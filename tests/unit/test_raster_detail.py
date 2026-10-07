import copy
import hashlib

import pytest
from PIL import Image

from diagex.vision.raster_detail import RasterDetailClient, detail_sheets
from diagex.vision.raster_review import RasterReviewClient


def guide(identity="d1", **box):
    return {"proposal_id": identity, "bbox_normalized": {"x": .1, "y": .2, "w": .1, "h": .2, **box}}


def test_sheet_crops_exact_source_pixels_and_keeps_captions_outside_ink():
    source = Image.new("RGB", (200, 100), "white")
    source.paste("black", (20, 20, 40, 40))
    guides = [guide(), guide("edge", x=.98, y=.95, w=.02, h=.05)]
    original = copy.deepcopy(guides)
    sheets = detail_sheets(source, guides)
    assert guides == original and len(sheets) == 1
    sheet, metadata = sheets[0]
    assert sheet.size == (448, 224)
    for cell in metadata["cells"]:
        crop = source.crop(cell["source_view_bbox_pixels"])
        assert cell["source_crop_rgb_sha256"] == hashlib.sha256(crop.tobytes()).hexdigest()
        assert cell["display_origin"][1] >= cell["caption_band_xyxy"][3]
        assert cell["display_origin"][1] + cell["display_size"][1] <= 224
    assert metadata["cells"][1]["source_view_bbox_pixels"][2:] == [200, 100]


def test_all_eighty_guides_fit_without_silently_dropping_the_tail():
    guides = [guide(str(i)) for i in range(80)]
    sheets = detail_sheets(Image.new("RGB", (100, 100), "white"), guides)
    assert len(sheets) == 5
    assert [c["proposal_id"] for _, r in sheets for c in r["cells"]] == [str(i) for i in range(80)]
    assert all(max(image.size) <= 2000 for image, _ in sheets)


@pytest.mark.parametrize("bad", [[guide(), guide()], [guide(x=float("nan"))], [guide(x=.95, w=.2)]])
def test_invalid_guides_cannot_generate_misleading_crops(bad):
    with pytest.raises(ValueError):
        detail_sheets(Image.new("RGB", (100, 100)), bad)


def test_client_preserves_primary_image_and_coordinates_while_appending_details(monkeypatch):
    seen = []
    monkeypatch.setattr(RasterReviewClient, "messages_create", lambda self, **kw: seen.append(kw))
    client = RasterDetailClient.__new__(RasterDetailClient)
    client.begin_raster_view({"raster_proposal_guidance": {"guides": [guide()]}}, Image.new("RGB", (100, 100), "white"))
    messages = [{"role": "user", "content": [{"type": "image", "source": {"data": "primary"}}, {"type": "text", "text": "original coordinates"}]}]
    original = copy.deepcopy(messages)
    client.messages_create(messages=messages, tools=[{"name": "submit_pid_objects"}], max_tokens=6000)
    assert messages == original
    assert seen[0]["messages"][0]["content"][:2] == original[0]["content"]
    assert len(seen[0]["messages"][0]["content"]) == 4
    assert seen[0]["max_tokens"] == 6000 and len(client.raster_detail_manifest) == 1
    assert "FIRST detail image" in seen[0]["messages"][0]["content"][2]["text"]
