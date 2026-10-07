"""Raster fallback must work across OpenCV Hough output layouts."""
import pytest
from PIL import Image, ImageDraw

from diagex.vision.evidence import PageEvidence
from diagex.vision.topology import extract_raster_paths


def page():
    return PageEvidence(page_index=0, source_ref="fixture", width=300, height=100,
                        dpi=72, effective_dpi=72, is_scanned=True)


@pytest.mark.parametrize("shape", [(2, 1, 4), (2, 4)])
def test_hough_layouts_preserve_line_coordinates(monkeypatch, shape):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    lines = np.array([[20, 30, 280, 30], [50, 10, 50, 90]], dtype=np.int32).reshape(shape)
    monkeypatch.setattr(cv2, "HoughLinesP", lambda *a, **kw: lines)
    paths, warning = extract_raster_paths(page=page(), image=Image.new("RGB", (300, 100), "white"))
    assert warning is None
    assert [p.points for p in paths] == [[(20, 30), (280, 30)], [(50, 10), (50, 90)]]
    assert all(p.origin == "raster_cv" for p in paths)


def test_installed_opencv_extracts_visible_stroke():
    pytest.importorskip("cv2")
    image = Image.new("RGB", (300, 100), "white")
    ImageDraw.Draw(image).line((20, 50, 280, 50), fill="black", width=2)
    paths, warning = extract_raster_paths(page=page(), image=image)
    assert warning is None
    assert any(p.bbox.w >= 200 and abs(p.points[0][1] - 50) <= 2 for p in paths)
