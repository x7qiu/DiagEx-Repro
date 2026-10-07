"""Native multiword caption identity, without cross-line or spatial guesses."""
import pytest
from PIL import Image, ImageDraw

from diagex.extractors.pid_legend import _matching_label_span, _select_legend_crop
from diagex.vision.evidence import PageEvidence, TextEvidence
from diagex.vision.models import BBox, DiagramPage


def words():
    return [
        TextEvidence(id='a', text='Process', bbox=BBox(x=300, y=100, w=60, h=20),
                     block_index=2, line_index=0, word_index=0),
        TextEvidence(id='b', text='Lines', bbox=BBox(x=366, y=100, w=40, h=20),
                     block_index=2, line_index=0, word_index=1),
    ]


def evidence(spans):
    return PageEvidence(page_index=0, source_ref='public-mock', width=800, height=800,
                        dpi=144, effective_dpi=144, is_scanned=False,
                        text_spans=spans, paths=[])


def test_exact_caption_reconstruction_uses_consecutive_words_and_stable_source_identity():
    spans = words()
    result = _matching_label_span('Process Lines', evidence(spans))
    assert result.text == 'Process Lines'
    assert result.bbox == BBox(x=300, y=100, w=106, h=20)
    assert (result.block_index, result.line_index, result.word_index) == (2, 0, 0)
    assert result.id.startswith('legend-caption-')
    assert _matching_label_span('Process Lines', evidence(list(reversed(spans)))) == result
    assert [span.id for span in spans] == ['a', 'b']


@pytest.mark.parametrize('field,value', [
    ('block_index', None), ('line_index', None), ('word_index', None),
    ('block_index', 3), ('line_index', 1), ('word_index', 2),
    ('word_index', 0), ('word_index', -1), ('origin', 'raster_ocr'),
])
def test_missing_conflicting_or_unrelated_native_identity_is_not_joined(field, value):
    spans = words()
    spans[1] = spans[1].model_copy(update={field: value})
    assert _matching_label_span('Process Lines', evidence(spans)) is None


@pytest.mark.parametrize('bbox', [
    BBox(x=600, y=100, w=40, h=20),  # Separate distant column.
    BBox(x=366, y=130, w=40, h=20),  # Separate visual line.
    BBox(x=200, y=100, w=40, h=20),  # Reversed or corrupt native ordering.
])
def test_geometrically_unrelated_words_are_not_joined(bbox):
    spans = words()
    spans[1] = spans[1].model_copy(update={'bbox': bbox})
    assert _matching_label_span('Process Lines', evidence(spans)) is None


def test_legacy_whole_span_match_survives_missing_metadata_and_wins_over_reconstruction():
    direct = TextEvidence(id='legacy', text='Process Lines', bbox=BBox(x=10, y=10, w=100, h=20))
    assert _matching_label_span('Process Lines', evidence([*words(), direct])) == direct
    assert _matching_label_span('Process Line', evidence(words())) is None


def test_reconstructed_caption_reuses_existing_crop_checks_without_relocating_pixels():
    source = Image.new('RGB', (800, 800), 'white')
    draw = ImageDraw.Draw(source)
    draw.line((100, 110, 150, 110), fill='black', width=2)
    draw.rectangle((100, 600, 150, 640), outline='black', width=2)
    page = DiagramPage(page_index=0, source_ref='public-mock', image=source,
                       width=800, height=800, dpi=144, effective_dpi=144, is_scanned=False)
    local = BBox(x=96, y=106, w=58, h=10)
    valid = _select_legend_crop(page=page, evidence=evidence(words()),
                               label='Process Lines', model_bbox=local)
    assert valid[0] is not None and valid[1] == local and valid[-1] == 'accepted'
    assert valid[2] == BBox(x=300, y=100, w=106, h=20)
    # A distant, nonblank equipment glyph cannot be substituted for this caption.
    distant = BBox(x=96, y=596, w=58, h=48)
    rejected = _select_legend_crop(page=page, evidence=evidence(words()),
                                  label='Process Lines', model_bbox=distant)
    assert rejected[0] is None and rejected[1] == distant
    assert rejected[-1] == 'rejected_text_overlap'  # Existing rejection vocabulary.
