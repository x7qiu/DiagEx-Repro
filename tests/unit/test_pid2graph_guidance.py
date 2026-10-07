from types import SimpleNamespace

from eval.pid2graph.guidance import view_guidance


def test_source_boxes_map_into_view_without_becoming_native_evidence():
    box = SimpleNamespace(x=100, y=50, w=200, h=100)
    rows = [{"id": "detector-1", "bbox": [220, 120, 260, 160], "confidence": 0.8, "label": "valve"},
            {"id": "outside", "bbox": [10, 10, 20, 20], "confidence": 0.9, "label": "pump"},
            {"id": "clipped", "bbox": [190, 90, 220, 120], "confidence": 0.5, "label": "general"}]
    result = view_guidance(rows, box, (2, 2))
    assert "native_symbol_candidates" not in result
    guides = result["raster_proposal_guidance"]["guides"]
    assert [g["proposal_id"] for g in guides] == ["detector-1", "clipped"]
    assert guides[0]["bbox_normalized"] == {"x": 0.05, "y": 0.1, "w": 0.1, "h": 0.2}
    assert guides[0]["partially_visible"] is False
    assert guides[1]["partially_visible"] is True
    assert "candidate_id" not in guides[0]
