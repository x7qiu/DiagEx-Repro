from eval.pid2graph.coverage_data import contained, drawing_plan, overlap


def test_large_symbols_boundaries_and_small_neighbors_remain_covered():
    nodes = [
        {"id": "large", "label": "tank", "bbox": [0, 0, 1600, 700]},
        {"id": "small", "label": "valve", "bbox": [10, 10, 25, 25]},
        {"id": "edge", "label": "pump", "bbox": [1950, 1950, 2000, 2000]},
        {"id": "pipe", "label": "connector", "bbox": [900, 1500, 910, 1510]},
    ]
    physical, crops = drawing_plan(nodes, [2000, 2000])
    assert {n["id"] for n in physical} == {"large", "small", "edge"}
    for n in physical:
        assert any(n["id"] in c["covered_ids"] and contained(n["bbox"], c["bbox"]) for c in crops)
    assert any(c["anchor"] == "small" for c in crops), "Downscaled tiny neighbor needs a full-resolution crop"
    for c in crops:
        if c["kind"] == "background":
            assert all(overlap(n["bbox"], c["bbox"]) == 0 for n in physical)


def test_empty_drawing_can_supply_a_real_negative():
    physical, crops = drawing_plan([{"id": "line", "label": "connector", "bbox": [10, 10, 100, 100]}], [1000, 1000])
    assert physical == [] and len(crops) == 1 and crops[0]["kind"] == "background"
