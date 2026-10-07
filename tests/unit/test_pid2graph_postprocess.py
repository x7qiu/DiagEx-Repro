from PIL import Image, ImageDraw

from eval.pid2graph.postprocess import ink_guard, suppress_duplicates


def row(identity, box, label="valve", confidence=0.9):
    return {"id": identity, "bbox": box, "label": label, "confidence": confidence}


def test_blank_source_and_outside_boxes_cannot_pass_ink_guard():
    image = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(image).rectangle((10, 10, 20, 20), outline="black")
    kept, rejected = ink_guard(image, [row("ink", [8, 8, 22, 22]), row("blank", [50, 50, 60, 60]),
                                     row("outside", [-30, -30, -10, -10])])
    assert [p["id"] for p in kept] == ["ink"]
    assert len(rejected) == 2


def test_suppression_keeps_distinct_classes_and_adjacent_symbols():
    predictions = [row("first", [0, 0, 10, 10]), row("duplicate", [1, 1, 11, 11], confidence=0.6),
                   row("adjacent", [12, 0, 22, 10]), row("other_kind", [0, 0, 10, 10], "instrumentation")]
    kept, rejected = suppress_duplicates(predictions)
    assert {p["id"] for p in kept} == {"first", "adjacent", "other_kind"}
    assert rejected[0]["retained_id"] == "first"
