import pytest

from eval.pid2graph.data import read_graph
from eval.pid2graph.scoring import score_drawing


def node(identity, label="valve", bbox=None, confidence=0.9):
    return {"id": identity, "label": label, "bbox": bbox or [0, 0, 10, 10], "confidence": confidence}


def test_duplicate_predictions_cannot_inflate_recall():
    score = score_drawing({"nodes": [node("truth")]}, [node("a"), node("b")])
    assert score["symbols"]["tp"] == 1
    assert score["symbols"]["fp"] == 1
    assert score["symbols"]["recall"] == 1
    assert score["symbols"]["precision"] == 0.5


def test_wrong_class_is_localized_but_not_classified():
    score = score_drawing({"nodes": [node("truth")]}, [node("a", "pump")])
    assert score["symbols"]["tp"] == 0
    assert score["localization"]["tp"] == 1
    assert score["per_class"]["valve"]["fn"] == 1
    assert score["per_class"]["pump"]["fp"] == 1


def test_misses_and_helpers_do_not_disappear():
    graph = {"nodes": [node("truth"), node("helper", "connector")]}
    score = score_drawing(graph, [])
    assert score["symbols"]["actual"] == 1
    assert score["symbols"]["recall"] == 0
    assert score["symbols"]["precision"] is None
    assert score["symbols"]["f1"] == 0


def test_matching_uses_geometry_not_matching_ids():
    score = score_drawing({"nodes": [node("same")]}, [node("same", bbox=[20, 20, 30, 30])])
    assert score["symbols"]["tp"] == 0


def test_bad_predictions_fail_visibly():
    with pytest.raises(ValueError):
        score_drawing({"nodes": []}, [node("bad", confidence=float("nan"))])
    with pytest.raises(ValueError):
        score_drawing({"nodes": []}, [node("same"), node("same")])


def test_graphml_keys_are_resolved_by_name_not_number(tmp_path):
    p = tmp_path / "truth.graphml"
    p.write_text('''<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
    <key id="d0" for="node" attr.name="label" attr.type="string"/>
    <key id="d1" for="node" attr.name="xmax" attr.type="double"/>
    <key id="d2" for="node" attr.name="xmin" attr.type="double"/>
    <key id="d3" for="node" attr.name="ymax" attr.type="double"/>
    <key id="d4" for="node" attr.name="ymin" attr.type="double"/>
    <graph edgedefault="undirected"><node id="n"><data key="d0">valve</data>
    <data key="d1">30</data><data key="d2">10</data>
    <data key="d3">50</data><data key="d4">20</data></node></graph></graphml>''')
    graph = read_graph(p)
    assert graph["nodes"][0]["bbox"] == [10, 20, 30, 50]
    assert graph["directed"] is False
