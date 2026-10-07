import json

from PIL import Image

from eval.pid2graph.data import SYMBOLS, digest, save_new, sha256
from eval.pid2graph.detector import prepare_training


def test_training_preparation_never_opens_reserved_sources(tmp_path):
    image = tmp_path / "train.png"
    graph = tmp_path / "train.graphml"
    Image.new("RGB", (32, 32), "white").save(image)
    keys = "".join(f'<key id="{k}" for="node" attr.name="{k}"/>'
                   for k in ("label", "xmin", "ymin", "xmax", "ymax"))
    values = {"label": "valve", "xmin": 2, "ymin": 3, "xmax": 10, "ymax": 12}
    data = "".join(f'<data key="{k}">{v}</data>' for k, v in values.items())
    graph.write_text(f'<graphml xmlns="http://graphml.graphdrawing.org/xmlns">{keys}'
                     f'<graph edgedefault="undirected"><node id="v">{data}</node></graph></graphml>')
    training = {"id": "train", "group": "train-group", "split": "train", "image": image.name,
                "graph": graph.name, "image_sha256": sha256(image), "graph_sha256": sha256(graph)}
    # These paths deliberately do not exist. Training must not open them.
    reserved = [{**training, "id": split, "group": split, "split": split,
                 "image": "reserved.png", "graph": "reserved.graphml"} for split in ("validation", "test")]
    manifest = {"dataset_root": str(tmp_path), "drawings": [training, *reserved]}
    manifest["manifest_sha256"] = digest(manifest)
    save_new(tmp_path / "manifest.json", manifest)
    prepare_training(tmp_path / "manifest.json", tmp_path / "crops", crop_size=32, crops_per_drawing=2)
    result = json.loads((tmp_path / "crops/training.json").read_text())
    assert {r["source_id"] for r in result["samples"]} == {"train"}
    assert {r["split"] for r in result["sources"]} == {"train"}
    assert all(r["boxes"] == [[2, 3, 10, 12]] and r["labels"] == [SYMBOLS.index("valve") + 1]
               for r in result["samples"])
