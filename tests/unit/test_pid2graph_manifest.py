import json

import pytest

from eval.pid2graph.data import digest, freeze, save_new
from eval.pid2graph.runner import prepare, run


def test_grouped_split_and_source_only_panel(tmp_path):
    rows = []
    for i, collection in enumerate(["Synthetic", "Synthetic", "PID2Graph OPEN100", "PID2Graph OPEN100"]):
        rows.append({"id": f"{collection}/{i}", "collection": collection,
                     "image": f"Complete/{collection}/{i}.png", "graph": f"Complete/{collection}/{i}.graphml",
                     "image_sha256": "same" if i < 2 else str(i), "thumbnail_sha256": str(i),
                     "layout_sha256": str(i), "dhash": f"{i * 100:064x}", "size": [100, 100],
                     "node_counts": {"valve": 99}, "patch_count": 10})
    audit = {"complete": rows, "root": str(tmp_path), "inventory_sha256": "inventory", "errors": []}
    save_new(tmp_path / "audit.json", audit)
    manifest = freeze(tmp_path / "audit.json", tmp_path / "manifest.json")
    by_id = {r["id"]: r for r in manifest["drawings"]}
    assert by_id["Synthetic/0"]["split"] == by_id["Synthetic/1"]["split"]
    assert by_id["Synthetic/0"]["group"] == by_id["Synthetic/1"]["group"]
    assert by_id["PID2Graph OPEN100/2"]["split"] == "development_exposed"
    assert by_id["PID2Graph OPEN100/3"]["split"] == "development_exposed"
    panel = prepare(tmp_path / "manifest.json", tmp_path / "panel.json")
    for cases in panel["panels"].values():
        for case in cases:
            assert set(case) == {"id", "image", "image_sha256", "size", "collection"}
    with pytest.raises(FileExistsError):
        freeze(tmp_path / "audit.json", tmp_path / "manifest.json")


def test_final_inference_is_blocked_before_client_creation(tmp_path):
    panel = {"manifest_sha256": "x", "panels": {"test": []}}
    panel["panel_sha256"] = digest(panel)
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(panel))
    with pytest.raises(ValueError, match="sealed selection"):
        run(path, tmp_path / "out", "no-ledger", "no-prices", split="test")


def test_manifest_tampering_is_rejected_before_inference(tmp_path):
    panel = {"manifest_sha256": "x", "panels": {"validation": []}, "panel_sha256": "stale"}
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(panel))
    with pytest.raises(ValueError, match="content changed"):
        run(path, tmp_path / "out", "no-ledger", "no-prices")
