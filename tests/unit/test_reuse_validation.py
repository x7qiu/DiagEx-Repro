import json

import pytest

from eval.pid2graph.data import digest, save_new, sha256
from eval.pid2graph.reuse_validation import seed


def fixture(tmp_path, *, second_split="validation"):
    code = tmp_path / "code.py"
    code.write_text("# method implementation")
    image = tmp_path / "image.png"
    image.write_bytes(b"source bytes")
    rows = [{"id": name, "collection": "synthetic", "image": str(image), "image_sha256": sha256(image), "size": [20, 10]}
            for name in ["first", "second"]]
    manifest = {"drawings": [{**r, "split": "validation" if i == 0 else second_split} for i, r in enumerate(rows)]}
    manifest["manifest_sha256"] = digest(manifest)
    save_new(tmp_path / "manifest.json", manifest)
    panels = []
    for name, subset in [("source", rows[:1]), ("target", rows)]:
        p = {"manifest_sha256": manifest["manifest_sha256"], "panels": {"validation": subset, "test": []}}
        p["panel_sha256"] = digest(p)
        save_new(tmp_path / (name + ".json"), p)
        panels.append(p)
    run = tmp_path / "run"
    config = {"variant": "baseline", "split": "validation", "model": "model", "panel_sha256": panels[0]["panel_sha256"]}
    save_new(run / "config.json", config)
    price = {"price": 0.001}
    price_path = run / "price-snapshots" / "temp.json"
    save_new(price_path, price)
    price_hash = sha256(price_path)
    price_path.rename(price_path.with_name(price_hash + ".json"))
    tile = {"tile_id": "p0-r0-c0", "status": "failed", "error": "TimeoutError: no response", "predictions": [],
            "config_sha256": digest(config), "ledger_request_ids": ["request-id"], "price_snapshot_sha256": price_hash,
            "runtime_seconds": 2, "reserved_or_charged_usd": .2}
    case = run / digest("first")[:16]
    save_new(case / "p0-r0-c0.json", tile)
    save_new(case / "predictions.json", {"id": "first", "image_sha256": rows[0]["image_sha256"], "config_sha256": digest(config),
            "tiles": 1, "completed_tiles": 0, "predictions": [], "ledger_request_ids": ["request-id"],
            "errors": [{"tile_id": tile["tile_id"], "error": tile["error"]}], "runtime_seconds": 2, "reserved_or_charged_usd": .2})
    registration = {"split": "validation", "panel_sha256": panels[0]["panel_sha256"], "variant": "baseline", "model": "model",
                    "method_fingerprints": {str(code): sha256(code)}}
    save_new(tmp_path / "registration.json", registration)
    return (tmp_path / "source.json", tmp_path / "target.json", run, tmp_path / "expanded",
            tmp_path / "registration.json", tmp_path / "manifest.json")


def test_reuse_preserves_failed_attempt_and_billing_identity(tmp_path):
    args = fixture(tmp_path)
    result = seed(*args)
    assert result["reused_drawings"] == ["first"] and result["unattempted_drawings"] == ["second"]
    assert result["new_paid_requests"] == 0
    old_path = args[2] / digest("first")[:16] / "p0-r0-c0.json"
    new = json.loads((args[3] / digest("first")[:16] / "p0-r0-c0.json").read_text())
    old = json.loads(old_path.read_text())
    assert new["validation_reuse"]["source_sha256"] == sha256(old_path)
    assert new["ledger_request_ids"] == old["ledger_request_ids"]
    assert new["status"] == "failed" and new["error"] == old["error"]
    assert new["config_sha256"] != old["config_sha256"]
    assert {k: v for k, v in new.items() if k not in {"config_sha256", "validation_reuse"}} == {k: v for k, v in old.items() if k != "config_sha256"}
    with pytest.raises(FileExistsError):
        seed(*args)


@pytest.mark.parametrize("change", ["image", "implementation", "tile_coverage", "billing", "test_run", "stopped"])
def test_changed_or_incomplete_inputs_are_not_reused(tmp_path, change):
    args = fixture(tmp_path)
    tile = args[2] / digest("first")[:16] / "p0-r0-c0.json"
    if change == "image":
        (tmp_path / "image.png").write_bytes(b"different pixels")
    elif change == "implementation":
        (tmp_path / "code.py").write_text("# changed")
    elif change == "tile_coverage":
        tile.unlink()
    elif change == "billing":
        data = json.loads(tile.read_text())
        data["ledger_request_ids"] = ["another-request"]
        tile.write_text(json.dumps(data))
    elif change == "stopped":
        (args[2] / "stop-request.json").write_text("{}")
    else:
        path = args[2] / "config.json"
        data = json.loads(path.read_text())
        data["split"] = "test"
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        seed(*args)
    assert not args[3].exists()


def test_panel_cannot_relabel_a_test_drawing_as_validation(tmp_path):
    args = fixture(tmp_path, second_split="test")
    with pytest.raises(ValueError, match="frozen validation"):
        seed(*args)
