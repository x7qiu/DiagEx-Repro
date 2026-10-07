import json

import pytest

from eval.pid2graph.__main__ import score
from eval.pid2graph.data import digest, save_new, sha256


@pytest.mark.parametrize("schema", [1, 2])
def test_concurrent_legend_charge_is_not_attributed_to_symbol_run(tmp_path, schema):
    fields = {"label": "valve", "xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}
    keys = "".join(f'<key id="{k}" for="node" attr.name="{k}"/>' for k in fields)
    values = "".join(f'<data key="{k}">{v}</data>' for k, v in fields.items())
    graph = tmp_path / "drawing.graphml"
    graph.write_text(f'<graphml xmlns="http://graphml.graphdrawing.org/xmlns">{keys}'
                     f'<graph edgedefault="undirected"><node id="v">{values}</node></graph></graphml>')
    row = {"id": "drawing", "collection": "synthetic", "graph": graph.name,
           "graph_sha256": sha256(graph), "image_sha256": "image"}
    manifest = {"dataset_root": str(tmp_path), "drawings": [row], "limits": []}
    manifest["manifest_sha256"] = digest(manifest)
    panel = {"manifest_sha256": manifest["manifest_sha256"], "panels": {"validation": [{"id": "drawing"}]}}
    panel["panel_sha256"] = digest(panel)
    config = {"model": "m", "variant": "baseline", "split": "validation", "panel_sha256": panel["panel_sha256"]}
    save_new(tmp_path / "manifest.json", manifest)
    save_new(tmp_path / "panel.json", panel)
    save_new(tmp_path / "run/config.json", config)
    case = tmp_path / "run" / digest("drawing")[:16]
    save_new(case / "predictions.json", {"config_sha256": digest(config), "image_sha256": "image", "errors": [],
             "runtime_seconds": 1, "reserved_or_charged_usd": 2.1,
             "predictions": [{"id": "p", "bbox": [0, 0, 10, 10], "label": "valve", "confidence": 0.9,
                              "disposition": "graph_eligible"}]})
    save_new(case / "tile.json", {"ledger_request_ids": ["symbol", "legend"]})
    ledger = {"limit_usd": 60, "requests": [
        {"id": "symbol", "category": "baseline", "model": "m", "charged_usd": 0.1},
        {"id": "legend", "category": "legend_transfer", "model": "m", "charged_usd": 2}]}
    if schema == 2:
        ledger.update(schema_version=2, prior_spend={"usd": 0.64})
        ledger["requests"][0].update(actual_billed_usd=None, exposure_usd=0.1, status="unresolved")
        ledger["requests"][1].update(actual_billed_usd=2, exposure_usd=2, status="billed")
    save_new(tmp_path / "spending.json", ledger)
    report = score(tmp_path / "manifest.json", tmp_path / "panel.json", tmp_path / "run", tmp_path / "report.json", "validation")
    assert report["reserved_or_charged_usd"] == 0.1
    assert report["summary"]["symbols"]["tp"] == 1
    if schema == 2:
        assert report["billing"]["actual_billed_usd"] == 0
        assert report["billing"]["unresolved_upper_usd"] == 0.1
        assert not report["billing"]["actual_billing_complete"]
    assert json.loads((tmp_path / "spending.json").read_text()) == ledger
