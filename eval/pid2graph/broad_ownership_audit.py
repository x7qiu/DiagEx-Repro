"""Replay completed broad responses to explain geometry filtering, without inference.

Only validation images and saved responses are read. GraphML is not used.
Replayed boxes, classes, ordering and source proposal IDs must match the saved
reviews on every successful tile before any ownership conclusion is reported.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from diagex.config import ScanConfig, TilingConfig
from diagex.vision.loader import iter_pages, load
from diagex.vision.perception import (
    NormalizedBBox,
    _bbox_center_is_owned,
    _project_normalized_bbox,
)
from diagex.vision.raster_broad import normalize_broad
from diagex.vision.tiling import AspectAwareStrategy, ownership_core, tile
from diagex.vision.views import ViewProvider

from .data import digest, save_new, sha256
from .epoch_train import sealed
from .raster_broad_runner import record


def audit_signature(audit):
    """Ignore generated exception wording, whose dict repr changes on JSON save.

    Exact statuses, proposal IDs, rejected indices and all wire rows still match.
    Visible reasons for symbol/reject/uncertain decisions remain exact.
    """
    value = copy.deepcopy(audit)
    value.pop("raw_input", None)
    for row in value["decisions"]:
        if row["status"] == "unresolved":
            row.pop("reason", None)
    for row in value["rejected_discoveries"]:
        row.pop("reason", None)
    return value


def verify_predictions(raw, sx, sy):
    expected = []
    for index, review in enumerate(raw["outcome"]["candidate_reviews"]):
        obj = review["object"]
        attributes = {**obj["attributes"], **{k: obj.get(k) for k in
                                             ("equipment_class", "valve_type", "actuation")}}
        expected.append(record(f"{raw['tile_id']}:proposal:{index}", obj["kind"],
            attributes, review["bbox"], obj.get("confidence", "medium"), sx, sy, "review_proposal"))
    if expected != raw["predictions"]:
        raise ValueError("Saved predictions differ from the replayed reviews")


def replay(raw, *, page, info, core):
    """Return normalized observations and their exact saved review identities."""
    valid = []
    for attempt in raw["raster_review_attempts"]:
        for result in attempt["tool_results"]:
            if "decisions" not in result:
                continue  # An invalid format attempt did not produce observations.
            observations, audit = normalize_broad(result["raw_input"], attempt["proposal_ids"])
            if audit_signature(audit) != audit_signature(result):
                raise ValueError("Normalization differs from the saved response audit")
            valid.append(observations)
    if len(valid) != 1:
        raise ValueError("Successful tile must have exactly one accepted tool payload")
    observations, expected, decisions = valid[0], [], []
    for index, row in enumerate(observations):
        box = _project_normalized_bbox(NormalizedBBox.model_validate(row["bbox"]),
                                       page=page, view_info=info)
        owned = _bbox_center_is_owned(box, core, page)
        identity = row.get("proposal_id")
        record = {"object_index": index, "bbox": box.model_dump(),
                  "label": row["broad_category"], "proposal_id": identity,
                  "confidence": row["confidence"]}
        decisions.append({**record, "owned": owned})
        if owned:
            expected.append(record)
    actual = [{"object_index": r["object_index"], "bbox": r["bbox"],
               "label": r["object"]["attributes"]["broad_category"],
               "proposal_id": r["object"]["attributes"].get("raster_proposal_id"),
               "confidence": r["object"]["confidence"]}
              for r in raw["outcome"]["candidate_reviews"]]
    if expected != actual:
        raise ValueError("Ownership replay differs from saved broad reviews")
    return decisions


def run(panel_path, run_path, audit_path, output):
    panel = sealed(panel_path, "panel_sha256")
    root = Path(run_path)
    config = json.loads((root / "config.json").read_text())
    audit = json.loads(Path(audit_path).read_text())
    if (config["split"] != "validation" or config["variant"] != "broad_raster_recognition"
            or config["panel_sha256"] != panel["panel_sha256"]
            or audit["run_config_sha256"] != sha256(root / "config.json")
            or not audit["attempted_coverage_complete"]):
        raise ValueError("Requires matching, fully attempted broad validation records")
    sources = {}
    source_root = Path(__file__).resolve().parents[2] / "src/diagex"
    if sha256(Path(__file__).with_name("raster_broad_runner.py")) != config["runner_source_sha256"]:
        raise ValueError("Frozen broad runner changed")
    for name, expected in config["source_fingerprints"].items():
        actual = sha256(source_root / name)
        if actual != expected:
            raise ValueError(f"Frozen source changed: {name}")
        sources[name] = actual
    sources["vision/views.py"] = sha256(source_root / "vision/views.py")
    tc, sc = TilingConfig(**config["tiling"]), ScanConfig(**config["scan"])
    lookup = {r["id"]: r for r in audit["cases"]}
    if len(lookup) != len(audit["cases"]) or set(lookup) != {r["id"] for r in panel["panels"]["validation"]}:
        raise ValueError("Decision audit coverage differs")
    cases, successful, failed, checked_objects, discoveries = [], 0, 0, 0, 0
    for row in panel["panels"]["validation"]:
        if sha256(row["image"]) != row["image_sha256"]:
            raise ValueError("Validation source changed")
        lost = lookup[row["id"]]["ids"]["recognized_without_retained_output"]
        observations = {i: [] for i in lost}
        failed_recognitions = {i: [] for i in lost}
        page = next(iter_pages(load(row["image"], tc, sc)))
        sx, sy = row["size"][0] / page.width, row["size"][1] / page.height
        grid = tile(page, AspectAwareStrategy(tc.max_tokens_per_tile, tc.overlap_frac, tc.token_per_pixel))
        views = ViewProvider(page, grid)
        for current in grid:
            path = root / digest(row["id"])[:16] / (current.id + ".json")
            raw = json.loads(path.read_text())
            if raw["config_sha256"] != digest(config) or raw["tile_id"] != current.id:
                raise ValueError("Saved tile identity or configuration differs")
            if raw["status"] == "failed":
                failed += 1
                for attempt in raw["raster_review_attempts"]:
                    for result in attempt["tool_results"]:
                        for decision in result.get("decisions", []):
                            identity = decision["proposal_id"]
                            if identity in failed_recognitions and decision["status"] == "symbol":
                                failed_recognitions[identity].append(current.id)
                continue
            if raw["status"] != "complete":
                raise ValueError("A tile is still unattempted")
            _, info = views.get_tile(current.id)
            core = ownership_core(current, grid)
            rows = replay(raw, page=page, info=info, core=core)
            verify_predictions(raw, sx, sy)
            successful += 1
            checked_objects += len(rows)
            discoveries += sum(r["proposal_id"] is None for r in rows)
            for value in rows:
                identity = value["proposal_id"]
                if identity in observations:
                    observations[identity].append({**value, "tile": current.id,
                        "tile_sha256": sha256(path), "ownership_core": core.model_dump()})
        unexplained = [i for i in lost if not observations[i] or any(r["owned"] for r in observations[i])
                       or failed_recognitions[i]]
        cases.append({"id": row["id"], "collection": row["collection"],
            "recognized_without_output": len(lost), "unowned_observations": observations,
            "recognized_on_failed_tiles": failed_recognitions,
            "not_explained_by_successful_tile_ownership_alone": unexplained})
        print(row["id"], len(lost), "recognized without output;", len(unexplained), "need other explanations", flush=True)
    save_new(output, {"panel_sha256": panel["panel_sha256"], "split": "validation",
        "cases": cases, "drawings": len(cases), "successful_tiles_replayed": successful,
        "failed_tiles_preserved": failed, "checked_observations_including_discoveries": checked_objects,
        "discovery_observations": discoveries,
        "recognized_without_output": sum(c["recognized_without_output"] for c in cases),
        "not_explained_by_ownership_alone": sum(len(c["not_explained_by_successful_tile_ownership_alone"]) for c in cases),
        "saved_reviews_reproduced_on_every_successful_tile": True,
        "saved_predictions_reproduced_on_every_successful_tile": True,
        "audit_comparison": "Exact structured decisions and rejected wire rows; generated exception wording excluded because JSON sorting can change embedded dict repr.",
        "audit_sha256": sha256(audit_path), "run_config_sha256": sha256(root / "config.json"),
        "source_sha256": sources, "reporter_sha256": sha256(__file__), "api_calls": 0,
        "ground_truth_read": False, "test_images_read": False,
        "interpretation": "Ownership replay identifies deterministic output filtering only when the saved successful reviews are reproduced exactly. It does not establish why the VLM did not recognize an object in its owning view. Failed tiles are preserved and cannot be declared ownership losses. No threshold, prompt, prediction or inference changes."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "run", "audit", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    run(args.panel, args.run, args.audit, args.out)
