"""Join completed proposal decisions to scored detector matches, after inference only."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .data import digest, save_new, sha256
from .proposal_retention import compare

GROUPS = ("never_requested", "rejected_only", "unresolved_without_recognition",
          "recognized_without_retained_output", "retained_in_output")


def join_case(detector, hybrid, audit):
    if len({r["id"] for r in (detector, hybrid, audit)}) != 1:
        raise ValueError("Drawing identity differs")
    tp = {m["prediction_id"]: m for m in detector["matches"]}
    fp = set(detector["false_positive_ids"])
    if len(tp) != len(detector["matches"]) or tp.keys() & fp:
        raise ValueError("Detector assignments overlap")
    domain = tp.keys() | fp
    groups = {k: set(audit["ids"][k]) for k in GROUPS}
    if (set.union(*groups.values()) != domain or sum(map(len, groups.values())) != len(domain)
            or audit["detector_proposals"] != len(domain)):
        raise ValueError("Decision groups must partition the scored proposals")
    found = {m["truth_id"] for m in hybrid["matches"]}
    localized = {m["truth_id"] for m in hybrid["localization_matches"]}
    result = {}
    for name, ids in groups.items():
        matched = ids & tp.keys()
        recovered = {i for i in matched if tp[i]["truth_id"] in found}
        lost = matched - recovered
        wrong_class = {i for i in lost if tp[i]["truth_id"] in localized}
        result[name] = {"proposals": len(ids), "detector_true_positives": len(matched),
                        "detector_false_positives": len(ids & fp),
                        "truth_found_in_hybrid": len(recovered), "truth_lost_in_hybrid": len(lost),
                        "lost_but_localized": len(wrong_class),
                        "lost_without_localization": len(lost - wrong_class)}
    return {"id": detector["id"], "collection": detector["collection"], "groups": result}


def join(detector, hybrid, audit, config_sha256):
    compare(detector, hybrid)  # Require identical manifest, panel, split and attempted coverage.
    if hybrid["config"].get("guidance", {}).get("detector_config") != detector["config"]:
        raise ValueError("Hybrid used a different detector")
    if digest(hybrid["config"]) != hybrid["config_sha256"]:
        raise ValueError("Hybrid configuration seal differs")
    if (not audit["attempted_coverage_complete"] or audit["panel_sha256"] != hybrid["panel_sha256"]
            or audit["split"] != hybrid["split"] or audit["run_config_sha256"] != config_sha256):
        raise ValueError("Decision audit provenance differs")
    a = {r["id"]: r for r in audit["cases"]}
    h = {r["id"]: r for r in hybrid["cases"]}
    if a.keys() != h.keys() or len(a) != len(audit["cases"]):
        raise ValueError("Decision audit case coverage differs")
    cases = [join_case(d, h[d["id"]], a[d["id"]]) for d in detector["cases"]]

    def pool(rows):
        result = {}
        for group in GROUPS:
            counts = Counter()
            for row in rows:
                counts.update(row["groups"][group])
            result[group] = dict(counts)
        return result

    return {"panel_sha256": hybrid["panel_sha256"], "split": hybrid["split"], "drawings": len(cases),
            "groups": pool(cases), "cases": cases,
            "per_collection": {c: pool([r for r in cases if r["collection"] == c])
                               for c in sorted({r["collection"] for r in cases})},
            "interpretation": "Reporting-only association, not causal attribution. Groups partition detector proposal IDs within each drawing. A rejected proposal's truth may be recovered by another proposal or discovery. Retained output can have wrong class or geometry. Unresolved includes uncertain, missing, invalid and failed responses; it does not mean rejection. Detector false positives include duplicates and class/localization errors under frozen matching. No engineering approval or new inference."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("detector", "hybrid", "audit", "run", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    inputs = {k: getattr(args, k) for k in ("detector", "hybrid", "audit")}
    reports = {k: json.loads(p.read_text()) for k, p in inputs.items()}
    config = args.run / "config.json"
    if json.loads(config.read_text()) != reports["hybrid"]["config"]:
        raise ValueError("Run configuration differs from scored report")
    result = join(**reports, config_sha256=sha256(config))
    result["inputs"] = {k: {"path": str(p), "sha256": sha256(p)} for k, p in inputs.items()}
    result["inputs"]["run_config"] = {"path": str(config), "sha256": sha256(config)}
    save_new(args.out, result)
    print(json.dumps(result["groups"], indent=2))


if __name__ == "__main__":
    main()
