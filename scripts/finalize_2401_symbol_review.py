"""Record final metadata-only source corrections without changing frozen eval truth."""

import json
from pathlib import Path

from diagex.extractors.evidence_checkpoint import atomic_write_json
from diagex.review.detection import DetectionReviewStore

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/2401-delivery/audit"


def main():
    report = OUT / "final-symbol-corrections.json"
    if report.exists():
        raise ValueError("Final metadata review already applied")
    store = DetectionReviewStore(ROOT / "runs/2401/2026-09-09_audited_r-4344-derived")
    tags = {
        102: ("2401-V-001", "txt-54d631dd0cbaec02"),
        181: ("2401-V-002", "txt-08011bab22b87e84"),
        316: ("2401-V-003", "txt-6721c3fe97abf522"),
    }
    explicit_solenoids = {26, 41, 82, 94, 205, 232, 295, 304}
    corrections = []
    for row in store.read()["symbols"]:
        if row["status"] != "confirmed":
            continue
        d = row["detection"]
        a = d["attributes"]
        i = a.get("source_audit_index")
        reasons = []
        if i in tags:
            label, tid = tags[i]
            d.update(
                label=label,
                raw_text=label,
                source_text_ids=sorted(set(d.get("source_text_ids", []) + [tid])),
            )
            a.update(printed_tag=label, canonical_tag=label, source_text_ids=d["source_text_ids"])
            reasons.append("Bind the printed vessel body label; preserve source spelling")
        if (
            a.get("actuation")
            and i not in explicit_solenoids
            and not str(a.get("source_audit_id", "")).startswith("omission-")
        ):
            old = a.pop("actuation")
            a["actuation_review"] = (
                "Unknown: previous model attribute is not substantiated by the audited glyph"
            )
            reasons.append(f"Remove unsupported prior actuation {old}; no inferred default")
        if not reasons:
            continue
        refs = [
            f"audit/page-{d['page_index'] + 1:02d}.png#bbox={','.join(str(d['bbox'][k]) for k in ('x', 'y', 'w', 'h'))}"
        ]
        if i in tags:
            refs.append(
                f"original-native-evidence/page-{d['page_index'] + 1:04d}.json#{tags[i][1]}"
            )
        store.apply(
            dict(
                action="save_symbol",
                id=row["id"],
                detection=d,
                status="confirmed",
                draft_review=True,
                revision=store.read()["revision"],
                rater="Codex source auditor",
                actor_type="agent",
                evidence_refs=refs,
                note="; ".join(reasons),
            )
        )
        corrections.append(
            dict(
                id=d["id"],
                page_index=d["page_index"],
                reasons=reasons,
                evidence_refs=refs,
                detection=d,
            )
        )
    frozen = store.snapshot(store.read()["revision"], draft=True)
    atomic_write_json(OUT / "reviewed-snapshot-final.json", frozen)
    atomic_write_json(
        report,
        dict(
            review_revision=frozen["revision"],
            review_origin="agent",
            changes=corrections,
            frozen_evaluation_unchanged=True,
            graph_policy="Apply identity/attribute changes by stable node ID before final export; no geometry changes",
        ),
    )
    print(json.dumps({"corrections": len(corrections), "revision": frozen["revision"]}), flush=True)


if __name__ == "__main__":
    main()
