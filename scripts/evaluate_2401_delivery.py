"""Frozen, ledger-bounded production workflow comparison (two repeats)."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from threading import Event

from PIL import Image

from diagex.config import LLMConfig, RuntimeBudgets, SymbolPerceptionConfig
from diagex.extractors.evidence_checkpoint import atomic_write_json
from diagex.llm.client import LLMClient
from diagex.llm.cost import CostTracker
from diagex.llm.model_policy import ESCALATION_MODEL, FAST_MODEL
from diagex.ui.progress import NullReporter
from diagex.vision.evidence import PageEvidence
from diagex.vision.models import BBox, DiagramPage, Tile
from diagex.vision.perception import perceive_tile
from diagex.vision.symbol_candidates import SymbolCandidate
from diagex.vision.views import ViewProvider

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/2401-delivery/evaluation"
RUN = ROOT / "runs/2401/2026-09-08T04-47-37_deepseek-deepseek-v4-flash-vision-exp_r-4344"
PREFLIGHT = OUT.parent / "model-preflight"
CLOSED = "anthropic/claude-opus-4.7"
STOP = Event()
CYCLE = 2


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def owned(row, box):
    b = row.get("bbox", row.get("bbox_global"))
    return (
        box["x"] <= b["x"] + b["w"] / 2 < box["x"] + box["w"]
        and box["y"] <= b["y"] + b["h"] / 2 < box["y"] + box["h"]
    )


def public_truth(name):
    rows = [
        json.loads(line)
        for line in (ROOT / f"eval/datasets/{name}/annotations.truth.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    if name == "butane1":
        result = [
            dict(r, bbox=r["bbox_global"])
            for r in rows
            if r["label"] in {"E-234-010A", "E-234-010B", "PG-2914", "TW-2921", "TW-2922"}
        ]
        # Source-audited glyphs omitted from the original public fixture.
        for i, (x, y, w, h) in enumerate(
            [
                (207, 58, 32, 54),
                (208, 266, 30, 54),
                (210, 386, 30, 53),
                (325, 386, 31, 53),
                (207, 596, 32, 54),
                (205, 726, 33, 56),
                (800, 782, 31, 53),
                (11, 365, 51, 28),
                (-23, 685, 54, 28),
            ]
        ):
            result.append(
                dict(
                    id=f"audited-butane-valve-{i}",
                    kind="equipment",
                    label="",
                    bbox=dict(x=x + 2800, y=y + 2220, w=w, h=h),
                    attributes={"equipment_class": "valve"},
                )
            )
        return result
    return [
        dict(
            id="audited-filter",
            kind="equipment",
            label="FL-399",
            bbox=dict(x=2750, y=2420, w=245, h=80),
            attributes={"equipment_class": "filter"},
        ),
        dict(
            id="audited-LA004",
            kind="instrument",
            label="LA 004",
            bbox=dict(x=2595, y=2280, w=112, h=99),
            attributes={},
        ),
        dict(
            id="audited-V008",
            kind="equipment",
            label="V 008",
            bbox=dict(x=3055, y=2550, w=62, h=32),
            attributes={"equipment_class": "valve"},
        ),
        dict(
            id="audited-V030",
            kind="equipment",
            label="V 030",
            bbox=dict(x=3200, y=2647, w=30, h=48),
            attributes={"equipment_class": "valve"},
        ),
    ]


def prepare():
    if (OUT / "manifest.json").exists():
        raise ValueError("Frozen manifest exists; do not overwrite evaluation inputs")
    snapshot_path = OUT.parent / "audit/reviewed-snapshot.json"
    snapshot = read(snapshot_path)
    bundle = read(RUN / "detection.json")
    manifest = read(OUT / "frozen-regions.json")
    manifest.update(
        source_snapshot_sha256=sha(snapshot_path),
        cycle=1,
        fast=FAST_MODEL,
        escalation=ESCALATION_MODEL,
        closed=CLOSED,
        matching={
            "ownership": "glyph center in frozen region; context crop has 180px margin",
            "geometry": "IoU >= .25 OR intersection/min(area) >= .65; maximum-cardinality one-to-one, descending overlap traversal",
            "class": "kind + equipment family; subtype separately; unknown subtype never counted correct",
            "tags": "exact after whitespace and hyphen removal; original strings retained",
            "scope": "equipment, instrument, opc; exclude valve label callouts and passive piping fittings",
        },
        truth_review="Agent source audit; no independent human validation",
        maximum_cycles=2,
    )
    atomic_write_json(OUT / "legend.json", snapshot["legend_pack"]["entries"])
    for region in manifest["regions"]:
        pi = region["page_index"]
        name = region["dataset"]
        box = region["bbox"]
        if name == "2401":
            evidence = PageEvidence.model_validate_json(
                (RUN / f"evidence/page-{pi + 1:04d}.json").read_text()
            )
            image_path = OUT.parent / f"audit/page-{pi + 1:02d}.png"
            truth = [r for r in snapshot["detections"] if r["page_index"] == pi and owned(r, box)]
            candidates = [c for c in bundle["candidates"] if c["page_index"] == pi]
        else:
            image_path = OUT / f"{name}-page.png"
            with Image.open(image_path) as im:
                w, h = im.size
            evidence = PageEvidence(
                page_index=0,
                source_ref=f"{name}#page=1",
                width=w,
                height=h,
                dpi=72,
                effective_dpi=72,
                is_scanned=True,
                role="pid",
            )
            truth = [r for r in public_truth(name) if owned(r, box)]
            candidates = []
        truth = [
            r
            for r in truth
            if r["kind"] in {"equipment", "instrument", "opc"}
            and r.get("attributes", {}).get("equipment_class") != "piping_component"
        ]
        case = dict(
            region,
            evidence=evidence.model_dump(mode="json"),
            image_path=str(image_path),
            image_sha256=sha(image_path),
            truth=truth,
            candidates=candidates,
        )
        atomic_write_json(OUT / f"cases/{region['id']}.json", case)
        region.update(truth_count=len(truth), case_sha256=sha(OUT / f"cases/{region['id']}.json"))
    manifest["legend_sha256"] = sha(OUT / "legend.json")
    atomic_write_json(OUT / "manifest.json", manifest)
    print(
        json.dumps([{k: r[k] for k in ("id", "truth_count")} for r in manifest["regions"]]),
        flush=True,
    )


def family(row):
    if row["kind"] != "equipment":
        return row["kind"]
    value = row.get("attributes", {}).get("equipment_class", "unclassified_equipment")
    return "valve" if "valve" in value else value


def score(truth, detections):
    def overlap(a, b):
        a = a["bbox"]
        b = b["bbox"]
        i = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])) * max(
            0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
        )
        aa = a["w"] * a["h"]
        bb = b["w"] * b["h"]
        iou = i / max(1, aa + bb - i)
        return iou if iou >= 0.25 or i / max(1, min(aa, bb)) >= 0.65 else 0

    m = [[overlap(t, d) if family(t) == family(d) else 0 for d in detections] for t in truth]
    matched = {}

    def augment(i, seen):
        for j in sorted(range(len(detections)), key=lambda j: (-m[i][j], j)):
            if m[i][j] <= 0 or j in seen:
                continue
            seen.add(j)
            if j not in matched or augment(matched[j], seen):
                matched[j] = i
                return True
        return False

    for i in range(len(truth)):
        augment(i, set())
    pairs = sorted((i, j) for j, i in matched.items())
    def norm(s):
        return "".join(str(s or "").upper().replace("-", "").split())
    tagged = [(i, j) for i, j in pairs if truth[i].get("label")]
    known_subtypes = [
        (i, j)
        for i, j in pairs
        if truth[i].get("attributes", {}).get("valve_type") not in {None, "general", "unknown"}
    ]
    return dict(
        tp=len(pairs),
        fp=len(detections) - len(pairs),
        fn=len(truth) - len(pairs),
        truth=len(truth),
        predictions=len(detections),
        matches=pairs,
        tag_correct=sum(
            norm(truth[i].get("label")) == norm(detections[j].get("label")) for i, j in tagged
        ),
        tag_matched=len(tagged),
        tag_truth=sum(bool(t.get("label")) for t in truth),
        subtype_correct=sum(
            truth[i]["attributes"]["valve_type"]
            == detections[j].get("attributes", {}).get("valve_type")
            for i, j in known_subtypes
        ),
        subtype_matched=len(known_subtypes),
    )


def run_case(region, condition, repeat, base):
    dest = OUT / f"results-cycle-{CYCLE}/{condition}/{region['id']}-{repeat}.json"
    if dest.exists():
        return read(dest)
    if STOP.is_set():
        return {"status": "not_dispatched_after_systemic_failure"}
    case_path = OUT / f"cases/{region['id']}.json"
    assert sha(case_path) == region["case_sha256"]
    case = read(case_path)
    assert sha(Path(case["image_path"])) == case["image_sha256"]
    evidence = PageEvidence.model_validate(case["evidence"])
    im = Image.open(case["image_path"]).convert("RGB")
    page = DiagramPage(
        **{
            k: getattr(evidence, k)
            for k in (
                "page_index",
                "source_ref",
                "width",
                "height",
                "dpi",
                "effective_dpi",
                "is_scanned",
            )
        },
        image=im,
    )
    provider = ViewProvider(page, [])
    box = BBox(**case["bbox"])
    view, info = provider.get_region(box.x - 180, box.y - 180, box.w + 360, box.h + 360)
    tile = Tile(id=f"evaluation-{region['id']}", page_index=page.page_index, bbox=info.page_bbox)
    adaptive = condition != "fixed-fast-open-weight"
    model = CLOSED if condition == "adaptive-closed-reference" else FAST_MODEL
    cfg = replace(
        base,
        transport="openrouter",
        model=model,
        vision_model=None,
        reasoning_model=None,
        escalation_model=None,
        production_open_weight=model != CLOSED,
        reasoning_mode="auto",
        spending_ledger=str(PREFLIGHT / "spending.json"),
        verified_prices=str(PREFLIGHT / "verified-prices.json"),
        spending_category="extraction",
    )
    client = LLMClient(cfg, budgets=RuntimeBudgets(retry_attempts=1))
    stronger = (
        LLMClient(replace(cfg, model=ESCALATION_MODEL), budgets=RuntimeBudgets(retry_attempts=1))
        if adaptive and model != CLOSED
        else None
    )
    cost = CostTracker()
    started = time.monotonic()
    diagnostics = []
    result = dict(
        id=region["id"],
        split=region["split"],
        dataset=region["dataset"],
        condition=condition,
        repeat=repeat,
        model=model,
        escalation_model=ESCALATION_MODEL if stronger else model,
        cycle=CYCLE,
    )
    try:
        outcome = perceive_tile(
            client=client,
            escalation_client=stronger,
            cost_tracker=cost,
            reporter=NullReporter(),
            page=evidence,
            tile=tile,
            view_image=view,
            view_info=info,
            ownership_bbox=box,
            legend_summary=read(OUT / "legend.json"),
            step=1,
            candidates=[SymbolCandidate.model_validate(c) for c in case["candidates"]],
            policy=SymbolPerceptionConfig(
                workflow="adaptive" if adaptive else "fixed", transport_attempts=1
            ),
            overview_image=provider.get_overview()[0] if adaptive else None,
            region_provider=provider.get_region if adaptive else None,
            on_diagnostic=diagnostics.append,
            page_context={
                "dataset": case["dataset"],
                "legend_scope": "2401 definitions are project-specific; on public drawings transfer graphical shapes only, never assert unprinted project tags.",
            },
        )
        dets = [
            d.model_dump(mode="json")
            for d in outcome.detections
            if d.kind in {"equipment", "instrument", "opc"}
            and d.attributes.get("equipment_class") != "piping_component"
        ]
        result.update(
            status="complete" if not outcome.contract_failed else "contract_failed",
            detections=dets,
            batch=outcome.batch.model_dump(mode="json"),
            attempts=outcome.attempts,
            metrics=score(case["truth"], dets),
        )
    except Exception as exc:
        from diagex.llm.client import is_non_retryable_api_error

        if is_non_retryable_api_error(exc):
            STOP.set()
        result.update(
            status="failed", error=f"{type(exc).__name__}: {exc}", metrics=score(case["truth"], [])
        )
    result.update(
        elapsed_s=time.monotonic() - started,
        usage=[asdict(s) for s in cost.steps],
        diagnostics=diagnostics,
    )
    atomic_write_json(dest, result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k in {"id", "condition", "repeat", "status", "elapsed_s", "metrics", "error"}
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return result


def run(conditions):
    manifest = read(OUT / "manifest.json")
    assert sha(OUT / "legend.json") == manifest["legend_sha256"]
    base = LLMConfig.from_env()
    # Complete the open-weight comparison before spending on a closed reference.
    for condition in conditions:
        jobs = [
            (region, condition, repeat, base) for repeat in (1, 2) for region in manifest["regions"]
        ]
        if condition == "adaptive-closed-reference":
            regions = sorted(
                manifest["regions"],
                key=lambda r: ({"heldout": 0, "public": 1, "diagnostic": 2}[r["split"]], r["id"]),
            )
            jobs = [(region, condition, repeat, base) for region in regions for repeat in (1, 2)]
        with ThreadPoolExecutor(
            max_workers=1 if condition == "adaptive-closed-reference" else 3
        ) as pool:
            for future in as_completed([pool.submit(run_case, *job) for job in jobs]):
                future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "fixed-fast-open-weight",
            "adaptive-fast-open-weight",
            "adaptive-closed-reference",
        ],
    )
    args = parser.parse_args()
    prepare() if args.action == "prepare" else run(args.conditions)
