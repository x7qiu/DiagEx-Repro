"""Frozen first-pass symbol reasoning comparison; does not modify production runs.

prepare is offline. run sends only the predeclared crops and repetitions from
the manifest (no semantic or HTTP retries). Provider routing is pinned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import fitz
import httpx
from PIL import Image

from diagex.config import Config
from diagex.llm.cost import CostTracker
from diagex.ui.progress import NullReporter
from diagex.vision.evidence import PageEvidence, text_spans_intersecting
from diagex.vision.models import BBox, DiagramPage, Tile
from diagex.vision.perception import parse_perception_response, perceive_tile, project_batch
from diagex.vision.symbol_candidates import SymbolCandidate
from diagex.vision.tiling import AspectAwareStrategy, ownership_core, tile
from diagex.vision.views import ViewInfo, ViewProvider

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/2401/2026-09-06T23-30-43_deepseek-deepseek-v4-flash-vision-exp_r-5427"
OUT = ROOT / "output/reasoning-ab-r5427"
TILES = [
    "p3-r2-c1",
    "p7-r1-c1",
    "p8-r1-c1",
    "p3-r1-c1",
    "p5-r1-c2",
    "p10-r3-c2",
    "p4-r1-c4",
    "p6-r2-c1",
    "p8-r3-c0",
]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


class Capture:
    def messages_create(self, **kwargs):
        self.request = {k: v for k, v in kwargs.items() if k != "on_stream_delta"}
        data = json.loads(kwargs["messages"][0]["content"][1]["text"].split("\n")[-1])
        return SimpleNamespace(
            content=[
                {
                    "type": "tool_use",
                    "name": "submit_pid_objects",
                    "input": {
                        "objects": [],
                        "candidate_decisions": [
                            {
                                "candidate_id": c["candidate_id"],
                                "decision": "uncertain",
                                "reason": "Offline fixture capture",
                            }
                            for c in data["native_symbol_candidates"]
                        ],
                    },
                }
            ],
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
        )


def prepare():
    if (OUT / "manifest.json").exists():
        raise RuntimeError("Frozen benchmark already exists; do not silently change its inputs")
    bundle = json.loads((RUN / "detection.json").read_text())
    wb = json.loads((RUN / "workbench.json").read_text())
    source = Path(wb["source_path"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == bundle["source_sha256"]
    truth = [
        dict(t, page_index=t["page"] - 1)
        for t in json.loads((ROOT / "output/detection-audit/reviewed-instances.json").read_text())
        if t["category"] != "valve_callout"
    ]
    cfg = Config()
    strategy = AspectAwareStrategy(
        max_tokens_per_tile=cfg.tiling.max_tokens_per_tile,
        overlap_frac=cfg.tiling.overlap_frac,
        token_per_pixel=cfg.tiling.token_per_pixel,
    )
    cases = []
    with fitz.open(source) as pdf:
        for tid in TILES:
            pi = int(tid.split("-")[0][1:])
            ep = RUN / "evidence" / f"page-{pi + 1:04d}.json"
            evidence = PageEvidence.model_validate_json(ep.read_text())
            pix = pdf[pi].get_pixmap(
                matrix=fitz.Matrix(evidence.dpi / 72, evidence.dpi / 72), alpha=False
            )
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            assert image.size == (evidence.width, evidence.height)
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
                image=image,
            )
            tiles = tile(page, strategy)
            current = next(t for t in tiles if t.id == tid)
            core = ownership_core(current, tiles)
            original = json.loads(
                (RUN / "checkpoints/perception" / f"p{pi + 1:04d}__{tid}.json").read_text()
            )
            assert core.model_dump() == original["ownership_core"]
            candidates = [SymbolCandidate.model_validate(c) for c in original["native_candidates"]]
            view, info = ViewProvider(page, tiles).get_tile(tid)
            client = Capture()
            perceive_tile(
                client=client,
                cost_tracker=CostTracker(),
                reporter=NullReporter(),
                page=evidence,
                tile=current,
                view_image=view,
                view_info=info,
                ownership_bbox=core,
                legend_summary=bundle["legend_pack"]["entries"],
                step=1,
                page_context={"coverage": "deterministic fixed grid"},
                candidates=candidates,
            )
            request = client.request
            request.pop("thinking")
            request.update(
                model=wb["settings"]["vision_model"],
                max_tokens=12000,
                output_config={"effort": "low"},
                provider={
                    "only": ["fireworks"],
                    "allow_fallbacks": False,
                },
            )
            request["system"] = [
                {"type": "text", "text": request["system"], "cache_control": {"type": "ephemeral"}}
            ]
            request["tools"][-1]["cache_control"] = {"type": "ephemeral"}
            owned_truth = [
                t
                for t in truth
                if t["page_index"] == pi
                and core.x <= t["bbox"]["x"] + t["bbox"]["w"] / 2 < core.x2
                and core.y <= t["bbox"]["y"] + t["bbox"]["h"] / 2 < core.y2
            ]
            case = {
                "tile_id": tid,
                "page_index": pi,
                "tile": current.model_dump(mode="json", exclude={"image"}),
                "core": core.model_dump(),
                "view_info": {**asdict(info), "page_bbox": info.page_bbox.model_dump()},
                "candidates": [c.model_dump(mode="json") for c in candidates],
                "truth": owned_truth,
                "request_sha256": digest(request),
                "evidence_sha256": hashlib.sha256(ep.read_bytes()).hexdigest(),
            }
            save(OUT / "inputs" / f"{tid}.json", request)
            save(OUT / "cases" / f"{tid}.json", case)
            view.save(OUT / "cases" / f"{tid}.png")
            cases.append(case)
    manifest = {
        "source_sha256": bundle["source_sha256"],
        "source_run": str(RUN),
        "model": wb["settings"]["vision_model"],
        "provider": "fireworks",
        "repeats": 2,
        "maximum_calls": 36,
        "max_tokens": 12000,
        "effort": "low",
        "conditions": {
            "off": {"type": "disabled"},
            "on": {"type": "adaptive", "display": "summarized"},
        },
        "request_difference": "thinking only",
        "semantic_retries": 0,
        "http_retries": 0,
        "scoring": "geometry plus predeclared class-family rules, not subtype/tag accuracy",
        "cases": [
            {k: c[k] for k in ("tile_id", "request_sha256", "evidence_sha256")} for c in cases
        ],
        "reviewed_instances": sum(len(c["truth"]) for c in cases),
        "native_candidates": sum(len(c["candidates"]) for c in cases),
        "truth_sha256": digest([c["truth"] for c in cases]),
    }
    save(OUT / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def score_response(case, payload):
    ep = RUN / "evidence" / f"page-{case['page_index'] + 1:04d}.json"
    assert hashlib.sha256(ep.read_bytes()).hexdigest() == case["evidence_sha256"]
    p = PageEvidence.model_validate_json(ep.read_text())
    t = Tile.model_validate(case["tile"])
    v = dict(case["view_info"])
    v["page_bbox"] = BBox(**v["page_bbox"])
    info = ViewInfo(**v)
    try:
        batch = parse_perception_response(SimpleNamespace(**payload), page=p, view_info=info)
        detections = project_batch(
            batch=batch,
            page=p,
            tile=t,
            view_info=info,
            nearby_text_ids={s.id for s in text_spans_intersecting(p, t.bbox)},
            ownership_bbox=BBox(**case["core"]),
            candidates=[SymbolCandidate.model_validate(c) for c in case["candidates"]],
        )
        return {
            "detections": [d.model_dump(mode="json") for d in detections],
            "batch": batch.model_dump(mode="json"),
        }
    except Exception as exc:
        return {"detections": [], "parse_error": str(exc)}


def run(pilot=False):
    m = json.loads((OUT / "manifest.json").read_text())
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OpenRouter key unavailable")
    order = []
    for repeat in range(m["repeats"]):
        for index, entry in enumerate(m["cases"]):
            tid = entry["tile_id"]
            if repeat >= m.get("case_repeats", {}).get(tid, m["repeats"]):
                continue
            modes = ["off", "on"] if (repeat + index) % 2 == 0 else ["on", "off"]
            order.extend((repeat, tid, mode) for mode in modes)
    if pilot:
        order = order[:2]
    for repeat, tid, mode in order:
        target = OUT / "responses" / f"{tid}--{mode}--{repeat}.json"
        if target.exists():
            continue
        request = json.loads((OUT / "inputs" / f"{tid}.json").read_text())
        case = json.loads((OUT / "cases" / f"{tid}.json").read_text())
        assert digest(request) == case["request_sha256"]
        request["thinking"] = m["conditions"][mode]
        record = {
            "tile_id": tid,
            "repeat": repeat,
            "condition": mode,
            "request_sha256": digest(request),
            "shared_request_sha256": case["request_sha256"],
            "started_at": time.time(),
        }
        # Persist before dispatch. An interrupted/uncertain request is never
        # repeated automatically and cannot silently exceed the planned budget.
        save(target, record)
        try:
            with httpx.Client(timeout=300) as client:
                response = client.post(
                    "https://openrouter.ai/api/v1/messages",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "anthropic-version": "2023-06-01",
                        "X-OpenRouter-Title": "DiagEx bounded reasoning benchmark",
                    },
                    json=request,
                )
            record.update(
                http_status=response.status_code,
                response=response.json(),
                wall_seconds=time.time() - record["started_at"],
            )
            if response.is_success:
                record.update(score_response(case, record["response"]))
                record["thinking_chars"] = sum(
                    len(block.get("thinking", ""))
                    for block in record["response"].get("content", [])
                    if block.get("type") == "thinking"
                )
            save(target, record)
            print(
                json.dumps(
                    {
                        k: record.get(k)
                        for k in (
                            "tile_id",
                            "condition",
                            "repeat",
                            "http_status",
                            "wall_seconds",
                            "thinking_chars",
                            "parse_error",
                        )
                    }
                ),
                flush=True,
            )
            if not response.is_success:
                raise RuntimeError(f"Provider HTTP {response.status_code}; stopped without retry")
            if record.get("parse_error"):
                raise RuntimeError(
                    "No usable structured output; stopped for inspection without retry"
                )
        except Exception as exc:
            record.update(error=str(exc), wall_seconds=time.time() - record["started_at"])
            save(target, record)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "pilot", "run"])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        run(pilot=args.action == "pilot")
