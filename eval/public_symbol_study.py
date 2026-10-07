"""Source-only public symbol experiments. Prepare/run never read annotations.

Run production perception on a frozen raw crop, saving unvalidated proposals
separately from accepted detections. Scoring is a separate, offline command.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tarfile
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from diagex.config import LLMConfig, SymbolPerceptionConfig
from diagex.knowledge.resolver import knowledge_snapshot, resolve
from diagex.llm.billing import AccountedSpendingLedger, summarize
from diagex.llm.budget import BudgetExceeded
from diagex.llm.client import LLMClient
from diagex.llm.cost import CostTracker
from diagex.ui.progress import NullReporter
from diagex.vision.evidence import PageEvidence
from diagex.vision.models import BBox, Tile
from diagex.vision.views import ViewInfo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / "output/legend-symbol-improvement-20261005"
PUBLIC_ROOT = ROOT / "tests/p-ids-public"
WEBSITE = "projectmaterials.pid-symbols"
QUERY = "symbol valve pump equipment instrument connector"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def load(path):
    return json.loads(Path(path).read_text())


def checked_input(path):
    """Copy an explicit allowlist; arbitrary manifest prose never reaches models."""
    raw = load(path)
    pdf = Path(raw["source_pdf"]).resolve()
    if not pdf.is_relative_to(PUBLIC_ROOT.resolve()):
        raise ValueError("Only repository public PDF fixtures are eligible")
    if sha(pdf) != raw["source_sha256"]:
        raise ValueError("Public PDF checksum changed")
    image = Path(raw["image"]).resolve()
    if sha(image) != raw["image_sha256"]:
        raise ValueError("Source crop checksum changed")
    with Image.open(image) as img:
        size = list(img.size)
    if size != raw["image_pixel_size"] or max(size) > 2000:
        raise ValueError("Crop dimensions changed or exceed the fixed 2000-pixel view limit")
    core = BBox.model_validate(raw["evaluation_core_bbox_local"])
    if core.x < 0 or core.y < 0 or core.x2 > size[0] or core.y2 > size[1]:
        raise ValueError("Scoring core must lie inside the source crop")
    return {
        k: raw[k]
        for k in (
            "panel_id",
            "source_pdf",
            "source_sha256",
            "page_index",
            "render_dpi",
            "image",
            "image_sha256",
            "image_pixel_size",
            "context_bbox_global",
            "evaluation_core_bbox_local",
        )
    }


def baseline_module(study, relative, name):
    target = study / "public-harness-baseline" / relative
    with tarfile.open(study / "baseline-source.tar.gz") as archive:
        content = archive.extractfile(relative).read()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != content:
        raise ValueError("Archived baseline module was changed")
    target.write_bytes(content)
    spec = importlib.util.spec_from_file_location(name, target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def implementation(study, variant):
    from diagex.vision import symbol_interpretation

    if variant == "current":
        return symbol_interpretation
    return baseline_module(
        study,
        "src/diagex/vision/symbol_interpretation.py",
        "diagex.vision._public_baseline_symbol_interpretation",
    )


def ablate_reference_images(context, mode):
    """Study-only ablation; preserve definitions while withholding visual evidence."""
    from diagex.knowledge.library import model_assets
    from diagex.knowledge.resolver import digest

    if mode not in {"all", "none"}:
        raise ValueError("Unknown reference image policy")
    focused = deepcopy(context)
    audit = {"policy": mode, "withheld_model_assets": [], "withheld_reference_asset_ids": []}
    if mode == "all" or not focused:
        return focused, audit
    audit["withheld_model_assets"] = model_assets(focused)
    for entry in focused.get("references", []):
        audit["withheld_reference_asset_ids"].extend(
            {
                "reference_id": entry["id"],
                "reference_version": entry["version"],
                "asset_id": asset["id"],
                "sha256": asset.get("sha256"),
            }
            for asset in entry.get("assets", [])
        )
        if entry.get("source", {}).get("image") and not entry.get("assets"):
            audit["withheld_reference_asset_ids"].append(
                {
                    "reference_id": entry["id"],
                    "reference_version": entry["version"],
                    "asset_id": "legacy-image",
                    "sha256": entry.get("image_sha256"),
                }
            )
        entry["assets"] = []
        entry.setdefault("source", {})["image"] = None
        if "image_sha256" in entry:
            entry["image_sha256"] = None
        if entry.get("catalog"):
            entry["catalog"]["displayed_asset_ids"] = []
            entry["catalog"]["model_asset_id"] = None
    focused["reference_image_policy"] = "none"
    focused["reference_image_notice"] = (
        "Reference illustrations are deliberately withheld for this experiment. "
        "Only textual definitions are supplied. Do not claim a visual variant match."
    )
    focused["supplied_assets"] = model_assets(focused)
    focused["model_assets"] = []
    focused.pop("identity", None)
    focused["identity"] = digest(focused)
    assert not model_assets(focused)
    return focused, audit


def source_view_specs(size, grid):
    """Fixed source-only quadrants; no scoring annotations determine these views."""
    width, height = size
    if grid not in {1, 2} or min(size) < grid:
        raise ValueError("Grid must be 1 or 2 and fit the source image")
    margin_x, margin_y = int(round(width * 0.1)), int(round(height * 0.1))
    specs = []
    for row in range(grid):
        for col in range(grid):
            x, y = col * width // grid, row * height // grid
            right, bottom = (col + 1) * width // grid, (row + 1) * height // grid
            left, top = max(0, x - margin_x), max(0, y - margin_y)
            x2, y2 = min(width, right + margin_x), min(height, bottom + margin_y)
            specs.append(
                {
                    "id": "public-crop" if grid == 1 else f"public-grid-{row}-{col}",
                    "bbox": BBox(x=left, y=top, w=x2 - left, h=y2 - top).model_dump(),
                    "ownership_bbox": BBox(x=x, y=y, w=right - x, h=bottom - y).model_dump(),
                }
            )
    return specs


def prepare(args):
    return _prepare_checked_source(args, checked_input(args.input))


def _prepare_checked_source(args, source):
    """Shared preparation after a dataset-specific source-only integrity check."""
    if args.out.exists():
        raise ValueError("Choose a new prepared directory")
    module = implementation(args.study, args.variant)
    if args.backend == "broad_review" and (args.variant != "current" or not args.cv_guidance):
        raise ValueError("Broad review requires current production code and CV guidance")
    context = {}
    if args.knowledge == "website":
        snapshot = knowledge_snapshot("general", source_ids=[WEBSITE])
        # No unowned/general book references may cross this public-only boundary.
        snapshot["entries"] = [
            e for e in snapshot["entries"] if e.get("source", {}).get("document_id") == WEBSITE
        ]
        if args.variant == "baseline":
            resolver = baseline_module(
                args.study,
                "src/diagex/knowledge/resolver.py",
                "diagex.knowledge._public_baseline_resolver",
            ).resolve
        else:
            resolver = resolve
        context["knowledge"] = resolver(snapshot, "symbol_interpretation", query=QUERY)
        if any(
            e.get("source", {}).get("document_id") != WEBSITE
            for e in context["knowledge"].get("references", [])
        ):
            raise ValueError("Nonpublic knowledge entered the experiment")
    image_ablation = {
        "policy": args.reference_images,
        "withheld_model_assets": [],
        "withheld_reference_asset_ids": [],
    }
    if context.get("knowledge"):
        context["knowledge"], image_ablation = ablate_reference_images(
            context["knowledge"], args.reference_images
        )
    cv_sha = None
    if args.cv_guidance:
        from diagex.vision.raster_guidance import load_guidance, view_guidance

        guidance = load_guidance(args.cv_guidance, source["image"])
        width, height = source["image_pixel_size"]
        context.update(
            view_guidance(guidance["predictions"], BBox(x=0, y=0, w=width, h=height), (1, 1))
        )
        cv_sha = sha(args.cv_guidance)
    policy = SymbolPerceptionConfig(
        request_timeout_s=120, reasoning_timeout_s=120, transport_attempts=1, workflow=args.workflow
    )
    support = {
        str(p.relative_to(ROOT)): sha(p)
        for directory in ("src/diagex/knowledge", "src/diagex/vision", "src/diagex/llm")
        for p in (ROOT / directory).glob("*.py")
    }
    manifest = {
        "schema_version": 1,
        "input": source,
        "input_manifest_sha256": sha(args.input),
        "variant": args.variant,
        "backend": args.backend,
        "knowledge": args.knowledge,
        "reference_images": args.reference_images,
        "reference_image_ablation": image_ablation,
        "workflow": args.workflow,
        "grid": args.grid,
        "view_specs": source_view_specs(source["image_pixel_size"], args.grid),
        "grid_policy": "Equal source-image quadrants with a 10% full-image margin on each side; disjoint half-open center ownership. No annotations determine views.",
        "policy": asdict(policy),
        "context": context,
        "perception_module_sha256": sha(module.__file__),
        "support_code": support,
        "cv_guidance_path": str(args.cv_guidance.resolve()) if args.cv_guidance else None,
        "cv_guidance_sha256": cv_sha,
        "created_at": time.time(),
        "public_only": True,
        "legend_entries": [],
        "annotations_read": False,
        "baseline_scope": "Archived symbol_interpretation and knowledge resolver; shared support modules and current public library explicitly hashed. Not a whole historical application replay.",
        "deployment_difference": "Fixed 120-second request limit, one transport attempt, no semantic reasoning escalation; identical across compared variants.",
        "cv_scope": "Production broad_review route"
        if args.backend == "broad_review"
        else "Optional CV hint experiment in perceive_tile; not broad_review production routing",
    }
    save(args.out / "prepared.json", manifest)
    print(
        json.dumps(
            {
                "prepared": str(args.out),
                "panel": source["panel_id"],
                "variant": args.variant,
                "knowledge": args.knowledge,
                "backend": args.backend,
            },
            indent=2,
        )
    )


class BoundedLedger(AccountedSpendingLedger):
    """Atomically enforce run exposure/call caps and final-validation reserve."""

    def __init__(self, *args, run_cap, calls_cap, final_reserve, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_cap, self.calls_cap, self.final_reserve = run_cap, calls_cap, final_reserve

    @contextmanager
    def _locked(self):
        with super()._locked() as state:
            original_count = len(state["requests"])
            yield state
            if len(state["requests"]) <= original_count:
                return
            own = [r for r in state["requests"] if r["category"] == self.category]
            if len(own) > self.calls_cap or sum(r["exposure_usd"] for r in own) > self.run_cap:
                raise BudgetExceeded("Public experiment run cap reached before dispatch")
            if (
                state["prior_spend"]["usd"] + summarize(state)["budget_exposure_usd"]
                > state["limit_usd"] - self.final_reserve
            ):
                raise BudgetExceeded("Preserve the final validation allowance before dispatch")


def verify_prepared(prepared):
    source = prepared["input"]
    if (
        sha(source["source_pdf"]) != source["source_sha256"]
        or sha(source["image"]) != source["image_sha256"]
    ):
        raise ValueError("Frozen public source changed")
    for path, expected in prepared["support_code"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Prepared code changed: {path}; prepare a new variant")
    if (
        prepared["cv_guidance_path"]
        and sha(prepared["cv_guidance_path"]) != prepared["cv_guidance_sha256"]
    ):
        raise ValueError("CV guidance changed")
    if prepared.get("grid", 1) == 2 and prepared.get("view_specs") != source_view_specs(
        source["image_pixel_size"], 2
    ):
        raise ValueError("Prepared grid differs from the fixed source-only partition")


def invoke(prepared, module, client, diagnostics):
    if prepared.get("grid", 1) == 1:
        return invoke_view(prepared, module, client, diagnostics)
    results, failures = [], []
    for spec in prepared["view_specs"]:
        view_diagnostics = []
        try:
            result = invoke_view(prepared, module, client, view_diagnostics, spec)
            results.append({"view": spec, **result})
        except Exception as exc:
            failures.append(
                {"view_id": spec["id"], "error_type": type(exc).__name__, "error": str(exc)}
            )
            if isinstance(exc, BudgetExceeded):
                break
        finally:
            diagnostics.extend({"view_id": spec["id"], **item} for item in view_diagnostics)
    return {
        "detections": [d for result in results for d in result["detections"]],
        "proposals": [p for result in results for p in result["proposals"]],
        "batch": {
            "uncertainties": [
                u for result in results for u in result["batch"].get("uncertainties", [])
            ],
            "candidate_reviews": [
                r for result in results for r in result["batch"].get("candidate_reviews", [])
            ],
        },
        "per_view": results,
        "view_failures": failures,
        "attempts": sum(result["attempts"] for result in results),
        "contract_failed": bool(failures) or any(result["contract_failed"] for result in results),
        "recovery_diagnostics": [d for result in results for d in result["recovery_diagnostics"]],
        "coordinate_frame": "original_source_crop_pixels",
    }


def invoke_view(prepared, module, client, diagnostics, spec=None):
    """Model inputs contain only source pixels, fixed scope, and selected references."""
    source = prepared["input"]
    image = Image.open(source["image"]).convert("RGB")
    width, height = image.size
    box = BBox.model_validate(spec["bbox"]) if spec else BBox(x=0, y=0, w=width, h=height)
    ownership = BBox.model_validate(spec["ownership_bbox"]) if spec else box
    view_image = image.crop((box.x, box.y, box.x2, box.y2)) if spec else image
    page = PageEvidence(
        page_index=0,
        source_ref="public-source-crop",
        width=width,
        height=height,
        dpi=source["render_dpi"],
        effective_dpi=source["render_dpi"],
        is_scanned=True,
        role="pid",
        text_spans=[],
        paths=[],
    )
    tile = Tile(id=spec["id"] if spec else "public-crop", page_index=0, bbox=box, image=view_image)
    view = ViewInfo("tile", (box.x, box.y), 1, 1, view_image.size, box, tile.id)
    context = deepcopy(prepared["context"])
    if spec and prepared.get("cv_guidance_path"):
        from diagex.vision.raster_guidance import load_guidance, view_guidance

        guidance = load_guidance(prepared["cv_guidance_path"], source["image"])
        context.update(view_guidance(guidance["predictions"], box, (1, 1)))
    policy = SymbolPerceptionConfig(**prepared["policy"])
    kwargs = dict(
        client=client,
        cost_tracker=CostTracker(),
        reporter=NullReporter(),
        page=page,
        tile=tile,
        view_image=view_image,
        view_info=view,
        ownership_bbox=ownership,
        legend_summary=[],
        step=1,
        page_context=context,
        candidates=[],
        reasoning_mode="disabled",
        policy=policy,
        on_diagnostic=diagnostics.append,
        deadline=time.monotonic() + 360,
    )
    if policy.workflow == "adaptive":

        def region(x, y, w, h):
            left, top = max(0, int(x)), max(0, int(y))
            right, bottom = min(width, int(x + w)), min(height, int(y + h))
            area = BBox(x=left, y=top, w=max(1, right - left), h=max(1, bottom - top))
            crop = image.crop((area.x, area.y, area.x2, area.y2))
            return crop, ViewInfo("region", (left, top), 1, 1, crop.size, area)

        kwargs.update(overview_image=image, region_provider=region)
    if prepared["backend"] == "broad_review":
        from diagex.vision.raster_pipeline import perceive_broad_with_semantics

        outcome = perceive_broad_with_semantics(**kwargs)
    else:
        outcome = module.perceive_tile(**kwargs)
    proposals = []
    for index, obj in enumerate(outcome.batch.objects):
        if obj.bbox is None:
            continue
        bbox = module._project_normalized_bbox(obj.bbox, page=page, view_info=view)
        if spec and not module._bbox_center_is_owned(bbox, ownership, page):
            continue
        proposals.append(
            {
                "id": f"{tile.id}-proposal-{index}" if spec else f"proposal-{index}",
                "kind": obj.kind,
                "label": obj.label,
                "bbox": bbox.model_dump(),
                "attributes": obj.graph_attributes(),
                "confidence": obj.confidence,
                "validation_status": "unvalidated_model_proposal",
            }
        )
    # Broad production routing has its raw inventory in candidate_reviews.
    if prepared["backend"] == "broad_review":
        proposals = [
            {
                **r.get("object", {}),
                "id": f"{tile.id}-proposal-{i}" if spec else f"proposal-{i}",
                "bbox": r["bbox"],
                "validation_status": "unvalidated_model_proposal",
            }
            for i, r in enumerate(outcome.batch.candidate_reviews)
            if r.get("object")
            and r.get("bbox")
            and (
                not spec
                or module._bbox_center_is_owned(BBox.model_validate(r["bbox"]), ownership, page)
            )
        ]
    return {
        "detections": [d.model_dump(mode="json") for d in outcome.detections],
        "proposals": proposals,
        "batch": outcome.batch.model_dump(mode="json"),
        "attempts": outcome.attempts,
        "contract_failed": outcome.contract_failed,
        "recovery_diagnostics": outcome.recovery_diagnostics,
    }


def select_provider(prices, model, provider_tag=None):
    """Narrow routing in memory without changing verified reservation ceilings."""
    price = prices.get(model)
    if price is None or time.time() > float(price["valid_until"]):
        raise BudgetExceeded("Verified model pricing unavailable or expired")
    verified = list(price.get("provider_tags", []))
    if not verified:
        raise BudgetExceeded("Verified provider allowlist unavailable")
    if provider_tag is not None and provider_tag not in verified:
        raise ValueError("Requested provider is outside the verified provider allowlist")
    effective = [provider_tag] if provider_tag is not None else verified
    prices[model] = {**deepcopy(price), "provider_tags": effective}
    return {
        "requested_provider_tag": provider_tag,
        "verified_provider_tags": verified,
        "effective_provider_tags": effective,
        "verified_at": price.get("verified_at"),
        "valid_until": price["valid_until"],
        "unchanged_price_ceilings": {
            key: price.get(key, 0)
            for key in ("context_length", "input_per_token", "output_per_token", "request_usd")
        },
        "effective_endpoint_tool_support": [
            {
                "tag": endpoint.get("tag"),
                "supports_tool_choice": endpoint.get("supports_tool_choice"),
            }
            for endpoint in price.get("endpoint_snapshots", [])
            if endpoint.get("tag") in effective
        ],
    }


def controlled_tool_choice(request, mode):
    """Study-only transport control; leave the production response contract intact."""
    if mode not in {"auto", "forced"}:
        raise ValueError("Unknown tool choice policy")
    if mode == "auto" and request.get("tools"):
        return {**request, "tool_choice": {"type": "auto"}}
    return dict(request)


def run(args):
    prepared = load(args.prepared / "prepared.json")
    verify_prepared(prepared)
    module = implementation(args.study, prepared["variant"])
    if sha(module.__file__) != prepared["perception_module_sha256"]:
        raise ValueError("Frozen perception implementation changed")
    if args.out.exists():
        raise ValueError("Existing run may contain billable attempts; use a new run directory")
    if args.max_usd <= 0 or args.max_calls < 1:
        raise ValueError("Run limits must be positive")
    category = "public_symbol_" + uuid.uuid4().hex
    cfg = LLMConfig.from_env()
    cfg.transport, cfg.model, cfg.vision_model, cfg.reasoning_model = (
        "openrouter",
        args.model,
        args.model,
        args.model,
    )
    cfg.reasoning_mode, cfg.production_open_weight = "disabled", False
    cfg.spending_ledger = str(args.study / "spending.json")
    cfg.verified_prices = str(args.study / "verified-prices.json")
    cfg.spending_category = category
    if load(cfg.spending_ledger).get("schema_version") != 2:
        raise ValueError("Use the existing cumulative schema-version-2 ledger")
    client = LLMClient(cfg)
    client.spending = BoundedLedger(
        cfg.spending_ledger,
        cfg.verified_prices,
        category,
        run_cap=args.max_usd,
        calls_cap=args.max_calls,
        final_reserve=0 if args.final_validation else 5,
    )
    provider_control = select_provider(client.spending.prices, args.model, args.provider_tag)
    # Price validation/reservations remain in the same metered production client.
    save(
        args.out / "manifest.json",
        {
            "prepared": prepared,
            "harness_sha256": sha(__file__),
            "public_source_attribution": prepared.get("source_attribution") or (load(PUBLIC_ROOT / "SOURCES.json")
            if (PUBLIC_ROOT / "SOURCES.json").exists()
            else {
                "record": str(PUBLIC_ROOT / "SOURCES.md"),
                "sha256": sha(PUBLIC_ROOT / "SOURCES.md"),
            }),
            "prepared_sha256": sha(args.prepared / "prepared.json"),
            "model": args.model,
            "provider_control": provider_control,
            "tool_choice_policy": args.tool_choice,
            "response_contract": "Unchanged production tool-name and schema validation",
            "spending_category": category,
            "max_usd": args.max_usd,
            "max_calls": args.max_calls,
            "final_validation": args.final_validation,
            "prices_sha256": sha(cfg.verified_prices),
            "started_at": time.time(),
        },
    )
    original = client.messages_create
    calls = []

    def dispatch(**kwargs):
        kwargs = controlled_tool_choice(kwargs, args.tool_choice)
        index = len(calls) + 1
        request = {k: v for k, v in kwargs.items() if not callable(v)}
        save(args.out / f"request-{index:02}.json", request)
        record = {"call": index, "started_at": time.time()}
        calls.append(record)
        try:
            response = original(**kwargs)
            save(args.out / f"response-{index:02}.json", response.model_dump(mode="json"))
            record["status"] = "completed"
            return response
        except Exception as exc:
            record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            record["elapsed_s"] = time.time() - record["started_at"]
            save(args.out / "calls.json", calls)

    client.messages_create = dispatch
    diagnostics, started = [], time.monotonic()
    result = {"status": "failed"}
    try:
        result = invoke(prepared, module, client, diagnostics)
        result["status"] = "partial" if result.get("view_failures") else "completed"
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
    finally:
        state = load(cfg.spending_ledger)
        ids = [r["id"] for r in state["requests"] if r["category"] == category]
        result.update(
            elapsed_s=time.monotonic() - started, request_ids=ids, billing=summarize(state, ids)
        )
        save(args.out / "result.json", result)
        save(args.out / "diagnostics.json", diagnostics)
    print(
        json.dumps(
            {
                "status": result["status"],
                "detections": len(result.get("detections", [])),
                "proposals": len(result.get("proposals", [])),
                "billing": result["billing"],
                "out": str(args.out),
            },
            indent=2,
        )
    )


def box_iou(left, right):
    a, b = BBox.model_validate(left), BBox.model_validate(right)
    intersection = max(0, min(a.x2, b.x2) - max(a.x, b.x)) * max(0, min(a.y2, b.y2) - max(a.y, b.y))
    return intersection / max(1, a.w * a.h + b.w * b.h - intersection)


def _in_core(box, core):
    return (
        core["x"] <= box["x"] + box["w"] / 2 < core["x"] + core["w"]
        and core["y"] <= box["y"] + box["h"] / 2 < core["y"] + core["h"]
    )


def broad_class(row):
    if row.get("kind") == "raster_symbol":
        category = row.get("attributes", {}).get("broad_category", "unclassified")
        return "opc" if category == "inlet/outlet" else category
    if row.get("kind") == "instrument":
        return "instrumentation"
    if row.get("kind") == "opc":
        return "opc"
    value = (
        row.get("attributes", {}).get("equipment_class")
        or row.get("equipment_class")
        or "unclassified"
    )
    value = str(value).lower().replace("-", "_").replace(" ", "_")
    if value.endswith("_valve"):
        return "valve"
    return {
        "shell_and_tube_heat_exchanger": "heat_exchanger",
        "shell_tube_heat_exchanger": "heat_exchanger",
        "plate_heat_exchanger": "heat_exchanger",
        "off_page_connector": "opc",
    }.get(value, value)


def maximum_matches(predictions, targets, threshold):
    """Maximum-cardinality IoU-qualified matching, deterministic IoU-first ties."""
    edges = {
        i: sorted(
            ((box_iou(p["bbox"], t["bbox_context"]), j) for j, t in enumerate(targets)),
            reverse=True,
        )
        for i, p in enumerate(predictions)
    }
    owner = {}

    def augment(i, seen):
        for overlap, j in edges[i]:
            if overlap < threshold or j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    for i in range(len(predictions)):
        augment(i, set())
    return sorted(
        (i, j, box_iou(predictions[i]["bbox"], targets[j]["bbox_context"]))
        for j, i in owner.items()
    )


def inventory_metrics(predictions, annotations, source, threshold):
    """Scoring only: source-reviewed annotations never enter prepare/invoke."""
    core = source["evaluation_core_bbox_local"]
    owned = [p for p in predictions if _in_core(p["bbox"], core)]
    targets = annotations["targets"]
    pairs = maximum_matches(owned, targets, threshold)
    matched = {i for i, j, v in pairs}
    origin = source["context_bbox_global"]
    excluded = []
    for role in ("ambiguous_glyphs", "secondary_components_and_exclusions"):
        for row in annotations.get(role, []):
            b = row.get("bbox_context")
            if b is None:
                g = row["bbox_global"]
                b = {**g, "x": g["x"] - origin["x"], "y": g["y"] - origin["y"]}
            excluded.append(
                {
                    "id": row["id"],
                    "role": role,
                    "bbox": b,
                    "negative": role != "ambiguous_glyphs"
                    and (
                        row.get("role") == "valve identity callout"
                        or "not independent instrument" in row.get("scoring", "").lower()
                        or "not physical equipment" in row.get("scoring", "").lower()
                    ),
                }
            )
    ignored, false, negative_hits = [], [], []
    for i, p in enumerate(owned):
        if i in matched:
            continue
        # A giant box cannot disappear merely by enclosing a tiny ignored glyph.
        alternatives = [(box_iou(p["bbox"], e["bbox"]), e) for e in excluded]
        overlap, exclusion = max(alternatives, key=lambda x: x[0], default=(0, None))
        if overlap >= threshold and not exclusion["negative"]:
            ignored.append(
                {
                    "prediction_id": p["id"],
                    "annotation_id": exclusion["id"],
                    "role": exclusion["role"],
                    "iou": overlap,
                }
            )
        else:
            false.append(i)
            if overlap >= threshold and exclusion["negative"]:
                negative_hits.append(
                    {"prediction_id": p["id"], "annotation_id": exclusion["id"], "iou": overlap}
                )
    details = []
    for i, j, overlap in pairs:
        p, t = owned[i], targets[j]
        tag = t.get("tag_or_reference")

        def normal(value):
            return "".join(c for c in str(value or "").upper() if c.isalnum())

        predicted_tag = (
            p.get("label") or p.get("printed_tag") or p.get("raw_text") or p.get("canonical_tag")
        )
        if not predicted_tag:
            predicted_tag = p.get("attributes", {}).get("drawing_ref") or p.get("drawing_ref")
        details.append(
            {
                "prediction_id": p["id"],
                "target_id": t["id"],
                "iou": overlap,
                "expected_class": t["broad_class"],
                "predicted_class": broad_class(p),
                "class_correct": broad_class(p) == t["broad_class"],
                "expected_tag": tag,
                "predicted_tag": predicted_tag,
                "tag_exact_normalized": normal(predicted_tag) == normal(tag) if tag else None,
                "box_area_ratio": p["bbox"]["w"]
                * p["bbox"]["h"]
                / max(1, t["bbox_context"]["w"] * t["bbox_context"]["h"]),
            }
        )
    true, fp, fn = len(pairs), len(false), len(targets) - len(pairs)
    precision = true / (true + fp) if true + fp else 0
    recall = true / (true + fn) if true + fn else 0
    tagged = [d for d in details if d["tag_exact_normalized"] is not None]
    duplicate = [
        owned[i]["id"]
        for i in false
        if any(box_iou(owned[i]["bbox"], t["bbox_context"]) >= threshold for t in targets)
    ]
    attrs = [p.get("attributes", {}) for p in owned]
    per_class = {}
    for cls in sorted({t["broad_class"] for t in targets}):
        these = [d for d in details if d["expected_class"] == cls]
        count = sum(t["broad_class"] == cls for t in targets)
        per_class[cls] = {
            "targets": count,
            "localized": len(these),
            "correctly_interpreted": sum(d["class_correct"] for d in these),
        }
    return {
        "target_count": len(targets),
        "owned_prediction_count": len(owned),
        "outside_core_count": len(predictions) - len(owned),
        "true_positive_localization": true,
        "false_positive_localization": fp,
        "false_negative_localization": fn,
        "precision_localization": precision,
        "recall_localization": recall,
        "f1_localization": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0,
        "mean_iou_of_matches": sum(d["iou"] for d in details) / true if true else None,
        "interpretation_accuracy_on_localized": sum(d["class_correct"] for d in details) / true
        if true
        else None,
        "tag_exact_normalized_on_tagged_matches": sum(d["tag_exact_normalized"] for d in tagged)
        / len(tagged)
        if tagged
        else None,
        "tagged_match_count": len(tagged),
        "duplicate_prediction_ids": duplicate,
        "false_positive_prediction_ids": [owned[i]["id"] for i in false],
        "missed_target_ids": [
            t["id"] for j, t in enumerate(targets) if j not in {j for i, j, v in pairs}
        ],
        "ignored_secondary_or_ambiguous": ignored,
        "negative_example_hits": negative_hits,
        "scoring_policy": "v2: explicit non-object callouts/break marks count as false positives; passive fittings and ambiguous decomposition are ignored",
        "matches": details,
        "per_class": per_class,
        "attribution_trace": {
            "with_knowledge_match": sum(bool(a.get("knowledge_matches")) for a in attrs),
            "invalid_match_reports": sum(bool(a.get("knowledge_match_errors")) for a in attrs),
            "with_legend_match": sum(bool(a.get("legend_entry_ids")) for a in attrs),
            "meaning": "Trace counts, not source correctness judgments; semantic attribution requires visual review.",
        },
    }


def render_overlay(path, source, annotations, predictions, title):
    """Evaluation artifact only. Never replaces the frozen model input image."""
    from PIL import ImageDraw

    image = Image.open(source["image"]).convert("RGB")
    margin = 35
    canvas = Image.new("RGB", (image.width * 2, image.height + margin), "white")
    canvas.paste(image, (0, margin))
    canvas.paste(image, (image.width, margin))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 8), "Source-checked annotations (green); fixed evaluation core (gray)", fill="black"
    )
    draw.text((image.width + 8, 8), title + " (blue)", fill="black")
    core = source["evaluation_core_bbox_local"]
    for offset in (0, image.width):
        draw.rectangle(
            (
                offset + core["x"],
                margin + core["y"],
                offset + core["x"] + core["w"],
                margin + core["y"] + core["h"],
            ),
            outline="gray",
            width=2,
        )
    for row in annotations["targets"]:
        b = row["bbox_context"]
        draw.rectangle(
            (b["x"], margin + b["y"], b["x"] + b["w"], margin + b["y"] + b["h"]),
            outline="green",
            width=3,
        )
        draw.text(
            (b["x"], margin + max(0, b["y"] - 12)),
            row["id"],
            fill="green",
            stroke_width=1,
            stroke_fill="white",
        )
    for index, row in enumerate(predictions):
        b = row["bbox"]
        if not _in_core(b, core):
            continue
        x = image.width + b["x"]
        draw.rectangle(
            (x, margin + b["y"], x + b["w"], margin + b["y"] + b["h"]), outline="blue", width=2
        )
        label = row.get("label") or broad_class(row)
        draw.text(
            (x, margin + max(0, b["y"] - 12)),
            f"{index}: {label}",
            fill="blue",
            stroke_width=1,
            stroke_fill="white",
        )
    canvas.save(path)


def score_cv(args):
    from diagex.vision.raster_guidance import load_guidance

    source = checked_input(args.input)
    guidance = load_guidance(args.guidance, source["image"])
    annotations = load(args.annotations)
    if (
        annotations["panel_id"] != source["panel_id"]
        or annotations["source"]["pdf_sha256"] != source["source_sha256"]
        or annotations["region"]["inference_image_sha256"] != source["image_sha256"]
    ):
        raise ValueError("CV scoring annotations do not match the public source")
    if not 0 < args.iou <= 1:
        raise ValueError("IoU threshold must be in (0,1]")
    predictions = []
    for row in guidance["predictions"]:
        x, y, x2, y2 = row["bbox"]
        label = row["label"]
        kind = (
            "instrument"
            if label == "instrumentation"
            else "opc"
            if label in ("arrow", "inlet/outlet")
            else "equipment"
        )
        predictions.append(
            {
                "id": row["id"],
                "bbox": {
                    "x": round(x),
                    "y": round(y),
                    "w": max(1, round(x2 - x)),
                    "h": max(1, round(y2 - y)),
                },
                "kind": kind,
                "label": "",
                "attributes": {"equipment_class": label},
                "confidence": row["confidence"],
            }
        )
    report = {
        "scope": "Unvalidated CV proposals, not production detections. No training or model calls.",
        "annotation_file_sha256": sha(args.annotations),
        "guidance_sha256": sha(args.guidance),
        "detector": guidance["detector"],
        "elapsed_s": guidance.get("runtime_seconds"),
        "iou_threshold": args.iou,
        "metrics": inventory_metrics(predictions, annotations, source, args.iou),
    }
    if args.out.exists():
        raise ValueError("Choose a new CV score artifact")
    save(args.out, report)
    render_overlay(
        args.out.with_suffix(".overlay.png"),
        source,
        annotations,
        predictions,
        "Unvalidated CV proposals",
    )
    print(json.dumps(report, indent=2))


def score(args):
    run_manifest = load(args.run / "manifest.json")
    result = load(args.run / "result.json")
    prepared = run_manifest["prepared"]
    source = prepared["input"]
    annotations = load(args.annotations)
    if annotations["panel_id"] != source["panel_id"]:
        raise ValueError("Annotations belong to another panel")
    if (
        annotations["source"]["pdf_sha256"] != source["source_sha256"]
        or annotations["region"]["inference_image_sha256"] != source["image_sha256"]
        or annotations["region"]["context_bbox_global"] != source["context_bbox_global"]
    ):
        raise ValueError("Annotation source geometry/checksums differ from inference")
    if not 0 < args.iou <= 1:
        raise ValueError("IoU threshold must be in (0,1]")
    report = {
        "panel_id": source["panel_id"],
        "run_status": result["status"],
        "run": str(args.run.resolve()),
        "annotation_file_sha256": sha(args.annotations),
        "annotation_status": annotations.get("status"),
        "annotator": annotations.get("annotator"),
        "iou_threshold": args.iou,
        "scope": "Small source-reviewed public crop; no independent expert ground truth. Raw proposals and production outputs scored separately. Not whole-page or topology accuracy.",
        "run_result_sha256": sha(args.run / "result.json"),
        "prepared_sha256": run_manifest["prepared_sha256"],
        "billing": result["billing"],
        "elapsed_s": result["elapsed_s"],
    }
    for name, key in [("raw_proposals", "proposals"), ("production_detections", "detections")]:
        report[name] = inventory_metrics(result.get(key, []), annotations, source, args.iou)
    report["uncertainty"] = {
        "model_uncertainties": result.get("batch", {}).get("uncertainties", []),
        "candidate_reviews": result.get("batch", {}).get("candidate_reviews", []),
        "ambiguous_source_glyph_count": len(annotations.get("ambiguous_glyphs", [])),
    }
    output = args.out or args.run / "score.json"
    if output.exists():
        raise ValueError("Score output already exists; select a new artifact")
    save(output, report)
    render_overlay(
        output.with_suffix(".overlay.png"),
        source,
        annotations,
        result.get("detections", []),
        "Production detections",
    )
    print(
        json.dumps(
            {
                name: report[name]
                for name in ("panel_id", "run_status", "raw_proposals", "production_detections")
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--variant", choices=("baseline", "current"), default="current")
    p.add_argument("--knowledge", choices=("off", "website"), default="off")
    p.add_argument("--reference-images", choices=("all", "none"), default="all")
    p.add_argument("--workflow", choices=("fixed", "adaptive"), default="fixed")
    p.add_argument("--grid", type=int, choices=(1, 2), default=1)
    p.add_argument("--backend", choices=("baseline", "broad_review"), default="baseline")
    p.add_argument("--cv-guidance", type=Path)
    p = sub.add_parser("run")
    p.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    p.add_argument("--provider-tag", help="Restrict routing to one currently verified provider tag")
    p.add_argument("--tool-choice", choices=("auto", "forced"), default="forced")
    p.add_argument("--max-usd", type=float, default=1)
    p.add_argument("--max-calls", type=int, default=3)
    p.add_argument("--final-validation", action="store_true")
    p = sub.add_parser("score")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--iou", type=float, default=0.5)
    p = sub.add_parser("score-cv")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--guidance", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    globals()[args.command.replace("-", "_")](args)


if __name__ == "__main__":
    main()
