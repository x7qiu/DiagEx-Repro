"""Offline contracts for the public-only model study and its spending guard."""

import json
import time
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from PIL import Image

from diagex.config import SymbolPerceptionConfig
from diagex.llm.budget import BudgetExceeded
from eval import public_symbol_study as study


def ledger(tmp_path, **limits):
    state = {"schema_version": 2, "limit_usd": 10, "prior_spend": {"usd": 0}, "requests": []}
    study.save(tmp_path / "ledger.json", state)
    study.save(
        tmp_path / "prices.json",
        {
            "models": {
                "model": {
                    "context_length": 100,
                    "input_per_token": 0.01,
                    "output_per_token": 0.01,
                    "valid_until": time.time() + 1000,
                }
            }
        },
    )
    return study.BoundedLedger(
        tmp_path / "ledger.json",
        tmp_path / "prices.json",
        "run",
        run_cap=limits.get("run_cap", 3),
        calls_cap=limits.get("calls_cap", 2),
        final_reserve=limits.get("final_reserve", 5),
    )


def test_atomic_run_cap_and_final_reserve(tmp_path):
    guard = ledger(tmp_path, run_cap=1.5)
    with pytest.raises(BudgetExceeded, match="run cap"):
        guard.reserve("model", 100)
    assert study.load(guard.path)["requests"] == []
    guard = ledger(tmp_path, run_cap=8, final_reserve=9)
    with pytest.raises(BudgetExceeded, match="validation allowance"):
        guard.reserve("model", 1)
    assert study.load(guard.path)["requests"] == []


def test_settlement_does_not_count_as_another_call(tmp_path):
    guard = ledger(tmp_path, calls_cap=1)
    identity = guard.reserve("model", 10)
    guard.settle(identity, {"id": "gen-1", "usage": {"cost": 0.1}})
    with pytest.raises(BudgetExceeded, match="run cap"):
        guard.reserve("model", 10)
    rows = study.load(guard.path)["requests"]
    assert len(rows) == 1 and rows[0]["actual_billed_usd"] == 0.1


def test_source_allowlist_does_not_forward_arbitrary_manifest_prose(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "PUBLIC_ROOT", tmp_path)
    pdf, image = tmp_path / "public.pdf", tmp_path / "source.png"
    pdf.write_bytes(b"public fixture")
    Image.new("RGB", (100, 100), "white").save(image)
    raw = {
        "panel_id": "public",
        "source_pdf": str(pdf),
        "source_sha256": study.sha(pdf),
        "page_index": 0,
        "render_dpi": 300,
        "image": str(image),
        "image_sha256": study.sha(image),
        "image_pixel_size": [100, 100],
        "context_bbox_global": {"x": 0, "y": 0, "w": 100, "h": 100},
        "evaluation_core_bbox_local": {"x": 10, "y": 10, "w": 80, "h": 80},
        "annotations": [{"label": "FORBIDDEN_GOLD"}],
        "scope": "FORBIDDEN_GOLD",
    }
    study.save(tmp_path / "input.json", raw)
    result = study.checked_input(tmp_path / "input.json")
    assert "FORBIDDEN_GOLD" not in json.dumps(result)
    monkeypatch.setattr(study, "PUBLIC_ROOT", tmp_path / "other")
    with pytest.raises(ValueError, match="public PDF"):
        study.checked_input(tmp_path / "input.json")


def test_production_invocation_sends_source_only_and_keeps_proposals_unvalidated(tmp_path):
    from diagex.vision import symbol_interpretation as module

    image = tmp_path / "source.png"
    Image.new("RGB", (200, 100), "white").save(image)
    requests = []

    class Client:
        def messages_create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                content=[
                    {
                        "type": "tool_use",
                        "name": "submit_pid_objects",
                        "input": {
                            "candidate_results": [],
                            "proposals": [
                                {
                                    "kind": "equipment",
                                    "equipment_class": "valve",
                                    "bbox": {"x": 0.2, "y": 0.2, "w": 0.1, "h": 0.2},
                                    "confidence": "medium",
                                }
                            ],
                        },
                    }
                ],
                usage=None,
            )

    prepared = {
        "input": {"image": str(image), "render_dpi": 300, "annotations": "FORBIDDEN_GOLD"},
        "policy": asdict(SymbolPerceptionConfig(transport_attempts=1, request_timeout_s=120)),
        "context": {},
        "backend": "baseline",
    }
    result = study.invoke(prepared, module, Client(), [])
    assert len(result["detections"]) == 1 and len(result["proposals"]) == 1
    assert result["proposals"][0]["bbox"] == {"x": 40, "y": 20, "w": 20, "h": 20}
    assert result["proposals"][0]["validation_status"] == "unvalidated_model_proposal"
    assert len(requests) == 1
    encoded = json.dumps({k: v for k, v in requests[0].items() if not callable(v)})
    assert "FORBIDDEN_GOLD" not in encoded and "2401" not in encoded
    assert requests[0]["max_attempts"] == 1 and requests[0]["time_budget_s"] == 120


def test_scoring_one_to_one_duplicates_ambiguity_and_scope():
    def prediction(identity, x, cls="valve", y=10, size=10):
        return {
            "id": identity,
            "bbox": {"x": x, "y": y, "w": size, "h": size},
            "kind": "equipment",
            "attributes": {"equipment_class": cls},
            "label": "V-1",
        }

    source = {
        "evaluation_core_bbox_local": {"x": 0, "y": 0, "w": 100, "h": 100},
        "context_bbox_global": {"x": 100, "y": 100, "w": 100, "h": 100},
    }
    annotations = {
        "targets": [
            {
                "id": "a",
                "bbox_context": {"x": 10, "y": 10, "w": 10, "h": 10},
                "broad_class": "valve",
                "tag_or_reference": "V 1",
            },
            {
                "id": "b",
                "bbox_context": {"x": 40, "y": 10, "w": 10, "h": 10},
                "broad_class": "valve",
                "tag_or_reference": None,
            },
        ],
        "ambiguous_glyphs": [
            {"id": "ambiguous", "bbox_global": {"x": 170, "y": 110, "w": 10, "h": 10}}
        ],
    }
    predictions = [
        prediction("a", 10),
        prediction("duplicate", 10),
        prediction("b", 40, "pump"),
        prediction("ambiguous", 70),
        prediction("outside", 120),
    ]
    result = study.inventory_metrics(predictions, annotations, source, 0.5)
    assert result["true_positive_localization"] == 2
    assert result["false_positive_localization"] == 1
    assert result["recall_localization"] == 1
    assert result["precision_localization"] == pytest.approx(2 / 3)
    assert result["interpretation_accuracy_on_localized"] == 0.5
    assert result["tag_exact_normalized_on_tagged_matches"] == 1
    assert len(result["ignored_secondary_or_ambiguous"]) == 1
    assert len(result["duplicate_prediction_ids"]) == 1
    assert result["outside_core_count"] == 1
    # A large enclosing box must not be hidden by a tiny ambiguous glyph.
    result = study.inventory_metrics([prediction("giant", 50, size=40)], annotations, source, 0.5)
    assert result["false_positive_localization"] == 1


def test_matching_maximizes_cardinality_without_reusing_target():
    predictions = [
        {"bbox": {"x": 0, "y": 0, "w": 40, "h": 20}},
        {"bbox": {"x": 0, "y": 0, "w": 20, "h": 20}},
    ]
    targets = [
        {"bbox_context": {"x": 0, "y": 0, "w": 20, "h": 20}},
        {"bbox_context": {"x": 20, "y": 0, "w": 20, "h": 20}},
    ]
    assert len(study.maximum_matches(predictions, targets, 0.5)) == 2


def test_prepared_code_changes_are_refused(tmp_path, monkeypatch):
    file = tmp_path / "code.py"
    file.write_text("before")
    image = tmp_path / "image.png"
    image.write_bytes(b"input")
    monkeypatch.setattr(study, "ROOT", tmp_path)
    prepared = {
        "input": {
            "source_pdf": str(image),
            "source_sha256": study.sha(image),
            "image": str(image),
            "image_sha256": study.sha(image),
        },
        "support_code": {"code.py": study.sha(file)},
        "cv_guidance_path": None,
    }
    study.verify_prepared(prepared)
    file.write_text("changed")
    with pytest.raises(ValueError, match="Prepared code changed"):
        study.verify_prepared(prepared)


def test_explicit_nonobject_examples_are_false_positives_not_ignored():
    source = {
        "evaluation_core_bbox_local": {"x": 0, "y": 0, "w": 100, "h": 100},
        "context_bbox_global": {"x": 0, "y": 0, "w": 100, "h": 100},
    }
    annotation = {
        "targets": [],
        "secondary_components_and_exclusions": [
            {
                "id": "callout",
                "bbox_global": {"x": 10, "y": 10, "w": 20, "h": 20},
                "role": "valve identity callout",
                "scoring": "not independent instrument",
            },
            {
                "id": "break",
                "bbox_global": {"x": 40, "y": 10, "w": 10, "h": 30},
                "role": "curved line-break mark",
                "scoring": "not physical equipment, valve, instrument or off-page connector",
            },
        ],
    }
    predictions = [
        {"id": r["id"], "bbox": r["bbox_global"], "kind": "instrument"}
        for r in annotation["secondary_components_and_exclusions"]
    ]
    result = study.inventory_metrics(predictions, annotation, source, 0.5)
    assert result["false_positive_localization"] == 2
    assert result["ignored_secondary_or_ambiguous"] == []
    assert len(result["negative_example_hits"]) == 2


def test_broad_route_keeps_uninterpreted_inventory_separate(tmp_path):
    from diagex.vision import symbol_interpretation as module

    image = tmp_path / "source.png"
    Image.new("RGB", (200, 100), "white").save(image)
    requests = []

    class Client:
        def messages_create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                id="test-generation",
                usage=None,
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        name="submit_raster_symbols",
                        input={
                            "raster_results": [
                                {
                                    "proposal_id": "cv-1",
                                    "decision": "symbol",
                                    "broad_category": "valve",
                                    "confidence": "medium",
                                    "bbox": {"x": 0.2, "y": 0.2, "w": 0.1, "h": 0.2},
                                    "reason": "two opposing triangles",
                                }
                            ],
                            "discoveries": [],
                        },
                    )
                ],
            )

    prepared = {
        "input": {"image": str(image), "render_dpi": 300},
        "policy": asdict(SymbolPerceptionConfig(transport_attempts=1, request_timeout_s=120)),
        "context": {
            "raster_proposal_guidance": {
                "guides": [{"proposal_id": "cv-1"}],
                "guides_omitted_by_limit": 0,
            }
        },
        "backend": "broad_review",
    }
    result = study.invoke(prepared, module, Client(), [])
    assert result["detections"] == []
    assert len(result["proposals"]) == 1
    assert study.broad_class(result["proposals"][0]) == "valve"
    assert result["proposals"][0]["bbox"] == {"x": 40, "y": 20, "w": 20, "h": 20}
    assert len(requests) == 2
    assert requests[1]["tools"][0]["name"] == "submit_raster_semantics"
    assert (
        result["batch"]["candidate_reviews"][0]["legend_interpretation"]["decision"] == "unresolved"
    )


def test_text_only_reference_ablation_keeps_meaning_but_no_image_or_variant_trace(tmp_path):
    from diagex.knowledge.library import model_assets, supplied_trace, validate_match
    from diagex.vision import symbol_interpretation as module

    entry = {
        "id": "public-reference",
        "version": 2,
        "explanation": "Defined geometry phrase",
        "source": {"document_id": study.WEBSITE, "image": "never-load-this.png"},
        "catalog": {
            "role": "catalog",
            "model_asset_id": "symbol",
            "displayed_asset_ids": ["symbol"],
        },
        "assets": [
            {
                "id": "symbol",
                "role": "variant",
                "path": "never-load-this.png",
                "sha256": "a" * 64,
                "derived_from": [],
            }
        ],
        "text_slots": [],
    }
    original = {
        "identity": "original-identity",
        "references": [entry],
        "reference_ids": [entry["id"]],
    }
    original["supplied_assets"] = model_assets(original)
    original_copy = json.loads(json.dumps(original))
    context, audit = study.ablate_reference_images(original, "none")
    assert original == original_copy
    assert context["identity"] != original["identity"]
    assert context["references"][0]["explanation"] == entry["explanation"]
    assert context["supplied_assets"] == context["model_assets"] == model_assets(context) == []
    assert supplied_trace(context)["assets"] == []
    assert audit["withheld_model_assets"][0]["asset_id"] == "symbol"
    assert audit["withheld_reference_asset_ids"][0]["reference_id"] == entry["id"]
    assert (
        validate_match(context, entry["id"], "symbol", "Source triangles")[1]
        == "variant_not_supplied"
    )
    assert validate_match(context, entry["id"], None, "Source triangles")[0] is not None
    image = tmp_path / "source.png"
    Image.new("RGB", (200, 100), "white").save(image)
    image_hash = study.sha(image)
    requests = []

    class Client:
        def messages_create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                content=[
                    {
                        "type": "tool_use",
                        "name": "submit_pid_objects",
                        "input": {
                            "candidate_results": [],
                            "proposals": [
                                {
                                    "kind": "equipment",
                                    "equipment_class": "valve",
                                    "bbox": {"x": 0.2, "y": 0.2, "w": 0.1, "h": 0.2},
                                    "confidence": "medium",
                                    "knowledge_reference_id": entry["id"],
                                    "knowledge_variant_id": "symbol",
                                    "knowledge_evidence": "Source triangles",
                                }
                            ],
                        },
                    }
                ],
                usage=None,
            )

    prepared = {
        "input": {"image": str(image), "render_dpi": 300},
        "policy": asdict(SymbolPerceptionConfig(transport_attempts=1, request_timeout_s=120)),
        "context": {"knowledge": context},
        "backend": "baseline",
    }
    result = study.invoke(prepared, module, Client(), [])
    blocks = requests[0]["messages"][0]["content"]
    assert sum(block["type"] == "image" for block in blocks) == 1
    assert "Defined geometry phrase" in json.dumps(blocks)
    attrs = result["detections"][0]["attributes"]
    assert attrs["supplied_knowledge"]["assets"] == []
    assert attrs["knowledge_match_error"] == "variant_not_supplied"
    assert "knowledge_match" not in attrs
    assert study.sha(image) == image_hash
    all_images, audit_all = study.ablate_reference_images(original, "all")
    assert all_images == original and audit_all["withheld_model_assets"] == []


def test_text_only_ablation_also_disables_legacy_source_images():
    from diagex.knowledge.library import model_assets

    context = {
        "identity": "original",
        "references": [
            {
                "id": "legacy",
                "version": 1,
                "source": {"image": "legacy.png"},
                "image_sha256": "b" * 64,
                "explanation": "Legacy meaning",
            }
        ],
    }
    updated, audit = study.ablate_reference_images(context, "none")
    assert model_assets(updated) == []
    assert updated["references"][0]["source"]["image"] is None
    assert audit["withheld_model_assets"][0]["asset_id"] == "legacy-image"


def test_provider_control_narrows_verified_routes_without_changing_prices(tmp_path):
    guard = ledger(tmp_path)
    price = guard.prices["model"]
    price["provider_tags"] = ["first", "second"]
    price["endpoint_snapshots"] = [
        {"tag": "first", "supports_tool_choice": {"auto": True, "function": False}},
        {"tag": "second", "supports_tool_choice": {"auto": True, "function": True}},
    ]
    audit = study.select_provider(guard.prices, "model", "first")
    assert price["provider_tags"] == ["first", "second"]
    assert guard.prices["model"]["provider_tags"] == ["first"]
    assert audit["verified_provider_tags"] == ["first", "second"]
    assert audit["effective_provider_tags"] == ["first"]
    assert len(audit["effective_endpoint_tool_support"]) == 1
    for key, value in price.items():
        if key != "provider_tags":
            assert guard.prices["model"][key] == value
    identity = guard.reserve("model", 10)
    row = study.load(guard.path)["requests"][0]
    assert row["id"] == identity and row["reserved_usd"] == 1.1
    assert row["price"]["provider_tags"] == ["first"]
    # The pricing file remains unchanged: this is only a run-specific restriction.
    assert "provider_tags" not in study.load(tmp_path / "prices.json")["models"]["model"]
    with pytest.raises(ValueError, match="outside the verified"):
        study.select_provider({"model": price}, "model", "unknown")
    with pytest.raises(BudgetExceeded, match="expired"):
        study.select_provider({"model": {**price, "valid_until": 0}}, "model", "first")


def test_auto_tool_choice_changes_transport_only_and_rejects_unstructured_response(tmp_path):
    from diagex.vision import symbol_interpretation as module

    request = {
        "tools": [{"name": "submit_pid_objects"}],
        "tool_choice": {"type": "tool", "name": "submit_pid_objects"},
    }
    assert study.controlled_tool_choice(request, "forced") == request
    assert study.controlled_tool_choice(request, "auto")["tool_choice"] == {"type": "auto"}
    assert request["tool_choice"]["type"] == "tool"
    image = tmp_path / "source.png"
    Image.new("RGB", (200, 100), "white").save(image)
    requests = []

    class Client:
        def messages_create(self, **kwargs):
            requests.append(study.controlled_tool_choice(kwargs, "auto"))
            return SimpleNamespace(
                content=[{"type": "text", "text": "There is a valve."}], usage=None
            )

    prepared = {
        "input": {"image": str(image), "render_dpi": 300},
        "policy": asdict(SymbolPerceptionConfig(transport_attempts=1, request_timeout_s=120)),
        "context": {},
        "backend": "baseline",
    }
    with pytest.raises(module.PerceptionResponseFormatError, match="unstructured"):
        study.invoke(prepared, module, Client(), [])
    assert requests and all(r["tool_choice"] == {"type": "auto"} for r in requests)


def test_grid_views_depend_only_on_source_dimensions_and_partition_ownership():
    specs = study.source_view_specs((201, 101), 2)
    assert len(specs) == 4
    assert study.source_view_specs((201, 101), 1) == [
        {
            "id": "public-crop",
            "bbox": {"x": 0, "y": 0, "w": 201, "h": 101},
            "ownership_bbox": {"x": 0, "y": 0, "w": 201, "h": 101},
        }
    ]
    for x in (0, 99, 100, 200):
        for y in (0, 49, 50, 100):
            assert (
                sum(
                    study._in_core({"x": x, "y": y, "w": 0, "h": 0}, spec["ownership_bbox"])
                    for spec in specs
                )
                == 1
            )
    assert specs[0]["bbox"]["w"] == 120
    assert specs[1]["bbox"]["x"] == 80


def test_grid_uses_global_projection_single_ownership_and_remapped_cv(tmp_path, monkeypatch):
    from diagex.vision import symbol_interpretation as module

    image = tmp_path / "source.png"
    Image.new("RGB", (200, 100), "white").save(image)
    original_hash = study.sha(image)
    guidance_path = tmp_path / "cv.json"
    study.save(
        guidance_path,
        {
            "image_sha256": original_hash,
            "size": [200, 100],
            "coordinate_frame": "original_image_pixels",
            "status": "proposals_require_interpretation",
            "detector": {"detector_sha256": "f" * 64},
            "predictions": [
                {
                    "id": "cv-1",
                    "bbox": [96, 46, 106, 56],
                    "label": "valve",
                    "confidence": 0.8,
                    "disposition": "review_proposal",
                }
            ],
        },
    )
    calls, kwargs_by_view = [], []
    production = module.perceive_tile

    def captured(**kwargs):
        kwargs_by_view.append(kwargs)
        return production(**kwargs)

    monkeypatch.setattr(module, "perceive_tile", captured)

    class Client:
        def messages_create(self, **kwargs):
            calls.append(kwargs)
            view = kwargs_by_view[-1]["view_info"]
            bbox = {
                "x": (96 - view.origin[0]) / view.view_size[0],
                "y": (46 - view.origin[1]) / view.view_size[1],
                "w": 10 / view.view_size[0],
                "h": 10 / view.view_size[1],
            }
            return SimpleNamespace(
                content=[
                    {
                        "type": "tool_use",
                        "name": "submit_pid_objects",
                        "input": {
                            "candidate_results": [],
                            "proposals": [
                                {
                                    "kind": "equipment",
                                    "equipment_class": "valve",
                                    "bbox": bbox,
                                    "confidence": "medium",
                                }
                            ],
                        },
                    }
                ],
                usage=None,
            )

    prepared = {
        "input": {"image": str(image), "render_dpi": 300, "annotations": "FORBIDDEN_GOLD"},
        "policy": asdict(SymbolPerceptionConfig(transport_attempts=1, request_timeout_s=120)),
        "context": {},
        "backend": "baseline",
        "grid": 2,
        "view_specs": study.source_view_specs((200, 100), 2),
        "cv_guidance_path": str(guidance_path),
    }
    result = study.invoke(prepared, module, Client(), [])
    assert len(calls) == len(result["per_view"]) == 4 and result["view_failures"] == []
    assert len(result["detections"]) == len(result["proposals"]) == 1
    assert result["detections"][0]["bbox"] == {"x": 96, "y": 46, "w": 10, "h": 10}
    assert result["proposals"][0]["id"] == "public-grid-1-1-proposal-0"
    for kwargs in kwargs_by_view:
        view = kwargs["view_info"]
        guide = kwargs["page_context"]["raster_proposal_guidance"]["guides"][0]
        assert guide["bbox_normalized"]["x"] == pytest.approx(
            (96 - view.origin[0]) / view.view_size[0]
        )
        assert kwargs["view_image"].size == (120, 60)
        assert kwargs["page"].width == 200 and kwargs["page"].height == 100
    assert study.sha(image) == original_hash
    assert "FORBIDDEN_GOLD" not in json.dumps(
        [{k: v for k, v in call.items() if not callable(v)} for call in calls]
    )


def test_grid_retains_completed_views_if_budget_stops_remaining_work(monkeypatch):
    count = 0

    def fake_view(*args):
        nonlocal count
        count += 1
        if count == 2:
            raise BudgetExceeded("stop")
        return {
            "detections": [{"id": "first"}],
            "proposals": [],
            "batch": {},
            "attempts": 1,
            "contract_failed": False,
            "recovery_diagnostics": [],
        }

    monkeypatch.setattr(study, "invoke_view", fake_view)
    result = study.invoke(
        {"grid": 2, "view_specs": study.source_view_specs((200, 100), 2)}, None, None, []
    )
    assert count == 2 and result["detections"] == [{"id": "first"}]
    assert result["contract_failed"] and len(result["view_failures"]) == 1
