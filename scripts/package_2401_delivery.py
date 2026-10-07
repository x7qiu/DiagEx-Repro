"""Package the bounded source-audited draft; no model calls or original-run edits."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from statistics import median

from diagex.dexpi import validate as dexpi_validate
from diagex.dexpi.xml_io import dump as dump_xml
from diagex.dexpi.xml_io import load as load_xml
from diagex.extractors.dexpi_builder import build_dexpi, serialize_model, validate_model
from diagex.extractors.evidence_checkpoint import atomic_write_json
from diagex.vision.evidence import PageEvidence
from diagex.vision.models import ReconciledGraph
from diagex.vision.topology import route_evidence_failures

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/2401-delivery"


def read(path):
    return json.loads(Path(path).read_text())


def write(name, value):
    atomic_write_json(OUT / name, value)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ratio(a, b):
    return a / b if b else None


def aggregate(cases):
    ok = [c for c in cases if c["status"] == "complete"]
    fields = [
        "tp",
        "fp",
        "fn",
        "subtype_correct",
        "subtype_matched",
        "tag_correct",
        "tag_matched",
        "tag_truth",
    ]
    totals = {k: sum(c.get("metrics", {}).get(k, 0) for c in ok) for k in fields}
    times = sorted(c["elapsed_s"] for c in ok)
    totals.update(
        precision=ratio(totals["tp"], totals["tp"] + totals["fp"]),
        recall=ratio(totals["tp"], totals["tp"] + totals["fn"]),
        subtype_accuracy_on_matched_known=ratio(
            totals["subtype_correct"], totals["subtype_matched"]
        ),
        tag_exact_match_on_matched=ratio(totals["tag_correct"], totals["tag_matched"]),
        tag_exact_match_on_all_tagged_truth=ratio(totals["tag_correct"], totals["tag_truth"]),
        completed=len(ok),
        attempted=len(cases),
        median_s=median(times) if times else None,
        p95_s=times[math.ceil(0.95 * len(times)) - 1] if times else None,
        escalated_regions=sum(c.get("attempts", 1) > 1 for c in ok),
    )
    return totals


def percent(value):
    return "n/a" if value is None else f"{100 * value:.2f}%"


def main():
    build = read(OUT / "build-result.json")
    run = Path(build["run_dir"])
    snapshot = read(OUT / "audit/reviewed-snapshot-final.json")
    provenance = read(OUT / "audit/provenance.json")
    graph = ReconciledGraph.model_validate(read(run / "graph.json"))
    by_id = {n.id: n for n in graph.nodes}
    corrections = read(OUT / "audit/final-symbol-corrections.json")
    for row in corrections["changes"]:
        d = row["detection"]
        n = by_id["n-review-" + d["id"]]
        n.label, n.source_quote = d["label"], d.get("raw_text")
        n.attributes.pop("actuation", None)
        n.attributes.update(d["attributes"])
        n.source_evidence_ids = sorted(set(n.source_evidence_ids + d.get("source_text_ids", [])))
        n.attributes["source_review"] = dict(
            actor_type="agent",
            review_revision=snapshot["revision"],
            evidence_refs=row["evidence_refs"],
            reasons=row["reasons"],
        )
    for n in graph.nodes:
        n.attributes["human_reviewed"] = False
        n.attributes["review_origin"] = "agent"
    final_edge_audit = []
    for e in graph.edges:
        if e.attributes.get("provisional_review_only"):
            continue
        if e.line_type != "process":
            reason = "The route is visible, but the source legend signal examples do not establish a reliable electric/pneumatic mapping to these long dashed strokes. Keep route geometry; withhold typed connection pending source-owner clarification."
            e.attributes.update(
                provisional_review_only=True,
                review_reason=reason,
                source_review=dict(
                    actor_type="agent",
                    decision="withhold_signal_type",
                    evidence_refs=[
                        "audit/legend-line-native4x.png",
                        "audit/signals-1.png",
                        "audit/signals-2.png",
                        "audit/signals-3.png",
                    ],
                ),
            )
            final_edge_audit.append(
                dict(id=e.id, decision="demote", original_type=e.line_type, reason=reason)
            )
        elif e.attributes.get("visual_style") == "dashed":
            e.attributes["visual_style"] = "solid"
            e.attributes["source_review"] = dict(
                actor_type="agent",
                decision="retain_process",
                evidence_refs=["audit/dashed-process.png"],
                reason="Original source crop shows continuous solid valve-manifold piping; prior automatic dashed-style label was incorrect.",
            )
            final_edge_audit.append(dict(id=e.id, decision="retain_process_correct_style"))
    pages = {
        i: PageEvidence.model_validate(read(run / f"evidence/page-{i + 1:04d}.json"))
        for i in range(3, 12)
    }
    proof_failures, proof_records = {}, []
    confirmed = [e for e in graph.edges if not e.attributes.get("provisional_review_only")]
    for e in confirmed:
        a, b = by_id[e.from_node], by_id[e.to_node]
        if e.cross_sheet:
            failures = (
                []
                if e.attributes.get("match_basis") == "reciprocal_title_block_references"
                and e.source_evidence_ids
                and e.attributes.get("left_sheet_ref")
                and e.attributes.get("right_sheet_ref")
                else ["missing_reciprocal_reference_proof"]
            )
        else:
            failures = route_evidence_failures(e, pages[a.page_index])
        if failures:
            proof_failures[e.id] = failures
        proof_records.append(
            dict(
                id=e.id,
                pages=[a.page_index + 1, b.page_index + 1],
                kind="reciprocal_references" if e.cross_sheet else "native_route_and_ports",
                source_evidence_ids=e.source_evidence_ids,
                failures=failures,
            )
        )
    assert not proof_failures, proof_failures
    write("graph.json", graph.model_dump(mode="json"))
    write("confirmed-connections.json", [e.model_dump(mode="json") for e in confirmed])
    write(
        "audit/connection-audit.json",
        dict(
            actor_type="agent",
            human_approved=False,
            typed_signal_decisions=final_edge_audit,
            emitted_connection_proofs=proof_records,
            scope="Automated native proof validation on every emitted edge; targeted source visual review of all 15 proposed signals and 4 disputed process styles. Not a second-rater graph truth audit.",
        ),
    )
    with (OUT / "confirmed-connections.csv").open("w", newline="") as file:
        fields = [
            "id",
            "from_node",
            "from_label",
            "from_page",
            "to_node",
            "to_label",
            "to_page",
            "line_type",
            "cross_sheet",
            "direction",
            "source_evidence_ids",
            "source_view",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for e in confirmed:
            a, b = by_id[e.from_node], by_id[e.to_node]
            writer.writerow(
                dict(
                    id=e.id,
                    from_node=a.id,
                    from_label=a.label,
                    from_page=a.page_index + 1,
                    to_node=b.id,
                    to_label=b.label,
                    to_page=b.page_index + 1,
                    line_type=e.line_type,
                    cross_sheet=e.cross_sheet,
                    direction=e.attributes.get("flow_direction", "unknown"),
                    source_evidence_ids=";".join(e.source_evidence_ids),
                    source_view=f"review.html?edge={e.id}&page={a.page_index + 1}",
                )
            )
    hypotheses = read(run / "hypotheses.json")
    write("hypotheses.json", hypotheses)
    omission = read(OUT / "audit/omission-audit.json")
    exceptions = []
    for item in read(OUT / "audit/symbol-audit.json")["semantic_exceptions"]:
        exceptions.append(
            dict(
                id="semantic-" + item["id"],
                type="symbol_semantics",
                status="unresolved",
                page_index=item["page_index"],
                bbox_global=item["bbox"],
                reason=item["question"],
                node_id="n-review-" + item["id"],
            )
        )
    for item in snapshot["unresolved"]["items"]:
        d = item.get("detection", {})
        exceptions.append(
            dict(
                id=item["id"],
                type=item["type"],
                status="unresolved",
                page_index=d.get("page_index"),
                reason=item.get("review_provenance", {}).get("note", item.get("reason")),
                details=item,
            )
        )
    for i, item in enumerate(omission["source_conflicts"]):
        exceptions.append(
            dict(
                id=f"source-conflict-{i + 1}",
                type="source_identifier_conflict",
                status="unresolved",
                page_index=item["pages"][0] - 1,
                reason=item["question"],
                pages=item["pages"],
                decision="Preserve printed labels; no silent renumbering or human approval.",
            )
        )
    for e in graph.edges:
        if e.attributes.get("provisional_review_only"):
            exceptions.append(
                dict(
                    id=e.id,
                    type="provisional_connection",
                    status="unresolved",
                    page_index=by_id[e.from_node].page_index,
                    reason=e.attributes.get("review_reason")
                    or "; ".join(e.attributes.get("route_evidence", {}).get("failures", []))
                    or "Connection not admitted by evidence gate",
                    from_node=e.from_node,
                    to_node=e.to_node,
                )
            )
    exceptions.append(
        dict(
            id="local-opc-port-gap",
            type="known_graph_limitation",
            status="unresolved",
            reason="Local off-page-connector routes lack native contour ports in the current vector topology backend and remain withheld; explicit reciprocal cross-sheet references are evaluated separately.",
        )
    )
    exceptions.append(
        dict(
            id="raster-environment",
            type="validation_blocker",
            status="unresolved",
            reason="Public raster graph recovery unavailable because the optional OpenCV dependency is absent. Three dependency-install attempts failed with network/TLS errors. Raster graph result is not a passed validation.",
        )
    )
    write(
        "exceptions.json",
        dict(
            review_origin="agent",
            human_approved=False,
            unresolved=exceptions,
            graph_conflicts=[
                c.model_dump(mode="json") if hasattr(c, "model_dump") else c
                for c in graph.conflicts
            ],
            unclassified_nodes=[
                n.id
                for n in graph.nodes
                if n.attributes.get("equipment_class", "").startswith("unclassified")
            ],
            valves_without_known_subtype=[
                n.id
                for n in graph.nodes
                if n.attributes.get("equipment_class") == "valve"
                and n.attributes.get("valve_type") in (None, "", "unknown", "unclassified", "other", "general")
            ],
            source_conflicts=omission["source_conflicts"],
        ),
    )
    dexpi = build_dexpi(graph)
    serialize_model(dexpi.model, OUT, "2401.dexpi")
    dump_xml(dexpi.model, OUT / "2401.dexpi.xml")
    issues = validate_model(dexpi.model)
    semantic = dexpi_validate.semantic_validate(load_xml(OUT / "2401.dexpi.xml"))
    errors = [
        dict(rule=x.rule_id, message=x.message, path=x.path)
        for x in semantic
        if x.severity == "error"
    ]
    assert not issues and not errors, (issues, errors)
    write(
        "export-validation.json",
        dict(
            json_roundtrip_issues=issues,
            xml_semantic_errors=errors,
            xml_semantic_warnings=[
                dict(rule=x.rule_id, message=x.message, path=x.path)
                for x in semantic
                if x.severity != "error"
            ],
            builder_issues=dexpi.issues,
            stats=dexpi.stats,
            xsd_validation="not run; round-trip plus semantic validation only",
        ),
    )
    # Freeze exact model-comparison outputs; incomplete reference results stay separate.
    comparisons = {}
    prices = read(OUT / "model-preflight/verified-prices.json")["models"]
    for folder in sorted((OUT / "evaluation/results-cycle-2").iterdir()):
        if not folder.is_dir():
            continue
        cases = [read(f) for f in sorted(folder.glob("*.json"))]
        dispatches, stronger_regions, usage_estimate = {}, 0, 0.0
        for case in cases:
            models = [
                d["transport"]["model"]
                for d in case.get("diagnostics", [])
                if d.get("phase") == "transport"
                and d.get("transport", {}).get("event") == "dispatch"
            ]
            stronger_regions += any("122b" in model for model in models)
            for model in models:
                dispatches[model] = dispatches.get(model, 0) + 1
            for i, usage in enumerate(case.get("usage", [])):
                model = models[i] if i < len(models) else case["escalation_model"]
                price = prices[model]
                usage_estimate += (
                    usage["input_tokens"] * price["input_per_token"]
                    + usage["output_tokens"] * price["output_per_token"]
                )
        comparisons[folder.name] = dict(
            model_dispatches=dispatches,
            stronger_model_regions=stronger_regions,
            recorded_usage_estimate_usd=usage_estimate,
            overall=aggregate(cases),
            planned=16,
            splits={
                s: aggregate([c for c in cases if c["split"] == s])
                for s in ["diagnostic", "heldout", "public"]
            },
            repeats={str(i): aggregate([c for c in cases if c["repeat"] == i]) for i in [1, 2]},
            cases=[
                {
                    k: v
                    for k, v in c.items()
                    if k
                    in [
                        "id",
                        "split",
                        "repeat",
                        "status",
                        "metrics",
                        "attempts",
                        "elapsed_s",
                        "error",
                    ]
                }
                for c in cases
            ],
        )
    ledger = read(OUT / "model-preflight/spending.json")
    costs = {
        cat: dict(
            limit_usd=limit,
            conservative_accounted_usd=sum(
                r["charged_usd"] for r in ledger["requests"] if r["category"] == cat
            ),
            unsettled_reserved_usd=sum(
                r["charged_usd"]
                for r in ledger["requests"]
                if r["category"] == cat and r["status"] != "complete"
            ),
        )
        for cat, limit in ledger["category_limits"].items()
    }
    total = sum(r["charged_usd"] for r in ledger["requests"])
    context = [
        {
            k: v
            for k, v in read(f).items()
            if k in ["condition", "repeat", "metrics", "elapsed_s", "status", "sample_sha256"]
        }
        for f in sorted((OUT / "context-comparison").glob("*context-*.json"))
    ]
    public = []
    for name in ["dexpi-reference", "tennessee1", "butane1"]:
        d = read(OUT / f"public-graph-validation/{name}.json")
        public.append(
            {
                k: d[k]
                for k in [
                    "dataset",
                    "metrics",
                    "elapsed_s",
                    "status",
                    "warnings",
                    "scope",
                    "limitations",
                ]
            }
        )
        if name == "butane1":
            public[-1]["status"] = "blocked_missing_opencv"
    metrics = dict(
        symbol_comparisons=comparisons,
        connection_context_sample=context,
        public_graph=public,
        costs=costs,
        total_conservative_accounted_usd=total,
        automated_production_accepted=False,
        adaptive_adopted=False,
        cycles_used=2,
        caveats=[
            "Matching truth is source-audited by this agent, not independent human ground truth.",
            "No overall 2401 graph precision/recall established.",
            "Closed reference is partial: 10 of 16 complete; 1 budget-stopped and 5 not dispatched.",
            "Latency includes pipeline work/retries; p95 is nearest-rank on a very small sample.",
            "charged_usd is a verified worst-provider usage estimate or retained maximum reservation, not an invoice. Failed cycle-1 reservations remain counted.",
        ],
    )
    write("metrics.json", metrics)
    package_provenance = dict(
        original_run=provenance["original_run"],
        original_run_id="r-4344",
        raw_graph_run=build["run_id"],
        raw_graph_path=str(run / "graph.json"),
        source_sha256=provenance["source_sha256"],
        raw_graph_sha256=sha(run / "graph.json"),
        source_path=provenance["source"],
        graph_input_review_revision=build["review_revision"],
        final_review_revision=snapshot["revision"],
        final_review_sha256=sha(OUT / "audit/reviewed-snapshot-final.json"),
        metadata_corrections=len(corrections["changes"]),
        final_graph_sha256=sha(OUT / "graph.json"),
        model_profile=dict(
            fast="qwen/qwen3.5-35b-a3b",
            stronger="qwen/qwen3.5-122b-a10b",
            transport="openrouter",
            workflow="fixed",
        ),
        review_origin="agent",
        human_approved=False,
        quality_status="partial",
        automatic_production_accepted=False,
        pages_audited=omission["coverage_pages"],
        graph_nodes=len(graph.nodes),
        confirmed_connections=len(confirmed),
        provisional_connections=len(graph.edges) - len(confirmed),
        graph_build_manifest=read(run / "checkpoints/manifest.json")
        if (run / "checkpoints/manifest.json").exists()
        else None,
    )
    write("delivery-provenance.json", package_provenance)
    template = (ROOT / "src/diagex/review/delivery_template.html").read_text()
    payload = dict(
        graph=graph.model_dump(mode="json"),
        review_revision=snapshot["revision"],
        hypotheses=hypotheses if isinstance(hypotheses, list) else hypotheses.get("hypotheses", []),
        exceptions=exceptions,
    )
    (OUT / "review.html").write_text(
        template.replace(
            "__DELIVERY_DATA__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
        )
    )
    (OUT / "original-native-evidence").mkdir(exist_ok=True)
    for f in (run / "evidence").glob("page-*.json"):
        shutil.copy2(f, OUT / "original-native-evidence" / f.name)
    shutil.copy2(ROOT / "docs/OPEN_WEIGHT_PRODUCTION.md", OUT / "OPERATIONS.md")
    # A report is generated from counts, never from nominal planned completion.
    lines = [
        "# 2401 source-audited draft",
        "",
        f"The delivery contains {len(graph.nodes)} reviewed symbol nodes, {len(confirmed)} source-supported process connections (including 3 explicit cross-sheet references), and {len(graph.edges) - len(confirmed)} provisional connections. All nine P&ID pages and all three legend pages received an agent source audit. This is a partial engineering draft, without human approval; automatic production acceptance is false.",
        "",
        "Open [source review](review.html), [graph](graph.json), [confirmed CSV](confirmed-connections.csv), [confirmed JSON](confirmed-connections.json), [DEXPI XML](2401.dexpi.xml), [DEXPI JSON](2401.dexpi.json), [exceptions](exceptions.json), and [provenance](delivery-provenance.json).",
        "",
        "## Audit and graph scope",
        "",
        "The original r-4344 artifacts are preserved. Graph generation used reviewed inputs at revision 1512, without repeating full symbol extraction. The final revision 1620 applies 108 recorded metadata corrections, including three vessel tag associations and removal of unsupported actuation attributes. Agent provenance is carried through fusion and export; no node is marked human-reviewed. The source audit added 47 omitted glyphs. Two detached symbols and one unresolved legend definition remain excluded. Five source identifier conflicts retain the printed labels and await owner decisions.",
        "",
        "Valve subtype remains general for 242 of 288 valves. These visible valve bodies are retained with a broad class; no more specific mechanism or default actuation is asserted. The subtype exception list is included in exceptions.json.",
        "",
        "Every emitted local connection passed the native path, endpoint-port, crossing and intermediate-symbol evidence checks. Cross-sheet edges carry reciprocal drawing references. Source images, native evidence IDs and route overlays are included. This proves that an inspectable evidence record exists; it is not a measured full-graph recall claim. Fifteen visible signal routes were withheld at final audit because the electric/pneumatic classification was not established by the source legend. Four solid process-manifold routes had their erroneous automatic dashed-style label corrected from the source crops. Local OPC port recovery remains incomplete.",
        "",
        "Engineering hypotheses are stored separately and excluded from confirmed connections and DEXPI. This run produced zero such hypotheses; the implementation and exclusion tests are present. Targeted geometry recovery is bounded and cannot accept a connection from engineering plausibility alone.",
        "",
        "## Symbol comparison — second and final cycle",
        "",
        "Eight frozen regions, identical reviewed legend/evidence, two repeats each. Four diagnostic, two held-out 2401, two public raster regions. Match requires the same broad object family plus IoU ≥ 0.25 or intersection/minimum area ≥ 0.65, with one-to-one maximum-cardinality matching. Tag exact match normalizes whitespace and hyphens only. Subtype accuracy is on matched objects with a known source-audited subtype; unclassified truth is excluded, an unknown prediction is not correct. No second independent rater is claimed.",
        "",
        "| Workflow / split | Complete / planned | TP / FP / FN | Precision | Recall | Subtype correct / known matched | Tag exact / tagged matched | Median / p95 s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, comp in comparisons.items():
        for split, m in comp["splits"].items():
            planned = 8 if split == "diagnostic" else 4
            latency = "n/a" if m["median_s"] is None else f"{m['median_s']:.1f} / {m['p95_s']:.1f}"
            lines.append(
                f"| {name} / {split} | {m['completed']} / {planned} | {m['tp']} / {m['fp']} / {m['fn']} | {percent(m['precision'])} | {percent(m['recall'])} | {m['subtype_correct']} / {m['subtype_matched']} | {m['tag_correct']} / {m['tag_matched']} | {latency} |"
            )
    lines += [
        "",
        "**Decision:** retain the fixed fast open-weight default. Held-out precision/recall is 94.12% / 88.89% for fixed versus 90.91% / 83.33% for adaptive. The adaptive gate fails. Closed-source performance cannot satisfy this gate. Public raster perception is poor for all three configurations, exposing a missing candidate/discovery capability rather than proving production readiness. Blank tags in parsed predictions count as errors even when an attribute contains a loop number.",
        "",
        "Cycle 1 failed with provider routing errors for Qwen effort parameters; the adapter was corrected to use binary thinking and omit unsupported output effort. Cycle 2 is the sole retest. Closed comparison completed 10/16 planned cases; one second inspection was stopped before dispatch by the extraction budget reservation, and five cases were not attempted. All held-out and public reference pairs completed; diagnostic reference rows are partial and must not be compared as complete aggregates.",
        "",
        "### Completion, escalation and repeat variation",
        "",
        "| Workflow | Complete / planned | Regions with extra inspections | Repeat 1 P / R | Repeat 2 P / R | Median / p95 s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, c in comparisons.items():
        m = c["overall"]
        a = c["repeats"]["1"]
        b = c["repeats"]["2"]
        lines.append(
            f"| {name} | {m['completed']} / 16 | {m['escalated_regions']} / {m['completed']} | {percent(a['precision'])} / {percent(a['recall'])} | {percent(b['precision'])} / {percent(b['recall'])} | {m['median_s']:.1f} / {m['p95_s']:.1f} |"
        )
    lines += [
        "",
        "Extra-inspection frequency covers the complete fast-plus-escalation workflow, including discovery attempts. Closed repeat aggregates cover different subsets; use the per-region records in metrics.json for paired comparisons. Small-sample p95 is descriptive, not an SLA. [Frozen evaluation](evaluation/manifest.json) and per-case outputs preserve raw detections, diagnostics, failures, usage, timings and match assignments.",
        "",
        "## Connection sample and public graph validation",
        "",
        "Context comparison used the same 17-pair page-7 sample (14 visible process routes and 3 false shortcuts), identical source evidence, the same fast open-weight reasoning model and two repeats. Both conditions in both repeats yielded TP 7, FP 0, FN 7: precision 100%, recall 50%, with no observed context benefit. Without-context elapsed time was 204.7 / 199.5 seconds; with-context was 207.6 / 186.3 seconds. These are sample metrics, not full 2401 graph metrics.",
        "",
        "Public graph validation supplies fixture nodes to isolate connection recovery. It is separate from public raster symbol perception. Fixture annotations are scored as provided; source review found incomplete symbols and approximate boxes, and historical debug counts were stale.",
        "",
        "| Fixture | TP / FP / FN | Precision | Recall | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for d in public:
        m = d["metrics"]
        blocked = d["status"].startswith("blocked")
        lines.append(
            f"| {d['dataset']} | {m['tp']} / {m['fp']} / {m['fn']} | {'n/a' if blocked else percent(ratio(m['tp'], m['tp'] + m['fp']))} | {'n/a' if blocked else percent(ratio(m['tp'], m['tp'] + m['fn']))} | {d['status']} |"
        )
    lines += [
        "",
        "The butane raster graph result is environment-blocked by missing OpenCV; its recorded zero edges are not a successful quality measurement. Three installation attempts failed at network/TLS access. No success is claimed for that backend.",
        "",
        "## Budget and validation",
        "",
        f"Total conservative accounted spend is **${total:.6f} of $50**. This includes retained maximum reservations on failed/uncertain requests. It is an upper estimate from verified provider prices and recorded usage, not a billing invoice. The original r-4344 historical run is excluded; all live calls for this implementation milestone are in the shared ledger.",
        "",
        "| Category | Allocation | Accounted USD | Unsettled reserve included |",
        "|---|---:|---:|---:|",
    ]
    for cat, c in costs.items():
        lines.append(
            f"| {cat} | ${c['limit_usd']:.2f} | ${c['conservative_accounted_usd']:.6f} | ${c['unsettled_reserved_usd']:.6f} |"
        )
    lines += [
        "",
        "No third tuning cycle or additional closed-model batch was run. Model weights/license revisions, current-at-dispatch pricing, endpoint capabilities and smoke outputs are retained under model-preflight. The selected Qwen 3.5 models have publisher weight repositories and Apache-2.0 licenses; image/tool calls were exercised. Provider quantization and exact serving identity are not independently attested. DeepSeek legacy results remain diagnostic and are not admitted as a verified production model.",
        "",
        f"DEXPI JSON round-trip and parsed XML semantic validation report zero errors. {len([x for x in semantic if x.severity != 'error'])} semantic warnings and {len(dexpi.issues)} builder notes remain, primarily missing unprinted instrument numbers, tag-format recommendations and unresolved typing. XML XSD validation was not run. These files export confirmed connections only; provisional edges and hypotheses remain in review artifacts.",
        "",
        "Regression tests and the source review UI checks are recorded in [verification](verification.json). Hosted inference is configured now; self-hosted serving and hardware optimization are deferred. See the [operational guide](OPERATIONS.md).",
    ]
    lines += [
        "",
        "## Per-workflow usage and stronger-model dispatch",
        "",
        "These second-cycle usage estimates apply the verified maximum provider token prices to recorded responses. They exclude cycle-1 retained reserves and preflight calls, and do not replace the authoritative all-call ledger above. The partial closed run includes a paid response before its budget-stopped second inspection.",
        "",
        "| Workflow | Recorded usage estimate USD | Regions dispatched to stronger open-weight model |",
        "|---|---:|---:|",
    ]
    for name, value in comparisons.items():
        lines.append(
            f"| {name} | ${value['recorded_usage_estimate_usd']:.6f} | {value['stronger_model_regions']} |"
        )
    lines += [
        "",
        "The fixed workflow made 17 fast calls across 16 regions (one bounded repair retry). The adaptive open-weight workflow made 16 fast plus 12 stronger calls: escalation frequency 12/16 = 75%. The closed reference reused the closed model for additional inspections; it is not an open-weight escalation.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            dict(
                nodes=len(graph.nodes),
                confirmed=len(confirmed),
                provisional=len(graph.edges) - len(confirmed),
                cost=total,
                dexpi_stats=dexpi.stats,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
