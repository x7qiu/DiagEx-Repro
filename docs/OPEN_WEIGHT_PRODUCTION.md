# Open-weight 2401 delivery workflow

The web workbench now defaults to `deepseek/deepseek-v4.1-flash` through OpenRouter for vision, reasoning, and reinspection. The Qwen profile described below is an explicit selection in the web form; the CLI `--production-open-weight` behavior is unchanged. Custom evaluation configurations use the selected reasoning model for reinspection and do not inherit the Qwen allowlist.

The supported hosted profile uses Qwen 3.5 35B-A3B for ordinary vision and reasoning, with Qwen 3.5 122B-A10B available for bounded targeted inspection. The fixed extraction workflow remains the default because the adaptive configuration worsened both held-out precision and recall. Automatic extraction has not passed acceptance.

The 2401 delivery is in `output/2401-delivery/`. Open `review.html` in a browser or serve this directory on loopback. The view supports page/category filters, tag search, pan/zoom, endpoint navigation and direct connection links. The CSV, graph and DEXPI files share stable node/edge identities. The final agent audit and original machine graph are separate artifacts.

## Starting a production-profile run

Use the existing provider credential environment; never put credentials in a configuration report. From the repository root:

```sh
python scripts/preflight_open_weight.py --out output/model-preflight
export DIAGEX_SPENDING_LEDGER="$PWD/output/model-preflight/spending.json"
export DIAGEX_VERIFIED_PRICES="$PWD/output/model-preflight/verified-prices.json"
export DIAGEX_SPENDING_CATEGORY=graph
diagex extract-pid drawing.pdf --production-open-weight --fresh
```

Check `diagex extract-pid --help` for positional/output options. The web upload form also offers the production profile; it fixes the model roles and Evidence v2 engine. Custom model fields are for explicit evaluation configurations. `--adaptive-inspection` is an experimental opt-in; it is not the accepted default.

`--process-overview overview.txt` and `--engineering-rules rules.txt` add source-attributed context. Context can motivate a proposed route; it cannot supply drawing evidence. Unproven suggestions have endpoints, rationale and a question, stay in `hypotheses.json`, and never become standard DEXPI connections. A proposed missing route can trigger native geometry lookup and at most one fast plus one stronger visual inspection. Candidate/discovery inspection similarly has at most two additional inspections per flagged region. Remaining uncertainty stays queued.

The preflight reads publisher weight files and Apache-2.0 licenses, records exact revisions, and checks provider image/tool support. It makes no inference calls. Refresh prices when their 24-hour validity expires. Refreshing does not reset an existing spending ledger. Retained reservations must not be discarded after an uncertain response. The ledger reserves the worst supported request cost before every dispatch, includes retries, and stops when either category or total funds are insufficient. Its amounts are conservative accounting estimates, not provider invoices.

The milestone's existing ledger is `output/2401-delivery/model-preflight/spending.json`; do not start a new ledger to circumvent its $50 cap or the two-cycle stopping rule. A closed-reference comparison can be skipped when its maximum reservation would consume protected delivery funds.

## Review and recovery behavior

Fresh bypasses machine legend interpretation and machine symbol checkpoints. Independently persisted reviewed artifacts are preserved. Resume requires matching source/configuration/stage hashes; interrupted work retains bounded retry diagnostics and a visible paused/incomplete state. Changing legend definitions invalidates dependent classifications. Row boundary/caption edits and splitting mixed rows are supported.

Detection review offers the existing fully reviewed build and an explicit draft build. A draft includes only resolved detections/legend entries, lists unchecked coverage and unresolved items, and preserves agent versus human provenance. Agent confirmation does not become human approval. The r-4344 source artifacts were copied into a derived review workspace before any audit edits.

The adapter owns OpenRouter-specific behavior. Qwen 3.5 uses binary thinking and omits unsupported effort parameters. Production model validation rejects a closed-source identifier in every role; there is no automatic closed fallback. The application interfaces remain portable, but self-hosted serving, hardware sizing, provider quantization verification and exact deployed weight attestation are deferred.

## Reproducible evidence

- `scripts/evaluate_2401_delivery.py`: frozen identical regions and matching rules; retained cycle-1 routing failures and final cycle-2 outputs. Existing files prevent accidental reruns.
- `scripts/compare_2401_context.py`: fixed connection sample with/without context.
- `scripts/validate_public_graph_delivery.py`: existing public fixture graph stage with supplied truth nodes; separate from symbol perception. The butane raster backend was blocked by absent OpenCV and network/TLS installation failure.
- `scripts/build_2401_delivery.py`: single graph build from the reviewed detections, not a repeated full extraction.
- `scripts/finalize_2401_symbol_review.py`: source-audited metadata corrections.
- `scripts/package_2401_delivery.py`: deterministic local packaging and evidence/export validation; no inference calls. Keeps the original machine run unchanged.

The report separates symbol P/R, known subtype correctness, tag exact match, sample connection P/R, latency, repeat variation, completion and costs. No full-2401 graph recall figure or independent second-rater agreement is claimed. The final source gate withheld 15 ambiguous signal types. Three cross-sheet process links have reciprocal references; local OPC port recovery remains incomplete. DEXPI JSON and parsed XML semantic checks pass with reported warnings; an XSD validation pass is not claimed.
