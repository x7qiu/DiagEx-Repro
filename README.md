# diagex — reproducibility package

This repository is the public reproducibility package for the manuscript

> **DiagEx: Zero-Shot P&ID-to-DEXPI Digitization with a Vision-Language Agent**
> Heiko Koziolek, Thilo Braun, Tabea Bordis
> ABB Corporate Research Center, Mannheim, Germany
> *Manuscript under review — venue and DOI to be added on acceptance.*

It contains:

- The diagex Python package (`src/diagex/`) that performs the extraction.
- The evaluation harness (`eval/`) used to produce every number in the
  paper.
- Ground-truth corpora (`eval/datasets/`) for the 10 public P&IDs the paper
  evaluates against.
- The 10 source PDFs (`tests/p-ids-public/`).
- Final results CSVs, tables, and figures (`results/`) for Phase 1, Phase 2,
  the no-tile ablation, and the GPT-4.1 baseline.
- Replay cassettes (`eval/cassettes/`) so reviewers can re-derive results without
  an Anthropic API key.
- The unreviewed manuscript PDF (`paper/manuscript.pdf`) and the design-time
  evaluation plan (`paper/evaluation-plan.md`).

Per-run agent transcripts, tile crops, intermediate DEXPI JSON, and full
debug reports live in the **separate Zenodo supplementary archive** linked
from `.zenodo.json` (cross-DOI). They are not in this repository because
they are large (~150–250 MB) and not needed to verify any number.

## Quick start (no API key required)

```bash
git clone <REPLACE_WITH_REPO_URL> diagex-repro && cd diagex-repro
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .

# Run the unit suite (validates eval logic + DEXPI 2.0 round-trip).
pytest tests/unit/

# Re-derive Phase 1 results from cassettes.
python eval/run_paper_eval.py \
    --fixtures eval/datasets/manifest.yaml \
    --phase 1 --conditions baseline --use-cassettes \
    --out /tmp/repro
diff /tmp/repro/results.csv results/phase1/results.csv
```

If the diff is empty, your environment reproduces the published Phase 1
numbers byte-for-byte. See `docs/REPRODUCE.md` for the equivalent commands
for Phase 2, the ablation, and the GPT-4.1 baseline.

## Live mode

To re-run the agent against the real PDFs (Anthropic API key required):

```bash
export ANTHROPIC_API_KEY=...
python eval/run_paper_eval.py --conditions baseline --phase 1 --out /tmp/live
```

For the browser-based workflow, start the local workbench instead:

```bash
diagex web
```

The dashboard configures the provider, API key, vision and reasoning models,
reasoning mode, effort, and extraction engine. It accepts a P&ID upload, lets
the operator resume compatible artifacts or start a fresh run, preserves the
live extraction log, and opens a completed graph in human review. For Evidence v2,
**Detect legends & symbols** stops before fusion, connectivity and export. Open
**Review legends & symbols**, check the legend, correct/add/reject symbols and
check each P&ID page for missed symbols. **Build graph…** returns to model settings;
**Build reviewed graph** then uses the saved decisions in a new run without
repeating legend or symbol model calls. The original detection run stays available.
The dashboard lists runs containing `graph.json` or `detection.json`; older runs can be linked to
their original PDF before review, with checkpoint source hashes enforced when
available. API keys entered in the dashboard remain in process memory and are
not written to run artifacts. The server binds to `127.0.0.1` by default;
non-loopback use has no authentication or TLS and is intended only for trusted
networks.

OpenRouter is supported through its Anthropic Messages-compatible endpoint:

```bash
export DIAGEX_LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=...
export DIAGEX_MODEL=provider/model-slug
# auto follows --effort; enabled/disabled force all model calls
export DIAGEX_REASONING=auto

diagex extract-pid path/to/drawing.pdf --effort medium
```

Choose a model that supports both image input and tool calling. `OPENROUTER_MODEL`
is accepted as an alias for `DIAGEX_MODEL`. Optional attribution settings are
`OPENROUTER_HTTP_REFERER` and `OPENROUTER_APP_TITLE`.

`DIAGEX_REASONING` accepts `auto`, `enabled`, or `disabled` (`on`/`off` and
`true`/`false` are aliases). `auto` retains the existing effort-based behavior.
In evidence-v2, object and line perception remain non-thinking regardless of
this setting; `DIAGEX_REASONING` controls the separate semantic relationship
solver. Legacy extraction continues to apply it to its model-driven stages.

### Evidence-first extraction (opt in)

See [How vector P&IDs become a connected graph](docs/PID_RECONSTRUCTION.md)
for the reconstruction flowchart, ownership rules, assembly review behavior,
validation procedure and current limitations.

The staged evidence-first engine is available alongside the paper-compatible
legacy agent:

```bash
export DIAGEX_PID_ENGINE=evidence-v2
# Optional: use a fast vision model and a stronger page-relationship solver.
export DIAGEX_VISION_MODEL=qwen/qwen3.7-flash
export DIAGEX_REASONING_MODEL=qwen/qwen3.8-max

diagex extract-pid path/to/drawing.pdf --effort medium
# Equivalent one-run override:
diagex extract-pid path/to/drawing.pdf --engine evidence-v2
```

To inspect only native PDF text, printed tags, and automatically detected
legend pages, without running object perception, topology, relationship
solving, or DEXPI export:

```bash
diagex inspect-pid-evidence path/to/drawing.pdf
```

The command scans every page deterministically so it can classify page roles,
then sends only the detected legend pages to the vision model. Vector-PDF
abbreviation tables such as `通用缩写` and `仪表类型缩写` are reconstructed from
positioned native text and merged into the same project legend used by a full
evidence-v2 run. It writes a human-readable `legend.learned.md`, a compact
`legend.learned.json`, the full image-bearing `legend.json`,
`legend.abbreviations.json`, and a self-contained `legend.review.html` with
source-row crops, filters, editable normalized fields, approve/reject state,
and reviewed-JSON download. Per-page native evidence and
`evidence/native-text-inventory.json` are also retained beneath the inspection
run directory.
Because engineering-object perception is intentionally skipped, tag inventory
items are candidates rather than node assignments.

Evidence-v2 extracts positioned PDF text and vector paths before making
stateless raw-symbol calls over a deterministic overlapping crop grid. Each native
candidate gets one symbol, non-symbol, or uncertain outcome; assembly membership
and graph eligibility are decided later. Every symbol crop starts with reasoning
off and a 6,000-token allowance. Explicitly enabling reasoning permits one targeted
16,000-token check for uncertain or rejected native candidates, geometry conflicts,
and ambiguous rounded equipment bodies, with enlarged source details. Malformed responses get a short
repair instead. Requests have time and retry limits, and the stage stops early on
repeated contract failures or after 30 minutes, preserving completed work and
diagnostics. It fuses the objects once, then runs a bounded contextual-symbol check only for
small unclassified glyphs next to valve bodies. That check compares an
annotated page overview and high-resolution local crops with the saved
project-legend symbol images. A high-confidence, directly attached composite
can be represented as one valve with an `actuation` attribute; unsupported or
ambiguous merges remain review conflicts. Line topology is reconstructed only
after this correction, preventing actuator strokes from becoming process
connections. Native PDF tag text and project-legend abbreviations then
deterministically enrich or correct unambiguous node kinds; conflicting
specific object evidence is preserved as a review conflict rather than being
overwritten. For vector
PDFs it keeps visual line style separate from engineering meaning: explicit PDF
dash metadata is preserved, and regularly spaced collinear fragments are
grouped into dashed/dotted/dash-dot evidence when CAD exports encode every mark
as an independent solid stroke. A bounded non-thinking visual pass then reports
only route visibility, endpoint contact, stroke style, visible arrows, and
project-legend matches. Deterministic endpoint rules reject unsupported signal
combinations and calculate evidence-based edge confidence. Finally, the
text-only relationship solver divides ambiguous candidates into bounded groups.
Each group uses a thinking call without tools to produce a compact engineering
memo, followed by a low-effort call that serializes that same memo into the
strict graph schema. Models that permit reasoning to be disabled use the more
reliable non-thinking serializer; reasoning-mandatory models such as GLM-5.3
keep reasoning enabled at low effort and use that same model. Provider aliases
that newly require reasoning recover once from the explicit validation error.
This avoids introducing a separate serializer model while retaining the
DeepSeek-compatible split between extended reasoning and forced tool selection.
Set `DIAGEX_REASONING_MODEL` explicitly when the reasoning model differs from
the vision model; the selected models are recorded in `result.json`.
Scanned PDFs use an optional OpenCV fallback, installed with
`pip install -e '.[vision]'`; raster dash-pattern inference is not yet enabled.

Object fusion requires overlapping observations of compatible physical symbols
or supporting native vector geometry. Nearby tags and large enclosing boxes
cannot merge distinct objects; uncertain matches remain separate review candidates.
Fused geometry retains a representative observed box, or a closed native outline
when complementary crops prove a single symbol. Fusion provenance is saved in
node attributes. A fusion-version upgrade reuses native evidence and perception;
completed runs are upgraded in a new directory to preserve their graph and review.
Quality metrics report accepted and provisional connectivity separately.

To audit fusion offline without making model calls or changing the original run:

```bash
python scripts/evaluate_fusion.py --run-dir runs/<drawing>/<run-id> --out tmp/fusion-audit
```

This also checks reviewed instances from the two public vector fixtures with
simulated duplicate crop observations. Its scores evaluate fusion, not perception
or end-to-end connection accuracy.

Fusion and topology share a native-vector symbol scene. Page-wide face recovery
extends clipped detections to supported physical outlines; open glyphs use local
non-collinear symbol strokes. Ownership is recorded per segment, so a PDF path
can contain both a symbol and an external pipe. Separate observations supported
by one physical symbol share its evidence identity. Conflicting printed tags
remain review candidates and cannot become accepted connections to each other.

Each accepted local route must have independently checked native ports, source
segments, and distinct physical endpoints. Box contact alone stays provisional.
Nozzle/flange chains can establish ports outside the body. Missing large rounded
bodies are retained as unclassified review candidates, including outlines made
from flattened curves; these hypotheses cannot certify accepted connections.
Ordinary closed piping loops are not automatically classified as equipment.
Crossings remain separate unless native junction evidence supports a connection.
Model scores cannot override the structural acceptance gate.

A displaced detection can recover a uniquely supported nearby native glyph with
matching dimensions, overlapping extent and external pipe contact. Another
object's well-localized glyph cannot be borrowed. Failed route alternatives in
one native network share a review decision; distinct supported ports remain
separate. Each alternative retains its original geometry and evidence for route
selection in the review workbench. Same-symbol and zero-length proposals become
instance/geometry conflicts instead of connection edges. Connection approval or
rejection also resolves its linked route conflicts, with atomic undo; object
risk indicators follow the remaining unresolved conflicts. Independent object
classification decisions remain explicit.

Native arrowheads determine flow direction independently of endpoint ordering.
Unknown direction is retained in graph/DEXPI attributes and rendered without a
flow arrow. Existing graph interfaces remain compatible; `route_evidence` records
ports, supporting paths, and reasons for uncertainty. The checkpoint marker
`port_topology: 4.0.0` invalidates topology and downstream results while retaining
native evidence and perception. Fusion version 4.0.0 also invalidates contextual
interpretation because instance geometry can change. Native evidence on rotated
PDF pages is aligned with the rendered page; only legacy rotated native layers
are regenerated during reuse. Restart a running workbench before reusing a run
so it loads the new code. Older native evidence without fill metadata leaves
otherwise ambiguous four-way junctions unresolved.

For an offline topology replay using the fusion audit's saved objects:

```bash
python scripts/evaluate_topology.py --run-dir runs/<drawing>/<run-id> \
  --objects tmp/fusion-audit/objects.replayed.json --out tmp/topology-audit
```

The report compares local endpoint pairs against the public fixture graphs and
keeps structurally supported candidates separate from provisional candidates.
It does not rerun semantic review or claim end-to-end extraction accuracy.

Each run writes `evidence/page-XXXX.json`,
`evidence/native-text-inventory.json` (assigned, excluded, ambiguous, and
unresolved printed tag candidates),
`checkpoints/contextual/page-XXXX.json`,
`checkpoints/line_evidence/page-XXXX.json`,
`checkpoints/page_graph/page-XXXX.json`, `checkpoints/manifest.json`, and
`quality.report.json` in addition to the existing graph and DEXPI artifacts. An
interrupted matching run resumes missing crops or page solves automatically;
source, configuration, model, and checkpoint schema hashes prevent incompatible
runs from being reused. The default remains `legacy` until evidence-v2 clears
the evaluation gates.

For a live Phase 2 A/B run on the existing ground-truth fixtures:

```bash
python eval/run_paper_eval.py --phase 2 \
  --conditions baseline,evidence-v2 --out /tmp/diagex-v2-ab
```

Kimi Code K3 is also supported with the Kimi Code Console key:

```bash
export DIAGEX_LLM_PROVIDER=kimi
export KIMI_API_KEY=...
export KIMI_BASE_URL=https://api.kimi.com/coding/v1
export DIAGEX_MODEL=k3

diagex extract-pid path/to/drawing.pdf --effort medium
```

DiagEx accepts Kimi's OpenAI-style `/coding/v1` setting above, but uses Kimi's
Anthropic-compatible `/coding/v1/messages` endpoint internally so image and tool
blocks do not need conversion. `KIMI_MODEL` is accepted as a model-name alias.

New run directories include the sanitized model name after the timestamp, for
example `runs/2401/2026-08-16T10-30-00_qwen-qwen3.7-flash_r-ab12/`.

Live results are non-deterministic; expect ±0.02 macro-F1 around the
published numbers per the evaluation plan §8.1.

## Human review workbench

Review an extraction beside its original PDF in a local browser:

```bash
diagex review runs/<drawing>/<run-id> \
  --pdf path/to/drawing.pdf \
  --rater "Reviewer name"
```

The left pane preserves the source page and the right pane shows editable,
source-aligned entities and connections. Every action is autosaved under the
run's `review/` directory, so Ctrl+C is safe and the same command resumes the
session. Final export remains disabled until every page, entity, connection,
and extraction conflict has an explicit disposition. Completion writes
`graph.reviewed.json`, `pid.reviewed.dexpi.json`,
`pid.reviewed.dexpi.xml`, and `review.report.json` without changing the
original `graph.json`.

The queue opens on **Needs your decision**, with identifier, symbol, and
connection filters, search, page selection, source-location links, and a
**Next finding** button. **Awaiting human approval** contains the full sign-off
worklist. Clearing flagged findings does not complete human review.

Additional audit questions can be supplied in `review-findings.json` beside
`graph.json`. It contains the matching `graph_sha256` and `source_sha256`, and a
`findings` list of `{ "id": "stable-id", "conflict": { ... } }` records.
Each conflict needs a `type`; optional `title`, `question`, `queue_category`,
`page_index`, `node_id`, and `source_locations` provide the review context.
Each source location has `page_index`, `bbox_global`, and `label` in the graph's
page coordinates. Imports preserve existing review events and reject changed
content under an existing ID. Audit decisions require a written clarification.

## Layout

```
src/diagex/         diagex Python package (the extractor)
eval/               evaluation harness, ground-truth corpora, replay cassettes
  ├─ *.py           harness modules
  ├─ datasets/      ground-truth corpora (manifest + per-fixture truth)
  └─ cassettes/     offline replay artefacts (no API key needed)
scripts/            operational scripts (rendering, rescoring, build)
tests/
  ├─ unit/          eval-harness + DEXPI 2.0 layer unit tests
  ├─ fixtures/      shared test fixtures (DEXPI 2.0 reference XML)
  └─ p-ids-public/  the 10 public source PDFs
results/            final CSVs, tables, figures, reports
docs/               reproduction runbook, data card, architecture sketch
paper/              the manuscript PDF + evaluation plan
data/               supporting illustration assets (e.g. two_tanks_hires.png)
```

A full file-by-file map lives in `docs/REPRODUCE.md` (which paper number
maps to which CSV row to which command).

## Licensing

- diagex source: Apache-2.0 (see `LICENSE`).
- DEXPI 2.0 specification sources, vendored under
  `src/diagex/dexpi/codegen/vendored/`: CC-BY 4.0 (DEXPI Initiative; see
  `NOTICE` and `src/diagex/dexpi/codegen/vendored/PROVENANCE.md`).
- All other third-party code: their original licenses, listed in `NOTICE`.

## Citing

See `CITATION.cff` (GitHub renders it as a "Cite this repository" widget).

## Limitations

- Reproduction needs Python 3.11+ on Linux/macOS. The `[codegen]` extra
  (which requires Python 3.12+ and `dexpi.specificator==1.0.0`) is **not**
  required to reproduce results — the DEXPI 2.0 model is checked in under
  `src/diagex/dexpi/_generated/`.
- The 10-fixture corpus is the public subset; private customer P&IDs that
  appear in some paper figures are not redistributable and are not in this
  archive.
- Live-mode reproduction depends on Anthropic API availability and current
  pricing of `claude-opus-4-7`. Cassette mode is fully offline.
