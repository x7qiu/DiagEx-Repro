# PID2Graph extraction study — final results

The completed study supports a small, optional precision improvement to the VLM
symbol pipeline. On the reserved 16-drawing test panel, the fixed source-ink filter
removed 28 false positives without losing a matched symbol. Macro drawing F1 rose
from 0.2207 to 0.2238; recall remained 0.2227. This does not establish reliable
general P&ID extraction. Detector-guided validation showed promising preliminary
results but stopped after provider failures; it remains experimental and was
excluded from selection. On 2026-09-19 the user reported actual OpenRouter spending
of $0.64. The previously reported $58.0094 was an overconservative internal budget
accounting total, dominated by unresolved reservations, not actual spending.

All artifact paths below are relative to the DiagEx project directory. Detailed
run history is in `docs/PID2GRAPH_GOAL.md`; the current process checkpoint is
`output/pid2graph-development/checkpoint-initial.json`.

## Dataset and evaluation contract

The audit found 762 complete image–GraphML pairs and 35,799 patch graphs. The
complete drawings consist of 750 synthetic drawings and 12 previously exposed
OPEN100 drawings. The frozen split contains 528 training, 107 validation, 115 test
and 12 exposed-development drawings. Patches inherit their parent drawing's split
and are not independent samples. Exact-image, thumbnail, annotation-layout and
near-image duplicate relationships were grouped before splitting. Original
annotations were preserved; object-array mapping files were not unpickled.

The paid evaluation uses a panel frozen before inference: 16 validation drawings
and 16 reserved test drawings, each balanced between the two synthetic
collections. These panel results do not describe every drawing in the larger
validation/test splits. Shared generator vocabularies remain a limitation, and
undetected redraw relationships remain possible. OPEN100 is not an unseen-project
test.

GraphML provides the reference boxes and seven broad labels: general, valve,
pump, tank, instrumentation, arrow and inlet/outlet. Connector helpers, crossings,
borders and background are excluded from physical-symbol truth. Predictions are
matched one-to-one at IoU 0.5 with fixed confidence ordering. Primary selection
uses macro drawing F1; ties use cost, runtime and name. Class-agnostic localization
and classification conditional on localization are reported separately. Failed
tiles remain misses in full-drawing denominators. Unattempted drawings cannot
qualify a candidate for selection.

The inference panel contains source images, hashes and dimensions, not GraphML
labels. Training reads training annotations; validation/test inference does not.
Scoring and annotated failure figures are separate from inference. GraphML does
not validate detailed legend meanings, actuation, process service, tags or DEXPI
engineering semantics.

Frozen identities:

- Manifest: `80f54f9d717c528998da3fb704f8244d3ecb3c512ba8fced692017706ddb36e0`
- Panel: `edfa31b1b70b28f76b4b77d125ffef441ed9af7b083485f149db2b4b1810f805`
- Files: `output/pid2graph-development/manifest-v1.json` and `panel-v1.json`.

## Complete VLM baseline and ink-filter comparison

The baseline uses `deepseek/deepseek-v4.1-flash` through OpenRouter, the existing
symbol schema, fixed image tiles and an explicit 120-second perception-request
allowance. It supplies neither native candidate IDs nor detector hints for these
raster drawings. The exact tiling, scan, provider and perception settings and
source fingerprints are saved in `baseline-validation/config.json`.
No drawing-specific legend is supplied in this raster-symbol benchmark; legend
interpretation is checked separately below. These scores measure the symbol
stage, not complete PDF-to-graph accuracy.

All 400 tiles across all 16 validation drawings were attempted: 371 succeeded and
29 failed. Five drawings had no failed tiles; eleven had recorded failures. All
1,085 physical reference symbols remain in the denominator.

| Method | TP | FP | FN | Precision | Recall | Micro F1 | Macro drawing F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Existing VLM baseline | 220 | 439 | 865 | 0.3338 | 0.2028 | 0.2523 | 0.2211 |
| Fixed source-ink filter | 220 | 398 | 865 | 0.3560 | 0.2028 | 0.2584 | 0.2251 |

The filter rejects a predicted box when fewer than 0.1% of its original source
pixels are darker than 180/255. It removed 41 false positives and no class-aware
matches. Eight drawings improved and eight were unchanged. This is a modest
precision gain; it does not address most missed symbols.

The baseline's class-agnostic localization result is 341 matches, precision
0.5175, recall 0.3143 and F1 0.3911. Of those 341 localization matches, 219 have
the correct broad class. Conditional matching is independently computed, so this
count need not equal the 220 matches from class-aware matching. The largest
conditional confusions are general→valve (54) and general→instrumentation (27).

| Reference class | Reference symbols | Baseline TP | Precision | Recall |
|---|---:|---:|---:|---:|
| Arrow | 18 | 0 | — | 0.0000 |
| General | 397 | 21 | 0.4200 | 0.0529 |
| Inlet/outlet | 0 | 0 | 0.0000 | — |
| Instrumentation | 187 | 85 | 0.3864 | 0.4545 |
| Pump | 14 | 0 | 0.0000 | 0.0000 |
| Tank | 16 | 2 | 0.3333 | 0.1250 |
| Valve | 453 | 112 | 0.3003 | 0.2472 |

The current VLM output schema has no separate arrow category. Arrow misses
therefore include a representation limitation, and remain in the denominator.
There are no reference inlet/outlet symbols in this panel; its seven predictions
are false positives, and recall is undefined.

Baseline macro F1 is 0.2786 on Dataset PID and 0.1636 on PID2Graph Synthetic.
The inspected failure sheets show class mistakes despite reasonable localization,
boxes that encompass excess piping/text, and glyphs with no overlapping prediction.
These examples are in `baseline-failure-figures/index.json`, with scoring-only
annotated images that must never be sent to inference.

Reports under `output/pid2graph-development`:

- `baseline-validation-report.json`
- `ink-guard-validation-report.json`
- `baseline-analysis.json`
- `baseline-failure-figures/index.json`

## Reserved final test and integration decision

Selection was sealed from validation before the first final-test API request.
All 16 reserved drawings and all 400 expected tiles were attempted: 386 tiles
succeeded and 14 failed. Eleven drawings had no tile failures; five were partial.
All 1,091 reference symbols remain in the denominator. The report field
`complete: false` records those tile failures; it does not mean that drawings were
omitted. `final-coverage-audit.json` verifies every expected tile ID, configuration,
source fingerprint, pricing snapshot, and drawing summary. The driver exited zero.

| Method | TP | FP | FN | Precision | Recall | Micro F1 | Macro drawing F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cached VLM baseline | 243 | 505 | 848 | 0.3249 | 0.2227 | 0.2643 | 0.2207 |
| Selected source-ink filter | 243 | 477 | 848 | 0.3375 | 0.2227 | 0.2684 | 0.2238 |

The paired comparison uses the same cached VLM responses and makes no extra API
calls. Nine drawings improved and seven were unchanged. The macro F1 increase is
0.00309 (0.31 percentage points), consistent with the small validation gain.
These are descriptive results on a fixed synthetic panel, not a significance
claim or measured industrial-PDF accuracy. No parameters were tuned after testing.

Selected class-agnostic localization has 367 matches, precision 0.5097, recall
0.3364 and F1 0.4053. Conditional classification is correct for 242/367 localized
matches (65.94%). The largest confusions are general→valve (61) and
general→instrumentation (22). Independently matched class-aware TP is 243.

| Reference class | Reference symbols | Selected TP | Precision | Recall |
|---|---:|---:|---:|---:|
| Arrow | 19 | 0 | — | 0.0000 |
| General | 406 | 19 | 0.3585 | 0.0468 |
| Inlet/outlet | 0 | 0 | 0.0000 | — |
| Instrumentation | 186 | 77 | 0.3850 | 0.4140 |
| Pump | 15 | 0 | 0.0000 | 0.0000 |
| Tank | 20 | 4 | 0.5000 | 0.2000 |
| Valve | 445 | 143 | 0.3185 | 0.3213 |

Macro F1 changes from 0.2857 to 0.2901 on Dataset PID and from 0.1556 to 0.1574
on PID2Graph Synthetic. The three lowest-F1 final drawings (Synthetic 199, 63 and
208) all have successful tile coverage. Their inspected detail sheets show
incorrect broad classes despite boxes overlapping source glyphs, partial or
oversized boxes, and entirely missed glyphs including a small inline valve.
These failures therefore cannot all be attributed to API outages. Annotated
figures and the visual-review record are in `final-failure-figures/` and
`final-visual-review.json`; they were created only after inference finished.

Keep `--raster-ink-filter` as an optional supported refinement in the existing
Evidence v2 CLI. Preserve the baseline default and fallback: the filter reduces
some false positives but leaves most misses and class errors. Keep
`--raster-proposals` experimental because its guided VLM validation did not finish.
Do not promote the legend-caption prompt, containment cleanup, or model swap as
proven VLM improvements. The trained detector and its proposal interface are
delivered for reproducible further work; they are not engineering approvals.

Final recorded baseline runtime is 6,337.59 seconds; filtering adds 2.66 seconds.
This excludes the separately recorded 600-second operational pause and unsaved
interrupted-invocation overhead. Final requests account for $9.9785 in charges
plus reservations. Failures comprise nine rate limits, two connection errors,
two API status errors and one interrupted response stream. The final pricing
limits were fixed before final inference; earlier validation used explicitly
documented pricing revisions, so runtime/cost differences across splits are not
controlled performance comparisons.

Final artifacts under `output/pid2graph-development`:

- `final-baseline-report.json` and `final-ink-guard-report.json`
- `final-analysis.json` and `final-coverage-audit.json`
- `final-failure-figures/index.json` and `final-visual-review.json`
- `completion-audit-final.json`

## Five controlled improvements

The registry is `output/pid2graph-development/experiments-v7.json`. Implementation
parity and workflow tests are software checks, not additional accuracy experiments.

| Experiment | Evidence available | Current decision |
|---|---|---|
| Source-ink filtering | Full validation and reserved-test paired comparisons | Small supported precision gain; optional integration retained |
| Supervised detector guidance to the VLM | Detector component complete; guided VLM arm stopped for provider reliability/budget | Incomplete panel; excluded from selection |
| Legend-caption guidance | One paired check of 13 printed legend rows | Inconclusive; targeted generic-valve rejection remained |
| Containment cleanup of detector fragments | Full detector-component comparison | Component evidence only; no downstream VLM benefit established |
| Qwen model swap | Stopped after coordinate-compatibility failure | Excluded from selection |

On the first four completed paired drawings, all from Dataset PID, guided VLM
macro F1 is 0.5803 versus baseline 0.3088. Matched symbols increase from 103 to 199,
false positives decrease from 163 to 73, and recall rises from 0.2601 to 0.5025.
Each drawing improves. This is preliminary, with recorded failures retained;
it cannot establish full-panel performance or justify selection. The other
synthetic collection is not represented in these four results. Details are in
`guided-paired-interim-4.json`.

The guided arm stopped at a tile boundary after 156 of 400 tiles. Six drawings
were fully attempted and a seventh has six attempted tiles; nine whole drawings
remain missing. It produced 444 predictions, including 322 class-aware matches
and 122 false positives. The full-panel report retains all 1,085 truth symbols,
including 763 misses, and reports macro F1 0.2293. This incomplete result is not
selection-eligible and is not a generalization claim: the second synthetic
collection was never reached. A five-minute pause produced no new requests,
but upstream rate limits resumed afterward. See
`guided-stopped-validation-report.json` and `guided-validation/stop-request.json`.
The guided and Qwen arms will not be resumed during final evaluation.

The sealed selection is `selection-v1.json`, SHA256
`db4cf47ff6a2461b374cce00ad6702a2a4d93b97d4a231f401bc1d67f9695f0a`.
Only the completed baseline and ink-filter validation reports were eligible.
The final run uses the same baseline visual method and the selected fixed ink
postprocessing. No final-test results were inspected before this decision.

The Qwen arm used `qwen/qwen3.8-flash` with the existing symbol workflow and no
detector guidance. The stopped run contains 84 attempted tiles across four drawing
summaries, 215 predictions and zero IoU-0.5 localization matches. The remaining
12 drawings are missing. Displaced boxes were inspected before stopping; this
is a negative compatibility result for this coordinate workflow, not a claim
that the model cannot interpret legends or native candidate IDs. Its report is
`qwen-stopped-validation-report-v2.json`.

## Learned detector component

A local Faster R-CNN with MobileNet V3 FPN was trained from official COCO weights
using a prepared pool of 2,640 crops from the 528 training drawings. Only 2,000
crops were actually presented (1,000 steps × batch size 2): 0.758 passes through
that pool. All 528 training drawings were represented, but five prepared crops per
drawing do not cover all annotated symbols. This was a limited training pilot.
The predeclared final checkpoint
is step 1,000; no validation-based checkpoint search was performed. Training took
approximately 1,280 seconds on an Apple M1 Pro using MPS. A failed initial run and
its correction for empty/background proposal batches are retained in the history.

| Component proposals | TP | FP | FN | Precision | Recall | Micro F1 | Macro drawing F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw detector | 1074 | 673 | 11 | 0.6148 | 0.9899 | 0.7585 | 0.6630 |
| Fixed containment cleanup | 1073 | 525 | 12 | 0.6715 | 0.9889 | 0.7999 | 0.7028 |

These are review proposals requiring interpretation, not confirmed engineering
objects or a replacement for the VLM pipeline. Cleanup removes 148 false positives
but loses one match; its effect on the VLM has not been established. Training has
no positive inlet/outlet examples, and pump/tank examples are scarce.

Weights: `output/pid2graph-development/detector-v2/step-1000.pt`

SHA256: `8d05717fd762edc88658c809a119e6a9d079777933807a6d99903944fb0ee2e1`

The full recipe, environment records, thresholds, source snapshot, limitations and
new-image CLI command are in `docs/PID2GRAPH_DETECTOR.md`. Raw component inference
took 221.7 seconds for all 16 validation drawings and made no API calls.

## Optional integration and native PDF checks

`diagex.vision.raster_detector` generates source-bound proposal JSON for a new
raster drawing. Its outputs matched all 1,747 saved validation proposals exactly.
The existing CLI accepts these through the experimental `--raster-proposals`
option for Evidence v2. It passes fallible hints to the VLM after the legend stage;
it does not create native evidence IDs, approvals or detections from those hints.
Omitting the option preserves the baseline.

All 400 tile hint payloads match the frozen evaluation method exactly. Sixteen
new integration tests cover source/coordinate validation, cache separation,
baseline fallback, CLI forwarding and preflight rejection. The option currently
requires a single raster image, fixed inspection and deskew disabled. It remains
disabled by default and experimental: the incomplete guided arm does not support
promoting it to the selected pipeline.

The selected source-ink filter is available through the optional
`--raster-ink-filter` flag on `extract-pid --engine evidence-v2`. It runs after
legend resolution and VLM symbol inspection, before the symbol review bundle and
fusion. It filters both raw detections and unanchored review proposals in original
image pixels. Rejected observations, source hash, thresholds and implementation
hash remain in `raster-ink-filter.json`; original tile checkpoints are preserved.
Reviewed graph inputs bypass perception and are not filtered again.

The flag requires a single raster image, fixed inspection and deskew disabled.
Omitting it preserves the baseline and its cache identity. Filtered runs have a
separate cache identity. The production pixel filter and coordinate adapter match
all 659 validation predictions exactly, including the same 41 rejections
(`ink-production-parity.json`). Eight new filter tests and the related guidance,
CLI, review, evidence, response-contract and topology checks total 125 passing
tests. This verifies implementation behavior. Final results above support the
small precision gain; the filter remains optional because recall remains low.

Example using an existing drawing legend:

```sh
.venv/bin/diagex extract-pid drawing.png --engine evidence-v2 --legend customer-legend.pdf --raster-ink-filter
```

The production request-timeout default is unchanged; the benchmark explicitly
uses 120 seconds. These integration checks do not imply that every production
configuration reproduces the benchmark's accuracy or reliability.

Native vector-PDF text and geometry extraction remain the existing path. Four
selected tiles from two previously exposed PDFs were checked qualitatively:

| PDF sample | Native text spans / paths | VLM detections | Fused nodes | Topology edges |
|---|---:|---:|---:|---:|
| 2401, page 4 | 197 / 5789 | 1 | 1 | 0 |
| Covestro, sheet 1 | 1046 / 5602 | 15 | 16 | 12 |

The extra Covestro node is an unclassified native candidate requiring review;
one topology edge is uncertain. Internal geometric route checks are not
independent engineering validation. Four selected tiles do not establish
full-sheet recall, and neither PDF has an independent reference graph here.
Evidence and inspected overlay hashes are in `pdf-transfer-review.json`.

Legend interpretation was checked separately against 13 printed rows in a Chinese
PDF. The baseline was source-consistent for 11 rows; the caption-guided run was
consistent for 12. Both still rejected the targeted generic-valve row. One fresh
run per prompt on an exposed source is insufficient to claim a general improvement,
so that prompt change was not integrated. See `legend-comparison-v1.json`.

A controlled PDF fixture exercised legend prerequisite enforcement, an agent's
symbol-label edit, a frozen review snapshot and the real graph runner. It produced
two nodes and one edge, preserved reviewed labels/boxes and left the source run
unchanged. Quality remained partial and review provenance was explicitly agent,
not human. See `real-graph-workflow-v3/report.json`. This check found and fixed an
OpenCV Hough-line output-shape crash in `diagex.vision.topology`; the affected
topology/review tests passed. This is workflow evidence, not an engineering sign-off.

The paused P&ID Inspector service was not restarted or developed for this work.

## Runtime, cost and reproducibility

Recorded baseline runtime is 8,251.91 seconds. It includes saved tile processing
and cooldown durations plus page preparation, but excludes interrupted invocation
overhead that was not recorded in a tile. Ink filtering adds approximately 2.40
seconds. The provider produced connection errors and rate limits; these failures
remain in the accuracy results rather than being silently retried until successful.

Cost correction (2026-09-19): the user reports actual OpenRouter spending of
$0.64. This has not been independently reconciled against individual requests.
The old reservation total must not be presented as actual spending or as proof
that the API allowance was almost exhausted. Future runs need separate accounting
for actual billed cost, live reservations and unresolved requests. Historical
ledger entries remain preserved for audit. This correction supersedes the cost
interpretation in the earlier completion report; accuracy results are unchanged.

The historical internal ledger totals $58.0094: baseline $28.9769, experiments $18.9127,
legend/transfer/workflow checks $0.1413 and final test $9.9785. These are charges
plus conservative reservations, not an invoice. No paid goal process remains.

The user-authorized cumulative OpenRouter cap is $60. The ledger reserves a verified
maximum request cost before dispatch and settles successful calls from token usage
at price ceilings. Uncertain failures retain reservations. Baseline drawing
checkpoints account for $25.9264 charged/reserved; the baseline category totals
$28.9769 including smoke calls and interrupted-call overhead. Neither figure is
an actual invoice. The separate authenticated key-usage snapshot reports $0.3305
for the key's current UTC day; it is key-wide and does not reconcile individual
requests. No uncertain reservations were released based on that snapshot.

A prospective pricing revision was activated in `prices-v5.json`. Fresh endpoint
metadata lists DeepInfra and Fireworks as the allowed endpoints supporting the
required named-tool call, with maxima of $0.22/M input and $0.66/M output tokens.
The prepared guard uses those maxima through the server's `max_price` filter,
instead of the earlier $0.30/M and $1.20/M bounds. It preserves the allowed tags
and the currently eligible provider set; future repricing could still affect
availability. The change is operational and must remain explicit when comparing
cost/reliability. It does not release historical reservations or alter source
images, prompts, labels, detector settings or the scoring contract.

The guided run resumed with this revision after its original driver stopped at
an internal allocation limit. The 116 existing tile hashes were verified unchanged,
and a new live request used the lower reservation (`pricing-guard-live-check.json`).
The pricing revision initially set allocations to baseline $28.99, experiments
$20.51, final $10.30 and legend transfer $0.20, totaling $60. That final allocation
preserved at least the old $14 allocation's request capacity because the maximum
per-token price ratio is 0.7333. After the non-final arms closed, their unused
allowances were assigned to final evaluation: current allocations are baseline
$28.98, experiments $18.92, legend transfer $0.15 and final $11.95, still totaling
$60 (`final-allocation-revision.json`). No historical reservations were released. Consult `pricing-guard-revision.json` and the current checkpoint for status.
New tiles archive their exact pricing snapshot
and record its hash. The selection CLI now requires `--prices` and pins the final
provider price limits; final inference rejects changed limits before creating an
API client. Five focused budget-resume/selection tests pass. No actual test-panel
images were used by those tests.

The shared ledger is `output/pid2graph-development/spending.json`. Allocations and
live progress are in `checkpoint-initial.json`; operational evidence includes
`budget-stop-live-check.json`, `connection-accounting-policy.json`,
`budget-resume-audit.json` and `key-usage-probe.json`. An attempted tile is saved
before a budget exception; an untouched tile stays resumable. Failed or completed
tile records are not overwritten on resume.

To re-score the completed baseline without API calls, from the DiagEx directory:

```sh
.venv/bin/python -m eval.pid2graph score --manifest output/pid2graph-development/manifest-v1.json --panel output/pid2graph-development/panel-v1.json --run output/pid2graph-development/baseline-validation --out output/recomputed-baseline.json --split validation
```

To reproduce conditional classification, confusions, per-drawing filter deltas
and raw tile failure counts from the saved validation reports:

```sh
.venv/bin/python -m eval.pid2graph.analyze --report output/pid2graph-development/ink-guard-validation-report.json --paired output/pid2graph-development/baseline-validation-report.json --run output/pid2graph-development/baseline-validation --out output/recomputed-validation-analysis.json
```

The analysis command reads saved scores and optional tile checkpoints; it makes
no API calls and does not change matching rules. It rejects paired reports with
different splits, manifests, panels, case IDs or truth denominators. Its validation
check reproduced the original baseline conditional classification and confusion
counts, all 16 filter deltas, and all 400 tile states and failure types. Saved
output is `validation-analysis-reproduced.json`. The same command can summarize
the final reports after evaluation finishes; paired deltas are descriptive and
do not establish statistical significance.

Use a new output path: reports intentionally refuse overwrites. Other run
directories can be scored with the same command and their frozen split. Paid
inference commands, saved environments and price verification are recorded in the
goal log and model card. Price ceilings must be current before new paid calls;
credentials are supplied through the existing environment and are not artifacts.

To reproduce the final scores and paired analysis without inference:

```sh
.venv/bin/python -m eval.pid2graph score --manifest output/pid2graph-development/manifest-v1.json --panel output/pid2graph-development/panel-v1.json --run output/pid2graph-development/final-baseline --out output/recomputed-final-baseline.json --split test
.venv/bin/python -m eval.pid2graph score --manifest output/pid2graph-development/manifest-v1.json --panel output/pid2graph-development/panel-v1.json --run output/pid2graph-development/final-ink-guard --out output/recomputed-final-ink.json --split test
.venv/bin/python -m eval.pid2graph.analyze --report output/recomputed-final-ink.json --paired output/recomputed-final-baseline.json --run output/pid2graph-development/final-baseline --out output/recomputed-final-analysis.json
```

The requirement-by-requirement completion audit is
`output/pid2graph-development/completion-audit-final.json`. The remaining technical
limitations are low symbol recall, broad-class errors, the arrow schema gap,
incomplete guided-VLM evidence, synthetic-domain coverage and qualitative-only
legend/PDF transfer evidence. Addressing those would require a subsequent study;
the completed final panel must not become a tuning set disguised as an unseen test.
