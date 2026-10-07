# Detector-assisted VLM study, phase 2

This study is active and incomplete. It extends the first pilot without changing
its archived measurements or reusing its final panel as unseen evidence.

## Frozen scope and data

Keep the original 528/107/115 drawing-group train/validation/test split and the
12 exposed development drawings. All patches inherit their source drawing split.
Keep the original IoU 0.5 matching and full-drawing denominators. Do not read new
test annotations during development or use annotated overlays as inference input.

`output/pid2graph-phase2/panel-pilot.json` retains the original 16 validation
drawings. `panel-broad.json` includes all 107 validation drawings. Both share a
new final panel of 16 previously unused test drawings, selected by deterministic
ID hash, eight from each synthetic collection. All 16 previously tested drawing
groups are excluded; 99 candidates were available. `freeze.json` records the
source manifest identity and the prior test-run configurations inspected. No new
test image or graph was opened by panel preparation.

## Findings that determine the next work

The pilot prepared five 768-pixel crops for each training drawing, but its
1,000 steps at batch size two presented only 2,000 of 2,640 crops. Reconstructing
the deterministic sample order confirms that every training drawing appeared,
but only 6,134 of 45,021 physical symbols appeared with at least half their box
retained (13.6%). Only 5,275 were fully visible. Even the entire prepared pool
covered only 7,843 symbols. `pilot-training-coverage.json` reports each class and
drawing. More epochs on the same crop pool would not solve its coverage gap.

On the six fully attempted guided validation drawings, detector-only class-aware
TP was 626/628; the existing VLM matched 152 and the guided VLM matched 309.
317 detector-matched reference symbols were absent from guided class-aware
matches. There were also 20 failed guided API tiles. This evidence is confined to
Dataset PID and cannot distinguish every model rejection from API failure or
classification/localization loss. The old output lacks explicit proposal decisions.
See `pilot-proposal-retention.json`.

## Work sequence and stopping rules

1. Repair billing before paid calls. Preserve the previous ledger. Initialize
   this phase with the user's reported $0.64 historical spend, identified as
   user-supplied rather than request-level reconciliation. Track billed cost,
   active reservations, and unresolved requests separately. Capture generation
   IDs even on partial streams and reconcile authoritative usage when available.
   A read-only current-key snapshot has been recorded; it is key-wide, not a
   historical study invoice. Cumulative study authorization remains $60.
2. Prepare training crops that cover the annotated physical symbols, preserving
   suitable resolution and complete boxes. Include background/pipe negatives and
   rare-class sampling. Report exact annotation coverage and exceptions. Do not
   silently exclude large symbols or use validation/test patches for training.
3. Measure local training throughput on the prepared data. Before the full run,
   record a finite maximum epoch count, early-stopping patience and checkpoint
   selection rule. Train for multiple epochs, validate checkpoints on the pilot
   panel, and compare with the delivered first-study checkpoint. Report epochs,
   crop presentations, coverage, convergence and per-class performance.
4. Complete controlled VLM comparisons on the pilot panel, using the same source
   images and common model/provider policy across each paired experiment. Compare
   detector-only, baseline VLM and detector-assisted VLM. Separate failed calls,
   explicit model decisions, box/class corrections and unresolved proposals.
5. Evaluate up to two validation-selected hybrid finalists, the reference VLM
   and relevant detector component on the broader 107-drawing validation split.
   Select using validation macro drawing F1, with recall, precision, per-class,
   per-collection, accepted/review inventories and actual cost visible together.
   Freeze the winner before one final evaluation on the new reserved panel.
6. Integrate only supported changes with baseline fallback. Preserve legend-first
   interpretation and native PDF text/geometry, verify the review-to-graph workflow,
   and inspect representative qualitative PDF transfer examples. Deliver code,
   weights, commands, final metrics, failures and billing records.

All four hybrid experiment slots are registered: existing advisory guidance;
explicit decisions for each proposal; proposal-centered source crops; and
broad-class-first recognition with detailed legend interpretation kept separate.
Exact configurations are registered before validation inference, including which
variables change. Do not exceed four variants or quietly alter an existing variant.
Checkpoint comparison and software verification are not extra hybrid variants.

`validation-promotion-policy.json` fixes the remaining finite sequence. Finish
all four registered pilots before choosing the top two algorithms by validation
macro drawing F1. After training finishes, compare those two algorithms using
the best post-pilot epoch (epoch >= 1, chosen by detector validation macro F1)
against their existing pilot-weight results. Register replacement proposal
hashes before those at most two additional pilot runs. Carry each algorithm's
stronger checkpoint configuration to broader validation, for at most two hybrids
plus the baseline. The best post-pilot epoch is still compared if epoch zero
wins the detector-only selection; training improvements are not presumed to
improve VLM interpretation. No final selection/test starts before these stages
and the training convergence report finish.

Selection uses the original macro drawing F1 primary metric. For exact ties,
version-2 reports now use fully accounted actual billed cost, then runtime and
method name. Unknown billing totals rank after known totals; reservations and
unresolved maximum exposure never masquerade as prices. A higher accuracy score
still wins regardless of billing completeness. Legacy report selection remains
separate and cannot be mixed with actual-billing reports. Five selection tests
and eight validation-reuse tests pass, including actual-cost versus reservation
ranking, unknown-cost handling, unchanged final signatures, and test exclusion.

Slot 1 is registered in `hybrid-01-advisory-pilot-registration.json`: the
existing advisory guidance with the delivered pilot detector, all 16 validation
drawings and 1,747 source-only proposals. It changes only the guidance context
relative to the new VLM baseline. Its run `advisory-pilot-v1` has finished all
400 attempts, using the same locked cumulative spending ledger. A later comparison
using selected trained weights must register the replacement proposal hashes
before inference.

Slot 2 is registered in `hybrid-02-explicit-pilot-registration.json`, with
`raster_review_runner.py` / `raster_review_drive.py` and the optional
`diagex.vision.raster_review` adapter. It keeps the existing symbol ontology,
tile images, coordinate projection, discovery schema, and token/transport limits,
while requiring a symbol/reject/uncertain decision for every raster hint. Missing,
duplicate, unknown, or malformed decisions never become accepted symbols. Raw
decisions, proposal IDs, and omissions are preserved per tile. The existing
baseline/advisory source is unchanged. Validation for slot 2 has finished in
`explicit-pilot-v1`; its complete attempted coverage and partial failures are
reported below.

Eleven adapter/geometry tests and two decision-audit tests pass. A source-only
CPU detector produced three proposals on one training crop. The first live smoke
exposed the scanned perception path's promotion of accepted outputs to graph
eligibility; that experimental revision is archived under
`explicit-review-train-prep/source-v1`. The corrected wrapper retains accepted
raster observations in the geometry review queue. The repeated live smoke
`explicit-review-train-smoke-v2` returned two symbol decisions and one rejection,
retained two review proposals, and produced zero graph-eligible raster outputs.
Its completed request billed $0.0001817928. The rejected proposal was explicitly
called a flow arrow outside the current physical-symbol ontology, useful evidence
for the planned broad-class variant, not a validation result. Neither smoke is
used for accuracy selection. The final slot-2 sources are archived in `source-v2`.

`eval/pid2graph/raster_decision_audit.py` distinguishes explicit rejection,
uncertainty, missing response, conflicting decisions across views, and recognized
proposals absent from final geometry output. It verifies source/configuration
provenance, refuses unfinished tile coverage, and scopes IDs per drawing.
These observations supplement the independent truth-matched retention report;
neither VLM confirmation nor matching GraphML conveys engineering approval.

Slot 3 (`hybrid-03-detail-pilot-registration.json`) adds source-crop contact sheets
to slot 2: every supplied hint, up to 80, receives context padding and a labeled
crop, with at most 16 cells per sheet. The original image and its output coordinate
frame remain unchanged. Six checks cover crop provenance, caption separation,
all-80-hint coverage, invalid input, and message preservation. The initial live
training smoke recognized glyphs but omitted boxes, so those decisions remained
unresolved. After clarifying the same mandatory box contract, a new smoke returned
three review proposals without coordinate autofill. The flow arrow was mapped to
`general` under the unchanged detailed ontology. Both revisions are retained;
only the corrected revision is registered. The crop sheet was visually inspected.

Slot 4 (`hybrid-04-broad-pilot-registration.json`) uses the seven broad classes
directly, including flow arrows. Its smaller schema and prompt omit detailed
engineering subtype/actuation fields. It uses the same original tile images,
detector hints, token limits, ownership, and coordinate projection, without crop
sheets. Caller-prepared legend context is preserved; no legend is synthesized
from GraphML. Seven checks cover malformed/duplicate decisions, missing boxes,
native-candidate protection, bounded format recovery, and source-frame geometry.
The live training smoke returned arrow/instrumentation/valve observations, all
requiring review, with no native IDs or graph eligibility. Detailed legend-based
semantic refinement/integration remains required if this method is selected.
Neither smoke is accuracy evidence. Slot 3 validation has finished in
`detail-pilot-v1`; slot 4 validation has finished in `broad-pilot-v1`. Both use
the unchanged registered sources and original pilot-detector proposals.

## Completed advisory comparison

`advisory-pilot-v1` finished 400 attempts over all 16 validation drawings, with
399 successful tiles and one failed tile retained in the denominators. Its score
report has `complete: false` because one drawing is partial; attempted coverage
is finished, not an interrupted run. TP/FP/FN are 543/401/542; precision is
0.575212, recall 0.500461, micro F1 0.535239, and macro drawing F1 0.415229,
compared with baseline macro F1 0.236081. Localization TP is 815. The independent
class-agnostic assignment has 542 correctly classified matches out of 815;
this differs from the class-aware TP count because the assignments are separate.

The gain is strongly collection-dependent: Dataset PID micro F1 is 0.607533
(baseline 0.256927), while PID2Graph Synthetic is 0.224543 (baseline 0.221538).
The latter still confuses general symbols with valves and instruments. Across
both collections, the largest localized confusions are general→instrumentation
(105) and general→valve (83). Per-class and paired drawing results are preserved
in `advisory-pilot-analysis-v2.json`. These are descriptive validation comparisons,
not final test findings or evidence of detailed engineering correctness.

`advisory-pilot-retention.json` finds 541 retained detector truth matches out of
1,074 (50.37%). Among the 533 lost matches, 269 remain localized without a correct
class match and 264 have no localization match. Two additional truth symbols
are recovered. This advisory method does not emit explicit rejection decisions,
so the losses cannot all be called deliberate VLM rejections. The explicit
review variant is designed to resolve that uncertainty.

After reconciling the failed stream's dated model ID, the advisory requests have
$0.1269804532 fully accounted actual billing, with no active or unresolved
reservation. `advisory-pilot-report-v2.json` and the matching analysis/confirmed
reports include this correction. `advisory-billing-refresh.json` verifies that
all accuracy, coverage, configuration and runtime fields are unchanged; the
original report remains an archived pre-reconciliation billing snapshot.
Recorded inference time is 2009.80 seconds, excluding process startup overhead.
Three lowest-F1 detail sheets were visually inspected: they show near-aligned
wrong-class boxes, text mistaken for symbols, and partial glyph boxes.
`advisory-pilot-visual-review.json` records the observations and figure provenance.
The annotated figures remain evaluation-only.

## Completed explicit-decision comparison

`explicit-pilot-v1` finished all 400 tile attempts over 16 drawings: 397 successful,
three failed (retained in every denominator), 13 complete cases and three partial.
`explicit-pilot-report.json` records 586 TP, 370 FP, 499 FN, precision 0.612971,
recall 0.540092, micro F1 0.574228 and macro drawing F1 **0.435388**. The baseline
macro was 0.236081 and advisory macro 0.415229. Localization matches 819/1085;
conditional classification is 586/819 (0.715507). Dataset PID micro F1 is 0.650746;
PID2Graph Synthetic micro F1 is only 0.224044. No finalist is selected until all
four registered algorithms finish.

All 956 predictions qualify as explicit VLM-confirmed observations under the
secondary metric; none is automatically graph-eligible. The full reconciled cost
is **$0.2468214672**, including 403 settled requests and retries, with no remaining
reservation or unknown charge for this run. Recorded runtime is 4804.29 seconds.
`billing-snapshot-002.json` fixes the accounting evidence used for scoring.

`explicit-pilot-retention.json` finds 584/1074 correct detector truth matches
retained, 490 lost (232 still localized with a wrong class, 258 lacking a matched
localization), and two new matches. `explicit-pilot-decision-audit.json` shows all
1747 detector proposals were requested: 1120 recognized in at least one view,
956 retained in final output, 164 recognized without retained output, 181 only
rejected, and 446 unresolved without recognition. Recognition/rejection conflicts
occur for 185 proposals across views/attempts; these overlap the recognized group.
There were 26 requested proposal-view instances with no response, zero omitted by
the guidance limit, and one unrecognized output row.

The reporting-only join in `eval/pid2graph/decision_outcomes.py` connects these
decisions to frozen detector matching, with drawing-scoped IDs and exact run,
panel, detector and report checks. Five new tests plus four existing audit and
retention tests pass; Ruff passes. `explicit-pilot-decision-outcomes.json` partitions
the 490 lost detector truth matches:

| Proposal decision/output group | Correct detector matches lost | Still localized |
| --- | ---: | ---: |
| Only rejected | 36 | 0 |
| Unresolved without recognition | 111 | 1 |
| Recognized but absent from final output | 106 | 3 |
| Retained in final output | 237 | 228 |

The rejected group also contains 145 detector false positives. Thus rejection
does remove incorrect detections, but it is not the main explanation for recall
loss. Class errors on retained observations and losses after recognition are
substantial. These are associations under the frozen match assignment, not proof
of causal mechanisms; duplicate proposals can represent the same physical symbol.
The source-crop and broad-category trials address two relevant hypotheses and
continue under their existing registrations without mid-run changes.

`explicit-pilot-ownership-audit.json` further identifies the deterministic
mechanism for the 164 recognized proposals absent from output. A geometry-only
replay of all 400 tiles and 1758 final-batch raster observations exactly reproduces
the saved retained proposal IDs on every tile. Each of those 164 missing IDs is
present in at least one parsed final batch, but every such observation's projected
center lies outside that observing tile's ownership core. This group includes
107 correct detector matches, of which 106 are lost in hybrid scoring. These are
ownership-filter losses rather than parse rejection or loss in the report writer.
The replay does not establish why the owning view omitted a usable recognition,
and removing ownership could introduce duplicates and partial glyphs. No inference
or scoring method is changed. The audit uses validation images and archived
responses only, with zero API calls and no GraphML reads; Ruff passes.

```sh
PYTHONPATH=src .venv/bin/python -m eval.pid2graph.ownership_audit --panel output/pid2graph-phase2/panel-pilot.json --run output/pid2graph-phase2/explicit-pilot-v1 --audit output/pid2graph-phase2/explicit-pilot-decision-audit.json --out output/pid2graph-phase2/explicit-pilot-ownership-audit-new.json
```

Three low-F1 failure sheets were inspected (Synthetic 132, 10, 192). They show
near-aligned wrong-class boxes, partial/duplicate boxes on whole symbols, and
whole symbols with no matched output. `explicit-pilot-visual-review.json` records
observations and image hashes. No claim about prevalence follows from three
sheets, and annotated images remain scoring-only inputs.

```sh
PYTHONPATH=src .venv/bin/python -m eval.pid2graph.decision_outcomes --detector output/pid2graph-phase2/detector-epochs-v1/validation-epoch-000.json --hybrid output/pid2graph-phase2/explicit-pilot-report.json --audit output/pid2graph-phase2/explicit-pilot-decision-audit.json --run output/pid2graph-phase2/explicit-pilot-v1 --out output/pid2graph-phase2/explicit-pilot-decision-outcomes-new.json
```

## Completed source-crop comparison

`detail-pilot-v1` completed all 400 tile attempts, with 395 successes and five
failed tiles across five drawings. Full-inventory scoring yields 678 TP, 621 FP,
407 FN, precision 0.521940, recall **0.624885**, micro F1 0.568792 and macro drawing
F1 **0.426569**. Compared with explicit review, recall rises from 0.540092 but
macro F1 falls from 0.435388. Localization recall rises to 0.897696 (974 matches),
while conditional classification is 676/974 (0.694045). Dataset PID micro F1 is
0.663522; Synthetic micro F1 is 0.189076. The analysis is paired against explicit
review on the same 16 drawings, not a significance claim or final selection.

`detail-pilot-retention.json` records 677/1074 correct detector matches retained
(0.630354), 397 lost, and one new match. Of the losses, 294 still have a localization
match but wrong class, and 103 lack a localization match. Per-proposal decisions
show 1425 recognized IDs, 1298 retained IDs, 127 recognized without retained output,
102 rejected-only and 220 unresolved. There is also one discovery without a source
detector ID. All 1299 outputs qualify as VLM-confirmed observations in the secondary
metric, but none is automatically graph-eligible.

The decision/outcome join partitions the 397 lost detector truth matches into
10 rejected-only, 43 unresolved, 48 recognized without output, and 296 retained
with incorrect final matching (287 still localized). Geometry replay verified
all 400 tiles and 2452 raster objects: all 127 recognized IDs absent from output
were filtered by tile ownership, as in explicit review. Crops reduce this loss
and increase recognition, but substantial broad-class errors remain. The largest
conditional confusions are general→valve (109) and general→instrumentation (100).

The run has **$0.2068613288 known actual charges**, 403 settled requests, and two
unresolved requests without generation IDs whose combined maximum exposure is
$0.46929344. Thus $0.6761547688 is budget exposure, not an actual bill. Recorded
runtime is 5797.70 seconds. `billing-snapshot-004.json` fixes this report's ledger
evidence; unknown charges must not be presented as zero or used as a cheap-cost
tie breaker. The latest reconciliation separately settled two other captured
generation IDs for $0.000893424 (`billing-reconciliation-005.json`).

Failure sheets for Synthetic 53, 33 and 132 were visually inspected. They show
whole-glyph boxes with wrong broad classes and partial boxes coexisting with
whole boxes; a missed pump lacks a matched output. Detailed observations and
image hashes are in `detail-pilot-visual-review.json`. These selected examples
do not establish prevalence. No fifth hybrid variant or mid-run method change
is introduced, and the broad-category pilot remains required before shortlisting.

## Completed broad-category pilot and algorithm shortlist

`broad-pilot-v1` finished all 400 tile attempts, with 395 successful tiles and five
failed tiles across five drawings. It records 812 TP, 474 FP, 273 FN, precision
0.631415, recall **0.748387**, micro F1 0.684943 and macro drawing F1 **0.566602**.
Localization recall is 0.914286 (992 matches); conditional classification is
811/992 (0.817540). Dataset PID micro F1 is 0.761805 and Synthetic micro F1 is
0.369892. Both collections improve descriptively over the other VLM pilots,
although broad-category performance still trails the original detector alone.
The model finds 15/18 arrows correctly, while the detailed-schema methods find
none under the frozen broad-class mapping. This is not proof of detailed legend
semantics or real-PDF accuracy.

Retention is 810/1074 correct detector matches (0.754190), with 264 lost and two
new matches, including one corrected detector class. Of those 264 losses, 179
remain localized with a wrong class and 85 lack a matched localization. Explicit
decision records contain 1283 recognized IDs, 1158 retained IDs, 125 recognized
without output, 358 rejected-only and 106 unresolved. The 358 rejected-only
proposals include 338 detector false positives and 20 correct detector matches.
The decision/outcome join attributes 20 lost truth matches to rejected-only IDs,
18 to unresolved IDs, 54 to recognized-without-output IDs, and 172 to retained IDs
(167 still localized). The ownership replay for detailed-schema methods was not
applied to this different object schema; no equivalent filtering proof is claimed.

There are 128 additional discoveries without detector IDs. All 1286 predictions
qualify as VLM-confirmed broad observations under the secondary metric, while
engineering review and legend interpretation remain separate. Known actual
charges total **$0.2479034164**, with 403 settled requests and two unresolved
no-generation-ID requests bounded at $0.46929344 combined. Runtime is 5764.55
seconds. `billing-snapshot-005.json` anchors this score. Three low-F1 failure
sheets (Synthetic 132, 10, 33) show partial compound-glyph boxes, near-aligned
wrong-class boxes, and missing small symbols. Image hashes and observations are
recorded in `broad-pilot-visual-review.json`.

All four completed pilots are now ranked under the frozen promotion policy:

| Pilot method | Macro drawing F1 | Recall | TP | FP | Failed tiles |
| --- | ---: | ---: | ---: | ---: | ---: |
| Broad-category recognition | 0.566602 | 0.748387 | 812 | 474 | 5 |
| Explicit proposal review | 0.435388 | 0.540092 | 586 | 370 | 3 |
| Proposal source crops | 0.426569 | 0.624885 | 678 | 621 | 5 |
| Advisory guidance | 0.415229 | 0.500461 | 543 | 401 | 1 |
| Baseline reference | 0.236081 | 0.221198 | 240 | 588 | 1 |

`pilot-shortlist.json` seals broad-category recognition and explicit proposal
review as the two algorithms for trained-checkpoint comparison. The reproducible
`pilot_shortlist` command verifies all four registrations and current source
fingerprints, full attempted coverage, drawing-score aggregates, and billing
against the same immutable snapshot. All four reports' existing billing sections
match that snapshot exactly; no rescoring is necessary. Ruff passes. This record
does not authorize test inference or select final weights. Training convergence,
both trained-weight comparisons, broader validation and final evaluation remain
required under the original policy.

```sh
PYTHONPATH=src .venv/bin/python -m eval.pid2graph.pilot_shortlist --study output/pid2graph-phase2 --ledger output/pid2graph-phase2/billing-snapshot-005.json --candidate output/pid2graph-phase2/advisory-pilot-report-v2.json output/pid2graph-phase2/advisory-pilot-v1 --candidate output/pid2graph-phase2/explicit-pilot-report.json output/pid2graph-phase2/explicit-pilot-v1 --candidate output/pid2graph-phase2/detail-pilot-report.json output/pid2graph-phase2/detail-pilot-v1 --candidate output/pid2graph-phase2/broad-pilot-report.json output/pid2graph-phase2/broad-pilot-v1 --out output/pid2graph-phase2/pilot-shortlist-new.json
```

## Broader validation reuse and confirmation metrics

`eval/pid2graph/reuse_validation.py` rebases finished identical pilot validation
records into the broader validation configuration without new API requests.
It verifies panel/manifest seals, original validation membership, source hashes,
registered method hashes, complete attempted tile coverage, summary consistency,
and billing IDs. Failed attempts remain failed; test and stopped runs are refused.
For hybrid reuse, both proposal runs must also contain identical common-drawing
predictions under the same perception configuration. Eight focused tests pass.

`baseline-broad-v1` was seeded with all 16 pilot drawings and 416 copied records;
paid inference on the remaining 91 drawings is now running. The recorded zero-call dry run confirmed that the
normal runner accepts the expanded configuration. `reuse-verification.json`
checks prediction/error/usage/request-ID preservation. Reused request IDs are
the same study charges, never additional spending. The ongoing broader run uses
the ordinary frozen baseline driver and shared locked spending ledger.

`detector-pilot-broad-v1` supplies the broader detector-only reference using the
original pilot weights and unchanged proposal settings. It was registered before
new inference in `detector-pilot-broad-registration.json`. All 16 existing source
predictions were reused exactly after checking original validation membership,
image hashes, model identity and detector settings; only their panel/configuration
binding changes, with per-record source provenance retained. The ordinary detector
driver finished the remaining 91 drawings locally. This used no API calls or test
inputs. All 107 drawings were scored after inference finished, against an immutable
billing snapshot (the detector itself has no API charges).
Any VLM reuse still requires the separate exact-proposal checks. Runtime reports
retain observed durations, including differences in local compute contention;
they are not isolated hardware benchmarks.

`detector-pilot-broad-report.json` records 13,406 predictions against 9,259 physical
symbols: TP 9,200, FP 4,206, FN 59; precision 0.686260, recall 0.993628, micro F1
0.811824, and macro drawing F1 0.737536. Every drawing is complete.
`detector-pilot-broad-verification.json` confirms that the guidance loader accepts
all 107 outputs and the original 16 pilot predictions remain exactly identical.
The second collection is much harder for precision: Dataset PID micro F1 is
0.861956 (precision 0.759177, recall 0.996921), while PID2Graph Synthetic is
0.499522 (precision 0.337646, recall 0.959559). These are broader validation
diagnostics for the original weights, not selected trained-checkpoint results.

`detector-pilot-broad-analysis.json` includes all classes and localization
confusions. Conditional classification is 9,125/9,218 (98.99%) under the separate
class-agnostic assignment; duplicate predictions mean this numerator differs
from class-aware TP. Three low-F1 sheets show pipe-junction false positives,
partial/nested boxes alongside whole-glyph matches, and overlapping class
hypotheses. `detector-pilot-broad-visual-review.json` records the inspected examples
without asserting that they establish frequency or crop-level causality. Frozen
training/proposal settings are unchanged. Annotated overlays remain evaluation-only.

`eval/pid2graph/confirmed_metrics.py` reports a secondary inventory of accepted
VLM observations. Detector confidence alone never qualifies; explicit accepted
raster decisions and valid broad discoveries do. This filter is reporting-only,
not another candidate pipeline. Full truth denominators and failures are retained.
Two focused tests pass. All baseline and advisory predictions qualify under the
existing parser, so their confirmed reports equal their primary symbol metrics.
This does not confer engineering approval.

## Checkpoint export for CLI integration

`eval/pid2graph/checkpoint_export.py` creates a separate compatible checkpoint
from weights with a completed matching validation report. The epoch trainer's
missing RPN threshold can only be recovered after checking its recorded detector
source hash and inspecting the reconstructed builder's actual threshold. Smoke
checkpoints, changed classes/transforms, mismatched reports, and incomplete or
test evaluations are rejected. The running trainer and frozen inference methods
are unchanged. Seven focused tests pass.

An actual export of the pilot weights preserved all 284 model tensors exactly.
The production loader produced exactly equal CPU forward predictions on source
validation crops from both collections, with no GraphML or API calls. Evidence
is in `checkpoint-export-pilot-verification/verification.json`.

The subsequent full-drawing MPS check reproduced every one of the 1,747 reference
proposals exactly on all 16 validation drawings, including boxes, classes,
confidence, IDs and review disposition. `checkpoint-export-full-parity/report.json`
records every drawing and prediction hash. `detector_parity.py` now accepts an
export only with explicit source/export checkpoint bindings and unchanged
exporter/builder/production-loader identities. Seven focused provenance checks
pass. There were no API calls, GraphML reads or test inputs. This verifies the
pilot export path, not final model selection; the chosen trained checkpoint
still requires its own export and full-drawing parity check.

Reproduce that check with a fresh output directory:

```sh
PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-cv/bin/python -m eval.pid2graph.detector_parity --panel output/pid2graph-phase2/panel-pilot.json --reference output/pid2graph-phase2/detector-epochs-v1/validation-epoch-000 --checkpoint output/pid2graph-phase2/checkpoint-export-pilot-verification/detector.pt --export-verification output/pid2graph-phase2/checkpoint-export-pilot-verification/verification.json --out output/pid2graph-phase2/checkpoint-export-full-parity-new
```

## Completed current baseline

`baseline-pilot-v1` finished all 400 tile attempts over the 16 frozen validation
drawings; 399 succeeded and one transport failure remains in the denominators.
`baseline-pilot-report.json` records TP 240, FP 588, FN 845, precision 0.289855,
recall 0.221198, micro F1 0.250915, and macro drawing F1 0.236081. Class-agnostic
localization matched 390 objects, with 240/390 correctly classified. The two
collection micro F1 values are 0.256927 (Dataset PID) and 0.221538 (PID2Graph
Synthetic). These are a fresh stochastic reference run, not a claimed improvement
from a changed method. Detector-assisted accuracy comparisons are still running.

Known billed baseline usage is $0.117774804. One failed attempt lacks a generation
ID and has unresolved maximum exposure $0.23464672. This exposure is not an actual
charge and has not been released. Recorded inference time is 1788.76 seconds;
the existing runtime-scope limitations apply. `baseline-pilot-analysis-v2.json`
preserves the separate billed/reserved/unresolved fields; its statistics match
the earlier analysis exactly. `baseline-pilot-retention.json` finds that 239 of
1,074 detector truth matches were also correctly found by this unassisted VLM:
689 lost localization matches and 146 localized-but-misclassified matches.

Three low-F1 detail sheets were inspected and recorded in
`baseline-pilot-visual-review.json`: examples include closely aligned boxes with
wrong broad classes, absent small paired-line glyphs, partial glyph boxes, and
empty/text-adjacent predictions. These annotated figures are evaluation-only;
they must never be passed to validation or test inference.

`eval/pid2graph/proposal_retention.py` compares completed attempted coverage on
identical panels, separating shared class-aware truth matches, localization
loss, class errors, and recoveries. It rejects unfinished drawing coverage and
keeps API failures in the denominator. Two tests exercise those distinctions
and repeated truth IDs across different drawings. Applied to the archived
unassisted baseline (`historical-baseline-retention.json`), 219 of 1,074 detector
true positives were also correctly found by the VLM. Of the 855 not found
correctly, 736 had no VLM localization match and 119 had a localization match
without a class-aware match. The VLM recovered one additional truth symbol.
Eleven drawings had API failures. These are post-inference associations, not
evidence that the unassisted baseline received or explicitly rejected detector
proposals. They prioritize source coverage and localization in the hybrids.

Source-only detector proposals and uncertain review proposals must remain
distinguishable from VLM-confirmed observations and engineering approvals.
Report accepted-only metrics alongside the original review-inventory metrics;
retaining unverified detector output must not be claimed as VLM confirmation.
GraphML does not establish detailed legend semantics or real-PDF accuracy.

Completion requires finished required comparisons, an untouched final evaluation,
and an evidence-backed integration decision. A measured negative result is valid;
an interrupted required experiment remains incomplete. Persist bounded jobs and
checkpoints, use efficient waits, and avoid redundant polling/audits. Use local
compute; no cloud GPU rental. The Inspector and port 8000 remain retired.

## Current reproducibility commands

The resumable epoch trainer is implemented in `eval/pid2graph/epoch_train.py`.
It saves model, optimizer momentum, Python/CPU/MPS random states, sample cursor,
global learning-rate step, and validation history. Every epoch includes all
19,543 base crops plus the frozen background/rare-class repeats, for 22,259
presentations. The final short batch is retained. A process lock prevents two
writers; source/configuration hashes prevent silently changing a resumed run.
SIGTERM/SIGINT request a checkpoint after the current batch. Abrupt termination
can lose work since the latest 500-step checkpoint; loss logs are reconciled to
that checkpoint on resume. Training continues the pilot weights with a fresh
optimizer and all parameters of the reconstructed FrozenBatchNorm architecture
trainable. MPS bitwise determinism across processes is not claimed.

Three sampling/selection/schedule tests pass. A real local MPS smoke ran two
batches, saved state, and resumed through two more batches with the expected
16 presentations and 100 optimizer momentum states. Its weights are excluded
from model selection (`trainer-resume-smoke/verification.json`). The study run
`detector-epochs-v1` has completed epoch-zero validation: macro drawing F1
0.6630185360425482, reproducing the archived pilot result. Epoch one then improved
macro F1 to 0.7761567011290226: 1082 TP, 378 FP, three FN, precision 0.741096,
recall 0.997235 and micro F1 0.850295. Dataset PID micro F1 is 0.904478 and
Synthetic micro F1 0.646729. `detector-epoch-001-analysis.json` records per-class,
per-collection and paired results. Epoch two is running; convergence and final
checkpoint selection remain unproven. The checkpoint selection includes epoch zero as
a fallback, uses the earliest epoch on exact metric ties, and evaluates only
the frozen pilot validation panel after each full epoch.

Start or resume only after confirming the previous process is terminal:

```sh
PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-cv/bin/python -m eval.pid2graph.epoch_train --training output/pid2graph-phase2/training-data/training.json --schedule output/pid2graph-phase2/training-schedule-v1.json --manifest output/pid2graph-development/manifest-v1.json --out output/pid2graph-phase2/detector-epochs-v1
```

`prices-v1.json` records fresh endpoint capabilities and the same provider
allowlist/server-enforced price ceilings as the prior study. The single
training-crop billing smoke returned completed `usage.cost` of $0.00014966,
captured as actual billed cost with no outstanding reservation. The independent
generation lookup returned 404; it does not corroborate the amount or make the
request free. See `billing-smoke-v1/result.json` and
`billing-smoke-v1/generation-verification.json`. The $0.64 earlier-phase amount
remains user-reported and separate. Prices expire after 24 hours and must be
reverified for later requests.

The new 16-drawing VLM baseline is registered in
`baseline-pilot-registration.json` and finished in `baseline-pilot-v1`, with
120-second request allowances and four new tiles per process. Failed attempts
remain in the evaluation denominators. Synthetic drawings supply no detailed
legend mapping; their empty legend context is explicit, not inferred from
GraphML. Active and terminal job handles are recorded separately in
`output/pid2graph-phase2/checkpoint.json`; a status file alone is not proof that
a process is still alive.

Bootstrap command (requires a new output directory):

```sh
.venv/bin/python -m eval.pid2graph.study_v2 --previous output/pid2graph-development --out output/pid2graph-phase2
```

The bootstrap code and generated panel/coverage assertions passed Ruff and an
offline artifact consistency check. Paid phase-2 inference and the full epoch
trainer subsequently started under the frozen configuration.

## Training preparation and accounting implementation

`training-plan.json` and `training-data/training.json` now contain 19,543 rendered
crops from the 528 training drawings. All 45,021 physical symbols have a fully
contained crop, and 315 crops provide annotation-free pipe/background negatives.
All crop hashes and box bounds were checked; minimum retained box side is 11
pixels. Three source/box examples were visually inspected in `training-qa.png`.
That annotated QA image is not a training or inference input. The training seal
is `8e7c3d6aeb6e17116b996a7fdb1ba7e6b7c8512f015d38530c61d212c588f9fd`.

Reproduction commands:

```sh
.venv/bin/python -m eval.pid2graph.coverage_data plan --source output/pid2graph-development/manifest-v1.json --out output/new-training-plan.json
.venv/bin/python -m eval.pid2graph.coverage_data render --source output/new-training-plan.json --out output/new-training-data
```

`training-schedule-v1.json` was frozen before the full run. Two short local MPS
benchmarks measured 1.52 images/second at batch two and 2.19 at batch four after
startup. The selected batch size is four. Each epoch visits every base crop once,
then adds seeded background and rare-class draws, for 22,259 presentations.
The schedule allows six epochs, requires at least two, validates after each epoch,
and stops after two epochs without a 0.002 macro-F1 improvement over the patience
reference. Best-checkpoint selection still uses the highest validation macro F1.
Continue the delivered pilot model weights; reset the optimizer under the recorded
schedule. The short benchmark weights are not accuracy candidates.

Estimated training time is 2.82 hours per epoch and 16.93 hours at the maximum,
excluding validation, checkpoints and runtime variation. The resumable epoch
driver is now running under this schedule. No cloud compute is used.

`eval/pid2graph/training_report.py` reconstructs actual committed exposure from
the frozen sampler, saved epoch and cursor. It distinguishes crop presentations,
unique crops, drawing groups, repeated annotation occurrences, and unique physical
symbols. Physical IDs are scoped by drawing and deduplicated across overlapping
crops. The reporter checks the source/configuration hashes, archived validation
report/checkpoint identities, chronological epochs, selection rule and (when
finished) the minimum/maximum epoch and early-stopping conditions. It reads no
test inputs or GraphML. Seven focused checks cover overlap/repeat accounting,
same IDs on different drawings, the short final batch, invalid training positions,
wrong splits, and inconsistent physical inventories.

`training-exposure-snapshot-001.json` describes committed step 3,000, at 12,000
presentations: 10,864 unique crops, all 528 drawings/groups, 1,036 background
presentations, and 27,884/45,021 physical symbols fully contained at least once
(61.94%). This compares with 5,275 fully visible symbols in the prior pilot.
Per-class and per-drawing coverage are included. This is a historical partial
snapshot. `training-exposure-snapshot-002.json` now verifies a committed full epoch:
22,259 presentations, all 19,543 base crops, all 528 drawings/groups, 1,955 background
presentations, and all 45,021 physical symbols fully visible at least once.
All six present classes have complete unique physical-symbol exposure. This
establishes actual coverage, not convergence or generalization to real PDFs.

Generate an immutable progress or final convergence report with a new output path:

```sh
PYTHONPATH=src .venv/bin/python -m eval.pid2graph.training_report --training output/pid2graph-phase2/training-data/training.json --schedule output/pid2graph-phase2/training-schedule-v1.json --run output/pid2graph-phase2/detector-epochs-v1 --out output/pid2graph-phase2/training-report-next.json
```

Version-2 billing is implemented in `diagex.llm.billing`, selected only by a fresh
version-2 ledger. `spending.json` starts with the user-reported historical $0.64
and zero new calls. Known actual costs, active reservations and unresolved bounds
have separate fields. The client retains stream generation IDs even without UI
callbacks; completed response costs or matching generation metadata establish
actual billing. Token-price arithmetic remains an upper bound, not a bill.
Missing generation records and aggregate key snapshots do not make failed calls
free. Pre-dispatch exposure checks still enforce the cumulative $60 limit.

The legacy `charged_usd` field remains an exposure alias for old consumers. New
score reports have a distinct `billing` section; phase-2 spending reports must use
that section rather than interpret the legacy field as actual spending.

A later shared stream interruption affected one tile in each of the four active
API runs. The existing connection reset and cooldown recovered; every driver
subsequently recorded successful new tiles. Failed tile records remain immutable
and in the score denominators. Six captured generation IDs were queried in
`billing-reconciliation-001.json`; all triggered reconciliation validation errors,
so their unresolved charges were retained pending inspection of the returned
metadata. One older unresolved attempt has no generation ID.
No unknown charge was marked free, and no inference was repeated for this audit.

Inspection showed HTTP 200 responses whose model ID was
`deepseek/deepseek-v4.1-flash-20260910`, while the request used the undated alias.
The provider endpoint snapshots already stored with each reservation explicitly
bind that dated ID to the requested alias and provider. Reconciliation now accepts
only that exact archived mapping, with matching generation ID, provider, allowed
provider tag, and verification time; it never strips date suffixes heuristically
or relies on a possibly changed current alias. The original billed model and
mapping evidence are retained in each ledger record.

`billing-reconciliation-002.json` records six successful reconciliations totaling
$0.003667142, replacing $1.40788032 in maximum reservations. The one older no-ID
request remains unresolved at an upper bound of $0.23464672. Active requests keep
their reservations. The accounting/connection suite passed 32 tests, including
wrong dated model/provider/generation rejection and direct active-attempt
protection. Historical score reports retain their original billing snapshots;
completed runs must be rescored against the current ledger before selection and
final cost reporting. Their predictions, failures and accuracy metrics do not
change. Lookup logs now preserve the HTTP status separately from any validation
error so an HTTP 200 identity mismatch cannot be mistaken for a 404.

The reconciliation command now skips active streams even when their generation
ID is already known, preserving their reservation until the owner finishes.
Twelve billing tests pass, including a check that an active ID is not queried
for settlement and becomes eligible only after the attempt ends. The ledger's
historical `unresolved_request_count` summary field includes all requests lacking
actual billing, including active ones; the reconciliation artifact separately
reports active and finished-unresolved counts to avoid confusing those states.

The targeted suite passed 79 cases; adding the new version-2 scoring case then
passed both scoring cases (80 distinct cases across the resulting suite). A local
SDK check verified that parsed stream events retain the OpenRouter cost extension.
The new reconciliation command also completed a read-only live key-usage lookup.
`accounting-implementation-check.json` records scope and source hashes. A paid
training-split smoke subsequently verified live cost/generation capture under
the refreshed model prices and provider policy, as recorded above.

```sh
.venv/bin/python -m eval.pid2graph.billing_reconcile summary --ledger output/pid2graph-phase2/spending.json
```

Billing references consulted:

- https://openrouter.ai/docs/api/api-reference/api-keys/get-current-key
- https://openrouter.ai/docs/api/api-reference/generations/get-request-&-usage-metadata-for-a-generation
- https://openrouter.ai/docs/cookbook/administration/usage-accounting

The PDF, printed-legend and review-to-graph check scripts now use
`eval/pid2graph/transfer_accounting.py` to report actual billed amounts, active
reservations and finished unresolved upper bounds separately, with request IDs
and a ledger snapshot hash. Historical legacy-ledger runs remain explicitly
exposure-only. Four focused tests cover unknown amounts, request scoping and
invalid IDs; all three callers import and Ruff passes. An offline check on the
403 completed explicit-pilot requests exactly reproduces its authoritative
$0.2468214672 bill (`transfer-accounting-check.json`). No PDF inference was rerun
for this accounting change; selected-method transfer and integration remain due.

Remaining milestones are multi-epoch convergence, the other registered hybrid
comparisons, broader validation, a frozen final selection/test, and supported
CLI/workbench integration with representative PDF transfer checks.
# Broad observation review compatibility

## Automatic trained-checkpoint comparisons

`eval/pid2graph/checkpoint_comparison.py` now waits on the active trainer's OS
lock, then verifies completed training exposure, validation history and the
frozen stopping rule. It selects the best post-pilot epoch by detector macro
drawing F1, earliest epoch on exact ties, as required by the existing promotion
policy. It registers exact replacement configurations and proposal hashes for
the two shortlisted algorithms before any paid inference. Only detector weights
and resulting hints change. The frozen runners validate their prewritten
configurations, share the existing cumulative spending ledger, preserve failed
attempts, and write separate billing snapshots and validation reports.

The waiting/running job is session **31905**, output
`checkpoint-comparison-v1`; its command and six successful software tests are
recorded in `checkpoint-comparison-launch.json`. An actual premature preparation
attempt against the currently incomplete trainer was rejected before creating
any registration or API request. An interrupted trainer or failed comparison
stops this controller with an explicit failed status; it does not call an
unfinished experiment complete or move to the test panel. Do not start another
copy or edit its loaded sources while it is active. Resume the same command
only after confirming its prior session is terminal.

This automation ends after the two trained-weight pilot comparisons. Billing
reconciliation, error analysis, per-method checkpoint choice, broader finalist
evaluation, final selection and the untouched test evaluation still remain.

The broad-recognition pilot returns `raster_symbol` observations rather than
engineering objects. The existing review initializer attempted to parse every
proposal as `PerceivedObject`, which rejected these observations. The review
store now preserves them as pending, explicitly unclassified items with their
original source observation. The workbench keeps an unclassified selection
instead of silently defaulting to equipment. Confirmation requires a compatible
engineering class and the existing legend-review gate. Pending items stay out
of graph inputs; flow arrows can be excluded while retaining their original
evidence. Existing native and typed detection behavior remains covered.

`review-integration-check-001.json` records 21 passing unit tests, Ruff and
JavaScript syntax checks, and the synthetic browser workflow. One existing HTTP
test required a rerun with permission to bind a temporary loopback port. The
browser check used a temporary port and shut down; port 8000 stayed retired.
The browser graph-build runner captures review inputs, so this is software
workflow evidence, not a new end-to-end extraction accuracy result.

`broad-review-inventory-check-v1/verification.json` replays all 400 saved pilot
tile attempts through the actual review store: 1,286 observations from 16
validation drawings, all preserved as pending, none graph-eligible. The five
failed tile attempts remain represented in page coverage. No GraphML or API
calls were used, and frozen inference source hashes were checked unchanged.
Reproduce with `python -m eval.pid2graph.review_inventory --panel
output/pid2graph-phase2/panel-pilot.json --run
output/pid2graph-phase2/broad-pilot-v1 --out <new-output-directory>` under
`PYTHONPATH=src` in `.venv`.

This is preparation for integration, not a final method selection. The opt-in
semantic stage described below is implemented and software-tested; selected
checkpoint verification and selected-method PDF/graph transfer checks remain
incomplete. Training and broader baseline evaluation continue independently.

## Opt-in legend interpretation for broad observations

`vision/raster_semantics.py` sends the original source view, immutable symbol
IDs/bounds, bounded prepared legend entries and reference thumbnails through
the existing metered client. Replies may suggest engineering classes with
known compatible legend references, mark non-node evidence, or remain
unresolved. Geometry/native-ID injection, missing or duplicate decisions,
unsupported legend references and contradictory kinds cannot create accepted
semantic suggestions. Provider failures preserve recognized symbols and
completed interpretation chunks, flag unfinished interpretation, and stop
further semantic chunks in that view. Empty applicable legend context makes no
semantic API call. Every result remains pending for normal review.

The production route is explicitly opt-in:

```sh
.venv/bin/diagex extract-pid drawing.png --engine evidence-v2 \
  --raster-proposals proposals.json --raster-symbol-mode broad_review
```

It accepts the source-bound single-image proposal format and the new separately
validated PDF-page format (`vision/pdf_raster_guidance.py`),
requires persistence and fixed inspection without deskew, and stops at
detection review before building a graph. Reviewed inputs can subsequently
build through the existing workbench. Baseline is the default; native
candidates use their existing perception route. This does not yet provide
selected-checkpoint PDF proposal generation or a final deployment decision.
The PDF adapter preserves native evidence and binds hints to exact rendered
pixels; omitted pages retain baseline perception. Its 78-test integration check
and reduced-resolution CPU smoke are recorded in `pdf-guidance-integration-check.json`.
Selected-weight, production-resolution PDF evaluation is still required.
The frozen four-hybrid benchmark methods and their primary symbol metrics are
unchanged: interpretation adds review suggestions without changing their broad
inventory or geometry.

`raster-semantics-integration-check.json` records 61 passing software tests
(60 in the combined run plus the corrected public-workflow fixture), browser
verification and frozen-source fingerprints. Checks cover the actual production
tile route, CLI opt-in/cache identity, bounded failures and missing legends,
native fallback, review gating and reviewed-instance graph construction. Model
responses and drawing fixtures are controlled; this is not a measured claim
about detailed semantic accuracy or real-PDF generalization. No paid calls were
made by these integration checks; training and broader baseline continue in
their existing sessions. Full selected-method validation remains required.

### Live raster-PDF integration check

`eval.pid2graph.raster_pdf_check` runs a bounded, one-page PDF through the real
evidence extraction entry point, shared billing ledger, broad recognition,
legend-conditioned semantic suggestions, and persisted symbol review. It uses
built-in ISA definitions when no project legend is supplied, and stops before
graph construction. It does not read GraphML or the final test panel.

The first check (`raster-pdf-live-smoke-v1/report.json`) exposed a routing bug:
the sparse-first-page rule classified the scanned `two-tanks.pdf` as a cover,
and skipped all symbol calls. The check correctly reported incomplete and cost
zero. Source-bound PDF proposal pages now fail open to raster inspection when
they are scanned, have no native text or paths, and were weakly classified as
covers. Explicit whole-page legends, positive classifications, unselected pages,
and native evidence are preserved. Changed routing metadata and its prior reason
are recorded in `pdf-proposal-routing.json`; it is not an engineering assertion
that the page is a P&ID.

The fresh second check (`raster-pdf-live-smoke-v2/report.json`) completed one
tile with 42 pending observations: 29 interpreted suggestions, six non-node
suggestions, and seven unresolved. Three settled API calls cost $0.0067699968.
No symbols became approved graph nodes. This used old pilot weights, a reduced
1500-pixel rendering and built-in definitions; it is an interface check, not
selected-weight transfer accuracy. Visual inspection also found unsupported
text and subtype claims and misplaced boxes; see its `visual-review.json`.

The detection-stage progress and cost summary now include both broad recognition
and semantic tool calls instead of counting only the baseline tool name. The
82-test raster/PDF/review regression suite passed after this reporting fix.
The live check above preceded the reporting-only change; no extra paid rerun was
needed. Frozen training, benchmark inference and comparison-controller sources
remain unchanged.

Reproduce with a **new** output directory (existing live checks are immutable):

```sh
PYTHONPATH=src .venv/bin/python -m eval.pid2graph.raster_pdf_check \
  --source tests/p-ids-public/two-tanks.pdf \
  --proposals output/pid2graph-phase2/pdf-proposal-smoke-v1/proposals.json \
  --out output/pid2graph-phase2/NEW-raster-pdf-check \
  --ledger output/pid2graph-phase2/spending.json \
  --prices output/pid2graph-phase2/prices-v1.json
```

### Second completed detector epoch

`training-exposure-snapshot-003.json` verifies two complete sampling epochs:
44,518 committed crop presentations, all 19,543 prepared crops and 528 training
drawings, 3,910 background presentations, and full coverage of all 45,021 physical
training symbols. Repeated/overlapping annotations are not additional physical
symbols. The inlet/outlet class has no training instances.

On the unchanged 16-drawing pilot-validation panel, detector macro drawing F1
rose from 0.776157 after epoch 1 to 0.814417 after epoch 2 (pilot checkpoint:
0.663019). Epoch 2 has 1,082 true positives, 301 false positives and three misses;
epoch 1 had the same true-positive/miss totals and 378 false positives. Precision
is 0.782357 and recall 0.997235. Dataset PID macro F1 is 0.916641 and PID2Graph
Synthetic macro F1 is 0.712193, both higher than epoch 1.

| Broad class | Epoch 1 F1 | Epoch 2 F1 |
| --- | ---: | ---: |
| general | 0.784936 | 0.804878 |
| valve | 0.890424 | 0.922290 |
| pump | 0.848485 | 0.875000 |
| tank | 0.507937 | 0.627451 |
| instrumentation | 0.966408 | 0.984211 |
| arrow | 0.900000 | 0.837209 |
| inlet/outlet | unscored | unscored |

The aggregate gain does not mean every error improved: arrow false positives
increased from four to seven, and one previously found general symbol in
Dataset PID/74 was missed while a previously missed general symbol in
PID2Graph Synthetic/53 was recovered. The exact identities and paired metrics
are preserved in `detector-epoch-002-findings.json` and
`detector-epoch-002-analysis.json`.

The frozen early-stopping rule reports zero stale epochs, so training continues
into epoch 3. Coverage and multiple epochs are now demonstrated; convergence,
trained-weight hybrid comparisons, broader finalist validation and the untouched
final evaluation remain incomplete. No test data informed this checkpoint.

### Checkpoint promotion handoff

`eval.pid2graph.checkpoint_promotion` implements the next offline gate after
`checkpoint_comparison` finishes. It revalidates the completed comparison and
frozen sources, verifies every source tile and request ID, then rescores the four
relevant runs (two shortlisted methods, two checkpoints each) against one billing
snapshot. Metrics and runtime must remain identical to the original reports;
only reconciled accounting may change. Failed attempts stay in the denominators,
and active requests or missing drawings prevent promotion.

For each method, the existing frozen ranking rule chooses the stronger checkpoint
and preserves pilot-first order on exact ties. The output contains source
registrations accepted by `reuse_validation`, allowing identical pilot drawing
records and request IDs to be reused in broader validation. This creates no paid
calls, launches no inference, and grants no final-test access. Broader detector
hints must still be generated/verified before reuse and the frozen drivers run
the remaining validation drawings. Baseline remains a required comparison.

```sh
PYTHONPATH=src .venv/bin/python -m eval.pid2graph.checkpoint_promotion \
  --study output/pid2graph-phase2 \
  --comparison output/pid2graph-phase2/checkpoint-comparison-v1 \
  --manifest output/pid2graph-development/manifest-v1.json \
  --ledger output/pid2graph-phase2/spending.json \
  --out output/pid2graph-phase2/checkpoint-promotion-v1
```

Reconcile available bills before invoking this command; unknown charges remain
unknown rather than zero. Use a new output directory because promotion records
are immutable. `checkpoint-promotion-implementation-check.json` records 30 passing
tests across promotion, selection and validation reuse. A real-state check accepted
all 400 attempted tiles from each completed shortlisted pilot and correctly
refused to create a promotion while the trained comparisons were unfinished.

`eval.pid2graph.reuse_detector_validation` prepares the corresponding detector
expansion. Supply `--source-panel`, `--target-panel`, `--source-run`, `--out`,
`--checkpoint`, and `--manifest`. It accepts only completed validation records
bound to the supplied checkpoint, checks original images and proposal geometry,
and preserves exact hints and measured runtimes. The output is still incomplete:
run the existing `eval.pid2graph.detector infer` command with the same checkpoint,
device/settings and broader panel to compute its remaining drawings before using
it for VLM reuse. Choose trained weights only from the completed comparison plan.

The helper passed 37 combined detector-reuse, VLM-reuse and checkpoint-promotion
tests. `detector-reuse-verification-v1/verification.json` additionally records an
offline replay of the original pilot-weight expansion: all 1,747 proposals over
16 drawings exactly match the existing broader reference, including runtime and
source/configuration identity. The verification copy leaves 91 drawings
unattempted and makes no inference or API calls. It is not a selected trained
checkpoint or a new accuracy evaluation.

### Queued broader trained-detector evaluation

`eval.pid2graph.broader_detector` now waits on the trainer's OS lock, verifies
the completed training record, and selects the same best post-pilot checkpoint
as the trained-weight comparison controller. It uses the verified reuse helper
to preserve the 16 pilot drawings and runs the unchanged detector on the other
91 broader-validation drawings. This is a required detector reference, makes no
API calls, and does not select a VLM checkpoint or authorize the final test.

```sh
PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m eval.pid2graph.broader_detector \
  --study output/pid2graph-phase2 \
  --manifest output/pid2graph-development/manifest-v1.json \
  --python .venv-cv/bin/python --wait-for-training
```

The worker has its own exclusive lock and immutable launch, registration and
completion records. It verifies reused predictions before inference and checks
the full validation report afterward. Completed resumes do not launch inference
again; changed configuration, provenance or completed proposal files are rejected.
The implementation passed 25 combined worker, reuse and checkpoint-comparison
tests plus Ruff. A real-state check correctly refused inference before training
completion. Session 2360 was launched and reported `waiting_for_training`; do not
launch a second broader detector job. Its source and dependencies are now frozen
while it waits/runs. See `broader-detector-implementation-check.json` and the
`detector-trained-broad-v1-*` records for provenance.

### Broader baseline completed

The baseline finished all 107 validation drawings: 2,675 attempted tiles,
2,663 successful and 12 failed. Failures remain in the scoring denominator;
the report's `complete=false` describes those partial drawing failures, not
unattempted drawings. Sixteen pilot drawings were reused and 91 were newly run.
The frozen source fingerprints, exact tile identities, source/configuration
bindings and request-ID provenance passed the coverage check in
`baseline-broad-verification.json`.

`baseline-broad-report.json` records macro drawing F1 **0.236488**, symbol
precision **0.300719**, recall **0.221406**, and micro F1 **0.255039**
(2,050 TP, 4,767 FP, 7,209 FN). Collection macro F1 is 0.266885 for Dataset PID
and 0.178978 for PID2Graph Synthetic. Class-agnostic localization recall is
0.324657; classification accuracy among localized matches is 0.680639.
The most frequent confusions are general→valve (299), general→instrumentation
(218), and valve→general (175). See `baseline-broad-analysis.json`,
`baseline-broad-confirmed.json`, and `baseline-broad-findings.json`.

The baseline matches 2,043 of the old detector's 9,200 true-positive truth IDs
and seven additional truth IDs. This is output overlap, not evidence of
proposal rejection: the baseline received no detector hints. The corresponding
record is `baseline-broad-detector-overlap.json`.

Baseline known billed cost is **$0.861609996**, with **$7.27404832** held as an
unresolved upper bound across 31 requests, and no active reservations. Recorded
runtime is 21,195.23 seconds, including reused drawing runtimes; it excludes
interrupted-invocation overhead. Reconciliation 011 queried 14 new exact
generation IDs, recovered three charges totaling $0.000836052, and left eleven
404 responses unresolved. Across phase 2, known billing is $1.69862784;
including the user-reported prior $0.64, known cumulative spending is
**$2.33862784**, with a separate **$8.2126352** unresolved upper bound.
These bounds are not actual charges. The immutable billing source is
`billing-snapshot-011.json`.

Epoch 3 also completed: macro drawing F1 0.815444, 1,081 TP, 301 FP and four
FN. Its gain over epoch 2 was 0.001027, below the frozen 0.002 patience threshold;
the trainer correctly recorded one stale epoch and continued into epoch 4.
`training-exposure-snapshot-004.json` verifies three complete epochs, 66,777
crop presentations, all 19,543 crops, 528 drawings and 45,021 physical symbols.
The final training checkpoint, trained-weight hybrid comparisons, broader
hybrid finalists, fresh final test and selected integration remain incomplete.

`baseline-broad-failure-figures/index.json` and
`baseline-broad-visual-review.json` document 16 inspected examples from the two
lowest-F1 drawings in each collection among cases without failed tiles. These
deliberately selected failures are not a prevalence sample. Saved matches confirm
tank→general (IoU 0.939), pump→instrumentation (0.793), tank→valve (0.723), and
instrumentation→general (0.663). Other inspected crops show offset boxes near
text/instrument bubbles and missed small paired-line end marks annotated general.
This distinguishes class errors on localized symbols from absent or inaccurate
boxes without attributing them to API failures. The figures contain scoring-only
annotations and must never be supplied to validation or test inference. This was
agent inspection, not human review or a change to frozen experiment settings.


## Epoch 4 validation and training exposure

The frozen 16-drawing validation panel improved from macro drawing F1 0.815444 to 0.826902 (+0.011458). Eleven drawings improved, two worsened, and three tied. Correct detections stayed at 1,081 and false negatives at four; false positives decreased from 301 to 268. Micro F1 is 0.888250, precision 0.801334, and recall 0.996313. Dataset PID macro F1 is 0.929010; PID2Graph Synthetic is 0.724794.

Per-class F1: general 0.822060, valve 0.923234, pump 0.823529, tank 0.800000, instrumentation 0.981627, and arrow 0.878049. Inlet/outlet has no validation support. Pump F1 regressed from 0.903226 because false positives increased from three to six. Independent class-agnostic localization matching gives conditional classification 1,079/1,081, with two tank-to-general confusions; this matching is distinct from class-aware symbol matching. Synthetic/53 general41 was recovered, while Synthetic/33 general55 became missed. These are validation findings, not final-test estimates.

`training-exposure-snapshot-005.json` verifies four completed sampling epochs, 89,036 crop presentations, all 19,543 prepared crops and all 528 training drawings/groups, 7,820 background presentations, and full visibility of all 45,021 physical training symbols. Repeated crops are not counted as additional physical symbols. Epoch 4 improves beyond the frozen 0.002 patience threshold, becomes the best checkpoint so far, and resets stale epochs to zero. Training remains incomplete and continues under the original six-epoch maximum; hybrid checkpoint selection remains gated on completion.

Artifacts: `detector-epoch-004-analysis.json`, `detector-epoch-004-findings.json`, and `training-exposure-snapshot-005.json`. Validation report SHA-256: `fa4ff836bbe567f9a2099f58772469268c7fcee826324aea2c96e361c1103d09`. No test data or paid API requests were used for this analysis.


### Epoch 5 validation and training exposure

The frozen 16-drawing validation panel improved from macro drawing F1 0.8269021850 at epoch 4 to 0.8375682016 at epoch 5 (+0.0106660166). Class-aware counts are 1,080 TP, 255 FP and 5 FN (1,335 predictions; 1,085 references), giving precision 0.8089887640, recall 0.9953917051 and micro F1 0.8925619835. Localization alone has 1,081 TP, 254 FP and 4 FN. Conditional classification on independent class-agnostic matches is 1,079/1,081; one general→tank and one tank→general confusion remain. These match assignments are independent of class-aware scoring.

Dataset PID macro F1 decreased slightly from 0.9290103799 to 0.9285204659; PID2Graph Synthetic increased from 0.7247939902 to 0.7466159374. The new miss is `general51` in Synthetic/192. Per-class F1 is general 0.8338624339, valve 0.9279835391, pump 0.875, tank 0.6808510638, instrumentation 0.9816272966 and arrow 0.8372093023. Inlet/outlet remains unsupported by this panel. Tank false positives increased from 8 to 15 and arrow false positives from 5 to 7; the aggregate improvement does not erase these regressions.

`training-exposure-snapshot-006.json` verifies five complete sampling epochs at step 27,825: 111,295 crop presentations, all 19,543 prepared crops, all 528 training drawings/groups and all 45,021 physical symbols fully covered. This includes 9,775 background presentations. Exposure counts describe the saved epoch boundary, not the live cursor. The validation gain exceeds the frozen 0.002 threshold, resets stale epochs to zero, and selects epoch 5 provisionally. The sixth and final scheduled epoch is running; convergence and hybrid checkpoint selection remain incomplete.

Artifacts: `detector-epoch-005-analysis.json`, `detector-epoch-005-findings.json`, `detector-epochs-v1/validation-epoch-005.json` (SHA-256 `78d1e3501259bab6680d9d8165d9d4fb8761bc700160e6591ceeb7a6b5314fb9`) and the exposure snapshot. This milestone used no new API calls or sealed test data.

### Epoch 6 completion and training coverage

The trainer exited successfully after the registered maximum of six epochs: 33,390 optimizer steps and 133,554 crop presentations, including 11,730 background presentations. The completed exposure reconstruction (`training-exposure-snapshot-007.json`) verifies all 19,543 prepared crops, 528 drawing groups and 45,021 physical symbols were presented with their full boxes. Repeated and overlapping crops do not increase the unique physical-symbol count. There are no training examples for inlet/outlet.

Pilot-validation macro drawing F1 across epochs 0–6 was 0.663019, 0.776157, 0.814417, 0.815444, 0.826902, 0.837568 and 0.846651. Epoch 6 is the selected trained checkpoint. The last gain was 0.009083; training stopped at `maximum_epochs`, not early stopping. A validation plateau is therefore not established, and the registered schedule was not extended. Recorded training time was 57,708.676 seconds, excluding validation.

On the same 16 validation drawings, epoch 6 produced 1,080 TP, 239 FP and 5 FN: precision 0.818802, recall 0.995392 and micro F1 0.898502. Localization had 1,081 TP, 238 FP and 4 FN. Conditional classification was 1,080/1,081, using independent class-agnostic matches. Relative to epoch 5, 11 drawings improved, 2 worsened and 3 tied. Collection macro F1 was 0.931128 for Dataset PID and 0.762175 for PID2Graph Synthetic.

Class F1: general 0.846402, valve 0.929897, pump 0.903226, tank 0.666667, instrumentation 0.979058 and arrow 0.857143; inlet/outlet is unsupported by this panel. Tank false positives increased from 15 to 16, and instrumentation false positives from 7 to 8. Aggregate improvement does not remove these limitations. Full metrics and convergence history are in `detector-epoch-006-findings.json`; report SHA256 is `92bad5fb5959a2958faf06a5c9368640da7bb7549fe6f9f1458d04ccef003848`.

The existing checkpoint-comparison controller started the epoch-6 broad hybrid pilot run. The broader detector controller initially failed before new inference because sandboxed MPS was unavailable, then resumed the same registered command with local Mac GPU access. No model, scoring, split or inference configuration changed for that recovery. Broader hybrid comparisons, validation-only method selection, the untouched final evaluation and selected production transfer remain incomplete.

### Completed trained detector on broader validation

The epoch-6 detector reference completed all 107 validation drawings (16 reused pilot predictions, 91 new drawings) and exited0. The completion report hash and registered configuration were verified; the comparison with pilot weights holds every detector setting fixed except the checkpoint. No API inference or test access was used.

Macro drawing F1 increased from 0.737536 to 0.879982, with improvement on every drawing. Class-aware TP increased from9,200 to9,244, FP fell from4,206 to1,620 and FN from59 to15. Precision is0.850884, recall0.998380 and microF1 0.918750. Localization TP/FP/FN is9,247/1,617/12; conditional classification is9,232/9,247. Independent matching explains why conditional classification counts differ from class-aware TP.

Collection macroF1 is0.936327 for70 Dataset PID drawings and0.773381 for37 PID2Graph Synthetic drawings. Per-class F1 is general0.862416, valve0.945450, pump0.874172, tank0.737288, instrumentation0.983080 and arrow0.914761. Inlet/outlet has no reference examples. Tank precision0.587838 and Synthetic precision0.633648 remain limitations. Saved drawing inference times sum to1,027.745 seconds including reused pilot times; this excludes failed startup and orchestration.

Fifty previous misses were recovered and six new misses introduced. Visual inspection of all six regressions found four localization failures (IoU0.4115,0.4873,0.4920,0.4553), one tank classified as general (IoU0.9645), and one small valve with no overlapping proposal. The accompanying high-confidence false-positive examples include duplicate and partial-symbol boxes. These targeted cases are not a prevalence sample. See `detector-trained-broad-findings.json`, `detector-trained-broad-analysis.json`, `detector-trained-broad-failure-figures/index.json` and `detector-trained-broad-visual-review.json`. Figures are evaluation-only and were not sent to inference.

The report SHA256 is `dee1669e5a59af2ff0582c00a79758309700ddf2d0d322143aaa5bd4627b69e4`; checkpoint SHA256 is `b98615fb338f7684517e4407b7f489e5cea42308c6bd773536e3c518dba8e111`. This establishes improved detector proposals on validation, not completed VLM comparisons or detailed legend-semantic accuracy. Hybrid finalists, final method selection and fresh held-out evaluation remain pending.

### Epoch 6 production-loader compatibility

The candidate export preserves all 284 model tensors exactly. CPU forward predictions match on one validation crop from each collection. Full-drawing verification through the production `RasterDetector` loader also reproduced all 1,319 predictions exactly across the 16 pilot validation drawings on MPS, including boxes, scores and classes. The full-drawing check took 156.981 seconds. It read no GraphML or test images and made no API calls.

The exported weights are `checkpoint-export-epoch006/detector.pt` (SHA256 `7f27e06d3b0998adc598ae703e4278dc2817a68cf819360a6761595f2411fd3e`). Provenance is in `checkpoint-export-epoch006/verification.json`; full parity is in `checkpoint-export-epoch006-full-parity/report.json` (SHA256 `2b94ab9ea7451e35ed430d0fc9d5991a5e50da01ff8d4d94d1e3608476428a9b`). `checkpoint-export-epoch006-completion.json` binds those artifacts. This verifies a deployable candidate, not final hybrid selection or PDF transfer accuracy. Baseline behavior has not been changed.

Reproduce with new output directories:

```sh
PYTHONPATH=src .venv-cv/bin/python -m eval.pid2graph.checkpoint_export --checkpoint output/pid2graph-phase2/detector-epochs-v1/epoch-006.pt --report output/pid2graph-phase2/detector-epochs-v1/validation-epoch-006.json --panel output/pid2graph-phase2/panel-pilot.json --out output/pid2graph-phase2/checkpoint-export-epoch006
PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-cv/bin/python -m eval.pid2graph.detector_parity --panel output/pid2graph-phase2/panel-pilot.json --reference output/pid2graph-phase2/detector-epochs-v1/validation-epoch-006 --checkpoint output/pid2graph-phase2/checkpoint-export-epoch006/detector.pt --export-verification output/pid2graph-phase2/checkpoint-export-epoch006/verification.json --out output/pid2graph-phase2/checkpoint-export-epoch006-full-parity
```

### Candidate preparation at production PDF resolution

The exported epoch-6 candidate generated 388 unconfirmed proposals on public `two-tanks.pdf` using default production rendering: target DPI 300 and maximum page dimension 6,000. The scan renders at 5,550 × 4,044 pixels; the loader records DPI/effective DPI as 72/72 for this source. Source and rendered-pixel hashes, box coordinates and scan routing were verified. Inference took 9.982 seconds on local MPS and used no API or held-out test data. The proposal counts are 196 general, 126 instrumentation, 51 valve, six tank, five arrow and four pump. These are prediction counts, not physical-symbol counts or accuracy estimates.

Qualitative agent inspection found substantial class and scale transfer limitations. All four pump-labelled proposal crops enclose circular V identification bubbles. Two tank-labelled boxes cover pump-like symbols, two cover curved pipe sections, and two overlap instrument/tag and valve assemblies. The whole-page overview lacks boxes enclosing the two large storage tanks. Small instruments and valves receive proposals, alongside boxes on nozzle labels and pipe portions. These observations use an unannotated PDF and a targeted set of ten crops; they establish neither aggregate accuracy nor the cause of the failures. No benchmark settings were changed in response.

Artifacts are in `pdf-proposals-epoch006-production`: `proposals.json`, `verification.json`, `overview.png`, `tank-pump-details.png` with its JSON index, and `visual-review.json`. This is candidate preparation only. Selected-method VLM interpretation, legend-conditioned review and graph verification at production resolution remain required.

```sh
PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-cv/bin/python -m diagex.vision.pdf_raster_guidance --pdf tests/p-ids-public/two-tanks.pdf --checkpoint output/pid2graph-phase2/checkpoint-export-epoch006/detector.pt --checkpoint-sha256 7f27e06d3b0998adc598ae703e4278dc2817a68cf819360a6761595f2411fd3e --device mps --out output/pid2graph-phase2/pdf-proposals-epoch006-production/proposals.json
```

### Billing reconciliation during trained-checkpoint comparison

`billing-reconciliation-012.json` records two read-only generation-metadata lookups for newly unresolved requests with exact generation IDs and no previous lookup. Both returned HTTP 200: request `997e32c08d344cad90b9a92f5aecf572` cost $0.00069604 and `1553dacbfeba4470b4ec45f93a9146f2` cost $0.000579. These interrupted requests were billable; their reservations were replaced with the confirmed amounts. Older 404 results and requests without generation IDs were not repeatedly queried.

At immutable `billing-snapshot-012.json`, phase-two known actual usage is $1.7883856184 across 4,436 billed requests. Active reservations are zero at that instant; 37 unresolved requests retain $8.68192864 of conservative exposure. Adding the separately recorded user-reported phase-one $0.64 gives known cumulative usage $2.4283856184 and cumulative budget exposure $11.1103142584, below the $60 cap. Exposure is not actual spending. The comparison remains active, so these are point-in-time figures, not final experiment costs.

Pricing verification was subsequently refreshed from the public OpenRouter model-endpoints API before expiry. `price-refresh-20260920-verification.json` verifies identical model revisions, context size, provider allowlist and price ceilings; `deepseek-endpoints-refresh-20260920.json` preserves the HTTP 200 response. The active `prices-v1.json` now has SHA256 `a691ef1bcc28f6450df3dee7981cafed08866c227dc62ab768f4be1d6689037e` and expires at Unix time `1789983948.9566772`. The former bytes remain in `price-verifications/40dab6860090ca6a8e6c16a3cd067168b0d71df65889ca5ca99a590a75c3353c.json` and prior request records retain their own pricing evidence. Server-enforced maximum prices continue to exclude scheduled provider rates above the frozen ceilings. This was a metadata lookup, with no additional inference, provider substitution or benchmark change.

### Completed trained broad-hybrid checkpoint comparison

The epoch-6 broad hybrid finished attempted coverage of all 16 pilot validation drawings: 400 tiles, 393 successful and seven failed. Nine drawings have full successful coverage and seven have partial coverage. The saved report therefore has `complete: false` for failures, but no drawings or tiles remain unattempted and `stopped` is false. Failures are included in all metrics. The existing controller scored this run and started the trained explicit-decision comparison; final checkpoint promotion remains gated on both comparisons finishing.

Macro drawing F1 increased from 0.566602 with pilot weights to 0.594506 with epoch-6 weights (+0.027904). Ten drawings improved and six worsened. TP increased from 812 to 818, FP decreased from 474 to 363 and FN decreased from 273 to 267. Most of the gain is fewer false positives. Precision is 0.692633, recall 0.753917 and micro F1 0.721977. Collection macro F1 is 0.781277 for Dataset PID and 0.407735 for PID2Graph Synthetic. Independent class-agnostic localization gives 1,004 TP, 177 FP and 81 FN; conditional classification is 818/1,004 (0.814741). The most common class mismatches are general→instrumentation (74) and general→valve (60).

The trained detector alone found 1,080 correct symbols. The hybrid retains 818 of these (75.7407%), loses 262 and recovers no additional reference symbol. Of those losses, 185 remain localized with a class mismatch and 77 lack a matching localization. Joining explicit decisions to detector matches associates 21 lost correct symbols with rejection, 13 with unresolved responses, 43 with recognition without retained output, and 185 with retained outputs (184 wrong-class localized matches and one localization failure). These groups describe associations under the frozen matcher, not causal proof. API failures, geometry ownership and class interpretation must remain distinct explanations.

All 1,319 detector proposals were requested. The decision audit records 1,137 recognized, 1,070 retained, 67 recognized without output, 122 rejected-only and 60 unresolved; 204 had recognition/rejection conflicts across views or attempts. Visual inspection of eight examples from the lowest-F1 fully successful drawing in each collection found overlapping localized predictions with GraphML class mismatches. The figures exclude failed-response drawings for this targeted inspection, but the full metrics retain every failed tile. These examples do not establish detailed engineering semantics or overall error prevalence.

Recorded inference runtime is 5,025.992 seconds. The run has $0.2167001344 confirmed billed usage across 405 requests, no active reservation, and three unresolved requests retaining $0.70394016 exposure. That exposure is not a bill. `billing-reconciliation-013.json` separately records successful provider lookups for two interrupted requests costing $0.00055006 and $0.00051782. Its whole-study snapshot has known phase-two usage $1.9067121728, plus the prior user-reported $0.64. The study remains live; these are not final cumulative totals.

Artifacts under `checkpoint-comparison-v1` are `broad_raster_recognition-report.json` (SHA256 `2c1a60897188383f98abeef4ac8ef21c79d6ac2ad46b13ae6aa2006effd82604`), `broad-trained-analysis.json`, `broad-trained-attempt-verification.json`, `broad-trained-retention.json`, `broad-trained-decisions.json`, `broad-trained-decision-outcomes.json`, `broad-trained-findings.json`, `broad-trained-failure-figures/index.json` and `broad-trained-visual-review.json`. The verification checks source images, configuration, saved tile/drawing predictions and request attribution. No new inference, benchmark changes or test access were used for this analysis. Broader hybrid validation and final selection remain incomplete.

### Broad-response ownership replay

`eval.pid2graph.broad_ownership_audit` replays the saved broad tool payloads through the frozen normalizer, coordinate projection and tile ownership checks. It reproduced saved review boxes, classes, ordering, confidences and source proposal IDs, then reproduced the full saved prediction records, on every one of the 393 successful tiles. The replay checked 2,224 normalized observations, including 227 discovery observations across views. Seven failed tiles remain separate and are not counted as successful replays.

All 67 recognized proposals absent from the final output occurred only in successful observations whose projected box centers were outside the observing tile's ownership region. None required another explanation or had a recognized response on a failed tile. The decision-outcome join identifies 44 correct detector proposals in this group; 43 reference symbols remain missed, while one is recovered by another output. This verifies a deterministic filtering mechanism. It does not establish why recognition was absent in the owning view, or that removing ownership would improve accuracy; duplicate and partial observations remain a concern. No inference setting was changed and no fifth hybrid was introduced.

An initial replay stopped before producing a report because JSON key sorting changed dict representations embedded in Pydantic validation-error text. The audit now compares exact structured decisions, statuses and rejected wire rows independently of generated exception wording. Successful review and prediction equality remains exact. Seven focused tests cover serialized error text, changed boxes/classes/source IDs, changed predictions and duplicate accepted payloads; Ruff passes. Reports are `checkpoint-comparison-v1/broad-trained-ownership-audit.json` and `broad-trained-ownership-verification.json`. No GraphML, test input or API call is used by this replay.

```sh
PYTHONPATH=src .venv/bin/python -m eval.pid2graph.broad_ownership_audit --panel output/pid2graph-phase2/panel-pilot.json --run output/pid2graph-phase2/checkpoint-comparison-v1/broad_raster_recognition --audit output/pid2graph-phase2/checkpoint-comparison-v1/broad-trained-decisions.json --out output/pid2graph-phase2/checkpoint-comparison-v1/broad-trained-ownership-audit-new.json
```

### Broader hybrid validation controller

`eval.pid2graph.broader_hybrids` now executes the existing pilot-to-broader policy after checkpoint promotion. It validates both promoted registrations and their frozen method fingerprints, checks the selected checkpoints against the paired validation ranking, verifies that broader detector hints preserve the inference signature, and seeds the selected pilot attempts through `reuse_validation`. The frozen broad and explicit drivers process the remaining validation drawings sequentially. Reused predictions, failed attempts, runtime and billing IDs are preserved; they do not cause new API calls. Reports retain failed tiles in their denominators. Final candidate selection and test access are separate steps.

The controller is resumable, holds an exclusive process lock, and records its own orchestration source hashes. A software-fixture integration test exercises both possible checkpoint choices, new failed requests with unresolved billing, and a second completed invocation with no additional driver calls. Refusal tests cover absent promotion, extra variants, altered source, wrong detector weights and changed destination configuration. The promotion and reuse test suites pass all 31 tests, and Ruff passes. Twenty existing frozen training/inference source fingerprints remain unchanged. Evidence is `broader-hybrids-implementation-check.json`.

At preparation time the explicit checkpoint comparison was still live: ten complete drawing summaries, 265 attempted tiles (264 successful, one failed), with Synthetic/33 in progress. The controller has **not** been launched on study data. Finish that comparison, reconcile available bills, analyze the result and create `checkpoint-promotion-v1` first. Then prepare using:

```sh
PYTHONPATH=src .venv/bin/python -m eval.pid2graph.broader_hybrids --study output/pid2graph-phase2 --promotion output/pid2graph-phase2/checkpoint-promotion-v1/promotion.json --manifest output/pid2graph-development/manifest-v1.json --out output/pid2graph-phase2/hybrid-broad-v1 --pilot-proposals output/pid2graph-phase2/detector-pilot-broad-v1 --trained-proposals output/pid2graph-phase2/detector-trained-broad-v1 --prepare-only
```

Remove `--prepare-only` to run or resume the prepared experiment. The existing metered clients enforce spending and current pricing verification; this controller does not change those limits. Completion here means attempted coverage of the two broader hybrid arms, including recorded failures, not completion of the overall study. The completed broader baseline remains required for final validation-only selection.

### Billing lookup during the explicit checkpoint comparison

`billing-reconciliation-014.json` records one new read-only lookup for interrupted request `b5dca6d3dfa84893945c1908c8d90474`. Its exact generation ID returned HTTP 404. The request remains unresolved with its original conservative exposure; no free-request assumption or repeated lookup was made. Existing unresolved requests with no generation ID or a previous lookup were skipped.

At immutable `billing-snapshot-014.json`, confirmed phase-two charges total $2.0902334412 across 4,979 billed requests. Active reservations are $0.23464672 and unresolved exposure is $9.3858688. Including the user-reported prior $0.64 gives $2.7302334412 in known cumulative charges and $12.3507489612 in cumulative budget exposure, below the $60 cap. Exposure is not billed spending, and the ongoing comparison makes these point-in-time figures rather than final totals.

### Completed explicit checkpoint comparison and promotion

The comparison controller exited successfully after finishing both trained-weight arms. The explicit-decision arm attempted all 400 tiles on 16 pilot validation drawings: 398 successful and two failed (`APIConnectionError`), leaving 14 fully successful drawings and two partial drawings. All failures remain in the scores; the report's `complete: false` denotes those failures, with no unattempted inputs and `stopped: false`.

Explicit macro drawing F1 improved from 0.435388 to 0.484952 (+0.049564); 13 drawings improved and three worsened. Counts changed by +57 TP, −53 FP and −57 FN, giving 643 TP, 317 FP and 442 FN. Precision is 0.669792, recall 0.592627 and micro F1 0.628851. Collection macro F1 is 0.694318 for Dataset PID and 0.275587 for PID2Graph Synthetic. Per-class F1 is general 0.311111, valve 0.788594, pump 0.111111, tank 0.388889, instrumentation 0.687090 and arrow 0.0. Inlet/outlet has no reference support and four false positives. Localization alone gives 903 TP, 57 FP and 182 FN; conditional classification is 643/903. The largest class confusions are general→valve (121) and general→instrumentation (76).

The explicit hybrid retains 643 of the detector's 1,080 correct symbols (59.5370%) and loses 437: 260 remain localized with a class mismatch and 177 lack a matching localization. It recovers no reference symbols beyond the detector and corrects no detector class errors. All 1,319 detector proposals were requested; the decision audit finds 1,057 recognized, 958 retained, 99 recognized without output, 56 rejected-only and 206 unresolved, with 99 recognition/rejection conflicts across observations. Outcome joins associate 14 lost correct symbols with rejection, 86 with unresolved responses, 67 with recognition without output, and 270 with retained outputs. These are associations under the frozen matcher, not causal attributions.

Ownership replay checks all 400 tile records and 1,776 normalized raster objects. All 99 recognized proposals missing from output appear only in observations outside the observing tile's ownership region. Their detector matches include 67 correct symbols, none recovered elsewhere. This establishes the filtering mechanism, not the cause of absent recognition in the owning tile, and does not justify removing duplicate-control rules. No inference settings were changed.

Agent inspection covered eight error-example panels from the lowest-F1 fully successful drawing per collection: Synthetic/132 (F1 0.139535) and Dataset PID/184 (F1 0.606061). All eight have an independent localization match but mismatched broad classes. Examples include synthetic polygon/grid and flared glyphs, the INS pipe assembly, a circular DDL tag, and short parallel end marks. These targeted examples do not measure prevalence or establish detailed engineering semantics. The figures remain evaluation-only and were not supplied to inference.

Recorded explicit inference runtime is 5,301.049 seconds. Reconciliation `015` obtained an HTTP 200 generation record for interrupted request `f944e79c7a8847c5b904fd83cea0da8e`, confirming a charge of $0.000544129. The refreshed explicit report has $0.2197054662 in actual charges across 401 billed requests, zero active reservations and two unresolved requests with $0.46929344 exposure. The immutable whole-study snapshot has $2.1350334406 in phase-two actual charges, or $2.7750334406 including prior user-reported spending. Forty unresolved requests retain $9.3858688 exposure; cumulative exposure is $12.1609022406, not a billed total.

Artifacts under `checkpoint-comparison-v1` include `explicit_proposal_decisions-report.json` (SHA256 `be17474e917aea9c467214287c6b1bdd9b16d0de303484d5056f4fd48d489746`), `explicit-trained-analysis.json`, `explicit-trained-retention.json`, `explicit-trained-decisions.json`, `explicit-trained-decision-outcomes.json`, `explicit-trained-ownership-audit.json`, `explicit-trained-attempt-verification.json`, `explicit-trained-findings.json`, and the failure figures and visual review. Attempt verification checks source images, configuration, tile/drawing prediction equality and billing identities.

After reconciliation, `checkpoint_promotion` revalidated both complete checkpoint comparisons and scored all four candidate reports against one immutable billing snapshot. The frozen ranking selected trained epoch-6 weights for both shortlisted methods. Promotion seal: `24818ad38824a469f2b1ed0ba1f1a61168a45c427fb0a72e7dcac0e9e28c8bbd`; records are in `checkpoint-promotion-v1`. This chooses weights for broader validation, not the final method.

The prepared `hybrid-broad-v1` experiment has started under the documented controller command without `--prepare-only`. Each arm reuses 16 drawings (400 attempted tiles) and will process 91 new validation drawings; the broad variant runs first, followed by explicit decisions. Failed pilot attempts and billing identities remain unchanged. The baseline and both detector references already cover the same 107 drawings. Broader hybrid results, validation-only final selection, fresh held-out evaluation and selected production transfer remain incomplete; the final test inputs remain untouched.

### Final selection gate prepared while broader validation runs

`eval.pid2graph.final_selection` provides the next offline gate. It requires a completed broader controller, completed training evidence bound to both checkpoint comparisons, unchanged promoted methods and a finished baseline report. It checks tile/drawing prediction equality, failure counts and request provenance for all three broader candidates, then refreshes their billing against one immutable ledger snapshot without changing their predictions or metrics. Baseline requests are verified in the `baseline` ledger category; hybrid requests use `experiments`. The running frozen verifier and inference sources are unchanged.

The gate applies the existing accuracy/cost/runtime ranking and stores one choice under `final-selection-v1/selection.json`. Resume preserves that choice even if later billing evidence arrives. Changed source reports or signatures cause refusal. It performs no inference and scores validation GraphML only. It does not substitute for the subsequent selected-method final evaluation or integration verification.

Forty-one promotion, orchestration and selection tests pass, including fixtures where each of the three candidates wins, billing reconciliation changes case costs, failed requests remain unresolved, and incomplete or altered evidence prevents selection. Ruff passes. A negative check against the real live study returned `Both broader hybrid comparisons must finish before final selection` before creating an output directory. Evidence is `final-selection-implementation-check.json`; no final selection or test access occurred.

After both broader arms finish and available bills have been reconciled, run:

```sh
PYTHONPATH=src .venv/bin/python -m eval.pid2graph.final_selection --study output/pid2graph-phase2 --hybrids output/pid2graph-phase2/hybrid-broad-v1 --comparison output/pid2graph-phase2/checkpoint-comparison-v1 --manifest output/pid2graph-development/manifest-v1.json
```

The broader process remains live. Its first new drawing, Dataset PID/0, finished all 25 tiles successfully; Dataset PID/100 tile 15 failed after a connection interruption and later tiles continued successfully. This failed attempt will remain in the broader scores.

### Broader-run billing reconciliation 016

Two newly interrupted requests with exact generation IDs were reconciled through read-only provider lookups. Both returned HTTP 200: `f8351adaf4314808bc747842aacb72e7` cost $0.00026344 and `cd47e27753a24e7d9d98ec49a5a70616` cost $0.00053508. Their combined confirmed charge is $0.00079852; unresolved reservations were replaced with those bills. No inference was repeated and no older unresolved lookup was retried.

At `billing-snapshot-016.json`, phase-two actual charges total $2.2532382902 across 5,277 billed requests. Active reservations were zero at that instant. Forty unresolved requests retain $9.3858688 of exposure. Including the user-reported prior $0.64 gives $2.8932382902 in known cumulative charges and $12.2791070902 in cumulative budget exposure. The ongoing run makes these snapshot values, not final totals; exposure is not a bill. Full lookup evidence is `billing-reconciliation-016.json`.

The broad-recognition arm subsequently finished Dataset PID/12 with one failed tile (13) retained in the run and began Dataset PID/130. Broader validation remains incomplete, and no final-test input was opened.

`billing-reconciliation-017.json` subsequently settled interrupted request `302a0d9cd59a4c79981bda01241aa132` through an HTTP 200 generation lookup: actual charge $0.00051982. At snapshot 017, phase-two actual charges are $2.3348447234 across 5,380 billed requests, with no active reservations at that instant and 40 unresolved requests retaining $9.3858688 exposure. Adding prior user-reported spending gives $2.9748447234 known cumulative charges and $12.3607135234 cumulative budget exposure. These are live-study snapshot values, not final totals. No previous failed lookup or inference was repeated. Dataset PID/139 then finished with failed tile 11 retained, and Dataset PID/14 began.

`billing-reconciliation-018.json` records an HTTP 404 lookup for timed-out request `14ef9b99c43d43e4a56ac6198b3ac3e2`. No bill was available, so its unresolved exposure was retained. At snapshot 018, phase-two actual charges total $2.469188135 across 5,556 billed requests; active reservations are $0.23464672 and unresolved exposure is $9.85516224. Including the prior user-reported $0.64 gives $3.109188135 known cumulative charges and $13.198997095 cumulative exposure. The timeout recovered on retry, and Dataset PID/196 finished all 25 tiles successfully before Dataset PID/205 began. No inference or older billing lookup was repeated; final evaluation remains pending.

`billing-reconciliation-019.json` records an HTTP 404 lookup for interrupted request `52f9ce84709b4c09a3dceae4d2bd9e8b`; its actual bill remains unknown and its reserve is retained. At snapshot 019, phase-two known charges are $2.4894554926 across 5,579 billed requests, active reservations are $0.23464672, and unresolved exposure is $10.08980896. Including prior spending gives $3.1294554926 known cumulative charges and $13.4539111726 cumulative exposure, not a billed total. Dataset PID/205 finished with failed tile 13 retained, and Dataset PID/221 began. Neither inference nor an old billing lookup was repeated.


### Billing reconciliation 020

The broader broad-recognition run completed Dataset PID/292 with tile 18 failed after an interrupted response. The failed tile remains in evaluation. Exact generation lookup for request `d458d6ca88f54b649c44e9ba790d3df6` returned HTTP 200 and established an actual charge of $0.00074704. Evidence is `billing-reconciliation-020.json` and the immutable `billing-snapshot-020.json`.

At that snapshot, phase-two billed usage was $2.7305039502, active reservations were $0.23464672, and unresolved upper bounds were $10.32445568. Including the user-reported prior $0.64, known cumulative actual spending was $3.3705039502 and cumulative budget exposure was $13.9296063502. Exposure is not billed cost, and this snapshot is not the final total while inference continues. The raw unresolved count includes pending requests. No unknown failed request was treated as free.


### Billing reconciliation 021

Dataset PID/293 completed all 25 tiles successfully; two interrupted responses recovered under the frozen retry policy. Exact generation lookups established actual charges of $0.00025616 and $0.00051110 for requests `91ff536892d9429bb8c07cc5dabc54d8` and `09911b03b3fe475eab16fbe6c45d7c20`. Evidence is `billing-reconciliation-021.json` and immutable `billing-snapshot-021.json`.

At that snapshot, phase-two billed usage was $2.7552529518, active reservations were $0.23464672, and unresolved upper bounds were $10.32445568. Including the user-reported prior $0.64, known cumulative actual spending was $3.3952529518 and cumulative budget exposure was $13.9543553518. Exposure is not billed cost, and this is not the final total while inference continues. The raw unresolved count includes pending requests.


### Billing reconciliation 022

The ongoing broader run exposed one newly eligible unresolved request with an exact generation ID. Provider lookup for `f82d2f51bf4440edb94c62ff80be7ae2` returned HTTP 200 and established an actual charge of $0.00051432. Evidence is `billing-reconciliation-022.json` and immutable `billing-snapshot-022.json`. No inference or older billing lookup was repeated.

At that snapshot, phase-two billed usage was $3.4370439002 across 7,010 billed requests, active reservations were $0.23464672, and unresolved upper bounds were $11.96698272. Including the user-reported prior $0.64, known cumulative actual spending was $4.0770439002 and cumulative budget exposure was $16.2786733402. Exposure is not billed cost; these are snapshot values while inference continues. The raw unresolved count includes pending requests, and unknown charges remain reserved.

Synthetic/205 subsequently completed with failed tile 12 retained under the frozen retry policy; Synthetic/223 began. Broader comparisons, final selection and untouched final evaluation remain incomplete.


### Billing reconciliation 023

Synthetic/223 completed with tile 5 failed after an interrupted response; the failure remains in evaluation. Exact provider lookup for request `ae7ac769e9e1436492fa6dae5af76a02` returned HTTP 200 and established an actual charge of $0.00033754. Evidence is `billing-reconciliation-023.json` and immutable `billing-snapshot-023.json`. No inference or older billing lookup was repeated.

At that snapshot, phase-two billed usage was $3.4440925486 across 7,031 billed requests, active reservations were $0.23464672, and unresolved upper bounds were $11.96698272. Including the user-reported prior $0.64, known cumulative actual spending was $4.0840925486 and cumulative budget exposure was $16.2857219886. Exposure is not billed cost; these are snapshot values while inference continues. The raw unresolved count includes pending requests, and unknown charges remain reserved. Synthetic/232 began; broader comparisons, final selection and untouched final evaluation remain incomplete.


### Billing reconciliation 024

Synthetic/26 completed with tile 25 failed after an interrupted response; the failure remains in evaluation. Exact provider lookup for request `d9720b9848e0475dbdf1325843bd2570` returned HTTP 200 and established an actual charge of $0.00034646. Evidence is `billing-reconciliation-024.json` and immutable `billing-snapshot-024.json`. No inference or older billing lookup was repeated.

At that snapshot, phase-two billed usage was $3.4848931566 across 7,153 billed requests, active reservations were $0.23464672, and unresolved upper bounds were $11.96698272. Including the user-reported prior $0.64, known cumulative actual spending was $4.1248931566 and cumulative budget exposure was $16.3265225966. Exposure is not billed cost; these are snapshot values while inference continues. The raw unresolved count includes pending requests, and unknown charges remain reserved. Synthetic/36 began; broader comparisons, final selection and untouched final evaluation remain incomplete.


### Billing reconciliation 025

Synthetic/6 completed with tile 24 failed after an interrupted response; the failure remains in evaluation. Exact provider lookup for request `20feb54eb43b4545b49d38e2ed5c237e` returned HTTP 404. Its actual charge remains unknown and its reservation is retained. Evidence is `billing-reconciliation-025.json` and immutable `billing-snapshot-025.json`. No inference or older billing lookup was repeated.

At that snapshot, phase-two billed usage was $3.4986507998 across 7,199 billed requests, active reservations were zero at that instant, and unresolved upper bounds were $12.20162944 across 52 unresolved requests. Including the user-reported prior $0.64, known cumulative actual spending was $4.1386507998 and cumulative budget exposure was $16.3402802398. Exposure is not billed cost; these are snapshot values while inference continues. Synthetic/60 began; broader comparisons, final selection and untouched final evaluation remain incomplete.


### Billing reconciliation 026

Synthetic/75 completed with tile 4 failed and retained; tile 6 recovered from a provider timeout under the frozen retry policy. Exact provider lookup for interrupted request `c552a0f18e2248469bffed7d0211236f` returned HTTP 404. Its actual charge remains unknown and its reservation is retained. Evidence is `billing-reconciliation-026.json` and immutable `billing-snapshot-026.json`. No inference or older billing lookup was repeated.

At that snapshot, phase-two billed usage was $3.5321037382 across 7,299 billed requests, active reservations were $0.23464672, and unresolved upper bounds were $13.37486304. Including the user-reported prior $0.64, known cumulative actual spending was $4.1721037382 and cumulative budget exposure was $17.7816134982. Exposure is not billed cost; these are snapshot values while inference continues. The raw unresolved count includes pending requests. Synthetic/76 began; broader comparisons, final selection and untouched final evaluation remain incomplete.


### Billing reconciliation 027

Synthetic/76 completed with tiles 13 and 16 failed and retained; later interrupted requests recovered under the frozen retry policy. Three exact provider lookups returned HTTP 200: `cc648b5904d443fd90810c5786d04a4e` cost $0.00051502, `1157803e7b434ac7937f92e6bd51bfd8` cost $0.00040652, and `903b0daa806c4086a2632ec78b72d014` cost $0.00035360. Their combined confirmed charge is $0.00127514. Evidence is `billing-reconciliation-027.json` and immutable `billing-snapshot-027.json`. No inference or older billing lookup was repeated.

At that snapshot, phase-two billed usage was $3.5400333694 across 7,327 billed requests, active reservations were $0.23464672, and unresolved upper bounds were $13.60950976. Including the user-reported prior $0.64, known cumulative actual spending was $4.1800333694 and cumulative budget exposure was $18.0241898494. Exposure is not billed cost; these are snapshot values while inference continues. The raw unresolved count includes pending requests. Synthetic/89 began with tile 5 failed; broader comparisons, final selection and untouched final evaluation remain incomplete.


### Billing reconciliation 028

The interrupted Synthetic/89 request `5130f561fa7f43ff84a472ebd3be90c6` was subsequently reconciled through an exact HTTP 200 generation lookup at $0.00039410. Its failed tile remains in evaluation. Evidence is `billing-reconciliation-028.json` and immutable `billing-snapshot-028.json`; no inference or old lookup was repeated.

At snapshot 028, phase-two billed usage was $3.542689595 across 7,336 billed requests, active reservations were zero at that instant, and 58 unresolved requests retained $13.60950976 exposure. Including prior user-reported spending gives $4.182689595 known cumulative charges and $17.792199355 cumulative exposure. These live-study snapshot values are not final totals, and exposure is not billed cost.


### Billing reconciliation 029

Synthetic/9 completed with tile 17 failed after an interrupted response; the failure remains in evaluation. Exact provider lookup for request `1673d1ad57d24f6c911433a53f93606e` returned HTTP 200 and established an actual charge of $0.00040796. Evidence is `billing-reconciliation-029.json` and immutable `billing-snapshot-029.json`. No inference or older billing lookup was repeated.

At that snapshot, phase-two billed usage was $3.5552538418 across 7,375 billed requests, active reservations were $0.23464672, and unresolved upper bounds were $13.60950976. Including the user-reported prior $0.64, known cumulative actual spending was $4.1952538418 and cumulative budget exposure was $18.0394103218. Exposure is not billed cost; these are snapshot values while inference continues. The raw unresolved count includes pending requests. Synthetic/93, the final drawing of the first broader hybrid arm, began; both broader comparisons, final selection and untouched final evaluation remain incomplete.


### Completed broader broad-recognition comparison

The promoted epoch-six broad-recognition hybrid finished all 107 validation drawings: 2,645 successful tiles and 30 failed tiles, with 81 complete and 26 partial drawings. All failures remain in scoring. Macro drawing F1 is 0.6503097185, compared with baseline 0.2364875355 and trained detector 0.8799815104 on the same panel. The hybrid improves on baseline for 106 drawings and worsens one. Its class-aware totals are TP 7,383, FP 2,743 and FN 1,876; precision is 0.7291131740, recall 0.7973863268, and micro F1 0.7617229817. Localization F1 is 0.8971885478. Independent localization matches give conditional classification accuracy 7,361/8,696 (0.8464811408).

Dataset PID macro F1 is 0.8022645322 across 70 drawings; PID2Graph Synthetic macro F1 is 0.3628276386 across 37. The largest localized class confusions are general to instrumentation (491) and general to valve (468). Broad labels and differing collection conventions do not establish detailed engineering semantics.

The hybrid retains 7,382 of 9,244 detector-correct truths (79.8572%). Of 1,862 lost truths, 1,310 remain localized but misclassified and 552 lack localization under frozen matching. It recovers one truth beyond the detector and corrects no independently localized detector class error. Explicit response audits show 455 recognized proposals without retained output. Of these, 282 correspond to detector true positives; 276 truths are absent from the hybrid and six are recovered elsewhere. Ownership replay is running before making a filtering attribution.

Reproducible reporting artifacts under `hybrid-broad-v1` are `broad-analysis-vs-baseline.json`, `broad-retention.json`, `broad-decisions.json`, `broad-decision-outcomes.json`, and `broad-findings.json`. They use the existing `eval.pid2graph.analyze`, `proposal_retention`, `raster_decision_audit`, and `decision_outcomes` CLIs without inference changes. Runtime including reused cases is 48,705.7663 seconds. At the arm report snapshot, actual charges are $1.6448063996, active reservations zero, and unresolved upper bounds $4.92758112. These are this arm's charges, not cumulative study charges, and remaining unknown costs prevent a final actual-cost total.

The controller has started the required explicit-decision comparison; Dataset PID/0 tile 2 failed and remains in evaluation. Neither final method selection nor test evaluation has occurred.

### Billing reconciliation 030

The newly interrupted explicit-arm request `cabdc32a95e24817a9fc52191c496d86` was reconciled through an exact HTTP 200 generation lookup at $0.00075558. Evidence is `billing-reconciliation-030.json` and immutable `billing-snapshot-030.json`; no inference or old lookup was repeated.

At snapshot 030, phase-two billed usage was $3.5675506746 across 7,408 billed requests, active reservations zero at that instant, and 58 unresolved requests retained $13.60950976 exposure. Including prior user-reported spending gives $4.2075506746 known cumulative charges and $17.8170604346 cumulative exposure. These live-study snapshot values are not final totals, and exposure is not billed cost.


### Broader broad-recognition ownership and visual findings

The validation-only ownership replay completed successfully over all 107 drawings. It reproduced the saved reviews and predictions on all 2,645 successful tiles, preserved 30 failed tiles separately, and checked 18,590 normalized observations including 2,017 discoveries. Every one of the 455 recognized proposals absent from output was observed only outside the recognizing tile's ownership region; none required another filtering explanation. This establishes deterministic output filtering, not why the VLM failed to recognize the same proposal in its owning view. Of these 455 proposals, 282 were detector true positives; six truths were recovered elsewhere and 276 were lost (272 without localization, four still localized with class mismatch). No threshold or inference change was made. Evidence is `broad-ownership-audit.json` and `broad-ownership-verification.json` under `hybrid-broad-v1`, superseding the pending-replay statement in the earlier findings artifact.

Agent visual inspection covered eight detail panels from the lowest-F1 fully successful newly evaluated drawing in each collection, excluding reused pilot cases: Synthetic/172 and Dataset PID/339. Synthetic/172 has TP 5, FP 25, FN 16 and F1 0.1961; its displayed symbols have close boxes but general-to-valve/instrumentation class mismatches. Dataset PID/339 has TP 83, FP 49, FN 28 and F1 0.6831; examples include a tapered inline symbol classified as instrumentation, a valve prediction on blank background, and paired strokes classified as inlet/outlet or valve against general truth. These are broad-label validation errors, not a detailed engineering interpretation or proof of real-PDF accuracy. The full report was not rescored from the figure subset. Evidence is `broad-figure-input.json`, `broad-failure-figures/index.json`, and `broad-visual-review.json`; annotated images remain scoring-only.


### Billing reconciliation 031

The explicit-decision arm completed Dataset PID/102 with tile 9 failed after an interrupted response; the failure remains in evaluation. Exact provider lookup for request `9020eeba489a4819b44408c53ced8324` returned HTTP 200 and established an actual charge of $0.00068026. Evidence is `billing-reconciliation-031.json` and immutable `billing-snapshot-031.json`. No inference or older billing lookup was repeated.

At that snapshot, phase-two billed usage was $3.606431553 across 7,465 billed requests, active reservations were $0.23464672, and unresolved upper bounds were $13.84415648. Including the user-reported prior $0.64, known cumulative actual spending was $4.246431553 and cumulative budget exposure was $18.325234753. Exposure is not billed cost; these are snapshot values while inference continues. The raw unresolved count includes pending requests. Dataset PID/106 began; the second broader comparison, final selection and untouched final evaluation remain incomplete.


### Billing reconciliation 032

Three newly eligible completed requests were reconciled using exact generation IDs; all returned HTTP 200, establishing $0.003298629 in charges. Evidence is `billing-reconciliation-032.json` and immutable `billing-snapshot-032.json`. No inference or older lookup was repeated.

At that snapshot, phase-two billed usage was $3.8066661620 across 7,731 billed requests, active reservations were $0.23464672, and unresolved upper bounds were $14.78274336. Including prior user-reported spending of $0.64 gives $4.4466661620 known cumulative charges and $19.4640562420 cumulative exposure. Exposure is not billed cost; the raw unresolved count includes pending requests. The explicit-decision broader comparison remains live, and final method selection and untouched evaluation have not occurred.


### Billing reconciliation 033

Explicit-arm Dataset PID/181 tile 8 failed after an interrupted response and remains in evaluation. Exact generation lookup for request `31d47ece13d344ecb3713ef263f98a47` returned HTTP 200, establishing $0.0008708 billed. Evidence is `billing-reconciliation-033.json` and immutable `billing-snapshot-033.json`. No inference or older lookup was repeated.

At that snapshot, phase-two billed usage was $3.8563159076 across 7,794 billed requests, active reservations were $0.23464672, and unresolved upper bounds were $14.78274336. Including prior user-reported $0.64 gives $4.4963159076 known cumulative charges and $19.5137059876 cumulative exposure. Exposure is not billed cost; these are live-study snapshot values, and raw unresolved counts include pending requests. The second broader comparison, final selection and untouched final evaluation remain incomplete.


### Billing reconciliation 034

Explicit-arm Dataset PID/195 finished with tile 21 failed after an interrupted response; that failure remains in evaluation. Exact generation lookup for request `6b550e38ea8041adbe616923d5fe1737` returned HTTP 200, establishing $0.000266369 billed. Evidence is `billing-reconciliation-034.json` and immutable `billing-snapshot-034.json`. No inference or older lookup was repeated.

At that snapshot, phase-two billed usage was $3.8805584694 across 7,839 billed requests, active reservations were $0.23464672, and unresolved upper bounds were $14.78274336. Including prior user-reported $0.64 gives $4.5205584694 known cumulative charges and $19.5379485494 cumulative exposure. Exposure is not billed cost; these are live-study snapshot values, and raw unresolved counts include pending requests. The second broader comparison, final selection and untouched final evaluation remain incomplete.


### Billing reconciliation 035

Explicit-arm Dataset PID/221 finished with tile 13 failed after an interrupted response; that failure remains in evaluation. Exact generation lookup for request `87cbb17b6bae4ce6b0b642eff419719f` returned HTTP 200, establishing $0.001498089 billed. Evidence is `billing-reconciliation-035.json` and immutable `billing-snapshot-035.json`. No inference or older lookup was repeated.

At that snapshot, phase-two billed usage was $3.9334122936 across 7,904 billed requests, active reservations were $0.23464672, and unresolved upper bounds were $15.01739008. Including prior user-reported $0.64 gives $4.5734122936 known cumulative charges and $19.8254490936 cumulative exposure. Exposure is not billed cost; these are live-study snapshot values, and raw unresolved counts include pending requests. The second broader comparison, final selection and untouched final evaluation remain incomplete.


### Billing reconciliation 036

Explicit-arm Dataset PID/227 tile 13 failed after an interrupted response and remains in evaluation. Exact generation lookup for request `2188a56bea8047568ef7ca9b2cb44c4b` returned HTTP 200, establishing $0.00110376 billed. Evidence is `billing-reconciliation-036.json` and immutable `billing-snapshot-036.json`. No inference or older lookup was repeated.

At that snapshot, phase-two billed usage was $3.9545638352 across 7,927 billed requests, active reservations were $0.23464672, and unresolved upper bounds were $15.01739008. Including prior user-reported $0.64 gives $4.5945638352 known cumulative charges and $19.8466006352 cumulative exposure. Exposure is not billed cost; these are live-study snapshot values, and raw unresolved counts include pending requests. The second broader comparison, final selection and untouched final evaluation remain incomplete.


### Billing reconciliation 037

Explicit-arm Dataset PID/234 tiles 11 and 12 failed after interrupted responses and remain in evaluation. Two newly eligible exact generation lookups returned HTTP 200: request `81fa111f88a449af87037ad8b47bdac1` billed $0.00075418 and `cbc0456a6ae94112a2b30346a828f29b` billed $0.00082628, totaling $0.00158046. Evidence is `billing-reconciliation-037.json` and immutable `billing-snapshot-037.json`. No inference or older lookup was repeated.

At that snapshot, phase-two billed usage was $3.9730376360 across 7,954 billed requests, active reservations were $0.23464672, and unresolved upper bounds were $15.01739008. Including prior user-reported $0.64 gives $4.6130376360 known cumulative charges and $19.8650744360 cumulative exposure. Exposure is not billed cost; these are live-study snapshot values, and raw unresolved counts include pending requests. The second broader comparison, final selection and untouched final evaluation remain incomplete.


### Billing reconciliation 038

Explicit-arm Dataset PID/237 tile 13 failed after an interrupted response and remains in evaluation. Exact generation lookup for request `bd05c01eb7774ecc830ddea18d4dca6a` returned HTTP 200, establishing $0.00094822 billed. Evidence is `billing-reconciliation-038.json` and immutable `billing-snapshot-038.json`. No inference or older lookup was repeated. A later tile 20 timeout recovered through the frozen built-in retry.

At that snapshot, phase-two billed usage was $3.9956360272 across 7,981 billed requests, active reservations were $0.23464672, and unresolved upper bounds were $15.01739008. Including prior user-reported $0.64 gives $4.6356360272 known cumulative charges and $19.8876728272 cumulative exposure. Exposure is not billed cost; these are live-study snapshot values, and raw unresolved counts include pending requests. The second broader comparison, final selection and untouched final evaluation remain incomplete.


### Billing reconciliation 039

Dataset PID/237 finished with failed tiles 13 and 23 retained; tile 20 recovered through the built-in retry. Two additional eligible exact generation lookups were made: request `f76f0032b2a84a668b7e21545bbe2269` returned HTTP 200 and billed $0.000338609; `eaebd578b02b4b74ad604fc54cae6d8e` returned HTTP 404 and remains unresolved with its reservation retained. Evidence is `billing-reconciliation-039.json` and immutable `billing-snapshot-039.json`. No inference or older lookup was repeated.

At that snapshot, phase-two billed usage was $3.9995582442 across 7,988 billed requests, active reservations were $0.23464672, and unresolved upper bounds were $15.25203680. Including prior user-reported $0.64 gives $4.6395582442 known cumulative charges and $20.1262417642 cumulative exposure. Exposure is not billed cost; these are live-study snapshot values, and raw unresolved counts include pending requests. The second broader comparison, final selection and untouched final evaluation remain incomplete.


### Transport interruption, pricing refresh, and billing reconciliation 040

Session 11383 exited with code 1 after the existing perception failure limit was reached on Dataset PID/242. Saved tiles 13, 15, 16 and 17 failed with timeout/connection errors; all attempted records remain immutable and in scoring. Pricing verification had also expired by inspection. Public endpoint evidence was refreshed with the same allowed providers, model revision, context, scheduled pricing and server-enforced ceilings. Evidence is `price-refresh-20260921-verification.json`; previous price bytes are archived by SHA-256. Resume is limited to untouched tiles, with pre-resume hashes recorded in `hybrid-broad-v1/transport-resume-001.json`. No inference code or selection policy changed.

Reconciliation 040 returned HTTP 200 for two exact generation IDs, establishing $0.001749478 in charges. Snapshot phase-two billed cost is $4.0109138278, active reservations are zero, and unresolved upper bounds are $16.89456384. Including prior user-reported $0.64 gives $4.6509138278 known cumulative cost and $21.5454776678 exposure. Unknown charges are not treated as free. Final selection and test evaluation remain incomplete.


### Billing reconciliation 041

Dataset PID/263 completed successfully. Dataset PID/267 tile 2 failed after an interrupted response and remains in evaluation; tiles 3 and 6 recovered through built-in retries. Two newly eligible exact generation lookups returned HTTP 200: request `34727f5f76ca435d8e332617a634d2ae` billed $0.0007315 and `125d21ee188546d8bfb84094bc573121` billed $0.000493309, totaling $0.001224809. Evidence is `billing-reconciliation-041.json` and immutable `billing-snapshot-041.json`. No inference or older lookup was repeated.

At that snapshot, phase-two billed usage was $4.0824921388 across 8,095 billed requests, active reservations were $0.23464672, and unresolved upper bounds were $17.12921056. Including prior user-reported $0.64 gives $4.7224921388 known cumulative charges and $22.0863494188 cumulative exposure. Exposure is not billed cost; these are live-study snapshot values, and raw unresolved counts include pending requests. The second broader comparison, final selection and untouched final evaluation remain incomplete.


### Billing reconciliation 042

Exact generation lookup for previously eligible request `c6860c35b616414fbbf92db0612a0c45` returned HTTP 200, establishing $0.001200729 billed. Evidence is `billing-reconciliation-042.json` and immutable `billing-snapshot-042.json`. No inference or older lookup was repeated. Dataset PID/267 tile 7 recovered through a built-in retry; the comparison remains live.

At that snapshot, phase-two billed usage was $4.0861058014 across 8,099 billed requests, active reservations were $0.23464672, and unresolved upper bounds were $17.36385728. Including prior user-reported $0.64 gives $4.7261058014 known cumulative charges and $22.3246098014 cumulative exposure. Exposure is not billed cost; these are live-study snapshot values, and raw unresolved counts include pending requests. The second broader comparison, final selection and untouched final evaluation remain incomplete.


### Billing reconciliation 043

Dataset PID/287 tiles 4 and 7 recovered after upstream 429 responses through the existing cooldown/retry policy. A subsequent 429 raised cooldown to 60 seconds. Exact generation lookups for requests `a0b86642203240e9964c044ecf07de7a` and `f0410ba73c7248449cf0b5d2b6f2e393` both returned HTTP 404; their costs remain unresolved and reservations are retained. Evidence is `billing-reconciliation-043.json` and immutable `billing-snapshot-043.json`. No inference, old lookup, model routing or retry policy was changed.

At that snapshot, phase-two billed usage was $4.1111749018 across 8,122 billed requests, active reservations were $0.23464672, and unresolved upper bounds were $17.83315072. Including prior user-reported $0.64 gives $4.7511749018 known cumulative charges and $22.8189723418 cumulative exposure. Exposure is not billed cost; these are live-study snapshot values, and raw unresolved counts include pending requests. The second broader comparison, final selection and untouched final evaluation remain incomplete.


### Billing reconciliation 044

Dataset PID/287 tile 8 failed and remains in scoring. Exact generation lookup for request `639f1d4a3cb84aed9a53742973786198` returned HTTP 404; its cost remains unresolved and reservation retained. Evidence is `billing-reconciliation-044.json` and immutable `billing-snapshot-044.json`. No inference, older lookup, model routing or retry policy was changed. The same session 4848 remains live.

At that snapshot, phase-two billed usage was $4.1133256210 across 8,124 billed requests, active reservations were $0.00000000, and unresolved upper bounds were $18.06779744. Including prior user-reported $0.64 gives $4.7533256210 known cumulative charges and $22.8211230610 cumulative exposure. Exposure is not billed cost; these are live-study snapshot values. The second broader comparison, final selection and untouched final evaluation remain incomplete.


### Billing reconciliation 045

Dataset PID/292 tiles 7 and 8 failed and remain in scoring; tile 8 encountered a provider 429 and the existing cooldown/retry policy. Exact generation lookups for requests `1b97221f21514eec8cbb6fb421c00c73` and `9d537c941288436cb2a3cf700af06615` both returned HTTP 404; costs remain unresolved and reservations retained. Evidence is `billing-reconciliation-045.json` and immutable `billing-snapshot-045.json`. No inference or older lookup was repeated, and the same session 4848 remains live.

At that snapshot, phase-two billed usage was $4.1314721018 across 8,147 billed requests, active reservations were $0.00000000, and unresolved upper bounds were $18.53709088. Including prior user-reported $0.64 gives $4.7714721018 known cumulative charges and $23.3085629818 cumulative exposure. Exposure is not billed cost; these are live-study snapshot values. The second broader comparison, final selection and untouched final evaluation remain incomplete.


### Billing reconciliation 046

Previously eligible exact generation lookup for request `8e5c6a9ae729451fab11d9ec50e01563` returned HTTP 404; its cost remains unresolved and reservation retained. Evidence is `billing-reconciliation-046.json` and immutable `billing-snapshot-046.json`. No inference or older lookup was repeated. Dataset PID/292 tiles 9–11 completed successfully; the same session 4848 remains live.

At that snapshot, phase-two billed usage was $4.1329663610 across 8,149 billed requests, active reservations were $0.00000000, and unresolved upper bounds were $18.77173760. Including prior user-reported $0.64 gives $4.7729663610 known cumulative charges and $23.5447039610 cumulative exposure. Exposure is not billed cost; these are live-study snapshot values. The second broader comparison, final selection and untouched final evaluation remain incomplete.


Billing reconciliation 047: two newly recorded generation IDs returned HTTP 404; both reservations remain unresolved. The snapshot records phase-2 actual charges of $4.228496835, active reservations of $0.23464672, and unresolved upper bounds of $19.71032448. Including the user-reported prior $0.64, known cumulative actual spending is $4.868496835 and cumulative exposure is $24.813468035, within the $60 authorization. These are point-in-time values, not a final bill. Evidence: `output/pid2graph-phase2/billing-reconciliation-047.json` and `billing-snapshot-047.json`.


Transport resume 002: session 4848 exited with the existing perception-failure limit after timeout/SSL connection failures on Dataset PID/326 tiles 22–24. All three failures remain in evaluation; tile 25 was untouched. There were no active billing reservations or newly eligible generation-ID lookups. Pricing remained valid, and a non-billed endpoint metadata request returned HTTP 200. The identical controller resumed as session 20922. All 1,324 prior tile records and the frozen plan were verified unchanged. Evidence: `output/pid2graph-phase2/hybrid-broad-v1/transport-resume-002.json`, its connectivity check, billing snapshot, and verification artifact. This resume does not complete the broader comparison.


Billing reconciliation 048: the generation lookup for timed-out Dataset PID/351 tile 22 returned HTTP 200 with an actual charge of $0.00076286. The tile remains failed in extraction metrics. The snapshot records phase-2 actual charges of $4.3576304754, active reservations of $0.23464672, and unresolved upper bounds of $20.88355808. Including the user-reported prior $0.64, known cumulative actual spending is $4.9976304754 and cumulative exposure is $26.1158352754. These are point-in-time values, not a final bill. Evidence: `output/pid2graph-phase2/billing-reconciliation-048.json` and `billing-snapshot-048.json`. Tile 23 also failed with a connection error; tile 24 recovered through the existing retry policy.


### Billing reconciliation 049

Dataset PID/395 tile 6 failed with `TimeoutError: stream exceeded request time budget`; tile 7 subsequently completed in the same live session 20922. The failed tile was preserved and not manually retried. Exact generation `gen-1790027684-HAIMdeH1R9ATWK9IM4Zv` for request `08deba8376ae4f02b63a3c4bbba1fe09` returned HTTP 200 with billed cost $0.001214869. Evidence: `output/pid2graph-phase2/billing-reconciliation-049.json` and immutable `billing-snapshot-049.json`.

At that snapshot, phase-2 billed usage was $4.4272662712 across 8,546 billed requests; active reservations were $0; unresolved upper exposure was $21.1182048 across 90 unresolved requests. Including the prior user-reported $0.64, known cumulative charges were $5.0672662712 and conservative cumulative exposure was $26.1854710712, below the $60 cap. These are point-in-time values; unknown requests remain reserved and are not treated as free.


### Billing reconciliation 050

Dataset PID/395 tile 8 failed with `TimeoutError: stream exceeded request time budget`; tile 9 subsequently completed in the same live session 20922. The failed tile was preserved and not manually retried. Exact generation `gen-1790027969-uiaHwylMRn3smsSh7UOh` for request `81a5becdb75140969bde5eac4bb0892f` returned HTTP 200 with billed cost $0.001191349. Evidence: `output/pid2graph-phase2/billing-reconciliation-050.json` and immutable `billing-snapshot-050.json`.

At that snapshot, phase-2 billed usage was $4.4289031898 across 8,548 billed requests; active reservations were $0.23464672; unresolved upper exposure was $21.1182048, with 91 unresolved requests in the raw summary (including active requests). Including the prior user-reported $0.64, known cumulative charges were $5.0689031898 and conservative cumulative exposure was $26.4217547098, below the $60 cap. These are point-in-time values; unknown requests remain reserved and are not treated as free.


### Billing reconciliation 051

Dataset PID/429 tile 9 encountered an interrupted-stream RemoteProtocolError and the existing built-in retry, then failed with `TimeoutError: response exceeded request time budget`; tile 10 subsequently completed in the same live session 20922. The failed tile was preserved and not manually retried. Exact generation `gen-1790039162-30MQ64VqEv1GexCe6dqw` for request `28dfed5f8ce54becace2590925de867c` returned HTTP 200 with billed cost $0.000547489. Evidence: `output/pid2graph-phase2/billing-reconciliation-051.json` and immutable `billing-snapshot-051.json`.

At that snapshot, phase-2 billed usage was $4.5436900652 across 8,699 billed requests; active reservations were $0.23464672; unresolved upper exposure was $21.1182048, with 91 unresolved requests in the raw summary (including active requests). Including the prior user-reported $0.64, known cumulative charges were $5.1836900652 and conservative cumulative exposure was $26.5365415852, below the $60 cap. These are point-in-time values; unknown requests remain reserved and are not treated as free.


### Billing reconciliation 052

Dataset PID/429 tile 14 failed with `TimeoutError: stream exceeded request time budget`; tile 15 failed with `TimeoutError: response exceeded request time budget`. Tiles 16 and 17 subsequently completed in the same live session 20922. Failed tiles were preserved and not manually retried. Exact generation `gen-1790040007-WdgY7O2BmOQdeeaZBXMx` for request `26b20da558f344a6bafb8b1fe5ed2a6b` returned HTTP 200 with billed cost $0.000437729. No further finished unresolved request was eligible for exact-ID lookup after this reconciliation. Evidence: `output/pid2graph-phase2/billing-reconciliation-052.json` and immutable `billing-snapshot-052.json`.

At that snapshot, phase-2 billed usage was $4.5483461118 across 8,706 billed requests; active reservations were $0.23464672; unresolved upper exposure was $21.1182048, with 91 unresolved requests in the raw summary (including active requests). Including the prior user-reported $0.64, known cumulative charges were $5.1883461118 and conservative cumulative exposure was $26.5411976318, below the $60 cap. These are point-in-time values; unknown requests remain reserved and are not treated as free.


### Billing reconciliation 053

Dataset PID/450 finished all 25 tiles successfully. Before Dataset PID/452 tile 2, DeepInfra returned an upstream rate limit; the existing runner applied a 30-second cooldown and built-in retry. Tiles 2 and 3 then completed in live session 20922. Exact generation `gen-1790041358-LGKwW8ioKoeCPDy4HIDl` for request `183b048856094ee1842b42721cf57ea1` returned HTTP 404 at billing lookup. The charge remains unknown and its reservation was retained. Evidence: `output/pid2graph-phase2/billing-reconciliation-053.json` and immutable `billing-snapshot-053.json`.

At that snapshot, phase-2 billed usage was $4.5733428346 across 8,742 billed requests; active reservations were $0; unresolved upper exposure was $21.35285152 across 91 unresolved requests. Including the prior user-reported $0.64, known cumulative charges were $5.2133428346 and conservative cumulative exposure was $26.5661943546, below the $60 cap. These are point-in-time values; unknown requests remain reserved and are not treated as free.


### Billing reconciliation 054

Dataset PID/452 finished all 25 tiles successfully. Tiles 24 and 25 recovered from provider timeouts (14,778ms and 14,291ms) using the existing built-in retries, then drawing 456 started in live session 20922. Exact generation `gen-1790042104-c793x4T7krdLZ1UJ24eT` for request `43da09094c0e48f89ee5069f925f97bd` and generation `gen-1790042142-Y0qFX7RzmVcQDXh2HTu6` for request `8181b0c2bdd8462aa2f0d894a858e0e5` both returned HTTP 404 at billing lookup. Both charges remain unknown and their reservations were retained. Evidence: `output/pid2graph-phase2/billing-reconciliation-054.json` and immutable `billing-snapshot-054.json`.

At that snapshot, phase-2 billed usage was $4.5914590418 across 8,770 billed requests; active reservations were $0.23464672; unresolved upper exposure was $21.82214496, with 94 unresolved requests in the raw summary (including active requests). Including the prior user-reported $0.64, known cumulative charges were $5.2314590418 and conservative cumulative exposure was $27.2882507218, below the $60 cap. These are point-in-time values; unknown requests remain reserved and are not treated as free.


### Billing reconciliation 055

Dataset PID/457 tile 7 encountered an interrupted-stream RemoteProtocolError and the existing built-in retry, then failed with `APIConnectionError: Connection error.` Tiles 8 through 11 subsequently completed in live session 20922. The failed tile was preserved and not manually retried. Exact generation `gen-1790042967-x7RVjlHeYbGUaihGs2MP` for request `b5e1ea9fc6bd48e2b7f355e34e515d8a` returned HTTP 200 with billed cost $0.000679089. Evidence: `output/pid2graph-phase2/billing-reconciliation-055.json` and immutable `billing-snapshot-055.json`.

At that snapshot, phase-2 billed usage was $4.6116514596 across 8,800 billed requests; active reservations were $0; unresolved upper exposure was $21.82214496 across 93 unresolved requests. Including the prior user-reported $0.64, known cumulative charges were $5.2516514596 and conservative cumulative exposure was $27.0737964196, below the $60 cap. These are point-in-time values; unknown requests remain reserved and are not treated as free.


### Billing reconciliation 056

Dataset PID/58 tile 8 encountered an interrupted-stream RemoteProtocolError and the existing built-in retry, then failed with `APIConnectionError: Connection error.` Tiles 9 through 13 subsequently completed in live session 20922. The failed tile was preserved and not manually retried. Exact generation `gen-1790043672-KknmdVpFptT0Acy1WaZx` for request `b43f827211e54669933ce52ea04ba32f` returned HTTP 404 at billing lookup. The charge remains unknown and its reservation was retained. Evidence: `output/pid2graph-phase2/billing-reconciliation-056.json` and immutable `billing-snapshot-056.json`.

At that snapshot, phase-2 billed usage was $4.6316629588 across 8,828 billed requests; active reservations were $0; unresolved upper exposure was $22.05679168 across 94 unresolved requests. Including the prior user-reported $0.64, known cumulative charges were $5.2716629588 and conservative cumulative exposure was $27.3284546388, below the $60 cap. These are point-in-time values; unknown requests remain reserved and are not treated as free.

### Billing reconciliation 057

At Unix time 1790044920.913522, queried the one newly eligible finished unresolved request by its exact generation ID.
Request `4e820a1419c24e798469834552a145b9`, generation `gen-1790044713-xK2sSMIMTBICPgKb7g8Q`: HTTP 404. No charge established; the unresolved reservation remains, and this is not treated as free.

Point-in-time phase-2 actual billing: $4.6585692956; active reservations: $0.23464672; unresolved upper bound: $22.29143840. Raw unresolved request count: 96 (includes active requests). Including prior user-reported $0.64, known cumulative actual charges are $5.2985692956 and cumulative budget exposure is $27.8246544156, below the $60 authorization. Actual billing is incomplete.

Evidence: `output/pid2graph-phase2/billing-reconciliation-057.json` and `billing-snapshot-057.json`. Existing experiment continued unchanged; no failed tile was rerun.

### Billing reconciliation 058

At Unix time 1790045234.5578399, queried the one newly eligible finished unresolved request by its exact generation ID.
Request `ab7d48cf44f549c9a50b84d6eb156fdc`, generation `gen-1790045056-Z33FmBmiQg0HoQZLK5Xf`: HTTP 200; actual charge $0.00073038. Replaced this request's unresolved reservation with actual billing; the failed tile remains failed and was not rerun.

Point-in-time phase-2 actual billing: $4.6645712524; active reservations: $0.23464672; unresolved upper bound: $22.29143840. Raw unresolved request count: 96 (includes active requests). Including prior user-reported $0.64, known cumulative actual charges are $5.3045712524 and cumulative budget exposure is $27.8306563724, below the $60 authorization. Other unresolved requests remain reserved; actual billing is incomplete.

Evidence: `output/pid2graph-phase2/billing-reconciliation-058.json` and `billing-snapshot-058.json`. No inference configuration changed.

### Billing reconciliation 059

At Unix time 1790045962.530835, queried the newly eligible finished unresolved attempt by its exact generation ID.
Request `35e16941edae42f8932795d72b668576`, generation `gen-1790045823-E5433esqtz2nWjYfEixR`: HTTP 404. Actual charge remains unknown; retained the unresolved reservation. The existing provider cooldown and retry recovered the tile without changing the experiment.

Point-in-time phase-2 actual billing: $4.6820651820; active reservations: $0.00000000; unresolved upper bound: $22.76073184. Raw unresolved request count: 97. Including prior user-reported $0.64, known cumulative actual charges are $5.3220651820 and cumulative budget exposure is $28.0827970220, below the $60 authorization. Actual billing is incomplete; unknown charges are not treated as free.

Evidence: `output/pid2graph-phase2/billing-reconciliation-059.json` and `billing-snapshot-059.json`. No inference configuration changed.

### Billing reconciliation 060

At Unix time 1790046362.8400152, queried the newly eligible finished unresolved attempt by its exact generation ID.
Request `bfe8ac8cacce4ff5bbdca866f6b60afc`, generation `gen-1790046232-diWqwjsMNjDVROG2AVKN`: HTTP 404. Actual charge remains unknown; retained the unresolved reservation. Dataset PID/8 tile 17 remains failed; no failed tile was rerun.

Point-in-time phase-2 actual billing: $4.6914850372; active reservations: $0.23464672; unresolved upper bound: $22.99537856. Raw unresolved request count: 99 (includes active requests). Including prior user-reported $0.64, known cumulative actual charges are $5.3314850372 and cumulative budget exposure is $28.5615103172, below the $60 authorization. Actual billing is incomplete; unknown charges are not treated as free.

Evidence: `output/pid2graph-phase2/billing-reconciliation-060.json` and `billing-snapshot-060.json`. No inference configuration changed.

### Billing reconciliation 061

At Unix time 1790047933.152719, queried the newly eligible finished unresolved attempt by its exact generation ID.
Request `2f0b1a4d5da04714b2d13c91ac88950d`, generation `gen-1790047843-NVty2eaQmvCpZ9E8so8n`: HTTP 404. Actual charge remains unknown; retained the unresolved reservation. PID2Graph Synthetic/103 tile 13 remains failed; no failed tile was rerun.

Point-in-time phase-2 actual billing: $4.7319352204; active reservations: $0.23464672; unresolved upper bound: $23.69931872. Raw unresolved request count: 102 (includes active requests). Including prior user-reported $0.64, known cumulative actual charges are $5.3719352204 and cumulative budget exposure is $29.3059006604, below the $60 authorization. Actual billing is incomplete; unknown charges are not treated as free.

Evidence: `output/pid2graph-phase2/billing-reconciliation-061.json` and `billing-snapshot-061.json`. No inference configuration changed.

### Billing reconciliation 062

At Unix time 1790050736.200827, queried the newly eligible finished unresolved attempt by its exact generation ID.
Request `292ac2cbb3954674a97db0e4567fbc6b`, generation `gen-1790050608-QXa4K8NpDCAhe2v6MZJ8`: HTTP 404. Actual charge remains unknown; retained the unresolved reservation. PID2Graph Synthetic/162 tile 8 remains failed; no failed tile was rerun.

Point-in-time phase-2 actual billing: $4.8224865528; active reservations: $0.23464672; unresolved upper bound: $24.63790560. Raw unresolved request count: 106 (includes active requests). Including prior user-reported $0.64, known cumulative actual charges are $5.4624865528 and cumulative budget exposure is $30.3350388728, below the $60 authorization. Actual billing is incomplete; unknown charges are not treated as free.

Evidence: `output/pid2graph-phase2/billing-reconciliation-062.json` and `billing-snapshot-062.json`. No inference configuration changed.

### Billing reconciliation 063

At Unix time 1790052907.515327, queried the newly eligible finished unresolved attempt by its exact generation ID.
Request `f0bc91362fba45b28b6f8c5636759b41`, generation `gen-1790052781-zcvsLbo6F2vxpwj8evrI`: HTTP 200; actual charge $0.000478049. Replaced this request's unresolved reservation with actual billing. Synthetic/232 tile 24 remains failed and was not rerun.

Point-in-time phase-2 actual billing: $4.8790797662; active reservations: $0.00000000; unresolved upper bound: $24.87255232. Raw unresolved request count: 106. Including prior user-reported $0.64, known cumulative actual charges are $5.5190797662 and cumulative budget exposure is $30.3916320862, below the $60 authorization. Other unresolved requests remain reserved; actual billing is incomplete.

Evidence: `output/pid2graph-phase2/billing-reconciliation-063.json` and `billing-snapshot-063.json`. No inference configuration changed.

### Billing reconciliation 064

At Unix time 1790053441.639759, queried the newly eligible finished unresolved attempt by its exact generation ID.
Request `5bdf8108431143a3828254a7b6aa13bf`, generation `gen-1790053335-oarcvNk7LmASqGN8a8hJ`: HTTP 404. Actual charge remains unknown; retained the unresolved reservation. Synthetic/246 tile 12 recovered using the unchanged built-in retry.

Point-in-time phase-2 actual billing: $4.8917695230; active reservations: $0.23464672; unresolved upper bound: $25.10719904. Raw unresolved request count: 108 (includes active requests). Including prior user-reported $0.64, known cumulative actual charges are $5.5317695230 and cumulative budget exposure is $30.8736152830, below the $60 authorization. Actual billing is incomplete; unknown charges are not treated as free.

Evidence: `output/pid2graph-phase2/billing-reconciliation-064.json` and `billing-snapshot-064.json`. No inference configuration changed.

### Billing reconciliation 065

At Unix time 1790057189.302586, queried the newly eligible finished unresolved attempt by its exact generation ID.
Request `1ab4e47ca5194fc994b4f994839d0cda`, generation `gen-1790057049-0ZLzj8IuDp5UvkBBc72v`: HTTP 404. Actual charge remains unknown; retained the unresolved reservation. Synthetic/89 tile 22 recovered using the unchanged built-in cooldown and retry.

Point-in-time phase-2 actual billing: $4.9719177470; active reservations: $0.23464672; unresolved upper bound: $26.04578592. Raw unresolved request count: 112 (includes active requests). Including prior user-reported $0.64, known cumulative actual charges are $5.6119177470 and cumulative budget exposure is $31.8923503870, below the $60 authorization. Actual billing is incomplete; unknown charges are not treated as free.

Evidence: `output/pid2graph-phase2/billing-reconciliation-065.json` and `billing-snapshot-065.json`. No inference configuration changed.

### Completed broader explicit-decision comparison

Both broader arms finished; session 20922 exited 0. Explicit decisions attempted all 2,675 tiles across 107 validation drawings: 2,619 succeeded and 56 failed, yielding 64 complete and 43 partial drawings. All failures remain in scoring. The original 16 pilot drawings are reused unchanged.

Explicit macro drawing F1 is 0.533958602, versus baseline 0.236487535, broad recognition 0.650309719, and detector reference 0.879981510. It improves 98 drawings over baseline and worsens nine. Its symbol totals are TP 5,559 / FP 2,292 / FN 3,700 (precision 0.708063, recall 0.600389); localization totals are 7,468 / 383 / 1,791. Conditional class accuracy is 5,557/7,468 = 0.744108 under the independent localization matching. Collection macro F1: Dataset PID 0.689834045; PID2Graph Synthetic 0.239059115. Runtime is 69,146.004 seconds including reused pilot runtime.

The hybrid retains 5,558/9,244 detector-correct truths (60.1255%), loses 3,686 (1,908 still localized with class errors; 1,778 without localization), and recovers one truth beyond the detector. The largest confusions are general→valve (967) and general→instrumentation (526). There are 856 recognized detector proposals absent from output. Geometry replay verified every saved tile and explains all 856 through ownership filtering; this does not establish why the VLM failed to recognize them in their owning views. Explicit rejection loses 187 detector-correct truths; unresolved responses lose 847; recognized-but-filtered proposals lose 691; retained outputs lose 1,961, mostly through class errors.

Eight detail examples from the lowest-F1 fully successful new drawing in each collection were visually inspected: Synthetic/246 and Dataset PID/320. They show tight boxes with wrong broad classes, paired-stroke misclassification, and an unlocalized tank. These are scoring-only annotated images, never inference inputs, and are not engineering approval or PDF accuracy claims.

Evidence: `output/pid2graph-phase2/hybrid-broad-v1/explicit-findings.json`, `explicit-retention.json`, `explicit-decision-outcomes.json`, `explicit-ownership-verification.json`, and `explicit-visual-review.json`. Detailed legend interpretation remains separate. No final test inputs have been opened.

### Validation-only final selection

The final-selection controller exited 0 after checking all three eligible VLM candidates against the frozen validation panel, source fingerprints, prediction/attempt coverage, and one immutable billing snapshot. It selected `broad_raster_recognition` with seal `0c0899ead0286f580923e0ca391f0585d69cc83df2fc4884ed98e7cb7e9a5a3c`. Candidate macro F1: broad recognition 0.650309719; explicit decisions 0.533958602; baseline 0.236487535. The detector reference scores 0.879981510 but is excluded from VLM-pipeline selection by the predeclared policy because it does not perform legend interpretation. This gap remains an important limitation.

The selection snapshot records phase-2 actual charges $4.9917024830, active reservations $0.00000000, and unresolved upper bound $26.28043264. Unknown charges remain reserved. Including prior user-reported $0.64, known cumulative actual charges are $5.6317024830 and total budget exposure is $31.9121351230. No newly eligible exact-generation lookups remained before sealing.

Only after sealing did final detector inference begin on the previously frozen, untouched 16-drawing test panel. The final inference uses epoch006, MPS, threshold 0.15, and the unchanged selected settings; no test results will be used to tune the method. Evidence: `output/pid2graph-phase2/final-selection-v1/selection.json`, `inputs.json`, `coverage.json`, `billing-snapshot.json`, and refreshed validation reports.

### Billing reconciliation 066

At Unix time 1790058863.0189838, request `f930ff57baf14307839a92896e0fa8da`, generation `gen-1790058738-Yw1lrfRVdJjDDlPZiViU`, returned HTTP200 with actual charge $0.00043414. Its unresolved reservation was replaced by the charge. The failed final-test tile remains failed; no repeat inference or tuning occurred.

Point-in-time phase-2 actual billing is $4.9981778390, active reservations $0.23464672, unresolved upper bound $26.28043264. Raw unresolved request count 113 includes active requests. With prior user-reported $0.64, known cumulative actual charges are $5.6381778390 and total exposure $32.1532571990. Actual billing remains incomplete. Evidence: `output/pid2graph-phase2/billing-reconciliation-066.json` and `billing-snapshot-066.json`.


### Billing reconciliation 067

One finished request (`d65dff6903154d94876e41a3281d625f`, generation `gen-1790059723-eBhGEl5cHpQbHfrh3Tr7`) returned HTTP 404 from generation lookup. No actual charge was inferred; its unresolved reservation remains. At this point-in-time snapshot, phase-2 actual billed usage was $5.0819051070, active reservations $0.46929344, and unresolved upper bounds $27.45366624. The raw unresolved request count was 119, including active requests. Including the prior user-reported $0.64, known cumulative actual usage was $5.7219051070 and conservative cumulative exposure $33.6448647870. These reservations are not bills. Evidence: `billing-reconciliation-067.json` and `billing-snapshot-067.json`.

### Selected PDF transfer in progress

The selected epoch-6 export and source-bound 5,550 × 4,044 render passed the preflight in `selected-pdf-transfer-preflight.json`. Live check session 41239 is processing all 25 tiles via the public extraction entry point with built-in ISA definitions prepared first. Saved broad observations stay in review candidates, explaining progress messages with zero graph-ready objects. Intermediate records include invalid engineering-kind/class combinations left unresolved by strict semantic validation. These are transfer limitations to inspect after completion; no semantic accuracy or completed integration claim is made. Both live jobs and current counters are recorded in `final-progress-1790060243.json` and `checkpoint.json`.
