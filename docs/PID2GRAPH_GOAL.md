# PID2Graph extraction development

Active goal: improve DiagEx legend and symbol extraction through measured VLM,
conventional-vision, and optionally supervised-learning experiments. The paused
P&ID Inspector is outside scope.

## Authorization and resources

- User authorized **$60 total OpenRouter API spending** on 2026-09-14.
- Shared ledger: `output/pid2graph-development/spending.json`. Every request must
  use the existing pre-dispatch reservation mechanism. No unmetered clients.
- Initial allocation: baseline $15, improvements $30, final $10, legend/PDF transfer $5.
  These are allocations within the $60 total, not additional authorizations.
- Local Apple M1 Pro, 32 GiB RAM. No cloud GPU rental authorized.
- VLM prices and their time-based maxima are recorded in `prices.json`.

## Evaluation contract

Audit all complete drawings and patch GraphML files before freezing. Preserve
source annotations. Resolve GraphML key names, never assume d1/d2 coordinate order.
Separate physical-symbol classes from connector/crossing/border/background nodes.
The dataset contains 750 synthetic plans and 12 previously exposed OPEN100 sheets.
OPEN100 stays together in development only. Synthetic splits are within generator
distributions, not proof of new-project generalization. Group identical images,
normalized annotation layouts, and conservatively detected near-duplicates.

Freeze training/validation/test groups and a deterministic budgeted evaluation
panel before inference. Panels are sampled by drawing ID, not prediction quality
or annotation density. Report both the full split and actual evaluation coverage.
No patch from another split may enter training. No test inference or scoring until
candidate selection is sealed. Do not reclassify or remove difficult truth after
seeing predictions. Keep source-only inference manifests separate from scoring data.

At IoU 0.5, report symbol precision, recall, F1, per-class counts and localization
independent of class. Rank predictions by confidence and match one-to-one. Missing
drawings/failed requests remain visible and do not reduce recall denominators.
Selection uses macro drawing F1; report precision/recall tradeoffs and per-domain
results, with cost/runtime tie-breakers. The symbol-review inventory includes VLM
proposals; graph-eligible detections must also be scored separately. A proposal is
not an accepted engineering object. GraphML cannot validate legend meanings, tags,
flow direction, or detailed DEXPI semantics.

## Experiments and completion

Baseline calls the existing DiagEx perception code with its current fixed workflow.
Inspect baseline failures before choosing up to five controlled improvements:
source-derived raster proposals, crop scale/coverage, VLM reinspection, and/or
train-only learned proposals. Preserve the existing VLM legend interpretation and
native PDF evidence. No default change without measured support; retain fallback.

After validation selection, evaluate the frozen candidate on the final panel.
Separately verify legend review → symbol review → graph building and representative
PDF overlays. Transfer inspection is qualitative without independent labels.
Deliver code, checkpoints/weights if trained, environment versions, commands,
metrics, failure overlays, integration decision and limitations. Negative results
are acceptable. The goal remains active until all required evidence exists.

## Progress

- Added read-only inventory/grouping and independent symbol scorer.
- Scorer adversarial checks cover duplicate predictions, wrong classes, missing
  detections, misleading matching IDs, malformed outputs, and GraphML key order.
- Inventory in progress; encountered reversed **background** boxes. These are
  preserved as non-symbol annotations; physical-symbol geometry is checked separately.
- Live inspection corrected the initial quarantine hypothesis: `perceive_tile`
  passes `candidates=None` for scanned/image inputs, allowing raster detections.
  Native-PDF unanchored proposals are quarantined. Both inventories remain reported.
- Full audit completed: 762 drawings, 35,799 patch graphs, no invalid physical-symbol
  boxes. All 762 NPY shift files require object unpickling; unused, not loaded.
- Frozen split: 528 train, 107 validation, 115 test, 12 OPEN100 development-exposed.
  Budgeted panels: 16 validation, 16 test, 4 train smoke, 2 OPEN100 development.
- Nine evaluation-contract checks passed. Transport/related checks: 44 passed.
- Transport preflight isolated OpenRouter rejection of effort + disabled reasoning.
  Fixed client to omit effort for disabled OpenRouter thinking; other transports
  and enabled reasoning retain their behavior. Checks saved in transport-checks.json.
- Fireworks-only smoke hit upstream rate limits. `prices-v3.json` permits the
  verified DeepSeek/DeepInfra/Fireworks endpoints and uses their maximum prices;
  Qwen remains Alibaba-only. All comparisons must record the same allowed providers.
- `baseline-smoke-v4` completed 25 tile attempts on training drawing Dataset PID/229:
  3 timed out, 51 predicted symbols, 24 correct matches among 93 reference symbols.
  Training-only ink guard removed 5 false positives without losing a true positive;
  NMS did not change this smoke. Not evidence of held-out improvement.
- Main baseline now runs with an explicit 120-second request allowance. Original
  application defaults are unchanged. Provider cooldown waits occur outside the
  individual request deadline; runtime includes those waits.
- Active process handle and exact resume command are in
  `output/pid2graph-development/checkpoint-initial.json` (exec session 43926 at creation).
  Revalidate that live handle before restarting. Results: `baseline-validation/`.
- Scoring/grouping/transport checks passed (44), plus 2 image-postprocessing checks.
  No final-test inference or candidate selection has occurred.
- Local supervised detector probe passed on MPS, including backward propagation
  and identical predictions after rebuilding/loading weights. Torchvision's
  pretrained model uses FrozenBatchNorm; inference reconstructs that architecture
  without requiring a new pretrained-weight download.
- Prepared 2,640 crops exclusively from 528 frozen training drawings. Crop targets
  are visually aligned. A split-exclusion test uses nonexistent validation/test
  paths to verify preparation never opens reserved sources. The local detector
  trains for a predeclared 1,000 steps; use the final checkpoint, with no validation
  checkpoint search. Environment and initialization-weight hash are saved.
- Registered two controlled hypotheses in `experiments-v1.json`: an image-only
  blank-box guard and supervised raster guidance for the existing VLM. The latter
  uses the existing page-context argument and explicitly labels detections as
  fallible hints, never native geometry or accepted engineering facts. Default
  production perception is unchanged. No validation improvement is claimed yet.
- Baseline session 43926 ended after three consecutive connection failures.
  Public HTTP connectivity recovered. Session 43323 resumes untouched tiles;
  cached failures and their charges remain in outputs and scoring. The circuit
  breaker now counts fresh failures within the resumed invocation so historical
  failures cannot permanently prevent processing the remaining tiles.
- Local training runs in session 99069. Revalidate live handles before restarting.
  All paid calls share the cumulative $60 ledger; approximately $6.66 was reserved
  or conservatively charged at this checkpoint, including failed requests.
- Fifty-three regression checks passed across training exclusion, image filtering,
  legend context, native legend rows, detection review, provenance, and inspection.
  Separate source-coordinate guidance and manifest checks also passed. A browser
  smoke passed legend review, symbol edits, persistence, source overlays and the
  reviewed-input handoff to a stub graph runner. This checks UI flow, not actual
  extraction or graph accuracy. PDF transfer and actual graph checks remain due.
- Detector v1 stopped at step 644 on an all-negative batch. The same batch produced
  nonfinite classifier loss on both CPU and MPS because the MobileNet detector's
  default RPN cutoff removed every background ROI. Setting that cutoff to zero
  retained background examples and produced finite loss and gradients. V2 restarted
  from original pretrained weights with the same data/seed/learning-rate/1,000-step
  plan; it has passed the failed batch. Use `detector-v2/step-1000.pt` after completion.
  Its exact training source snapshot is saved and matches its recorded SHA256.
- Baseline session 43323 later ended after another connection outage. Fresh HTTP
  connectivity succeeded; session 48809 resumes untouched tiles. The evaluation
  runner now rebuilds a broken SDK connection after transport errors while keeping
  failed outputs, spending reservations and provider cooldown state.
- Added selection sealing and final inference signatures for model/weights,
  thresholds, source code and perception settings. Selection requires all validation
  drawings attempted; partial drawings retain failed-tile denominators. No selection
  has been sealed and no final-test inference has occurred.
- Scoring now attributes shared-ledger charges by recorded request IDs and category,
  excluding concurrent legend requests from symbol-run cost without releasing any
  reservations. Sixteen evaluation-contract tests passed, including this case.
- Fresh legend check: 13 printed definitions from the first 2401 legend page,
  with native text/row geometry regenerated from the PDF. Baseline broad source
  consistency was 11/13. Experimental caption guidance gave 12/13, but **did not fix**
  the generic-valve row being rejected as a heading; the change came from a throttle
  row changing from uncertain to accepted. This single paired run is inconclusive;
  no default prompt change. See `legend-comparison-v1.json` and `experiments-v3.json`.
- Fresh native PDF smoke completed four selected tile calls across 2401 page 4 and
  a Covestro sheet, then ran actual fusion and deterministic topology. Source text
  and vector paths remain present. Results and inspected overlays are recorded in
  `pdf-transfer-review.json`. This is partial, qualitative transfer evidence, not
  full-sheet recall or engineering validation. The final integrated review-to-graph
  workflow remains due after selection.
- Detector v2 completed all 1,000 planned training steps in 1,280 seconds. Weights
  and checksum are saved at `detector-v2/step-1000.pt`. Source-only inference over
  the 16-image validation panel is active in session 2920. The detector is still
  experimental; its training loss is not evidence of validation improvement.

## Reproduction commands

Current checkpoint (2026-09-14):

- Detector validation completed all 16 frozen drawings: precision 0.6148, recall
  0.9899, macro drawing F1 0.6630. Same-class containment cleanup removed 148 false
  positives and one true positive, giving precision 0.6715, recall 0.9889 and macro
  F1 0.7028. These are proposal-component metrics, not VLM pipeline improvements.
- Registered experiment 4 for that fixed containment rule before scoring it, based
  on validation failure overlays and a train-only nesting audit. Registered
  experiment 5 for a Qwen model swap with otherwise unchanged symbol workflow.
  The current registry is `experiments-v5.json`; no sixth experiment is planned.
- Qwen's Alibaba route rejected forced tool calls. `prices-v4.json` records the
  compatible Makora endpoint and its verified ceiling; its two-tile training-only
  smoke passed. DeepSeek's existing price entry is unchanged. Qwen validation is
  running with the same frozen panel and scoring contract.
- DeepSeek's baseline and detector-guided runs still encounter upstream 429s and
  intermittent transport failures. Short subprocess batches preserve cooldowns
  across restarts; this has not eliminated the failures. Cached failures remain
  misses, and uncertain requests retain their full spending reservations.
- Shared category allocations are now baseline $22, experiments $22, final $14,
  legend/PDF transfer $2, within the same cumulative $60 authorization. The shared
  ledger, not these notes, is authoritative for current charges and reservations.
- Added `eval.pid2graph.workflow_check` to exercise agent legend/symbol review,
  immutable snapshots and the real graph pipeline on a controlled PDF fixture.
  This complements the browser component smoke; it is software verification and
  does not represent human engineering review or extraction accuracy.
- The real workflow fixture uncovered a raster fallback crash: this OpenCV build
  returns Hough lines as `(N, 4)` instead of `(N, 1, 4)`. Normalizing both layouts
  fixes it; three regressions passed, including real installed OpenCV inference.
  `real-graph-workflow-v3/report.json` records the completed real pipeline:
  two reviewed nodes, one connection, unchanged reviewed boxes/labels and agent
  provenance. It retains a partial-quality flag; this is not a claim of complete
  graph accuracy. Symbol-inference fingerprints were not changed by this fix.
- Qwen model-swap validation was stopped after the first two completed drawings
  scored zero localization matches and failure overlays showed displaced boxes.
  The stopped output contains three drawing summaries (186 predictions, zero
  matches) and nine attempted tiles of the fourth. Its full-panel report retains
  missing denominators, and the incomplete arm is excluded from selection. The
  successful tool-call smoke did not establish coordinate compatibility. Keep
  `qwen-validation/stop-request.json`; do not restart this unchanged arm.
- Added an outer 270-second evaluation guard, allowing two 120-second format
  passes plus cleanup. A guided stream took 360.48 seconds before yielding an
  event that exposed its expired deadline; socket timeouts alone were insufficient.
  That failure remains recorded. The recovery signal arrived after the following
  tile began, so that interrupted tile also has an explicit failed checkpoint and
  retained reservation. Two wall-clock guard regressions passed. Core inference
  payloads/settings remain unchanged; this is an operational evaluation safeguard.
- Broader topology and detection-review regression run after the OpenCV fix:
  41 tests passed. No default VLM or detector integration has been selected yet.
- The stopped Qwen report was finalized to include its fourth drawing's nine
  attempted tiles and their cost, rather than silently dropping that unfinished
  case. `qwen-stopped-validation-report-v2.json` contains 215 predictions and zero
  matches, with 12 entirely unattempted drawings still missing. The selection
  guard explicitly rejects stopped arms. `experiments-v6.json` is the progress
  registry; it still contains exactly five hypotheses.
- Billing reconciliation remains inconclusive: official OpenRouter guidance
  acknowledges some charged 429/partial-output cases; prior generation lookups
  returned 404 and the browser Activity page requires sign-in. No unknown-cost
  reservations were released. See `billing-evidence-review.json` before repeating
  these checks. The total cap remains $60.
- Current allocations: baseline $25, experiments $20, final $14, legend/PDF $1.
  This reallocates within $60 after the model-swap stop and completed transfer
  checks. Both DeepSeek runs later hit a shared connection outage. Read-only HTTP
  probes recovered through the configured local proxy; causality is not proven.
  The baseline now runs alone before the guided arm resumes. Connection resets
  also schedule a 30-second wait, preserving any longer provider cooldown.
- Added prospective proof for unsent final attempts. The evaluation runner
  disables redirects and hidden SDK retries, captures exact reservation IDs, and
  recognizes only typed HTTPX connection-establishment failures at the original
  POST endpoint. Such a final attempt can be recorded as not dispatched, keeping
  its original reservation and proof in the ledger while counting its cost as zero.
  Earlier retries, HTTP errors and interrupted responses remain reserved. No
  historical unknown-cost reservations were modified. Fourteen focused checks
  passed; see `connection-accounting-policy.json` for the precise contract.
- Detector weights, training recipe, component performance and domain/schema
  limitations are documented in `docs/PID2GRAPH_DETECTOR.md`.

Run from `DiagEx-Repro`; use new output paths for audits, manifests, and reports.
Inference can resume only an identical configuration and source fingerprint.

```sh
.venv/bin/python -m eval.pid2graph audit --root ../testdata/PID2Graph --out output/new-audit
.venv/bin/python -m eval.pid2graph freeze --audit output/new-audit/inventory.json --out output/new-manifest.json
.venv/bin/python -m eval.pid2graph prepare --manifest output/new-manifest.json --out output/new-panel.json
.venv/bin/python -m eval.pid2graph run --panel output/pid2graph-development/panel-v1.json --out output/pid2graph-development/baseline-validation --ledger output/pid2graph-development/spending.json --prices output/pid2graph-development/prices-v3.json --split validation --request-timeout 120
.venv/bin/python -m eval.pid2graph score --manifest output/pid2graph-development/manifest-v1.json --panel output/pid2graph-development/panel-v1.json --run output/pid2graph-development/baseline-validation --out output/pid2graph-development/baseline-validation-report.json --split validation
```

Do not reinitialize the spending ledger. The $60 authorization is cumulative across
all goal turns, smoke requests, failures, comparisons, and final evaluation. Refresh
expired provider price verification before further inference, retaining earlier snapshots.

Local supervised experiment commands (use existing outputs/handles when live):

```sh
.venv-cv/bin/python -m eval.pid2graph.detector prepare --manifest output/pid2graph-development/manifest-v1.json --out output/pid2graph-development/detector-data-v1
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-cv/bin/python -m eval.pid2graph.detector train --training output/pid2graph-development/detector-data-v1/training.json --out output/pid2graph-development/detector-v2 --steps 1000 --device mps
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-cv/bin/python -m eval.pid2graph.detector infer --checkpoint output/pid2graph-development/detector-v2/step-1000.pt --panel output/pid2graph-development/panel-v1.json --out output/pid2graph-development/detector-validation --split validation
.venv/bin/python -m eval.pid2graph run --panel output/pid2graph-development/panel-v1.json --out output/pid2graph-development/guided-validation --ledger output/pid2graph-development/spending.json --prices output/pid2graph-development/prices-v3.json --variant supervised_guidance --proposal-run output/pid2graph-development/detector-validation --split validation --request-timeout 120
.venv/bin/python -m eval.pid2graph.postprocess --panel output/pid2graph-development/panel-v1.json --source output/pid2graph-development/baseline-validation --out output/pid2graph-development/ink-guard-validation-v2 --variant ink_guard
```


### Reusable detector parity and budget-stop checkpoint (2026-09-14)

The reusable `diagex.vision.raster_detector` implementation produced exactly the same 1,747 proposals on all 16 validation drawings as the frozen benchmark, with zero API calls and no GraphML/test-image reads. Its report is `detector-production-parity/report.json`; the model card includes the new-image CLI and separate reuse environment. This establishes implementation parity only; default VLM integration remains pending.

Baseline session31992 stopped at its internal25 allocation after11 completed drawing summaries. Reallocated within the authorized60 to baseline28, experiments17, final14, legend transfer1, under the shared lock; no reservations were released. Resumed baseline as session48587. Guided70631 remains stopped pending baseline completion. Provider429 responses continue, with persisted cooldown.

Fixed an operational gap in the evaluation runner: a budget exception after an already reserved attempt now saves that tile as failed, including its exact request IDs and cost, before stopping. If no attempt was reserved, the untouched tile remains resumable. Sixteen budget-resume/connection/accounting tests passed and Ruff passed. No benchmark perception configuration or prior tile results changed. `budget-resume-audit.json` distinguishes category totals from checkpoint-associated costs; historical interrupted-call overhead remains charged/reserved even when it cannot be reliably assigned to a drawing. Full validation, sealed selection, final evaluation, and the supported integration decision remain due.


### Optional raster guidance in the existing VLM stage (2026-09-14)

Added experimental `diagex extract-pid --engine evidence-v2 --raster-proposals FILE.json`. It binds local detector proposals to the exact raster hash/dimensions, validates coordinates and broad review-only classes, and passes the frozen fallible-hint payload to the existing VLM stage after legend processing. Hints never become native candidate IDs or confirmed detections. Omission retains the baseline, and guided cache identity includes hints and adapter code. The option requires fixed inspection and deskew disabled; PDF extraction remains its native path. Persisted runs retain source-bound hints and the adapter hash.

All400 validation-tile hint payloads exactly match the frozen benchmark; all16 reusable detector artifacts passed validation without GraphML/test-image reads or API calls. Sixteen new integration tests passed, and89 existing CLI/evidence/response-contract tests passed. The initial two failures were blank-fixture page-role setup and were corrected explicitly; no extraction logic was weakened. New code lint passes; the CLI retains pre-existing Typer/default and exception-chaining findings. Evidence and source hashes are recorded in `raster-guidance-integration-check/integration.json`.

This is software integration preparation for the existing supervised-guidance experiment, not a sixth experiment or a declaration of extraction improvement. Default enablement and the supported integration decision remain pending full validation and sealed final evaluation. Baseline48587 remains live; guided70631 is still held until baseline finishes.


### Full baseline and ink-filter validation (2026-09-14)

All16 frozen validation-panel drawings and400tiles have been attempted:371tiles succeeded and29failed. All1085 physical reference symbols remain in the denominator. Five drawings had no failed tile; eleven are partial because of recorded failures. Reports: `baseline-validation-report.json`, `ink-guard-validation-report.json`, and `baseline-analysis.json`.

| Method | TP / FP / FN | Precision | Recall | Micro F1 | Macro drawing F1 |
|---|---|---:|---:|---:|---:|
| Existing VLM baseline | 220 / 439 / 865 | 0.33384 | 0.20276 | 0.25229 | 0.22110 |
| Fixed source-ink filter | 220 / 398 / 865 | 0.35599 | 0.20276 | 0.25837 | 0.22512 |

The filter removed41false positives without losing a class-aware match, improving8/16drawings and leaving the others unchanged. This modest precision improvement does not solve recall. Baseline localization alone matches341symbols;219of those localization matches have the correct broad class. The most frequent conditional confusion is general→valve(54), followed by general→instrumentation(27). Dataset PID macroF1 is0.27862; PID2Graph Synthetic is0.16359.

Three scoring-side detail sheets were visually inspected (`baseline-failure-figures/index.json`): several predicted boxes cover the right glyph but have the wrong broad category, some surround excessive nearby piping/text, and some symbols have no prediction. These annotated figures are never inference inputs. Detailed engineering semantics remain outside the GraphML contract.

Recorded baseline runtime is8251.91seconds, including saved processing/cooldown durations, with25.9264USD charged/reserved linked to drawing checkpoints. The baseline category total28.9769USD also includes smoke and interrupted-call overhead. These are conservative reservations/price ceilings, not invoices. A new authenticated read-only `GET /api/v1/key` snapshot reports0.33052255USD key-wide usage for the current UTC day; it is not exact per-goal/per-request attribution and releases no reservations (`key-usage-probe.json`). The live budget-stop regression preserved the attempted tile9 failure and its0.3217728USD reservation before resuming at tile10 (`budget-stop-live-check.json`).

Baseline58545 and scorer60946 completed. Guided validation resumed as45611 using unchanged detector hints, model, source inputs and scoring. Current allocations are baseline29.5, experiments16.3, final14, legend_transfer0.2, total60. Registryv7 updates evidence only. No winner has been sealed and the final test remains untouched.


### Consolidated report and prospective price guard (2026-09-14)

Created `docs/PID2GRAPH_RESULTS.md` with dataset/scoring scope, full baseline/per-class results, five experiment statuses, detector component evidence, optional integration, separate legend/PDF/workflow checks, cost interpretation, reproduction command and explicit remaining gates. Guided45611 remains live; its current invocation uses original pricesv3.

Fresh official endpoint metadata confirms the allowed named-tool-compatible endpoints are DeepInfra and Fireworks, priced at most0.22USD/M input and0.66USD/M output. Direct DeepSeek remains incompatible with forced named-tool choice. Prepared pricesv5 with these server-enforced max_price caps and unchanged provider tags; the currently eligible provider set is unchanged. Reservation for a6000-output-token call falls from0.3217728 to0.23464672 prospectively. No historical reservations are released. The price filter changes explicitly and future repricing can affect availability; this operational change must remain visible in final reporting. Do not activate by interrupting a paid call: usev5 at the next natural driver resume.

New tile records retain a price-snapshot hash and exact archived file. The selection CLI requires --prices and seals final price limits; test inference checks them before API client creation. Identical re-verification may refresh dates without changing limits. Five focused budget-resume/selection tests and Ruff pass. A misplaced test assertion was corrected before the passing final run. No actual test-panel image or truth was read. See `pricing-guard-revision.json` and `deepseek-endpoints-price-review.json`.


### Price guard activation and four-drawing guided comparison (2026-09-14)

Guided45611 terminated at its internal allocation after saving the attempted tile16 failure. Activated pricesv5 on the new guided driver61895. All116oldertile hashes remain unchanged; new live requests reserve0.23464672USD (`pricing-guard-live-check.json`). Under the shared ledger lock, reallocated to baseline28.99, experiments20.51, final10.3, legend transfer0.2, still60total. Final10.3 exceeds14times the maximum new/old per-token price ratio0.7333, retaining at least the earlier final request capacity. No historical reservations were released.

The first four completed paired drawings, all from Dataset PID, improve from baseline macroF1 0.30882565 toguided0.58028475. TP increases103to199, FP falls163to73 and recall rises0.26010to0.50253; eachdrawingimproves. Failed tiles remain in the denominators. This is preliminary one-collection evidence only: `guided-paired-interim-4.json` is explicitly ineligible for selection. The complete16-drawing guided comparison and full final test remain due. Updated the consolidated results report with these limits and active pricing provenance.


### Guided stop, sealed selection and final-test start

Guided driver 61895 exited after the requested tile-boundary stop: 156/400 tiles,
six fully attempted drawings and one partial. Repeated 429s continued after the
five-minute quiet interval. The arm is explicitly stopped and ineligible; its
full-denominator report is guided-stopped-validation-report.json. No uncertainty
holds were released and no source/inference settings were changed.

selection-v1.json seals ink_guard from the two eligible full-validation reports.
The final baseline-inference driver is session 59712; apply frozen ink_guard
postprocessing with that seal after inference. The original final allocation
remains $10.30 and total authorization $60. No test tuning is permitted.
Optional production --raster-ink-filter is being verified, with default false,
source-pixel rejection audit, separate cache and unchanged baseline fallback.


### Final-test operational cooldown

Repeated upstream429 responses continued during the second final drawing. The
outer final driver41584/session59712 is held while its current bounded child
finishes; controller92877 then enforces600seconds with no new requests and resumes
the same verified driver. Event: final-rate-limit-pause-v1.json. Source images,
model/provider price guards, prompts and scoring remain frozen. The intentional
outer-driver pause is recorded separately from tile processing runtime. No failed
records are retried or billing reservations released.


### Remaining allowance assigned to final evaluation

The shared ledger now allocates baseline28.98, experiments18.92, legend0.15 and
final11.95, totaling the original authorized60. All non-final paid arms are closed;
unused allowance is available to the same live final driver59712 without a restart.
No historical holds were released and no inference/provider settings changed.
See final-allocation-revision.json. Further substantial funding would require a
new user authorization; the remaining rounding slack is under two cents.

### Final evaluation finished and scored

Driver session 59712 exited zero after all 16 drawings/400 tiles: 386 successful, 14 failed. Selection/configuration/pricing and source hashes verified before scoring. Frozen ink filtering removed 28 FP and no TP on the final panel; macro drawing F1 0.2206831254 to 0.2237758886, recall 0.2227314390. Three final failure detail sheets inspected; all three cases had successful API coverage, with class and localization/miss errors remaining. Optional ink integration retained, baseline default preserved; detector guidance remains experimental. Ledger charges plus reservations total $58.00939082 under the $60 cap. See docs/PID2GRAPH_RESULTS.md and final artifacts; final completion audit follows.

Final requirement audit passed: all 762 complete source pairs and 2640 crop hashes unchanged; training/model source and delivered weights verified; all final coverage and report artifacts present. Completion evidence is in completion-audit-final.json. No live paid evaluation process remains.
