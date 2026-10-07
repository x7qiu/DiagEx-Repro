# PID2Graph raster proposal detector

This experimental detector supplies locations and broad categories for VLM
inspection. Its predictions are review proposals, not confirmed engineering
objects. The validation figures below measure this component alone. They do not
establish an improvement in the complete VLM pipeline or in real PDF extraction.

## Phase 2 selected pipeline — final evaluation running

The six-epoch continuation is complete. Epoch 6 was selected from training
checkpoints by pilot-validation macro drawing F1. This selects a detector
candidate. The detector-assisted VLM checkpoint comparisons and both broader
hybrid comparisons are complete. Validation-only selection chose broad raster
recognition with epoch 6 proposals. The selected pipeline's once-only evaluation
on the previously untouched 16-drawing test panel is running; no final-test
accuracy is yet claimed. Baseline remains the application default.

The trained broad-hybrid pilot has since completed attempted coverage: macro
drawing F1 is 0.594506 versus 0.566602 with pilot weights, including seven failed
tiles. It retains 818 of the trained detector's 1,080 correct detections; 185
of the 262 lost symbols remain localized with a class mismatch. The trained
explicit-decision comparison reached macro F1 0.484952 versus 0.435388 with
pilot weights, including two failed tiles. It retains 643 of 1,080 correct
detector detections; 260 of the 437 losses remain localized with a class
mismatch. Both hybrids still trail detector-only broad-category accuracy;
this does not evaluate detailed legend semantics. These pilot results informed
checkpoint promotion; the completed broader comparison below determined final
method selection. Full analysis is in `docs/PID2GRAPH_PHASE2.md`.

### Completed broader pipeline comparison

All methods used the same 107 validation drawings. Failed tiles remain in the
score denominators; reused pilot predictions were preserved unchanged.

| Method | Macro drawing F1 | Failed tiles / attempted tiles |
|---|---:|---:|
| Existing VLM baseline | 0.236488 | 12 / 2675 |
| Broad-recognition hybrid, selected | 0.650310 | 30 / 2675 |
| Explicit-decision hybrid | 0.533959 | 56 / 2675 |
| Epoch 6 detector reference | 0.879982 | Not an API tile run |

The frozen selection rule compares complete VLM pipelines and keeps detector-only
performance as a reference. Selection seal:
`0c0899ead0286f580923e0ca391f0585d69cc83df2fc4884ed98e7cb7e9a5a3c`.

Broad recognition retains 7,382 of 9,244 detector-correct symbols (79.8572%);
explicit decisions retain 5,558 (60.1255%). Both lose useful detector proposals
and introduce class errors. The broad hybrid loses 1,862 detector-correct truths,
including 1,310 still localized with the wrong class. Explicit decisions lose
3,686, including 1,908 still localized with the wrong class. Geometry replay
explains all 455 broad and 856 explicit recognized-but-missing proposals through
tile ownership filtering. This identifies the filtering mechanism, not why the
VLM failed to recognize the symbol in its owning tile. Removing that filter has
not been evaluated and could add duplicates.

The selected hybrid improves on the VLM baseline but still trails the detector
on broad GraphML labels. This is not evidence that it improves detailed legend
semantics, real-PDF accuracy, or DEXPI correctness. Evidence is saved under
`output/pid2graph-phase2/hybrid-broad-v1` and `final-selection-v1`.

Training checkpoint: `output/pid2graph-phase2/detector-epochs-v1/epoch-006.pt`

SHA256: `b98615fb338f7684517e4407b7f489e5cea42308c6bd773536e3c518dba8e111`

Production-compatible export:
`output/pid2graph-phase2/checkpoint-export-epoch006/detector.pt`

Export SHA256: `7f27e06d3b0998adc598ae703e4278dc2817a68cf819360a6761595f2411fd3e`

The export preserves all 284 model tensors. Its production loader reproduced
all 1,319 predictions exactly on the 16 pilot validation drawings on MPS,
including coordinates, classes and scores. CPU forward predictions also match
on a crop from each collection. See
`output/pid2graph-phase2/checkpoint-export-epoch006-completion.json` for bound
artifact hashes. Compatibility is not a deployment or accuracy approval.

### Training coverage and stopping

Training continued the pilot weights with a fresh SGD optimizer. The frozen
schedule used batch size four, learning rate 0.001, momentum 0.9, weight decay
0.0005, gradient clipping at 10, 100 warmup steps and cosine decay to 0.0001.
Augmentation used horizontal flips with probability 0.5 and contrast factors
0.9–1.1. All prepared crops were presented once per epoch, with additional
background, pump and tank sampling. The drawing-group splits were preserved;
overlapping crops were not counted as additional drawings or physical symbols.

The completed run used 33,390 optimizer steps and 133,554 crop presentations,
including 11,730 background presentations. It covered all 19,543 prepared
crops, all 528 training drawing groups and all 45,021 physical symbols with
full bounding boxes. Unique class coverage was 14,858 general, 19,898 valve,
313 pump, 516 tank, 8,236 instrumentation and 1,200 arrow symbols. There were
no inlet/outlet training examples. Exposure reconstruction is recorded in
`output/pid2graph-phase2/training-exposure-snapshot-007.json`.

The schedule was fixed before training: at least two and at most six epochs,
with early stopping after two epochs lacking a 0.002 improvement over the
patience reference. The run reached the maximum of six epochs; it did not
early-stop. Pilot-validation macro F1 across epochs 0–6 was 0.663019, 0.776157,
0.814417, 0.815444, 0.826902, 0.837568 and 0.846651. The last checkpoint still
improved, so a convergence plateau is not established. Training time was
57,708.676 seconds, excluding validation.

### Component validation and transfer limits

Evaluation uses class-aware one-to-one matching at IoU 0.5. Macro F1 weights
drawings equally. The table compares raw detectors under the same frozen
inference settings; containment cleanup is not applied.

| Validation inputs | Weights | Precision | Recall | Micro F1 | Macro drawing F1 | TP / FP / FN |
|---|---|---:|---:|---:|---:|---:|
| 16 pilot drawings | Epoch 6 | 0.818802 | 0.995392 | 0.898502 | 0.846651 | 1080 / 239 / 5 |
| 107 broader drawings | Pilot | 0.686260 | 0.993628 | 0.811824 | 0.737536 | 9200 / 4206 / 59 |
| 107 broader drawings | Epoch 6 | 0.850884 | 0.998380 | 0.918750 | 0.879982 | 9244 / 1620 / 15 |

The broader panel contains 70 Dataset PID and 37 PID2Graph Synthetic drawings.
Sixteen pilot predictions were reused with unchanged provenance; 91 drawings
were newly inferred. Every drawing's F1 improved over the pilot weights.
Collection macro F1 is 0.936327 and 0.773381 respectively. Broader per-class F1
is general 0.862416, valve 0.945450, pump 0.874172, tank 0.737288,
instrumentation 0.983080 and arrow 0.914761. Inlet/outlet has no validation
support. Tank precision remains 0.587838, and Synthetic precision is 0.633648.
These are detector validation results, not held-out or complete-pipeline results.
See `output/pid2graph-phase2/detector-trained-broad-report.json` and its analysis,
findings and failure figures for the full evidence.

Real-PDF transfer has observed limitations. On unannotated public
`two-tanks.pdf` at default production rendering (5,550 × 4,044 pixels), the
candidate produced 388 proposals. All four pump-labelled crop examples enclosed
circular identification bubbles; tank-labelled examples included pipes,
instrument assemblies and pump symbols. The two large tanks lacked whole-symbol
boxes in the overview. The targeted inspection is qualitative, not an accuracy
sample. Artifacts are in
`output/pid2graph-phase2/pdf-proposals-epoch006-production/visual-review.json`.
VLM interpretation, legend review and symbol review remain required. Broad
GraphML classes do not establish detailed legend semantics or DEXPI correctness.

### Use the exported candidate on a new raster

```sh
PYTHONPATH=src .venv-cv/bin/python -m diagex.vision.raster_detector --image /absolute/path/new-drawing.png --checkpoint output/pid2graph-phase2/checkpoint-export-epoch006/detector.pt --checkpoint-sha256 7f27e06d3b0998adc598ae703e4278dc2817a68cf819360a6761595f2411fd3e --device mps --out output/new-drawing-proposals.json
```

Inference retains 768-pixel crops, stride 576, confidence threshold 0.15,
class-aware NMS IoU 0.4 and RPN score threshold zero. Output remains fallible
review proposals. Native PDF text and geometry continue through their existing
path. Full training, export, validation and PDF preparation commands are in
`docs/PID2GRAPH_PHASE2.md`; the sealed final panel must not be used for exploratory
inference. No paid API calls were needed for detector training or validation.

## Historical pilot weights and reproduction

Checkpoint: `output/pid2graph-development/detector-v2/step-1000.pt`

SHA256: `8d05717fd762edc88658c809a119e6a9d079777933807a6d99903944fb0ee2e1`

The checkpoint contains the final step of the predeclared 1,000-step training
schedule. No checkpoint was chosen using validation or test performance. The
exact training source is saved beside it as `source-detector.py`; its hash matches
the training configuration. Environment versions and initialization provenance
are in `detector-environment.txt` and `detector-training-provenance.json` under
`output/pid2graph-development`.

The model is torchvision Faster R-CNN with a MobileNet V3 Large FPN backbone,
initialized from the official COCO V1 weights. Its input transform uses 640 pixels.
It trained locally on an Apple M1 Pro through MPS, using Python 3.13, torch 2.14.0
and torchvision 0.29.0. The 1,000 steps took approximately 1,280 seconds.

Training used 2,640 crops from 528 frozen training drawings, with seeded horizontal
flips, batch size two and initial SGD learning rate 0.005. The prepared pool had
2,640 crops; the 1,000-step run presented only 2,000 of them (0.758 epochs). The
seeded sample order includes all 528 training drawings, but only a small subset of
each drawing is covered. This is a pilot, not exhaustive use of the training data.
Validation/test drawings
and their patches were excluded. A failed first training attempt revealed that
the default RPN cutoff could discard all background proposals in a negative
batch; the restarted run retained negative examples with an RPN score cutoff of
zero. Both the failure and its CPU/MPS diagnosis are preserved.

From the DiagEx directory:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-cv/bin/python -m eval.pid2graph.detector infer --checkpoint output/pid2graph-development/detector-v2/step-1000.pt --panel output/pid2graph-development/panel-v1.json --out output/new-detector-validation --split validation
```

To reproduce crop preparation and training into new output directories:

```sh
PYTHONPATH=src .venv-cv/bin/python -m eval.pid2graph.detector prepare --manifest output/pid2graph-development/manifest-v1.json --out output/reproduced-detector-data
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src .venv-cv/bin/python -m eval.pid2graph.detector train --training output/reproduced-detector-data/training.json --out output/reproduced-detector --steps 1000 --device mps
```

The current training and preparation functions match the archived training source;
only the inference function subsequently gained final-selection enforcement.
The archived source hash is
`0b4beabb56aa1709dbfa6319f33759fae284f3684a649cc13f6bc745b41b231c`.
Use the saved environment versions and seed for reproduction. MPS/device or
library differences can prevent bit-identical retraining; use the delivered
checkpoint and its checksum to reproduce the measured inference result.

Inference uses original raster coordinates, 768-pixel crops at stride 576,
confidence threshold 0.15 and class-aware NMS at IoU 0.4. The reserved test panel
requires a sealed selection; do not use it for exploratory inference.

## Historical pilot reuse on a new raster drawing

The optional `diagex.vision.raster_detector` module loads the weights once and
accepts arbitrary raster images without the benchmark manifest or GraphML.
From the DiagEx directory, substitute the new image path:

```sh
PYTHONPATH=src .venv-cv/bin/python -m diagex.vision.raster_detector --image /absolute/path/new-drawing.png --checkpoint output/pid2graph-development/detector-v2/step-1000.pt --checkpoint-sha256 8d05717fd762edc88658c809a119e6a9d079777933807a6d99903944fb0ee2e1 --device mps --out output/new-drawing-proposals.json
```

The output records the source-image hash, original image dimensions, original
pixel coordinates, weight and implementation hashes, and the requirement for
interpretation. Existing output files are not overwritten. Torch loads only when
constructing the detector; the ordinary DiagEx runtime does not require it.
The reusable runtime is recorded separately in `detector-reuse-environment.txt`;
the original training environment record remains unchanged.

All 1,747 proposals across the 16 validation drawings matched the benchmark
outputs exactly, including IDs, boxes, categories, confidence and disposition.
The parity run took 215.1 seconds and read neither GraphML nor test images.
Its report is `output/pid2graph-development/detector-production-parity/report.json`.
This establishes implementation parity only and does not establish transfer accuracy.

The existing Evidence v2 CLI can optionally supply these proposals to its VLM:

```sh
.venv/bin/diagex extract-pid /absolute/path/new-drawing.png --engine evidence-v2 --raster-proposals output/new-drawing-proposals.json
```

This experimental option uses the configured VLM and the existing legend stage.
It passes locations and broad classes as fallible search hints; they do not become
native candidates or confirmed symbols. Omit `--raster-proposals` to use the
baseline. Guided validation stopped after 156 of 400 tiles because repeated
provider failures threatened the final-test budget. Its promising early results
are incomplete and were excluded from selection. The option remains experimental
and is not recommended as a default. Source-ink filtering was selected from the
completed validation comparisons; see `docs/PID2GRAPH_RESULTS.md`.

The original proposal format requires a matching single-frame raster image,
fixed inspection, and deskew disabled. A separate PDF page format is now available
as described below; the image format still cannot be attached to a PDF. Native
PDF extraction continues through its existing text/geometry path.
The source hash, image dimensions, proposal bounds/classes/confidence and detector
weight hash are validated. Guided cache identity includes the complete proposal
payload and adapter implementation hash, so changing hints cannot reuse a baseline
or differently guided perception checkpoint. Persisted runs retain the proposals
and adapter hash in `raster-proposals.json`.

All 400 validation tile hint payloads matched the frozen benchmark exactly;
`raster-guidance-integration-check/report.json` records the comparison. All 16
reusable proposal artifacts also passed source/coordinate validation. Sixteen new
tests cover source binding, cache separation, baseline fallback, CLI forwarding,
preflight rejection and the production perception loop. These are software
integration checks, not additional extraction-accuracy experiments.

## PDF page proposals (phase 2, experimental)

`python -m diagex.vision.pdf_raster_guidance --pdf SOURCE.pdf --checkpoint
EXPORTED_CHECKPOINT.pt --checkpoint-sha256 SHA256 --device cpu --out proposals.json`
generates hints on the same rendered pages used by production extraction.
`--pages` accepts zero-based page indices and defaults to every page. The source
PDF remains untouched. The optional detector environment needs PyMuPDF and
python-dotenv in addition to its existing Torch/Pillow/Pydantic dependencies;
the checked local versions are PyMuPDF 1.28.0 and python-dotenv 1.2.3.

Pass the result to the existing `--raster-proposals` option with the original
PDF as input. The default generation settings match the CLI's render settings.
If a programmatic caller changes DPI, page dimensions or scan settings, it must
use the same settings for generation and extraction. Each artifact records the
source checksum, selected pages, renderer source hashes and exact page pixel
hashes. Hints are rejected if the actual rendered page differs. Unselected pages
use baseline/native perception. Existing native candidates retain their normal
interpretation path; hints never become native geometry.

The optional `--raster-symbol-mode broad_review` runs broad recognition plus
legend-conditioned semantic suggestions on eligible raster pages and stops for
review. It does not automatically approve graph nodes. Broad recognition is now
the validation-selected VLM method. The mode and PDF adapter remain experimental
while the final evaluation and selected production-resolution transfer check run.

`output/pid2graph-phase2/pdf-proposal-smoke-v1/verification.json` records a CPU
interface smoke with exported pilot weights on `two-tanks.pdf` at a reduced
1500-pixel page size. All 50 hints passed bounds and cross-environment pixel
identity checks. Visual inspection found oversized pipe/assembly/text boxes and
many small glyphs without proposals. This is not measured accuracy or final
transfer evidence. Final transfer requires the selected checkpoint, VLM and
production-resolution rendering. A vector-PDF software check separately preserved
496 text spans, 2067 paths and 15 native candidates. The expanded integration
suite passed 78 tests; no API calls or final-test inputs were used by these checks.

## Historical pilot component validation

All 16 frozen validation-panel drawings were evaluated. Matching uses class-aware,
one-to-one IoU 0.5. Macro F1 weights drawings equally; micro F1 pools symbols.

| Proposals | Precision | Recall | Micro F1 | Macro drawing F1 | TP / FP / FN |
|---|---:|---:|---:|---:|---:|
| Raw detector | 0.6148 | 0.9899 | 0.7585 | 0.6630 | 1074 / 673 / 11 |
| Containment cleanup | 0.6715 | 0.9889 | 0.7999 | 0.7028 | 1073 / 525 / 12 |

Raw inference took 221.7 seconds for the panel and made no paid API calls. The
cleanup removed smaller same-class fragments mostly contained in sufficiently
confident larger boxes. Its fixed thresholds and source are recorded in the
experiment registry. The cleanup has component evidence only; a downstream VLM
benefit has not been established.

Raw precision differs substantially by collection: 0.7488 for Dataset PID and
0.3128 for PID2Graph Synthetic. Failure overlays show partial-symbol boxes and
duplicate detections, especially in the second collection. Inspect
`detector-failure-figures/index.json` with the full JSON reports for examples.

## Limits and intended use

- Both measured collections are synthetic and share generator families across
  splits. These results do not qualify an unseen industrial project.
- The seven labels are general, valve, pump, tank, instrumentation, arrow and
  inlet/outlet. Training contains no inlet/outlet examples; that class is not
  supported by learned positive evidence. Pump and tank examples are scarce.
- GraphML does not validate detailed legend meanings, tags, flow direction,
  process service, valve actuation or DEXPI semantics. Those require separate
  VLM/source checks and engineering review.
- The phase-one VLM symbol schema has no separate arrow category. Its arrow score
  therefore reflects a representational limitation as well as extraction behavior;
  the scoring contract keeps those reference symbols in the denominator.
- Detector guidance must remain visibly fallible and must not fabricate native
  candidate IDs, PDF path evidence, or approval. Native vector-PDF extraction
  remains the existing text/geometry pipeline.
- The phase-one detector-guided pipeline was not selected because its validation run is
  incomplete. Final evaluation of the selected ink-filter method is tracked in
  `docs/PID2GRAPH_RESULTS.md`; this card does not claim a detector test result.
