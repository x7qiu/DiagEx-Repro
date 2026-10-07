# Independently improvable extraction stages

The refactor preserves the existing extraction behavior and adds reusable stage
boundaries. It does not train new weights, change the frozen PID2Graph experiment,
start a model server, or enable extra model calls in the full extraction workflow.

## Responsibilities

| Responsibility | Implementation | Inputs | Outputs |
|---|---|---|---|
| Symbol detection | `vision/symbol_detection.py` | Page geometry; optional rendered image and CV detector | Native candidates and immutable CV proposals |
| Text detection/recognition | `vision/text_detection.py` | Native PDF words or an explicit text recognizer | Positioned literal text with its actual origin |
| Raster text backend | `vision/text_recognition_vlm.py` | Image, injected model client | Literal text and boxes; no symbol assignment |
| Line detection | `vision/line_detection.py` | PDF paths and optional page image | Paths and visual evidence; no graph endpoints |
| Symbol interpretation | `vision/symbol_interpretation.py`, `raster_pipeline.py`, `raster_semantics.py` | Candidates, source views, prepared legend | Classification, interpretation, and unresolved/rejected decisions |
| Text assignment | `vision/text_assignment.py` | Symbol inventory, positioned text, equipment scopes, legend | Copied/enriched nodes, bindings, conflicts, source IDs |
| Connection inference | `vision/topology.py`, `connection_inference.py` | Symbols, saved lines, legend and visual assessments | Geometric candidates and validated semantic edges |

`perception.py` and `page_graph.py` remain compatibility imports. Existing callers
and monkeypatches address the same implementation modules, not duplicated copies.
The CV detector is still loaded separately from its exported weights. Its public
protocol accepts pixels and returns proposals; it has no model-client dependency.
All language-model backends receive a client from their caller. A serving engine
such as vLLM would be a transport adapter, not a symbol or connection algorithm.
No vLLM server or new serving transport is introduced by this refactor.

The full workflow retains its existing perception, contextual-resolution,
and assembly logic. Symbol interpretation still reads printed tags when required
by its original prompt; separating OCR into a new backend does not silently remove
that ability. Dedicated VLM text recognition is opt-in through the stage runner.
It has software tests, not a measured real-PDF OCR accuracy claim.

## Stage execution

### Optional CV in the web workbench

The **Symbol detection** selector offers the existing VLM/native-evidence route
(default) and **AI model + trained symbol detector (experimental)**. Enabling CV reveals a local detector
checkpoint path and a CPU/MPS/CUDA device selector. The server generates
source-bound proposals for the uploaded PDF or image before invoking the VLM,
then uses the existing `broad_review` interpretation route. Validated legend-supported classifications continue into normal fusion and graph construction.
Unresolved observations stay in the machine audit and cannot become nodes automatically.
This enables the tested hybrid; it does not implement a new fusion algorithm.

Set these in the environment used to start the workbench, if needed:

```bash
export DIAGEX_CV_CHECKPOINT=/absolute/path/to/detector.pt
export DIAGEX_CV_PYTHON=/absolute/path/to/torch-environment/bin/python
```

The checkpoint can also be entered in the form. Without an environment override,
the standard exported checkpoint under `output/pid2graph-phase2/` is used when
present. The worker defaults to the server's Python; `DIAGEX_CV_PYTHON` allows the
existing separate Torch environment to be reused without adding Torch to the
web environment. The CV runtime needs Torch, Torchvision, Pillow, PyMuPDF,
Pydantic, NumPy and python-dotenv. No packages or model weights are downloaded by
this option.

CV failures stop the job before VLM extraction. The run records the selected
checkpoint hash, device and proposal path in its workbench metadata. Proposal
files are retained under the workbench storage directory's `cv/JOB_ID/` folder.
CV proposals are regenerated for each new web job; normal compatible VLM
checkpoints can still be resumed.
### Independent commands

After activating an environment with DiagEx dependencies, run from this checkout:

```bash
PYTHONPATH=src python -m diagex.cli stage-list
PYTHONPATH=src python -m diagex.cli stage-run request.json --out output/stages
```

Requests have this envelope (the concrete `inputs` schema depends on the stage):

```json
{
  "schema_version": "1.0",
  "stage": "text_assignment",
  "backend": "geometry",
  "inputs": {
    "pages": [],
    "nodes": [],
    "legend_pack": null,
  },
  "assets": {}
}
```

That minimal request runs an empty inventory, useful for checking installation.
Real assignment inputs use `PageEvidence` and `ReconciledNode` JSON records.
`vision/stage_contracts.py` defines all request models; obtain their JSON schemas
with `StageRequest.model_json_schema()` and the input classes listed in
`vision/stages.py:STAGES`.

Images are already rendered page images in the same coordinates as `PageEvidence`.
They are not automatically resized PDFs. Assets use explicit checksums:

```json
"assets": {
  "image": {"path": "page.png", "sha256": "<64 hexadecimal characters>"},
  "checkpoint": {"path": "detector.pt", "sha256": "<64 hexadecimal characters>"}
}
```

`checkpoint` is only accepted by symbol detection; `source_pdf` is additionally
accepted by text detection to extract native words directly. Relative paths are
resolved against the request file. A changed asset is rejected before inference.
For `symbol_detection/cv`, use the environment containing Torch and Torchvision;
other stages do not require those packages. Native and raster proposals retain
separate identities, so CV boxes are never relabelled as native PDF evidence.

Model-based requests specify `model` and `transport` and require `--live`:

```bash
PYTHONPATH=src python -m diagex.cli stage-run request.json --out output/stages --live
```

Credentials come from the configured environment, never from saved requests.
OpenRouter calls require the existing `DIAGEX_SPENDING_LEDGER` and
`DIAGEX_VERIFIED_PRICES`. The shared client preserves budget accounting.
The saved `usage_estimate` is per-stage token usage and estimated cost; actual
billing and unresolved reservations remain authoritative in the spending ledger.
No live API experiments were needed to implement this refactor.

Symbol interpretation accepts a tile, its ownership box, candidates, legend,
page context and fixed-view policy. `broad_review` preserves raw guidance in the
request and records VLM decisions separately. Connection `geometry` requires
saved `LineDetectionResult`; it does not rerun the line detector. Connection
`vlm` additionally requires saved topology. Optional `line_interpretation/vlm`
collects visual assessments before semantic connection inference.

Independent VLM connection runs use one injected client and the solver's default
settings. They do not reproduce every production routing option (such as separate
inspection/escalation clients or additional process context). Fixed-view symbol
requests are supported; adaptive inspection remains in the full workflow.

## Saved inputs and caching

The normal pipeline saves module requests under `RUN/module_inputs/` for native
text, native symbol detection, symbol interpretation, text assignment, line
geometry and geometric connection inference. Rendered page assets are shared
by checksum. The original checkpoint/review artifacts remain available.

For example, assignment can be rerun without perception or any model calls:

```bash
PYTHONPATH=src python -m diagex.cli stage-run \
  RUN/module_inputs/text_assignment/objects.json --out output/assignment-experiment
```

Each independent run writes `request.json`, `identity.json` and `result.json`
under `OUT/STAGE/IDENTITY/`. The identity covers the inputs, asset checksums,
backend, relevant implementation files, package versions, and public model
routing. A per-identity lock prevents duplicate concurrent execution. A matching
cached result is reused after checking its output checksum. Transport exceptions
produce `failure.json` and do not produce a successful result. Structured partial
symbol results are explicitly marked `partial`; unresolved decisions are evidence,
not automatically approved graph objects.

Changing assignment logic invalidates assignment replay, without rerunning a
saved symbol detector. Full-pipeline checkpoint versions separately invalidate
downstream graph stages when assignment or line logic changes. The engine schema
changes for this refactor, preventing accidental reuse of a pre-refactor run as
if it contained the new module artifacts.

## Independent evaluation

```bash
PYTHONPATH=src python -m diagex.cli stage-score reference.json result.json
```

Reference formats and metrics:

- Symbols: `{"symbols": [{"bbox": {"x":0,"y":0,"w":10,"h":10}, "category":"valve"}]}`.
  Localization and class-aware matching at IoU 0.5, with one-to-one matches.
  Native detections use glyph shapes; CV uses broad categories; interpreted
  engineering detections use `kind`. Use the corresponding reference vocabulary.
- Text: `{"text_spans": [{"bbox": {...}, "text":"PT-101"}]}`. Localization,
  exact matched transcriptions, and spatial character error rate including missed
  and extra text. This is not reading-order paragraph OCR evaluation.
- Lines: `{"paths": [{"points": [[0,0],[100,0]]}]}`. Polyline matching with a
  configurable pixel tolerance and reversed orientation allowed. This measures
  matching polylines, not segmentation-invariant line coverage or junction quality.
- Assignment: `{"bindings": [{"text_id":"t1", "target_id":"s1"}]}`. Correct
  text-to-symbol/scope relationships; upstream IDs must be fixed.
- Connections: `{"edges": [{"from_node":"s1", "to_node":"s2", "line_type":"process"}]}`.
  Directed, typed edge precision/recall/F1 with fixed upstream symbol IDs.

These are module development metrics. Keep the existing frozen PID2Graph scorer
for study comparisons; do not replace it with this evaluator midway through a
study. GraphML cannot supply missing transcription or detailed legend labels.
No metric establishes engineering approval. Always also run the complete pipeline
on a representative panel to detect propagation of upstream errors.

## Refactor isolation and further work

This branch starts from `f458f4c`. It was initially developed in a separate
worktree to avoid changing files used by evaluation. At the user's request it
has been consolidated into the ordinary `DiagEx-Repro` checkout on
`codex/modular-extraction`; the extra worktree has been removed. Uploaded files
and run data were copied across, with a separate backup retained. Use this one
directory for development and testing. No training or paid study reruns were
needed for the refactor.

The first iteration preserves behavior. Improved OCR, new detector weights,
assignment algorithms and connection algorithms can now be developed behind the
stage contracts and compared using the same saved inputs. Existing contextual
symbol resolution and cross-sheet assembly remain orchestration dependencies;
modules are replaceable stages, not independent microservices.

## Validation

- Full unit suite: **1,049 passed, 1 skipped**. Model calls in the new tests use
  fakes; no live API spending was needed.
- Original versus refactored deterministic execution on `dexpi-reference.pdf`
  produced identical serialized evidence, candidates, fused objects and topology:
  one page, 496 text spans, 2,292 paths, and 14 native symbol candidates. This is
  behavior parity, not an extraction accuracy measurement.
- Tests cover immutable assignment inputs, saved line-to-connection replay,
  production assignment capture/replay, cache invalidation, corrupt artifacts,
  asset checksums, model-call accounting, compatibility imports and stage metrics.
- No new Ruff findings. The 40 existing CLI findings remain unchanged.
- The budget-resume unit test now supplies its own model setting rather than
  depending on an untracked local environment file.
- Web CV tests cover enabled/disabled modes, invalid settings, detector failure
  before VLM calls, and PDF/image proposal validation. English and Chinese
  controls were checked in a temporary browser preview.
  A CPU smoke run with the actual exported epoch-six weights generated two
  proposals from a small synthetic PDF using the separate CV environment.
