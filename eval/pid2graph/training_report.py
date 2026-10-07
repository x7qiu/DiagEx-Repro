"""Report committed training exposure and validation convergence without test data."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .data import SYMBOLS, save_new, sha256
from .epoch_train import epoch_order, sealed, selection_state


def exposure(data, schedule, state):
    samples = data["samples"]
    epoch_size, batch = schedule["sampling"]["epoch_presentations"], schedule["batch_size"]
    epoch, cursor = state["epoch"], state["cursor"]
    if not 1 <= epoch <= schedule["maximum_epochs"] + 1 or not 0 <= cursor <= epoch_size:
        raise ValueError("Invalid committed training position")
    if cursor != epoch_size and cursor % batch:
        raise ValueError("Committed cursor is not a complete batch")
    if epoch == schedule["maximum_epochs"] + 1 and cursor:
        raise ValueError("Position exceeds maximum epochs")
    count = (epoch - 1) * epoch_size + cursor
    steps = (epoch - 1) * math.ceil(epoch_size / batch) + math.ceil(cursor / batch)
    if state["presentations"] != count or state["step"] != steps:
        raise ValueError("Committed steps/presentations disagree with the frozen sampler")
    source_lookup = {r["id"]: r for r in data["sources"]}
    if len(source_lookup) != len(data["sources"]) or any(r["split"] != "train" for r in data["sources"]):
        raise ValueError("Training source identities or splits differ")
    labels = {}
    for sample in samples:
        source = source_lookup[sample["source_id"]]
        if sample["source_group"] != source["group"]:
            raise ValueError("Crop belongs to a different drawing group")
        if len(sample["reference_ids"]) != len(sample["labels"]):
            raise ValueError("Reference IDs and labels differ")
        if not set(sample["fully_covered_ids"]) <= set(sample["reference_ids"]):
            raise ValueError("Fully covered symbol is absent from the crop")
        for identity, label in zip(sample["reference_ids"], sample["labels"], strict=True):
            if not 1 <= label <= len(SYMBOLS):
                raise ValueError("Unknown training label")
            key = (sample["source_id"], identity)
            category = SYMBOLS[label - 1]
            if key in labels and labels[key] != category:
                raise ValueError("Overlapping crops disagree on a physical symbol's class")
            labels[key] = category
    totals = Counter(labels.values())
    if totals != Counter(data["symbol_counts"]):
        raise ValueError("Sample reference inventory differs from physical-symbol totals")
    frequencies = Counter()
    for number in range(1, epoch):
        frequencies.update(epoch_order(samples, schedule, number))
    if cursor:
        frequencies.update(epoch_order(samples, schedule, epoch)[:cursor])
    fully_seen, partially_or_fully_seen = set(), set()
    annotation_presentations = Counter()
    drawing_presentations, group_presentations = Counter(), Counter()
    background_presentations = 0
    for index, frequency in frequencies.items():
        sample = samples[index]
        source = sample["source_id"]
        drawing_presentations[source] += frequency
        group_presentations[sample["source_group"]] += frequency
        if not sample["labels"]:
            background_presentations += frequency
        for identity, label in zip(sample["reference_ids"], sample["labels"], strict=True):
            partially_or_fully_seen.add((source, identity))
            annotation_presentations[SYMBOLS[label - 1]] += frequency
        fully_seen.update((source, identity) for identity in sample["fully_covered_ids"])
    full_counts = Counter(labels[k] for k in fully_seen)
    seen_counts = Counter(labels[k] for k in partially_or_fully_seen)
    by_drawing = defaultdict(lambda: {"physical_symbols": 0, "fully_seen_symbols": 0})
    for key in labels:
        by_drawing[key[0]]["physical_symbols"] += 1
        by_drawing[key[0]]["fully_seen_symbols"] += int(key in fully_seen)
    return {"completed_sampling_epochs": epoch - 1 + int(cursor == epoch_size),
            "current_epoch": epoch, "current_epoch_presentations": cursor,
            "committed_steps": steps, "crop_presentations": count,
            "unique_crops_presented": len(frequencies), "prepared_crops": len(samples),
            "drawings_presented": len(drawing_presentations), "training_drawings": len(source_lookup),
            "groups_presented": len(group_presentations),
            "training_groups": len({r["group"] for r in data["sources"]}),
            "background_presentations": background_presentations,
            "fully_seen_physical_symbols": len(fully_seen), "physical_symbols": len(labels),
            "per_class": {category: {"physical_symbols": totals[category],
                          "fully_seen_symbols": full_counts[category],
                          "partially_or_fully_seen_symbols": seen_counts[category],
                          "annotation_presentations_including_repeats": annotation_presentations[category]}
                          for category in SYMBOLS},
            "per_drawing": {identity: {**by_drawing[identity], "group": source["group"],
                            "crop_presentations": drawing_presentations[identity]}
                            for identity, source in source_lookup.items()}}


def report(training_path, schedule_path, run, output):
    data = sealed(training_path, "training_sha256")
    schedule = sealed(schedule_path, "schedule_sha256")
    run = Path(run)
    state_path = run / ("completed.json" if (run / "completed.json").exists() else "progress.json")
    raw = state_path.read_bytes()
    state = json.loads(raw)
    config = json.loads((run / "config.json").read_text())
    if config.get("software_smoke_only"):
        raise ValueError("Software smoke is not accuracy training")
    if (config["training_sha256"] != data["training_sha256"]
            or schedule["training_sha256"] != data["training_sha256"]
            or config["schedule_sha256"] != schedule["schedule_sha256"]
            or state["schedule_sha256"] != schedule["schedule_sha256"]):
        raise ValueError("Training/schedule identity differs")
    for path, expected in config["source_hashes"].items():
        if sha256(path) != expected:
            raise ValueError("Frozen training or validation source changed")
    measured = exposure(data, schedule, state)
    history = state["history"]
    expected_epochs = list(range(measured["completed_sampling_epochs"] + 1))
    # A saved full-epoch cursor can precede its validation report after an
    # interruption. It is exposure evidence, not a validated checkpoint.
    actual_epochs = [row["epoch"] for row in history]
    awaiting_validation = (state["cursor"] == schedule["sampling"]["epoch_presentations"]
                           or (state["epoch"] == 1 and state["cursor"] == 0))
    if actual_epochs != expected_epochs and not (awaiting_validation and actual_epochs == expected_epochs[:-1]):
        raise ValueError("Validation history does not follow completed epochs")
    validation = []
    for row in history:
        if sha256(row["report"]) != row["report_sha256"]:
            raise ValueError("Archived validation report changed")
        scored = json.loads(Path(row["report"]).read_text())
        if (scored["split"] != "validation" or not scored["complete"]
                or scored["config"]["detector_sha256"] != sha256(row["checkpoint"])
                or scored["panel_sha256"] != config["panel_sha256"]
                or scored["summary"]["macro_drawing_f1"] != row["macro_drawing_f1"]):
            raise ValueError("Validation checkpoint or measurement differs")
        validation.append({**row, "summary": scored["summary"], "per_collection": scored["per_collection"]})
    chosen = selection_state(history, schedule) if history else None
    if chosen != state.get("selection"):
        raise ValueError("Checkpoint selection differs from frozen rule")
    complete = state_path.name == "completed.json"
    if complete and (not chosen or state["cursor"] or measured["completed_sampling_epochs"] < schedule["minimum_epochs"]
                     or not (chosen["early_stop"] or measured["completed_sampling_epochs"] == schedule["maximum_epochs"])
                     or len(history) != len(expected_epochs)):
        raise ValueError("Training completion is not supported by the stopping rule")
    result = {"complete": complete, "state_path": str(state_path.resolve()),
              "state_sha256": hashlib.sha256(raw).hexdigest(),
              "training_sha256": data["training_sha256"], "schedule_sha256": schedule["schedule_sha256"],
              "reporter_sha256": sha256(__file__), "exposure": measured, "validation": validation,
              "selection": chosen, "training_seconds": state["training_seconds"],
              "stop_reason": state.get("reason"),
              "scope": "Committed checkpoint exposure reconstructed from the frozen epoch sampler. Physical IDs are scoped by drawing; overlapping and repeated crops do not create additional physical symbols. Pending batches after the saved checkpoint are excluded. Augmentation preserves the included boxes. No test inputs or GraphML are read. A progress snapshot is not evidence that a process is currently live."}
    save_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "schedule", "run", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = report(args.training, args.schedule, args.run, args.out)
    print(json.dumps({"complete": result["complete"], "exposure": {
        k: v for k, v in result["exposure"].items() if k not in {"per_class", "per_drawing"}}}, indent=2))


if __name__ == "__main__":
    main()
