"""Versioned, atomic checkpoints for the evidence-first extractor."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

CHECKPOINT_SCHEMA_VERSION = "2.0.0"


class CheckpointError(BaseModel):
    stage: str
    item: str
    detail: str


class CheckpointManifest(BaseModel):
    schema_version: str = CHECKPOINT_SCHEMA_VERSION
    engine: Literal["evidence-v2"] = "evidence-v2"
    source_sha256: str
    config_sha256: str
    run_id: str
    status: Literal["running", "partial", "paused", "complete", "error"] = "running"
    pause_reason: str | None = None
    completed: dict[str, list[str]] = Field(default_factory=dict)
    errors: list[CheckpointError] = Field(default_factory=list)
    stage_versions: dict[str, str] = Field(default_factory=dict)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True),
    )


class CheckpointStore:
    def __init__(self, run_dir: Path, manifest: CheckpointManifest) -> None:
        self.run_dir = Path(run_dir)
        self.root = self.run_dir / "checkpoints"
        self.path = self.root / "manifest.json"
        self.manifest = manifest
        self._reused: dict[str, set[str]] = {}
        self._computed: dict[str, set[str]] = {}

    @classmethod
    def create(
        cls,
        *,
        run_dir: Path,
        source_sha256: str,
        config_sha256: str,
        run_id: str,
    ) -> CheckpointStore:
        store = cls(
            run_dir,
            CheckpointManifest(
                source_sha256=source_sha256,
                config_sha256=config_sha256,
                run_id=run_id,
            ),
        )
        store.save()
        return store

    @classmethod
    def load(cls, run_dir: Path) -> CheckpointStore:
        path = Path(run_dir) / "checkpoints" / "manifest.json"
        manifest = CheckpointManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if manifest.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"checkpoint schema {manifest.schema_version!r} is incompatible with "
                f"{CHECKPOINT_SCHEMA_VERSION!r}"
            )
        return cls(Path(run_dir), manifest)

    def save(self) -> None:
        atomic_write_text(self.path, self.manifest.model_dump_json(indent=2))

    def matches(self, *, source_sha256: str, config_sha256: str) -> bool:
        return (
            self.manifest.engine == "evidence-v2"
            and self.manifest.source_sha256 == source_sha256
            and self.manifest.config_sha256 == config_sha256
        )

    def is_done(self, stage: str, item: str) -> bool:
        return item in self.manifest.completed.get(stage, [])

    def seed_raw_evidence_from(
        self, source: CheckpointStore, *, include_contextual: bool = False,
    ) -> None:
        """Upgrade a completed run in a new directory, preserving its review.

        Copy bytes rather than linking them: subsequent checkpoint writes must
        never change the original graph, evidence, or human review history.
        """
        if self.manifest.completed or not source.matches(
            source_sha256=self.manifest.source_sha256,
            config_sha256=self.manifest.config_sha256,
        ):
            raise ValueError("raw evidence reuse requires a fresh compatible checkpoint")
        stages = ("inspection", "perception", "contextual") if include_contextual else ("inspection", "perception")
        for relative in ("evidence", *(f"checkpoints/{stage}" for stage in stages)):
            folder = source.run_dir / relative
            if folder.is_dir():
                # Run preparation already creates evidence/. Populate that
                # directory while still copying independent checkpoint bytes.
                shutil.copytree(folder, self.run_dir / relative, dirs_exist_ok=True)
        self.manifest.completed = {
            stage: list(source.manifest.completed[stage])
            for stage in stages if stage in source.manifest.completed
        }
        self.manifest.stage_versions = dict(source.manifest.stage_versions)
        self.save()

    def record_reuse(self, stage: str, item: str) -> None:
        self._reused.setdefault(stage, set()).add(item)

    def reuse_summary(self) -> dict[str, Any]:
        reused = {stage: len(items) for stage, items in sorted(self._reused.items())}
        computed = {stage: len(items) for stage, items in sorted(self._computed.items())}
        reused_total = sum(reused.values())
        computed_total = sum(computed.values())
        total = reused_total + computed_total
        return {
            "reused_by_stage": reused,
            "computed_by_stage": computed,
            "reused_total": reused_total,
            "computed_total": computed_total,
            "reuse_percent": round(100.0 * reused_total / total, 1) if total else 0.0,
        }

    def mark_done(self, stage: str, item: str) -> None:
        self._computed.setdefault(stage, set()).add(item)
        values = self.manifest.completed.setdefault(stage, [])
        if item not in values:
            values.append(item)
            values.sort()
        self.manifest.errors = [
            error
            for error in self.manifest.errors
            if not (error.stage == stage and error.item == item)
        ]
        self.manifest.status = "running"
        self.manifest.pause_reason = None
        self.save()

    def mark_error(self, stage: str, item: str, detail: str) -> None:
        self.manifest.errors = [
            error
            for error in self.manifest.errors
            if not (error.stage == stage and error.item == item)
        ]
        self.manifest.errors.append(CheckpointError(stage=stage, item=item, detail=detail))
        self.manifest.status = "partial"
        self.save()

    def invalidate(self, *stages: str) -> None:
        """Invalidate derived stages after an upstream checkpoint changes."""
        changed = False
        for stage in stages:
            if self.manifest.completed.pop(stage, None) is not None:
                changed = True
        remaining_errors = [error for error in self.manifest.errors if error.stage not in stages]
        if len(remaining_errors) != len(self.manifest.errors):
            self.manifest.errors = remaining_errors
            changed = True
        if changed:
            self.manifest.status = "running"
            self.save()

    def ensure_stage_version(
        self,
        stage_group: str,
        version: str,
        *,
        invalidate: tuple[str, ...] = (),
    ) -> bool:
        """Invalidate derived checkpoints once when their implementation changes."""
        if self.manifest.stage_versions.get(stage_group) == version:
            return False
        self.invalidate(*invalidate)
        # A copied checkpoint of an older version must not survive as apparent
        # output when recomputing its replacement fails. Keep source evidence.
        for name in invalidate:
            folder = self.root / name
            if folder.is_dir():
                shutil.rmtree(folder)
            if name == "perception":
                diagnostics = self.root / "perception_diagnostics"
                if diagnostics.is_dir():
                    shutil.rmtree(diagnostics)
        self.manifest.stage_versions[stage_group] = version
        self.manifest.status = "running"
        self.save()
        return True

    def set_status(self, status: Literal["running", "partial", "paused", "complete", "error"]) -> None:
        self.manifest.status = status
        self.save()

    def artifact_path(self, stage: str, item: str, *, suffix: str = ".json") -> Path:
        safe_item = item.replace("/", "_").replace("..", "_")
        return self.root / stage / f"{safe_item}{suffix}"

    def write_json_artifact(self, stage: str, item: str, value: Any) -> Path:
        path = self.artifact_path(stage, item)
        atomic_write_json(path, value)
        self.mark_done(stage, item)
        return path

    def read_json_artifact(self, stage: str, item: str) -> Any:
        value = json.loads(self.artifact_path(stage, item).read_text(encoding="utf-8"))
        self.record_reuse(stage, item)
        return value


def find_resumable_run(
    *,
    runs_root: Path,
    source_sha256: str,
    config_sha256: str,
) -> CheckpointStore | None:
    store, _ = find_resumable_run_with_report(
        runs_root=runs_root,
        source_sha256=source_sha256,
        config_sha256=config_sha256,
    )
    return store


def find_resumable_run_with_report(
    *,
    runs_root: Path,
    source_sha256: str,
    config_sha256: str,
    required_stage_versions: dict[str, str] | None = None,
) -> tuple[CheckpointStore | None, dict[str, Any]]:
    """Find a resumable run and explain why a new run may be required."""

    report: dict[str, Any] = {
        "considered_runs": 0,
        "incompatible_manifests": 0,
        "source_mismatches": 0,
        "config_mismatches": 0,
        "matching_complete_runs": 0,
        "reason": "no prior run directories",
    }
    if not runs_root.exists():
        return None, report
    candidates = sorted(
        (path for path in runs_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    current_complete_seen = False
    for candidate in candidates:
        path = candidate / "checkpoints" / "manifest.json"
        if not path.is_file():
            continue
        report["considered_runs"] += 1
        try:
            store = CheckpointStore.load(candidate)
        except (OSError, ValueError):
            report["incompatible_manifests"] += 1
            continue
        if store.manifest.source_sha256 != source_sha256:
            report["source_mismatches"] += 1
            continue
        if store.manifest.config_sha256 != config_sha256:
            report["config_mismatches"] += 1
            continue
        if store.manifest.status == "complete":
            report["matching_complete_runs"] += 1
            if not any(
                store.manifest.stage_versions.get(group) != version
                for group, version in (required_stage_versions or {}).items()
            ):
                current_complete_seen = bool(required_stage_versions)
                continue
            if current_complete_seen:
                # A newer run has already performed this upgrade. Do not
                # repeatedly seed another run from the older completed input.
                continue
            report["reason"] = "matching checkpoint needs derived-stage upgrade"
            report["selected_run_dir"] = str(candidate)
            return store, report
        report["reason"] = "matching incomplete checkpoint found"
        report["selected_run_dir"] = str(candidate)
        return store, report
    if report["matching_complete_runs"]:
        report["reason"] = "matching prior run is complete; completed runs are not resumed"
    elif report["config_mismatches"]:
        report["reason"] = "prior incomplete run uses different extraction configuration"
    elif report["source_mismatches"]:
        report["reason"] = "prior incomplete run is for different source content"
    elif report["incompatible_manifests"]:
        report["reason"] = "prior checkpoint schema is incompatible"
    elif report["considered_runs"]:
        report["reason"] = "no compatible incomplete checkpoint found"
    return None, report
