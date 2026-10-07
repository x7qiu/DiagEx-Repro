"""Dependency-free local HTTP server for the DiagEx extraction workbench.

The workbench deliberately keeps provider credentials in process memory.  It
persists uploaded source files and redacted run metadata so a completed run can
be opened in the existing human-review application after a browser refresh.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import mimetypes
import re
import threading
import time
import uuid
import webbrowser
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import parse_qs, unquote, urlencode, urlsplit

from rich.console import Console

from diagex.config import Config, LLMConfig
from diagex.extractors.evidence_checkpoint import atomic_write_json
from diagex.review.detection import DetectionReviewStore, ReviewConflict

STATIC_DIR = Path(__file__).with_name("static")
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
ALLOWED_UPLOAD_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_SAFE_FILENAME_RE = re.compile(r"[^\w.()\-\u3400-\u9fff]+", re.UNICODE)
_TERMINAL_JOB_STATES = {"succeeded", "failed", "paused"}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _safe_filename(value: str) -> str:
    name = Path(value).name.strip().replace("\x00", "")
    name = _SAFE_FILENAME_RE.sub("_", name).strip("._")
    return name[:180] or "drawing.pdf"


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provider_base_url(config: LLMConfig) -> str:
    if config.transport == "openrouter":
        return config.openrouter_base_url
    if config.transport == "kimi":
        return config.kimi_base_url
    if config.transport == "azure":
        return config.azure_endpoint or ""
    return ""


def _provider_has_key(config: LLMConfig, provider: str) -> bool:
    return bool(
        {
            "anthropic": config.anthropic_api_key,
            "openrouter": config.openrouter_api_key,
            "kimi": config.kimi_api_key,
            "azure": config.azure_api_key,
        }.get(provider)
    )


@dataclass
class UploadedDiagram:
    id: str
    filename: str
    path: Path
    content_type: str
    size: int
    created_at: str = field(default_factory=_utc_now)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "filename": self.filename,
            "content_type": self.content_type,
            "size": self.size,
            "created_at": self.created_at,
        }


@dataclass
class ExtractionJob:
    id: str
    upload_id: str
    filename: str
    source_path: Path
    settings: dict[str, Any]
    status: str = "queued"
    created_at: str = field(default_factory=_utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    run_dir: Path | None = None
    result: dict[str, Any] | None = None
    _logs: deque[tuple[int, str]] = field(default_factory=lambda: deque(maxlen=10000))
    _next_log_id: int = 1
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def append_log(self, text: str) -> None:
        normalised = text.replace("\r", "\n")
        with self._lock:
            for line in normalised.splitlines():
                line = line.rstrip()
                if not line:
                    continue
                self._logs.append((self._next_log_id, line))
                self._next_log_id += 1

    def public(self, *, after: int = 0, include_logs: bool = True) -> dict[str, Any]:
        with self._lock:
            logs = [
                {"id": log_id, "text": text}
                for log_id, text in self._logs
                if include_logs and log_id > after
            ]
            return {
                "id": self.id,
                "upload_id": self.upload_id,
                "filename": self.filename,
                "settings": dict(self.settings),
                "status": self.status,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "error": self.error,
                "run_dir": str(self.run_dir) if self.run_dir else None,
                "result": copy.deepcopy(self.result),
                "logs": logs,
                "log_cursor": self._next_log_id - 1,
            }


class _JobLogWriter(io.TextIOBase):
    def __init__(self, job: ExtractionJob) -> None:
        self.job = job
        self._pending = ""
        self._lock = threading.Lock()

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        with self._lock:
            self._pending += value
            parts = self._pending.replace("\r", "\n").split("\n")
            self._pending = parts.pop()
            for line in parts:
                self.job.append_log(line)
        return len(value)

    def flush(self) -> None:
        with self._lock:
            if self._pending.strip():
                self.job.append_log(self._pending)
            self._pending = ""


class WorkbenchError(RuntimeError):
    """User-facing workbench request error."""


class WorkbenchConflictError(WorkbenchError):
    """The requested action conflicts with current workbench state."""


class Workbench:
    """Own uploads, extraction jobs, and child review servers for one process."""

    def __init__(
        self,
        config: Config,
        *,
        storage_dir: Path | None = None,
        runner: Callable[..., Any] | None = None,
    ) -> None:
        self.config = copy.deepcopy(config)
        runs_root = Path(self.config.runs_dir).expanduser().resolve()
        self.storage_dir = (storage_dir or runs_root / ".web").expanduser().resolve()
        self.uploads_dir = self.storage_dir / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self.detection_stores: dict[str, DetectionReviewStore] = {}
        self.detection_images: dict[tuple[str, int], bytes] = {}
        self.uploads: dict[str, UploadedDiagram] = {}
        self.jobs: dict[str, ExtractionJob] = {}
        self.review_servers: dict[str, tuple[ThreadingHTTPServer, threading.Thread, str]] = {}
        self._source_hash_cache: dict[Path, tuple[int, int, str]] = {}
        self._lock = threading.RLock()

    def public_config(self) -> dict[str, Any]:
        from diagex.web.model_profiles import DEFAULT_MODEL_POLICY, MODEL_PROFILES

        llm = self.config.llm
        profile = MODEL_PROFILES[DEFAULT_MODEL_POLICY]
        return {
            "model_policy": DEFAULT_MODEL_POLICY,
            "model_profiles": copy.deepcopy(MODEL_PROFILES),
            "provider": profile["provider"],
            "base_url": "https://openrouter.ai/api",
            "model": profile["vision_model"],
            "vision_model": profile["vision_model"],
            "reasoning_model": profile["reasoning_model"],
            "reasoning_mode": llm.reasoning_mode,
            "engine": profile["engine"],
            "effort": "medium",
            "configured_keys": {
                provider: _provider_has_key(llm, provider)
                for provider in ("openrouter", "kimi", "anthropic", "azure")
            },
            "runs_dir": str(Path(self.config.runs_dir).expanduser().resolve()),
            "upload_limit_bytes": MAX_UPLOAD_BYTES,
        }

    def save_upload(
        self,
        *,
        filename: str,
        content_type: str,
        source: BinaryIO,
        length: int,
    ) -> UploadedDiagram:
        if length <= 0:
            raise WorkbenchError("uploaded file is empty")
        if length > MAX_UPLOAD_BYTES:
            raise WorkbenchError("uploaded file exceeds the 500 MiB limit")
        safe_name = _safe_filename(filename)
        suffix = Path(safe_name).suffix.lower()
        if suffix not in ALLOWED_UPLOAD_SUFFIXES:
            raise WorkbenchError("upload a PDF or supported image file")
        upload_id = "u-" + uuid.uuid4().hex[:12]
        upload_dir = self.uploads_dir / upload_id
        upload_dir.mkdir()
        # Preserve the safe original basename so run directories and exported
        # artifacts stay recognisable to the operator.
        path = upload_dir / safe_name
        remaining = length
        try:
            with path.open("xb") as handle:
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise WorkbenchError(
                            "upload ended before Content-Length bytes were received"
                        )
                    handle.write(chunk)
                    remaining -= len(chunk)
        except Exception:
            if path.exists():
                path.unlink()
            upload_dir.rmdir()
            raise
        upload = UploadedDiagram(
            id=upload_id,
            filename=safe_name,
            path=path,
            content_type=content_type or "application/octet-stream",
            size=length,
        )
        with self._lock:
            self.uploads[upload.id] = upload
        return upload

    def start_extraction(self, body: dict[str, Any]) -> ExtractionJob:
        upload_id = str(body.get("upload_id") or "")
        reviewed = None
        if body.get("reviewed_run"):
            review_store = self.detection_store(str(body["reviewed_run"]))
            reviewed = review_store.snapshot(body.get("reviewed_revision"), draft=bool(body.get("draft", False)))
            source, verified = self._resolve_run_source(review_store.run_dir)
            if source is None or not verified:
                raise WorkbenchError("Attach the original P&ID before building the graph")
            upload = UploadedDiagram(
                id="reviewed",
                filename=source.name,
                path=source,
                content_type="",
                size=source.stat().st_size,
                created_at=_utc_now(),
            )
        with self._lock:
            if reviewed is None:
                upload = self.uploads.get(upload_id)
            if upload is None:
                raise WorkbenchError("select and upload a P&ID first")
            if any(job.status not in _TERMINAL_JOB_STATES for job in self.jobs.values()):
                raise WorkbenchConflictError("another extraction is already running")

        config, redacted = self._config_for_request(body)
        from diagex.vision.process_context import context_document
        config.process_context = [context_document(body[key], source="workbench:" + key, kind=key)
                                  for key in ("process_overview", "engineering_rules") if body.get(key)]
        redacted["process_context"] = config.process_context
        fresh = bool(body.get("fresh", False))
        effort = str(body.get("effort") or "medium").strip().lower()
        if effort not in {"low", "medium", "high", "xhigh"}:
            raise WorkbenchError("effort must be low, medium, high, or xhigh")
        engine = str(body.get("engine") or "evidence-v2").strip().lower().replace("_", "-")
        if config.llm.production_open_weight and engine != "evidence-v2":
            raise WorkbenchError("The open-weight production profile requires evidence-v2")
        if engine not in {"legacy", "evidence-v2"}:
            raise WorkbenchError("engine must be legacy or evidence-v2")
        stop_after = str(body.get("stop_after") or "graph")
        if stop_after not in {"graph", "detection"}:
            raise WorkbenchError("Unknown workflow stage")
        if (stop_after == "detection" or reviewed is not None) and engine != "evidence-v2":
            raise WorkbenchError("Legend and symbol review requires Evidence v2")
        if reviewed is not None:
            fresh, stop_after = True, "graph"
        redacted["stop_after"] = stop_after
        if reviewed is not None:
            redacted["reviewed_run"] = reviewed["source_run"]
            redacted["reviewed_revision"] = reviewed["revision"]
        config.pid.engine = engine  # type: ignore[assignment]
        redacted.update({"fresh": fresh, "effort": effort, "engine": engine})

        job = ExtractionJob(
            id="j-" + uuid.uuid4().hex[:12],
            upload_id=upload.id,
            filename=upload.filename,
            source_path=upload.path,
            settings=redacted,
        )
        with self._lock:
            if any(existing.status not in _TERMINAL_JOB_STATES for existing in self.jobs.values()):
                raise WorkbenchConflictError("another extraction is already running")
            self.jobs[job.id] = job
        thread = threading.Thread(
            target=self._run_extraction,
            args=(job, config, effort, engine, fresh, reviewed),
            name=f"diagex-{job.id}",
            daemon=True,
        )
        thread.start()
        return job

    def _config_for_request(self, body: dict[str, Any]) -> tuple[Config, dict[str, Any]]:
        from diagex.web.model_profiles import MODEL_PROFILES

        policy = body.get("model_policy") or "evaluation"
        if policy not in {*MODEL_PROFILES, "evaluation"}:
            raise WorkbenchError("unknown model profile")
        production = policy == "production-open-weight"
        profile = MODEL_PROFILES.get(policy)
        if profile:
            body = {**body, **profile}
        provider = str(body.get("provider") or self.config.llm.transport).strip().lower()
        if provider not in {"openrouter", "kimi", "anthropic", "azure"}:
            raise WorkbenchError("unknown LLM provider")
        vision_model = str(body.get("vision_model") or body.get("model") or "").strip()
        reasoning_model = str(body.get("reasoning_model") or vision_model).strip()
        if not vision_model:
            raise WorkbenchError("choose a vision model")
        if not reasoning_model:
            reasoning_model = vision_model
        reasoning_mode = str(body.get("reasoning_mode") or "auto").strip().lower()
        if reasoning_mode not in {"auto", "enabled", "disabled"}:
            raise WorkbenchError("reasoning must be auto, enabled, or disabled")
        api_key = str(body.get("api_key") or "").strip()
        base_url = str(body.get("base_url") or "").strip().rstrip("/")

        cfg = copy.deepcopy(self.config)
        llm = replace(
            cfg.llm,
            transport=provider,
            model=vision_model,
            vision_model=vision_model,
            reasoning_model=reasoning_model,
            reasoning_mode=reasoning_mode,
            escalation_model=profile["escalation_model"] if profile else reasoning_model,
            production_open_weight=False,
        )
        if provider == "openrouter":
            llm.openrouter_api_key = api_key or cfg.llm.openrouter_api_key
            llm.openrouter_base_url = base_url or "https://openrouter.ai/api"
            if not llm.openrouter_api_key:
                raise WorkbenchError("enter an OpenRouter API key")
        elif provider == "kimi":
            llm.kimi_api_key = api_key or cfg.llm.kimi_api_key
            llm.kimi_base_url = base_url or "https://api.kimi.com/coding/v1"
            if not llm.kimi_api_key:
                raise WorkbenchError("enter a Kimi API key")
        elif provider == "azure":
            llm.azure_api_key = api_key or cfg.llm.azure_api_key
            llm.azure_endpoint = base_url or cfg.llm.azure_endpoint
            llm.azure_deployment = vision_model
            if not llm.azure_api_key or not llm.azure_endpoint:
                raise WorkbenchError("enter an Azure API key and endpoint")
        else:
            llm.anthropic_api_key = api_key or cfg.llm.anthropic_api_key
            if not llm.anthropic_api_key:
                raise WorkbenchError("enter an Anthropic API key")
        cfg.llm = llm
        if production:
            from diagex.llm.model_policy import apply_production_profile
            apply_production_profile(cfg)
        elif profile:
            cfg.pid.engine = profile["engine"]
            cfg.symbol_perception = replace(cfg.symbol_perception, workflow="fixed")
        return cfg, {
            "model_policy": policy,
            "provider": provider,
            "base_url": base_url or _provider_base_url(llm),
            "vision_model": vision_model,
            "reasoning_model": reasoning_model,
            "reasoning_mode": reasoning_mode,
            "api_key_source": "entered" if api_key else "environment",
        }

    def _run_extraction(
        self,
        job: ExtractionJob,
        config: Config,
        effort: str,
        engine: str,
        fresh: bool,
        reviewed_inputs: dict | None = None,
    ) -> None:
        job.status = "running"
        job.started_at = _utc_now()
        job.append_log(f"Starting {engine} extraction for {job.filename}")
        writer = _JobLogWriter(job)
        console = Console(file=writer, force_terminal=False, color_system=None, width=160)
        started = time.monotonic()
        try:
            runner = self.runner
            if runner is None:
                from diagex.extractors.pid import run_pid_extract

                runner = run_pid_extract
            result = runner(
                diagram=job.source_path,
                symbol_standard="isa-5.1",
                legend_path=None,
                legend_pages=None,
                legend_region=None,
                no_legend=False,
                legend_key=None,
                effort=effort,
                max_steps=None,
                engine=engine,
                config=config,
                persist=True,
                fresh=fresh,
                stop_after=job.settings.get("stop_after", "graph"),
                reviewed_inputs=reviewed_inputs,
                out_path=None,
                confidence_report_path=None,
                console=console,
            )
            writer.flush()
            job.run_dir = Path(result.run_dir).resolve() if result.run_dir else None
            job.result = self._public_result(result)
            job.status = "succeeded"
            if job.run_dir is not None:
                manifest_path = job.run_dir / "checkpoints" / "manifest.json"
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text())
                    if manifest.get("status") == "paused":
                        job.status = "paused"
                        job.result["pause_reason"] = manifest.get("pause_reason")
            if job.run_dir is not None:
                atomic_write_json(
                    job.run_dir / "workbench.json",
                    {
                        "schema_version": "1.0.0",
                        "job_id": job.id,
                        "upload_id": job.upload_id,
                        "source_path": str(job.source_path.resolve()),
                        "source_filename": job.filename,
                        "settings": job.settings,
                        "created_at": job.created_at,
                        "finished_at": _utc_now(),
                    },
                )
            job.append_log(
                f"Extraction finished with quality status {result.quality_status or 'unknown'} "
                f"in {time.monotonic() - started:.1f}s"
            )
        except Exception as exc:  # noqa: BLE001 - error is returned to the local operator
            writer.flush()
            job.status = "failed"
            job.error = str(exc)
            job.append_log(f"Extraction failed: {exc}")
        finally:
            job.finished_at = _utc_now()
            # The only remaining references are redacted job settings; the
            # request-specific Config (and therefore entered key) can be freed.
            del config

    @staticmethod
    def _public_result(result: Any) -> dict[str, Any]:
        cost = dict(result.cost_summary or {})
        return {
            "workflow_stage": getattr(result, "workflow_stage", "graph"),
            "diagram_stem": result.diagram_stem,
            "run_id": result.run_id,
            "engine": result.engine,
            "model": result.model,
            "effort": result.effort,
            "quality_status": result.quality_status or None,
            "dexpi_json_path": str(result.dexpi_json_path) if result.dexpi_json_path else None,
            "stats": dict(result.dexpi_stats or {}),
            "build_issue_count": len(result.dexpi_issues or []),
            "validation_issue_count": len(result.validation_issues or []),
            "legend": {
                "source": result.legend_source,
                "entry_count": result.legend_entry_count,
            },
            "tokens": int(cost.get("total_tokens", 0) or 0),
            "elapsed_s": float(cost.get("wall_clock_s", 0.0) or 0.0),
            "retries": int(cost.get("retries", 0) or 0),
        }

    def job(self, job_id: str) -> ExtractionJob:
        with self._lock:
            try:
                return self.jobs[job_id]
            except KeyError as exc:
                raise WorkbenchError("unknown extraction job") from exc

    def _source_hash(self, path: Path) -> str:
        resolved = path.expanduser().resolve()
        stat = resolved.stat()
        cached = self._source_hash_cache.get(resolved)
        fingerprint = (stat.st_size, stat.st_mtime_ns)
        if cached is not None and cached[:2] == fingerprint:
            return cached[2]
        value = _file_sha256(resolved)
        self._source_hash_cache[resolved] = (*fingerprint, value)
        return value

    @staticmethod
    def _expected_source_hash(run_dir: Path) -> str | None:
        for path in (
            run_dir / "detection.json",
            run_dir / "checkpoints" / "manifest.json",
            run_dir / "review" / "session.json",
            run_dir / "workbench.json",
        ):
            value = str(_read_json_object(path).get("source_sha256") or "").strip().lower()
            if re.fullmatch(r"[0-9a-f]{64}", value):
                return value
        return None

    def _source_candidates(
        self, run_dir: Path, graph: dict[str, Any], *, allow_inferred: bool
    ) -> list[Path]:
        root = Path(self.config.runs_dir).expanduser().resolve()
        raw_candidates: list[str | Path] = []
        for path in (run_dir / "workbench.json", run_dir / "review" / "session.json"):
            source = _read_json_object(path).get("source_path")
            if source:
                raw_candidates.append(str(source))
        graph_source = str(graph.get("source_path") or "").strip()
        if graph_source:
            source = Path(graph_source).expanduser()
            if source.is_absolute():
                raw_candidates.append(source)
            elif allow_inferred:
                raw_candidates.extend((run_dir / source, root.parent / source, Path.cwd() / source))
            if allow_inferred:
                basename = source.name
                raw_candidates.extend(
                    candidate
                    for candidate in self.uploads_dir.glob("*/*")
                    if candidate.name == basename
                )

        candidates: list[Path] = []
        seen: set[Path] = set()
        for value in raw_candidates:
            candidate = Path(value).expanduser().resolve()
            if candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)
        return candidates

    def _resolve_run_source(
        self, run_dir: Path, graph: dict[str, Any] | None = None
    ) -> tuple[Path | None, bool | None]:
        graph_value = graph if graph is not None else _read_json_object(run_dir / "graph.json")
        expected = self._expected_source_hash(run_dir)
        for candidate in self._source_candidates(
            run_dir, graph_value, allow_inferred=expected is not None
        ):
            if not candidate.is_file() or candidate.suffix.lower() not in ALLOWED_UPLOAD_SUFFIXES:
                continue
            if expected is None:
                return candidate, None
            try:
                if self._source_hash(candidate) == expected:
                    return candidate, True
            except OSError:
                continue
        return None, None

    def _persist_source_link(self, run_dir: Path, source_path: Path, source_sha256: str) -> None:
        path = run_dir / "workbench.json"
        manifest = _read_json_object(path)
        manifest.update(
            {
                "schema_version": "1.0.0",
                "source_path": str(source_path.resolve()),
                "source_filename": source_path.name,
                "source_sha256": source_sha256,
                "source_linked_at": _utc_now(),
            }
        )
        atomic_write_json(path, manifest)

    def recent_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        root = Path(self.config.runs_dir).expanduser().resolve()
        records: list[dict[str, Any]] = []
        graph_paths = sorted(
            {
                p.parent / "graph.json"
                for name in ("graph.json", "detection.json")
                for p in root.glob(f"*/*/{name}")
            },
            key=lambda path: path.parent.stat().st_mtime_ns,
            reverse=True,
        )
        for graph_path in graph_paths:
            try:
                run_dir = graph_path.parent.resolve()
                graph = _read_json_object(graph_path)
                result = _read_json_object(run_dir / "result.json")
                quality = _read_json_object(run_dir / "quality.report.json")
                workbench_manifest = _read_json_object(run_dir / "workbench.json")
                source_path, source_verified = self._resolve_run_source(run_dir, graph)
                source_name = (
                    source_path.name
                    if source_path is not None
                    else Path(str(graph.get("source_path") or "")).name
                )
                records.append(
                    {
                        "run_dir": str(run_dir),
                        "source_available": source_path is not None,
                        "source_verified": source_verified,
                        "source_filename": source_name or None,
                        "filename": workbench_manifest.get("source_filename")
                        or source_name
                        or run_dir.parent.name,
                        "diagram": run_dir.parent.name,
                        "finished_at": workbench_manifest.get("finished_at")
                        or run_dir.name[:19].replace("T", " "),
                        "run_id": result.get("run_id") or run_dir.name.rsplit("_", 1)[-1],
                        "model": result.get("model"),
                        "engine": result.get("engine"),
                        "quality_status": result.get("quality_status") or quality.get("status"),
                        "has_graph": graph_path.is_file(),
                        "has_detection": (run_dir / "detection.json").is_file(),
                        "workflow_stage": result.get("workflow_stage", "graph"),
                        "review_started": (run_dir / "review" / "session.json").is_file(),
                        "node_count": len(graph.get("nodes") or []),
                        "edge_count": len(graph.get("edges") or []),
                    }
                )
            except (OSError, ValueError, TypeError):
                continue
            if len(records) >= limit:
                break
        return records

    def launch_review(
        self,
        *,
        rater: str,
        job_id: str | None = None,
        run_dir: str | None = None,
        upload_id: str | None = None,
        kind: str = "graph",
    ) -> str:
        reviewer = rater.strip()
        if not reviewer:
            raise WorkbenchError("enter the reviewer name")
        target: Path | None = None
        source_path: Path | None = None
        if job_id:
            job = self.job(job_id)
            if job.status not in {"succeeded", "paused"} or job.run_dir is None:
                raise WorkbenchConflictError("the extraction must finish before review")
            target = job.run_dir
            source_path = job.source_path
        elif run_dir:
            target = Path(run_dir).expanduser().resolve()
        if target is None:
            raise WorkbenchError("select an extraction run")
        runs_root = Path(self.config.runs_dir).expanduser().resolve()
        target = target.resolve()
        if not target.is_relative_to(runs_root):
            raise WorkbenchError("the selected run is outside the configured runs directory")
        if kind not in {"graph", "detection"}:
            raise WorkbenchError("Unknown review type")
        artifact = target / ("detection.json" if kind == "detection" else "graph.json")
        if not artifact.is_file():
            raise WorkbenchError(
                f"The selected run has no {artifact.name}; run the corresponding extraction first"
            )

        if upload_id:
            with self._lock:
                upload = self.uploads.get(upload_id)
            if upload is None:
                raise WorkbenchError("the selected source upload is unavailable")
            source_path = upload.path
        elif source_path is None:
            source_path, _verified = self._resolve_run_source(target)
        if source_path is None or not source_path.is_file():
            raise WorkbenchError("attach the original P&ID before opening this run")

        source_sha = self._source_hash(source_path)
        expected_sha = self._expected_source_hash(target)
        if expected_sha is not None and source_sha != expected_sha:
            raise WorkbenchError("the selected P&ID does not match this run's source hash")
        self._persist_source_link(target, source_path, source_sha)

        if kind == "detection":
            self.detection_store(str(target))
            return "/detection-review?" + urlencode({"run_dir": str(target), "rater": reviewer})

        key = str(target.resolve())
        existing = self.review_servers.get(key)
        if existing is not None and existing[1].is_alive():
            return existing[2]

        from diagex.review.core import ReviewStore
        from diagex.review.server import make_handler as make_review_handler

        store = ReviewStore.open(target, source_path=source_path, rater=reviewer)
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_review_handler(store))
        thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.25},
            name=f"diagex-review-{target.name}",
            daemon=True,
        )
        thread.start()
        url = f"http://127.0.0.1:{int(server.server_address[1])}/"
        self.review_servers[key] = (server, thread, url)
        return url

    def detection_store(self, run_dir: str) -> DetectionReviewStore:
        target = Path(run_dir).expanduser().resolve()
        root = Path(self.config.runs_dir).expanduser().resolve()
        if not target.is_relative_to(root) or not (target / "detection.json").is_file():
            raise WorkbenchError("Select a completed legend and symbol detection run")
        source, verified = self._resolve_run_source(target)
        if source is None or not verified:
            raise WorkbenchError("Attach the original P&ID before reviewing detections")
        with self._lock:
            key = str(target)
            if key not in self.detection_stores:
                self.detection_stores[key] = DetectionReviewStore(target)
            return self.detection_stores[key]

    def detection_page(self, run_dir: str, index: int) -> bytes:
        import fitz
        from PIL import Image

        store = self.detection_store(run_dir)
        page = next((p for p in store.bundle["pages"] if p["page_index"] == index), None)
        if page is None:
            raise WorkbenchError("Unknown page")
        cache_key = (str(store.run_dir), index)
        if cache_key in self.detection_images:
            return self.detection_images[cache_key]
        source, _ = self._resolve_run_source(store.run_dir)
        scale = min(1.0, 2400 / max(page["width"], page["height"]))
        size = (max(1, round(page["width"] * scale)), max(1, round(page["height"] * scale)))
        if source.suffix.lower() == ".pdf":
            with fitz.open(source) as document:
                native = document[index]
                pix = native.get_pixmap(
                    matrix=fitz.Matrix(size[0] / native.rect.width, size[1] / native.rect.height),
                    alpha=False,
                )
                data = pix.tobytes("png")
                self.detection_images[cache_key] = data
                return data
        with Image.open(source) as original:
            original.seek(index)
            image = original.convert("RGB")
            image = image.resize(size)
            output = io.BytesIO()
            image.save(output, format="PNG")
            data = output.getvalue()
            self.detection_images[cache_key] = data
            return data

    def detection_crop(self, run_dir: str, index: int, bounds: list[int]) -> bytes:
        import fitz
        from PIL import Image

        store = self.detection_store(run_dir)
        page = next((p for p in store.bundle["pages"] if p["page_index"] == index), None)
        if page is None or len(bounds) != 4:
            raise WorkbenchError("Invalid source crop")
        x, y, width, height = bounds
        if (
            min(x, y) < 0
            or min(width, height) <= 0
            or x + width > page["width"]
            or y + height > page["height"]
        ):
            raise WorkbenchError("Crop must be inside its source page")
        source, _ = self._resolve_run_source(store.run_dir)
        if source.suffix.lower() == ".pdf":
            with fitz.open(source) as document:
                native = document[index]
                sx, sy = native.rect.width / page["width"], native.rect.height / page["height"]
                clip = fitz.Rect(x * sx, y * sy, (x + width) * sx, (y + height) * sy)
                # Re-render native vector strokes; do not magnify the page preview.
                scale = min(128.0, 1200 / max(clip.width, clip.height))
                pix = native.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
                return pix.tobytes("png")
        with Image.open(source) as original:
            original.seek(index)
            sx, sy = original.width / page["width"], original.height / page["height"]
            crop = original.convert("RGB").crop(
                (round(x * sx), round(y * sy), round((x + width) * sx), round((y + height) * sy))
            )
            crop.thumbnail((1200, 1200))
            output = io.BytesIO()
            crop.save(output, format="PNG")
            return output.getvalue()

    def close(self) -> None:
        for server, thread, _url in list(self.review_servers.values()):
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        self.review_servers.clear()


def make_handler(workbench: Workbench) -> type[BaseHTTPRequestHandler]:
    class WorkbenchHandler(BaseHTTPRequestHandler):
        server_version = "DiagExWorkbench/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            if args and str(args[1]).startswith(("4", "5")):
                super().log_message(fmt, *args)

        def _headers(self, status: int, content_type: str, length: int) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; style-src 'self'; "
                "script-src 'self'; connect-src 'self'",
            )
            self.end_headers()

        def _send_json(self, value: Any, status: int = HTTPStatus.OK) -> None:
            payload = _json_bytes(value)
            self._headers(status, "application/json; charset=utf-8", len(payload))
            self.wfile.write(payload)

        def _send_file(self, path: Path, content_type: str | None = None) -> None:
            candidate = path.resolve()
            static_root = STATIC_DIR.resolve()
            if not candidate.is_file() or candidate.parent != static_root:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            payload = candidate.read_bytes()
            mime = (
                content_type
                or mimetypes.guess_type(candidate.name)[0]
                or "application/octet-stream"
            )
            self._headers(HTTPStatus.OK, mime, len(payload))
            self.wfile.write(payload)

        def _route(self) -> str:
            return unquote(urlsplit(self.path).path)

        def _query(self) -> dict[str, list[str]]:
            return parse_qs(urlsplit(self.path).query, keep_blank_values=True)

        def _read_json(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise WorkbenchError("invalid Content-Length") from exc
            if length <= 0 or length > MAX_JSON_BYTES:
                raise WorkbenchError("JSON body must be between 1 byte and 1 MiB")
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise WorkbenchError("request body must be a JSON object")
            return value

        def do_GET(self) -> None:  # noqa: N802
            try:
                self._handle_get()
            except ReviewConflict as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
            except (WorkbenchError, ValueError) as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:  # noqa: BLE001
                self._send_json(
                    {"error": f"internal workbench error: {exc}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def _handle_get(self) -> None:
            route = self._route()
            if route in {"/", "/index.html"}:
                self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
                return
            if route == "/detection-review":
                self._send_file(STATIC_DIR / "detection.html", "text/html; charset=utf-8")
                return
            if route in {"/api/detection-review", "/api/detection-page", "/api/detection-crop"}:
                run_dir = (self._query().get("run_dir") or [""])[0]
                if route == "/api/detection-review":
                    self._send_json(workbench.detection_store(run_dir).public())
                else:
                    index = int((self._query().get("page") or ["0"])[0])
                    if route == "/api/detection-crop":
                        bounds = [int(n) for n in (self._query().get("box") or [""])[0].split(",")]
                        data = workbench.detection_crop(run_dir, index, bounds)
                    else:
                        data = workbench.detection_page(run_dir, index)
                    self._headers(HTTPStatus.OK, "image/png", len(data))
                    self.wfile.write(data)
                return
            if route.startswith("/static/"):
                name = route.removeprefix("/static/")
                if not name or "/" in name or "\\" in name or name.startswith("."):
                    self._send_json({"error": "invalid static path"}, HTTPStatus.BAD_REQUEST)
                    return
                self._send_file(STATIC_DIR / name)
                return
            if route == "/api/health":
                self._send_json({"ok": True})
                return
            if route == "/api/config":
                self._send_json(workbench.public_config())
                return
            if route == "/api/jobs":
                jobs = [
                    job.public(include_logs=False)
                    for job in sorted(
                        workbench.jobs.values(), key=lambda item: item.created_at, reverse=True
                    )
                ]
                self._send_json({"jobs": jobs})
                return
            if route.startswith("/api/jobs/"):
                job_id = route.removeprefix("/api/jobs/")
                after_text = (self._query().get("after") or ["0"])[0]
                try:
                    after = max(0, int(after_text))
                except ValueError as exc:
                    raise WorkbenchError("after must be an integer") from exc
                self._send_json(workbench.job(job_id).public(after=after))
                return
            if route == "/api/runs":
                self._send_json({"runs": workbench.recent_runs()})
                return
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            route = self._route()
            try:
                if route == "/api/uploads":
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                    except ValueError as exc:
                        raise WorkbenchError("invalid Content-Length") from exc
                    filename = (self._query().get("filename") or [""])[0]
                    upload = workbench.save_upload(
                        filename=filename,
                        content_type=self.headers.get("Content-Type", ""),
                        source=self.rfile,
                        length=length,
                    )
                    self._send_json({"upload": upload.public()}, HTTPStatus.CREATED)
                    return
                body = self._read_json()
                if route == "/api/extractions":
                    job = workbench.start_extraction(body)
                    self._send_json({"job": job.public()}, HTTPStatus.ACCEPTED)
                    return
                if route == "/api/detection-review/actions":
                    store = workbench.detection_store(str(body.get("run_dir") or ""))
                    self._send_json(store.apply(body))
                    return
                if route == "/api/reviews":
                    url = workbench.launch_review(
                        kind=str(body.get("kind") or "graph"),
                        rater=str(body.get("rater") or ""),
                        job_id=str(body.get("job_id") or "") or None,
                        run_dir=str(body.get("run_dir") or "") or None,
                        upload_id=str(body.get("upload_id") or "") or None,
                    )
                    self._send_json({"url": url})
                    return
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (WorkbenchConflictError, ReviewConflict) as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
            except (WorkbenchError, ValueError, KeyError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:  # noqa: BLE001
                self._send_json(
                    {"error": f"internal workbench error: {exc}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )

    return WorkbenchHandler


def serve_workbench(
    workbench: Workbench,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    on_ready: Callable[[str], None] | None = None,
) -> str:
    """Serve the extraction dashboard until interrupted."""
    server = ThreadingHTTPServer((host, port), make_handler(workbench))
    actual_port = int(server.server_address[1])
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{display_host}:{actual_port}/"
    if on_ready is not None:
        on_ready(url)
    if open_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        workbench.close()
    return url
