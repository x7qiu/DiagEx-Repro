"""Local CV preparation for the workbench; no language-model calls or credentials."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from diagex.extractors.evidence_checkpoint import atomic_write_json


def defaults():
    checkpoint = os.environ.get("DIAGEX_CV_CHECKPOINT", "")
    if not checkpoint:
        candidate = Path("output/pid2graph-phase2/checkpoint-export-epoch006/detector.pt")
        if candidate.is_file():
            checkpoint = str(candidate.resolve())
    return {
        "mode": "off",
        "checkpoint": checkpoint,
        "device": "cpu",
        "python": os.environ.get("DIAGEX_CV_PYTHON") or sys.executable,
    }


def settings(body):
    mode = body.get("cv_mode", "off")
    if mode not in {"off", "guided", "broad_review"}:
        raise ValueError("CV mode must be off, guided, or broad_review")
    if mode == "off":
        return {"mode": "off"}
    checkpoint = str(body.get("cv_checkpoint") or defaults()["checkpoint"]).strip()
    if not checkpoint or not Path(checkpoint).expanduser().is_file():
        raise ValueError("CV requires an existing local detector checkpoint (.pt)")
    device = body.get("cv_device", "cpu")
    if device not in {"cpu", "mps", "cuda"}:
        raise ValueError("CV device must be cpu, mps, or cuda")
    from diagex.vision.stages import file_hash

    checkpoint = Path(checkpoint).expanduser().resolve()
    python = Path(defaults()["python"]).expanduser().absolute()
    # Preserve a virtualenv's executable path; resolving its symlink loses the venv.
    if not python.is_file():
        raise ValueError("DIAGEX_CV_PYTHON must name an existing Python executable")
    return {
        "mode": mode,
        "checkpoint": str(checkpoint),
        "device": device,
        "checkpoint_sha256": file_hash(checkpoint),
        "python": str(python),
    }


def prepare(source, config, selected, folder, log):
    """Run in an optional dedicated Torch environment, then validate the result."""
    from diagex.vision.pdf_raster_guidance import load_pdf_guidance
    from diagex.vision.raster_guidance import load_guidance

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    request = folder / "request.json"
    output = folder / "proposals.json"
    atomic_write_json(
        request,
        {
            "source": str(Path(source).resolve()),
            "checkpoint": selected["checkpoint"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "device": selected["device"],
            "tiling": asdict(config.tiling),
            "scan": asdict(config.scan),
        },
    )
    log(f"Running CV symbol detector on {selected['device']} before VLM interpretation")
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    )
    try:
        process = subprocess.run(
            [
                selected["python"],
                "-m",
                "diagex.web.cv",
                "--request",
                str(request),
                "--out",
                str(output),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "CV preparation exceeded its one-hour limit; VLM extraction was not started"
        ) from exc
    if process.returncode:
        detail = (process.stderr or process.stdout)[-2500:]
        raise RuntimeError(
            "CV preparation failed; VLM extraction was not started. Check the checkpoint, device, "
            "and DIAGEX_CV_PYTHON environment (Torch/Torchvision required).\n" + detail
        )
    payload = (
        load_pdf_guidance(output, source, config)
        if Path(source).suffix.lower() == ".pdf"
        else load_guidance(output, source)
    )
    if payload["detector"]["detector_sha256"] != selected["checkpoint_sha256"]:
        raise ValueError("CV proposal checkpoint differs from the selected weights")
    count = (
        sum(len(p["predictions"]) for p in payload["pages"])
        if "pages" in payload
        else len(payload["predictions"])
    )
    log(f"CV completed: {count} symbol proposals; starting legend and VLM interpretation")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.request.read_text())
    from diagex.config import Config, LLMConfig, ScanConfig, TilingConfig
    from diagex.vision.pdf_raster_guidance import generate
    from diagex.vision.raster_detector import RasterDetector

    detector = RasterDetector(
        spec["checkpoint"], device=spec["device"], expected_sha256=spec["checkpoint_sha256"]
    )
    config = Config(
        llm=LLMConfig(), tiling=TilingConfig(**spec["tiling"]), scan=ScanConfig(**spec["scan"])
    )
    source = Path(spec["source"])
    payload = (
        generate(source, detector, config)
        if source.suffix.lower() == ".pdf"
        else detector.predict_file(source)
    )
    atomic_write_json(args.out, payload)


if __name__ == "__main__":
    main()
