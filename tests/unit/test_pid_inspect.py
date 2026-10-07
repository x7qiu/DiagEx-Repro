from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

import fitz
from PIL import Image, ImageDraw

from diagex.config import Config, LLMConfig
from diagex.extractors.pid_inspect import run_pid_evidence_inspection
from diagex.vision.legend_models import LegendEntry, LegendPack


def _test_symbol_b64() -> str:
    image = Image.new("RGB", (16, 16), "white")
    ImageDraw.Draw(image).rectangle((3, 3, 12, 12), outline="black", width=2)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_pid_evidence_inspection_auto_detects_legend_and_stops_before_perception(
    tmp_path: Path, monkeypatch
) -> None:
    pdf = tmp_path / "drawing.pdf"
    document = fitz.open()
    legend_page = document.new_page(width=500, height=300)
    legend_page.insert_text((40, 40), "INSTRUMENT LEGEND")
    legend_page.insert_text((40, 100), "QZ project quality transmitter")
    pid_page = document.new_page(width=500, height=300)
    pid_page.insert_text((40, 40), "P&ID TEST")
    pid_page.insert_text((100, 120), "PI")
    pid_page.insert_text((94, 140), "00203")
    document.save(pdf)
    document.close()

    captured: dict[str, object] = {}

    class FakeClient:
        retries_total = 0

        def __init__(self, config, budgets=None) -> None:
            self.config = config

    def fake_resolve_legend(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            source="explicit_pages",
            pack=LegendPack(
                source_ref="drawing#pages=0",
                entries=[
                    LegendEntry(
                        label="QZ",
                        description="project quality transmitter",
                        symbol_class="transmitter",
                        kind="instrument",
                        image_b64=_test_symbol_b64(),
                        attributes={
                            "instrument_function": "transmitter",
                            "measured_variable": "analysis",
                        },
                        source="legend_extracted",
                    )
                ],
            ),
        )

    monkeypatch.setattr("diagex.extractors.pid_inspect.LLMClient", FakeClient)
    monkeypatch.setattr("diagex.extractors.pid_legend.resolve_legend", fake_resolve_legend)
    cfg = Config(llm=LLMConfig(model="fake", anthropic_api_key="unused"))
    cfg.runs_dir = tmp_path / "runs"

    result = run_pid_evidence_inspection(
        diagram=pdf,
        symbol_standard="isa-5.1",
        legend_key=None,
        config=cfg,
        console=None,
    )

    assert result.legend_pages == [0]
    assert captured["legend_pages"] == [0]
    assert result.page_roles == {0: "legend", 1: "pid"}
    assert result.learned_entries[0]["label"] == "QZ"
    assert result.learned_entries[0]["has_image"] is True
    assert (result.run_dir / result.learned_entries[0]["image_file"]).is_file()
    assert (result.run_dir / "legend.learned.md").is_file()
    assert (result.run_dir / "legend.learned.json").is_file()
    assert (result.run_dir / "legend.abbreviations.json").is_file()
    assert not (result.run_dir / "legend.review.html").exists()
    assert not (result.run_dir / "graph.json").exists()
    assert not list(result.run_dir.glob("*.dexpi.json"))

    inventory = json.loads(
        (result.run_dir / "evidence" / "native-text-inventory.json").read_text(
            encoding="utf-8"
        )
    )
    assert inventory["inspection_only"] is True
    pi = next(item for item in inventory["items"] if item["normalised_text"] == "PI00203")
    assert pi["candidate_kind"] == "instrument_tag"
    assert pi["expected_kind"] == "instrument"
