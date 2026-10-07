"""Opt-in raster text recognition through an injected, accounted model client.

This backend reads literal text only. It neither names symbols nor assigns tags.
It is not enabled implicitly by the existing full extraction workflow.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from diagex.vision.encode import encode_image_block
from diagex.vision.evidence import TextEvidence, stable_evidence_id
from diagex.vision.models import BBox


class Word(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    # Coordinates are normalized to the supplied page image, not symbol boxes.
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)


class Transcription(BaseModel):
    model_config = ConfigDict(extra="forbid")
    words: list[Word]


class VLMTextRecognizer:
    def __init__(self, *, client, cost_tracker, max_tokens=6000):
        self.client, self.cost_tracker, self.max_tokens = client, cost_tracker, max_tokens

    def recognize(self, *, page, image):
        tool = {
            "name": "submit_text",
            "description": "Transcribe visible text and its boxes",
            "input_schema": Transcription.model_json_schema(),
        }
        response = self.client.messages_create(
            system=(
                "Read the literal printed text in the image, including tags and annotations. "
                "Return one box per word in normalized image coordinates. Do not infer missing "
                "letters, engineering classes or relationships. Omit illegible text."
            ),
            messages=[{"role": "user", "content": [encode_image_block(image)]}],
            tools=[tool],
            tool_choice={"type": "tool", "name": "submit_text"},
            max_tokens=self.max_tokens,
            thinking={"type": "disabled"},
            reasoning_mode_override="disabled",
        )
        self.cost_tracker.record(
            response,
            step=len(self.cost_tracker.steps) + 1,
            tile_id="text_detection",
            page_index=page.page_index,
        )
        calls = [
            b.input for b in response.content if b.type == "tool_use" and b.name == "submit_text"
        ]
        if len(calls) != 1:
            raise ValueError("Exactly one submit_text response is required")
        words = Transcription.model_validate(calls[0]).words
        spans = []
        for index, word in enumerate(words):
            if word.x + word.width > 1.000001 or word.y + word.height > 1.000001:
                raise ValueError("Text box lies outside the page")
            left, top = round(word.x * page.width), round(word.y * page.height)
            box = BBox(
                x=left,
                y=top,
                w=max(1, round((word.x + word.width) * page.width) - left),
                h=max(1, round((word.y + word.height) * page.height) - top),
            )
            spans.append(
                TextEvidence(
                    id=stable_evidence_id(
                        "ocr", page.page_index, index, word.text, box.model_dump_json()
                    ),
                    text=word.text,
                    bbox=box,
                    origin="raster_vlm",
                )
            )
        return spans
