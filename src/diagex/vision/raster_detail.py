"""Extra source-image detail sheets for the explicit raster-review experiment."""
from __future__ import annotations

import copy
import hashlib
import math

from PIL import Image, ImageDraw, ImageOps

from diagex.vision.encode import encode_image_block
from diagex.vision.raster_review import RasterReviewClient

CELL_SIZE = 224
CAPTION_HEIGHT = 24
SHEET_COLUMNS = 4
PER_SHEET = 16

DETAIL_INSTRUCTION = """Additional images are contact sheets of source crops around
the supplied raster proposals. Each caption is a generated proposal_id outside
the source ink. Inspect the glyph near the center of that crop; surrounding ink
is context. These crops do not introduce new candidates, legend definitions, or
coordinate systems. Return every bbox normalized to the FIRST detail image,
never to a contact sheet or individual crop. Every decision="symbol" row MUST
include both kind and bbox; a proposal_id or crop does not replace the required
box. You may repeat a hint's normalized box only after verifying it against the
first source image; otherwise correct it. Keep the same decision contract."""


def detail_sheets(source, guides):
    if len(guides) > 80:
        raise ValueError("Detail sheets must use the same bounded 80-guide inventory")
    if len({r["proposal_id"] for r in guides}) != len(guides):
        raise ValueError("Duplicate raster guide ID")
    source = source.convert("RGB")
    result = []
    for start in range(0, len(guides), PER_SHEET):
        group = guides[start:start + PER_SHEET]
        cols = min(SHEET_COLUMNS, len(group))
        sheet = Image.new("RGB", (cols * CELL_SIZE, math.ceil(len(group) / cols) * CELL_SIZE), "white")
        painter = ImageDraw.Draw(sheet)
        cells = []
        for index, guide in enumerate(group):
            b = guide["bbox_normalized"]
            values = [b[k] for k in ("x", "y", "w", "h")]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
                raise ValueError("Invalid raster guide coordinates")
            x, y, w, h = values
            if not (0 <= x < 1 and 0 <= y < 1 and w > 0 and h > 0 and x + w <= 1 + 1e-9 and y + h <= 1 + 1e-9):
                raise ValueError("Raster guide lies outside the source view")
            x, y, w, h = x * source.width, y * source.height, w * source.width, h * source.height
            pad = max(12, w / 2, h / 2)
            box = [max(0, math.floor(x - pad)), max(0, math.floor(y - pad)),
                   min(source.width, math.ceil(x + w + pad)), min(source.height, math.ceil(y + h + pad))]
            crop = source.crop(box)
            display = ImageOps.contain(crop, (CELL_SIZE - 16, CELL_SIZE - CAPTION_HEIGHT - 16), Image.Resampling.BICUBIC)
            # Upscale small glyphs while preserving aspect ratio; contain() can enlarge.
            cx, cy = index % cols * CELL_SIZE, index // cols * CELL_SIZE
            caption = guide["proposal_id"] + (" [clipped]" if guide.get("partially_visible") else "")
            painter.text((cx + 8, cy + 5), caption, fill="black")
            dx = cx + (CELL_SIZE - display.width) // 2
            dy = cy + CAPTION_HEIGHT + (CELL_SIZE - CAPTION_HEIGHT - display.height) // 2
            sheet.paste(display, (dx, dy))
            cells.append({"proposal_id": guide["proposal_id"], "source_view_bbox_pixels": box,
                          "source_crop_size": list(crop.size), "display_size": list(display.size),
                          "display_origin": [dx, dy], "caption_band_xyxy": [cx, cy, cx + CELL_SIZE, cy + CAPTION_HEIGHT],
                          "source_crop_rgb_sha256": hashlib.sha256(crop.tobytes()).hexdigest()})
        result.append((sheet, {"size": list(sheet.size), "cells": cells,
                               "sheet_rgb_sha256": hashlib.sha256(sheet.tobytes()).hexdigest()}))
    return result


class RasterDetailClient(RasterReviewClient):
    def begin_raster_view(self, guidance, source_image):
        super().begin_raster_view(guidance)
        sheets = detail_sheets(source_image, guidance["raster_proposal_guidance"]["guides"])
        self.raster_detail_blocks = []
        self.raster_detail_manifest = []
        for sheet, record in sheets:
            block = encode_image_block(sheet)
            record["encoded_base64_sha256"] = hashlib.sha256(block["source"]["data"].encode()).hexdigest()
            self.raster_detail_manifest.append(record)
            self.raster_detail_blocks.extend([{"type": "text", "text": DETAIL_INSTRUCTION}, block])

    def messages_create(self, **kwargs):
        tools = kwargs.get("tools", [])
        if len(tools) == 1 and tools[0].get("name") == "submit_pid_objects":
            if not hasattr(self, "raster_detail_blocks"):
                raise ValueError("Source detail sheets must be prepared before requesting review")
            messages = copy.deepcopy(kwargs["messages"])
            if not messages or messages[0]["role"] != "user" or not isinstance(messages[0]["content"], list):
                raise ValueError("Raster review requires an image-bearing user message")
            messages[0]["content"].extend(self.raster_detail_blocks)
            kwargs = {**kwargs, "messages": messages}
        return super().messages_create(**kwargs)
