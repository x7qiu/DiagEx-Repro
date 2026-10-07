"""Reference-guided candidate retention, requiring independently located source ink."""

import math
import re


def connector_exception(detection, page, context):
    refs = [
        e["id"]
        for e in context.get("references", [])
        if e.get("behavior") == "inspect_connector_exception"
    ]
    if not refs:
        return None
    box = detection.bbox
    tolerance = max(3, min(box.w, box.h) * 0.15)

    def inside(point):
        return (
            box.x - tolerance <= point[0] <= box.x2 + tolerance
            and box.y - tolerance <= point[1] <= box.y2 + tolerance
        )

    # Closed outlines must cover the detected glyph, not unrelated tiny nearby ink.
    glyphs = [
        p
        for p in page.paths
        if p.closed
        and len(set(p.points)) == 5
        and p.primitive != "curve"
        and all(inside(point) for point in p.points)
        and p.bbox.w >= box.w * 0.6
        and p.bbox.h >= box.h * 0.6
    ]
    lines = [
        p
        for p in page.paths
        if not p.closed
        and len(p.points) >= 2
        and any(inside(point) for point in (p.points[0], p.points[-1]))
        and any(not inside(point) for point in (p.points[0], p.points[-1]))
    ]
    nearby = [
        s
        for s in page.text_spans
        if math.hypot(
            (s.bbox.x + s.bbox.w / 2) - (box.x + box.w / 2),
            (s.bbox.y + s.bbox.h / 2) - (box.y + box.h / 2),
        )
        <= max(box.w, box.h) * 2 + 30
    ]
    # A line label or utility prior alone is insufficient: need continuation language.
    labels = [
        s
        for s in nearby
        if re.search(r"\b(?:to|from|sheet|drawing|continuation)\b|至|来自|接图|续图", s.text, re.I)
    ]
    if not (glyphs and lines and labels):
        return None
    return {
        "reference_ids": refs,
        "knowledge_identity": context["knowledge_identity"],
        "profile_version": context.get("profile_version"),
        "glyph_path_ids": [p.id for p in glyphs],
        "attached_path_ids": [p.id for p in lines],
        "text_ids": [s.id for s in labels],
        "status": "requires_review",
        "reason": "Supported placement exception: source outline, attached line and continuation text. Destination remains unverified.",
    }
