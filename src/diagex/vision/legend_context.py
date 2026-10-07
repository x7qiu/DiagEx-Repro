"""Deterministic, bounded legend context for one raw-symbol view.

Text abbreviations have a separate budget. Graphical definitions are selected
fairly across candidate families, so a large abbreviation table cannot suppress
continuation or instrument conventions. Image references share the entry IDs.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from collections import deque
from typing import Any

from PIL import Image

from diagex.vision.encode import encode_image_block
from diagex.vision.legend_models import normalise_legend_role
from diagex.vision.symbol_candidates import SymbolCandidate

MAX_GRAPHICAL_ENTRIES = 32
MAX_LEGEND_IMAGES = 12
MAX_ABBREVIATIONS = 16
SHAPES = (
    "continuation_arrow",
    "valve_body",
    "open_inline_valve",
    "instrument_frame",
    "round_symbol",
    "capsule_body",
    "connected_frame",
)


def select_graph_legend_context(entries, node_labels, *, limit=80):
    """Reserve space for each graphical kind before adding relevant abbreviations."""
    tokens = _tokens(" ".join(str(label or "") for label in node_labels))
    groups = {}
    for entry in entries:
        attrs = entry.get("attributes") or {}
        if attrs.get("row_status") in {"reject", "uncertain"}:
            continue
        abbreviation = attrs.get("legend_kind") == "abbreviation"
        if abbreviation and str(entry.get("label", "")).upper() not in tokens:
            continue
        group = "abbreviation" if abbreviation else entry.get("kind", "other")
        groups.setdefault(group, []).append(entry)
    queues = [deque(sorted(group, key=lambda e: (-len(_tokens(str(e.get("label", ""))) & tokens), str(e.get("label", "")))))
              for key, group in sorted(groups.items(), key=lambda item: (item[0] == "abbreviation", item[0]))]
    selected = []
    while any(queues) and len(selected) < limit:
        for queue in queues:
            if queue and len(selected) < limit:
                selected.append(queue.popleft())
    return selected


def _tokens(text: str) -> set[str]:
    # Chinese compound descriptions have no spaces: 往复压缩机 and 螺杆压缩机
    # share useful printed evidence. Latin instrument codes still match exactly.
    tokens = set(re.findall(r"[A-Z]+[A-Z0-9]*|[\u4e00-\u9fff]+", text.upper()))
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.update(run[i : i + n] for n in (2, 3) for i in range(len(run) - n + 1))
    return tokens


def _families(entry: dict[str, Any]) -> set[str]:
    attrs = entry.get("attributes") or {}
    kind, cls = entry.get("kind"), entry.get("symbol_class", "")
    # Shape hints describe visual resemblance, not the role of a definition.
    # In particular, curled line-example ends and scope-boundary corners have
    # been mislabeled continuation_arrow. Keep those definitions available to
    # line/graph consumers, but never promote them into object exemplars here.
    roles = {
        entry.get("reference_type"),
        attrs.get("reference_type"),
        attrs.get("symbol_role"),
        attrs.get("line_type"),
    }
    if roles & {"line_style", "scope_boundary", "boundary", "line_annotation"} or cls in {
        "boundary_line", "scope_boundary", "line_style", "line_crossing", "line_annotation"
    }:
        return set()
    if kind == "line":
        # Older packs sometimes stored an explicitly typed off-page endpoint
        # as a line. Preserve that supported distinction; generic arrow shape
        # alone cannot override a line definition's role.
        if cls in {"opc", "off_page_connector", "offpage_connector", "drawing_reference"}:
            return {"continuation_arrow"}
        return set()
    try:
        explicit = json.loads(attrs.get("candidate_shapes", "[]"))
    except (TypeError, ValueError):
        explicit = []
    if isinstance(explicit, list) and (
        families := set(s for s in explicit if isinstance(s, str)) & set(SHAPES)
    ):
        # These are retrieval hints, not semantic exclusions. A source legend
        # may depict a connected compressor assembly while the native detector
        # isolates its capsule-shaped body. Retain both competing definitions.
        if (
            entry.get("kind") == "equipment"
            and attrs.get("symbol_role") != "actuator"
            and families & {"capsule_body", "connected_frame"}
        ):
            families |= {"capsule_body", "connected_frame"}
        if "valve_body" in families:
            families.add("open_inline_valve")
        return families
    if kind == "connector" or cls == "opc":
        return {"continuation_arrow"}
    if kind == "valve" or "valve" in cls or attrs.get("valve_type"):
        return {"valve_body", "open_inline_valve"}
    if kind == "instrument":
        return {"round_symbol", "instrument_frame"}
    if kind == "equipment":
        return {"round_symbol", "capsule_body", "connected_frame"}
    return set()


def cls_check(entry):
    return entry.get("symbol_class") == "check_valve"


def select_legend_context(
    entries: list[dict[str, Any]],
    candidates: list[SymbolCandidate],
    native_text: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    """Return compact entries, image blocks, and per-candidate reference IDs."""
    tokens = _tokens(" ".join([*native_text, *(text for c in candidates for text in c.text)]))
    families = {c.shape for c in candidates} | {"valve_body", "continuation_arrow", "round_symbol", "capsule_body"}
    graphical, abbreviations = [], []
    for entry in entries:
        entry = normalise_legend_role(entry)
        attrs = entry.get("attributes") or {}
        if entry.get("source") != "customer_override" and attrs.get("row_status") in {
            "uncertain",
            "reject",
        }:
            continue
        if str(entry.get("crop_quality") or "").startswith("rejected"):
            continue
        label = str(entry.get("label") or "")
        identity = (
            entry.get("source_row_id")
            or hashlib.sha256(
                json.dumps(
                    {
                        k: entry.get(k)
                        for k in (
                            "source",
                            "label",
                            "kind",
                            "symbol_class",
                            "source_page_index",
                            "source_bbox",
                        )
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()[:16]
        )
        item = {
            k: entry.get(k)
            for k in (
                "label",
                "kind",
                "symbol_class",
                "description",
                "source_page_index",
                "source_bbox",
                "source",
                "standard",
            )
        }
        item.update(
            legend_entry_id=str(identity),
            attributes={
                k: v
                for k, v in attrs.items()
                if k
                not in {
                    "classification_evidence",
                    "classification_repair_evidence",
                    "source_path_ids",
                    "source_text_ids",
                    "previous_rejection_reason",
                }
            },
        )
        # Exact printed codes only; substring matching would make P match every PI.
        if attrs.get("legend_kind") == "abbreviation":
            if label.strip().upper() in tokens:
                abbreviations.append(item)
            continue
        applicable = _families(entry) & families
        if not applicable:
            continue
        specific = any(c.shape == "open_inline_valve" for c in candidates) and (
            attrs.get("valve_type") == "check" or cls_check(entry)
        )
        rank = (
            not specific,
            -len(_tokens(label) & tokens),
            entry.get("source") != "customer_override",
            entry.get("source") != "legend_extracted",
            not bool(entry.get("image_b64")),
            str(identity),
        )
        graphical.append((rank, item, applicable, entry.get("image_b64")))
    queues = []
    for shape in SHAPES:
        if shape not in families:
            continue
        ranked, repetitions = [], {}
        for g in sorted((g for g in graphical if shape in g[2]), key=lambda g: g[0]):
            attrs = g[1]["attributes"]
            signature = (
                g[1]["kind"],
                g[1]["symbol_class"],
                attrs.get("symbol_role"),
                attrs.get("valve_type"),
                attrs.get("instrument_function"),
            )
            count = repetitions.get(signature, 0)
            repetitions[signature] = count + 1
            # Relevant text/override/source quality keep priority. Within a
            # tie, provide competing classes before repeating one class.
            ranked.append(((*g[0][:-1], count, g[0][-1]), g))
        queues.append(deque(g for _, g in sorted(ranked, key=lambda pair: pair[0])))
    selected, seen = [], set()
    while any(queues) and len(selected) < MAX_GRAPHICAL_ENTRIES:
        for queue in queues:
            while queue and queue[0][1]["legend_entry_id"] in seen:
                queue.popleft()
            if queue and len(selected) < MAX_GRAPHICAL_ENTRIES:
                g = queue.popleft()
                seen.add(g[1]["legend_entry_id"])
                selected.append(g)
    blocks, images = [], 0
    for _, item, _, encoded in selected:
        if not encoded or images >= MAX_LEGEND_IMAGES:
            continue
        try:
            with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as original:
                original.thumbnail((256, 256))
                block = encode_image_block(original)
        except (ValueError, OSError):
            continue
        item["has_reference_image"] = True
        blocks.extend(
            [
                {
                    "type": "text",
                    "text": f"Legend reference {item['legend_entry_id']}: {item['label']} (definition, not a detected object)",
                },
                block,
            ]
        )
        images += 1
    candidate_refs = {
        c.id: [g[1]["legend_entry_id"] for g in selected if c.shape in g[2]] for c in candidates
    }
    abbreviations.sort(key=lambda item: (item["label"], item["legend_entry_id"]))
    return [*(g[1] for g in selected), *abbreviations[:MAX_ABBREVIATIONS]], blocks, candidate_refs
