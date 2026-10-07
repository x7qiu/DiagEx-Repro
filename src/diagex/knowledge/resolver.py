"""Deterministic, bounded retrieval. Context is a prior, never drawing evidence."""

import hashlib
import json
import math
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

from .library import asset_path, load_library, model_assets
from .models import Override, Profile
from .sources import SOURCES, source_allowed, validate_sources

CORE = [
    "Explicit drawing legends and confirmed project definitions take precedence over general references.",
    "Industry and process-unit context guide inspection; they never prove equipment or connections exist.",
    "Keep conflicting applicable definitions visible; do not silently choose one.",
    "Separate drawing-quality recommendations from recognition and preserve uncertainty.",
    "Reference illustrations are teaching material, never source-drawing evidence.",
]


def library_root():
    local = Path(__file__).resolve().parents[3] / "knowledge" / "pid"
    return local if local.is_dir() else Path(__file__).parent / "pid"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def knowledge_snapshot(mode="off", profile=None, overrides=None, source_ids=None):
    if mode not in {"off", "general", "profile"}:
        raise ValueError("Unknown knowledge mode")
    if mode == "off":
        return {}
    selected_sources = validate_sources(source_ids)
    parsed = Profile.model_validate(profile) if mode == "profile" else None
    checked_overrides = [Override.model_validate(value) for value in (overrides or [])]
    if mode != "profile" and checked_overrides:
        raise ValueError("Drawing overrides require a confirmed profile")
    root = library_root().resolve()
    catalog = load_library(root)
    entries = catalog["entries"]
    ids = [e["id"] for e in entries]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate knowledge reference ID")
    replacements = catalog.get("replacements", [])
    retired_ids = {r["reference_id"] for r in replacements}
    if parsed and set(parsed.reference_ids) - set(ids) - retired_ids:
        raise ValueError("Profile selects an unknown knowledge reference")
    snapshot = {
        "schema_version": 2,
        "mode": mode,
        "selected_sources": selected_sources,
        "core_principles": CORE,
        "profile": parsed.model_dump(mode="json") if parsed else None,
        "overrides": [v.model_dump(mode="json", exclude_unset=True) for v in checked_overrides],
        "entries": entries,
        "documents": catalog["documents"],
        "mappings": catalog["mappings"],
        "replacements": replacements,
        "implementation_sha256": hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))
        ).hexdigest(),
    }
    snapshot["identity"] = digest(snapshot)
    return snapshot


def _norm(value):
    return str(value).strip().casefold()


def _applicable(entry, context):
    scope = entry["applicability"]
    for field in ("industry", "company", "project"):
        if scope.get(field) and _norm(scope[field]) != _norm(context.get(field) or ""):
            return False
    if scope.get("unit_types") and not (
        {_norm(x) for x in scope["unit_types"]} & {_norm(x) for x in context.get("unit_types", [])}
    ):
        return False
    for required in scope.get("standards", []):
        if not any(
            _norm(s["name"]) == _norm(required["name"])
            and (not required.get("edition") or s.get("edition") == required["edition"])
            for s in context.get("standards", [])
        ):
            return False
    return True


def _effective_context(snapshot, page_index):
    profile = snapshot.get("profile")
    context = dict(profile["context"]) if profile else {}
    # Selecting a standard explicitly enables that edition for this run. Existing
    # declarations of the same standard (including conflicting editions) win.
    standards = list(context.get("standards", []))
    for identity in snapshot.get("selected_sources") or []:
        standard = SOURCES[identity]["standard"]
        if standard and not any(_norm(s["name"]) == _norm(standard["name"]) for s in standards):
            standards.append(dict(standard))
    context["standards"] = standards
    applied = []
    for override in snapshot.get("overrides", []):
        if not override.get("pages") or page_index + 1 in override["pages"]:
            context.update(override["context"])
            applied.append(override)
    return context, applied


def applicable_references(snapshot, page_index=0):
    """All eligible references for UI inspection, before task/topic retrieval."""
    if not isinstance(page_index, int) or isinstance(page_index, bool) or page_index < 0:
        raise ValueError("Preview sheet must be a positive integer")
    context, applied = _effective_context(snapshot, page_index)
    return {
        "reference_ids": [
            e["id"] for e in snapshot.get("entries", [])
            if source_allowed(e, snapshot) and (e.get("catalog") or {}).get("role") != "background" and _applicable(e, context)
        ],
        "context": context,
        "drawing_overrides": applied,
        "page": page_index + 1,
    }


def _focused_entry(entry):
    """Keep original evidence in the catalog, supply only the curated interpretation."""
    metadata = entry.get("catalog")
    if not metadata:
        return entry  # Historical snapshots and older libraries preserve their contract.
    entry = deepcopy(entry)
    entry["concept"] = metadata.get("short_name") or entry["concept"]
    entry["explanation"] = metadata.get("interpretation") or entry["explanation"]
    legacy_image = entry["source"].get("image") if not entry["assets"] else None
    asset_ids = set(metadata.get("displayed_asset_ids", [])) | {metadata.get("model_asset_id")}
    entry["assets"] = [a for a in entry["assets"] if a["id"] in asset_ids]
    # New visual entries never fall back to a full source row. Preserve old single-image entries.
    legacy_image = legacy_image if not metadata.get("displayed_asset_ids") else None
    entry["source"] = {**entry["source"], "passage": None, "image": legacy_image}
    entry["notes"] = []  # Explicit supporting references carry the curated source notes.
    entry["table_data"] = []
    # Named matrix cells are interpretation rules, not the raw imported table.
    # Keep column identities, source wording and notes; omit only empty defaults.
    if entry.get("letter_matrix"):
        for note in entry["letter_matrix"]["notes"]:
            # The snapshot retains full provenance. Avoid repeating document titles
            # and image paths for every note within the bounded model context.
            same_document = note["source"].get("document_id") == entry["source"].get("document_id")
            keys = {"pdf_page", "page"} if same_document else {"document_id", "pdf_page", "page", "passage"}
            note["source"] = {k: v for k, v in note["source"].items() if k in keys and v is not None}
        for row in entry["letter_matrix"]["rows"]:
            # Row links are for browsing; only supplied entry IDs are citable.
            row.pop("reference_id", None)
            for cell in row["cells"].values():
                if not cell.get("footnotes"):
                    cell.pop("footnotes", None)
                if cell.get("text") is None:
                    cell.pop("text", None)
    entry["relations"] = [
        r for r in entry.get("relations", [])
        if r["reference_id"] in metadata.get("supporting_reference_ids", [])
    ]
    return entry


def resolve(snapshot, task, page_index=0, query="", drawing_definitions=None):
    if not snapshot:
        return {}
    profile = snapshot.get("profile")
    context, applied = _effective_context(snapshot, page_index)
    terms = set(re.findall(r"[\w-]+", query.casefold()))
    terms.update(_norm(x) for x in context.get("unit_types", []))
    selected_ids = set(profile.get("reference_ids", [])) if profile else set()
    for replacement in snapshot.get("replacements", []):
        if replacement["reference_id"] in selected_ids:
            selected_ids.update(replacement["replacement_ids"])
    eligible = [
        _focused_entry(e) for e in snapshot["entries"]
        if source_allowed(e, snapshot) and (e.get("catalog") or {}).get("role") != "background"
        and task in e["tasks"] and _applicable(e, context)
    ]
    by_id = {e["id"]: e for e in eligible}

    def bundle(entry):
        return [entry, *[
            by_id[id_] for id_ in (entry.get("catalog") or {}).get("supporting_reference_ids", [])
            if id_ in by_id and id_ != entry["id"]
        ]]
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "for",
        "with",
        "from",
        "by",
        "as",
        "is",
        "are",
    }
    terms -= stop

    def tokens(text):
        return set(re.findall(r"\w+", text.casefold())) - stop

    terms.update(tokens(query))
    indexed = []
    frequency = Counter()
    replacement_labels = {}
    for replacement in snapshot.get("replacements", []):
        for target in replacement["replacement_ids"]:
            replacement_labels.setdefault(target, []).append(replacement["source_label"])
    for entry in eligible:
        heading = tokens(entry["concept"] + " " + " ".join(entry.get("aliases", [])))
        # Search aliases only: website captions never replace the ISA definition.
        heading.update(tokens(" ".join(replacement_labels.get(entry["id"], []))))
        body = tokens(entry["explanation"])
        tags = tokens(" ".join([*entry["tags"], *(entry.get("catalog") or {}).get("search_terms", [])]))
        frequency.update(heading | body | tags)
        indexed.append((entry, heading, body, tags))
    scored = []
    for entry, heading, body, tags in indexed:
        score = 0.0
        for term in terms & (heading | body | tags):
            rarity = math.log(1 + len(eligible) / (1 + frequency[term]))
            score += rarity * (
                4 if term in heading else 2 if term in tags else 1 / (1 + len(body) / 120)
            )
        if entry["id"] in selected_ids:
            score += 100
        if score:
            scored.append((score, entry))
    # Bound supplied text as the library grows. Keep each chosen definition and
    # its notes intact rather than truncating away qualifications or exceptions.
    references = []
    text_size = 0
    for _, entry in sorted(scored, key=lambda pair: (-pair[0], pair[1]["id"])):
        additions = [e for e in bundle(entry) if e not in references]
        if len(references) + len(additions) > 8:
            continue
        size = len(json.dumps(additions, ensure_ascii=False))
        if references and text_size + size > 40000:
            continue
        references.extend(additions)
        text_size += size
        if len(references) == 8:
            break
    # Retain all sides of applicable definition conflicts even beyond retrieval cap.
    conflicts = []
    grouped = {}
    for entry in eligible:
        if entry.get("definition_key"):
            grouped.setdefault(entry["definition_key"], []).append(entry)
    for key, group in grouped.items():
        if len({e["definition_value"] for e in group}) > 1:
            conflicts.append({"key": key, "reference_ids": [e["id"] for e in group]})
            for entry in group:
                references.extend(e for e in bundle(entry) if e not in references)
    result = {
        "task": task,
        "page_index": page_index,
        "knowledge_identity": snapshot["identity"],
        "selected_sources": snapshot.get("selected_sources"),
        "profile_id": profile["id"] if profile else None,
        "profile_version": profile["version"] if profile else None,
        "context": context,
        "drawing_overrides": applied,
        "core_principles": snapshot["core_principles"],
        "drawing_definitions": drawing_definitions or [],
        "references": references,
        "reference_ids": [e["id"] for e in references],
        "conflicts": conflicts,
        "trace_semantics": "references supplied, not proof of model reliance",
    }
    result["identity"] = digest(result)
    result["supplied_assets"] = model_assets(result)
    return result


def reference_images(context):
    """Only load catalogued images whose bytes still match the run snapshot."""
    from PIL import Image

    for asset in model_assets(context):
        path = asset_path(library_root(), asset["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
            raise ValueError("Knowledge image changed since run configuration")
        with Image.open(path) as image:
            copy = image.convert("RGB")
            copy.thumbnail((1400, 1400))
            yield copy


def resolve_symbol_context(snapshot, page, candidates, legend_entries, ownership):
    """Retrieve by local glyph family; reserve bounded space across families.

    No family or retrieved reference is a classification. Source/edition checks,
    linked notes and conflicts still pass through the normal resolver.
    """
    if not snapshot:
        return {}
    from diagex.vision.legend_context import select_legend_context

    local = [
        c
        for c in candidates
        if ownership.x <= c.bbox.x + c.bbox.w / 2 < ownership.x2
        and ownership.y <= c.bbox.y + c.bbox.h / 2 < ownership.y2
    ]
    nearby = [t.text for t in page.text_spans if t.bbox.iou(ownership) > 0]
    definitions, _, _ = select_legend_context(legend_entries, local, nearby)
    queries = {
        "open_inline_valve": "止回阀 单向阀 check valve",
        "valve_body": "阀门 valve",
        "continuation_arrow": "connector off page 连接符",
        "round_symbol": "instrument 仪表 connector",
        "instrument_frame": "instrument 仪表",
        "capsule_body": "equipment vessel pump compressor 设备",
        "connected_frame": "equipment 设备",
    }
    shapes = sorted({c.shape for c in local}, key=lambda s: (s != "open_inline_valve", s))
    # No candidates must still allow references for independent visual discovery.
    shapes = shapes or ["valve_body", "round_symbol", "capsule_body", "continuation_arrow"]
    contexts = [
        resolve(snapshot, "symbol_interpretation", page.page_index, queries.get(s, s))
        for s in shapes
    ]
    context = deepcopy(contexts[0])
    refs, seen, size = [], set(), 0
    # Round-robin each family's definitions, keeping linked notes in its bundle.
    for rank in range(8):
        for result in contexts:
            if rank >= len(result["references"]):
                continue
            entry = result["references"][rank]
            linked = (entry.get("catalog") or {}).get("supporting_reference_ids", [])
            bundle = [entry, *[e for e in result["references"] if e["id"] in linked]]
            add = [e for e in bundle if e["id"] not in seen]
            n = len(json.dumps(add, ensure_ascii=False))
            if len(refs) + len(add) > 8 or (refs and size + n > 40000):
                continue
            refs.extend(add)
            seen.update(e["id"] for e in add)
            size += n
    # The resolver intentionally retains both sides of applicable conflicts.
    conflicts = {c["key"]: c for result in contexts for c in result["conflicts"]}
    for result in contexts:
        for entry in result["references"]:
            if entry["id"] not in seen and any(
                entry["id"] in c["reference_ids"] for c in conflicts.values()
            ):
                refs.append(entry)
                seen.add(entry["id"])
    context.update(
        references=refs,
        reference_ids=[e["id"] for e in refs],
        conflicts=list(conflicts.values()),
        drawing_definitions=definitions,
        retrieval_queries=[queries.get(s, s) for s in shapes],
        retrieval_basis="local glyph proposals; family hints are not matches",
    )
    context["identity"] = digest(
        {k: v for k, v in context.items() if k not in {"identity", "supplied_assets"}}
    )
    context["supplied_assets"] = model_assets(context)
    return context


