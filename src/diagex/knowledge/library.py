"""Shared, file-backed reference catalog. No source PDF is needed at runtime."""

import hashlib
import re
from functools import lru_cache
from pathlib import Path

import yaml
from PIL import Image

from .models import Document, Entry, Mapping, OverlapReview


def asset_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path.suffix.lower() != ".png":
        raise ValueError("Reference image is outside the knowledge library or is not PNG")
    if not path.is_file():
        raise ValueError(f"Missing reference image: {relative}")
    return path


def _stamp(path):
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino


@lru_cache(maxsize=4096)
def _yaml_record(path, stamp):
    # File metadata is only a memoization key; the bytes remain schema-validated.
    return yaml.load(path.read_text(), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))


@lru_cache(maxsize=4096)
def _image_record(path, stamp):
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        with Image.open(path) as image:
            return checksum, image.size
    except OSError as exc:
        raise ValueError("Invalid reference image") from exc


def image_hash(root, relative):
    path = asset_path(root, relative)
    return _image_record(path, _stamp(path))[0]


def _records(root, folder, model):
    return [
        model.model_validate(_yaml_record(p, _stamp(p))).model_dump(mode="json")
        for p in sorted((root / folder).glob("*.yaml"))
    ]


def load_library(root):
    root = Path(root).resolve()
    documents = _records(root, "sources", Document)
    docs = {d["id"]: d for d in documents}
    if len(docs) != len(documents):
        raise ValueError("Duplicate source document ID")
    entries = []
    for path in sorted(root.glob("*.yaml")):
        entry = Entry.model_validate(_yaml_record(path, _stamp(path))).model_dump(mode="json")
        for source in [entry["source"], *(n["source"] for n in entry["notes"])]:
            if source.get("document_id") and source["document_id"] not in docs:
                raise ValueError("Unknown source document")
        if entry["source"].get("image"):
            entry["image_sha256"] = image_hash(root, entry["source"]["image"])
        for asset in entry["assets"]:
            if image_hash(root, asset["path"]) != asset["sha256"]:
                raise ValueError(f"Reference image checksum mismatch: {asset['id']}")
            image_path = asset_path(root, asset["path"])
            if _image_record(image_path, _stamp(image_path))[1] != (
                asset["width"],
                asset["height"],
            ):
                raise ValueError("Reference image dimensions do not match manifest")
            if asset["crop"] and asset["crop"]["document_id"] not in docs:
                raise ValueError("Crop references unknown document")
            if asset.get("web"):
                document = docs.get(asset["web"]["document_id"])
                if document is None or document["source_type"] != "web":
                    raise ValueError("Web image references unknown or non-web document")
        entries.append(entry)
    ids = {e["id"] for e in entries}
    if len(ids) != len(entries):
        raise ValueError("Duplicate knowledge reference ID")
    for entry in entries:
        if entry.get("letter_matrix"):
            if any(r.get("reference_id") and r["reference_id"] not in ids
                   for r in entry["letter_matrix"]["rows"]):
                raise ValueError("Unknown letter-row reference")
            if any(n["source"].get("document_id") not in docs
                   for n in entry["letter_matrix"]["notes"]):
                raise ValueError("Unknown letter-note source document")
        if any(r["reference_id"] not in ids for r in entry["relations"]):
            raise ValueError("Unknown related reference")
        metadata = entry.get("catalog") or {}
        supporting = metadata.get("supporting_reference_ids", [])
        if set(supporting) - ids:
            raise ValueError("Unknown supporting reference")
        if any(
            (e.get("catalog") or {}).get("role") != "supporting"
            for e in entries if e["id"] in supporting
        ):
            raise ValueError("Catalog links must target supporting knowledge")
    mappings = _records(root, "mappings", Mapping)
    versions = {(e["id"], e["version"]) for e in entries}
    for mapping in mappings:
        if (mapping["reference_id"], mapping["reference_version"]) not in versions:
            raise ValueError("Mapping references an unknown reference version")
    replacements = load_replacements(root)
    for replacement in replacements:
        if replacement["reference_id"] in ids:
            raise ValueError("Replaced reference is still active; apply the overlap review")
        if not set(replacement["replacement_ids"]) <= ids:
            raise ValueError("Replacement references an unknown active definition")
    return {"entries": entries, "documents": documents, "mappings": mappings,
            "replacements": replacements}


def load_replacements(root):
    replacements = [r for review in _records(Path(root), "overlaps", OverlapReview)
                    for r in review["replacements"]]
    ids = [r["reference_id"] for r in replacements]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate reference replacement")
    return replacements


def catalog_asset(root, entry_id, asset_id=None):
    """Resolve only catalogued assets; never accept a caller-provided file path."""
    root = Path(root).resolve()
    if not re.fullmatch(r"[a-z0-9._-]+", entry_id):
        raise ValueError("Unknown knowledge reference")
    record = (root / f"{entry_id}.yaml").resolve()
    if not record.is_file() and any(
        r["reference_id"] == entry_id for r in load_replacements(root)
    ):
        # Retired records are asset-only history, never new retrieval candidates.
        record = (root / "retired" / f"{entry_id}.yaml").resolve()
    if not record.is_relative_to(root):
        raise ValueError("Reference record is outside the knowledge library")
    entry = None
    if record.is_file():
        entry = Entry.model_validate(_yaml_record(record, _stamp(record))).model_dump(mode="json")
        if entry["id"] != entry_id:
            entry = None
    if entry is None:
        # Legacy catalog files need not be named after their record IDs.
        entry = next((e for e in load_library(root)["entries"] if e["id"] == entry_id), None)
    if entry is None:
        raise ValueError("Unknown knowledge reference")
    if asset_id:
        asset = next((a for a in entry["assets"] if a["id"] == asset_id), None)
        if asset is None:
            raise ValueError("Unknown reference asset")
        path = asset_path(root, asset["path"])
        checksum, size = _image_record(path, _stamp(path))
        if checksum != asset["sha256"] or size != (asset["width"], asset["height"]):
            raise ValueError("Reference image does not match manifest")
        return path
    relative = entry["source"].get("image")
    if not relative:
        raise ValueError("Unknown reference illustration")
    registered = next((a for a in entry["assets"] if a["path"] == relative), None)
    if registered:
        return catalog_asset(root, entry_id, registered["id"])
    path = asset_path(root, relative)
    _image_record(path, _stamp(path))
    return path


def model_assets(context):
    """Two reference illustrations per request, each with explicit identity."""
    selected = []
    seen = set()
    for entry in context.get("references", []):
        assets = entry.get("assets", [])
        metadata = entry.get("catalog")
        if metadata:
            # Supporting prose and background illustrations must not consume the image budget.
            if metadata["role"] == "background" or (
                metadata["role"] == "supporting" and not entry["source"].get("image")
            ):
                continue
            sheet = next((a for a in assets if a["id"] == metadata.get("model_asset_id")), None)
            sheet = sheet or next(
                (a for a in assets if a["id"] in metadata.get("displayed_asset_ids", [])), None
            )
        else:
            sheet = next((a for a in assets if a["role"] == "variant_sheet"), None)
            sheet = sheet or next((a for a in assets if a["role"] == "variant"), None)
        if sheet:
            selected.append(
                {
                    "reference_id": entry["id"],
                    "reference_version": entry["version"],
                    "asset_id": sheet["id"],
                    "path": sheet["path"],
                    "sha256": sheet["sha256"],
                    "variant_ids": sheet["derived_from"] or [sheet["id"]],
                }
            )
        elif entry["source"].get("image"):
            selected.append(
                {
                    "reference_id": entry["id"],
                    "reference_version": entry["version"],
                    "asset_id": "legacy-image",
                    "path": entry["source"]["image"],
                    "sha256": entry["image_sha256"],
                    "variant_ids": [],
                }
            )
        else:
            continue
        key = (selected[-1]["path"], selected[-1]["sha256"])
        if key in seen:
            selected.pop()
            continue
        seen.add(key)
        if len(selected) == 2:
            break
    return selected


def supplied_trace(context):
    if not context:
        return {}
    return {
        "knowledge_identity": context.get("knowledge_identity"),
        "context_identity": context.get("identity"),
        "profile_id": context.get("profile_id"),
        "profile_version": context.get("profile_version"),
        "references": [
            {"id": e["id"], "version": e["version"], "source": {k: e.get("source", {}).get(k) for k in ("document_id", "document", "edition")}} for e in context.get("references", [])
        ],
        "assets": model_assets(context),
        "semantics": "supplied references, not evidence of a match",
    }


def validate_match(context, reference_id, variant_id, evidence):
    """Validate the attribution, not the engineering truth of a model's claim."""
    if not reference_id:
        return None, "missing_reference_id" if variant_id or evidence else None
    entry = next((e for e in context.get("references", []) if e["id"] == reference_id), None)
    if entry is None:
        return None, "reference_not_supplied"
    if not evidence or not evidence.strip():
        return None, "missing_drawing_evidence"
    if variant_id and not any(
        reference_id == a["reference_id"] and variant_id in a["variant_ids"]
        for a in model_assets(context)
    ):
        return None, "variant_not_supplied"
    return {
        "reference_id": reference_id,
        "reference_version": entry["version"],
        "source": {k: entry.get("source", {}).get(k) for k in ("document_id", "document", "edition", "page", "section")},
        "variant_id": variant_id,
        "drawing_evidence": evidence,
        "status": "model_reported",
        "semantics": "ID-validated attribution; not independently verified",
    }, None
