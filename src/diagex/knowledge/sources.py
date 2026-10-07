"""Explicit run source selection; browsing alone never enables a standard."""

SOURCES = {
    "isa-5.1-2009": {"name": "ISA 5.1:2009", "standard": {"name": "ISA 5.1", "edition": "2009"}},
    "sht-3101-2017": {"name": "SH/T 3101-2017", "standard": {"name": "SH/T 3101", "edition": "2017"}},
    "projectmaterials.pid-symbols": {"name": "Projectmaterials website", "standard": None},
}


def validate_sources(values):
    if values is None:
        return None  # Older callers retain context-based applicability.
    if not isinstance(values, list) or any(not isinstance(v, str) or v not in SOURCES for v in values):
        raise ValueError("Unknown knowledge source selection")
    return sorted(set(values))


def source_allowed(entry, snapshot):
    selected = snapshot.get("selected_sources")
    if selected is None:
        return True
    document = entry.get("source", {}).get("document_id")
    return bool(selected) and (document is None or document in selected)
