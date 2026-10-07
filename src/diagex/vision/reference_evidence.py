"""Validate reported reference use without treating supplied context as a match."""

from pydantic import BaseModel, ConfigDict


class KnowledgeCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reference_id: str
    variant_id: str | None = None
    drawing_evidence: str


def knowledge_matches(context, obj):
    from diagex.knowledge.library import validate_match
    citations = [(c.reference_id, c.variant_id, c.drawing_evidence) for c in obj.knowledge_citations]
    citations.append((obj.knowledge_reference_id, obj.knowledge_variant_id, obj.knowledge_evidence))
    matches, errors = [], []
    for identity, variant, evidence in citations:
        match, error = validate_match(context, identity, variant, evidence)
        if match and match not in matches:
            matches.append(match)
        if error:
            errors.append({"id": identity, "reason": error})
    return matches, errors


def legend_matches(ids, evidence, entries):
    supplied = {e["legend_entry_id"]: e for e in entries}
    matches, errors = [], []
    for identity in dict.fromkeys(ids):
        entry = supplied.get(identity)
        if entry is None:
            errors.append({"id": identity, "reason": "legend_not_supplied"})
        elif not evidence or not evidence.strip():
            errors.append({"id": identity, "reason": "missing_drawing_evidence"})
        else:
            matches.append({
                "legend_entry_id": identity, "label": entry.get("label"),
                "source": entry.get("source"), "standard": entry.get("standard"),
                "source_page_index": entry.get("source_page_index"),
                "source_bbox": entry.get("source_bbox"), "drawing_evidence": evidence,
                "status": "model_reported",
            })
    return matches, errors
