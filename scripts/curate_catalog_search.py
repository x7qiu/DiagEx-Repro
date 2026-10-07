"""Apply explicit search vocabulary and remove imported source-heading pollution."""

from pathlib import Path

import yaml

from diagex.knowledge.models import Entry

ROOT = Path(__file__).resolve().parents[1] / "knowledge" / "pid"


def curate(root=ROOT):
    recipe = yaml.safe_load((root / "recipes/catalog-search.yaml").read_text())
    vocabulary = {identity: family["terms"] for family in recipe["families"]
                  for identity in family["reference_ids"]}
    vocabulary.update({identity: [] for identity in recipe["excluded_from_family"]})
    if any(not (root / f"{identity}.yaml").is_file() for identity in vocabulary):
        raise ValueError("Search curation references an unknown entry")
    changed = 0
    for path in sorted(root.glob("*.yaml")):
        old = yaml.safe_load(path.read_text())
        entry = Entry.model_validate(old).model_dump(mode="json", exclude_none=True)
        if entry["id"] not in vocabulary and entry["source"].get("document_id") != "projectmaterials.pid-symbols":
            continue
        if entry["id"] in vocabulary:
            entry["catalog"]["search_terms"] = vocabulary[entry["id"]]
        if entry["source"].get("document_id") == "projectmaterials.pid-symbols":
            section = entry["source"].get("section")
            entry["tags"] = [tag for tag in entry["tags"] if tag != section]
            inherited = f"Projectmaterials labels this illustration ‘{entry['concept']}’ ({section})."
            clean = f"Projectmaterials labels this illustration ‘{entry['concept']}’."
            if entry["explanation"] == inherited:
                entry["explanation"] = clean
            if entry["catalog"]["interpretation"] == inherited:
                entry["catalog"]["interpretation"] = clean
        # Compare normalized records, so optional schema defaults do not cause version churn.
        if entry != Entry.model_validate(old).model_dump(mode="json", exclude_none=True):
            entry["version"] += 1
            path.write_text(yaml.safe_dump(entry, sort_keys=False, allow_unicode=True, width=100))
            changed += 1
    return changed


if __name__ == "__main__":
    print(f"Updated {curate()} reference records")
