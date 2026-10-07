"""Local immutable profile revisions and evidence-backed native-text suggestions."""

import fcntl
import hashlib
import os
import re
import tempfile
import uuid
from pathlib import Path

from .models import Profile


class ProfileStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self):
        latest = {}
        for path in sorted(self.root.glob("*.json")):
            profile = Profile.model_validate_json(path.read_text())
            if profile.version > latest.get(profile.id, {}).get("version", 0):
                latest[profile.id] = profile.model_dump(mode="json")
        return sorted(latest.values(), key=lambda p: p["name"].casefold())

    def get(self, profile_id, version):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", str(profile_id)):
            raise ValueError("Invalid context profile ID")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise ValueError("Select a confirmed context profile version")
        path = self.root / f"{profile_id}.{version}.json"
        if not path.is_file():
            raise ValueError("Unknown context profile revision")
        return Profile.model_validate_json(path.read_text()).model_dump(mode="json")

    def save(self, body):
        if body.get("confirmed") is not True:
            raise ValueError("Confirm project context before saving")
        profile_id = body.get("id") or "context-" + uuid.uuid4().hex[:12]
        with (self.root / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            existing = next((p for p in self.list() if p["id"] == profile_id), None)
            version = existing["version"] if existing else 0
            if body.get("expected_version", 0) != version:
                raise ValueError("Context profile changed; reload before confirming")
            profile = Profile.model_validate(
                {
                    k: v
                    for k, v in {**body, "id": profile_id, "version": version + 1}.items()
                    if k != "expected_version"
                }
            )
            path = self.root / f"{profile.id}.{profile.version}.json"
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.root, suffix=".tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(profile.model_dump_json(indent=2))
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        return profile.model_dump(mode="json")


LABELS = {
    "industry": r"(?:industry|行业)",
    "unit_types": r"(?:process units?|unit types?|装置类型|装置)",
    "company": r"(?:company|公司)",
    "project": r"(?:project|项目)",
    "standards": r"(?:drawing standard|applicable standard|采用标准|制图标准)",
}


def suggest(source):
    import fitz

    path = Path(source)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    suggestions = []
    with fitz.open(path) as document:
        for index, page in enumerate(document):
            for line in page.get_text().splitlines():
                for field, label in LABELS.items():
                    match = re.fullmatch(rf"\s*{label}\s*[:：]\s*(.{{1,200}})", line, re.I)
                    if not match:
                        continue
                    text = match[1].strip()
                    value = text
                    if field == "unit_types":
                        value = [v.strip() for v in re.split(r"[,，;；]", text) if v.strip()]
                    if field == "standards":
                        # Only parse an explicitly printed edition/year; never industry-derived.
                        standard = re.fullmatch(r"(.+?)(?:\s*[:：]\s*|\s+)(\d{4})", text)
                        value = [
                            {
                                "name": standard[1].strip() if standard else text,
                                "edition": standard[2] if standard else None,
                            }
                        ]
                    suggestions.append(
                        {
                            "field": field,
                            "value": value,
                            "source": path.name,
                            "source_sha256": sha,
                            "page": index + 1,
                            "passage": line,
                            "status": "unconfirmed",
                        }
                    )
    return {
        "suggestions": suggestions,
        "source_sha256": sha,
        "notice": "Suggestions use explicitly labelled native PDF text only. Unlabelled or scanned context must be entered manually; standards are never inferred from industry.",
    }
