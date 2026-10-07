"""Source-attributed context; engineering assumptions never become drawn facts."""
import hashlib
from pathlib import Path


def context_document(text, *, source, kind):
    if kind not in {"process_overview", "engineering_rules"}:
        raise ValueError("Unknown process-context kind")
    text = str(text).strip()
    if not text or len(text) > 40000:
        raise ValueError("Process context must contain 1 to 40000 characters")
    return {"kind": kind, "source": str(source), "sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text}


def context_file(path, *, kind):
    path = Path(path).resolve()
    return context_document(path.read_text(), source=path, kind=kind)
