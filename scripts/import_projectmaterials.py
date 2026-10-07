"""Import the publisher's individual table-cell symbols, with original web provenance.

Run with --download once, then without it to rebuild offline from the ignored cache.
Overview charts outside symbol cells are reported, not misrepresented as single symbols.
"""

import argparse
import hashlib
import io
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import yaml
from PIL import Image

from diagex.knowledge.library import load_replacements
from diagex.knowledge.models import Document, Entry

URL = "https://blog.projectmaterials.com/epc-projects/engineering/pid-symbols-list/"
DOC = "projectmaterials.pid-symbols"
ROOT = Path(__file__).resolve().parents[1]
SECTIONS = {
    "Piping P&ID Symbols": "piping",
    "Valve Symbols for P&ID": "valves_actuators",
    "Strainers P&ID Symbols": "piping",
    "Lines P&ID Symbols": "lines_signals",
    "General Plant Equipment P&ID Symbols": "equipment",
    "Pumps P&ID Symbols": "equipment",
    "Compressors P&ID Symbols": "equipment",
    "Heat Exchangers P&ID Symbols": "equipment",
    "Vessels P&ID Symbols": "equipment",
    "Motors, Generators, and Turbines P&ID Symbols": "equipment",
    "Filters P&ID Symbols": "equipment",
    "Instruments (P&ID Symbols)": "instruments",
}
BACKGROUND = {
    "AND Gate",
    "NOT Gate",
    "OR Gate",
    "Computer",
    "Computer Indicator",
    "Displayed Configurable",
    "Displayed Programmable Indicator",
    "Programmable Indicator",
    "Diamond",
    "Double",
    "Unit Control Panel",
}
QUALIFICATION = (
    "Publisher-provided visual reference, not a verified ISA or ISO definition. "
    "Explicit drawing legends and applicable project definitions take precedence. "
    "Match visible geometry and drawing text; do not infer objects or connections "
    "from this reference alone. Labels and numbers in this example are not drawing tags."
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def slug(value):
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


class SymbolParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.heading = ""
        self.heading_id = ""
        self.in_heading = False
        self.cell = None
        self.rows = []
        self.overviews = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("h2", "h3", "h4"):
            self.in_heading = True
            self.heading = ""
            self.heading_id = attrs.get("id", "")
        if tag == "td":
            self.cell = {"images": [], "text": ""}
        if tag == "img" and attrs.get("src"):
            if self.cell is not None and self.heading in SECTIONS:
                self.cell["images"].append(attrs)
            elif self.heading in SECTIONS:
                self.overviews.append({"section": self.heading, "url": urljoin(URL, attrs["src"])})

    def handle_data(self, data):
        if self.in_heading:
            self.heading += data
        if self.cell is not None:
            self.cell["text"] += data

    def handle_endtag(self, tag):
        if tag in ("h2", "h3", "h4"):
            self.in_heading = False
            self.heading = self.heading.strip()
        if tag == "td" and self.cell is not None:
            for img in self.cell["images"]:
                self.rows.append(
                    dict(
                        section=self.heading,
                        section_id=self.heading_id,
                        label=" ".join(self.cell["text"].split()) or img.get("alt", ""),
                        alt=img.get("alt", ""),
                        url=urljoin(URL, img["src"]),
                    )
                )
            self.cell = None


def parse(html):
    parser = SymbolParser()
    parser.feed(html)
    if not parser.rows:
        raise ValueError("No individual symbol cells found; source structure may have changed")
    return parser


def fetch(url, path):
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "blog.projectmaterials.com":
        raise ValueError("Only the requested publisher host is allowed")
    if not path.exists():
        temp = path.with_suffix(".part")
        subprocess.run(
            [
                "curl",
                "--fail",
                "--silent",
                "--show-error",
                "--retry",
                "2",
                "--max-time",
                "45",
                "-A",
                "Mozilla/5.0",
                url,
                "-o",
                str(temp),
            ],
            check=True,
        )
        temp.replace(path)
    return path.read_bytes()


def reference_id(row):
    return f"{DOC}.{slug(row['section'].replace(' P&ID Symbols', ''))}.{slug(row['label'])}"


def record(row, raw, retrieved_at, root, reviewed_hash=None):
    # Caption/section identity survives changes to the site's hashed asset filenames.
    identity = reference_id(row)
    relative = f"illustrations/projectmaterials/{identity}.png"
    output = root / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(raw)) as image:
        original_format = image.format
        image.load()
        encoded = io.BytesIO()
        image.convert("RGBA" if "A" in image.getbands() else "RGB").save(encoded, "PNG")
        width, height = image.size
    if reviewed_hash and sha(encoded.getvalue()) != reviewed_hash:
        raise ValueError(f"Reviewed symbol image changed; reconsider overlap: {identity}")
    output.write_bytes(encoded.getvalue())
    category = SECTIONS[row["section"]]
    role = "background" if row["alt"] in BACKGROUND else "catalog"
    meaning = f"Projectmaterials labels this illustration ‘{row['label']}’."
    entry = dict(
        id=identity,
        version=1,
        concept=row["label"],
        aliases=[row["alt"]],
        curation="source_transcribed",
        source_status="informative",
        reference_type=(
            "scope_boundary"
            if row["label"] == "Control Panel / Enclosure"
            else "line_style"
            if category == "lines_signals"
            else "symbol"
        ),
        kind="common_practice",
        explanation=meaning,
        exceptions=[QUALIFICATION],
        applicability={},
        tasks=["symbol_interpretation", "line_interpretation"]
        if category == "lines_signals"
        else ["symbol_interpretation"],
        tags=["Projectmaterials"],
        source=dict(
            document="Projectmaterials — P&ID Symbols List",
            document_id=DOC,
            url=URL + "#" + row["section_id"],
            section=row["section"],
            item=row["label"],
        ),
        assets=[
            dict(
                id="symbol",
                role="variant",
                label=row["label"],
                path=relative,
                sha256=sha(output.read_bytes()),
                width=width,
                height=height,
                web=dict(
                    document_id=DOC,
                    url=row["url"],
                    retrieved_at=retrieved_at,
                    original_sha256=sha(raw),
                    original_format=original_format,
                ),
            )
        ],
        catalog=dict(
            role=role,
            category=category,
            short_name=row["label"],
            interpretation=meaning,
            displayed_asset_ids=["symbol"],
            model_asset_id="symbol",
        ),
    )
    return Entry.model_validate(entry).model_dump(mode="json", exclude_none=True)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch missing source files")
    args = parser.parse_args()
    cache = ROOT / "data/knowledge-sources/projectmaterials"
    cache.mkdir(parents=True, exist_ok=True)
    html_path = cache / "page.html"
    if args.download:
        fetch(URL, html_path)
    html = html_path.read_bytes()
    parsed = parse(html.decode("utf-8"))
    root = ROOT / "knowledge/pid"
    replacements = {r["reference_id"]: r for r in load_replacements(root)}
    for replacement in replacements.values():
        for target in replacement["replacement_ids"]:
            target_path = (root / f"{target}.yaml").resolve()
            if not target_path.is_relative_to(root.resolve()) or not target_path.is_file():
                raise ValueError(f"Missing replacement definition: {target}")
    time_path = cache / "retrieved-at.txt"
    if not time_path.exists():
        time_path.write_text(datetime.now(UTC).isoformat())
    retrieved_at = time_path.read_text().strip()

    # Fetch completely before modifying records. Failed downloads abort visibly.
    def get(row):
        path = cache / (sha(row["url"].encode()) + ".image")
        return fetch(row["url"], path) if args.download else path.read_bytes()

    with ThreadPoolExecutor(max_workers=4) as pool:
        images = list(pool.map(get, parsed.rows))
    manifest = Document(
        id=DOC,
        title="Projectmaterials — P&ID Symbols List",
        sha256=sha(html),
        source_type="web",
        url=URL,
        publisher="Projectmaterials",
        retrieved_at=retrieved_at,
        usage_statement="All images are free to use for your engineering documentation.",
    )
    dump(root / "sources" / f"{DOC}.yaml", manifest.model_dump(mode="json", exclude_none=True))
    entries = []
    retired = []
    seen = set()
    for row, raw in zip(parsed.rows, images, strict=True):
        replacement = replacements.get(reference_id(row))
        e = record(
            row,
            raw,
            retrieved_at,
            root,
            reviewed_hash=replacement["source_image_sha256"] if replacement else None,
        )
        if e["id"] in seen:
            raise ValueError(f"Duplicate symbol identity: {e['id']}")
        seen.add(e["id"])
        active_path = root / f"{e['id']}.yaml"
        path = root / "retired" / active_path.name if replacement else active_path
        previous = active_path if active_path.exists() else path
        if previous.exists():
            old = yaml.safe_load(previous.read_text())
            # Preserve deliberately curated vocabulary on offline rebuilds.
            e["catalog"]["search_terms"] = old.get("catalog", {}).get("search_terms", [])
            e["version"] = old["version"]
            if e != old:
                e["version"] += 1
        dump(path, e)
        if replacement:
            active_path.unlink(missing_ok=True)
            retired.append(e)
        else:
            entries.append(e)
    report = dict(
        source=DOC,
        source_sha256=sha(html),
        imported=len(entries) + len(retired),
        active=len(entries),
        replaced_by_isa=len(retired),
        roles=dict(Counter(e["catalog"]["role"] for e in entries)),
        sections=dict(Counter(r["section"] for r in parsed.rows)),
        excluded_overview_images=parsed.overviews,
        notes=[
            "Individual table-cell images only; article prose and composite charts are not symbol entries.",
            f"Publisher advertises 407 symbols; this snapshot contains {len(parsed.rows)} individual table-cell images.",
            "Original pixel dimensions retained; PNG conversion does not restore detail lost in source WebP images.",
        ],
    )
    dump(root / "coverage" / f"{DOC}.yaml", report)
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
