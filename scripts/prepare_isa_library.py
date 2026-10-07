"""Reproduce the remaining ISA 5.1:2009 library from the checksum-pinned local PDF.

No model calls. Table rules and clause boundaries are specific to this edition.
Keeps the three hand-curated pilot records unchanged. Never modifies the PDF.
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import fitz
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DOC = "isa-5.1-2009"
TITLE = "ANSI/ISA-5.1-2009"
PILOTS = {f"{DOC}.t5.3.2-{n}" for n in (17, 18, 19)}
TASKS = ["symbol_interpretation", "text_assignment", "line_interpretation", "connections", "review"]
PROSE_PAGES = [*range(13, 30), *range(31, 36), 76, 77, *range(85, 99), *range(111, 127)]
TECHNICAL_PAGES = set(range(13, 110)) | set(range(111, 127)) | {9, 10, 11}
NOTE_GROUP = {
    "5.1.1": "5.3.1",
    "5.1.2": "5.3.1",
    **{f"5.2.{i}": "5.3.2" for i in range(1, 6)},
    "5.3.1": "5.3.3",
    "5.3.2": "5.3.3",
    **{f"5.4.{i}": "5.3.4" for i in range(1, 5)},
    "5.5": "5.3.5",
    "5.6": "5.3.6",
    "5.7": "5.3.7",
    "5.8": "5.3.8",
    "4.1": "4.2",
    "A.1": "A.16.1",
    "A.2": "A.16.2",
    "A.3.1": "A.16.3",
    "A.3.2": "A.16.3",
    "A.4": "A.16.4",
}
EXPECTED_ITEMS = {
    "5.1.1": 5,
    "5.1.2": 7,
    "5.2.1": 7,
    "5.2.3": 41,
    "5.2.4": 5,
    "5.2.5": 7,
    "5.3.1": 9,
    "5.3.2": 21,
    "5.4.1": 21,
    "5.4.2": 21,
    "5.4.3": 23,
    "5.4.4": 5,
    "5.5": 8,
    "5.6": 28,
    "5.7": 18,
    "5.8": 37,
}


def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def clause_key(text):
    m = re.match(r"^((?:[1-6](?:\.\d+)*|[AB]\.\d+(?:\.\d+)*))\s+(.+)", text)
    if not m:
        return None
    key = m[1]
    # Definitions are individual; other prose is kept in complete topic sections.
    if re.fullmatch(r"3\.1\.\d+|[1-6]\.\d+|[AB]\.\d+|[56]\.3\.\d+|A\.16\.\d+|[1-6]", key):
        return key, m[2]
    return None


class Importer:
    def __init__(self, pdf, root):
        self.root = root / "knowledge" / "pid"
        manifest = yaml.safe_load((self.root / "sources" / f"{DOC}.yaml").read_text())
        if hashlib.sha256(pdf.read_bytes()).hexdigest() != manifest["sha256"]:
            raise ValueError("PDF checksum differs from the curated pilot source")
        self.doc = fitz.open(pdf)
        if len(self.doc) != 128:
            raise ValueError("Unexpected pagination")
        self.original = {i + 1: (p.rotation, p.derotation_matrix) for i, p in enumerate(self.doc)}
        # Normalize landscape text/table geometry in memory, never save over the source.
        for p in self.doc:
            if p.rotation:
                p.remove_rotation()
        self.entries = {}
        self.page_assets = {}
        self.coverage = defaultdict(set)
        self.groups = {}
        self.item_inventory = defaultdict(list)

    def source(self, page, table=None, item=None, passage=None):
        return dict(
            document_id=DOC,
            document=TITLE,
            edition="2009",
            page=str(page),
            pdf_page=page,
            table=table,
            item=item,
            passage=passage,
        )

    def asset(
        self,
        page,
        rect,
        name,
        role="context",
        label=None,
        table=None,
        item=None,
        depiction="source_panel",
    ):
        p = self.doc[page - 1]
        rect = fitz.Rect(rect) & p.rect
        path = self.root / "illustrations" / DOC / "complete" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        pix = p.get_pixmap(clip=rect, dpi=300, alpha=False)
        pix.save(path)
        rotation, derotation = self.original[page]
        source_rect = rect * derotation
        return dict(
            id=name.split("/")[-1],
            role=role,
            label=label or name,
            path=path.relative_to(self.root).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            width=pix.width,
            height=pix.height,
            crop=dict(
                document_id=DOC,
                pdf_page=page,
                printed_page=str(page),
                table=table,
                item=item,
                rect=[round(v, 4) for v in source_rect],
                coordinates="unrotated_pdf_points_top_left",
                page_rotation=rotation,
                dpi=300,
                renderer=f"PyMuPDF {fitz.VersionBind}",
                processing="display orientation retained; lossless source crop",
            ),
            quality="clear",
            depiction=depiction,
        )

    def page_asset(self, n):
        if n not in self.page_assets:
            p = self.doc[n - 1]
            self.page_assets[n] = self.asset(
                n,
                (60, 60, p.rect.width - 60, p.rect.height - 65),
                f"page-{n}",
                label=f"Complete technical content on page {n}",
            )
        return dict(self.page_assets[n])

    def base(
        self,
        identity,
        concept,
        explanation,
        page,
        ref_type="convention",
        kind="definition",
        table=None,
        item=None,
    ):
        return dict(
            id=identity,
            version=1,
            concept=concept,
            aliases=[],
            reference_type=ref_type,
            kind=kind,
            curation="source_transcribed",
            source_status="informative" if page >= 85 or page < 13 else "normative",
            explanation=explanation,
            exceptions=[
                "Explicit drawing legends and confirmed project definitions take precedence.",
                "Source illustrations are reference material, not evidence that objects or connections exist in a drawing.",
            ],
            applicability={"standards": [{"name": "ISA 5.1", "edition": "2009"}]},
            tasks=TASKS.copy(),
            tags=[],
            source=self.source(page, table, item),
            assets=[],
            notes=[],
            text_slots=[],
            interpretation=[],
            relations=[],
        )

    def add(self, entry):
        if entry["id"] in self.entries:
            raise ValueError(f"Duplicate generated ID: {entry['id']}")
        if entry["id"] not in PILOTS:
            self.entries[entry["id"]] = entry
        for a in entry.get("assets", []):
            if a.get("crop"):
                self.coverage[a["crop"]["pdf_page"]].add(entry["id"])

    def prose(self):
        sections = []
        current = None
        for n in PROSE_PAGES:
            page = self.doc[n - 1]
            lines = []
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    text = clean("".join(span["text"] for span in line["spans"]))
                    if text and 60 <= line["bbox"][1] < 730:
                        lines.append((line["bbox"][1], line["bbox"][0], text, line["bbox"][3]))
            lines.sort()
            merged = []
            for y, x, text, bottom in lines:
                if merged and abs(merged[-1][0] - y) < 1:
                    yy, xx, previous, bb = merged[-1]
                    merged[-1] = (yy, min(x, xx), previous + " " + text, max(bb, bottom))
                else:
                    merged.append((y, x, text, bottom))
            lines = merged
            # Full-width heading clauses only; diagram labels do not start sections.
            for y, x, text, bottom in lines:
                heading = clause_key(text) if x < 88 else None
                if heading:
                    key, name = heading
                    current = {"key": key, "name": name, "segments": {}, "lines": []}
                    sections.append(current)
                if current:
                    current["lines"].append(text)
                    current["segments"].setdefault(n, [y, bottom])
                    current["segments"][n][1] = max(current["segments"][n][1], bottom)
        for idx, section in enumerate(sections):
            key = section["key"]
            pages = list(section["segments"])
            n = pages[0]
            text = "\n".join(section["lines"])
            if key.startswith("3.1."):
                concept = section["name"].rstrip(":")
                kind = "definition"
            else:
                concept = f"{key} — {section['name']}"
                kind = "common_practice" if key.startswith(("A.", "B.")) else "definition"
            entry = self.base(f"{DOC}.clause-{key.lower()}", concept, text, n, kind=kind)
            entry["source"]["passage"] = f"Clause {key}"
            entry["source"]["page"] = ", ".join(map(str, pages))
            entry["tags"] = [section["name"], key]
            if key.startswith(("4.2", "5.3.", "6.3.", "A.", "B.")):
                entry["source_status"] = "informative"
            if key.startswith("B."):
                entry["reference_type"] = (
                    "assembly_pattern" if key not in {"B.1", "B.2", "B.3", "B.4"} else "convention"
                )
                entry["kind"] = (
                    "possible_arrangement"
                    if entry["reference_type"] == "assembly_pattern"
                    else "common_practice"
                )
                entry["interpretation"].append(
                    "These are worked examples. Do not treat the complete diagram as one equipment symbol or assign all enclosed text to a group."
                )
            if key.startswith("6.") or key == "6":
                entry["kind"] = "drawing_quality_recommendation"
                entry["tasks"] = ["review"]
            for page, (top, bottom) in section["segments"].items():
                # Extend evidence to next topic heading (or content bottom) to retain diagrams.
                next_y = 727
                if idx + 1 < len(sections) and page in sections[idx + 1]["segments"]:
                    next_y = sections[idx + 1]["segments"][page][0] - 2
                if next_y <= top:
                    next_y = bottom + 2
                asset = self.asset(
                    page,
                    (65, max(60, top - 2), self.doc[page - 1].rect.width - 65, next_y),
                    f"clause-{key.lower()}-p{page}",
                    label=f"Clause {key}, page {page}",
                )
                entry["assets"].append(asset)
                entry["assets"].append(self.page_asset(page))
            entry["source"]["image"] = entry["assets"][0]["path"]
            self.groups[key] = entry
            self.add(entry)

    def link_notes(self, entry, table, key=None):
        key = key or NOTE_GROUP.get(table)
        if key and key in self.groups:
            notes = self.groups[key]
            entry["relations"].append(
                dict(
                    relation="related",
                    reference_id=notes["id"],
                    explanation=f"Explanatory notes for Table {table}: clause {key}.",
                )
            )
            entry["notes"].append(dict(source=notes["source"], explanation=notes["explanation"]))
            # Point at complete note crops, not duplicated or disconnected footnote fragments.
            for a in notes["assets"]:
                if a["id"].startswith("clause-"):
                    entry["assets"].append(dict(a, role="note"))
            if entry["notes"] and any(a["role"] == "note" for a in entry["assets"]):
                entry["notes"][-1]["asset_id"] = next(
                    a["id"] for a in entry["assets"] if a["role"] == "note"
                )

    def sheet(self, entry):
        variants = [a for a in entry["assets"] if a["role"] == "variant"]
        if not variants:
            return
        width = 1000
        thumbs = []
        for asset in variants:
            with Image.open(self.root / asset["path"]) as im:
                image = im.convert("RGB")
                image.thumbnail((940, 700))
            thumbs.append((asset, image))
        total = sum(im.height + 65 for _, im in thumbs)
        sheet = Image.new("RGB", (width, total), "white")
        draw = ImageDraw.Draw(sheet)
        y = 0
        for asset, im in thumbs:
            draw.text((20, y + 10), asset["label"], fill="black", font_size=19)
            sheet.paste(im, ((width - im.width) // 2, y + 45))
            y += im.height + 65
        path = (
            self.root
            / "illustrations"
            / DOC
            / "complete"
            / f"{entry['id'].removeprefix(DOC + '.')}-variants.png"
        )
        sheet.save(path)
        entry["assets"].append(
            dict(
                id="variants",
                role="variant_sheet",
                label="Labeled reference panels",
                path=path.relative_to(self.root).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                width=sheet.width,
                height=sheet.height,
                derived_from=[a["id"] for a in variants],
                quality="clear",
                depiction="source_panel",
            )
        )

    def row_tables(self):
        for n in range(36, 76):
            if n == 39:
                continue
            p = self.doc[n - 1]
            for t in p.find_tables().tables:
                if t.bbox[2] - t.bbox[0] < 400:
                    continue
                before = p.get_text(clip=fitz.Rect(65, 60, 550, t.bbox[1]), sort=True)
                titles = list(re.finditer(r"Table (5\.[\d.]+)\s*[—–-]\s*([^\n]+)", before))
                if not titles:
                    raise ValueError(f"No table title on page {n}")
                title = titles[-1]
                table = title[1]
                name = clean(title[2])
                rows = t.extract()
                itemcol = 1 if table in {"5.2.3", "5.2.4", "5.2.5"} else 0
                for row, geometry in zip(rows, t.rows, strict=True):
                    item = clean(row[itemcol])
                    if not item.isdigit() or not geometry.cells[itemcol]:
                        continue
                    self.item_inventory[table].append(int(item))
                    identity = f"{DOC}.t{table}-{item}"
                    if identity in PILOTS:
                        self.coverage[n].add(identity)
                        continue
                    bbox = geometry.cells[itemcol]
                    y0, y1 = bbox[1], bbox[3]
                    desc = clean(row[-1])
                    symbolcol = itemcol + 1
                    # Names in function/logic tables are separate from their displayed glyphs.
                    concept = (
                        clean(row[1])
                        if table in {"5.6", "5.7"}
                        else clean(desc.split("•")[1] if "•" in desc else desc)
                    )
                    concept = concept.rstrip(".") or f"{name}, item {item}"
                    if table == "5.1.1":
                        concept = f"Instrumentation location: {concept}"
                    ref_type = "line_style" if table in {"5.3.1", "5.3.2"} else "symbol"
                    if table == "5.4.4":
                        ref_type = "convention"
                    if table == "5.3.2" and int(item) >= 17:
                        ref_type = "symbol"
                    entry = self.base(identity, concept, desc, n, ref_type, table=table, item=item)
                    entry["tags"] = [name, table, concept]
                    entry["source"]["passage"] = f"Table {table}, item {item}: {desc}"
                    context = self.asset(
                        n,
                        (t.bbox[0] - 1, y0 - 1, t.bbox[2] + 1, y1 + 1),
                        f"t{table}-{item}-context",
                        label=f"Complete row: Table {table}, item {item}",
                        table=table,
                        item=item,
                    )
                    entry["assets"] = [context, self.page_asset(n)]
                    entry["source"]["image"] = context["path"]
                    columns = (
                        range(1, 5)
                        if table == "5.1.1"
                        else range(1, 3)
                        if table == "5.4.4"
                        else [symbolcol]
                    )
                    for col in columns:
                        cell = geometry.cells[col]
                        if not cell:
                            raise ValueError(f"Missing symbol cell {identity}/{col}")
                        rect = (cell[0] + 0.8, y0 + 0.8, cell[2] - 0.8, y1 - 0.8)
                        label = (
                            f"{'ABCD'[col - 1]}: "
                            + ["Primary / BPCS", "Alternate / SIS", "Computer systems", "Discrete"][
                                col - 1
                            ]
                            if table == "5.1.1"
                            else f"Method {'AB'[col - 1]}"
                            if table == "5.4.4"
                            else "Symbol panel (includes source annotations)"
                        )
                        entry["assets"].append(
                            self.asset(
                                n,
                                rect,
                                f"t{table}-{item}-panel-{col}",
                                role="variant",
                                label=label,
                                table=table,
                                item=item,
                            )
                        )
                    # Logic alternate glyphs can be in the definition column. Retain that
                    # complete panel as an alternative instead of losing or guessing glyphs.
                    if table == "5.7" and "Alternate symbol" in desc:
                        cell = geometry.cells[-1]
                        entry["assets"].append(
                            self.asset(
                                n,
                                (cell[0] + 0.8, y0 + 0.8, cell[2] - 0.8, y1 - 0.8),
                                f"t{table}-{item}-alternate",
                                role="variant",
                                label="Alternate symbol with definition and graph",
                                table=table,
                                item=item,
                            )
                        )
                    entry["interpretation"] = [
                        "The panels retain all source alternatives and annotations; they are not tight drawing-object bounding boxes.",
                        "Item numbers, method labels, note markers, equations, and illustrative tags are not automatically symbol geometry or actual drawing tags.",
                        "Apply each placeholder according to its own row and linked notes. Asterisks, question marks, letters, and numbers do not have one universal meaning.",
                        "Assign text, ports, coordinates, and connections only from the drawing occurrence; preserve uncertainty.",
                    ]
                    if table in {"5.3.1", "5.3.2"}:
                        entry["interpretation"].append(
                            "Distinguish process connections, instrument signals, and line crossings using the stated application and drawing legend."
                        )
                    # Extract explicit marker definitions only. Do not invent meanings from shape.
                    for marker, meaning in re.findall(r"(\([#*]+\)|\[[#*]+\])\s*=\s*([^•]+)", desc):
                        entry["text_slots"].append(
                            dict(
                                role="source_variable",
                                example_marker=marker,
                                location="As marked in the source panel",
                                meaning=meaning.strip(),
                            )
                        )
                    self.link_notes(entry, table)
                    if table == "5.7" and item in {"3", "4"}:
                        entry["exceptions"].append(
                            "Source inconsistency: on page 65, the NAND/NOR headings and their verbal/graphical truth definitions do not consistently agree. Preserve the original evidence; do not derive logic behavior from this row without corroboration."
                        )
                    self.sheet(entry)
                    self.add(entry)
        for table, count in EXPECTED_ITEMS.items():
            if sorted(self.item_inventory[table]) != list(range(1, count + 1)):
                raise ValueError(
                    f"Incomplete/duplicated table {table}: {self.item_inventory[table]}"
                )

    def other_tables(self):
        for n in [30, 39, *range(78, 85), *range(99, 110)]:
            p = self.doc[n - 1]
            heading = p.get_text(clip=fitz.Rect(60, 60, p.rect.width - 60, 117), sort=True)
            m = re.search(r"Table ([A-Z\d.]+)\s*[—–-]\s*([^\n]+)", heading)
            if not m:
                raise ValueError(f"No table heading p{n}: {heading}")
            table, name = m[1], clean(m[2])
            identity = f"{DOC}.t{table.lower()}-p{n}"
            text = p.get_text(
                clip=fitz.Rect(60, 60, p.rect.width - 60, p.rect.height - 65), sort=True
            )
            entry = self.base(
                identity,
                f"Table {table} — {name} (page {n})",
                f"{name}. Use the source table with its column headings and linked explanatory notes. Blank or merged cells must not be interpreted as new definitions.",
                n,
                table=table,
            )
            entry["source"]["passage"] = text
            entry["tags"] = (
                [name, table, "identification", "tag", "letters"]
                if n == 30 or n >= 99
                else [name, table, "measurement", "notation"]
            )
            entry["assets"] = [self.page_asset(n)]
            entry["source"]["image"] = entry["assets"][0]["path"]
            entry["table_data"] = [
                t.extract() for t in p.find_tables().tables if t.bbox[2] - t.bbox[0] > 400
            ]
            entry["interpretation"] = [
                "Table cells are source transcriptions; retain headers and merged-cell context when interpreting them. Use the image to resolve superscripts or extraction ambiguity.",
                "Example tags and numbering schemes illustrate conventions; never copy them as actual drawing identifiers.",
            ]
            if table.startswith("6."):
                entry["kind"] = "drawing_quality_recommendation"
                entry["tasks"] = ["review"]
                entry["tags"] = [name, table, "dimensions"]
                note = "6.3." + table.split(".")[1]
                NOTE_GROUP[table] = note
            else:
                entry["tasks"] = ["text_assignment", "symbol_interpretation", "review"]
            self.link_notes(entry, table)
            printed = re.search(r"Clause ([A-Z\d.]+)", heading)
            printed_key = printed[1].rstrip(".") if printed else None
            if printed_key in self.groups and printed_key != NOTE_GROUP.get(table):
                self.link_notes(entry, table, printed_key)
                entry["exceptions"].append(
                    f"The printed heading points to clause {printed_key}, while clause {NOTE_GROUP.get(table)} explicitly discusses Table {table}. Both references are retained."
                )
            self.add(entry)
            if n == 82:
                second = self.base(
                    f"{DOC}.t6.6-p82",
                    "Table 6.6 — Dimensions for Table 5.6 (page 82)",
                    "Dimension recommendations for signal processing function blocks. Table 6.6 is the lower panel on page 82; the upper panel belongs to Table 6.5.",
                    82,
                    kind="drawing_quality_recommendation",
                    table="6.6",
                )
                second["tasks"] = ["review"]
                second["tags"] = ["dimensions", "signal processing", "function blocks", "6.6"]
                second["assets"] = [self.page_asset(82)]
                second["source"]["image"] = second["assets"][0]["path"]
                second["source"]["passage"] = (
                    "Table 6.6, lower panel; explanatory notes in clause 6.3.6."
                )
                self.link_notes(second, "6.6", "6.3.6")
                self.add(second)
            # Identification letters deserve individual searchable, column-aware entries.
            if n == 30:
                t = max(p.find_tables().tables, key=lambda t: t.bbox[2] - t.bbox[0])
                names = [
                    "Measured/initiating variable",
                    "Variable modifier",
                    "Readout/passive function",
                    "Output/active function",
                    "Function modifier",
                ]
                for r in t.extract():
                    letter = clean(r[0])
                    if not re.fullmatch("[A-Z]", letter):
                        continue
                    desc = "; ".join(
                        f"{label}: {clean(v) or 'blank in source'}"
                        for label, v in zip(names, r[1:], strict=True)
                    )
                    e = self.base(
                        f"{DOC}.t4.1-letter-{letter.lower()}",
                        f"Identification letter {letter}",
                        desc,
                        n,
                        table="4.1",
                        item=letter,
                    )
                    e["tasks"] = ["text_assignment", "symbol_interpretation"]
                    e["tags"] = [
                        "identification",
                        "letter",
                        letter,
                        *[clean(v) for v in r[1:] if v],
                    ]
                    e["assets"] = [entry["assets"][0]]
                    e["source"]["image"] = e["assets"][0]["path"]
                    e["interpretation"] = [
                        "Letter meaning depends on its position in the instrument identification. Do not assign one meaning to every occurrence of this letter."
                    ]
                    self.link_notes(e, "4.1")
                    self.add(e)

    def finish(self):
        for n in [9, 10, 11]:
            p = self.doc[n - 1]
            text = p.get_text(clip=fitz.Rect(65, 60, 550, 730), sort=True)
            e = self.base(
                f"{DOC}.introduction-p{n}",
                f"Introduction and interpretation background (page {n})",
                text,
                n,
                kind="common_practice",
            )
            e["tasks"] = ["review"]
            e["assets"] = [self.page_asset(n)]
            e["source"]["image"] = e["assets"][0]["path"]
            self.add(e)
        missing = TECHNICAL_PAGES - set(self.coverage)
        if missing:
            raise ValueError(f"Uncovered technical pages: {sorted(missing)}")
        # Links to other tables are explicit reference relationships, not drawing topology.
        table_ids = defaultdict(list)
        for e in self.entries.values():
            if e["source"].get("table"):
                table_ids[e["source"]["table"]].append(e["id"])
        for e in self.entries.values():
            # Keep source text intact; browsing and retrieval use the same records.
            dump(self.root / f"{e['id']}.yaml", e)
        coverage = dict(
            document_id=DOC,
            source_sha256=yaml.safe_load((self.root / "sources" / f"{DOC}.yaml").read_text())[
                "sha256"
            ],
            generated_entry_count=len(self.entries),
            preserved_pilot_ids=sorted(PILOTS),
            technical_pages=sorted(TECHNICAL_PAGES),
            tables=sorted(table_ids),
            pages={str(p): sorted(ids) for p, ids in sorted(self.coverage.items())},
            numbered_tables={t: sorted(items) for t, items in sorted(self.item_inventory.items())},
            excluded_pages={
                str(
                    p
                ): "Cover, publication/committee information, contents, blank page, or publisher material; not extraction knowledge."
                for p in range(1, 129)
                if p not in TECHNICAL_PAGES
            },
            qualifications=[
                "Existing pilot records are hand-curated; added records are source-transcribed.",
                "Annotated panels retain alternatives; they are not uniformly isolated glyph crops.",
                "Page 65 NAND/NOR source inconsistency is retained and flagged.",
                "No DEXPI mappings or extraction accuracy improvement are inferred from import coverage.",
            ],
        )
        dump(self.root / "coverage" / f"{DOC}.yaml", coverage)
        print(
            json.dumps(
                {
                    "generated_entries": len(self.entries),
                    "technical_pages": len(TECHNICAL_PAGES),
                    "numbered_items": sum(map(len, self.item_inventory.values())),
                }
            )
        )


def prepare(pdf, root=ROOT):
    imp = Importer(pdf, root)
    imp.prose()
    imp.row_tables()
    imp.other_tables()
    imp.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT / "data/knowledge-sources" / f"{DOC}.pdf")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    prepare(args.pdf, args.root)
