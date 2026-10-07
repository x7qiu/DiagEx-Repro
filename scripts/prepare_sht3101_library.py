"""Rebuild the curated SH/T 3101:2017 scan library without OCR or model calls.

The checked recipe owns names, qualifications, and PDF-coordinate rectangles.
Keep the user's PDF under ignored data/knowledge-sources; runtime uses only PNG/YAML.
"""

import argparse
import hashlib
import math
import shutil
from pathlib import Path

import fitz
import yaml
from PIL import Image, ImageDraw

from diagex.knowledge.models import Document, Entry

ROOT = Path(__file__).resolve().parents[1]
DOC = "sht-3101-2017"
TITLE = "SH/T 3101—2017 石油化工流程图图例"
SCAN_NOTE = (
    "Scanned source, 1190 × 1684 pixels per page, without a text layer. "
    "Crops preserve native pixels, including faded strokes. The PDF declares "
    "1190 × 1684 point pages, so 72-DPI rendering preserves that raster size; "
    "this is not a measurement of the physical paper's scanning resolution. "
    "Enlargement cannot recover missing detail. Names and selected notes were "
    "checked against the source; this is a curated subset, not a certified transcription."
)


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True, width=100))


def prepare(pdf, root=ROOT, recipe_path=None):
    library = root / "knowledge" / "pid"
    recipe_path = recipe_path or library / "recipes" / f"{DOC}.yaml"
    recipe = yaml.safe_load(recipe_path.read_text())
    if checksum(pdf) != recipe["source_sha256"]:
        raise ValueError("Source checksum differs from the inspected scan; recheck crop coordinates first")
    original = root / "data" / "knowledge-sources" / f"{DOC}.pdf"
    original.parent.mkdir(parents=True, exist_ok=True)
    if original.exists() and checksum(original) != recipe["source_sha256"]:
        raise ValueError("A different local source exists; do not overwrite it")
    with fitz.open(pdf) as document:
        if len(document) != recipe["page_count"] or any(
            list(page.rect)[2:] != recipe["page_size"] or page.rotation for page in document
        ):
            raise ValueError("Unexpected scan pagination, dimensions or rotation")
        if pdf.resolve() != original.resolve():
            shutil.copyfile(pdf, original)
        manifest = Document(
            id=DOC, title=TITLE, edition="2017", sha256=recipe["source_sha256"],
            scan_note=SCAN_NOTE,
        )
        dump(library / "sources" / f"{DOC}.yaml", manifest.model_dump(exclude_none=True))
        pages = {}

        def page_image(number):
            if number not in pages:
                pix = document[number - 1].get_pixmap(dpi=recipe["render_dpi"], alpha=False)
                pages[number] = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return pages[number]

        def source(page, table, item=None):
            return dict(document_id=DOC, document=TITLE, edition="2017", pdf_page=page,
                        page=str(page - 4), table=table, item=str(item) if item else None)

        def asset(directory, page, table, item, asset_id, role, label, rect,
                  quality="clear", depiction="isolated_symbol", trim=False):
            if len(rect) != 4 or not all(math.isfinite(v) for v in rect):
                raise ValueError("Invalid recipe rectangle")
            x0, y0, x1, y1 = map(int, rect)
            if not (0 <= x0 < x1 <= 1190 and 0 <= y0 < y1 <= 1684):
                raise ValueError("Recipe rectangle exceeds the inspected page")
            image = page_image(page).crop((x0, y0, x1, y1))
            if trim:
                # Locate only the outer whitespace; never threshold or repaint source pixels.
                bounds = image.convert("L").point(lambda p: 255 if p < 245 else 0).getbbox()
                if not bounds:
                    raise ValueError(f"Empty symbol crop: {directory.name}/{asset_id}")
                a, b, c, d = bounds
                a, b, c, d = max(0, a - 8), max(0, b - 8), min(image.width, c + 8), min(image.height, d + 8)
                image = image.crop((a, b, c, d))
                x0, y0, x1, y1 = x0 + a, y0 + b, x0 + c, y0 + d
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{asset_id}.png"
            image.save(path)
            return dict(
                id=asset_id, role=role, label=label, path=path.relative_to(library).as_posix(),
                sha256=checksum(path), width=image.width, height=image.height,
                quality=quality, depiction=depiction,
                crop=dict(document_id=DOC, pdf_page=page, printed_page=str(page - 4),
                          table=table, item=str(item) if item else None, rect=[x0, y0, x1, y1],
                          coordinates="unrotated_pdf_points_top_left", page_rotation=0,
                          dpi=recipe["render_dpi"], renderer=f"PyMuPDF {fitz.VersionBind}",
                          processing="Native scan pixels; crop only, no resampling, thresholding, sharpening or reconstructed strokes"),
            )

        def save(entry):
            entry = Entry.model_validate(entry)
            path = library / f"{entry.id}.yaml"
            record = entry.model_dump(mode="json", exclude_none=True)
            if path.exists():
                previous = Entry.model_validate(yaml.safe_load(path.read_text())).model_dump(mode="json", exclude_none=True)
                # This importer handles generic panels, not verified letter matrices.
                # Source checksum is checked above; preserve their later curation.
                if previous.get("letter_matrix"):
                    return
                record["version"] = previous["version"]
                if record != previous:
                    record["version"] += 1
            dump(path, record)

        for row in recipe["rows"]:
            table, item, page = row["table"], row["item"], row["page"]
            identity = f"{DOC}.t{table}-{item}"
            directory = library / "illustrations" / DOC / f"t{table}-{item}"
            src = source(page, table, item)
            y0, y1 = row["row"]
            assets = [asset(directory, page, table, item, "context", "context", "Complete source row",
                            [row["columns"][0] - 4, y0 - 5, row["columns"][-1] + 5, y1 + 5],
                            depiction="source_panel")]
            for variant in row["variants"]:
                assets.append(asset(
                    directory, page, table, item, variant["id"], "variant", variant["label"],
                    variant["rect"], quality=row.get("quality", "clear"),
                    depiction=variant.get("depiction", "source_panel" if row["reference_type"] == "assembly_pattern" else "isolated_symbol"),
                    trim=True,
                ))
            displayed = [a["id"] for a in assets if a["role"] == "variant"]
            included = row.get("model_variant_ids", displayed)
            selected = [a for a in assets if a["id"] in included]
            # Label every variant without changing its original pixels. Padding is presentation only.
            cell_width = max(220, max(a["width"] for a in selected) + 24)
            cell_height = max(a["height"] for a in selected) + 50
            columns = min(2, len(selected))
            sheet = Image.new("RGB", (columns * cell_width, math.ceil(len(selected) / columns) * cell_height), "white")
            draw = ImageDraw.Draw(sheet)
            for i, a in enumerate(selected):
                x, y = (i % columns) * cell_width, (i // columns) * cell_height
                draw.text((x + 10, y + 8), a["id"], fill="black")
                with Image.open(library / a["path"]) as crop:
                    sheet.paste(crop, (x + (cell_width - crop.width) // 2, y + 32))
            sheet_path = directory / "variants.png"
            sheet.save(sheet_path)
            assets.append(dict(id="variants", role="variant_sheet", label="Labeled reference variants",
                               path=sheet_path.relative_to(library).as_posix(), sha256=checksum(sheet_path),
                               width=sheet.width, height=sheet.height, derived_from=included,
                               quality=row.get("quality", "clear")))
            src["image"] = assets[0]["path"]
            notes = [dict(source=source(page, table, item), explanation=n, asset_id="context") for n in row["notes"]]
            if table == "4.7":
                foot = asset(directory, 18, table, None, "actuator-note", "note", "Actuator-circle diameter note",
                             [118,675,1024,709], depiction="source_panel")
                assets.append(foot)
                notes.append(dict(source=source(18, table), explanation=row["notes"][0], asset_id=foot["id"]))
            if table in ("4.5", "4.8"):
                note_page, rect = (15, [162,1508,1064,1539]) if table == "4.5" else (18, [118,1510,1024,1548])
                foot = asset(directory, note_page, table, None, "table-note", "note", "Table footnote", rect, depiction="source_panel")
                assets.append(foot)
                notes.append(dict(source=source(note_page, table), explanation=row["notes"][0], asset_id=foot["id"]))
            tasks = ["symbol_interpretation", "text_assignment"]
            if row["reference_type"] in ("line_style", "scope_boundary", "convention") or table == "4.1":
                tasks += ["line_interpretation", "connections"]
            explanation = row["interpretation"]
            save(dict(
                id=identity, version=1, concept=" / ".join(row["names"]), aliases=row["names"],
                reference_type=row["reference_type"], kind="definition", explanation=explanation,
                applicability={"standards": [{"name": "SH/T 3101", "edition": "2017"}]},
                tasks=tasks, tags=[*row["names"], row["category"], row["reference_type"], "SH/T 3101"],
                source=src, assets=assets, notes=notes, text_slots=row["text_slots"],
                interpretation=["Source illustrations are references, not drawing evidence. Never copy example tags, coordinates or connections into extraction.", *row["notes"]],
                exceptions=["Explicit drawing legends and project definitions take precedence. Unclear drawing evidence must remain uncertain."],
                source_status="normative", curation="curated",
                catalog=dict(role=row["role"], category=row["category"], short_name=row["names"][0],
                             search_terms=row.get("search_terms", []),
                             interpretation=explanation, displayed_asset_ids=displayed, model_asset_id="variants",
                             supporting_reference_ids=[f"{DOC}.{v}" for v in row.get("supporting", [])]),
            ))
        for role, records in [("background", recipe["background"]), ("supporting", recipe["supporting"])]:
            for record in records:
                identity = f"{DOC}.{record['id']}"
                directory = library / "illustrations" / DOC / record["id"]
                src = source(record["page"], record["table"])
                illustration = asset(directory, record["page"], record["table"], None, "context", "context",
                                     "Source context", record["rect"], depiction="source_panel")
                src["image"] = illustration["path"]
                explanation = record.get("explanation", "Source material retained for browsing, outside the focused P&ID catalog. Full panels are not individual symbols and are not supplied to extraction.")
                save(dict(id=identity, version=1, concept=record["name"],
                          reference_type=record.get("reference_type", "convention"), kind="definition",
                          explanation=explanation, tasks=["symbol_interpretation", "text_assignment"],
                          applicability={"standards": [{"name": "SH/T 3101", "edition": "2017"}]},
                          source=src, assets=[illustration], source_status="informative" if "appendix" in record["id"] else "normative",
                          catalog=dict(role=role, short_name=record["name"], interpretation=explanation)))
    return len(recipe["rows"]) + len(recipe["background"]) + len(recipe["supporting"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    args = parser.parse_args()
    print(f"Prepared {prepare(args.pdf)} SH/T 3101:2017 references")
