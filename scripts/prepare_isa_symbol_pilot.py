"""Reproduce the curated ISA 5.1:2009 pilot from the user's original PDF.

Run from the project root with --pdf PATH. Coordinates were checked against
Table 5.3.2 on PDF page 47 and its explanatory note on page 34.
This is a fixed, auditable curation recipe, not a whole-document importer.
"""

import argparse
import hashlib
import shutil
from pathlib import Path

import fitz
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DOC_ID = "isa-5.1-2009"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))


def prepare(pdf, root=ROOT):
    library = root / "knowledge" / "pid"
    original = root / "data" / "knowledge-sources" / f"{DOC_ID}.pdf"
    original.parent.mkdir(parents=True, exist_ok=True)
    if pdf.resolve() != original.resolve():
        if original.exists() and original.read_bytes() != pdf.read_bytes():
            raise ValueError(
                "A different source already exists; preserve it and resolve the edition first"
            )
        shutil.copyfile(pdf, original)
    document = fitz.open(original)
    if len(document) != 128 or "Drawing-to-drawing signal connector" not in document[46].get_text():
        raise ValueError("Source does not match the inspected ISA 5.1:2009 pagination")
    dump(
        library / "sources" / f"{DOC_ID}.yaml",
        {
            "id": DOC_ID,
            "title": "ANSI/ISA-5.1-2009 Instrumentation Symbols and Identification",
            "edition": "2009",
            "sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        },
    )
    definitions = [
        (
            17,
            "Drawing-to-drawing signal connector",
            ["跨图信号连接符", "signal continuation"],
            "Carries a signal between drawings. The table describes signal flow from left to right and shows rounded and pointed alternatives. The upper field identifies the sending or receiving instrument; the lower field identifies the other drawing or sheet.",
            (71, 166.5, 541, 331),
            [
                ("a1", "a) upper rounded alternative", (121, 181, 215, 213)),
                ("a2", "a) lower rounded alternative", (121, 215, 215, 246)),
                ("b1", "b) upper pointed alternative", (123, 253, 215.3, 287)),
                ("b2", "b) lower pointed alternative", (121, 289, 215, 323)),
            ],
            [
                {
                    "role": "instrument_tag",
                    "example_marker": "(#)",
                    "location": "upper compartment",
                    "meaning": "Instrument sending or receiving the signal",
                },
                {
                    "role": "drawing_reference",
                    "example_marker": "(##)",
                    "location": "lower compartment",
                    "meaning": "Drawing or sheet sending or receiving the signal",
                },
            ],
        ),
        (
            18,
            "Signal input to a logic diagram",
            ["逻辑图信号输入", "logic signal input"],
            "Marks a signal input to a logic diagram. Its variable text can describe the input, identify its source, or give an instrument tag.",
            (71, 331, 541, 369),
            [("input", "Signal input", (131, 337, 186, 364))],
            [
                {
                    "role": "input_reference",
                    "example_marker": "(*)",
                    "location": "left of terminal stroke",
                    "meaning": "Input description, source, or instrument tag",
                }
            ],
        ),
        (
            19,
            "Signal output from a logic diagram",
            ["逻辑图信号输出", "logic signal output"],
            "Marks a signal output from a logic diagram. Its variable text can describe the output, identify its destination, or give an instrument tag.",
            (71, 369, 541, 407),
            [("output", "Signal output", (136, 375, 190, 402))],
            [
                {
                    "role": "output_reference",
                    "example_marker": "(*)",
                    "location": "right of terminal stroke",
                    "meaning": "Output description, destination, or instrument tag",
                }
            ],
        ),
    ]
    for item, concept, aliases, explanation, row, variants, slots in definitions:
        identity = f"{DOC_ID}.t5.3.2-{item}"
        directory = library / "illustrations" / DOC_ID / f"t5.3.2-{item}"
        directory.mkdir(parents=True, exist_ok=True)
        assets = []

        def asset(
            asset_id,
            role,
            label,
            rect,
            page=47,
            table="5.3.2",
            source_item=str(item),
            directory=directory,
            assets=assets,
        ):
            path = directory / f"{asset_id}.png"
            pix = document[page - 1].get_pixmap(clip=fitz.Rect(rect), dpi=300, alpha=False)
            pix.save(path)
            assets.append(
                {
                    "id": asset_id,
                    "role": role,
                    "label": label,
                    "path": path.relative_to(library).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "width": pix.width,
                    "height": pix.height,
                    "crop": {
                        "document_id": DOC_ID,
                        "pdf_page": page,
                        "printed_page": str(page),
                        "table": table,
                        "item": source_item,
                        "rect": list(rect),
                        "coordinates": "unrotated_pdf_points_top_left",
                        "page_rotation": document[page - 1].rotation,
                        "dpi": 300,
                        "renderer": f"PyMuPDF {fitz.VersionBind}",
                        "processing": "none",
                    },
                    "quality": "clear",
                }
            )

        asset("context", "context", "Complete source row", row)
        asset("table-heading", "context", "Table title and note convention", (71, 70, 541, 124))
        for variant_id, label, rect in variants:
            asset(variant_id, "variant", label, rect)
        notes = []
        if item == 17:
            asset("note-3", "note", "Clause 5.3.3, note (3)", (70, 116, 545, 145), 34, None, None)
            notes.append(
                {
                    "source": {
                        "document_id": DOC_ID,
                        "document": "ANSI/ISA-5.1-2009",
                        "edition": "2009",
                        "page": "34",
                        "pdf_page": 34,
                        "passage": "Clause 5.3.3, note (3)",
                    },
                    "explanation": "The user's engineering and design practices should document which alternative symbol is selected.",
                    "asset_id": "note-3",
                }
            )
        columns = min(2, len(variants))
        sheet = Image.new(
            "RGB", (columns * 550, ((len(variants) + columns - 1) // columns) * 230), "white"
        )
        draw = ImageDraw.Draw(sheet)
        for i, (variant_id, label, _) in enumerate(variants):
            x, y = (i % columns) * 550, (i // columns) * 230
            draw.text((x + 18, y + 10), f"{variant_id} - {label}", fill="black", font_size=18)
            with Image.open(directory / f"{variant_id}.png") as glyph:
                glyph.thumbnail((510, 175))
                sheet.paste(glyph, (x + (550 - glyph.width) // 2, y + 45))
        sheet_path = directory / "variants.png"
        sheet.save(sheet_path)
        assets.append(
            {
                "id": "variants",
                "role": "variant_sheet",
                "label": "Labeled visual variants",
                "path": sheet_path.relative_to(library).as_posix(),
                "sha256": hashlib.sha256(sheet_path.read_bytes()).hexdigest(),
                "width": sheet.width,
                "height": sheet.height,
                "derived_from": [v[0] for v in variants],
                "quality": "clear",
            }
        )
        dump(
            library / f"{identity}.yaml",
            {
                "id": identity,
                "version": 1,
                "concept": concept,
                "aliases": aliases,
                "reference_type": "symbol",
                "kind": "definition",
                "explanation": explanation,
                "exceptions": [
                    "This defines signal notation, not process-piping connectivity.",
                    "Use the explicit drawing legend when it differs; shape alone does not prove a connection.",
                ],
                "applicability": {"standards": [{"name": "ISA 5.1", "edition": "2009"}]},
                "tasks": ["symbol_interpretation", "text_assignment"],
                "tags": ["symbol", "signal", "connector", "logic", "instrument", *aliases],
                "source": {
                    "document_id": DOC_ID,
                    "document": "ANSI/ISA-5.1-2009",
                    "edition": "2009",
                    "page": "47",
                    "pdf_page": 47,
                    "table": "5.3.2",
                    "item": str(item),
                    "image": (directory / "context.png").relative_to(library).as_posix(),
                },
                "assets": assets,
                "notes": notes,
                "text_slots": slots,
                "interpretation": [
                    "Placeholder marks represent variable text, not literal identifiers to copy into drawing results.",
                    "Row numbers, alternative labels a)/b), and parenthesized note (3) are reference annotations, not symbol parts.",
                    "Retain uncertainty when the source glyph or actual text does not support a specific interpretation.",
                ]
                + (
                    []
                    if item == 17
                    else [
                        "A logic-diagram input/output mark alone does not prove an off-page continuation; do not force an equipment or piping-node classification."
                    ]
                ),
            },
        )
    document.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path)
    prepare(parser.parse_args().pdf)
