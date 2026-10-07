"""Apply the focused visual catalog to the preserved ISA 5.1:2009 source archive.

No model calls. Re-render lossless, tightly framed PDF panels; preserve originals.
Run after prepare_isa_library.py when rebuilding from the source PDF.
"""

import argparse
import hashlib
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

import fitz
import yaml
from PIL import Image, ImageChops, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DOC = "isa-5.1-2009"
CATEGORIES = {
    "5.1.1": "instruments", "5.2.1": "measurement", "5.2.3": "measurement",
    "5.2.4": "measurement", "5.2.5": "measurement", "5.3.1": "lines_signals",
    "5.3.2": "lines_signals", "5.4.1": "valves_actuators",
    "5.4.2": "valves_actuators", "5.4.3": "valves_actuators",
}
NAMES = {
    "t5.1.1-1": "Field instrument", "t5.1.1-2": "Main panel instrument",
    "t5.1.1-3": "Instrument behind main panel", "t5.1.1-4": "Local panel instrument",
    "t5.1.1-5": "Instrument behind local panel",
    "t5.2.1-1": "Primary measurement element", "t5.2.1-2": "Transmitter with integral element",
    "t5.2.1-3": "Transmitter with close-coupled element",
    "t5.2.1-4": "Transmitter with remote element",
    "t5.2.1-5": "Integral element on process equipment",
    "t5.2.1-6": "Close-coupled element on process equipment",
    "t5.2.1-7": "Remote element on process equipment",
    "t5.2.3-1": "Conductivity / moisture probe", "t5.2.3-2": "pH / ORP probe",
    "t5.2.3-27": "Coriolis flowmeter", "t5.3.2-1": "Instrument air / gas supply",
    "t5.3.2-5": "Pneumatic signal", "t5.3.2-6": "Electrical signal",
    "t5.3.2-7": "Continuous functional signal", "t5.3.2-11": "Wireless / unguided signal",
    "t5.3.2-12": "Shared control system bus", "t5.3.2-13": "Independent systems link",
    "t5.3.2-14": "Fieldbus link", "t5.3.2-15": "Remote calibration link",
    "t5.3.2-18": "Signal input connector", "t5.3.2-19": "Signal output connector",
    "t5.4.3-14": "Pressure regulator with relief and gauge",
}
SUPPORT = {
    "clause-5.3.1": (
        "Instrument shapes and location markings",
        "A circle represents an individual hardware instrument. A circle in a square represents "
        "the project's primary shared display/control system or BPCS; a diamond in a square "
        "represents its alternate shared display/control system or SIS. The project must document "
        "which meaning is selected: primary/alternate are system distinctions, not interchangeable "
        "symbol styles. Horizontal solid or dashed markings distinguish location and operator "
        "accessibility as specified in each table row. Accessibility includes viewing, setpoint "
        "adjustment, and operating-mode changes. Shapes alone do not prove safety-system membership.",
    ),
    "clause-5.3.2": (
        "Measurement element notation",
        "Measurements may use bubbles alone or bubbles with graphics. Bubble notation is used "
        "when a graphic is unavailable or the project does not use graphics. The transmitter "
        "function T in examples may instead be controller C, indicator I, recorder R, or switch S "
        "as supported by the drawing. The project documents its selected conventions. A depicted "
        "transmitter and primary element may be multiple components, not one object bounding box.",
    ),
    "clause-5.3.3": (
        "Signal and connection notation",
        "Signal arrows clarify direction when needed; absence of an arrow does not establish "
        "direction. Projects document their selected alternatives. Signal connectors are distinct "
        "from process-piping connectors. System-bus, independent-system, fieldbus and remote "
        "calibration links have different meanings: use the line definition and drawing evidence. "
        "Power supplies are indicated when nonstandard, independently required, or affected by "
        "controller/switch actions. Do not infer missing supplies or connections.",
    ),
    "clause-5.3.4": (
        "Valve bodies, actuators and assemblies",
        "Valve bodies and actuators are separate graphical components which can form an assembly. "
        "Users document their chosen symbols. Read the actual combination: a valve body alone "
        "does not establish its actuator, control function or failure action. Source notes include "
        "specific combination rules; retain ambiguity if a combination is unsupported. "
        "The source's note (4) lists actuator items 13-15 for on-off solenoid valves while note (8) "
        "lists 17-19; do not silently reconcile those statements. Item 21 denotes a motor controlling "
        "a process variable; an actuator combination may represent variable-speed control.",
    ),
    "clause-4.1": (
        "Instrument identification letters",
        "Interpret instrument letters according to their position: measured/initiating variable, "
        "variable modifier, readout/passive function, output/active function, or function modifier. "
        "User-choice letters and assigned blank positions require the project's documented "
        "definitions or drawing legend. Reference examples are not actual instrument tags.",
    ),
}


def compact(text):
    return re.sub(r"\s+", " ", text.replace("•", "")).strip()


def cleaned_panel_page(doc, page_index, rect):
    """Remove only top-margin note citations from an in-memory PDF page copy.

    This is edition-specific source curation, never a filter for drawing text.
    Text-only PDF redaction preserves every vector stroke and raster image.
    """
    copy = fitz.open()
    copy.insert_pdf(doc, from_page=page_index, to_page=page_index)
    page = copy[0]
    previous = fitz.TOOLS.set_small_glyph_heights()
    try:
        fitz.TOOLS.set_small_glyph_heights(True)
        words = page.get_text("words", clip=rect)
        notes = [w for w in words if re.fullmatch(r"\(\d+[a-z]?\)", w[4])
                 and w[1] < rect.y0 + 12 and w[0] < rect.x0 + 45]
        # Table 5.2.1 item 2 has a stray '(' inside its printed note-citation run.
        # Only remove punctuation bracketed by identified citations on the same baseline.
        if notes:
            notes += [w for w in words if w[4] in {"(", ")"}
                      and min(n[0] for n in notes) < w[0] < max(n[2] for n in notes)
                      and any(abs(w[1] - n[1]) < 0.5 for n in notes)]
        protected = [w for w in words if w not in notes]
        for word in notes:
            if any(fitz.Rect(word[:4]).intersects(fitz.Rect(w[:4])) for w in protected):
                copy.close()
                raise ValueError("Note reference overlaps symbol text; manual curation required")
            page.add_redact_annot(fitz.Rect(word[:4]), fill=False, cross_out=False)
        if notes:
            page.apply_redactions(images=0, graphics=0, text=0)
        excluded = [dict(text=w[4], rect=list(w[:4]), reason="source_note_reference") for w in notes]
        return copy, page, excluded
    finally:
        fitz.TOOLS.set_small_glyph_heights(previous)


def make_asset(root, doc, entry, original, label):
    """Exclude note citations, retaining symbol geometry and meaningful text."""
    crop = deepcopy(original["crop"])
    page = doc[crop["pdf_page"] - 1]
    rect = fitz.Rect(crop["rect"])
    if page.rotation:
        raise ValueError("Catalog curation expects unrotated source panels")
    working, page, excluded = cleaned_panel_page(doc, crop["pdf_page"] - 1, rect)
    pix = page.get_pixmap(clip=rect, dpi=300, alpha=False)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    ink = ImageChops.difference(image, Image.new("RGB", image.size, "white"))
    bbox = ink.convert("L").point(lambda p: 255 if p > 25 else 0).getbbox()
    if not bbox:
        raise ValueError(f"Empty symbol panel: {entry['id']}")
    # Re-render the resulting PDF rectangle so provenance describes the exact pixels.
    scale = 300 / 72
    rect = fitz.Rect(
        max(rect.x0, (pix.x + bbox[0] - 10) / scale),
        max(rect.y0, (pix.y + bbox[1] - 10) / scale),
        min(rect.x1, (pix.x + bbox[2] + 10) / scale),
        min(rect.y1, (pix.y + bbox[3] + 10) / scale),
    )
    id_ = "catalog-" + original["id"]
    relative = Path("illustrations") / DOC / "catalog" / f"{entry['id']}.{id_}.png"
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    pix = page.get_pixmap(clip=rect, dpi=300, alpha=False)
    pix.save(target)
    crop.update(rect=list(rect), dpi=300, renderer=f"PyMuPDF {fitz.VersionBind}",
                processing="Top-margin source-note citations removed using text-only PDF redaction; "
                "blank margins trimmed; vector geometry, alternative labels and placeholders retained",
                excluded_annotations=excluded)
    working.close()
    return dict(id=id_, role="variant", label=label, path=str(relative),
                sha256=hashlib.sha256(target.read_bytes()).hexdigest(), width=pix.width,
                height=pix.height, crop=crop, quality=original.get("quality", "clear"),
                depiction=original.get("depiction", "isolated_symbol"))


def sheet_asset(root, entry, variants):
    font = ImageFont.load_default(size=20)
    width, cell_height = 1100, 260
    image = Image.new("RGB", (width, len(variants) * cell_height), "white")
    draw = ImageDraw.Draw(image)
    for i, asset in enumerate(variants):
        y = i * cell_height
        draw.text((18, y + 8), f"{asset['id']} | {asset['label']}", fill="black", font=font)
        with Image.open(root / asset["path"]) as source:
            source = source.convert("RGB")
            source.thumbnail((width - 50, cell_height - 50))
            image.paste(source, ((width - source.width) // 2, y + 42))
    relative = Path("illustrations") / DOC / "catalog" / f"{entry['id']}.sheet.png"
    target = root / relative
    image.save(target)
    return dict(id="catalog-sheet", role="variant_sheet", label="Selected catalog variants",
                path=str(relative), sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                width=image.width, height=image.height, derived_from=[a["id"] for a in variants],
                quality="clear", depiction="source_panel")


def curate(pdf, root):
    manifest = yaml.safe_load((root / "sources" / f"{DOC}.yaml").read_text())
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Source PDF checksum mismatch")
    doc = fitz.open(pdf)
    paths = sorted(root.glob("*.yaml"))
    records = {p: yaml.safe_load(p.read_text()) for p in paths}
    notes_text = next(e["explanation"] for e in records.values() if e["id"] == f"{DOC}.clause-4.2")
    letter_notes = dict(re.findall(r"(?ms)^\((\d+)\) (.*?)(?=^\(\d+\) |\Z)", notes_text))
    counts = Counter()
    for path, entry in records.items():
        if entry["source"].get("document_id") not in (None, DOC):
            continue  # Other publishers have their own curation/import workflow.
        before = deepcopy(entry)
        # Hand-verified matrices are already curated supporting references.
        if entry.get("letter_matrix"):
            continue
        short_id = entry["id"].removeprefix(DOC + ".")
        table = entry["source"].get("table")
        item = int(entry["source"].get("item") or 0) if ".t5." in entry["id"] else 0
        metadata = {"role": "background"}
        if not entry["id"].startswith(DOC):
            metadata = {"role": "supporting", "interpretation": entry["explanation"]}
        elif short_id in SUPPORT:
            name, meaning = SUPPORT[short_id]
            metadata = dict(role="supporting", short_name=name, interpretation=meaning)
        elif short_id.startswith("t4.1-letter-"):
            notes = re.findall(r"\((\d+)[a-z]?\)", entry["explanation"])
            qualifications = " ".join(compact(letter_notes[n]) for n in dict.fromkeys(notes) if n in letter_notes)
            metadata = dict(role="supporting", short_name=entry["concept"],
                            interpretation=entry["explanation"] + " " + qualifications)
            entry["tasks"] = ["symbol_interpretation", "text_assignment"]
        elif table == "5.4.4":
            metadata = dict(role="supporting", short_name=entry["concept"],
                            interpretation=compact(entry["explanation"]) +
                            " Failure action must be supported by the drawing annotation, not assumed from valve type.")
        elif table in CATEGORIES and not (table == "5.3.2" and item in (20, 21)):
            category = "connectors" if table == "5.3.2" and item >= 17 else CATEGORIES[table]
            variants = [a for a in entry["assets"] if a["role"] == "variant" and not a["id"].startswith("catalog-")]
            if table == "5.1.1":
                variants = [next(a for a in variants if a["id"].endswith(f"panel-{n}")) for n in (4, 1, 2)]
            selected = []
            pilot = table == "5.3.2" and item in (17, 18, 19)
            for i, asset in enumerate(variants):
                if table == "5.1.1":
                    label = ["Individual instrument", "Primary control system", "Alternate control system"][i]
                elif entry.get("curation") != "source_transcribed":
                    label = asset["label"]
                else:
                    label = "Symbol / illustrated alternatives"
                selected.append(asset if pilot else make_asset(root, doc, entry, asset, label))
            if pilot:
                entry["assets"] = [a for a in entry["assets"] if not a["id"].startswith("catalog-")]
                sheet = next(a for a in entry["assets"] if a["role"] == "variant_sheet")
            else:
                sheet = sheet_asset(root, entry, selected)
                entry["assets"] = [a for a in entry["assets"] if not a["id"].startswith("catalog-")] + selected + [sheet]
            meaning = compact(entry["explanation"])
            if table == "5.2.1" and item > 1:
                meaning += " This is a measurement arrangement; identify its components separately from drawing evidence."
            support = "5.3.1" if table == "5.1.1" else "5.3.2" if table.startswith("5.2") else "5.3.3" if table.startswith("5.3") else "5.3.4"
            links = [f"{DOC}.clause-{support}"]
            if table == "5.1.1" or table.startswith("5.2"):
                links.append(f"{DOC}.clause-4.1")
            metadata = dict(role="catalog", category=category,
                            short_name=NAMES.get(short_id, entry["concept"]), interpretation=meaning,
                            displayed_asset_ids=[a["id"] for a in selected], model_asset_id=sheet["id"],
                            supporting_reference_ids=links)
        entry["catalog"] = metadata
        if entry != before:
            entry["version"] += 1
            path.write_text(yaml.safe_dump(entry, sort_keys=False, allow_unicode=True), encoding="utf-8")
        counts[metadata["role"]] += 1
    doc.close()
    print(dict(counts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT / "data/knowledge-sources/isa-5.1-2009.pdf")
    args = parser.parse_args()
    curate(args.pdf, ROOT / "knowledge/pid")
