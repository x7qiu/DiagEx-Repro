from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Standard(Strict):
    name: str = Field(min_length=1)
    edition: str | None = None


class Context(Strict):
    industry: str | None = None
    unit_types: list[str] = Field(default_factory=list)
    company: str | None = None
    project: str | None = None
    conventions: list[str] = Field(default_factory=list)
    standards: list[Standard] = Field(default_factory=list)


class Override(Strict):
    # One-based sheet numbers; an empty list means the entire uploaded drawing.
    pages: list[int] = Field(default_factory=list)
    context: Context

    @model_validator(mode="after")
    def valid_pages(self):
        if any(page < 1 for page in self.pages):
            raise ValueError("Drawing pages are one-based positive numbers")
        return self


class Profile(Strict):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    name: str = Field(min_length=1, max_length=160)
    version: int = Field(ge=1)
    confirmed: Literal[True]
    context: Context
    reference_ids: list[str] = Field(default_factory=list)
    confirmation_evidence: list[dict[str, Any]] = Field(default_factory=list)


class Source(Strict):
    document: str | None = None
    page: str | None = None
    edition: str | None = None
    image: str | None = None
    passage: str | None = None
    document_id: str | None = None
    pdf_page: int | None = Field(default=None, ge=1)
    table: str | None = None
    item: str | None = None
    url: HttpUrl | None = None
    section: str | None = None


class Document(Strict):
    id: str = Field(pattern=r"^[a-z0-9._-]+$")
    title: str
    edition: str | None = None
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_type: Literal["pdf", "web"] = "pdf"
    url: HttpUrl | None = None
    publisher: str | None = None
    retrieved_at: str | None = None
    usage_statement: str | None = None
    scan_note: str | None = None

    @model_validator(mode="after")
    def web_source_identity(self):
        if self.source_type == "web" and not (self.url and self.retrieved_at):
            raise ValueError("Web source requires a URL and retrieval date")
        return self


class WebImage(Strict):
    document_id: str
    url: HttpUrl
    retrieved_at: str
    original_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    original_format: str
    processing: str = "Lossless PNG conversion; no resizing or geometry edits"


class ExcludedAnnotation(Strict):
    text: str
    rect: tuple[float, float, float, float]
    reason: Literal["source_note_reference"] = "source_note_reference"


class Crop(Strict):
    document_id: str
    pdf_page: int = Field(ge=1)
    printed_page: str
    table: str | None = None
    item: str | None = None
    rect: tuple[float, float, float, float]
    coordinates: Literal["unrotated_pdf_points_top_left"] = "unrotated_pdf_points_top_left"
    page_rotation: int = 0
    dpi: int = Field(gt=0)
    renderer: str
    processing: str = "none"
    excluded_annotations: list[ExcludedAnnotation] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_rect(self):
        import math

        x0, y0, x1, y1 = self.rect
        if not all(math.isfinite(v) for v in self.rect) or min(x0, y0) < 0 or x1 <= x0 or y1 <= y0:
            raise ValueError("Crop requires a finite positive rectangle")
        return self


class Asset(Strict):
    id: str = Field(pattern=r"^[a-z0-9._-]+$")
    role: Literal["variant", "context", "note", "variant_sheet"]
    label: str
    path: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    crop: Crop | None = None
    web: WebImage | None = None
    derived_from: list[str] = Field(default_factory=list)
    quality: Literal["clear", "ambiguous"] = "clear"
    depiction: Literal["isolated_symbol", "source_panel"] = "isolated_symbol"

    @model_validator(mode="after")
    def evidence_origin(self):
        if self.crop is not None and self.web is not None:
            raise ValueError("Source image must have only one provenance type")
        if self.role == "variant_sheet":
            if not self.derived_from:
                raise ValueError("Variant sheet requires source asset IDs")
        elif self.crop is None and self.web is None:
            raise ValueError("Source image requires crop or web provenance")
        return self


class TextSlot(Strict):
    role: str
    example_marker: str
    location: str
    meaning: str


class SourceNote(Strict):
    source: Source
    explanation: str
    asset_id: str | None = None


class ReferenceRelation(Strict):
    relation: Literal["component", "alternative", "scope_convention", "related"]
    reference_id: str
    explanation: str


class Mapping(Strict):
    reference_id: str
    reference_version: int = Field(ge=1)
    target_model: str
    target_version: str
    target_kind: Literal["class", "relationship"]
    target: str
    conditions: list[str] = Field(default_factory=list)
    applicability: Context = Field(default_factory=Context)


class ReferenceReplacement(Strict):
    reference_id: str = Field(pattern=r"^[a-z0-9._-]+$")
    replacement_ids: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)
    source_image_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_label: str


class OverlapReview(Strict):
    version: int = Field(ge=1)
    preferred_source: str
    compared_source: str
    policy: str
    replacements: list[ReferenceReplacement]
    retained_examples: list[dict[str, str]] = Field(default_factory=list)


class CatalogMetadata(Strict):
    role: Literal["catalog", "supporting", "background"]
    category: Literal[
        "instruments", "valves_actuators", "measurement", "lines_signals", "connectors",
        "equipment", "piping"
    ] | None = None
    short_name: str | None = None
    # Curated search vocabulary, never inherited source-section headings.
    search_terms: list[str] = Field(default_factory=list)
    interpretation: str | None = None
    displayed_asset_ids: list[str] = Field(default_factory=list)
    model_asset_id: str | None = None
    supporting_reference_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def catalog_content(self):
        if self.role == "catalog" and not (
            self.category and self.short_name and self.interpretation and self.displayed_asset_ids
        ):
            raise ValueError("Catalog entries require a category, name, meaning, and visuals")
        return self


LetterColumn = Literal[
    "measured_variable", "variable_modifier", "readout_function",
    "output_function", "function_modifier",
]
LETTER_COLUMNS = (
    "measured_variable", "variable_modifier", "readout_function",
    "output_function", "function_modifier",
)


class LetterCell(Strict):
    state: Literal["defined", "blank", "user_choice", "unreadable"]
    text: str | None = None
    footnotes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_meaning(self):
        if self.state in {"defined", "user_choice"} and not (self.text or "").strip():
            raise ValueError("A defined letter cell requires its source text")
        if self.state == "blank" and self.text is not None:
            raise ValueError("A blank cell cannot supply a meaning")
        return self


class LetterRow(Strict):
    letter: str = Field(pattern="^[A-Z]$")
    reference_id: str | None = None
    cells: dict[LetterColumn, LetterCell]

    @model_validator(mode="after")
    def complete_columns(self):
        if set(self.cells) != set(LETTER_COLUMNS):
            raise ValueError("A letter row requires all five named columns")
        return self


class LetterNote(Strict):
    marker: str
    text: str
    source: Source
    status: Literal["transcribed", "unresolved"] = "transcribed"


class LetterMatrix(Strict):
    rows: list[LetterRow]
    notes: list[LetterNote] = Field(default_factory=list)
    header_footnotes: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def complete_matrix(self):
        import re
        import string

        if [r.letter for r in self.rows] != list(string.ascii_uppercase):
            raise ValueError("A letter matrix requires exactly A-Z in order")
        markers = [n.marker for n in self.notes]
        if len(set(markers)) != len(markers):
            raise ValueError("Duplicate letter-table note marker")
        refs = [n for r in self.rows for c in r.cells.values() for n in c.footnotes]
        refs += [n for values in self.header_footnotes.values() for n in values]
        if any(n not in markers and re.sub(r"(?<=\d)[a-z]$", "", n) not in markers for n in refs):
            raise ValueError("Unknown letter-table footnote")
        return self


class Entry(Strict):
    id: str = Field(pattern=r"^[a-z0-9._-]+$")
    version: int = Field(ge=1)
    concept: str
    # Absent on older entries and historical run snapshots: retain legacy behavior.
    catalog: CatalogMetadata | None = None
    curation: Literal["curated", "source_transcribed"] = "curated"
    source_status: Literal["normative", "informative", "unspecified"] = "unspecified"
    table_data: list[list[list[str | None]]] = Field(default_factory=list)
    letter_matrix: LetterMatrix | None = None
    aliases: list[str] = Field(default_factory=list)
    reference_type: Literal[
        "symbol", "line_style", "assembly_pattern", "scope_boundary", "convention"
    ] = "convention"
    kind: Literal[
        "definition", "common_practice", "possible_arrangement", "drawing_quality_recommendation"
    ]
    explanation: str
    exceptions: list[str] = Field(default_factory=list)
    applicability: Context = Field(default_factory=Context)
    tasks: list[
        Literal[
            "symbol_interpretation",
            "text_assignment",
            "line_interpretation",
            "connections",
            "review",
        ]
    ]
    tags: list[str] = Field(default_factory=list)
    source: Source
    assets: list[Asset] = Field(default_factory=list)
    notes: list[SourceNote] = Field(default_factory=list)
    text_slots: list[TextSlot] = Field(default_factory=list)
    interpretation: list[str] = Field(default_factory=list)
    relations: list[ReferenceRelation] = Field(default_factory=list)
    # Definitions with the same key and different values are an explicit conflict.
    definition_key: str | None = None
    definition_value: str | None = None
    behavior: Literal["inspect_connector_exception"] | None = None

    @model_validator(mode="after")
    def valid_asset_links(self):
        ids = [asset.id for asset in self.assets]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate reference asset ID")
        variants = {a.id for a in self.assets if a.role == "variant"}
        for asset in self.assets:
            if asset.derived_from and not set(asset.derived_from) <= variants:
                raise ValueError("Variant sheet references unknown variants")
        if any(n.asset_id and n.asset_id not in ids for n in self.notes):
            raise ValueError("Note references unknown asset")
        if self.catalog:
            if not set(self.catalog.displayed_asset_ids) <= variants:
                raise ValueError("Catalog display references unknown variant")
            if self.catalog.model_asset_id and self.catalog.model_asset_id not in ids:
                raise ValueError("Catalog references unknown model asset")
            model_asset = next(
                (a for a in self.assets if a.id == self.catalog.model_asset_id), None
            )
            if model_asset and not (
                model_asset.id in self.catalog.displayed_asset_ids
                or model_asset.role == "variant_sheet"
                and set(model_asset.derived_from) <= set(self.catalog.displayed_asset_ids)
            ):
                raise ValueError("Catalog model image contains undisplayed variants")
        return self
