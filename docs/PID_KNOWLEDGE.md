# Context-aware P&ID knowledge

The Evidence v2 workbench now has a **Drawing context / 图纸背景** section, separate from model
selection. Refresh the workbench at http://127.0.0.1:8765 to load it.

## Use it

Choose one of three conditions:

1. **Off — existing extraction**: compatibility default.
2. **General P&ID knowledge**: shared references with no asserted industry or standards.
3. **General knowledge + confirmed project**: select a saved profile or enter known context,
   then press **Confirm & save project context**. Editing it requires a new confirmation.

Profiles support an industry, multiple process-unit types, company, project, free-text project
conventions, and explicitly declared standards with optional editions. Blank fields remain
unknown. A standard is never selected from the industry name. The UI accepts one standard
per line, `name | edition`, and one project convention per line.

A new profile draft can suggest missing fields from an uploaded document. Suggestions contain
the exact supporting passage, one-based page, source filename and source hash. They are not
applied automatically: use individual suggestions in the draft and confirm the project once.
A confirmed profile can then be selected for later drawings without repeated confirmation.
Suggestions currently recognize explicitly labelled **native PDF text** (English/Chinese);
scanned images and unlabeled prose require manual context entry. No paid suggestion call is made.

**Drawing or sheet exceptions** overrides selected fields for specified one-based pages
(`1,3-5`) or all pages if blank. An enabled empty field clears the inherited value. Multiple
override groups are supported through the API; when groups overlap, the later explicit
field wins. These overrides are run-specific and do not modify the reusable profile.

Profiles are immutable local revisions under `runs/.web/context_profiles/`, independently
of model settings. Revisions are checked to prevent stale updates. They are runtime data,
not automatically committed. The curated references under `knowledge/pid/` are source files.

## Pipeline behavior and provenance

The resolver combines a confirmed profile, explicit sheet overrides, source/legend context,
and the stage's topic. Core principles always accompany enabled knowledge. Up to eight
relevant entries are retrieved using applicability and text search; all sides of structured
applicable definition conflicts are retained. General knowledge cannot override drawing
legends or confirmed project definitions. Free-text conventions and legend disagreements
remain available to interpretation/review; arbitrary prose is not compiled into hard filters.

- Symbol interpretation receives reference text and up to two labelled teaching illustrations,
  separated from source images. Both the normal and CV-guided broad symbol paths receive it.
- Text assignment keeps its geometric algorithm and source identities, records the applicable
  context, and exposes structured reference conflicts. Knowledge does not relocate text.
- Line interpretation and connection reasoning receive task-specific context. Native line
  detection and trained detector weights remain unchanged.
- Candidate fusion can retain an interior/vertical OPC when a source outline, attached line,
  and nearby continuation text independently support further inspection. It does not accept
  a bare arrow merely because the context is utilities. The retained candidate is marked
  `requires_human_review`, with source path/text IDs and reference IDs.
- Unconfirmed placement exceptions cannot automatically create cross-sheet links. Additional source evidence must establish their identity. Drawing quality recommendations do not delete valid symbols.

This first deterministic exception handler requires native/vector source evidence. It does
not assert the same automatic rescue for raster-only scans. VLM reference guidance remains
available for scans and their unresolved candidates remain in the extraction audit.

Each enabled run writes `knowledge.snapshot.json` and `knowledge.stages.json`. Semantic
checkpoints and replayable stage outputs include their supplied reference IDs, profile
version and knowledge identity. The stage trace explicitly means **references supplied**,
not a claim that the model causally used every entry. Geometry-only stage traces document
context without claiming a change to geometry algorithms.

The complete knowledge snapshot (entry contents, illustration hashes, profile version,
overrides and implementation fingerprint) participates in the run cache identity. Independent
stage requests also carry context in their identities. This deliberately invalidates the run
conservatively when knowledge changes. Teaching illustrations are checked against the frozen
hash before use; changing one during a run raises an error instead of silently mixing versions.

## Python and HTTP interfaces

```python
from diagex.knowledge.resolver import knowledge_snapshot

# Pass config to the existing Evidence v2 extractor.
config.knowledge = knowledge_snapshot("general")
# Or use a confirmed Profile dictionary and explicit per-drawing overrides:
config.knowledge = knowledge_snapshot("profile", saved_profile, [
    {"pages": [2], "context": {"unit_types": ["utilities"]}}
])
```

The local workbench exposes:

- `GET /api/context/profiles`: latest revision of each saved profile.
- `POST /api/context/profiles`: name, context, selected reference IDs, `confirmed: true`;
  updates also include ID and `expected_version`.
- `POST /api/context/suggestions`: `upload_id`; returns unconfirmed evidence-backed suggestions.
- `GET /api/context/library`: validated library snapshot.
- `GET /api/context/illustration?id=opc.placement-exceptions`: catalogued source illustration.
- Existing `POST /api/extractions`: `knowledge_mode`, `context_profile_id`,
  `context_profile_version`, and optional `drawing_overrides`.

The server reloads the saved revision and rejects unconfirmed/unknown profile references;
it does not trust an arbitrary profile embedded in an extraction request. Legacy extraction
rejects enabled context rather than silently ignoring it.

## Evaluation

Run the paired offline validation with no model calls:

```sh
.venv/bin/python -m diagex.knowledge.evaluate \
  tests/fixtures/knowledge/connectors.yaml \
  --output docs/knowledge-evaluation.yaml
```

The checked-in manifest contains six **synthetic, annotated source-evidence fixtures**.
The same source evidence and candidate inventory are replayed in all three conditions.
It tests supported interior/vertical connectors and unsupported arrows with missing text,
outline or line evidence. Separate unit tests cover conflicting definitions, unknown context,
explicit standards editions, sheet overrides, profile confirmation and cache invalidation.

The report counts retained true positives, false positives, false negatives, and candidates
still requiring review. Baseline misses both supported exceptions; general and profile
conditions recover both for review without retaining the four negative candidates. The
profile condition intentionally ties general knowledge: this small general seed does not
justify inventing an extra benefit from a utilities label.

This is a **candidate-retention regression**, not measured end-to-end real-PDF extraction
improvement. For real validation, populate the same manifest format with saved PageEvidence,
DetectionRecords, and independently reviewed `expected_connector_ids`. Run paired model
extraction separately for each condition with fixed drawings/models and compare against held-out
annotations; do not tune using the final held-out set. This implementation did not run that
paid study or retrain any detector. The full book has not yet been ingested.

The interactive reviewer and manual graph-build workflow have been removed. Uncertainty
metadata remains part of extraction; it does not imply a pending reviewer session.

## Projectmaterials web source

The visual catalog also contains a separate `projectmaterials.pid-symbols` source from
[Projectmaterials](https://blog.projectmaterials.com/epc-projects/engineering/pid-symbols-list/).
The current snapshot has 406 individual table-cell images, including 46 heat-exchanger
illustrations. After comparison with ISA, 91 overlapping website entries are retired.
315 Projectmaterials entries remain active: 308 in the catalog and 7 background entries
in **All references**, excluded from normal retrieval. Composite overview
charts are listed in the coverage manifest rather than treated as individual symbols.
The publisher advertises 407 symbols, but 406 individual table-cell images were present
in the downloaded page. These references are supplemental, not verified ISA/ISO definitions.

Select **Source filters → Source → Projectmaterials** to browse this source alone.
The new Equipment and Piping & fittings categories include equipment and components
outside the instrumentation-focused ISA source. Original captions and section names
are preserved; the import does not invent technical meanings or standard applicability.

`knowledge/pid/sources/projectmaterials.pid-symbols.yaml` records the page checksum,
URL, publisher, retrieval date, and publisher's engineering-documentation reuse statement.
Each asset has `web` provenance (original image URL, format, checksum and download date)
instead of PDF crop coordinates. PNG conversion preserves pixel dimensions; it does not
recover detail missing from small or compressed original images. Browsing is fully local.

Reproduce the import with `.venv/bin/python scripts/import_projectmaterials.py --download`.
Downloads are cached in ignored `data/knowledge-sources/projectmaterials/`; without
`--download`, the script rebuilds from that cache offline. Existing cached files are reused,
so this is a snapshot importer, not an automatic website refresh. YAML entries and PNGs
stay versioned. Source-specific IDs keep ISA definitions and DEXPI mapping records separate.
Changes to web provenance, meanings, or images participate in existing knowledge cache
identities. Existing two-image limits, drawing-legend precedence, supplied/matched tracing,
and the knowledge-off setting still apply. No extraction-accuracy improvement is claimed.


### ISA overlap preference

`knowledge/pid/overlaps/projectmaterials-isa.yaml` records each confirmed overlap,
its ISA replacement(s), a comparison reason, and the reviewed website image checksum.
The comparison covers meaning and illustrated form, including parameterized instrument
circles and instrument-letter rules; it is not an image-hash or name-only deduplication.
The retained examples document why several similar-looking symbols were not merged.
Heat-exchanger, pump and vessel references remain, as do distinct device subtypes and
unresolved variants. This review does not certify the website illustrations as ISA-compliant.

The 91 replaced website records are moved to `knowledge/pid/retired/`; their PNGs remain
available only for historical asset references. They are absent from both library views
and all new extraction snapshots, including general mode. The importer honors the review
on every rebuild and rejects changed reviewed images before overwriting the saved image.
Old reference links lead to the ISA replacement(s). Saved profiles may still select an old
reference ID: retrieval resolves it to ISA, while retaining the profile's original contents
and enforcing the explicitly declared standard and edition. No standard is inferred.
Website labels remain search terms for the retained ISA entries, without changing the
ISA definitions or assigning example instrument letters to detected objects.

## SH/T 3101:2017 scanned source

The supplied 49-page scan is a separate source, `sht-3101-2017`. The first curated
pass contains **305 records: 276 catalog definitions, 2 supporting notes and 27
background records**, with 318 individual visual variants. It covers process lines
and boundaries, piping components, valves, signal lines, instrument system/location
frames, actuators, failure positions, measurement elements, columns, vessels,
reactors, internals, exchangers, furnaces, pumps, compressors and separation equipment.

Of the background records, 11 are explicitly PFD-only depictions, one is an
unreliably faint tracing-line crop (Table 4.1 item 3), and 15 retain source panels
for letter tables, computing functions, other instruments, auxiliary equipment and
Appendix A arrangements. Those panels are browseable but are not supplied to
extraction. This is a useful P&ID subset, **not a complete machine-readable
transcription or certification of the standard**. Administrative pages and normative
wording explanations are not imported. No DEXPI classes have been assigned.

Browse **Knowledge Library → Source filters → SH/T 3101—2017 石油化工流程图图例**.
Use **All references** for background material. Source details include the original
row, numbered PDF/printed pages, crop rectangles, checksums and scan limitations.
The source PDF is not needed for browsing.

To enable these definitions for extraction, add `SH/T 3101 | 2017` to the confirmed
Drawing context profile's declared standards. Importing or browsing the source does
not enable it automatically. A different or unknown edition does not match, and
drawing overrides and explicit legends still take precedence. Existing ISA and website
entries are preserved: the earlier preference for ISA over overlapping website
illustrations does not replace definitions in a different declared standard.

Important interpretation distinctions retained:

- Table 4.6 distinguishes DCS, SIS, third-party systems and individual instruments.
  These are not ISA's primary/alternate/computer column meanings. `***` is a
  third-party system-type placeholder. Broken or extended instrument frames are
  allowed when the tag does not fit.
- `T1` and optional `T2` on process continuation arrows are drawing/location fields.
  These are process connectors, not instrument-signal connectors.
- `注` inside pump driver examples points to motor/turbine alternatives. It is not
  an actual tag. Reference drivers and positions cannot establish drawing equipment.
- Vendor scope is a boundary, not a page-sized equipment object or a default owner
  of all text inside it. The valve/positioner/solenoid example is an assembly pattern.
- Failure-position arrows are not flow arrows. Curved paired line-break marks are
  not letter S tags. This does not make an actual solenoid `S` meaningless.
- The seven shell-and-tube exchanger alternatives are explicitly P&ID forms.
  Annotated column detail panels remain visible, but only the unannotated primary
  column image is supplied for model comparison.

Rebuild offline with:

```sh
.venv/bin/python scripts/prepare_sht3101_library.py --pdf '/path/to/SHT3101-2017石油化工流程图图例.pdf'
```

The fixed, checked recipe is `knowledge/pid/recipes/sht-3101-2017.yaml`; a different
PDF checksum is rejected before importing. The unmodified PDF is copied to ignored
`data/knowledge-sources/sht-3101-2017.pdf`. PNG/YAML evidence is versioned. On-device
OCR helped draft the curation, but neither OCR nor paid model calls are needed to
rebuild or use it. Each scan page has 1190 × 1684 native pixels; its declared PDF
size is also 1190 × 1684 points. Therefore 72-DPI rendering preserves native pixels.
It does not imply the physical paper was scanned at 72 DPI. Crops are lossless,
without sharpening, resampling or reconstructed strokes. Enlarging a crop does not
recover missing detail; a clearer source is needed for the faint tracing convention.

Checks cover source identity, crop dimensions/checksums, variant selection, model
image budget, scope/assembly distinctions, placeholders, edition applicability,
sheet overrides, knowledge-off behavior and background exclusion. They establish
library/integration contracts, not a measured extraction-accuracy improvement.

## Catalog search

Normal browsing searches symbol names, aliases, and explicitly curated family terms.
Direct name matches rank before aliases, which rank before family matches. All query
words must match within one field; English plurals and hyphens are normalized, and
Chinese terms are supported. A full reference ID can also be searched directly.
Retired website names remain searchable aliases for their ISA replacements.

“Also search descriptions & source notes” deliberately expands the search to
interpretation text, notes and source sections. These results rank below normal
matches and carry a visible explanation. A mention in a source heading is not an
equipment classification. Source/category/edition filters apply in either mode.

`knowledge/pid/recipes/catalog-search.yaml` records the initial curated heat-exchanger
family and deliberate exclusions. `scripts/curate_catalog_search.py` applies it
idempotently, removes inherited Projectmaterials headings from semantic tags and
descriptions, and preserves the original source locations. Reimporting sources
preserves the curated terms. The resolver can use those terms for retrieval, and
changes to them participate in cache identities. This improves search relevance;
it does not establish improved extraction accuracy.

## Selecting run sources and inspecting evidence

Drawing context includes independent checkboxes for ISA 5.1:2009, SH/T 3101-2017
and the Projectmaterials website. Changing a checkbox enables reference knowledge
if it was off. Turning knowledge off explicitly still disables all references.
Checking a standard makes that edition available for this run; an existing project
declaration of a different edition takes precedence, and sheet overrides are applied
last, including explicit empty standards. An empty source selection supplies no
reference entries. Nonempty selections also allow applicable shared general notes.
Selections restrict supporting references and conflict expansion as well as primary
retrieval. They are recorded in the run snapshot/settings and cache identity. Older
callers that omit `knowledge_sources` retain the previous context-based behavior.

The saved-run viewer shows each symbol's recorded evidence origin. Validated model
citations identify the knowledge source, reference/version and variant, or the
drawing legend entry/page and supporting drawing description. Both may contribute.
Built-in legend definitions and customer overrides are labeled separately. Merely
supplying references does not establish that the model used them. Invalid citations
are shown as unaccepted; uncertain symbols are retained. Uncited new VLM results are
labeled as VLM recognition with no reference cited; older records without provenance
say the origin was not recorded. The labels describe recorded evidence, not access
to the model's internal reasoning. No paid extraction is needed to validate these
contracts; a new run is needed to collect new per-symbol citations.

Explicit run source selections also exclude the old unversioned ISA/ISO/SAMA
fallback legend packs. Drawing-derived legends and customer definitions remain
available; versioned standard references come only through the selected library
and applicability resolver. Legacy callers keep their existing fallback behavior.
A symbol can cite multiple knowledge entries using `knowledge_citations`; the
validated `knowledge_matches` list preserves all of them, while `knowledge_match`
retains the first citation for older readers.

## Instrument-letter tables

The library's **Letter tables / 字母含义表** collection (`/knowledge?view=letters`)
contains ISA 5.1:2009 Table 4.1 and SH/T 3101-2017 Table 4.4. These are supporting
conventions, not symbol cards. The existing IDs remain
`isa-5.1-2009.t4.1-p30` and `sht-3101-2017.letter-table`.

An optional `letter_matrix` field on a reference contains exactly A–Z rows and
five named cells per row: `measured_variable`, `variable_modifier`,
`readout_function`, `output_function`, and `function_modifier`. Each cell stores
`state` (`defined`, `blank`, `user_choice`, or `unreadable`), original `text`, and
separate `footnotes`. Blank cells never inherit a meaning from a different
standard. ISA rows link to the existing per-letter reference IDs. Matrix notes
retain source locations and distinguish transcribed notes from unresolved markers.
The SH/T scan's printed `(8)` in the X/readout cell has no corresponding numbered
note on the inspected table/notes pages; it remains explicitly unresolved.

The shared API exposes the matrices. Retrieved references retain their named
cells and note text, with redundant provenance omitted only from model context;
full provenance remains in the library and run snapshot. The existing eight-entry,
40,000-character selection and two-image budgets are unchanged. Source selection,
edition applicability, drawing overrides, and explicit drawing definitions continue
to govern use. Matrix references do not supply missing tags or create drawing objects.

The ISA table and notes were checked against printed pages 25–30. SH/T was checked
against printed pages 10–11 (PDF pages 14–15), preserving the source's bilingual
wording and blank cells; semicolons join printed line breaks. Complete SH/T table
and notes crops supplement the older crops without replacing their asset bytes.
Generic catalog/SH/T recuration preserves these hand-verified matrix records.
A full ISA source reimport remains a destructive regeneration step and requires
reapplying curated data, as with other manual catalog curation.

This addition does not parse project letter matrices, implement tag-number rules,
or change the legend-extraction completeness gate. Validation is offline; no
extraction-accuracy improvement is established by these contract checks.
