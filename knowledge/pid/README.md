# Curated P&ID reference library

Each YAML file is one versioned reference entry, validated by `diagex.knowledge.models.Entry`.
Illustrations live under `illustrations/` and are content-hashed with the entry. Edit entries
here and increment their version when changing their interpretation. Git tracks the readable
library and its seed illustration; named project profiles select shared entries without
copying their contents.

The initial two entries derive only from the user-supplied off-page connector illustration.
Book title, edition and printed page are **unknown**. These are qualified teaching examples,
not a declaration of an industry standard or a rule for every company. The whole supplied
image is retained so readers can inspect the qualifications in context.

## Visual standards pilot

The Knowledge Library is available at `/knowledge` in the existing workbench. It is
read-only and independent of model settings and extraction runs. The ISA 5.1:2009
pilot contains Table 5.3.2 items 17, 18, and 19, including all four item-17 alternatives,
variable text fields, full source rows, table headings, and the referenced note on page 34.
These are **signal** references, not definitions of process-piping connections.

Drawing context includes a library link and an Applicable references panel, with
thumbnails and descriptions linking to the full reference in a separate tab. The
panel uses the same applicability rules as extraction, including sheet overrides;
its sheet selector previews the chosen page. Eligible references are available for
topic retrieval, not necessarily supplied to every stage. Knowledge-off and
unconfirmed/edited profiles show no active references, while all other entries
remain browsable under Not currently applied. Browsing never changes a profile.
The read-only `/api/context/applicable` POST accepts the extraction context fields
and a one-based `page`, and returns eligible IDs without saving any state.

The three layers stay separate:

1. Reference definitions describe appearances, meaning, applicability and evidence.
2. Drawing occurrences own actual geometry, tags, text assignments and connections.
3. Optional mapping records describe a target model/version and conditional class or
   relationship mapping. No DEXPI mappings are claimed by this pilot.

Root YAML files remain reference entries. `sources/*.yaml` are document manifests;
`mappings/*.yaml` are optional separate mapping records. `illustrations/` contains PNGs.
The original PDF is stored locally at `data/knowledge-sources/isa-5.1-2009.pdf` (ignored
by Git). It is not required to browse the library or run extraction. The manifest's
SHA-256 identifies the exact original used, without exposing its local path via HTTP.

Each new entry has:

- `reference_type`: symbol, line_style, assembly_pattern, scope_boundary or convention.
  This differs from `kind`, which distinguishes definitions from practices and advice.
- `aliases`, `text_slots`, `interpretation`, linked `notes`, and optional typed `relations`.
- `assets`: individually named variants, source context, note crops, and a labeled
  variant sheet. Every source crop records an unrotated, top-left PDF-point rectangle,
  page number, printed page, table/item, DPI, renderer, dimensions and checksum.
  A sheet identifies its component asset IDs. UI thumbnails never replace source crops.
- Legacy `source.image` remains supported. Existing entries need no migration.

To reproduce the pilot after obtaining the same source PDF:

```sh
.venv/bin/python scripts/prepare_isa_symbol_pilot.py --pdf '/path/to/ISA_5.1 (2009).pdf'
```

The recipe is explicitly curated for this PDF's pagination; it does not import arbitrary
standards. Visually inspect crops after any coordinate/rendering change. Increment entry
versions when evidence or interpretation changes and update image checksums together.
Low-quality scanned sources should retain their original evidence and use `quality:
ambiguous`; enlarging a scan does not recover missing detail.

## Runtime and downstream use

`knowledge.library.load_library` is the shared validated catalog. The existing
`/api/context/library` returns its normalized records in a version-2 knowledge snapshot.
`/api/context/asset?id=<reference-id>&asset=<asset-id>` serves catalogued PNGs only.
`/api/context/illustration?id=<reference-id>` preserves legacy single-image access.

Declare **ISA 5.1**, edition **2009**, in a confirmed Drawing context profile to make the
pilot applicable. General knowledge mode does not silently assume that standard.
Selecting or browsing a library card does not change profile applicability. Explicit
drawing legends take precedence, and drawing-specific overrides can remove a standard.

Symbol interpretation receives text definitions and up to two labeled reference images,
using variant sheets to preserve alternatives within the budget. Both VLM-only and
CV-assisted semantic interpretation use this interface. No detector weights change.
The source drawing remains the only evidence for objects, geometry and connections.

Stage contexts record supplied asset IDs, hashes and reference versions. Detected objects
can additionally carry `knowledge_match` with a reference/variant ID and a model-reported
description of supporting drawing evidence. These IDs are validated against supplied
references and images; they do **not** constitute independently verified matches. Invalid
attributions are recorded separately and do not erase otherwise valid native objects.
CV classifications relying solely on an invalid reference remain unresolved. Actual
source tags stay on occurrences; exact reference placeholder tokens are not accepted as tags.

Definitions, notes, document manifests and asset checksums participate in cache identities.
Image edits without updated manifests fail explicitly, as do missing assets and unsafe
paths. Old saved runs remain readable; fresh extraction uses the new snapshot identity.
Fixtures for future line styles, assemblies and scope boundaries live in tests, not in the
production knowledge catalog, and implement no new extraction or topology behavior.

An entry includes:

- Stable `id`, integer `version`, `concept`, `explanation`, and `exceptions`.
- `kind`: `definition`, `common_practice`, `possible_arrangement`, or
  `drawing_quality_recommendation`.
- `applicability`: industry, unit types, company, project, and explicitly named
  standards/editions. Empty applicability means general; a scoped reference needs a matching
  confirmed context. Selecting an entry does not override its applicability.
- `tasks`: symbol interpretation, text assignment, line interpretation, connections, review.
- `tags`: retrieval terms, including Chinese aliases where helpful.
- `source`: document, printed page, edition, supporting passage, relative image path.
- For structured definitions, `definition_key` and `definition_value`. Conflicting applicable
  values remain together in the resolver's `conflicts` list, including beyond the retrieval cap.

`behavior: inspect_connector_exception` is the one implemented deterministic behavior.
Do not add arbitrary instructions expecting them to become executable filters. Additional
behaviors require implementation and regression cases. Definitions and project conventions
are evidence for interpretation, never proof that an equipment item or edge exists.

The initial text search uses applicability plus tags/concept/explanation terms. No vector
service, detector retraining, or web service on port 8000 is involved.


## Complete ISA 5.1:2009 source import

The library now contains 451 ISA references: the three unchanged curated pilot entries
plus 448 source-transcribed entries. The two earlier general connector references remain.
Coverage includes all 263 numbered items in Tables 5.1.1 through 5.8, all 66 glossary
definitions, identification letters and matrices, measurement notations, dimension tables,
explanatory clauses, and informative Annexes A and B. Technical material on 116 pages is
accounted for in `coverage/isa-5.1-2009.yaml`; cover, contents, committee lists, blank pages,
and publisher material are explicitly excluded. Coverage measures source inclusion, not
engineering validation or measured extraction accuracy.

Reproduce the added records (the original PDF checksum must match the source manifest):

```sh
.venv/bin/python scripts/prepare_isa_library.py
```

The script validates complete item sequences, handles both tables on pages 43 and 82, normalizes
landscape pages in memory, and renders 300 DPI lossless evidence. Crop coordinates remain
in the original unrotated PDF coordinate system with the original rotation recorded.
The original PDF is never modified. This is an edition-specific importer, not a generic
parser for arbitrary standards. Rerunning overwrites generated records; preserve deliberate
curation edits before rerunning. The three original pilot files are not overwritten.

New records expose `curation: source_transcribed` and `source_status` (normative or
informative). The UI displays these distinctions. Symbol panels preserve annotations and
all alternatives, including alternate logic glyphs printed in the definition column;
`depiction: source_panel` distinguishes these from isolated glyph crops. Complete rows,
page context, explanatory notes, and linked clause records remain inspectable. Annex
worked examples are assembly patterns, not single equipment detections. Dimension
recommendations are retrieved for review, not symbol interpretation. Dense identification
matrices also retain `table_data` as raw cells; null/blank cells and merged headers require
consulting the source image. They are not silently filled with inferred values.

The source's page 65 NAND/NOR headings and truth descriptions appear inconsistent. Both
rows retain the original evidence and an explicit exception; the import does not silently
rewrite the logic or treat the row as verified executable behavior. The Table A.2 heading
points to A.16.3 while A.16.2 explicitly discusses that table; both references are linked
and the discrepancy is visible.

Browsing loads 24 cards at a time with lazy images and supports Show more and full-catalog
search, including source text. Image requests validate only the requested catalog asset;
file parsing and image metadata checks are memoized with modification/creation timestamps,
size, and inode so changed evidence is revalidated. Retrieval weights specific concept terms,
handles hyphenated terms, and caps ordinary selection at eight intact records and a soft
40,000-character serialized budget. A single oversized record and applicable definition
conflicts are preserved rather than truncated. The existing two-reference-image limit stays
in effect. ISA entries still require declared ISA 5.1 edition 2009, and knowledge-off stays off.

Validation includes complete coverage, source/table links, image hashes/dimensions, rotated
page provenance, applicability, legacy compatibility, mutation invalidation, bounded
retrieval, and browser checks. No training, paid model calls, DEXPI mappings, or accuracy
improvement claims are part of this import.

## Focused symbol catalog

The default `/knowledge` view shows 158 visual references in five categories:
instruments, valves/actuators, measurement devices, lines/signals, and connectors.
The Full standard view retains all 453 records, including the original computer
function variants, logic diagrams, glossary, drafting rules, and annex examples.

Optional `catalog` metadata is shared by the UI and resolver:

- `role`: `catalog`, `supporting`, or `background`.
- `category`, `short_name`, and `interpretation`: the concise browsing/prompt view.
- `displayed_asset_ids`: the ordered variant IDs visible in the simple catalog.
- `model_asset_id`: a labeled sheet containing only those selected variants.
- `supporting_reference_ids`: explicit links to relevant supporting definitions.

New extraction requests select from catalog and supporting records only, including
conflict expansion and profile-pinned selections. Linked applicable/task-relevant
notes accompany selected symbols within the existing eight-reference budget.
Original source text, raw table cells, source-page images, and unselected variants
remain available in the archive instead of being sent to the model. The existing
two-image budget, applicability checks, drawing-legend precedence, and supplied vs.
matched attribution remain in force. Old records/snapshots without catalog metadata
retain legacy behavior. Existing single-image references remain supported.

Instruments retain individual hardware, primary control-system, and alternate
control-system variants. Their system meanings are distinct; the project chooses
whether A/B denote primary/alternate or BPCS/SIS. Computer-system variant C stays
in the source archive. No standard is activated by merely browsing the library.

Reproduce the focused layer after a full source import:

```sh
.venv/bin/python scripts/curate_isa_catalog.py
```

This checksum-pinned script retains source files and original images, renders
300-DPI PNG crops with blank margins trimmed and top-left source-note citations
removed, and adds labeled selected variant sheets. Citation removal uses text-only
redaction on an in-memory PDF copy; original pages and crops remain unchanged.
Each cleaned crop records `excluded_annotations` (text, original PDF rectangle,
and reason). Geometry, meaningful letters and placeholders are preserved.
Panels may still contain alternative drawings and their a)/b) labels: neither a panel nor an example arrangement is automatically one detected
object. The three previously curated connector records retain their original
variant IDs. Regeneration is idempotent; changed records receive a new version.
Custom curation should be preserved before rerunning either preparation script.

Validation covers selection, tracing, legacy snapshots, API compatibility and
browser behavior; it does not demonstrate extraction accuracy improvements.
