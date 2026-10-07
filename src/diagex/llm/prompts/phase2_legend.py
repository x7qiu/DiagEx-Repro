"""Phase 2 legend-extraction system prompt (spec §7.2.2).

Used by `extractors.pid_legend.resolve_legend` when it runs a preparatory
ReAct pass over a legend sheet / abbreviation table. Distinct from the full
Phase 2 P&ID prompt: the agent's job here is to walk the legend rows and emit
one `annotate` per printed symbol, not to classify equipment in a live sheet.

Layout mirrors `phase1_query.py`: one stable cached block + one volatile
framing block so the starter text stays cache-stable across runs.
"""

from __future__ import annotations

from typing import Any

from diagex.dexpi_schema import (
    render_legend_equipment_list,
    render_legend_instrument_list,
    render_legend_valve_list,
)

# Stable — computed at module load from the DEXPI registry so the cache key
# is deterministic for a given registry state. Adding a class regenerates
# this string exactly once per process.
_STARTER_LEGEND_PROMPT_TEMPLATE = """\
You are extracting a symbol / abbreviation legend from an industrial P&ID.
The page (or region) you are viewing is a legend sheet: a grid of printed
symbols next to their meanings, or a table of abbreviations and classes.
Your output becomes a few-shot reference block for the downstream extractor;
accuracy of label + class matters more than geometric precision.

Your approach:
1. Call get_overview first to see the layout of rows / columns.
2. Call list_tiles and inspect every assigned detail tile. Python schedules
   the regions of the legend page. Inspect all columns and bottom rows within
   this region before finishing. Title blocks, revision tables, and borders
   are not legend entries; a region containing only these can yield no entries.
3. For each legible entry -- symbol glyph with a printed label -- emit one
   annotate call:
     - Annotate immediately after viewing the image that contains the glyph.
       Always set view_tag to the exact view_tag returned with that image.
       Never reuse overview, tile, or region coordinates with a different
       view_tag. bbox is in pixels of that named view, not page coordinates.
     - kind: one of "equipment", "instrument", "line", "connection"
       * Use "instrument" for ISA-style bubbles (FI, FIC, PT, LSH, ...).
       * Use "equipment" for pumps, tanks, vessels, heat exchangers, valves
         drawn as body-on-a-line glyphs.
       * Use "line" for pipe/line-style legends (process, signal, capillary).
       * Use "connection" for connector / off-page-connector style glyphs.
     - label: the printed name verbatim (e.g. "FIC", "butterfly valve",
       "signal, pneumatic"). Preserve case.
     - bbox: a tight box around the glyph itself (not the text column).
     - confidence: "high" only when the label text is clearly legible.
     - attributes MUST include:
         legend_kind: "symbol" for drawn glyphs, "abbreviation" for
           text-only abbreviation-table rows.
         legend_symbol_class: one of the DEXPI/ISA vocabulary values:
           instruments: %%INSTRUMENT_LIST%%
           equipment: %%EQUIPMENT_LIST%%
           valves (when kind=equipment and equipment_class=valve):
             %%VALVE_LIST%%
           lines/connectors: line, connector
       Pick the closest fit; if nothing fits, use "unclassified_equipment"
       or "unclassified_instrument".
     - attributes may include: legend_description (short gloss written next
       to the glyph), legend_standard (isa-5.1 / iso-10628 / sama / none).
4. When a row has no printed label or the label is illegible, skip it --
   do not guess.
5. Use get_region to zoom into rows where the label text is unclear before
   committing.
6. Call finish(answer="done") when every legible entry has been annotated.
   The Phase 2 legend pass does not produce a user-facing answer; `finish`
   just terminates the loop.

Never invent labels or classes. A smaller, accurate legend is better than a
padded one with guesses.\
"""


STARTER_LEGEND_PROMPT = (
    _STARTER_LEGEND_PROMPT_TEMPLATE
    .replace("%%INSTRUMENT_LIST%%", render_legend_instrument_list())
    .replace("%%EQUIPMENT_LIST%%", render_legend_equipment_list())
    .replace("%%VALVE_LIST%%", render_legend_valve_list())
)


def build_legend_system_prompt(*, effort_max_steps: int) -> list[dict[str, Any]]:
    """Return legend-extraction system-prompt blocks.

    Layout: [cached starter, volatile step-budget reminder]. The starter
    carries `cache_control`; the trailing block changes per effort and is
    not cached.
    """
    stable_block: dict[str, Any] = {
        "type": "text",
        "text": STARTER_LEGEND_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
    volatile_block: dict[str, Any] = {
        "type": "text",
        "text": (
            f"You have up to {effort_max_steps} tool-use steps for this "
            f"legend extraction. Finish early once every legible entry has "
            f"been annotated."
        ),
    }
    return [stable_block, volatile_block]
