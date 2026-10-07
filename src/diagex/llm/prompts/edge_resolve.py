"""System prompt for the focused edge-resolve ReAct sub-loop.

Used by `vision/edge_resolve.py` to give the agent a tight, single-purpose brief
when re-examining one dangling line endpoint at a time. Stays small so the
cached portion fits in a single block; only the per-target user message
mentions concrete coordinates.
"""

from __future__ import annotations

from typing import Any

STARTER_EDGE_RESOLVE_PROMPT = """\
You are auditing one suspect line endpoint on a P&ID extraction. The first
view returned by get_overview is a focused crop of the page centred on a
single dangling endpoint -- a pipe or signal line that the main extraction
either left mid-air or could not snap to a node. Your job is narrow:

1. Look at the focused overview. The dangling endpoint sits near the centre
   (the user message gives exact view-pixel coordinates). Identify what kind
   of terminator the line should reach: equipment nozzle, instrument bubble,
   off-page connector arrow, junction with another line, or an explicit
   dead-end glyph (cap, blind flange).

2. If the answer is unclear at the overview scale, call get_region with a
   smaller window (in page-pixel coordinates) to zoom in. You may zoom up to
   two levels deep. Stay within the focused area -- the rest of the drawing
   has already been processed.

3. Decide:
   (a) The line CONTINUES to a real terminator visible in the focus area.
       Emit one annotate(kind="line", endpoints=[...], ...) call whose first
       endpoint is within ~10 view pixels of the marked dangling point and
       whose last endpoint sits on the terminator (a nozzle of the equipment
       box, the centre of the instrument bubble, the tip of the OPC arrow,
       or the junction point on another line). Trace every visible bend.
       Set line_type to match the original line's type when given. Then call
       finish(answer="continued").
   (b) The line genuinely TERMINATES at the marked point (visible cap, blind
       flange, or it really does dead-end at a labelled boundary). Do not
       emit a new annotation. Call finish(answer="terminates").
   (c) UNCERTAIN even after zooming. Do not emit a guess. Call
       finish(answer="uncertain").

Hard constraints:
- Annotate ONLY kind="line" or kind="connection" in this pass. Do not record
  equipment, instruments, valves, or text -- those have already been
  extracted by the main walk and re-emitting them creates duplicates.
- Do not call list_tiles or get_tile -- only get_overview and get_region are
  available.
- Prefer confidence="medium". Use "low" if you traced the continuation with
  doubt; never use "high" in this pass since you only see a small window.
- One finish call ends the loop. Do not annotate after finishing.\
"""


def build_edge_resolve_system_prompt(
    *,
    line_type: str | None,
    endpoint_xy: tuple[int, int],
    focus_window_px: int,
) -> list[dict[str, Any]]:
    """Return cached system blocks for one dangling-endpoint resolution.

    The starter block carries `cache_control={"type": "ephemeral"}` so it
    re-uses prompt-cache hits across every target in the same run. The
    per-target framing (line type, endpoint coordinates, window size) lives
    in a small volatile trailing block.
    """
    starter_block: dict[str, Any] = {
        "type": "text",
        "text": STARTER_EDGE_RESOLVE_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }

    lt = line_type or "unknown"
    ex, ey = endpoint_xy
    volatile_text = (
        f"This target: line_type={lt}; dangling endpoint sits at "
        f"page-pixel ({ex}, {ey}); the focused overview is a "
        f"{2 * focus_window_px}x{2 * focus_window_px}px window centred there. "
        "Your view-pixel coordinates after get_overview are relative to the "
        "focused crop, NOT the full page; the runtime projects them back."
    )
    volatile_block: dict[str, Any] = {
        "type": "text",
        "text": volatile_text,
    }

    return [starter_block, volatile_block]
