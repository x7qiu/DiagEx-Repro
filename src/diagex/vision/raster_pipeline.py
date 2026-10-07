"""Opt-in production routing; the frozen broad study runner is unchanged."""
from __future__ import annotations

import hashlib
from pathlib import Path

from diagex.vision.legend_context import select_legend_context
from diagex.vision.perception import perceive_tile
from diagex.vision.raster_broad import BroadRasterClient, perceive_broad_raster
from diagex.vision.raster_semantics import interpret_raster_reviews


def implementation_sha256():
    root = Path(__file__).parent
    paths = [root / name for name in ("raster_pipeline.py", "raster_broad.py", "raster_semantics.py", "legend_context.py")]
    return hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()


def perceive_broad_with_semantics(**kwargs):
    # A raster inventory must not displace native candidate identities or their
    # established geometry interpretation, even when this mode is opted into.
    if kwargs.get("candidates"):
        return perceive_tile(**kwargs)
    context = kwargs.get("page_context") or {}
    if "raster_proposal_guidance" not in context:
        raise ValueError("Broad review requires source-bound raster guidance")
    client = kwargs["client"]
    # This initializer only stores per-view hints. Transport remains the same
    # metered client, preserving its budget, provider cooldown and request IDs.
    BroadRasterClient.begin_raster_view(client, context)
    entries = kwargs.get("legend_summary") or []
    compact, _, _ = select_legend_context(entries, [], [])
    before = len(kwargs["cost_tracker"].steps)
    try:
        outcome = perceive_broad_raster(**{**kwargs, "legend_summary": compact})
    finally:
        callback = kwargs.get("on_attempt")
        if callback:
            for _ in range(len(kwargs["cost_tracker"].steps) - before):
                callback("submit_raster_symbols")
    reviews, audit = interpret_raster_reviews(
        reviews=outcome.batch.candidate_reviews, client=client, cost_tracker=kwargs["cost_tracker"],
        page=kwargs["page"], tile=kwargs["tile"], view_image=kwargs["view_image"],
        view_info=kwargs["view_info"], legend_entries=entries, policy=kwargs["policy"],
        deadline=kwargs.get("deadline"), on_attempt=kwargs.get("on_attempt"),
    )
    outcome.batch.candidate_reviews = reviews
    outcome.recovery_diagnostics.append({"stage": "raster_legend_interpretation", **audit})
    outcome.attempts += len(audit["attempts"])
    if kwargs.get("on_diagnostic"):
        kwargs["on_diagnostic"](outcome.recovery_diagnostics[-1])
    return outcome
