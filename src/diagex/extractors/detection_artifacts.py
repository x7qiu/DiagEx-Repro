"""Persist raw machine detections and their extraction evidence."""
from diagex.extractors.evidence_checkpoint import atomic_write_json


def write_detection_bundle(
    run_dir, *, source_hash, pages, detections, legend_pack, per_page_status, candidates, reviews
):
    atomic_write_json(
        run_dir / "detection.json",
        {
            "version": 1,
            "source_sha256": source_hash,
            "pages": [
                {"page_index": p.page_index, "width": p.width, "height": p.height, "role": p.role}
                for p in pages
            ],
            "detections": [d.model_dump(mode="json") for d in detections],
            "legend_pack": legend_pack.model_dump(mode="json"),
            "per_page_status": per_page_status,
            "candidates": candidates,
            "reviews": reviews,
        },
    )
