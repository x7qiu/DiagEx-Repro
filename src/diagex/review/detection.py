"""Revisioned human decisions on immutable, pre-fusion observations."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from diagex.extractors.evidence_checkpoint import atomic_write_json
from diagex.review.raster import RasterReviewObservation, require_semantic_classification
from diagex.vision.legend_models import LegendEntry, LegendPack
from diagex.vision.perception import DetectionRecord, PerceivedObject


class ReviewConflict(ValueError):
    """The browser is editing an obsolete revision."""


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


class DetectionReviewStore:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.path = run_dir / "review" / "detection.json"
        self.undo_path = self.path.with_name("detection.undo.json")
        self.lock = threading.RLock()
        raw = (run_dir / "detection.json").read_bytes()
        self.digest = hashlib.sha256(raw).hexdigest()
        self.bundle = json.loads(raw)

    def _initial(self):
        symbols = []
        selected = set()
        for d in self.bundle["detections"]:
            candidate_id = d.get("attributes", {}).get("symbol_candidate_id")
            selected.add((d["page_index"], candidate_id))
            symbols.append(
                {
                    "id": d["id"],
                    "detection": d,
                    "origin": "detected",
                    "status": "pending",
                    "reason": "Raw model detection",
                }
            )
        decisions = {}
        for r in self.bundle.get("reviews", []):
            decisions.setdefault((r.get("page_index"), r.get("candidate_id")), []).append(r)
        for c in self.bundle.get("candidates", []):
            key = (c["page_index"], c["id"])
            if key in selected:
                continue
            rs = decisions.get(key, [])
            symbols.append(
                {
                    "id": c["id"],
                    "origin": "native_candidate",
                    "status": "pending",
                    "reason": "; ".join(str(r.get("reason", r.get("status", ""))) for r in rs)
                    or "Native candidate without a confirmed detection",
                    "detection": {
                        "id": c["id"],
                        "page_index": c["page_index"],
                        "tile_id": "review",
                        "kind": "equipment",
                        "label": "",
                        "bbox": c["bbox"],
                        "confidence": "low",
                        "source_text_ids": [],
                        "attributes": {
                            "symbol_candidate_id": c["id"],
                            "source_path_ids": c.get("source_path_ids", []),
                        },
                    },
                }
            )
        for index, r in enumerate(self.bundle.get("reviews", [])):
            obj = r.get("object")
            if not obj or not r.get("bbox"):
                continue
            rid = f"proposal-{index}"
            if obj.get("kind") == "raster_symbol":
                observation = self._validate_detection({
                    **obj, "id": rid, "page_index": r["page_index"],
                    "tile_id": r.get("tile_id", "review"), "bbox": r["bbox"],
                })
                if r.get("legend_interpretation"):
                    from diagex.vision.raster_semantics import suggested_detection

                    observation = self._validate_detection(suggested_detection(
                        observation, r["legend_interpretation"],
                    ))
                symbols.append({
                    "id": rid, "detection": observation, "origin": "proposal",
                    "status": "pending", "reason": r.get("reason", "Broad raster observation"),
                    "source_observation": deepcopy(r),
                })
                continue
            perceived = PerceivedObject.model_validate(obj)
            d = {
                **obj,
                "label": perceived.label,
                "raw_text": perceived.raw_text,
                "bbox": r["bbox"],
                "attributes": perceived.graph_attributes(),
                "id": rid,
                "page_index": r["page_index"],
                "tile_id": r.get("tile_id", "review"),
                "confidence": "low",
            }
            symbols.append(
                {
                    "id": rid,
                    "detection": DetectionRecord.model_validate(d).model_dump(mode="json"),
                    "origin": "proposal",
                    "status": "pending",
                    "reason": r.get("reason", "Unanchored proposal"),
                }
            )
        return {
            "version": 1,
            "base_sha256": self.digest,
            "revision": 0,
            "legend_revision": 0,
            "symbols": symbols,
            "legends": [
                {"id": f"legend-{i}", "entry": e, "status": "pending"}
                for i, e in enumerate(self.bundle["legend_pack"]["entries"])
            ],
            "coverage": {
                str(p["page_index"]): False for p in self.bundle["pages"] if p["role"] == "pid"
            },
            "events": [],
        }

    def read(self):
        with self.lock:
            if (
                hashlib.sha256((self.run_dir / "detection.json").read_bytes()).hexdigest()
                != self.digest
            ):
                raise ReviewConflict("Detection inputs changed. Reopen this review.")
            state = json.loads(self.path.read_text()) if self.path.exists() else self._initial()
            if state["base_sha256"] != self.digest:
                raise ReviewConflict(
                    "Detection inputs changed. Start review on a fresh detection run."
                )
            return state

    def public(self):
        state = self.read()
        return {
            **state,
            "pages": self.bundle["pages"],
            "per_page_status": self.bundle["per_page_status"],
            "run_dir": str(self.run_dir),
            "can_build": not self.blockers(state),
            "blockers": self.blockers(state),
            "can_undo": self.undo_path.exists(),
        }

    @staticmethod
    def blockers(state):
        result = []
        for key in ("legends", "symbols"):
            count = sum(r["status"] == "pending" for r in state[key])
            if count:
                result.append(f"{count} {key} still need review")
        count = sum(not v for v in state["coverage"].values())
        if count:
            result.append(f"{count} P&ID pages still need a visual check for missed symbols")
        return result

    def _validate_detection(self, data):
        model = RasterReviewObservation if data.get("kind") == "raster_symbol" else DetectionRecord
        d = model.model_validate(data)
        page = next((p for p in self.bundle["pages"] if p["page_index"] == d.page_index), None)
        b = d.bbox
        if (
            not page
            or min(b.x, b.y) < 0
            or min(b.w, b.h) <= 0
            or b.x2 > page["width"]
            or b.y2 > page["height"]
        ):
            raise ValueError("Symbol box must be non-empty and inside its source page")
        if page["role"] != "pid":
            raise ValueError("Add symbols only on P&ID pages")
        return d.model_dump(mode="json")

    def apply(self, body):
        rater = str(body.get("rater") or "").strip()
        if not rater:
            raise ValueError("Enter the reviewer name")
        actor_type = body.get("actor_type", "human")
        if actor_type not in {"human", "agent"}:
            raise ValueError("actor_type must be human or agent")
        evidence_refs = body.get("evidence_refs", [])
        if not isinstance(evidence_refs, list) or any(not isinstance(r, str) for r in evidence_refs):
            raise ValueError("evidence_refs must be a list of source references")
        if actor_type == "agent" and action_requires_evidence(body.get("action")) and not evidence_refs:
            raise ValueError("Agent review requires source evidence references")
        with self.lock:
            state = self.read()
            if body.get("revision") != state["revision"]:
                raise ReviewConflict("Another edit was saved. Reload before making this change.")
            previous = deepcopy(state)
            action = body.get("action")
            if action == "undo":
                if not self.undo_path.exists():
                    raise ValueError("No edit to undo")
                restored = json.loads(self.undo_path.read_text())
                for key in ("symbols", "legends", "coverage", "legend_revision"):
                    state[key] = restored[key]
            elif action == "coverage":
                key = str(body["page_index"])
                if key not in state["coverage"]:
                    raise ValueError("This is not a P&ID page")
                state["coverage"][key] = bool(body.get("checked"))
            elif action in {"confirm_detected", "confirm_legends", "reject_candidates"}:
                if action == "confirm_detected" and any(
                    r["status"] == "pending" for r in state["legends"]
                ) and not body.get("draft_review", False):
                    raise ValueError("Review the legend before confirming symbols")
                ids = set(body.get("ids", []))
                if not ids:
                    raise ValueError("Select the items to review")
                for row in state["legends" if action == "confirm_legends" else "symbols"]:
                    if row["id"] not in ids:
                        continue
                    if row["status"] != "pending" or row.get("stale"):
                        continue
                    if action == "confirm_detected" and row["origin"] != "detected":
                        continue
                    if action == "reject_candidates" and row["origin"] not in {
                        "native_candidate",
                        "proposal",
                    }:
                        continue
                    row["status"] = "rejected" if action == "reject_candidates" else "confirmed"
            elif action == "split_legend":
                parent = next((r for r in state["legends"] if r["id"] == body.get("id")), None)
                if parent is None:
                    raise ValueError("Unknown review item")
                entries = [LegendEntry.model_validate(e) for e in body.get("entries", [])]
                if len(entries) < 2:
                    raise ValueError("Splitting a legend requires at least two source definitions")
                for entry in entries:
                    source_page = next((p for p in self.bundle["pages"] if p["page_index"] == entry.source_page_index), None)
                    b = entry.source_bbox
                    if source_page is None or b is None or min(b.x,b.y) < 0 or min(b.w,b.h) <= 0 or b.x2 > source_page["width"] or b.y2 > source_page["height"]:
                        raise ValueError("Legend bounds must be inside the source page")
                    label = entry.source_label_bbox
                    if label is not None and (min(label.x,label.y)<0 or min(label.w,label.h)<=0 or label.x2>source_page['width'] or label.y2>source_page['height']):
                        raise ValueError("Legend caption bounds must be inside the source page")
                    entry.image_b64 = None
                    entry.crop_quality = "unavailable"
                    state["legends"].append({"id": "manual-" + uuid.uuid4().hex[:12], "origin": "manual",
                        "parent_id": parent["id"], "entry": entry.model_dump(mode="json"), "status": "pending"})
                parent["status"] = "rejected"
                parent["reason"] = "Split into separately reviewable source definitions"
                state["legend_revision"] += 1
                for symbol in state["symbols"]:
                    if symbol["status"] == "confirmed":
                        symbol.update(status="pending", stale=True, reason="Legend changed; recheck classification")
            elif action in {"save_symbol", "save_legend"}:
                legend = action == "save_legend"
                rows = state["legends" if legend else "symbols"]
                rid = body.get("id")
                row = next((r for r in rows if r["id"] == rid), None)
                if rid and row is None:
                    raise ValueError("Unknown review item")
                if row is None:
                    row = {"id": "manual-" + uuid.uuid4().hex[:12], "origin": "manual"}
                    rows.append(row)
                status = body.get("status", "confirmed")
                if status not in {"pending", "confirmed", "rejected"}:
                    raise ValueError("Invalid review status")
                if legend:
                    entry = LegendEntry.model_validate(body["entry"]).model_dump(mode="json")
                    if entry.get("source_bbox"):
                        page = next(
                            (
                                p
                                for p in self.bundle["pages"]
                                if p["page_index"] == entry.get("source_page_index")
                            ),
                            None,
                        )
                        b = entry["source_bbox"]
                        if (
                            not page
                            or min(b["x"], b["y"]) < 0
                            or min(b["w"], b["h"]) <= 0
                            or b["x"] + b["w"] > page["width"]
                            or b["y"] + b["h"] > page["height"]
                        ):
                            raise ValueError("Legend bounds must be inside the source page")
                    old_entry = row.get("entry", {})
                    label = entry.get("source_label_bbox")
                    if label:
                        page = next((p for p in self.bundle['pages'] if p['page_index']==entry.get('source_page_index')),None)
                        if not page or min(label['x'],label['y'])<0 or min(label['w'],label['h'])<=0 or label['x']+label['w']>page['width'] or label['y']+label['h']>page['height']:
                            raise ValueError("Legend caption bounds must be inside the source page")
                    if old_entry.get("source_bbox") != entry.get("source_bbox"):
                        entry["image_b64"] = None
                        entry["crop_quality"] = "unavailable"
                    # Re-rendering a source thumbnail does not change a symbol
                    # definition. Bounds, wording and semantics still invalidate.
                    def semantic_entry(value):
                        return {k: v for k, v in (value or {}).items()
                                if k not in {"image_b64", "crop_quality"}}
                    changed = semantic_entry(row.get("entry")) != semantic_entry(entry) or (row.get("status") == "rejected") != (
                        status == "rejected"
                    )
                    row["entry"] = entry
                    if changed:
                        state["legend_revision"] += 1
                        for symbol in state["symbols"]:
                            if symbol["status"] == "confirmed":
                                symbol.update(
                                    status="pending",
                                    stale=True,
                                    reason="Legend changed; recheck classification",
                                )
                else:
                    data = {**body["detection"], "id": row["id"]}
                    if data.get("kind") == "raster_symbol" and not row.get("source_observation"):
                        raise ValueError("Unclassified raster items must come from a source observation")
                    if row.get("source_observation"):
                        if status == "confirmed":
                            require_semantic_classification(data)
                        data["attributes"] = {
                            **data.get("attributes", {}),
                            "requires_legend_interpretation": status != "confirmed",
                        }
                    row["detection"] = self._validate_detection(data)
                    row.pop("stale", None)
                    row["legend_revision"] = state["legend_revision"]
                    if status == "confirmed" and any(
                        r["status"] == "pending" for r in state["legends"]
                    ) and not body.get("draft_review", False):
                        raise ValueError("Review the legend before confirming symbols")
                row["status"] = status
            else:
                raise ValueError("Unknown review action")
            provenance = {"actor_type": actor_type, "rater": rater,
                          "evidence_refs": evidence_refs, "note": str(body.get("note", ""))}
            # Attribute only items actually changed by this action, including bulk edits.
            for key in ("symbols", "legends"):
                before = {r["id"]: r for r in previous[key]}
                for row in state[key]:
                    if action != "undo" and row != before.get(row["id"]):
                        row["review_provenance"] = provenance
            state["revision"] += 1
            state["events"].append(
                {
                    "revision": state["revision"],
                    "action": action,
                    "item_id": body.get("id"),
                    "rater": rater,
                    "actor_type": actor_type,
                    "evidence_refs": evidence_refs,
                    "note": str(body.get("note", "")),
                    "item_ids": list(body.get("ids", [])),
                    "page_index": body.get("page_index"),
                    "at": datetime.now(UTC).isoformat(),
                }
            )
            atomic_write_json(self.undo_path, previous)
            atomic_write_json(self.path, state)
            if action == "undo":
                self.undo_path.unlink()
            return self.public()

    def snapshot(self, revision, *, draft=False):
        with self.lock:
            state = self.read()
            if revision != state["revision"]:
                raise ReviewConflict("Review changed. Reload before building the graph.")
            blockers = self.blockers(state)
            if blockers and not draft:
                raise ValueError("; ".join(blockers))
            pack = LegendPack.model_validate(self.bundle["legend_pack"])
            pack.entries = [
                LegendEntry.model_validate({**r["entry"], "attributes": {
                    **r["entry"].get("attributes", {}), "row_status": "accept",
                    "review_origin": r.get("review_provenance", {}).get("actor_type", "human"),
                }})
                for r in state["legends"]
                if r["status"] == "confirmed"
            ]
            return {
                "source_run": str(self.run_dir),
                "source_sha256": self.bundle["source_sha256"],
                "base_sha256": self.digest,
                "revision": revision,
                "build_mode": "draft" if draft else "reviewed",
                "review_origin": "agent" if any(e.get("actor_type") == "agent" for e in state["events"]) else "human",
                "unresolved": {
                    "items": [{"type": key, **deepcopy(r)} for key in ("legends", "symbols")
                              for r in state[key] if r["status"] == "pending"],
                    "unchecked_pages": [int(p) for p, checked in state["coverage"].items() if not checked],
                    "blockers": blockers,
                },
                "legend_pack": pack.model_dump(mode="json"),
                "detections": [
                    {**deepcopy(r["detection"]), "attributes": {
                        **deepcopy(r["detection"].get("attributes", {})),
                        "review_origin": r.get("review_provenance", {}).get("actor_type", "human"),
                        "source_review": deepcopy(r.get("review_provenance", {})),
                    }}
                    for r in state["symbols"] if r["status"] == "confirmed"
                ],
                "review": state,
            }


def action_requires_evidence(action):
    return action in {"coverage", "confirm_detected", "confirm_legends", "reject_candidates", "save_symbol", "save_legend", "split_legend"}
