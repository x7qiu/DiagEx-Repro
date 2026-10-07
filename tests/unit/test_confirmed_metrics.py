from eval.pid2graph.confirmed_metrics import affirmed


def test_detector_confidence_and_review_status_do_not_imply_vlm_confirmation():
    prediction = {"disposition": "review_proposal", "confidence": .999,
                  "attributes": {"geometry_basis": "supervised_raster_proposal"}}
    assert not affirmed(prediction)
    prediction["attributes"]["raster_vlm_decision"] = "uncertain"
    prediction["attributes"]["raster_proposal_id"] = "d1"
    assert not affirmed(prediction)
    prediction["attributes"]["raster_vlm_decision"] = "symbol"
    assert affirmed(prediction)


def test_legacy_acceptance_and_explicit_broad_discovery_are_recognition_only():
    assert affirmed({"disposition": "graph_eligible", "attributes": {}})
    prediction = {"disposition": "review_proposal", "attributes": {
        "geometry_basis": "vlm_broad_raster_observation", "broad_category": "arrow",
        "raster_vlm_reason": "Visible triangular arrowhead", "requires_legend_interpretation": True}}
    assert affirmed(prediction)
    assert prediction["disposition"] == "review_proposal" and prediction["attributes"]["requires_legend_interpretation"]
