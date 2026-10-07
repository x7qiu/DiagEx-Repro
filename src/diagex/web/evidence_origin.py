"""Human-readable projection of recorded attribution, without rewriting old runs."""

from diagex.knowledge.sources import SOURCES


def source_name(source, reference_id=""):
    identity = source.get("document_id")
    if not identity:
        identity = next((key for key in SOURCES if reference_id.startswith(key + ".")), None)
    return SOURCES.get(identity, {}).get("name") or source.get("document") or "未记录参考来源"


def evidence_origin(row, *, is_legend=False):
    attrs = row.get("attributes") or {}
    # Unresolved raster observations retain their interpretation outside attributes.
    interpretation = row.get("legend_interpretation") or {}
    sources = []
    if is_legend:
        origin = row.get("source")
        label = {
            "legend_extracted": "图纸图例表",
            "customer_override": "项目定义",
            "built_in": "内置符号定义",
        }.get(origin, "来源未记录")
        sources.append({"kind": origin or "unknown", "label": label,
                        "page_index": row.get("source_page_index"), "evidence": row.get("description")})
    matches = attrs.get("knowledge_matches") or interpretation.get("knowledge_matches")
    if matches is None:
        matches = [attrs.get("knowledge_match") or interpretation.get("knowledge_match")]
    for match in matches:
        if not isinstance(match, dict) or not match.get("reference_id"):
            continue
        sources.append({"kind": "knowledge", "label": source_name(match.get("source") or {}, match["reference_id"]),
                        "reference_id": match["reference_id"], "reference_version": match.get("reference_version"),
                        "variant_id": match.get("variant_id"), "evidence": match.get("drawing_evidence"),
                        "status": "model_reported"})
    for match in attrs.get("legend_matches", interpretation.get("legend_matches", [])):
        label = {"legend_extracted": "图纸图例表",
                 "customer_override": "项目定义",
                 "built_in": "内置符号定义"}.get(match.get("source"), "图例参考（来源未记录）")
        sources.append({"kind": "legend", "label": label, "entry_label": match.get("label"),
                        "legend_entry_id": match.get("legend_entry_id"),
                        "page_index": match.get("source_page_index"), "evidence": match.get("drawing_evidence"),
                        "status": "model_reported"})
    if not sources:
        if attrs.get("recognition_method") == "vlm" or attrs.get("raster_vlm_reason"):
            label = "视觉模型识别，未引用参考"
        elif attrs.get("legend_entry_ids"):
            label = "已记录图例引用，缺少来源详情"
        elif row.get("candidate_only"):
            label = "未分类候选"
        else:
            label = "未记录识别依据来源"
        sources.append({"kind": "other", "label": label,
                        "evidence": attrs.get("recognition_evidence") or attrs.get("raster_vlm_reason")})
    supplied = attrs.get("supplied_knowledge") or interpretation.get("supplied_knowledge") or {}
    supplied_names = sorted({source_name(e.get("source") or {}, e.get("id", "")) for e in supplied.get("references", [])})
    errors = [attrs.get("knowledge_match_error") or interpretation.get("knowledge_match_error")]
    errors.extend(e.get("reason") for e in attrs.get("knowledge_match_errors", interpretation.get("knowledge_match_errors", [])))
    errors.extend(e.get("reason") for e in attrs.get("legend_match_errors", interpretation.get("legend_match_errors", [])))
    return {"sources": sources, "summary": " + ".join(dict.fromkeys(s["label"] for s in sources)),
            "supplied_sources": supplied_names, "citation_errors": list(dict.fromkeys(e for e in errors if e)),
            "note": "引用由模型报告并校验编号，未经独立验证；提供参考不代表已采用。"}


def recognition_diagnostic(row):
    """Chinese presentation of saved states; never invent a missed detection."""
    attrs = row.get("attributes") or {}
    status = row.get("saved_status") or row.get("status")
    reason = row.get("reason") or row.get("saved_reason") or ""
    failure = row.get("request_failure")
    if failure:
        code, label = failure["code"], failure["label"]
    elif row.get("processing_status") == "not_processed":
        code, label = "not_processed", "运行已停止，此候选尚未处理"
    elif "contradict" in reason or "geometry" in reason:
        code, label = "validation_rejected", "校验未通过，已保留为待核查项"
    elif row.get("candidate_only") or status in {"uncertain", "unresolved", "unreviewed"}:
        code, label = "uncertain", "候选已定位，但类型尚未确定"
        if row.get("source_candidate_absent") or (row.get("object") and not row.get("candidate_id")):
            code, label = "no_native_candidate", "模型发现了额外图形，尚无原生候选，需校验定位"
    else:
        code, label = "detected", "已识别（模型判断，仍可人工核查）"
    supplied = attrs.get("supplied_knowledge") or row.get("supplied_knowledge") or {}
    matched = attrs.get("knowledge_matches") or attrs.get("knowledge_match")
    if matched:
        reference_status = "模型报告已使用知识库参考；编号已校验，语义未独立验证"
    elif supplied.get("references"):
        reference_status = "已提供知识库参考，但模型未报告使用；不能视为匹配成功"
    else:
        reference_status = "此项未记录已提供的知识库参考"
    return {"code": code, "label": label, "reference_status": reference_status}

