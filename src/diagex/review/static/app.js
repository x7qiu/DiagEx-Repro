"use strict";

const el = id => document.getElementById(id);
const I18N = {
  en: {
    showIssue:"Show issue", issueArea:"ISSUE AREA", issuePoint:"CHECK THIS POINT", relatedArea:"RELATED AREA", pageIssue:"Page-level finding: no exact location is recorded.", locatedIssue:"Highlighted in both panes", offPageIssue:"This finding is on another page. Use Show issue.",
    reviewScope:"Review scope", decisionQueue:"Needs your decision", humanSignoff:"Awaiting human approval", fullReview:"Full human review",
    findingCategories:"Finding categories", identifiers:"Identifiers", symbols:"Symbols", otherFindings:"Other findings", allFindings:"All findings",
    searchFindings:"Search label, reason or ID", filterByPage:"Filter by page", nextFinding:"Next finding →", findingCount:"{count} findings",
    showingFindings:"{count} shown", decisionHelp:"Start with flagged decisions across all pages. Routine approvals are in a separate view.",
    signoffHelp:"Every object still needs human approval, including items without a flagged problem.", noFindings:"No open findings match. Full human review may still be pending.",
    sourceLocations:"Jump to source", auditDecision:"Record decision / clarification", decisionNote:"Describe the clarification or why the printed value should be retained.",
    decisionRequired:"Record your decision before resolving this finding.", sourceAudit:"Source audit evidence", flaggedConnection:"Check the connection route and meaning.",
    untaggedObject:"Untagged {kind}", conflictVectorTitle:"Connection route needs confirmation", conflictVectorQuestion:"Check the visible route between {nodes}, including crossings and symbol ports.",
    title: "DiagEx Review", humanReview: "human review", previousPage: "Previous page", nextPage: "Next page",
    pageRole: "Page role", approvePage: "Approve page", waive: "Waive", zoomOut: "Zoom out", zoomIn: "Zoom in",
    fit: "Fit", inferenceBackground: "Inference background", backgroundFull: "PDF + overlays",
    backgroundDim: "Dimmed PDF", backgroundNone: "Overlays only", language: "Language", undo: "Undo",
    finishExport: "Finish & export", reviewQueue: "Review queue", loading: "Loading…", needsReview: "Needs review",
    allItems: "All items", allTypes: "All types", entities: "Entities", connections: "Connections", conflicts: "Conflicts",
    originalSource: "Original source", originalPidPage: "Original P&ID page", editableExtraction: "Editable extraction",
    reviewed: "reviewed", pending: "pending", rejected: "rejected", inferenceFrame: "Inference rendering frame",
    inspector: "Inspector", drawMissingEntity: "Draw missing entity", addConnection: "Add connection",
    comparisonView: "Comparison view", comparisonLayout: "Comparison layout", sideBySide: "Side by side", stacked: "Stacked",
    hideQueue: "Hide queue", showQueue: "Show queue", hideInspector: "Hide inspector", showInspector: "Show inspector",
    selectItem: "Select an entity, connection, or conflict.", pageCount: "Page {current} / {count}", pageShort: "page {page}",
    remaining: "{count} remaining", readyExport: "Ready to export", noMatches: "Nothing matches this filter.",
    entity: "entity", connection: "connection", conflict: "conflict", graphConflict: "Graph conflict",
    modelConfidence: "model confidence", review: "review", label: "Label", kind: "Kind",
    suggestedLabel: "Suggested label", chooseSuggestedLabel: "Choose a recovered label…", enterManually: "Enter another label…",
    equipmentClass: "Equipment class", valveType: "Valve type", actuatorType: "Actuator type", instrumentFunction: "Instrument function",
    sourceText: "Printed source text", attributesJson: "Attributes (JSON)", from: "From", to: "To", lineType: "Line type",
    visualStyle: "Detected appearance", styleConfidence: "style confidence", styleEvidence: "source style evidence",
    conflictInstanceGeometryTitle: "Overlapping object identities", conflictInstanceGeometryQuestion: "Check whether these observations describe one symbol or separate devices; they do not establish a pipe connection.", assembly: "Assembly", assemblies: "Assemblies", assemblyIdentity: "Assembly identity", assemblyHelp: "This identity belongs to a group of components. Confirm the group name; components and pipe connections remain separate.", assemblyMembers: "Components in this assembly", routeChoices: "Choose the visible route", routeChoice: "Route {number}", routePrompt: "Select a route to preview", polylineJson: "Polyline points (JSON)", saveChanges: "Save changes", approve: "Approve", reject: "Reject",
    resolved: "Resolved", waiveReason: "Waive with reason", rejectReason: "Reason for rejection (optional)",
    resolveReason: "How was this resolved? (optional)", conflictWaiveReason: "Reason for waiving this conflict",
    pageWaiveReason: "Reason for waiving this page", actionSaved: "{operation} saved",
    drawInstruction: "Drag a box around the missing entity in the editable pane",
    twoEntitiesRequired: "At least two active entities are required", fromEntityPrompt: "From entity ID or exact label",
    toEntityPrompt: "To entity ID or exact label", unresolvedEntity: "Could not resolve one of those entities",
    exportSuccess: "Reviewed DEXPI JSON and XML exported", loadFailed: "Failed to load review: {message}",
    unlabelled: "unlabelled", reasonPartialPage: "partial page", reasonGraphConflict: "graph conflict",
    reasonBuildIssue: "build issue", reasonMediumConfidence: "medium confidence", reasonLowConfidence: "low confidence",
    reasonUnclassified: "unclassified", reasonDanglingEndpoint: "dangling endpoint",
    candidates: "Suggested choices", candidateNodes: "Candidate entities", relatedConnections: "Related connections",
    competingLabels: "Possible printed labels", competingKinds: "Possible object types", candidatePairs: "Possible OPC pairs",
    dragCandidateHint: "Drag an entity onto From or To. On the drawing, you can also drag a connection endpoint onto an entity.",
    useAsFrom: "Use as From", useAsTo: "Use as To", view: "View", rawDetails: "Raw details",
    noCandidates: "No explicit alternatives were recovered.", crossing: "Treat as crossing", junction: "Treat as junction",
    keepEntity: "Keep entity", keepConnection: "Keep connection", editConnection: "Edit connection",
    reviewedAgainstSource: "reviewed against source", selectedCandidate: "selected candidate {value}", score: "score",
    warningMalformedAudit: "ignored malformed events.jsonl line {line}",
    warningMissingPage: "node {node} references missing page {page}",
    warningOutsidePage: "node {node} falls outside rendered page {page}",
    workspaceView: "Workspace view", nodeInventory: "Node inventory", missingCandidates: "Potential missing objects",
    inventorySearch: "Search tag, ID, or source text", evidenceSearch: "Search printed text", page: "Page", allPages: "All pages",
    subtype: "Subtype", isolated: "Isolated", duplicateTag: "Duplicate tag", printedText: "Printed text",
    candidateType: "Candidate type", disposition: "Disposition", nearbyEntities: "Nearby entities",
    linked: "Linked", dismissed: "Dismissed", linkToEntity: "Link to entity", drawFromCandidate: "Draw missing entity",
    dismissAs: "Dismiss as", reopen: "Reopen", recognizedNodes: "Recognized nodes", reviewedNodes: "Reviewed nodes",
    isolatedNodes: "Isolated nodes", unresolvedTags: "Unresolved tag candidates", pageCoverage: "Page coverage",
    evidenceCandidate: "source evidence candidate", chooseNearbyEntity: "Choose a nearby entity", blockingCandidate: "Must be reviewed",
    nonBlockingCandidate: "Context evidence", evidenceLinked: "Evidence linked", evidenceDismissed: "Evidence dismissed",
    noEvidenceArtifacts: "No native-text evidence artifacts were found for this run.",
    relatedConflicts: "Related conflicts", conflictCount: "{count} unresolved conflicts", reviewConflict: "Review conflict",
    nodeApprovalDoesNotResolve: "Approving this object confirms only its identity and properties. Resolve its relationship conflicts separately.",
    decisionNeeded: "Decision needed", affectedObjects: "Affected objects", modelEvidence: "Why this was flagged",
    currentInterpretation: "Current interpretation", suggestedInterpretation: "Suggested interpretation", currentGraph: "Current graph", detectedLine: "detected {style} line", noDirectConnection: "No direct connection is currently recorded",
    markNoConnection: "Confirm no direct connection", addAs: "Add as {type}", changeTo: "Change to {type}",
    confirmType: "Confirm {type}", supportingEvidence: "Supporting evidence", sameLoopEvidence: "Both instruments use loop {loop}",
    functionFlowEvidence: "Compatible function flow: {from} → {to}", alignedRouteEvidence: "The traced route aligns with both endpoints",
    legendMatchEvidence: "The visual/legend classification is {type}",
    rejectConnection: "Reject this connection", resolveNoChange: "Mark resolved without graph changes",
    conflictUnsupportedTitle: "Possible false connection", conflictRoleTitle: "Uncertain relationship type",
    conflictCandidateTitle: "Unconfirmed proposed connection", conflictTopologyTitle: "Ambiguous line routing",
    conflictCrossingTitle: "Crossing or junction?", conflictMissingTitle: "Possibly missing relationship",
    conflictEvidenceTitle: "Conflicting object evidence", conflictOpcTitle: "Uncertain off-page connection",
    conflictCompositeTitle: "Possible composite valve symbol", questionComposite: "Is the small glyph part of the adjacent valve actuator?",
    evidenceComposite: "The local symbol may be an actuator attached to the valve, but the project legend or geometry did not support automatic merging with high confidence.",
    mergeActuator: "Merge actuator into valve", suggestedActuation: "Actuator type",
    questionDirectConnection: "Are {nodes} directly connected by a {type}?",
    questionLineMeaning: "What does the {style} line between {nodes} represent?",
    questionConfirmSuggested: "Confirm that the relationship between {nodes} is a {type}?",
    questionProposedConnection: "Does a direct connection really exist between {nodes}?",
    questionCrossing: "Do these lines cross without connecting, or form a junction?",
    questionEvidence: "Which interpretation of {nodes} matches the printed drawing?",
    questionGenericConflict: "What should the reviewed graph record for {nodes}?",
    evidenceUnsupported: "The proposed relationship is incompatible with the identified engineering-object roles.",
    evidenceRole: "The route is visible, but its endpoints and appearance do not prove a single relationship type.",
    evidenceSuggested: "The system suggests {type} from the project legend, route, and endpoint semantics. Confirm it against the source drawing.",
    evidenceCandidate: "A deterministic route candidate was found, but the page-level solver could not confirm it.",
    evidenceTopology: "The extracted vector paths can be traced in more than one way through this area.",
    evidenceCrossing: "The geometry does not reliably distinguish a crossing from a connected junction.",
    evidenceMissing: "The page context suggests a relationship may be absent from the graph.",
    evidenceObject: "The visual detection and printed-document evidence disagree about this object.",
    evidenceOpc: "The off-page reference, direction, or matching sheet evidence is incomplete or inconsistent.",
    resolutionComplete: "This conflict has been resolved. Use Undo if the decision was incorrect.",
    interpretedMeaning: "Interpreted engineering meaning", modelSuggestedMeaning: "Model-suggested meaning",
    confirmedNoConnection: "confirmed that no direct connection exists",
    resolvedWithoutChange: "reviewed and resolved without changing the graph"
  },
  "zh-CN": {
    title: "DiagEx 人工复核", humanReview: "人工复核", previousPage: "上一页", nextPage: "下一页",
    pageRole: "页面类型", approvePage: "批准本页", waive: "豁免", zoomOut: "缩小", zoomIn: "放大",
    fit: "适应窗口", inferenceBackground: "推断视图背景", backgroundFull: "PDF + 标注层",
    backgroundDim: "淡化 PDF", backgroundNone: "仅显示标注层", language: "语言", undo: "撤销",
    showIssue:"定位问题", issueArea:"问题区域", issuePoint:"检查此处", relatedArea:"相关区域", pageIssue:"页面级问题：未记录精确位置。", locatedIssue:"已在两个窗格中突出标记", offPageIssue:"此问题位于其他页面，请点击定位问题。",
    reviewScope:"复核范围", decisionQueue:"需要您判断", humanSignoff:"等待人工确认", fullReview:"完整人工复核",
    findingCategories:"问题分类", identifiers:"编号冲突", symbols:"符号含义", otherFindings:"其他问题", allFindings:"全部问题",
    searchFindings:"搜索编号、原因或 ID", filterByPage:"按页筛选", nextFinding:"下一个问题 →", findingCount:"{count} 个问题",
    showingFindings:"显示 {count} 项", decisionHelp:"优先处理所有页面中的明确问题。常规人工确认可切换到单独视图。",
    signoffHelp:"每个对象都仍需人工确认，包括没有明确问题的项目。", noFindings:"当前筛选下没有待处理问题；完整人工复核可能仍未完成。",
    sourceLocations:"定位到原图", auditDecision:"记录判断或澄清", decisionNote:"请记录澄清结果，或说明保留原图编号的理由。",
    decisionRequired:"请先记录判断，再解决此问题。", sourceAudit:"原图审查证据", flaggedConnection:"确认连接路径及含义。",
    untaggedObject:"未编号的{kind}", conflictVectorTitle:"连接路径需确认", conflictVectorQuestion:"检查 {nodes} 之间的可见路径，包括交叉点和符号接口。",
    finishExport: "完成并导出", reviewQueue: "复核队列", loading: "正在加载…", needsReview: "待复核",
    allItems: "全部项目", allTypes: "全部类型", entities: "实体", connections: "连接", conflicts: "冲突",
    originalSource: "原始图纸", originalPidPage: "原始 P&ID 页面", editableExtraction: "可编辑提取结果",
    reviewed: "已复核", pending: "待处理", rejected: "已拒绝", inferenceFrame: "推断结果渲染页面",
    inspector: "属性检查器", drawMissingEntity: "框选缺失实体", addConnection: "添加连接",
    comparisonView: "对照视图", comparisonLayout: "对照布局", sideBySide: "左右排列", stacked: "上下排列",
    hideQueue: "隐藏队列", showQueue: "显示队列", hideInspector: "隐藏检查器", showInspector: "显示检查器",
    selectItem: "请选择实体、连接或冲突。", pageCount: "第 {current} / {count} 页", pageShort: "第 {page} 页",
    remaining: "剩余 {count} 项", readyExport: "可以导出", noMatches: "没有符合当前筛选条件的项目。",
    entity: "实体", connection: "连接", conflict: "冲突", graphConflict: "图结构冲突",
    modelConfidence: "模型置信度", review: "复核状态", label: "标签", kind: "种类",
    suggestedLabel: "候选标签", chooseSuggestedLabel: "请选择图纸中识别到的标签…", enterManually: "手动输入其他标签…",
    equipmentClass: "设备类别", valveType: "阀门类型", actuatorType: "执行机构类型", instrumentFunction: "仪表功能",
    sourceText: "图纸原文", attributesJson: "属性（JSON）", from: "起点", to: "终点", lineType: "线型",
    visualStyle: "检测到的线条外观", styleConfidence: "外观置信度", styleEvidence: "线型来源证据",
    conflictInstanceGeometryTitle: "对象身份或位置重叠", conflictInstanceGeometryQuestion: "请核对这些观察结果属于同一符号还是不同设备；它们尚不能证明存在管道连接。", assembly: "成套设备", assemblies: "成套设备", assemblyIdentity: "成套设备标识", assemblyHelp: "此标识属于一组组件。请确认整体名称；组件和管道连接分别保留。", assemblyMembers: "包含的组件", routeChoices: "选择图中实际路线", routeChoice: "路线 {number}", routePrompt: "选择路线以预览", polylineJson: "折线坐标（JSON）", saveChanges: "保存修改", approve: "批准", reject: "拒绝",
    resolved: "已解决", waiveReason: "填写理由并豁免", rejectReason: "拒绝理由（可选）",
    resolveReason: "如何解决此问题？（可选）", conflictWaiveReason: "请输入豁免此冲突的理由",
    pageWaiveReason: "请输入豁免本页的理由", actionSaved: "已保存：{operation}",
    drawInstruction: "请在可编辑视图中拖动鼠标，框选缺失实体",
    twoEntitiesRequired: "至少需要两个有效实体", fromEntityPrompt: "请输入起点实体 ID 或完整标签",
    toEntityPrompt: "请输入终点实体 ID 或完整标签", unresolvedEntity: "无法找到其中一个实体",
    exportSuccess: "已导出复核后的 DEXPI JSON 和 XML", loadFailed: "复核页面加载失败：{message}",
    unlabelled: "未命名", reasonPartialPage: "页面提取不完整", reasonGraphConflict: "图结构冲突",
    reasonBuildIssue: "构建问题", reasonMediumConfidence: "中置信度", reasonLowConfidence: "低置信度",
    reasonUnclassified: "未分类", reasonDanglingEndpoint: "连接端点悬空",
    candidates: "建议选项", candidateNodes: "候选实体", relatedConnections: "相关连接",
    competingLabels: "可能的图纸标签", competingKinds: "可能的对象类型", candidatePairs: "可能的跨页连接配对",
    dragCandidateHint: "将候选实体拖到起点或终点。也可在图中把连接端点直接拖到实体上。",
    useAsFrom: "设为起点", useAsTo: "设为终点", view: "查看", rawDetails: "原始详情",
    noCandidates: "未找到明确的候选项。", crossing: "按交叉处理", junction: "按节点处理",
    keepEntity: "保留实体", keepConnection: "保留连接", editConnection: "编辑连接",
    reviewedAgainstSource: "已对照原图复核", selectedCandidate: "已选择候选项 {value}", score: "评分",
    warningMalformedAudit: "已忽略 events.jsonl 中格式错误的第 {line} 行",
    warningMissingPage: "节点 {node} 引用了不存在的第 {page} 页",
    warningOutsidePage: "节点 {node} 位于第 {page} 页的渲染范围之外",
    workspaceView: "工作区视图", nodeInventory: "节点清单", missingCandidates: "可能遗漏的对象",
    inventorySearch: "搜索标签、ID 或图纸原文", evidenceSearch: "搜索图纸原文", page: "页码", allPages: "全部页面",
    subtype: "子类型", isolated: "孤立节点", duplicateTag: "重复标签", printedText: "图纸文字",
    candidateType: "候选类型", disposition: "处理结果", nearbyEntities: "附近实体",
    linked: "已关联", dismissed: "已排除", linkToEntity: "关联到实体", drawFromCandidate: "框选缺失实体",
    dismissAs: "排除为", reopen: "重新处理", recognizedNodes: "已识别节点", reviewedNodes: "已复核节点",
    isolatedNodes: "孤立节点", unresolvedTags: "未处理标签候选", pageCoverage: "页面覆盖情况",
    evidenceCandidate: "来源证据候选", chooseNearbyEntity: "选择附近实体", blockingCandidate: "必须复核",
    nonBlockingCandidate: "上下文证据", evidenceLinked: "证据已关联", evidenceDismissed: "证据已排除",
    noEvidenceArtifacts: "此运行没有可用的原生文字证据文件。",
    relatedConflicts: "相关冲突", conflictCount: "{count} 个未解决冲突", reviewConflict: "查看冲突",
    nodeApprovalDoesNotResolve: "批准此对象只确认其身份和属性；相关连接冲突需要单独处理。",
    decisionNeeded: "需要做出决定", affectedObjects: "涉及的对象", modelEvidence: "为什么被标记",
    currentInterpretation: "当前解释", suggestedInterpretation: "建议解释", currentGraph: "当前图结构", detectedLine: "检测到的{style}", noDirectConnection: "当前图中没有记录直接连接",
    markNoConnection: "确认没有直接连接", addAs: "添加为{type}", changeTo: "改为{type}",
    confirmType: "确认{type}", supportingEvidence: "支持依据", sameLoopEvidence: "两个仪表使用同一回路号 {loop}",
    functionFlowEvidence: "仪表功能关系合理：{from} → {to}", alignedRouteEvidence: "追踪到的线路与两个端点均对齐",
    legendMatchEvidence: "图例与视觉判断均指向{type}",
    rejectConnection: "拒绝此连接", resolveNoChange: "不修改图并标记为已解决",
    conflictUnsupportedTitle: "可能的错误连接", conflictRoleTitle: "关系类型不确定",
    conflictCandidateTitle: "尚未确认的候选连接", conflictTopologyTitle: "线路走向不明确",
    conflictCrossingTitle: "交叉还是连接节点？", conflictMissingTitle: "可能遗漏的关系",
    conflictEvidenceTitle: "对象证据相互冲突", conflictOpcTitle: "跨页连接不确定",
    conflictCompositeTitle: "可能的组合阀门符号", questionComposite: "这个小符号是否属于相邻阀门的执行机构？",
    evidenceComposite: "局部符号可能是安装在阀门上的执行机构，但图例或几何证据不足以支持自动合并。",
    mergeActuator: "将执行机构合并到阀门", suggestedActuation: "执行机构类型",
    questionDirectConnection: "{nodes} 之间是否确实存在一条{type}？",
    questionLineMeaning: "{nodes} 之间的{style}表示什么关系？",
    questionConfirmSuggested: "是否确认 {nodes} 之间是{type}关系？",
    questionProposedConnection: "{nodes} 之间是否真的存在直接连接？",
    questionCrossing: "这些线只是交叉通过，还是在此处连接？",
    questionEvidence: "{nodes} 的哪一种解释与原图一致？",
    questionGenericConflict: "复核后的图应当如何记录 {nodes}？",
    evidenceUnsupported: "候选关系与已识别工程对象的角色不兼容。",
    evidenceRole: "线路走向清晰可见，但端点角色和线条外观不足以唯一确定关系类型。",
    evidenceSuggested: "系统根据项目图例、线路走向和端点工程含义建议判定为{type}；请对照原图确认。",
    evidenceCandidate: "确定性拓扑发现了候选路径，但整页关系求解未能确认它。",
    evidenceTopology: "该区域的矢量路径存在多种可能的连接方式。",
    evidenceCrossing: "现有几何证据无法可靠区分线条交叉与连接节点。",
    evidenceMissing: "页面上下文表明图中可能遗漏了一条关系。",
    evidenceObject: "视觉检测结果与图纸文字证据对该对象的判断不一致。",
    evidenceOpc: "跨页编号、方向或目标图纸证据不完整或不一致。",
    resolutionComplete: "此冲突已解决。如果决定有误，请使用顶部的撤销按钮。",
    interpretedMeaning: "工程含义", modelSuggestedMeaning: "模型建议含义",
    confirmedNoConnection: "已确认不存在直接连接",
    resolvedWithoutChange: "已核对原图并在不修改图的情况下解决"
  }
};

const VALUE_LABELS_ZH = {
  unreviewed: "未复核", approved: "已批准", modified: "已修改", rejected: "已拒绝", waived: "已豁免", resolved: "已解决",
  approve: "批准", modify: "修改", reject: "拒绝", waive: "豁免", resolve: "解决", add: "添加", undo: "撤销",
  high: "高", medium: "中", low: "低", pid: "P&ID", legend: "图例", cover: "封面", notes: "说明", other: "其他",
  equipment: "设备", instrument: "仪表", line: "管线", connection: "连接", text: "文字", note: "注释", opc: "跨页连接",
  process: "工艺管线", signal_electric: "电信号线", signal_pneumatic: "气动信号线",
  instrument_capillary: "仪表毛细管", electrical_power: "电源线",
  solid: "实线", dashed: "虚线", dotted: "点线", dash_dot: "点划线", unknown: "未知",
  iou_grey_zone: "重叠判定不明确", unstitched_line_endpoint: "管线端点未拼接",
  dropped_edge_unsnappable: "丢弃的连接无法吸附", ambiguous_opc: "跨页连接不明确",
  linked: "已关联", dismissed: "已排除", equipment_tag: "设备标签", instrument_tag: "仪表标签",
  unknown_tag: "待判定标签", line_number: "管线编号", drawing_reference: "图纸编号", dimension: "尺寸", non_object: "非对象",
  tank: "储罐", vessel: "容器", column: "塔器", heat_exchanger: "换热器", pump: "泵", compressor: "压缩机",
  reactor: "反应器", agitator: "搅拌器", filter: "过滤器", separator: "分离器", fired_heater: "加热炉",
  cooling_tower: "冷却塔", fan_blower: "风机/鼓风机", turbine: "透平", centrifuge: "离心机", dryer: "干燥机",
  weigher: "称量设备", mixer: "混合器", transport_system: "输送系统", burner: "燃烧器", air_cooler: "空冷器",
  unclassified_equipment: "未分类设备", motor: "电机", cooler: "冷却器", level_gauge: "液位计",
  gate: "闸阀", globe: "截止阀", check: "止回阀", ball: "球阀", butterfly: "蝶阀", control: "调节阀",
  safety_relief: "安全泄压阀", three_way: "三通阀", needle: "针阀", plug: "旋塞阀", strainer: "过滤器",
  rupture_disc: "爆破片", indicator: "指示器", transmitter: "变送器", controller: "控制器", recorder: "记录器",
  element: "检测元件", switch: "开关", alarm: "报警", valve_actuator: "阀门执行机构", unclassified_instrument: "未分类仪表",
  manual: "手动执行机构", solenoid: "电磁执行机构", electric_motor: "电动执行机构", pneumatic: "气动执行机构",
  hydraulic: "液压执行机构", spring: "弹簧执行机构",
  pressure: "压力", temperature: "温度", flow: "流量", level: "液位", analysis: "分析", in: "入口", out: "出口",
  control_loop: "控制回路", instrumentation_loop: "仪表回路", unlabelled: "未命名"
};

const VALUE_LABELS_EN = {
  process: "process line", signal_electric: "electrical signal", signal_pneumatic: "pneumatic signal",
  instrument_capillary: "instrument capillary", electrical_power: "electrical power",
  equipment_tag: "equipment tag", instrument_tag: "instrument tag", unknown_tag: "unclassified tag",
  line_number: "line number", drawing_reference: "drawing reference", dimension: "dimension", non_object: "not an engineering object",
  heat_exchanger: "heat exchanger", fired_heater: "fired heater", cooling_tower: "cooling tower",
  fan_blower: "fan / blower", transport_system: "transport system", air_cooler: "air cooler",
  unclassified_equipment: "unclassified equipment", level_gauge: "level gauge", safety_relief: "safety / relief valve",
  three_way: "three-way valve", rupture_disc: "rupture disc", valve_actuator: "valve actuator",
  electric_motor: "electric motor actuator", pneumatic: "pneumatic actuator", hydraulic: "hydraulic actuator",
  solenoid: "solenoid actuator", manual: "manual actuator", spring: "spring actuator",
  unclassified_instrument: "unclassified instrument", control_loop: "control loop", instrumentation_loop: "instrumentation loop"
};

const DESCRIPTIVE_LABELS = {
  en: {
    showIssue:"Show issue", issueArea:"ISSUE AREA", issuePoint:"CHECK THIS POINT", relatedArea:"RELATED AREA", pageIssue:"Page-level finding: no exact location is recorded.", locatedIssue:"Highlighted in both panes", offPageIssue:"This finding is on another page. Use Show issue.",
    "压缩空气入口": "Compressed air inlet", "压缩空气出口": "Compressed air outlet",
    "仪表空气入口": "Instrument air inlet", "仪表空气出口": "Instrument air outlet",
    "冷却水入口": "Cooling-water inlet", "冷却水出口": "Cooling-water outlet",
    "氮气入口": "Nitrogen inlet", "氮气出口": "Nitrogen outlet"
  },
  "zh-CN": {
    "compressed air inlet": "压缩空气入口", "compressed air outlet": "压缩空气出口",
    "instrument air inlet": "仪表空气入口", "instrument air outlet": "仪表空气出口",
    "cooling water inlet": "冷却水入口", "cooling-water inlet": "冷却水入口",
    "cooling water outlet": "冷却水出口", "cooling-water outlet": "冷却水出口",
    "nitrogen inlet": "氮气入口", "nitrogen outlet": "氮气出口"
  }
};

function storedSetting(key) {
  try { return localStorage.getItem(key); } catch (_) { return null; }
}

function storeSetting(key, value) {
  try { localStorage.setItem(key, String(value)); } catch (_) { /* Setting still applies for this tab. */ }
}

const storedLayout = storedSetting("diagex.review.layout");
const storedWorkspaceView = storedSetting("diagex.review.workspaceView");
const app = {
  state: null, page: 0, zoom: 1, selection: null, syncing: false, drawMode: false,
  activeViewport: "inferenceViewport",
  sourceFocus: null, pendingEvidence: null, queueCategory: "all", queueSelection: null, decisionDrafts: {},
  lang: storedSetting("diagex.review.language") === "zh-CN" ? "zh-CN" : "en",
  layout: ["side-by-side", "stacked"].includes(storedLayout) ? storedLayout : (window.matchMedia("(max-width: 1600px)").matches ? "stacked" : "side-by-side"),
  workspaceView: ["comparison", "inventory", "evidence"].includes(storedWorkspaceView) ? storedWorkspaceView : "comparison",
  queueVisible: storedSetting("diagex.review.queueVisible") !== "false",
  inspectorVisible: storedSetting("diagex.review.inspectorVisible") !== "false"
};

function t(key, variables = {}) {
  const template = I18N[app.lang]?.[key] ?? I18N.en[key] ?? key;
  return template.replace(/\{(\w+)\}/g, (_, name) => String(variables[name] ?? `{${name}}`));
}

function displayValue(value) {
  if (value == null || value === "") return "";
  if (app.lang === "zh-CN" && VALUE_LABELS_ZH[value]) return VALUE_LABELS_ZH[value];
  if (app.lang === "en" && VALUE_LABELS_EN[value]) return VALUE_LABELS_EN[value];
  return String(value);
}

function containsChinese(value) { return /[\u3400-\u9fff]/.test(String(value || "")); }
function localizedDescription(value) {
  const raw=String(value || "").trim(); if(!raw)return "";
  return DESCRIPTIVE_LABELS[app.lang]?.[raw.toLowerCase()] || raw;
}
function bilingualDescription(node) {
  const description=String(node?.attributes?.structural_description || "").trim(); if(!description)return "";
  const match=/^(.+?)\s*\(([^()]+)\)\s*$/.exec(description);
  if(match&&containsChinese(match[1])&&/[A-Za-z]/.test(match[2]))return app.lang==="zh-CN"?match[1].trim():match[2].trim();
  if(app.lang==="zh-CN"&&!containsChinese(description))return "";
  return localizedDescription(description);
}
function displayNodeLabel(node) {
  if(!node)return "";
  const raw=String(node.label || node.id || "");
  if(raw==="unlabelled"){
    const attrs=node.attributes || {}, service=String(attrs.service || "").trim(), reference=String(attrs.drawing_ref || "").trim();
    if(node.kind==="opc"&&service&&reference)return attrs.direction==="in"?`${reference} → ${service}`:attrs.direction==="out"?`${service} → ${reference}`:`${service} · ${reference}`;
    if(node.kind==="opc"&&(service||reference))return service||reference;
    return bilingualDescription(node) || displayValue(raw);
  }
  if(node.kind==="opc"&&app.lang==="zh-CN"&&containsChinese(node.attributes?.service))return `${node.attributes.service}${node.attributes.direction?`（${displayValue(node.attributes.direction)}）`:""}`;
  if(app.lang==="zh-CN"&&!containsChinese(raw)){
    const source=String(node.source_quote || "").trim(); if(containsChinese(source))return source;
    const description=bilingualDescription(node); if(containsChinese(description))return description;
  }
  return localizedDescription(raw);
}
function nodeMeaning(node) {
  if(!node)return {text:"",suggested:false};
  const attrs=node.attributes || {};
  if(node.kind==="instrument"){
    const parts=[attrs.measured_variable,attrs.instrument_function].filter(value=>value&&value!=="other").map(displayValue);
    return {text:parts.join(" · "),suggested:false};
  }
  if(node.kind==="equipment"){
    const canonical=attrs.valve_type || attrs.equipment_class;
    if(canonical&&!["other","unclassified_equipment"].includes(canonical)){
      const parts=[displayValue(canonical),attrs.actuation?displayValue(attrs.actuation):""].filter(Boolean);
      return {text:parts.join(" · "),suggested:false};
    }
    if(attrs.actuation)return {text:displayValue(attrs.actuation),suggested:false};
    if(attrs.model_equipment_class)return {text:displayValue(attrs.model_equipment_class),suggested:true};
  }
  if(node.kind==="opc"){
    const parts=[attrs.direction,attrs.service].filter(Boolean).map(value=>displayValue(value)===String(value)?localizedDescription(value):displayValue(value));
    return {text:parts.join(" · "),suggested:false};
  }
  return {text:"",suggested:false};
}
function nodeMeaningHtml(node) {
  const meaning=nodeMeaning(node); if(!meaning.text)return "";
  return `<div class="meaning-summary"><span>${meaning.suggested?t("modelSuggestedMeaning"):t("interpretedMeaning")}</span><strong>${escapeHtml(meaning.text)}</strong></div>`;
}

function nodeLabelCandidates(node) {
  const attrs=node?.attributes || {}, values=[
    ...(attrs.label_candidates || []),
    ...(attrs.raw_text_candidates || []),
    attrs.canonical_tag,
    ...(node?.alternate_readings || []),
    node?.source_quote
  ];
  if(node?.kind==="opc"){
    const service=String(attrs.service || "").trim(), reference=String(attrs.drawing_ref || "").trim();
    if(service&&reference)values.unshift(attrs.direction==="in"?`${reference} → ${service}`:attrs.direction==="out"?`${service} → ${reference}`:`${service} · ${reference}`);
    values.push(service,reference);
  }
  const seen=new Set(), result=[];
  for(const value of values){const text=String(value || "").trim();if(!text||["unlabelled","unlabeled","unknown","none","na"].includes(text.toLowerCase()))continue;const key=text.toUpperCase();if(seen.has(key))continue;seen.add(key);result.push(text);}
  return result;
}

function nodeLabelEditorHtml(node) {
  const candidates=nodeLabelCandidates(node), current=String(node.label || "");
  if(!candidates.length)return inputField(t("label"),"nodeLabel",current);
  const selected=candidates.some(value=>value===current)?current:"";
  const options=[`<option value="">${escapeHtml(t("chooseSuggestedLabel"))}</option>`,...candidates.map(value=>`<option value="${escapeHtml(value)}" ${value===selected?"selected":""}>${escapeHtml(value)}</option>`),`<option value="__custom__">${escapeHtml(t("enterManually"))}</option>`].join("");
  return `<label class="field">${t("suggestedLabel")}<select id="nodeLabelCandidate">${options}</select></label>${inputField(t("label"),"nodeLabel",current)}`;
}

function bindNodeLabelEditor() {
  const select=el("nodeLabelCandidate"), input=el("nodeLabel"); if(!select||!input)return;
  select.onchange=()=>{if(select.value&&select.value!=="__custom__")input.value=select.value;if(select.value==="__custom__"){input.focus();input.select();}};
}

function translateReason(reason) {
  const keys = {
    "partial page": "reasonPartialPage", "graph conflict": "reasonGraphConflict", "build issue": "reasonBuildIssue",
    "medium confidence": "reasonMediumConfidence", "low confidence": "reasonLowConfidence",
    unclassified: "reasonUnclassified", "dangling endpoint": "reasonDanglingEndpoint"
  };
  return keys[reason] ? t(keys[reason]) : reason;
}

function translateWarning(warning) {
  let match = /^ignored malformed events\.jsonl line (\d+)$/.exec(warning);
  if (match) return t("warningMalformedAudit", {line: match[1]});
  match = /^node (\S+) references missing page (\d+)$/.exec(warning);
  if (match) return t("warningMissingPage", {node: match[1], page: match[2]});
  match = /^node (\S+) falls outside rendered page (\d+)$/.exec(warning);
  if (match) return t("warningOutsidePage", {node: match[1], page: match[2]});
  return warning;
}

const RELATION_CONFLICTS = new Set([
  "unsupported_endpoint_combination", "endpoint_role_uncertain", "page_graph_uncertain_candidate",
  "page_graph_uncertainty", "page_graph_missing", "visual_topology_disagreement"
]);

function unresolvedConflict(item) { return item && !["resolved", "waived"].includes(item.status); }
function conflictNodes(item) { return (item?.candidates?.nodes || []).map(candidate => nodeById(candidate.id) || candidate); }
function conflictEdges(item) {
  return (item?.candidates?.edges || []).map(candidate => edges().find(edge => edge.id === candidate.id) || candidate);
}
function relatedConflicts(targetType, targetId, unresolvedOnly = true) {
  return Object.entries(app.state?.reviews?.conflicts || {}).filter(([, item]) => {
    if (unresolvedOnly && !unresolvedConflict(item)) return false;
    if (targetType === "node") return (item.candidates?.nodes || []).some(node => node.id === targetId);
    if (targetType === "edge") return (item.candidates?.edges || []).some(edge => edge.id === targetId);
    return false;
  });
}
function conflictNodeNames(item) {
  const names = conflictNodes(item).map(reviewNodeName);
  return names.length ? names.join(" ↔ ") : t("affectedObjects").toLowerCase();
}
function conflictTitle(item) {
  if (item?.conflict?.title) return app.lang === "zh-CN" ? item.conflict.title_zh || item.conflict.title : item.conflict.title;
  const type = item?.conflict?.type || "";
  if (type === "unsupported_vector_route") return t("conflictVectorTitle");
  if (type === "physical_instance_geometry") return t("conflictInstanceGeometryTitle");
  if (type === "contextual_symbol_uncertainty") return t("conflictCompositeTitle");
  if (type === "unsupported_endpoint_combination") return t("conflictUnsupportedTitle");
  if (type === "endpoint_role_uncertain") return t("conflictRoleTitle");
  if (type === "page_graph_uncertain_candidate") return t("conflictCandidateTitle");
  if (type === "page_graph_uncertainty" || type === "visual_topology_disagreement") return t("conflictTopologyTitle");
  if (type === "crossing_or_junction") return t("conflictCrossingTitle");
  if (type === "page_graph_missing") return t("conflictMissingTitle");
  if (type.includes("opc")) return t("conflictOpcTitle");
  if (type.includes("evidence") || type === "rejected_non_connectable_text") return t("conflictEvidenceTitle");
  return t("graphConflict");
}
function conflictLineStyle(item) {
  const edge = conflictEdges(item)[0];
  const style = edge?.attributes?.visual_style || item?.conflict?.visual_style || "unknown";
  return displayValue(style === "unknown" ? "line" : style);
}
function conflictProposedType(item) {
  const edge = conflictEdges(item)[0];
  return item?.conflict?.proposed_line_type || edge?.attributes?.proposed_line_type || edge?.line_type || "connection";
}
function conflictSuggestedType(item) {
  const edge = conflictEdges(item)[0], visual=edge?.attributes?.system_confidence_evidence?.visual_assessment;
  const proposed = conflictProposedType(item);
  if(proposed && !["other", "connection"].includes(proposed))return proposed;
  const visuallyClassified=visual?.legend_class;
  if(visuallyClassified && !["other", "connection"].includes(visuallyClassified))return visuallyClassified;
  return proposed && !["other", "connection"].includes(proposed) ? proposed : null;
}
function conflictQuestion(item) {
  if (item?.conflict?.question) return app.lang === "zh-CN" ? item.conflict.question_zh || item.conflict.question : item.conflict.question;
  if (item?.conflict?.type === "physical_instance_geometry") return t("conflictInstanceGeometryQuestion");
  const type = item?.conflict?.type || "", nodesText = conflictNodeNames(item);
  if (type === "unsupported_vector_route") return t("conflictVectorQuestion", {nodes:nodesText});
  if (type === "contextual_symbol_uncertainty") return t("questionComposite");
  if (type === "unsupported_endpoint_combination") return t("questionDirectConnection", {nodes:nodesText,type:displayValue(conflictProposedType(item))});
  if (type === "endpoint_role_uncertain") {
    const suggested = conflictSuggestedType(item);
    return suggested ? t("questionConfirmSuggested", {nodes:nodesText,type:displayValue(suggested)}) : t("questionLineMeaning", {nodes:nodesText,style:conflictLineStyle(item)});
  }
  if (["page_graph_uncertain_candidate","page_graph_uncertainty","visual_topology_disagreement","page_graph_missing"].includes(type)) return t("questionProposedConnection", {nodes:nodesText});
  if (type === "crossing_or_junction") return t("questionCrossing");
  if (type.includes("evidence") || type === "rejected_non_connectable_text") return t("questionEvidence", {nodes:nodesText});
  return t("questionGenericConflict", {nodes:nodesText});
}

function conflictEvidence(item) {
  const type=item?.conflict?.type || "";
  if(type==="contextual_symbol_uncertainty")return t("evidenceComposite");
  if(type==="unsupported_endpoint_combination")return t("evidenceUnsupported");
  if(type==="endpoint_role_uncertain"){const suggested=conflictSuggestedType(item);return suggested?t("evidenceSuggested",{type:displayValue(suggested)}):t("evidenceRole");}
  if(type==="page_graph_uncertain_candidate")return t("evidenceCandidate");
  if(type==="page_graph_uncertainty"||type==="visual_topology_disagreement")return t("evidenceTopology");
  if(type==="crossing_or_junction")return t("evidenceCrossing");
  if(type==="page_graph_missing")return t("evidenceMissing");
  if(type.includes("opc"))return t("evidenceOpc");
  if(type.includes("evidence")||type==="rejected_non_connectable_text")return t("evidenceObject");
  return item?.conflict?.reason || item?.conflict?.message || item?.conflict?.detail || type || t("graphConflict");
}

function relatedConflictCards(targetType, targetId) {
  const conflicts = relatedConflicts(targetType, targetId);
  if (!conflicts.length) return "";
  return `<section class="related-conflicts"><div class="related-conflicts-heading"><h4>${t("relatedConflicts")}</h4><span>${t("conflictCount",{count:conflicts.length})}</span></div>
    <p>${t("nodeApprovalDoesNotResolve")}</p>
    ${conflicts.map(([id,item])=>`<button class="conflict-card" data-review-conflict="${escapeHtml(id)}"><strong>${escapeHtml(conflictTitle(item))}</strong><span>${escapeHtml(conflictQuestion(item))}</span><em>${t("reviewConflict")} →</em></button>`).join("")}</section>`;
}

function bindRelatedConflictCards(container) {
  container.querySelectorAll("[data-review-conflict]").forEach(button => {
    button.onclick=()=>select("conflict",button.dataset.reviewConflict,{focus:true});
  });
}

function applyLanguage() {
  document.documentElement.lang = app.lang;
  document.title = t("title");
  document.querySelectorAll("[data-i18n]").forEach(node => { node.textContent = t(node.dataset.i18n); });
  document.querySelectorAll("[data-i18n-title]").forEach(node => { node.title = t(node.dataset.i18nTitle); });
  document.querySelectorAll("[data-i18n-aria-label]").forEach(node => { node.setAttribute("aria-label", t(node.dataset.i18nAriaLabel)); });
  document.querySelectorAll("[data-i18n-alt]").forEach(node => { node.alt = t(node.dataset.i18nAlt); });
  document.querySelectorAll("[data-i18n-placeholder]").forEach(node => { node.placeholder = t(node.dataset.i18nPlaceholder); });
  el("languageSwitch").value = app.lang;
}

function applyWorkspaceLayout(refit = true) {
  const workspace = document.querySelector(".workspace"), compare = document.querySelector(".compare-panel");
  compare.classList.toggle("stacked", app.layout === "stacked");
  workspace.classList.toggle("queue-hidden", !app.queueVisible);
  workspace.classList.toggle("inspector-hidden", !app.inspectorVisible);
  el("layoutMode").value = app.layout;
  el("workspaceView").value = app.workspaceView;
  compare.classList.toggle("data-mode", app.workspaceView !== "comparison");
  el("inventoryView").classList.toggle("hidden", app.workspaceView !== "inventory");
  el("evidenceView").classList.toggle("hidden", app.workspaceView !== "evidence");
  el("toggleQueue").textContent = t(app.queueVisible ? "hideQueue" : "showQueue");
  el("toggleInspector").textContent = t(app.inspectorVisible ? "hideInspector" : "showInspector");
  el("toggleQueue").setAttribute("aria-pressed", String(!app.queueVisible));
  el("toggleInspector").setAttribute("aria-pressed", String(!app.inspectorVisible));
  if (refit && app.state && app.workspaceView === "comparison") requestAnimationFrame(fitView);
}

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const data = await response.json();
  if (!response.ok) {
    if (response.status === 409 && data.state) { app.state = data.state; renderAll(); }
    const error = new Error(data.error || `HTTP ${response.status}`); error.data = data; throw error;
  }
  return data;
}

function toast(message, error = false) {
  const box = el("toast"); box.textContent = message; box.style.background = error ? "#991b1b" : "#101828";
  box.classList.add("show"); clearTimeout(toast.timer); toast.timer = setTimeout(() => box.classList.remove("show"), 3200);
}

function pageInfo() { return app.state.session.pages.find(p => p.page_index === app.page); }
function nodes() { return app.state.graph.nodes; }
function edges() { return app.state.graph.edges; }
function nodeById(id) { return nodes().find(n => n.id === id); }
function evidenceCandidates() { return app.state.inventory?.evidence || []; }
function evidenceById(id) { return evidenceCandidates().find(item => item.id === id); }
function reviewClass(status) { return status === "rejected" ? "rejected" : (status === "unreviewed" ? "" : "reviewed"); }

function setStageSize() {
  const page = pageInfo(); if (!page) return;
  for (const id of ["sourceStage", "inferenceStage"]) {
    const stage = el(id); stage.style.width = `${page.width * app.zoom}px`; stage.style.height = `${page.height * app.zoom}px`;
  }
  for (const id of ["sourceOverlay", "graphOverlay"]) el(id).setAttribute("viewBox", `0 0 ${page.width} ${page.height}`);
  el("zoomLabel").textContent = `${Math.round(app.zoom * 100)}%`;
  renderIssueHighlight();
}

function viewportFocus(viewport) {
  const stage = viewport.querySelector(".stage"), page = pageInfo();
  const stageLeft = stage?.offsetLeft || 0, stageTop = stage?.offsetTop || 0;
  return {
    x: Math.max(0, Math.min(page.width, (viewport.scrollLeft + viewport.clientWidth / 2 - stageLeft) / app.zoom)),
    y: Math.max(0, Math.min(page.height, (viewport.scrollTop + viewport.clientHeight / 2 - stageTop) / app.zoom))
  };
}

function restoreViewportFocus(viewport, focus) {
  const stage = viewport.querySelector(".stage");
  const stageLeft = stage?.offsetLeft || 0, stageTop = stage?.offsetTop || 0;
  viewport.scrollLeft = stageLeft + focus.x * app.zoom - viewport.clientWidth / 2;
  viewport.scrollTop = stageTop + focus.y * app.zoom - viewport.clientHeight / 2;
}

function mergeBounds(left, right) {
  if (!right) return left;
  if (!left) return {...right};
  const x=Math.min(left.x,right.x), y=Math.min(left.y,right.y);
  const x2=Math.max(left.x+left.w,right.x+right.w), y2=Math.max(left.y+left.h,right.y+right.h);
  return {x,y,w:x2-x,h:y2-y};
}

function validBounds(value) {
  if (!value || typeof value !== "object") return null;
  const x=Number(value.x), y=Number(value.y), w=Number(value.w ?? value.width), h=Number(value.h ?? value.height);
  return [x,y,w,h].every(Number.isFinite) ? {x,y,w:Math.max(1,w),h:Math.max(1,h)} : null;
}

function boundsForPoints(points) {
  const valid=(Array.isArray(points)?points:[]).map(point=>Array.isArray(point)?[Number(point[0]),Number(point[1])]:[Number(point?.x),Number(point?.y)]).filter(point=>point.every(Number.isFinite));
  if (!valid.length) return null;
  const xs=valid.map(point=>point[0]), ys=valid.map(point=>point[1]);
  return {x:Math.min(...xs),y:Math.min(...ys),w:Math.max(1,Math.max(...xs)-Math.min(...xs)),h:Math.max(1,Math.max(...ys)-Math.min(...ys))};
}

function conflictPageIndex(item) {
  const conflict=item?.conflict || {}, candidates=[];
  if (Number.isInteger(conflict.page_index)) candidates.push(conflict.page_index);
  conflictNodes(item).forEach(node=>{if(Number.isInteger(node.page_index))candidates.push(node.page_index);});
  conflictEdges(item).forEach(edge=>candidates.push(edgePage(edge)));
  (conflict.source_locations || []).forEach(location=>candidates.push(location.page_index));
  (conflict.page_indices || []).forEach(page=>{if(Number.isInteger(page))candidates.push(page);});
  const unique=[...new Set(candidates)];
  return unique.includes(app.page) ? app.page : unique[0] ?? null;
}

function conflictExactBounds(item, pageIndex) {
  const c=item?.conflict||{};
  const location=(c.source_locations||[]).find(location=>location.page_index===pageIndex);
  if(location)return validBounds(location.bbox_global);
  if(c.page_index!=null&&c.page_index!==pageIndex)return null;
  for(const key of ["bbox_global","bbox","candidate_bbox","source_bbox"]) {
    const box=validBounds(c[key]);if(box)return box;
  }
  const point=boundsForPoints([c.point,c.intersection,c.crossing_point]);if(point)return point;
  if(c.x!=null&&c.y!=null&&Number.isFinite(Number(c.x))&&Number.isFinite(Number(c.y)))return {x:Number(c.x),y:Number(c.y),w:1,h:1};
  for(const key of ["polyline_global","points","route_points"]) {
    const box=boundsForPoints(c[key]);if(box)return box;
  }
  return null;
}
function conflictBounds(item, pageIndex) {
  const exact=conflictExactBounds(item,pageIndex);if(exact)return exact;
  let bounds=null;
  conflictNodes(item).filter(node=>node.page_index===pageIndex).forEach(node=>{bounds=mergeBounds(bounds,validBounds(node.bbox_global));});
  conflictEdges(item).forEach(edge=>{bounds=mergeBounds(bounds,edgeBoundsOnPage(edge,pageIndex));});
  return bounds;
}
function paddedIssueBounds(bounds, zoom, page) {
  const pad=14/zoom, minSize=72/zoom;
  const w=Math.max(minSize,bounds.w+2*pad),h=Math.max(minSize,bounds.h+2*pad);
  const x=Math.max(0,bounds.x+bounds.w/2-w/2),y=Math.max(0,bounds.y+bounds.h/2-h/2);
  return {x,y,w:Math.min(w,page.width-x),h:Math.min(h,page.height-y)};
}
function renderIssueHighlight() {
  for(const id of ["sourceOverlay","graphOverlay"])el(id)?.querySelector(".issue-spotlight")?.remove();
  const active=app.selection&&["edge","conflict"].includes(app.selection.type);
  el("showIssue").disabled=!active;
  const status=el("issueLocationStatus"); status.classList.toggle("hidden",!active);
  if(!active||!app.state)return;
  const page=pageInfo(), region=selectionRegion(app.selection.type,app.selection.id,app.page);
  const saved=app.sourceFocus;
  const bounds=saved?.id===app.selection.id&&saved.page_index===app.page?saved.bbox_global:region?.bounds;
  if(!bounds){status.textContent=t(region?.page!==app.page?"offPageIssue":"pageIssue");return;}
  status.textContent=t("locatedIssue");
  const b=paddedIssueBounds(bounds,app.zoom,page), point=bounds.w<=2&&bounds.h<=2;
  const conflict=app.selection.type==="conflict"?app.state.reviews.conflicts[app.selection.id]:null;
  const label=t(point?"issuePoint":conflict&&!conflictExactBounds(conflict,app.page)?"relatedArea":"issueArea");
  for(const id of ["sourceOverlay","graphOverlay"]) {
    const group=svg("g",{class:"issue-spotlight","pointer-events":"none","data-target-id":app.selection.id});
    if(id==="sourceOverlay"&&page.rotation_deg)group.setAttribute("transform",`rotate(${-page.rotation_deg} ${page.width/2} ${page.height/2})`);
    group.append(svg("rect",{x:b.x,y:b.y,width:b.w,height:b.h,class:"issue-halo"}));
    group.append(svg("rect",{x:b.x,y:b.y,width:b.w,height:b.h,class:"issue-region"}));
    const edge=app.selection.type==="edge"?edges().find(edge=>edge.id===app.selection.id):null;
    const routes=edge?[edge]:conflict?conflictEdges(conflict):[];
    routes.filter(edge=>edgePage(edge)===app.page&&edge.polyline_global?.length>1).forEach(edge=>{
      const points=edge.polyline_global.map(p=>p.join(",")).join(" ");
      group.append(svg("polyline",{points,class:"issue-route-halo"}));
      group.append(svg("polyline",{points,class:"issue-route"}));
    });
    if(point){const x=bounds.x,y=bounds.y,r=12/app.zoom;group.append(svg("path",{d:`M ${x-r} ${y} H ${x+r} M ${x} ${y-r} V ${y+r}`,class:"issue-crosshair"}));}
    const font=12/app.zoom,labelW=(label.length*7+18)/app.zoom,labelH=23/app.zoom,labelY=Math.max(0,b.y-labelH);
    group.append(svg("rect",{x:b.x,y:labelY,width:labelW,height:labelH,rx:3/app.zoom,class:"issue-label-bg"}));
    const text=svg("text",{x:b.x+8/app.zoom,y:labelY+16/app.zoom,"font-size":font,class:"issue-label"});text.textContent=label;group.append(text);
    el(id).append(group);
  }
}

function focusBounds(bounds, {minimumZoom=.2, maximumZoom=1.5} = {}) {
  const page=pageInfo(), viewport=el("inferenceViewport"); if(!page||!bounds||!viewport.clientWidth||!viewport.clientHeight)return;
  const margin=Math.max(40,Math.min(100,Math.min(viewport.clientWidth,viewport.clientHeight)*.14));
  const regionWidth=Math.max(24,bounds.w), regionHeight=Math.max(24,bounds.h);
  const fitZoom=Math.max(.05,Math.min(maximumZoom,(viewport.clientWidth-2*margin)/regionWidth,(viewport.clientHeight-2*margin)/regionHeight));
  const nextZoom=Math.max(.05,Math.min(maximumZoom,fitZoom));
  const focus={x:Math.max(0,Math.min(page.width,bounds.x+bounds.w/2)),y:Math.max(0,Math.min(page.height,bounds.y+bounds.h/2))};
  app.syncing=true; app.zoom=nextZoom; setStageSize();
  for(const id of ["sourceViewport","inferenceViewport"])restoreViewportFocus(el(id),focus);
  requestAnimationFrame(()=>{app.syncing=false;});
}

function edgeBoundsOnPage(edge, pageIndex) {
  let bounds=edgePage(edge)===pageIndex?boundsForPoints(edgePoints(edge)):null;
  for(const nodeId of [edge.from_node,edge.to_node]) {
    const node=nodeById(nodeId);
    if(node?.page_index===pageIndex)bounds=mergeBounds(bounds,validBounds(node.bbox_global));
  }
  return bounds;
}

function selectionRegion(type, id, preferredPage=null) {
  if (type === "assembly") { const a = assemblyById(id); return a ? {page:a.page_index,bounds:validBounds(a.bbox_global),minimumZoom:.1,maximumZoom:1} : null; }
  if(type==="node") {
    const node=nodeById(id);
    return node?{page:node.page_index,bounds:validBounds(node.bbox_global),minimumZoom:.45,maximumZoom:1.5}:null;
  }
  if(type==="evidence") {
    const item=evidenceById(id);
    return item?{page:item.page_index,bounds:validBounds(item.bbox_global),minimumZoom:.45,maximumZoom:1.5}:null;
  }
  if(type==="edge") {
    const edge=edges().find(item=>item.id===id); if(!edge)return null;
    const endpointPages=[nodeById(edge.from_node)?.page_index,nodeById(edge.to_node)?.page_index].filter(Number.isInteger);
    const page=endpointPages.includes(preferredPage)?preferredPage:endpointPages.includes(app.page)?app.page:(endpointPages[0]??edgePage(edge));
    return {page,bounds:edgeBoundsOnPage(edge,page),minimumZoom:.2,maximumZoom:1.15};
  }
  if(type==="conflict") {
    const item=app.state.reviews.conflicts[id]; if(!item)return null;
    const page=Number.isInteger(preferredPage)?preferredPage:conflictPageIndex(item);
    return page==null?null:{page,bounds:conflictBounds(item,page),minimumZoom:.2,maximumZoom:1.15};
  }
  return null;
}

function focusSelectionRegion(type, id, preferredPage=null) {
  const region=selectionRegion(type,id,preferredPage); if(!region||region.page==null)return;
  if(!region.bounds){if(region.page!==app.page)changePage(region.page,true);fitView();return;}
  if(region.page!==app.page)changePage(region.page,true);
  requestAnimationFrame(()=>requestAnimationFrame(()=>focusBounds(region.bounds,region)));
}

function changeZoom(nextZoom) {
  const active = el(app.activeViewport) || el("inferenceViewport");
  const focus = viewportFocus(active), clamped = Math.max(.05, Math.min(2.5, nextZoom));
  if (clamped === app.zoom) return;
  app.syncing = true;
  app.zoom = clamped;
  setStageSize();
  for (const id of ["sourceViewport", "inferenceViewport"]) restoreViewportFocus(el(id), focus);
  requestAnimationFrame(() => { app.syncing = false; });
}

function syncScroll(from, to) {
  if (app.syncing) return; app.syncing = true;
  const fx = from.scrollWidth - from.clientWidth, fy = from.scrollHeight - from.clientHeight;
  const tx = to.scrollWidth - to.clientWidth, ty = to.scrollHeight - to.clientHeight;
  to.scrollLeft = fx > 0 ? (from.scrollLeft / fx) * tx : 0;
  to.scrollTop = fy > 0 ? (from.scrollTop / fy) * ty : 0;
  requestAnimationFrame(() => { app.syncing = false; });
}

function svg(tag, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}

function edgePage(edge) { return nodeById(edge.from_node)?.page_index ?? nodeById(edge.to_node)?.page_index ?? 0; }
function edgePoints(edge) {
  if (edge.polyline_global?.length) return edge.polyline_global;
  const a = nodeById(edge.from_node), b = nodeById(edge.to_node); if (!a || !b) return [];
  return [[a.bbox_global.x + a.bbox_global.w / 2, a.bbox_global.y + a.bbox_global.h / 2], [b.bbox_global.x + b.bbox_global.w / 2, b.bbox_global.y + b.bbox_global.h / 2]];
}

function candidateNodesForEdge(edge) {
  const ids = [edge.from_node, edge.to_node];
  Object.values(app.state.reviews.conflicts).forEach(item => {
    const related = (item.candidates?.edges || []).some(candidate => candidate.id === edge.id);
    if (related) (item.candidates?.nodes || []).forEach(node => ids.push(node.id));
  });
  return [...new Set(ids)].map(nodeById).filter(node => node && app.state.reviews.nodes[node.id] !== "rejected");
}

function assemblyById(id) { return (app.state?.graph?.assemblies || []).find(a => a.id === id); }

function highlightedCandidateIds() {
  if (app.selection?.type === "assembly") return new Set(assemblyById(app.selection.id)?.member_node_ids || []);
  if (app.selection?.type === "edge") return new Set(candidateNodesForEdge(edges().find(edge => edge.id === app.selection.id) || {}).map(node => node.id));
  if (app.selection?.type === "conflict") {
    const item = app.state.reviews.conflicts[app.selection.id];
    return new Set((item?.candidates?.nodes || []).map(node => node.id));
  }
  return new Set();
}

function highlightedCandidateEdgeIds() {
  if (app.selection?.type !== "conflict") return new Set();
  const item = app.state.reviews.conflicts[app.selection.id];
  return new Set((item?.candidates?.edges || []).map(edge => edge.id));
}

function nodeAtPoint(point, edge, endpointField) {
  const opposite = endpointField === "from_node" ? edge.to_node : edge.from_node;
  return nodes().find(node => {
    if (node.page_index !== app.page || node.id === opposite || app.state.reviews.nodes[node.id] === "rejected") return false;
    const b = node.bbox_global;
    return point[0] >= b.x && point[0] <= b.x + b.w && point[1] >= b.y && point[1] <= b.y + b.h;
  });
}

function markDropTarget(overlay, nodeId) {
  overlay.querySelectorAll(".graph-node.drop-target").forEach(node => node.classList.remove("drop-target"));
  if (nodeId) overlay.querySelector(`.graph-node[data-node-id="${CSS.escape(nodeId)}"]`)?.classList.add("drop-target");
}

function renderOverlay() {
  const overlay = el("graphOverlay"); overlay.replaceChildren();
  if (app.selection?.type === "assembly") {
    const a = assemblyById(app.selection.id);
    if (a?.page_index === app.page) { const b = a.bbox_global; overlay.append(svg("rect", {x:b.x,y:b.y,width:b.w,height:b.h,class:"assembly-highlight"})); }
  }
  const highlighted = highlightedCandidateIds(), highlightedEdges = highlightedCandidateEdgeIds(), conflictFocused = app.selection?.type === "conflict"; let selectedEdgeParts = null;
  edges().filter(edge => edgePage(edge) === app.page).forEach(edge => {
    const points = edgePoints(edge); if (!points.length) return;
    const visualStyle = ["dashed", "dotted", "dash_dot"].includes((edge.attributes || {}).visual_style) ? (edge.attributes || {}).visual_style : "solid";
    const focusClass = conflictFocused ? (highlightedEdges.has(edge.id) ? "conflict-focus" : "conflict-muted") : "";
    const poly = svg("polyline", { points: points.map(p => p.join(",")).join(" "), class: `graph-edge style-${visualStyle} ${reviewClass(app.state.reviews.edges[edge.id])} ${app.selection?.id === edge.id ? "selected" : ""} ${focusClass}` });
    poly.addEventListener("click", event => { event.stopPropagation(); select("edge", edge.id); }); overlay.append(poly);
    if (app.selection?.type === "edge" && app.selection.id === edge.id) selectedEdgeParts = {edge, points, poly};
  });
  if (conflictFocused && !highlightedEdges.size) {
    const item=app.state.reviews.conflicts[app.selection.id], candidates=conflictNodes(item).filter(node=>node.page_index===app.page);
    if (RELATION_CONFLICTS.has(item?.conflict?.type) && candidates.length >= 2) {
      const centers=candidates.slice(0,2).map(node=>[node.bbox_global.x+node.bbox_global.w/2,node.bbox_global.y+node.bbox_global.h/2]);
      overlay.append(svg("polyline",{points:centers.map(point=>point.join(",")).join(" "),class:"conflict-proposal"}));
    }
  }
  nodes().filter(node => node.page_index === app.page).forEach(node => {
    const b = node.bbox_global;
    const focusClass = conflictFocused ? (highlighted.has(node.id) ? "conflict-focus" : "conflict-muted") : "";
    const rect = svg("rect", { x: b.x, y: b.y, width: b.w, height: b.h, "data-node-id": node.id, class: `graph-node ${reviewClass(app.state.reviews.nodes[node.id])} ${highlighted.has(node.id) ? "suggested" : ""} ${app.selection?.id === node.id ? "selected" : ""} ${focusClass}` });
    rect.addEventListener("click", event => { event.stopPropagation(); select("node", node.id); });
    overlay.append(rect);
    const label = svg("text", { x: b.x + 3, y: Math.max(12, b.y - 4), class: "graph-label", "font-size": Math.max(11, Math.min(24, b.h * .18)) }); label.textContent = displayNodeLabel(node) || node.id; overlay.append(label);
    if (app.selection?.type === "node" && app.selection.id === node.id) {
      enableNodeDrag(rect, node, overlay);
      addNodeHandle(overlay, node, rect);
    }
  });
  if (app.selection?.type === "evidence") {
    const candidate = evidenceById(app.selection.id);
    if (candidate?.page_index === app.page) {
      const b = candidate.bbox_global;
      overlay.append(svg("rect", {x:b.x, y:b.y, width:b.w, height:b.h, class:"evidence-candidate selected"}));
    }
  }
  if (selectedEdgeParts) addVertexHandles(overlay, selectedEdgeParts.edge, selectedEdgeParts.points, selectedEdgeParts.poly);
  overlay.onclick = () => { if (!app.drawMode) select(null, null); };
  renderSourceHighlight();
}

function pointerPoint(event, overlay) {
  const rect = overlay.getBoundingClientRect(), page = pageInfo();
  return [Math.max(0, Math.min(page.width, (event.clientX - rect.left) * page.width / rect.width)), Math.max(0, Math.min(page.height, (event.clientY - rect.top) * page.height / rect.height))];
}

function enableNodeDrag(rect, node, overlay) {
  rect.addEventListener("pointerdown", event => {
    event.stopPropagation(); rect.setPointerCapture(event.pointerId);
    const start = pointerPoint(event, overlay), original = {...node.bbox_global}; let moved = false;
    rect.onpointermove = move => {
      const point = pointerPoint(move, overlay), dx = point[0] - start[0], dy = point[1] - start[1];
      moved = moved || Math.abs(dx) + Math.abs(dy) > 2;
      node.bbox_global.x = Math.round(original.x + dx); node.bbox_global.y = Math.round(original.y + dy);
      rect.setAttribute("x", node.bbox_global.x); rect.setAttribute("y", node.bbox_global.y);
      renderSourceHighlight();
    };
    rect.onpointerup = async () => { rect.onpointermove = null; rect.onpointerup = null; if (moved) await submit("node", node.id, "modify", { bbox_global: roundBox(node.bbox_global) }); };
  });
}

function addNodeHandle(overlay, node, nodeRect) {
  const b = node.bbox_global, size = Math.max(10, 12 / app.zoom);
  const handle = svg("rect", { x: b.x + b.w - size / 2, y: b.y + b.h - size / 2, width: size, height: size, class: "handle" });
  handle.addEventListener("pointerdown", event => {
    event.stopPropagation(); handle.setPointerCapture(event.pointerId); const start = pointerPoint(event, overlay), original = {...b};
    handle.onpointermove = move => { const p = pointerPoint(move, overlay); node.bbox_global.w = Math.max(2, original.w + p[0] - start[0]); node.bbox_global.h = Math.max(2, original.h + p[1] - start[1]); nodeRect.setAttribute("width", node.bbox_global.w); nodeRect.setAttribute("height", node.bbox_global.h); handle.setAttribute("x", node.bbox_global.x + node.bbox_global.w - size / 2); handle.setAttribute("y", node.bbox_global.y + node.bbox_global.h - size / 2); renderSourceHighlight(); };
    handle.onpointerup = async () => { handle.onpointermove = null; await submit("node", node.id, "modify", { bbox_global: roundBox(node.bbox_global) }); };
  }); overlay.append(handle);
}

function addVertexHandles(overlay, edge, points, polyline) {
  points.forEach((point, index) => {
    const isStart = index === 0, isEnd = index === points.length - 1;
    const endpointField = isStart ? "from_node" : isEnd ? "to_node" : null;
    const circle = svg("circle", { cx: point[0], cy: point[1], r: Math.max(6, 8 / app.zoom), class: `vertex ${endpointField ? "endpoint" : ""}` });
    circle.addEventListener("pointerdown", event => {
      event.stopPropagation(); circle.setPointerCapture(event.pointerId);
      let target = null;
      circle.onpointermove = move => { const p = pointerPoint(move, overlay); points[index] = p; edge.polyline_global = points; circle.setAttribute("cx", p[0]); circle.setAttribute("cy", p[1]); polyline.setAttribute("points", points.map(item => item.join(",")).join(" ")); target = endpointField ? nodeAtPoint(p, edge, endpointField) : null; markDropTarget(overlay, target?.id); renderSourceHighlight(); };
      circle.onpointerup = async () => { circle.onpointermove = null; markDropTarget(overlay, null); if (endpointField && target) await updateEdgeEndpoint(edge.id, endpointField, target.id); else await submit("edge", edge.id, "modify", { polyline_global: points.map(p => p.map(Math.round)) }); };
    }); overlay.append(circle);
  });
}

function roundBox(b) { return { x: Math.round(b.x), y: Math.round(b.y), w: Math.max(1, Math.round(b.w)), h: Math.max(1, Math.round(b.h)) }; }

function renderSourceHighlight() {
  const overlay = el("sourceOverlay"); overlay.replaceChildren(); renderIssueHighlight(); if (!app.selection) return;
  const page = pageInfo(), group = svg("g", { transform: page.rotation_deg ? `rotate(${-page.rotation_deg} ${page.width / 2} ${page.height / 2})` : "" });
  if (app.selection.type === "assembly") {
    const a = assemblyById(app.selection.id); if (!a || a.page_index !== app.page) return;
    const b = a.bbox_global; group.append(svg("rect", {x:b.x,y:b.y,width:b.w,height:b.h,class:"assembly-highlight"}));
    a.member_node_ids.map(nodeById).filter(Boolean).forEach(n => { const b = n.bbox_global; group.append(svg("rect", {x:b.x,y:b.y,width:b.w,height:b.h,class:"source-highlight"})); });
  } else if (app.selection.type === "node") {
    const node = nodeById(app.selection.id); if (!node || node.page_index !== app.page) return;
    const b = node.bbox_global; group.append(svg("rect", { x: b.x, y: b.y, width: b.w, height: b.h, class: "source-highlight" }));
  } else if (app.selection.type === "evidence") {
    const candidate = evidenceById(app.selection.id); if (!candidate || candidate.page_index !== app.page) return;
    const b = candidate.bbox_global; group.append(svg("rect", {x:b.x, y:b.y, width:b.w, height:b.h, class:"source-highlight evidence"}));
  } else if (app.selection.type === "edge") {
    const edge = edges().find(item => item.id === app.selection.id); if (!edge || edgePage(edge) !== app.page) return;
    const points = edgePoints(edge); if (!points.length) return;
    const xs = points.map(point => point[0]), ys = points.map(point => point[1]), padding = 12;
    group.append(svg("rect", { x:Math.min(...xs)-padding, y:Math.min(...ys)-padding, width:Math.max(2,Math.max(...xs)-Math.min(...xs)+2*padding), height:Math.max(2,Math.max(...ys)-Math.min(...ys)+2*padding), class:"source-route-highlight" }));
    [points[0], points[points.length - 1]].forEach(point => group.append(svg("circle", { cx:point[0], cy:point[1], r:8, class:"source-route-endpoint" })));
  } else if (app.selection.type === "conflict") {
    const item=app.state.reviews.conflicts[app.selection.id]; if(!item)return;
    (item.conflict.source_locations||[]).filter(location=>location.page_index===app.page).forEach(location=>{const b=location.bbox_global;if(b)group.append(svg("rect",{x:b.x,y:b.y,width:b.w,height:b.h,class:"source-highlight conflict"}));});
    conflictNodes(item).filter(node=>node.page_index===app.page).forEach(node=>{const b=node.bbox_global;group.append(svg("rect",{x:b.x,y:b.y,width:b.w,height:b.h,class:"source-highlight conflict"}));});
    conflictEdges(item).filter(edge=>edgePage(edge)===app.page).forEach(edge=>{const points=edgePoints(edge);if(points.length)group.append(svg("polyline",{points:points.map(point=>point.join(",")).join(" "),class:"source-conflict-route"}));});
  }
  overlay.append(group);
  renderIssueHighlight();
}

function queueRows() {
  const scope=el("queueFilter").value;
  if(scope==="findings")return app.state.findings || [];
  const rows=[...app.state.queue];
  Object.entries(app.state.reviews.conflicts).filter(([,item])=>!item.decision_target).forEach(([id,item])=>rows.push({target_type:"conflict",target_id:id,page_index:conflictPageIndex(item),label:conflictTitle(item),status:item.status,reasons:[],conflict_ids:[id]}));
  return scope==="all"?rows:rows.filter(row=>row.status==="unreviewed");
}
function reviewNodeName(node) {
  if(!node)return t("unlabelled");
  node=nodeById(node.id)||node;
  if(node.label && node.label!=="unlabelled")return displayNodeLabel(node);
  return t("untaggedObject",{kind:node.attributes?.valve_type?displayValue("valve"):nodeMeaning(node).text||displayValue(node.kind)});
}
function queueRowText(row) {
  const item=row.target_type==="conflict"?app.state.reviews.conflicts[row.target_id]:null;
  const edge=row.target_type==="edge"?edges().find(edge=>edge.id===row.target_id):null;
  const label=item?conflictTitle(item):edge?[edge.from_node,edge.to_node].map(id=>reviewNodeName(nodeById(id))).join(" → "):row.label;
  const related=(row.conflict_ids||[]).map(id=>app.state.reviews.conflicts[id]).filter(Boolean);
  const reason=item?conflictQuestion(item):related.length?[...new Set(related.map(conflictQuestion))].join(" · "):(row.reasons||[]).map(translateReason).join(" · ");
  return {label,reason};
}
function visibleQueueRows() {
  const type=el("queueType").value, page=el("queuePage").value, query=el("queueSearch").value.trim().toLowerCase();
  return queueRows().filter(row=>{
    if(type!=="all"&&row.target_type!==type)return false;
    if(el("queueFilter").value==="findings"&&app.queueCategory!=="all"&&row.category!==app.queueCategory)return false;
    const pages=new Set([row.page_index]);
    for(const id of row.conflict_ids||[]) {
      const c=app.state.reviews.conflicts[id]?.conflict||{};
      (c.source_locations||[]).forEach(location=>pages.add(location.page_index));
      (c.page_indices||[]).forEach(index=>pages.add(index));
    }
    if(page!=="all"&&!pages.has(Number(page)))return false;
    const {label,reason}=queueRowText(row);
    return !query||[label,reason,row.target_id,...(row.conflict_ids||[]),...(row.reasons||[])].join(" ").toLowerCase().includes(query);
  });
}
function openQueueRow(row) {
  app.queueSelection=`${row.target_type}:${row.target_id}`;
  app.inspectorVisible=true; app.workspaceView="comparison"; applyWorkspaceLayout(false);
  const preferred=el("queuePage").value;
  select(row.target_type,row.target_id,{focus:true,preferredPage:preferred==="all"?row.page_index:Number(preferred)});
}
function nextQueueFinding() {
  const rows=visibleQueueRows(); if(!rows.length)return;
  const index=rows.findIndex(row=>`${row.target_type}:${row.target_id}`===app.queueSelection);
  openQueueRow(rows[(index+1)%rows.length]);
  el("queue").querySelector(".selected")?.scrollIntoView({block:"nearest"});
}
function renderQueue() {
  const findings=app.state.findings||[], focused=el("queueFilter").value==="findings";
  el("findingBadge").textContent=t("findingCount",{count:findings.length});
  el("queueHelp").textContent=t(focused?"decisionHelp":"signoffHelp");
  const categories=el("queueCategories");categories.replaceChildren();categories.classList.toggle("hidden",!focused);
  for(const [key,label] of [["all","allFindings"],["identifiers","identifiers"],["symbols","symbols"],["connections","connections"],["other","otherFindings"]]) {
    const count=key==="all"?findings.length:findings.filter(row=>row.category===key).length;
    if(!count&&key==="other")continue;
    const button=document.createElement("button");button.textContent=`${t(label)} ${count}`;button.setAttribute("aria-pressed",String(app.queueCategory===key));
    button.onclick=()=>{app.queueCategory=key;renderQueue();};categories.append(button);
  }
  const visible=visibleQueueRows(), box=el("queue");box.replaceChildren();
  el("queueCount").textContent=t("showingFindings",{count:visible.length});
  el("nextFinding").disabled=!visible.length;
  visible.forEach(row=>{
    const button=document.createElement("button");button.className=`queue-item ${app.selection?.id===row.target_id&&app.selection?.type===row.target_type?"selected":""}`;
    button.dataset.targetId=row.target_id;
    const typeLabel=t(row.category?(row.category==="other"?"otherFindings":row.category):row.target_type==="node"?"entity":row.target_type==="edge"?"connection":row.target_type);
    const item=row.target_type==="conflict"?app.state.reviews.conflicts[row.target_id]:null;
    const pages=[...new Set([row.page_index,...(item?.conflict?.source_locations||[]).map(location=>location.page_index)].filter(Number.isInteger))].sort((a,b)=>a-b);
    const pageLabel=pages.length?` · ${pages.map(page=>t("pageShort",{page:page+1})).join(", ")}`:"";
    const {label,reason}=queueRowText(row);
    button.innerHTML=`<span class="kind"><i class="status-dot ${row.status}"></i>${escapeHtml(typeLabel)}${escapeHtml(pageLabel)}</span><span class="label">${escapeHtml(label)}</span><span class="reason">${escapeHtml(reason)}</span>`;
    button.onclick=()=>openQueueRow(row);box.append(button);
  });
  if(!visible.length){const empty=document.createElement("div");empty.className="inspector empty";empty.textContent=t(focused?"noFindings":"noMatches");box.append(empty);}
}

function setWorkspaceView(view) {
  app.workspaceView = view;
  storeSetting("diagex.review.workspaceView", view);
  applyWorkspaceLayout(false);
  renderDataViews();
  if (view === "comparison") requestAnimationFrame(fitView);
}

function nodeSubtype(node) {
  const attrs = node.attributes || {};
  return attrs.valve_type || attrs.equipment_class || attrs.instrument_function || attrs.instrument_class || "";
}

function duplicateNodeIds() {
  const groups = new Map();
  nodes().forEach(node => {
    if (app.state.reviews.nodes[node.id] === "rejected") return;
    const key = `${node.page_index}:${String(node.label || "").toUpperCase().replace(/[^A-Z0-9\u4e00-\u9fff]/g, "")}`;
    if (!groups.has(key)) groups.set(key, []); groups.get(key).push(node.id);
  });
  return new Set([...groups.values()].filter(ids => ids.length > 1).flat());
}

function coverageCard(label, value, warning = false) {
  return `<article class="coverage-card ${warning ? "warning" : ""}"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></article>`;
}

function renderInventory() {
  const inventory = app.state.inventory || {counts:{}, degrees:{}};
  const counts = inventory.counts || {};
  el("inventorySummary").innerHTML = [
    coverageCard(t("recognizedNodes"), counts.nodes || 0),
    coverageCard(t("reviewedNodes"), `${counts.reviewed_nodes || 0} / ${counts.nodes || 0}`),
    coverageCard(t("isolatedNodes"), counts.isolated_nodes || 0, (counts.isolated_nodes || 0) > 0),
    coverageCard(t("unresolvedTags"), counts.unresolved_tag_candidates || 0, (counts.unresolved_tag_candidates || 0) > 0)
  ].join("");
  const search = el("inventorySearch").value.trim().toLowerCase();
  const pageFilter = el("inventoryPage").value, kind = el("inventoryKind").value, statusFilter = el("inventoryStatus").value;
  const duplicates = duplicateNodeIds();
  const rows = nodes().filter(node => {
    const status = app.state.reviews.nodes[node.id] || "unreviewed", attrs = node.attributes || {};
    const degree = inventory.degrees?.[node.id] || 0;
    if (pageFilter !== "all" && node.page_index !== Number(pageFilter)) return false;
    if (kind !== "all" && node.kind !== kind) return false;
    if (statusFilter === "unreviewed" && status !== "unreviewed") return false;
    if (statusFilter === "isolated" && degree !== 0) return false;
    if (statusFilter === "duplicate" && !duplicates.has(node.id)) return false;
    if (statusFilter === "unclassified" && !(node.kind === "equipment" && !attrs.equipment_class && !attrs.valve_type)) return false;
    const haystack = [node.id,node.label,node.source_quote,node.kind,nodeSubtype(node),JSON.stringify(attrs)].join(" ").toLowerCase();
    return !search || haystack.includes(search);
  }).sort((a,b) => a.page_index-b.page_index || a.bbox_global.y-b.bbox_global.y || a.bbox_global.x-b.bbox_global.x);
  el("inventoryRows").innerHTML = rows.map(node => {
    const degree = inventory.degrees?.[node.id] || 0, status = app.state.reviews.nodes[node.id] || "unreviewed";
    const issueBadges = [degree === 0 ? t("isolated") : "", duplicates.has(node.id) ? t("duplicateTag") : ""].filter(Boolean).map(value => `<span class="issue-chip">${escapeHtml(value)}</span>`).join(" ");
    return `<tr class="${status === "rejected" ? "rejected-row" : ""}"><td>${node.page_index+1}</td><td><strong>${escapeHtml(displayNodeLabel(node) || node.id)}</strong><small>${escapeHtml(node.id)}</small>${issueBadges}</td><td>${escapeHtml(displayValue(node.kind))}</td><td>${escapeHtml(displayValue(nodeSubtype(node)))}</td><td>${degree}</td><td>${escapeHtml(node.source_quote || "—")}</td><td>${escapeHtml(displayValue(status))}</td><td><button data-show-node="${escapeHtml(node.id)}">${t("view")}</button></td></tr>`;
  }).join("") || `<tr><td colspan="8" class="empty-cell">${t("noMatches")}</td></tr>`;
  el("inventoryRows").querySelectorAll("[data-show-node]").forEach(button => { button.onclick = () => { const node=nodeById(button.dataset.showNode); if(!node)return; setWorkspaceView("comparison"); select("node",node.id,{focus:true,preferredPage:node.page_index}); }; });
}

function renderEvidence() {
  const inventory = app.state.inventory || {counts:{}, evidence:[]}, counts = inventory.counts || {};
  el("evidenceSummary").innerHTML = evidenceCandidates().length ? [
    coverageCard(t("unresolvedTags"), counts.unresolved_tag_candidates || 0, (counts.unresolved_tag_candidates || 0) > 0),
    coverageCard(t("candidateType"), counts.blocking_tag_candidates || 0),
    coverageCard(t("pageCoverage"), `${(inventory.pages || []).filter(page => page.page_review?.status === "approved" || page.page_review?.status === "waived").length} / ${(inventory.pages || []).length}`)
  ].join("") : coverageCard(t("noEvidenceArtifacts"), "—", true);
  const search=el("evidenceSearch").value.trim().toLowerCase(), pageFilter=el("evidencePage").value, status=el("evidenceStatus").value, kind=el("evidenceKind").value;
  const rows=evidenceCandidates().filter(item => (pageFilter==="all" || item.page_index===Number(pageFilter)) && (status==="all" || (status==="blocking" ? item.blocking && item.review.status==="unreviewed" : item.review.status===status)) && (kind==="all" || item.candidate_kind===kind) && (!search || [item.text,item.normalised_text,item.candidate_kind,item.review.disposition].join(" ").toLowerCase().includes(search)));
  el("evidenceRows").innerHTML = rows.map(item => {
    const nearby=(item.nearby_node_ids || []).map(id=>displayNodeLabel(nodeById(id)) || id).slice(0,3).join(", ");
    const disposition=item.review.status === "linked" ? `${t("linked")}: ${displayNodeLabel(nodeById(item.review.linked_node_id)) || item.review.linked_node_id}` : item.review.status === "dismissed" ? `${t("dismissed")}: ${displayValue(item.review.disposition)}` : displayValue(item.review.status);
    return `<tr><td>${item.page_index+1}</td><td><strong>${escapeHtml(item.text)}</strong><small>${item.blocking?t("blockingCandidate"):t("nonBlockingCandidate")}</small></td><td>${escapeHtml(displayValue(item.candidate_kind))}</td><td>${escapeHtml(disposition)}</td><td>${escapeHtml(nearby || "—")}</td><td><button data-review-evidence="${escapeHtml(item.id)}">${t("view")}</button></td></tr>`;
  }).join("") || `<tr><td colspan="6" class="empty-cell">${t("noMatches")}</td></tr>`;
  el("evidenceRows").querySelectorAll("[data-review-evidence]").forEach(button => { button.onclick=()=>{const item=evidenceById(button.dataset.reviewEvidence);if(!item)return;setWorkspaceView("comparison");select("evidence",item.id,{focus:true,preferredPage:item.page_index});}; });
}

function renderDataViews() { if (!app.state) return; renderInventory(); renderEvidence(); }

function escapeHtml(value) { const div = document.createElement("div"); div.textContent = String(value ?? ""); return div.innerHTML; }

function inputField(label, id, value, type = "text") { return `<label class="field">${label}<input id="${id}" type="${type}" value="${escapeHtml(value)}"></label>`; }
function selectField(label, id, value, choices) { return `<label class="field">${label}<select id="${id}">${choices.map(c => `<option value="${escapeHtml(c)}" ${c === value ? "selected" : ""}>${escapeHtml(displayValue(c))}</option>`).join("")}</select></label>`; }

function candidateNodeCard(node) {
  return `<article class="candidate-card" draggable="true" data-candidate-node="${escapeHtml(node.id)}">
    <div><strong>${escapeHtml(displayNodeLabel(node) || node.id)}</strong><small>${escapeHtml(displayValue(node.kind))} · ${escapeHtml(node.id)}${node.page_index == null ? "" : ` · ${escapeHtml(t("pageShort", {page: node.page_index + 1}))}`}</small></div>
    <div class="candidate-actions"><button data-view-node="${escapeHtml(node.id)}">${t("view")}</button><button data-use-endpoint="from_node" data-node-id="${escapeHtml(node.id)}">${t("useAsFrom")}</button><button data-use-endpoint="to_node" data-node-id="${escapeHtml(node.id)}">${t("useAsTo")}</button></div>
  </article>`;
}

function endpointEditorHtml(edge, candidates) {
  const endpoint = id => displayNodeLabel(nodeById(id)) || id;
  return `<section class="candidate-section"><h4>${t("candidates")}</h4><p class="candidate-hint">${t("dragCandidateHint")}</p>
    <div class="endpoint-slots">
      <div class="endpoint-drop" data-endpoint="from_node"><span>${t("from")}</span><strong>${escapeHtml(endpoint(edge.from_node))}</strong></div>
      <div class="endpoint-drop" data-endpoint="to_node"><span>${t("to")}</span><strong>${escapeHtml(endpoint(edge.to_node))}</strong></div>
    </div>
    <div class="candidate-list">${candidates.length ? candidates.map(candidateNodeCard).join("") : `<p class="candidate-hint">${t("noCandidates")}</p>`}</div>
  </section>`;
}

function bindEndpointEditor(container, edge, candidates) {
  const ids = new Set(candidates.map(node => node.id));
  container.querySelectorAll("[data-candidate-node]").forEach(card => {
    card.addEventListener("dragstart", event => {
      event.dataTransfer.effectAllowed = "link";
      event.dataTransfer.setData("text/plain", card.dataset.candidateNode);
    });
    card.addEventListener("pointerdown", event => {
      if (event.target.closest("button")) return;
      event.preventDefault(); card.setPointerCapture(event.pointerId); card.classList.add("dragging");
      let zone = null;
      const clearZone = () => { if (zone) zone.classList.remove("drag-over"); zone = null; };
      card.onpointermove = move => {
        const next = document.elementFromPoint(move.clientX, move.clientY)?.closest(".endpoint-drop");
        if (next === zone) return; clearZone();
        if (next && container.contains(next)) { zone = next; zone.classList.add("drag-over"); }
      };
      card.onpointerup = () => {
        card.onpointermove = null; card.onpointerup = null; card.classList.remove("dragging");
        const destination = zone; clearZone();
        if (destination) updateEdgeEndpoint(edge.id, destination.dataset.endpoint, card.dataset.candidateNode);
      };
      card.onpointercancel = () => { card.onpointermove = null; card.onpointerup = null; card.classList.remove("dragging"); clearZone(); };
    });
  });
  container.querySelectorAll(".endpoint-drop").forEach(zone => {
    zone.addEventListener("dragover", event => { event.preventDefault(); zone.classList.add("drag-over"); event.dataTransfer.dropEffect = "link"; });
    zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
    zone.addEventListener("drop", event => {
      event.preventDefault(); zone.classList.remove("drag-over");
      const nodeId = event.dataTransfer.getData("text/plain");
      if (ids.has(nodeId)) updateEdgeEndpoint(edge.id, zone.dataset.endpoint, nodeId);
    });
  });
  container.querySelectorAll("[data-use-endpoint]").forEach(button => {
    button.onclick = () => updateEdgeEndpoint(edge.id, button.dataset.useEndpoint, button.dataset.nodeId);
  });
  container.querySelectorAll("[data-view-node]").forEach(button => {
    button.onclick = () => {
      const node = nodeById(button.dataset.viewNode); if (!node) return;
      select("node",node.id,{focus:true,preferredPage:node.page_index});
    };
  });
}

async function updateEdgeEndpoint(edgeId, field, nodeId) {
  const edge = edges().find(item => item.id === edgeId); if (!edge || !["from_node", "to_node"].includes(field)) return;
  await submit("edge", edgeId, "modify", {[field]: nodeId}, t("selectedCandidate", {value: displayNodeLabel(nodeById(nodeId)) || nodeId}));
}

async function applyConflictChoice(conflictId, targetType, targetId, operation, after, reason) {
  await submit(targetType, targetId, operation, after, reason);
  if (unresolvedConflict(app.state.reviews.conflicts[conflictId])) await submit("conflict", conflictId, "resolve", {}, reason);
}

async function applyConflictEdgeType(conflictId, edgeId, lineType) {
  const reason=t("selectedCandidate",{value:displayValue(lineType)});
  await applyConflictChoice(conflictId, "edge", edgeId, "modify", {line_type:lineType}, reason);
}

async function rejectConflictEdge(conflictId, edgeId) {
  await applyConflictChoice(conflictId, "edge", edgeId, "reject", {}, t("rejectConnection"));
}

async function addConflictRelation(conflictId, lineType) {
  const item=app.state.reviews.conflicts[conflictId], candidates=conflictNodes(item); if(candidates.length<2)return;
  const [fromNode,toNode]=candidates, center=node=>[Math.round(node.bbox_global.x+node.bbox_global.w/2),Math.round(node.bbox_global.y+node.bbox_global.h/2)];
  const reason=t("selectedCandidate",{value:displayValue(lineType)});
  await submit("edge","","add",{from_node:fromNode.id,to_node:toNode.id,line_type:lineType,polyline_global:[center(fromNode),center(toNode)],cross_sheet:fromNode.page_index!==toNode.page_index,confidence:"high",source_annotation_ids:[],attributes:{review_added:true,human_resolution:true,source_conflict_id:conflictId}},reason);
  await submit("conflict",conflictId,"resolve",{},reason);
}

async function confirmNoConnection(conflictId) {
  await submit("conflict",conflictId,"resolve",{},t("confirmedNoConnection"));
}

async function applyContextualMerge(conflictId, conflict, actuation) {
  const primary=nodeById(conflict.primary_node_id), absorbed=new Set(conflict.absorbed_node_ids || []);
  if(!primary || !absorbed.size)return;
  const attrs={...(primary.attributes || {}),actuation};
  if(conflict.suggested_valve_type)attrs.valve_type=conflict.suggested_valve_type;
  const reason=t("selectedCandidate",{value:displayValue(actuation)});
  await submit("node",primary.id,"modify",{kind:"equipment",attributes:attrs},reason);
  for(const nodeId of absorbed)if(nodeById(nodeId))await submit("node",nodeId,"reject",{},reason);
  for(const edge of edges().filter(edge=>absorbed.has(edge.from_node)||absorbed.has(edge.to_node))){
    if(app.state.reviews.edges[edge.id]!=="rejected")await submit("edge",edge.id,"reject",{},reason);
  }
  await submit("conflict",conflictId,"resolve",{},reason);
  select("node",primary.id,{focus:true,preferredPage:primary.page_index});
}

const HUMAN_LINE_TYPES=["process","signal_electric","signal_pneumatic","instrument_capillary","electrical_power"];
function lineTypeDecisionButtons(conflictId, edgeId=null, suggestedType=null) {
  return `<div class="decision-actions">${HUMAN_LINE_TYPES.map(type=>`<button class="${type===suggestedType?"primary":""}" data-conflict-line-type="${escapeHtml(type)}" ${edgeId?`data-conflict-edge-id="${escapeHtml(edgeId)}"`:""}>${escapeHtml(type===suggestedType?t("confirmType",{type:displayValue(type)}):edgeId?t("changeTo",{type:displayValue(type)}):t("addAs",{type:displayValue(type)}))}</button>`).join("")}
    <button class="danger" data-no-conflict-connection="${escapeHtml(conflictId)}">${edgeId?t("rejectConnection"):t("markNoConnection")}</button></div>`;
}

function relationshipEvidenceHtml(edge, suggestedType) {
  if (!edge) return "";
  const evidence=edge.attributes?.system_confidence_evidence || {}, visual=evidence.visual_assessment || null, left=nodeById(edge.from_node), right=nodeById(edge.to_node), facts=[];
  const loop=node=>String(node?.attributes?.loop_number||node?.attributes?.tag_number||node?.label?.match(/\d{2,6}[A-Z]?$/i)?.[0]||"").replace(/[^A-Z0-9]/gi,"").toUpperCase()||null;
  const instrumentFunction=node=>node?.attributes?.instrument_function||null;
  const leftLoop=loop(left), rightLoop=loop(right), leftFunction=instrumentFunction(left), rightFunction=instrumentFunction(right);
  const pair=evidence.instrument_pair || (left?.kind==="instrument"&&right?.kind==="instrument"?{same_loop:Boolean(leftLoop&&leftLoop===rightLoop),loop_number:leftLoop&&leftLoop===rightLoop?leftLoop:null,functions:[leftFunction,rightFunction],functional_direction:leftFunction==="element"&&["transmitter","indicator","controller","recorder"].includes(rightFunction)?"forward":rightFunction==="element"&&["transmitter","indicator","controller","recorder"].includes(leftFunction)?"reverse":null,compatible:false}:null);
  if(pair?.same_loop&&pair.loop_number)facts.push(t("sameLoopEvidence",{loop:pair.loop_number}));
  if(pair?.functional_direction&&Array.isArray(pair.functions)&&pair.functions.filter(Boolean).length===2){
    const ordered=pair.functional_direction==="reverse"?[pair.functions[1],pair.functions[0]]:pair.functions;
    facts.push(t("functionFlowEvidence",{from:displayValue(ordered[0]),to:displayValue(ordered[1])}));
  }
  if(visual?.route_visible==="yes"&&visual?.endpoint_alignment==="both")facts.push(t("alignedRouteEvidence"));
  if(suggestedType&&visual?.legend_class===suggestedType)facts.push(t("legendMatchEvidence",{type:displayValue(suggestedType)}));
  const unique=[...new Set(facts)];
  return unique.length?`<div class="relationship-evidence"><h5>${t("supportingEvidence")}</h5><ul>${unique.map(fact=>`<li>${escapeHtml(fact)}</li>`).join("")}</ul></div>`:"";
}

function bindConflictDecisionActions(container, conflictId) {
  container.querySelectorAll("[data-conflict-line-type]").forEach(button=>{button.onclick=()=>button.dataset.conflictEdgeId?applyConflictEdgeType(conflictId,button.dataset.conflictEdgeId,button.dataset.conflictLineType):addConflictRelation(conflictId,button.dataset.conflictLineType);});
  container.querySelectorAll("[data-no-conflict-connection]").forEach(button=>{const edgeId=button.closest(".relationship-card")?.dataset.edgeId;button.onclick=()=>edgeId?rejectConflictEdge(conflictId,edgeId):confirmNoConnection(conflictId);});
}

function pairSummary(pair) {
  const labels = (pair.node_ids || []).map(id => displayNodeLabel(nodeById(id)) || id).join(" ↔ ");
  const details = [pair.left_service, pair.right_service, pair.left_direction, pair.right_direction].filter(Boolean).join(" · ");
  const score = pair.score == null ? "" : ` · ${t("score")} ${Number(pair.score).toFixed(2)}`;
  return `<article class="candidate-card pair"><div><strong>${escapeHtml(labels || "—")}</strong><small>${escapeHtml(details)}${escapeHtml(score)}</small></div></article>`;
}

function renderInspector() {
  const box = el("inspector");
  if (!app.selection) { box.className = "inspector empty"; box.textContent = t("selectItem"); return; }
  box.className = "inspector"; const {type, id} = app.selection;
  if (type === "assembly") {
    const a = assemblyById(id); if (!a) return select(null, null);
    box.innerHTML = `<h3>${t("assemblyIdentity")}</h3><p>${t("assemblyHelp")}</p>
      <label class="field">${t("label")}<input id="assemblyLabel" list="assemblyLabels" value="${escapeHtml(a.label || "")}"></label>
      <datalist id="assemblyLabels">${a.label_candidates.map(v=>`<option value="${escapeHtml(v)}">`).join("")}</datalist>
      <div class="quick-choices">${a.label_candidates.map(v=>`<button data-assembly-label="${escapeHtml(v)}">${escapeHtml(v)}</button>`).join("")}</div>
      <h4>${t("assemblyMembers")}</h4>${a.member_node_ids.map(nodeById).filter(Boolean).map(n=>`<button class="wide-choice" data-assembly-member="${escapeHtml(n.id)}">${escapeHtml(displayNodeLabel(n)||n.id)}</button>`).join("")}
      <div class="actions"><button id="saveAssembly" class="primary">${t("approve")}</button><button id="rejectAssembly">${t("reject")}</button></div>`;
    box.querySelectorAll("[data-assembly-label]").forEach(b=>b.onclick=()=>{el("assemblyLabel").value=b.dataset.assemblyLabel;});
    box.querySelectorAll("[data-assembly-member]").forEach(b=>b.onclick=()=>select("node",b.dataset.assemblyMember,{focus:true}));
    el("saveAssembly").onclick=()=>submit("assembly",id,"modify",{label:el("assemblyLabel").value});
    el("rejectAssembly").onclick=()=>submit("assembly",id,"reject");
  } else if (type === "node") {
    const node = nodeById(id); if (!node) return select(null, null); const b = node.bbox_global;
    box.innerHTML = `<h3>${escapeHtml(displayNodeLabel(node) || node.id)}</h3><div class="meta">${escapeHtml(node.id)} · ${t("modelConfidence")} ${escapeHtml(displayValue(node.confidence))} · ${t("review")} ${escapeHtml(displayValue(app.state.reviews.nodes[id]))}</div>
      ${nodeMeaningHtml(node)}
      ${(node.attributes?.assembly_ids || []).map(aid=>assemblyById(aid)).filter(Boolean).map(a=>`<button data-open-assembly="${escapeHtml(a.id)}">${t("assembly")}: ${escapeHtml(a.label || a.label_candidates.join(" / "))}</button>`).join("")}
      ${relatedConflictCards("node",id)}
      ${nodeLabelEditorHtml(node)}${selectField(t("kind"), "nodeKind", node.kind, app.state.taxonomy.kinds)}
      ${selectField(t("equipmentClass"), "nodeEquipmentClass", (node.attributes || {}).equipment_class || "", ["", ...app.state.taxonomy.equipment_classes])}
      ${selectField(t("valveType"), "nodeValveType", (node.attributes || {}).valve_type || "", ["", ...app.state.taxonomy.valve_types])}
      ${selectField(t("actuatorType"), "nodeActuation", (node.attributes || {}).actuation || "", ["", ...(app.state.taxonomy.actuation_types || [])])}
      ${selectField(t("instrumentFunction"), "nodeInstrumentFunction", (node.attributes || {}).instrument_function || "", ["", ...app.state.taxonomy.instrument_functions])}
      <div class="geometry">${inputField("X", "nodeX", b.x, "number")}${inputField("Y", "nodeY", b.y, "number")}${inputField("W", "nodeW", b.w, "number")}${inputField("H", "nodeH", b.h, "number")}</div>
      ${node.source_quote ? `<label class="field">${t("sourceText")}<textarea class="evidence" readonly>${escapeHtml(node.source_quote)}</textarea></label>` : ""}
      <label class="field">${t("attributesJson")}<textarea id="nodeAttrs">${escapeHtml(JSON.stringify(node.attributes || {}, null, 2))}</textarea></label>
      <div class="actions"><button id="saveNode">${t("saveChanges")}</button><button id="approveNode" class="primary">${t("approve")}</button><button id="rejectNode" class="danger">${t("reject")}</button></div>`;
    box.querySelectorAll("[data-open-assembly]").forEach(b=>b.onclick=()=>select("assembly",b.dataset.openAssembly,{focus:true}));
    bindRelatedConflictCards(box); bindNodeLabelEditor();
    el("saveNode").onclick = async () => { try { const attrs=JSON.parse(el("nodeAttrs").value); const assign=(key,value)=>value?attrs[key]=value:delete attrs[key]; assign("equipment_class",el("nodeEquipmentClass").value); assign("valve_type",el("nodeValveType").value); assign("actuation",el("nodeActuation").value); assign("instrument_function",el("nodeInstrumentFunction").value); await submit("node", id, "modify", { label: el("nodeLabel").value, kind: el("nodeKind").value, bbox_global: { x:+el("nodeX").value, y:+el("nodeY").value, w:+el("nodeW").value, h:+el("nodeH").value }, attributes: attrs }); } catch (error) { toast(error.message, true); } };
    el("approveNode").onclick = () => submit("node", id, "approve"); el("rejectNode").onclick = () => submit("node", id, "reject", {}, prompt(t("rejectReason")) || "");
  } else if (type === "evidence") {
    const candidate=evidenceById(id); if(!candidate)return select(null,null);
    const review=candidate.review || {status:"unreviewed"};
    const nearby=(candidate.nearby_node_ids || []).map(nodeById).filter(Boolean);
    const options=nearby.map(node=>`<option value="${escapeHtml(node.id)}">${escapeHtml(displayNodeLabel(node) || node.id)} · ${escapeHtml(displayValue(node.kind))}</option>`).join("");
    const dispositions=app.state.taxonomy.evidence_dispositions.map(value=>`<option value="${escapeHtml(value)}">${escapeHtml(displayValue(value))}</option>`).join("");
    box.innerHTML=`<h3>${escapeHtml(candidate.text)}</h3><div class="meta">${escapeHtml(candidate.id)} · ${t("pageShort",{page:candidate.page_index+1})} · ${escapeHtml(displayValue(candidate.candidate_kind))} · ${escapeHtml(displayValue(review.status))}</div>
      <div class="style-summary">${candidate.blocking?t("blockingCandidate"):t("nonBlockingCandidate")}</div>
      <label class="field">${t("printedText")}<textarea class="evidence" readonly>${escapeHtml(candidate.text)}</textarea></label>
      ${nearby.length?`<label class="field">${t("chooseNearbyEntity")}<select id="evidenceNode">${options}</select></label><button id="linkEvidence" class="primary wide-choice">${t("linkToEntity")}</button>`:`<p class="candidate-hint">${t("noCandidates")}</p>`}
      <button id="drawEvidence" class="wide-choice">${t("drawFromCandidate")}</button>
      <label class="field">${t("dismissAs")}<select id="evidenceDisposition">${dispositions}</select></label><button id="dismissEvidence" class="wide-choice">${t("dismissAs")}</button>
      ${review.status!=="unreviewed"?`<button id="reopenEvidence" class="quiet wide-choice">${t("reopen")}</button>`:""}
      <details><summary>${t("rawDetails")}</summary><pre class="conflict-json">${escapeHtml(JSON.stringify(candidate,null,2))}</pre></details>`;
    if(el("linkEvidence"))el("linkEvidence").onclick=()=>submit("evidence",id,"link",{node_id:el("evidenceNode").value},t("evidenceLinked"));
    el("drawEvidence").onclick=()=>beginDrawNode(candidate);
    el("dismissEvidence").onclick=()=>submit("evidence",id,"dismiss",{disposition:el("evidenceDisposition").value},t("evidenceDismissed"));
    if(el("reopenEvidence"))el("reopenEvidence").onclick=()=>submit("evidence",id,"reopen");
  } else if (type === "edge") {
    const edge = edges().find(item => item.id === id); if (!edge) return select(null, null);
    const edgeAttrs = edge.attributes || {}, visualStyle = edgeAttrs.visual_style || "unknown", styleConfidence = edgeAttrs.visual_style_confidence;
    const styleSummary = `${t("visualStyle")}: ${displayValue(visualStyle)}${styleConfidence == null ? "" : ` · ${t("styleConfidence")} ${Math.round(Number(styleConfidence) * 100)}%`}`;
    const styleEvidence = (edgeAttrs.style_evidence_ids || []).join(", ");
    const nodeOptions = nodes().filter(n => app.state.reviews.nodes[n.id] !== "rejected").map(n => [n.id, `${displayNodeLabel(n) || n.id} (${n.id})`]);
    const options = value => nodeOptions.map(([id, label]) => `<option value="${id}" ${id === value ? "selected" : ""}>${escapeHtml(label)}</option>`).join("");
    const candidates = candidateNodesForEdge(edge);
    const alternatives = edgeAttrs.route_alternatives || [];
    box.innerHTML = `<h3>${escapeHtml((edge.attributes || {}).line_id || edge.id)}</h3><div class="meta">${escapeHtml(edge.id)} · ${t("modelConfidence")} ${escapeHtml(displayValue(edge.confidence))} · ${t("review")} ${escapeHtml(displayValue(app.state.reviews.edges[id]))}</div>
      ${relatedConflictCards("edge",id)}
      <div class="style-summary">${escapeHtml(styleSummary)}${styleEvidence ? `<br>${escapeHtml(t("styleEvidence"))}: ${escapeHtml(styleEvidence)}` : ""}</div>
      ${alternatives.length ? `<label class="field">${t("routeChoices")}<select id="routeChoice"><option value="">${t("routePrompt")}</option>${alternatives.map((route, index) => `<option value="${escapeHtml(route.id)}">${t("routeChoice", {number:index + 1})}</option>`).join("")}</select></label>` : ""}
      ${endpointEditorHtml(edge, candidates)}
      <label class="field">${t("from")}<select id="edgeFrom">${options(edge.from_node)}</select></label><label class="field">${t("to")}<select id="edgeTo">${options(edge.to_node)}</select></label>
      ${selectField(t("lineType"), "edgeType", edge.line_type || "other", app.state.taxonomy.line_types)}
      <label class="field">${t("polylineJson")}<textarea id="edgePoints">${escapeHtml(JSON.stringify(edge.polyline_global || [], null, 2))}</textarea></label>
      <label class="field">${t("attributesJson")}<textarea id="edgeAttrs">${escapeHtml(JSON.stringify(edge.attributes || {}, null, 2))}</textarea></label>
      <div class="actions"><button id="saveEdge">${t("saveChanges")}</button><button id="approveEdge" class="primary">${t("approve")}</button><button id="rejectEdge" class="danger">${t("reject")}</button></div>`;
    bindRelatedConflictCards(box); bindEndpointEditor(box, edge, candidates);
    el("saveEdge").onclick = async () => { try { await submit("edge", id, "modify", { from_node: el("edgeFrom").value, to_node: el("edgeTo").value, line_type: el("edgeType").value, polyline_global: JSON.parse(el("edgePoints").value), attributes: JSON.parse(el("edgeAttrs").value) }); } catch (error) { toast(error.message, true); } };
    if (el("routeChoice")) {
      el("approveEdge").disabled = true;
      el("routeChoice").onchange = () => {
        const route = alternatives.find(item => item.id === el("routeChoice").value);
        el("approveEdge").disabled = !route;
        if (route) {
          el("edgeFrom").value = route.from_node;
          el("edgeTo").value = route.to_node;
          el("edgePoints").value = JSON.stringify(route.polyline_global, null, 2);
          el("edgeAttrs").value = JSON.stringify({...edgeAttrs, ...route.attributes, selected_route_candidate_id:route.id}, null, 2);
        }
        for (const overlayId of ["graphOverlay", "sourceOverlay"]) {
          const overlay = el(overlayId);
          overlay.querySelector(".route-alternative-preview")?.remove();
          if (route) overlay.append(svg("polyline", {points:route.polyline_global.map(p=>p.join(",")).join(" "),class:"route-alternative-preview source-conflict-route"}));
        }
      };
    }
    el("approveEdge").onclick = () => submit("edge", id, "approve", alternatives.length ? {route_candidate_id:el("routeChoice").value} : {}); el("rejectEdge").onclick = () => submit("edge", id, "reject", {}, prompt(t("rejectReason")) || "");
  } else if (type === "conflict") {
    const item = app.state.reviews.conflicts[id]; if (!item) return select(null, null);
    const candidates = item.candidates || {nodes: [], edges: [], labels: [], kinds: [], pairs: [], decisions: []};
    const canDecide=unresolvedConflict(item);
    const auditFinding=id.startsWith("audit-");
    const locations=item.conflict.source_locations||[];
    const sourceLinks=locations.length?`<section class="candidate-section"><h4>${t("sourceLocations")}</h4><div class="source-locations">${locations.map((location,index)=>`<button data-source-location="${index}">${escapeHtml(t("pageShort",{page:location.page_index+1}))} · ${escapeHtml(location.label||t("originalSource"))} ↗</button>`).join("")}</div></section>`:"";
    const targetNode = candidates.nodes.find(node => node.id === item.conflict.node_id) || candidates.nodes[0];
    const labels = canDecide && targetNode && candidates.labels.length ? `<section class="candidate-section"><h4>${t("competingLabels")}</h4><div class="quick-choices">${candidates.labels.map(label => `<button data-conflict-label="${escapeHtml(label)}">${escapeHtml(label)}</button>`).join("")}</div></section>` : "";
    const kinds = canDecide && targetNode && candidates.kinds.length ? `<section class="candidate-section"><h4>${t("competingKinds")}</h4><div class="quick-choices">${candidates.kinds.map(kind => `<button data-conflict-kind="${escapeHtml(kind)}">${escapeHtml(displayValue(kind))}</button>`).join("")}</div></section>` : "";
    const actualEdges=conflictEdges(item), conflictEdge=actualEdges.length===1?actualEdges[0]:null;
    const nodeList = candidates.nodes.length ? `<section class="candidate-section"><h4>${t("affectedObjects")}</h4><div class="affected-node-list">${candidates.nodes.map(node=>`<button data-view-conflict-node="${escapeHtml(node.id)}"><strong>${escapeHtml(displayNodeLabel(node)||node.id)}</strong><span>${escapeHtml(displayValue(node.kind))}</span></button>`).join("")}</div></section>` : "";
    const pairs = candidates.pairs.length ? `<section class="candidate-section"><h4>${t("candidatePairs")}</h4><div class="candidate-list">${candidates.pairs.map(pairSummary).join("")}</div></section>` : "";
    const decisions = canDecide && candidates.decisions.length ? `<div class="quick-choices">${candidates.decisions.map(decision => `<button data-conflict-decision="${escapeHtml(decision)}">${t(decision)}</button>`).join("")}</div>` : "";
    const contextualMerge = canDecide && item.conflict.type==="contextual_symbol_uncertainty" && item.conflict.primary_node_id && (item.conflict.absorbed_node_ids || []).length ? `<section class="candidate-section"><h4>${t("mergeActuator")}</h4>${selectField(t("suggestedActuation"),"contextActuation",item.conflict.suggested_actuation||"other",app.state.taxonomy.actuation_types||["other"])}<button id="mergeContextActuator" class="primary wide-choice">${t("mergeActuator")}</button></section>` : "";
    const suggestedType=conflictSuggestedType(item);
    const relationship = conflictEdge ? `<section class="relationship-card" data-edge-id="${escapeHtml(conflictEdge.id)}"><h4>${t(suggestedType&&suggestedType!==conflictEdge.line_type?"suggestedInterpretation":"currentInterpretation")}</h4><strong>${escapeHtml(displayNodeLabel(nodeById(conflictEdge.from_node))||conflictEdge.from_node)} → ${escapeHtml(displayNodeLabel(nodeById(conflictEdge.to_node))||conflictEdge.to_node)}</strong><span>${escapeHtml(displayValue(suggestedType||conflictEdge.line_type))}${conflictEdge.attributes?.visual_style?` · ${escapeHtml(t("detectedLine",{style:displayValue(conflictEdge.attributes.visual_style)}))}`:""}</span>${relationshipEvidenceHtml(conflictEdge,suggestedType)}${canDecide?lineTypeDecisionButtons(id,conflictEdge.id,suggestedType):""}</section>` : RELATION_CONFLICTS.has(item.conflict.type)&&candidates.nodes.length>=2 ? `<section class="relationship-card"><h4>${t(suggestedType?"suggestedInterpretation":"currentGraph")}</h4><strong>${escapeHtml(conflictNodeNames(item))}</strong><span>${suggestedType?escapeHtml(displayValue(suggestedType)):escapeHtml(t("noDirectConnection"))}</span>${canDecide?lineTypeDecisionButtons(id,null,suggestedType):""}</section>` : "";
    box.innerHTML = `<h3>${escapeHtml(conflictTitle(item))}</h3><div class="meta">${escapeHtml(id)} · ${t("review")} ${escapeHtml(displayValue(item.status))}</div>
      <section class="decision-question"><span>${t("decisionNeeded")}</span><strong>${escapeHtml(conflictQuestion(item))}</strong></section>
      ${sourceLinks}${nodeList}${relationship}${contextualMerge}${decisions}${labels}${kinds}${pairs}
      <section class="model-evidence"><h4>${t(auditFinding?"sourceAudit":"modelEvidence")}</h4><p>${escapeHtml(conflictEvidence(item))}</p></section>
      ${auditFinding&&canDecide?`<label for="auditDecisionNote">${t("auditDecision")}</label><textarea id="auditDecisionNote" class="audit-note" placeholder="${escapeHtml(t("decisionNote"))}">${escapeHtml(app.decisionDrafts[id]||"")}</textarea>`:""}
      <details><summary>${t("rawDetails")}</summary><pre class="conflict-json">${escapeHtml(JSON.stringify(item.conflict, null, 2))}</pre></details>
      ${canDecide?`<div class="actions"><button id="resolveConflict">${t("resolveNoChange")}</button><button id="waiveConflict">${t("waiveReason")}</button></div>`:`<div class="resolution-complete">${t("resolutionComplete")}<p>${escapeHtml(item.reason||"")}</p></div>`}`;
    box.querySelectorAll("[data-source-location]").forEach(button=>{button.onclick=()=>{
      const location=locations[Number(button.dataset.sourceLocation)];changePage(location.page_index,true);app.sourceFocus={...location,id};renderOverlay();
      requestAnimationFrame(()=>focusBounds(validBounds(location.bbox_global),{maximumZoom:1.5}));
    };});
    if(el("auditDecisionNote"))el("auditDecisionNote").oninput=event=>{app.decisionDrafts[id]=event.target.value;};
    box.querySelectorAll("[data-conflict-label]").forEach(button => { button.onclick = () => applyConflictChoice(id, "node", targetNode.id, "modify", {label: button.dataset.conflictLabel}, t("selectedCandidate", {value: button.dataset.conflictLabel})); });
    box.querySelectorAll("[data-conflict-kind]").forEach(button => { button.onclick = () => applyConflictChoice(id, "node", targetNode.id, "modify", {kind: button.dataset.conflictKind}, t("selectedCandidate", {value: displayValue(button.dataset.conflictKind)})); });
    box.querySelectorAll("[data-conflict-decision]").forEach(button => { button.onclick = () => submit("conflict", id, "resolve", {}, t("selectedCandidate", {value: displayValue(button.dataset.conflictDecision)})); });
    bindConflictDecisionActions(box,id);
    if(el("mergeContextActuator"))el("mergeContextActuator").onclick=()=>applyContextualMerge(id,item.conflict,el("contextActuation").value);
    box.querySelectorAll("[data-view-conflict-node]").forEach(button=>{button.onclick=()=>{const node=nodeById(button.dataset.viewConflictNode);if(node)select("node",node.id,{focus:true,preferredPage:node.page_index});};});
    if(canDecide){el("resolveConflict").onclick = () => {
      const reason=auditFinding?(app.decisionDrafts[id]||"").trim():t("resolvedWithoutChange");
      if(!reason)return toast(t("decisionRequired"),true);
      return submit("conflict",id,"resolve",{},reason);
    };
    el("waiveConflict").onclick = () => { const reason = prompt(t("conflictWaiveReason")); if (reason) submit("conflict", id, "waive", {}, reason); };}
  }
}

function select(type, id, {focus=false,preferredPage=null} = {}) {
  app.sourceFocus=null;
  const conflict=type==="conflict"?app.state.reviews.conflicts[id]:null;
  if (conflict?.conflict?.assembly_id && assemblyById(conflict.conflict.assembly_id)) return select("assembly", conflict.conflict.assembly_id, {focus:true,preferredPage});
  if(conflict&&app.workspaceView!=="comparison")setWorkspaceView("comparison");
  const shouldFocus=focus||Boolean(conflict), region=shouldFocus?selectionRegion(type,id,preferredPage):null;
  const targetPage=region?.page??(Number.isInteger(preferredPage)?preferredPage:null);
  if(targetPage!=null&&targetPage!==app.page)changePage(targetPage,true);
  app.selection = type ? {type, id} : null; renderOverlay(); renderQueue(); renderInspector();
  if(shouldFocus)focusSelectionRegion(type,id,targetPage);
}

async function submit(target_type, target_id, operation, after = {}, reason = "") {
  try {
    const result = await api("/api/action", { method: "POST", body: JSON.stringify({ expected_revision: app.state.revision, target_type, target_id, operation, after, reason }) });
    app.state = result.state; if (result.event.target_id) app.selection = {type: result.event.target_type, id: result.event.target_id}; renderAll(); toast(t("actionSaved", {operation: displayValue(operation)})); return result;
  } catch (error) { toast(error.message, true); throw error; }
}

function changePage(index, preserveSelection = false) {
  const count = app.state.session.pages.length; app.page = Math.max(0, Math.min(count - 1, index)); if (!preserveSelection) app.selection = null;
  const page = pageInfo(); el("sourceImage").src = `/api/pages/${app.page}/source`; el("inferenceImage").src = `/api/pages/${app.page}/inference`;
  el("pageLabel").textContent = t("pageCount", {current: app.page + 1, count}); el("pageRole").value = app.state.reviews.pages[String(app.page)]?.role || "pid";
  setStageSize(); renderOverlay(); renderInspector(); renderQueue();
}

function renderProgress() {
  const c = app.state.completion;
  const separateConflicts = c.unresolved_conflicts.filter(id => !app.state.reviews.conflicts[id]?.decision_target);
  const remaining = (c.unreviewed_assemblies || []).length + c.unreviewed_pages.length + c.unreviewed_nodes.length + c.unreviewed_edges.length + separateConflicts.length + (c.unreviewed_evidence || []).length + (c.invalid_evidence_links || []).length;
  el("progressBadge").textContent = remaining ? t("remaining", {count: remaining}) : t("readyExport");
  el("finishButton").disabled = !c.complete;
  const warnings = app.state.warnings || []; el("warningBox").classList.toggle("hidden", !warnings.length); el("warningBox").textContent = warnings.map(translateWarning).join("\n");
}

function renderAll() { renderProgress(); changePage(app.page, true); renderDataViews(); applyWorkspaceLayout(false); }

function fitView() {
  const page=pageInfo(), viewport=el("inferenceViewport");
  app.zoom=Math.max(.05,Math.min(1,(viewport.clientWidth-28)/page.width,(viewport.clientHeight-28)/page.height));
  setStageSize();
}

function beginDrawNode(evidence = null) {
  if (evidence) { app.pendingEvidence=evidence.id; setWorkspaceView("comparison"); changePage(evidence.page_index, true); }
  app.drawMode = true; toast(t("drawInstruction")); const overlay = el("graphOverlay");
  overlay.onpointerdown = event => {
    if (!app.drawMode || event.target !== overlay) return; const start = pointerPoint(event, overlay), draft = svg("rect", { x:start[0], y:start[1], width:1, height:1, class:"graph-node selected" }); overlay.append(draft); overlay.setPointerCapture(event.pointerId);
    overlay.onpointermove = move => { const p = pointerPoint(move, overlay); draft.setAttribute("x", Math.min(start[0], p[0])); draft.setAttribute("y", Math.min(start[1], p[1])); draft.setAttribute("width", Math.abs(p[0]-start[0])); draft.setAttribute("height", Math.abs(p[1]-start[1])); };
    overlay.onpointerup = async move => { overlay.onpointermove = null; overlay.onpointerup = null; overlay.onpointerdown = null; app.drawMode = false; const p = pointerPoint(move, overlay), box = {x:Math.round(Math.min(start[0],p[0])), y:Math.round(Math.min(start[1],p[1])), w:Math.max(2,Math.round(Math.abs(p[0]-start[0]))), h:Math.max(2,Math.round(Math.abs(p[1]-start[1])))}; const candidate=app.pendingEvidence?evidenceById(app.pendingEvidence):null; const result=await submit("node", "", "add", {kind:candidate?.candidate_kind==="instrument_tag"?"instrument":"equipment",label:candidate?.text||t("unlabelled"),bbox_global:box,page_index:app.page,attributes:{review_added:true,...(candidate?{source_text_ids:candidate.source_ids,native_text_candidate_id:candidate.id}:{})},confidence:"medium",source_quote:candidate?.text||null,alternate_readings:[],source_annotation_ids:[],source_evidence_ids:candidate?.source_ids||[]}); if(candidate){app.pendingEvidence=null;await submit("evidence",candidate.id,"link",{node_id:result.event.target_id},t("evidenceLinked"));} };
  };
}

async function addEdge() {
  const active = nodes().filter(n => app.state.reviews.nodes[n.id] !== "rejected"); if (active.length < 2) return toast(t("twoEntitiesRequired"), true);
  const from = prompt(t("fromEntityPrompt")); if (!from) return; const to = prompt(t("toEntityPrompt")); if (!to) return;
  const resolve = value => active.find(n => n.id === value || n.label === value)?.id; const fromId = resolve(from), toId = resolve(to); if (!fromId || !toId) return toast(t("unresolvedEntity"), true);
  const a=nodeById(fromId), b=nodeById(toId), points=[[Math.round(a.bbox_global.x+a.bbox_global.w/2),Math.round(a.bbox_global.y+a.bbox_global.h/2)],[Math.round(b.bbox_global.x+b.bbox_global.w/2),Math.round(b.bbox_global.y+b.bbox_global.h/2)]];
  await submit("edge", "", "add", {from_node:fromId,to_node:toId,line_type:"process",polyline_global:points,cross_sheet:a.page_index!==b.page_index,confidence:"medium",source_annotation_ids:[],attributes:{review_added:true}});
}

function populatePageRoles() {
  const selected = el("pageRole").value;
  el("pageRole").replaceChildren();
  app.state.taxonomy.page_roles.forEach(role => {
    const option = document.createElement("option"); option.value = role; option.textContent = displayValue(role); el("pageRole").append(option);
  });
  el("pageRole").value = selected || app.state.reviews.pages[String(app.page)]?.role || "pid";
}

function populateDataFilters() {
  const pages = app.state.inventory?.pages || [];
  [["inventoryPage", "inventory"], ["evidencePage", "evidence"]].forEach(([id]) => {
    const select=el(id), selected=select.value || "all";
    select.replaceChildren();
    [["all",t("allPages")],...pages.map(page=>[String(page.page_index),t("pageShort",{page:page.page_index+1})])].forEach(([value,label])=>{const option=document.createElement("option");option.value=value;option.textContent=label;select.append(option);});
    select.value=selected==="all" || pages.some(page=>String(page.page_index)===selected) ? selected : "all";
  });
  const inventoryKind=el("inventoryKind"), selectedKind=inventoryKind.value || "all";
  inventoryKind.replaceChildren();
  [["all",t("allTypes")],...["equipment","instrument","opc"].map(value=>[value,displayValue(value)])].forEach(([value,label])=>{const option=document.createElement("option");option.value=value;option.textContent=label;inventoryKind.append(option);});
  inventoryKind.value=selectedKind;
  const evidenceKind=el("evidenceKind"), selectedEvidence=evidenceKind.value || "all", kinds=[...new Set(evidenceCandidates().map(item=>item.candidate_kind))].sort();
  evidenceKind.replaceChildren();
  [["all",t("allTypes")],...kinds.map(value=>[value,displayValue(value)])].forEach(([value,label])=>{const option=document.createElement("option");option.value=value;option.textContent=label;evidenceKind.append(option);});
  evidenceKind.value=kinds.includes(selectedEvidence)?selectedEvidence:"all";
}

function populateQueuePages() {
  const selected=el("queuePage").value||"all";
  el("queuePage").replaceChildren();
  for(const [value,label] of [["all",t("allPages")],...app.state.session.pages.map(page=>[String(page.page_index),t("pageShort",{page:page.page_index+1})])]) {
    const option=document.createElement("option");option.value=value;option.textContent=label;el("queuePage").append(option);
  }
  el("queuePage").value=selected;
}

async function init() {
  applyLanguage();
  applyWorkspaceLayout(false);
  app.state = await api("/api/state");
  populatePageRoles();
  populateDataFilters();
  populateQueuePages();
  el("languageSwitch").onchange = () => {
    app.lang = el("languageSwitch").value;
    storeSetting("diagex.review.language", app.lang);
    applyLanguage(); populatePageRoles(); populateDataFilters(); populateQueuePages(); applyWorkspaceLayout(false); renderAll();
  };
  el("workspaceView").onchange=()=>setWorkspaceView(el("workspaceView").value);
  el("layoutMode").onchange = () => { app.layout = el("layoutMode").value; storeSetting("diagex.review.layout", app.layout); applyWorkspaceLayout(); };
  el("toggleQueue").onclick = () => { app.queueVisible = !app.queueVisible; storeSetting("diagex.review.queueVisible", app.queueVisible); applyWorkspaceLayout(); };
  el("toggleInspector").onclick = () => { app.inspectorVisible = !app.inspectorVisible; storeSetting("diagex.review.inspectorVisible", app.inspectorVisible); applyWorkspaceLayout(); };
  for (const id of ["sourceViewport", "inferenceViewport"]) {
    const viewport = el(id);
    viewport.addEventListener("pointerenter", () => { app.activeViewport = id; });
    viewport.addEventListener("pointerdown", () => { app.activeViewport = id; });
  }
  el("sourceViewport").onscroll = () => syncScroll(el("sourceViewport"), el("inferenceViewport")); el("inferenceViewport").onscroll = () => syncScroll(el("inferenceViewport"), el("sourceViewport"));
  el("prevPage").onclick=()=>changePage(app.page-1); el("nextPage").onclick=()=>changePage(app.page+1);
  el("showIssue").onclick=()=>{if(app.selection){app.sourceFocus=null;setWorkspaceView("comparison");focusSelectionRegion(app.selection.type,app.selection.id);}};
  el("zoomIn").onclick=()=>changeZoom(app.zoom+.15); el("zoomOut").onclick=()=>changeZoom(app.zoom-.15); el("fitView").onclick=fitView;
  el("backgroundMode").onchange=()=>{ const img=el("inferenceImage"), mode=el("backgroundMode").value; img.style.opacity=mode==="none"?0:(mode==="dim"?.2:1); };
  el("approvePage").onclick=()=>submit("page",String(app.page),"approve",{role:el("pageRole").value}); el("waivePage").onclick=()=>{const reason=prompt(t("pageWaiveReason"));if(reason)submit("page",String(app.page),"waive",{role:el("pageRole").value},reason);};
  el("queueSearch").oninput=renderQueue; el("queuePage").onchange=renderQueue; el("nextFinding").onclick=nextQueueFinding;
  el("queueFilter").onchange=renderQueue; el("queueType").onchange=renderQueue; el("addNode").onclick=()=>beginDrawNode(); el("addEdge").onclick=addEdge;
  el("inventorySearch").oninput=renderInventory; el("inventoryPage").onchange=renderInventory; el("inventoryKind").onchange=renderInventory; el("inventoryStatus").onchange=renderInventory;
  el("evidenceSearch").oninput=renderEvidence; el("evidencePage").onchange=renderEvidence; el("evidenceStatus").onchange=renderEvidence; el("evidenceKind").onchange=renderEvidence;
  el("undoButton").onclick=()=>submit("", "", "undo");
  el("finishButton").onclick=async()=>{try{const result=await api("/api/finish",{method:"POST",body:JSON.stringify({expected_revision:app.state.revision})});app.state=result.state;renderAll();toast(t("exportSuccess"));}catch(error){toast(error.message,true);}};
  app.queueVisible=true;
  changePage(0); fitView(); renderProgress(); renderDataViews(); applyWorkspaceLayout(false);
  if(visibleQueueRows().length)openQueueRow(visibleQueueRows()[0]);
}

init().catch(error => toast(t("loadFailed", {message: error.message}), true));
