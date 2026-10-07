const $ = (id) => document.getElementById(id);

const I18N = {
  en: {
    subtitle: "P&ID extraction workbench", connecting: "Connecting", online: "Local server online",
    localWorkbench: "LOCAL WORKBENCH", title: "Turn a P&ID into an editable engineering graph",
    intro: "Configure the models, upload a drawing, and save the extraction results.",
    credentialsStayLocal: "Credentials are not saved", credentialsDetail: "Keys are kept in this server process, sent only to the selected API endpoint, and never saved to .env or run artifacts.",
    configure: "Configure extraction", configureHint: "Choose the provider and models used for this run.", provider: "Provider",
    baseUrl: "API endpoint", apiKey: "API key", apiKeyPlaceholder: "Leave blank to use the configured environment key", show: "Show", hide: "Hide",
    environmentKeyAvailable: "A configured key is available. Leave this field blank to use it.", keyRequired: "No configured key found. Enter one for this run.",
    visionModel: "AI model for reading symbols", visionHint: "Reads symbols and labels from the drawing, using the legend when available.",
    reasoningModel: "AI model for interpreting connections", reasoningModelHint: "Interprets how equipment and instruments connect, within and between pages.",
    reasoningHint: "Symbol detection starts with a fast pass. Enabled adds one limited check for uncertain or rejected symbols and ambiguous equipment bodies, using enlarged details.",
    reasoning: "Extra checking for uncertain symbols", automatic: "Automatic", enabled: "Enabled", disabled: "Disabled", effort: "Connection reasoning effort", low: "Low", medium: "Medium", high: "High", xhigh: "Extra high", engine: "Extraction workflow",
    chooseDrawing: "Choose the drawing", chooseDrawingHint: "PDF is recommended; vector PDFs give the strongest line evidence.",
    dropDrawing: "Drop a P&ID here", orBrowse: "or click to choose a file", replace: "Replace", artifactReuse: "Artifact reuse",
    reuseArtifacts: "Resume compatible artifacts", reuseHint: "Continue an interrupted matching run when possible.",
    startFresh: "Start a brand-new run", freshHint: "Bypass machine legend and symbol caches.",
    startExtraction: "Start extraction", queued: "Queued", extractionProgress: "Extraction progress", elapsed: "Elapsed", liveLog: "Live extraction log",
    autoScroll: "Auto-scroll", copy: "Copy", extractionFinished: "Extraction finished", runDirectory: "Run directory", recentRuns: "Saved runs",
    recentRunsHint: "Find saved extraction results and their output directories.", refreshRuns: "Refresh runs",
    filterRuns: "Find a run", filterRunsPlaceholder: "Drawing, run ID, or model",
    sourceVerified: "Source verified", sourceFound: "Source found", sourceNeeded: "Source PDF needed", sourceUploading: "Uploading source P&ID…",
    nodes: "nodes", connections: "connections", uploading: "Uploading drawing…", uploaded: "Drawing uploaded",
    running: "Running", succeeded: "Finished", failed: "Failed", quality: "Quality", equipment: "Equipment", valves: "Valves",
    instruments: "Instruments", segments: "Segments", opcs: "OPCs", tokens: "Tokens", buildIssues: "build issues", validationIssues: "validation issues",
    copied: "Log copied", sourceUnavailable: "Original source unavailable",
  },
  "zh-CN": {
    subtitle: "P&ID 提取工作台", connecting: "正在连接", online: "本地服务已连接",
    localWorkbench: "本地工作台", title: "将 P&ID 转换为可编辑的工程关系图",
    intro: "配置模型、上传图纸并运行提取，保存结果到输出目录。",
    credentialsStayLocal: "凭证不会被保存", credentialsDetail: "密钥仅保留在当前服务进程中并发送至所选 API 地址，不会写入 .env 或运行产物。",
    configure: "配置提取任务", configureHint: "选择本次运行使用的服务商和模型。", provider: "服务商",
    baseUrl: "API 地址", apiKey: "API 密钥", apiKeyPlaceholder: "留空则使用当前环境中已配置的密钥", show: "显示", hide: "隐藏",
    environmentKeyAvailable: "已检测到配置密钥；留空即可使用。", keyRequired: "没有检测到配置密钥，请输入本次运行使用的密钥。",
    visionModel: "识读符号的 AI 模型", visionHint: "用于原始符号识别；可在下方启用推理，帮助识别困难符号。",
    reasoningModel: "解释连接关系的 AI 模型", reasoningModelHint: "用于整页关系和跨页连接推理。",
    reasoningHint: "符号识别先进行快速检测；启用推理后，使用放大细节对不确定或被排除的符号及易混淆设备追加一次限时检查。",
    reasoning: "对不确定的符号追加检查", automatic: "自动", enabled: "启用", disabled: "禁用", effort: "连接关系推理强度", low: "低", medium: "中", high: "高", xhigh: "超高", engine: "提取流程",
    chooseDrawing: "选择图纸", chooseDrawingHint: "建议使用 PDF；矢量 PDF 可提供更可靠的线条证据。",
    dropDrawing: "将 P&ID 拖到这里", orBrowse: "或点击选择文件", replace: "更换", artifactReuse: "产物复用",
    reuseArtifacts: "复用兼容产物", reuseHint: "如存在匹配的中断任务，则从检查点继续。",
    startFresh: "开始全新运行", freshHint: "跳过机器图例和符号缓存。",
    startExtraction: "开始提取", queued: "排队中", extractionProgress: "提取进度", elapsed: "已用时间", liveLog: "实时提取日志",
    autoScroll: "自动滚动", copy: "复制", extractionFinished: "提取已完成", runDirectory: "运行目录", recentRuns: "已保存的运行",
    recentRunsHint: "查找已保存的提取结果和输出目录。", refreshRuns: "刷新运行列表",
    filterRuns: "查找运行", filterRunsPlaceholder: "图纸、运行 ID 或模型",
    sourceVerified: "原图已校验", sourceFound: "已找到原图", sourceNeeded: "需要关联原始图纸", sourceUploading: "正在上传原始 P&ID…",
    nodes: "节点", connections: "连接", uploading: "正在上传图纸…", uploaded: "图纸上传完成",
    running: "运行中", succeeded: "已完成", failed: "失败", quality: "质量状态", equipment: "设备", valves: "阀门",
    instruments: "仪表", segments: "管段", opcs: "跨页连接点", tokens: "令牌", buildIssues: "构建问题", validationIssues: "验证问题",
    copied: "日志已复制", sourceUnavailable: "原始图纸不可用",
  },
};

const state = {
  language: localStorage.getItem("diagex.web.language") || "zh-CN",
  config: null,
  upload: null,
  job: null,
  logCursor: 0,
  pollTimer: null,
  elapsedTimer: null,
  logLines: [],
  connected: false,
};

Object.assign(I18N.en, {detectedSymbols: "Detected symbols", detectSymbols: "Detect legends & symbols", detectHint: "Save legend and symbol detections without constructing a graph.", legendEntries: "Legend entries"});
Object.assign(I18N["zh-CN"], {detectedSymbols: "已检测符号", detectSymbols: "仅检测图例和符号", detectHint: "保存图例和符号检测结果，不构建连接图。", legendEntries: "图例条目"});
Object.assign(I18N.en, {
  cvMode: "Symbol detection", cvOff: "AI model with PDF text and lines (default)",
  cvGuided: "CV-guided AI detection", cvOn: "CV detection, then AI interpretation",
  cvHint: "The trained detector suggests symbol locations. The AI model checks the drawing and interprets the symbols. A matching legend or knowledge reference is optional evidence.",
  cvCheckpoint: "Trained detector file (.pt)",
  cvCheckpointHint: "Path on the machine running this server. Uses the configured checkpoint when provided.",
  cvDevice: "Run the symbol detector on", cvMissing: "Enter the path to the trained detector file.",
});
Object.assign(I18N["zh-CN"], {
  cvMode: "符号检测", cvOff: "AI 模型 + PDF 文字和线条（默认）", cvGuided: "CV 辅助 AI 检测", cvOn: "CV 检测后由 AI 单独解释",
  cvHint: "已训练的检测器提供符号位置建议，AI 模型核对图纸并解释符号。匹配的图例或知识库条目是可选的参考依据。",
  cvCheckpoint: "已训练的检测器文件（.pt）", cvCheckpointHint: "填写运行此服务的机器上的文件路径；如已配置权重，则自动填入。",
  cvDevice: "运行符号检测器的硬件", cvMissing: "请填写已训练的检测器文件路径。",
});

Object.assign(I18N.en, {
  modelSetup: "AI model setup", deepseekSetup: "DeepSeek (default)",
  qwenSetup: "Open weight model", customSetup: "Choose my own models",
  customSetupHint: "Choose the AI service and the models for reading symbols and interpreting connections.",
  currentWorkflow: "Current workflow", olderWorkflow: "Older workflow",
  cpuDevice: "Computer processor (CPU)", appleDevice: "Apple graphics processor", nvidiaDevice: "NVIDIA graphics card",
});
Object.assign(I18N["zh-CN"], {
  modelSetup: "AI 模型选择", deepseekSetup: "DeepSeek（默认）",
  qwenSetup: "开放权重模型", customSetup: "自行选择模型",
  customSetupHint: "选择 AI 服务商，以及识读符号和解释连接关系所用的模型。",
  currentWorkflow: "当前流程", olderWorkflow: "旧版流程",
  cpuDevice: "电脑处理器（CPU）", appleDevice: "Apple 图形处理器", nvidiaDevice: "NVIDIA 显卡",
});

function updateModelHint() {
  const profile = state.config?.model_profiles?.[$("modelPolicy").value];
  $("modelPolicyHint").textContent = (state.language === "zh-CN" ? profile?.hint_zh : profile?.hint) || t("customSetupHint");
}

function updateCV() {
  $("cvSettings").classList.toggle("hidden", $("cvMode").value === "off");
  if ($("cvMode").value !== "off") $("engine").value = "evidence-v2";
}

const providerDefaults = {
  openrouter: "https://openrouter.ai/api",
  kimi: "https://api.kimi.com/coding/v1",
  anthropic: "",
  azure: "",
};

function t(key) { return (I18N[state.language] || I18N.en)[key] || I18N.en[key] || key; }

function applyLanguage() {
  document.documentElement.lang = state.language;
  $("languageSwitch").value = state.language;
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    const value = t(node.dataset.i18n);
    if (value) node.textContent = value;
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
    node.placeholder = t(node.dataset.i18nPlaceholder);
  });
  updateKeyStatus();
  updateModelHint();
  if (state.connected) $("serverStatus").querySelector("span").textContent = t("online");
  if (state.job) renderJob(state.job);
  renderRecentRuns(window.__recentRuns || []);
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("show"), 2600);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const type = response.headers.get("content-type") || "";
  const payload = type.includes("json") ? await response.json() : { error: await response.text() };
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function humanBytes(value) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KiB`;
  return `${(value / 1024 ** 2).toFixed(1)} MiB`;
}

function formatTime(seconds) {
  const total = Math.max(0, Math.floor(seconds || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}` : `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function formatTokens(tokens) { return tokens >= 1e6 ? `${(tokens / 1e6).toFixed(3)}M` : `${Math.round(tokens / 1000)}k`; }

function updateKeyStatus() {
  if (!state.config) return;
  const provider = $("provider").value;
  const available = Boolean(state.config.configured_keys?.[provider]);
  $("keyStatus").textContent = available ? t("environmentKeyAvailable") : t("keyRequired");
  $("keyStatus").style.color = available ? "#138a58" : "#b76808";
}

function updateProvider() {
  const provider = $("provider").value;
  const field = $("baseUrlField");
  field.classList.toggle("hidden", provider === "anthropic");
  const current = $("baseUrl").value;
  if (!current || Object.values(providerDefaults).includes(current)) $("baseUrl").value = providerDefaults[provider];
  updateKeyStatus();
}

async function uploadDrawing(file) {
  const suffix = file.name.split(".").pop().toLowerCase();
  if (!["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"].includes(suffix)) {
    throw new Error("Upload a PDF or supported image file");
  }
  return api(`/api/uploads?filename=${encodeURIComponent(file.name)}`, {
    method: "POST", headers: { "Content-Type": file.type || "application/octet-stream" }, body: file,
  });
}

async function uploadFile(file) {
  $("uploadProgress").classList.remove("hidden");
  $("dropZone").classList.add("hidden");
  $("startButton").disabled = true;
  $("detectButton").disabled = true;
  try {
    const payload = await uploadDrawing(file);
    state.upload = payload.upload;
    window.maybeSuggestDrawingContext?.();
    $("fileName").textContent = state.upload.filename;
    $("fileSize").textContent = humanBytes(state.upload.size);
    $("fileSummary").classList.remove("hidden");
    $("startButton").disabled = false;
    $("detectButton").disabled = false;
    showToast(t("uploaded"));
  } finally {
    $("uploadProgress").classList.add("hidden");
  }
}

function extractionPayload() {
  return {
    ...window.drawingContextPayload(),
    upload_id: state.upload?.id,
    model_policy: $("modelPolicy").value,
    cv_mode: $("cvMode").value,
    cv_checkpoint: $("cvCheckpoint").value.trim(),
    cv_device: $("cvDevice").value,
    process_overview: $("processOverview").value.trim(),
    engineering_rules: $("engineeringRules").value.trim(),
    provider: $("provider").value,
    api_key: $("apiKey").value,
    base_url: $("baseUrl").value.trim(),
    vision_model: $("visionModel").value.trim(),
    reasoning_model: $("reasoningModel").value.trim(),
    reasoning_mode: $("reasoningMode").value,
    effort: $("effort").value,
    engine: $("engine").value,
    fresh: document.querySelector('input[name="runMode"]:checked').value === "fresh",
  };
}

async function startExtraction(stopAfter = "graph") {
  if (stopAfter === "detection") $("engine").value = "evidence-v2";
  $("formError").classList.add("hidden");
  $("startButton").disabled = true;
  try {
    if ($("cvMode").value !== "off") {
      if (!$("cvCheckpoint").value.trim()) throw new Error(t("cvMissing"));
      $("engine").value = "evidence-v2";
    }
    const payload = await api("/api/extractions", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({...extractionPayload(), stop_after: stopAfter}),
    });
    $("apiKey").value = "";
    state.job = payload.job;
    state.logCursor = 0;
    state.logLines = [];
    $("jobLog").textContent = "";
    $("runSection").classList.remove("hidden");
    $("resultSection").classList.add("hidden");
    $("runSection").scrollIntoView({ behavior: "smooth", block: "start" });
    renderJob(state.job);
    schedulePoll(250);
    startElapsedClock();
  } catch (error) {
    $("formError").textContent = error.message;
    $("formError").classList.remove("hidden");
    $("startButton").disabled = false;
    $("detectButton").disabled = false;
  }
}

function schedulePoll(delay = 1200) {
  clearTimeout(state.pollTimer);
  state.pollTimer = setTimeout(pollJob, delay);
}

async function pollJob() {
  if (!state.job) return;
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(state.job.id)}?after=${state.logCursor}`);
    state.job = job;
    state.logCursor = job.log_cursor;
    for (const entry of job.logs || []) state.logLines.push(entry.text);
    if (state.logLines.length > 10000) state.logLines = state.logLines.slice(-10000);
    $("jobLog").textContent = state.logLines.join("\n");
    if ($("autoScroll").checked) $("jobLog").scrollTop = $("jobLog").scrollHeight;
    renderJob(job);
    if (!["succeeded", "failed", "paused"].includes(job.status)) schedulePoll();
    else {
      clearInterval(state.elapsedTimer);
      $("startButton").disabled = false;
    $("detectButton").disabled = false;
      loadRecentRuns();
    }
  } catch (error) {
    $("jobError").textContent = error.message;
    $("jobError").classList.remove("hidden");
    schedulePoll(3000);
  }
}

function renderJob(job) {
  const label = job.status === "paused" ? (state.language === "zh-CN" ? "已暂停 · 可续跑" : "Paused · resumable") : (t(job.status) || job.status);
  $("jobState").textContent = label;
  $("jobState").className = `pill ${job.status === "failed" ? "error" : job.status === "succeeded" ? "ok" : job.status === "paused" ? "partial" : "running"}`;
  $("jobSubtitle").textContent = `${job.filename} · ${job.settings.vision_model} → ${job.settings.reasoning_model}`;
  $("jobError").classList.toggle("hidden", !job.error);
  $("jobError").textContent = job.error || "";
  $("activityBar").className = `activity-bar ${["succeeded", "paused"].includes(job.status) ? "done" : job.status === "failed" ? "failed" : ""}`;
  if (["succeeded", "paused"].includes(job.status) && job.result) renderResult(job);
}

function startElapsedClock() {
  clearInterval(state.elapsedTimer);
  state.elapsedTimer = setInterval(() => {
    if (!state.job?.started_at) return;
    const start = Date.parse(state.job.started_at);
    const end = state.job.finished_at ? Date.parse(state.job.finished_at) : Date.now();
    $("elapsedTime").textContent = formatTime((end - start) / 1000);
  }, 1000);
}

function metric(label, value) { return `<div class="metric"><strong>${value}</strong><span>${label}</span></div>`; }

function renderResult(job) {
  const result = job.result;
  const stats = result.stats || {};
  const quality = result.quality_status || "unknown";
  $("resultSection").classList.remove("hidden");
  $("qualityBadge").textContent = `${t("quality")}: ${quality}`;
  $("qualityBadge").className = `pill ${quality === "ok" ? "ok" : quality === "partial" ? "partial" : "error"}`;
  $("resultSummary").textContent = `${result.engine} · ${result.model} · ${result.build_issue_count} ${t("buildIssues")} · ${result.validation_issue_count} ${t("validationIssues")}`;
  $("metricGrid").innerHTML = [
    metric(t("equipment"), stats.equipment_count || 0), metric(t("valves"), stats.valve_count || 0),
    metric(t("instruments"), stats.instrument_count || 0), metric(t("segments"), stats.segment_count || 0),
    metric(t("opcs"), stats.opc_count || 0), metric(t("tokens"), formatTokens(result.tokens || 0)),
  ].join("");
  $("runPath").textContent = job.run_dir || "";
  $("viewRunButton").hidden = !job.run_dir;
  $("viewRunButton").href = "/runs/view?" + new URLSearchParams({run_dir:job.run_dir || ""});
  $("viewRunButton").textContent = state.language === "zh-CN" ? "查看结果" : "View results";
  const detectionOnly = result.workflow_stage === "detection";
  if (detectionOnly) {
    $("resultSummary").textContent = t("detectHint");
    $("metricGrid").innerHTML = [metric(t("detectedSymbols"), stats.detection_count || 0), metric(t("legendEntries"), result.legend.entry_count || 0), metric(t("tokens"), formatTokens(result.tokens || 0))].join("");
    $("qualityBadge").className = "pill partial";
  }
  if (result.pause_reason) $("resultSummary").textContent = `${result.pause_reason} · ${state.language === "zh-CN" ? "已保存检查点；选择续跑可继续未完成部分。" : "Checkpoints saved. Choose Resume to continue incomplete coverage."}`;
}

async function loadRecentRuns() {
  try {
    const payload = await api("/api/runs");
    window.__recentRuns = payload.runs || [];
    renderRecentRuns(window.__recentRuns);
  } catch (_error) { /* dashboard remains usable */ }
}

function renderRecentRuns(runs) {
  $("recentSection").classList.toggle("hidden", !runs.length);
  $("recentRuns").innerHTML = "";
  const query = ($("runFilter")?.value || "").trim().toLowerCase();
  const visibleRuns = query
    ? runs.filter((run) => [run.diagram, run.filename, run.run_id, run.model, run.engine, run.quality_status].some((value) => String(value || "").toLowerCase().includes(query)))
    : runs;
  for (const run of visibleRuns) {
    const row = document.createElement("div");
    row.className = "recent-run";
    const details = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = `${run.diagram || run.filename} · ${run.run_id || ""}`;
    const meta = document.createElement("div");
    meta.className = "recent-run-meta";
    for (const value of [
      run.finished_at,
      run.engine,
      run.model,
      run.quality_status,
      run.workflow_stage === "detection" ? t("detectSymbols") : `${run.node_count || 0} ${t("nodes")} · ${run.edge_count || 0} ${t("connections")}`,
    ]) {
      const item = document.createElement("span");
      item.textContent = value || "";
      meta.append(item);
    }
    details.append(title, meta);
    const location = document.createElement("code");
    location.textContent = run.run_dir;
    location.className = "recent-run-path";
    details.append(location);
    const view = document.createElement("a");
    view.className = "primary view-run";
    view.textContent = state.language === "zh-CN" ? "查看结果" : "View results";
    view.href = "/runs/view?" + new URLSearchParams({run_dir:run.run_dir});
    row.append(details, view);
    $("recentRuns").append(row);
  }
}

async function restoreActiveJob() {
  try {
    const payload = await api("/api/jobs");
    const job = payload.jobs?.[0];
    if (!job) return;
    state.job = job;
    state.logCursor = 0;
    state.logLines = [];
    $("runSection").classList.remove("hidden");
    renderJob(job);
    startElapsedClock();
    await pollJob();
  } catch (_error) { /* no in-memory jobs after a server restart */ }
}

async function initialise() {
  $("languageSwitch").value = state.language;
  applyLanguage();
  try {
    state.config = await api("/api/config");
    $("modelPolicy").value = state.config.model_policy || "evaluation";
    state.connected = true;
    $("serverStatus").classList.add("online");
    $("serverStatus").querySelector("span").textContent = t("online");
    $("provider").value = state.config.provider;
    $("baseUrl").value = state.config.base_url || providerDefaults[state.config.provider] || "";
    $("visionModel").value = state.config.vision_model || state.config.model || "";
    $("reasoningModel").value = state.config.reasoning_model || state.config.model || "";
    $("reasoningMode").value = state.config.reasoning_mode || "auto";
    $("effort").value = state.config.effort || "medium";
    $("engine").value = state.config.engine || "evidence-v2";
    $("cvCheckpoint").value = state.config.cv?.checkpoint || "";
    $("cvDevice").value = state.config.cv?.device || "cpu";
    updateCV();
    updateProvider();
    updateModelPolicy();
    await Promise.all([loadRecentRuns(), restoreActiveJob()]);
  } catch (error) {
    $("serverStatus").querySelector("span").textContent = error.message;
  }
}

$("languageSwitch").addEventListener("change", () => { state.language = $("languageSwitch").value; localStorage.setItem("diagex.web.language", state.language); applyLanguage(); });
$("provider").addEventListener("change", updateProvider);
function updateModelPolicy() {
  const profile = state.config?.model_profiles?.[$("modelPolicy").value];
  for (const id of ["provider", "visionModel", "reasoningModel", "engine"]) $(id).disabled = Boolean(profile);
  updateModelHint();
  if (profile) {
    $("provider").value = profile.provider;
    $("visionModel").value = profile.vision_model;
    $("reasoningModel").value = profile.reasoning_model;
    $("engine").value = profile.engine;
    $("baseUrl").value = providerDefaults[profile.provider];
    updateProvider();
  }
}
$("modelPolicy").addEventListener("change", updateModelPolicy);
$("cvMode").addEventListener("change", updateCV);
$("toggleKey").addEventListener("click", () => { const visible = $("apiKey").type === "text"; $("apiKey").type = visible ? "password" : "text"; $("toggleKey").textContent = t(visible ? "show" : "hide"); });
$("fileInput").addEventListener("change", () => { if ($("fileInput").files[0]) uploadFile($("fileInput").files[0]).catch((error) => { $("formError").textContent = error.message; $("formError").classList.remove("hidden"); $("dropZone").classList.remove("hidden"); }); });
$("replaceFile").addEventListener("click", () => { state.upload = null; $("fileSummary").classList.add("hidden"); $("dropZone").classList.remove("hidden"); $("fileInput").value = ""; $("startButton").disabled = true; });
for (const event of ["dragenter", "dragover"]) $("dropZone").addEventListener(event, (e) => { e.preventDefault(); $("dropZone").classList.add("dragging"); });
for (const event of ["dragleave", "drop"]) $("dropZone").addEventListener(event, (e) => { e.preventDefault(); $("dropZone").classList.remove("dragging"); });
$("dropZone").addEventListener("drop", (event) => { const file = event.dataTransfer.files[0]; if (file) uploadFile(file).catch((error) => { $("formError").textContent = error.message; $("formError").classList.remove("hidden"); $("dropZone").classList.remove("hidden"); }); });
$("startButton").addEventListener("click", () => startExtraction());
$("detectButton").addEventListener("click", () => startExtraction("detection"));
$("copyLog").addEventListener("click", async () => { await navigator.clipboard.writeText(state.logLines.join("\n")); showToast(t("copied")); });
$("refreshRuns").addEventListener("click", loadRecentRuns);
$("runFilter").addEventListener("input", () => renderRecentRuns(window.__recentRuns || []));

initialise();
