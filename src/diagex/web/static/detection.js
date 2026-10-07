"use strict";
const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
const runDir = params.get("run_dir") || "";
const CLASS_FIELDS = ["equipment_class", "valve_type", "instrument_class", "instrument_function", "connector_type"];
const BOX_FIELDS = ["x", "y", "w", "h"];
const MAX_ZOOM = 32;
const SELECTED_COLOR = "#c026d3";
let state, tab = "legends", selected = null, zoom = 1, drawingBox = false;
let drag = null, busy = false, draft = null;
function setting(key) { try { return localStorage.getItem(key); } catch { return null; } }
function saveSetting(key, value) { try { localStorage.setItem(key, value); } catch { /* Optional preference storage. */ } }
let language = setting("diagex.web.language") === "zh-CN" ? "zh-CN" : "en";
$("rater").value = params.get("rater") || "";
$("autoFocus").checked = setting("diagex.detection.autoFocus") !== "false";
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function t(key, values = {}) {
  return (DETECTION_I18N[language][key] ?? DETECTION_I18N.en[key] ?? key)
    .replace(/\{(\w+)\}/g, (_, name) => String(values[name] ?? ""));
}
const reasonKeys = {
  "Raw model detection": "rawDetection",
  "Native candidate without a confirmed detection": "nativeWithoutDetection",
  "Legend changed; recheck classification": "legendChanged",
  "Enter the reviewer name": "reviewerRequired",
  "Review the legend before confirming symbols": "reviewLegendFirst",
  "Another edit was saved. Reload before making this change.": "staleEdit",
  "Review changed. Reload before building the graph.": "staleBuild",
  "Symbol box must be non-empty and inside its source page": "insidePage",
  "No edit to undo": "emptyUndo",
  "Model did not select or assess this native candidate": "modelUnreviewed",
  "Model classified native glyph": "modelSelected",
  "conflicting or duplicate candidate decisions": "conflictingDecision",
  "Conflicting or duplicate candidate decisions": "conflictingDecision",
  "classification contradicts native glyph geometry": "shapeConflict",
  "Model proposal has no selected native glyph": "proposalUnanchored",
};
function translatedReason(value) {
  return value?.split("; ").map(part => reasonKeys[part] ? t(reasonKeys[part]) : part).join("; ");
}
function applyLanguage() {
  document.documentElement.lang = language;
  document.title = `${t("title")} · DiagEx`;
  $("languageSwitch").value = language;
  for (const node of document.querySelectorAll("[data-i18n]")) node.textContent = t(node.dataset.i18n);
  $("search").placeholder = t("searchHint");
  for (const [id, key] of [["languageSwitch","language"],["items","items"],["drawing","drawing"],["zoomIn","zoomIn"],["zoomOut","zoomOut"]]) {
    $(id).setAttribute("aria-label", t(key));
  }
}
async function api(path, body) {
  const response = await fetch(path, body ? {
    method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(body),
  } : {});
  const value = await response.json();
  if (!response.ok) throw Error(value.error || t("failed"));
  return value;
}
function error(e) {
  $("error").textContent = translatedReason(e.message);
  $("error").classList.remove("hidden");
}
async function act(body) {
  if (busy) return;
  busy = true;
  $("error").classList.add("hidden");
  try {
    state = await api("/api/detection-review/actions", {
      run_dir: runDir, revision: state.revision, rater: $("rater").value, ...body,
    });
    draft = null;
    render();
  } catch (e) { error(e); }
  finally { busy = false; }
}
function page() { return state?.pages.find(p => p.page_index === Number($("page").value)); }
function current() { return state?.[tab].find(row => row.id === selected); }
function legendType(entry) {
  if (entry.source === "built_in") return "library";
  if (entry.crop_quality === "omitted_abbreviation" || entry.attributes?.legend_kind === "abbreviation") return "text";
  return "drawing";
}
function legendTypeText(entry) { return t({drawing:"drawingLegend", text:"textLegend", library:"libraryLegend"}[legendType(entry)]); }
function rowName(row) { return (row.entry || row.detection).label || (row.entry || {}).symbol_class || row.detection?.attributes?.broad_category || t("unlabelled"); }
function rowLocation(row) {
  if (!row) return null;
  const index = row.entry ? row.entry.source_page_index : row.detection.page_index;
  const bounds = row.entry ? row.entry.source_bbox : row.detection.bbox;
  if (index == null || !bounds || !state.pages.some(p => p.page_index === index)) return null;
  return {pageIndex: index, bounds};
}
function rows() {
  const query = $("search").value.trim().toLowerCase();
  return state[tab].filter(row => {
    if (tab === "symbols" && row.detection.page_index !== page().page_index) return false;
    if (tab === "legends" && $("legendType").value !== "all" && legendType(row.entry) !== $("legendType").value) return false;
    if ($("filter").value !== "all" && row.status !== $("filter").value) return false;
    const obj = row.entry || row.detection;
    // Search meaningful text, never embedded thumbnail data.
    const text = [row.id, rowName(row), obj.kind, t(obj.kind), obj.symbol_class, JSON.stringify(obj.attributes || {}), row.reason, translatedReason(row.reason), t(row.origin || obj.kind)].join(" ");
    return text.toLowerCase().includes(query);
  });
}
function blockers() {
  const result = [];
  for (const [key, label] of [["legends","remainingLegends"],["symbols","remainingSymbols"]]) {
    const count = state[key].filter(row => row.status === "pending").length;
    if (count) result.push(t(label, {count}));
  }
  const count = Object.values(state.coverage).filter(checked => !checked).length;
  if (count) result.push(t("remainingPages", {count}));
  return result;
}
function select(id) {
  selected = id;
  draft = null;
  cancelDrawing();
  const location = rowLocation(current());
  if (location) $("page").value = String(location.pageIndex);
  render();
  if ($("autoFocus").checked) focusSelected();
}
function renderPages() {
  const prior = $("page").value;
  $("page").innerHTML = state.pages.map(p => {
    const incomplete = state.per_page_status[p.page_index] && state.per_page_status[p.page_index] !== "ok";
    return `<option value="${p.page_index}">${t("page")} ${p.page_index + 1} · ${esc(t(p.role))}${incomplete ? ` · ${t("incomplete")}` : ""}</option>`;
  }).join("");
  if (state.pages.some(p => String(p.page_index) === prior)) $("page").value = prior;
}
function render() {
  if (!state || !page()) return;
  $("runName").textContent = runDir.split("/").pop();
  const remaining = blockers();
  $("summary").textContent = `${t("summary", {
    legends: state.legends.filter(row => row.status === "confirmed").length,
    total: state.legends.length,
    symbols: state.symbols.filter(row => row.status === "confirmed").length,
    pending: state.symbols.filter(row => row.status === "pending").length,
    revision: state.revision,
  })}. ${remaining.join("; ") || t("ready")}`;
  $("build").disabled = !state.can_build;
  $("build").title = remaining.join("; ");
  $("buildDraft").disabled = !state.symbols.some(row => row.status === "confirmed");
  $("undo").disabled = !state.can_undo;
  $("legendTab").classList.toggle("active", tab === "legends");
  $("symbolTab").classList.toggle("active", tab === "symbols");
  $("legendTypeField").hidden = tab !== "legends";
  $("bulk").textContent = t(tab === "legends" ? "confirmLegends" : "confirmDetected");
  $("rejectCandidates").hidden = tab !== "symbols";
  $("add").textContent = t(tab === "legends" ? "addLegend" : "addSymbol");
  $("add").disabled = tab === "symbols" && page().role !== "pid";
  $("coverageLabel").hidden = tab !== "symbols" || page().role !== "pid";
  $("coverage").checked = !!state.coverage[String(page().page_index)];
  const location = rowLocation(current());
  $("focus").disabled = !location;
  $("selectionLabel").textContent = current()
    ? location ? t("selection", {name: rowName(current()), page: location.pageIndex + 1}) : `${rowName(current())} · ${t("noSource")}`
    : t("noSelection");
  $("selectionLabel").parentElement.classList.toggle("has-selection", !!location);
  $("items").replaceChildren();
  const visible = rows();
  for (const row of visible) {
    const button = document.createElement("button");
    button.className = row.id === selected ? "selected" : "";
    button.dataset.itemId = row.id;
    button.setAttribute("aria-pressed", String(row.id === selected));
    const origin = row.entry ? legendTypeText(row.entry) : t(row.origin);
    const reviewedBy = row.review_provenance ? ` · ${row.review_provenance.actor_type === "agent" ? "Agent reviewed" : "Human reviewed"}: ${row.review_provenance.rater}` : "";
    button.innerHTML = `${esc(rowName(row))}<small>${esc(t(row.status))} · ${esc(origin)}${esc(reviewedBy)}${row.stale ? ` · ${esc(t("legendChanged"))}` : ""}</small>`;
    button.onclick = () => select(row.id);
    $("items").append(button);
  }
  if (!visible.length) $("items").textContent = t("noItems");
  renderDrawing();
  renderInspector();
}
function svgElement(name, attributes) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  return element;
}
function contextBounds(bounds, sourcePage, multiplier = .4) {
  const margin = Math.max(25, Math.round(Math.max(bounds.w, bounds.h) * multiplier));
  const x = Math.max(0, bounds.x - margin), y = Math.max(0, bounds.y - margin);
  return {x, y, w: Math.min(sourcePage.width, bounds.x + bounds.w + margin) - x,
    h: Math.min(sourcePage.height, bounds.y + bounds.h + margin) - y};
}
function cropUrl(index, bounds) {
  return `/api/detection-crop?${new URLSearchParams({run_dir:runDir, page:index, box:BOX_FIELDS.map(key => bounds[key]).join(",")})}`;
}
function baseScale() {
  const p = page(), viewport = $("viewport");
  return Math.max(.001, Math.min((viewport.clientWidth - 32) / p.width, (viewport.clientHeight - 32) / p.height));
}
function viewportCenter() {
  const matrix = $("drawing").getScreenCTM();
  if (!matrix) return null;
  const rect = $("viewport").getBoundingClientRect();
  return new DOMPoint(rect.left + $("viewport").clientWidth / 2, rect.top + $("viewport").clientHeight / 2).matrixTransform(matrix.inverse());
}
function centerOn(point) {
  const viewport = $("viewport"), rect = viewport.getBoundingClientRect();
  const target = new DOMPoint(point.x, point.y).matrixTransform($("drawing").getScreenCTM());
  viewport.scrollLeft += target.x - rect.left - viewport.clientWidth / 2;
  viewport.scrollTop += target.y - rect.top - viewport.clientHeight / 2;
}
function focusSelected() {
  const location = rowLocation(current());
  if (!location) return;
  if (location.pageIndex !== page().page_index) $("page").value = String(location.pageIndex);
  const b = location.bounds, viewport = $("viewport");
  const scale = Math.min(viewport.clientWidth * .48 / Math.max(1, b.w), viewport.clientHeight * .48 / Math.max(1, b.h));
  zoom = Math.max(1, Math.min(MAX_ZOOM, scale / baseScale()));
  renderDrawing();
  centerOn({x:b.x + b.w / 2, y:b.y + b.h / 2});
}
function setZoom(value) {
  const center = viewportCenter();
  zoom = Math.max(1, Math.min(MAX_ZOOM, value));
  renderDrawing();
  if (center) centerOn(center);
}
function renderDrawing() {
  const p = page(), svg = $("drawing");
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${p.width} ${p.height}`);
  const scale = baseScale() * zoom;
  svg.style.width = `${p.width * scale}px`;
  svg.style.height = `${p.height * scale}px`;
  $("zoomLabel").textContent = `${Math.round(zoom * 100)}%`;
  svg.append(svgElement("image", {href:`/api/detection-page?${new URLSearchParams({run_dir:runDir, page:p.page_index})}`, width:p.width, height:p.height}));
  const location = rowLocation(current());
  const active = location && location.pageIndex === p.page_index ? location.bounds : null;
  if (active) {
    const context = contextBounds(active, p, 1.5);
    // Re-render the local PDF region so automatic zoom does not magnify a blurry preview.
    svg.append(svgElement("image", {href:cropUrl(p.page_index, context), x:context.x, y:context.y, width:context.w, height:context.h, class:"focused-source"}));
  }
  const boxes = tab === "symbols" ? state.symbols.filter(row => row.detection.page_index === p.page_index)
    : state.legends.filter(row => rowLocation(row)?.pageIndex === p.page_index);
  for (const row of boxes) {
    if (row.id === selected) continue;
    const b = rowLocation(row).bounds;
    const color = row.status === "confirmed" ? "#12804d" : row.status === "rejected" ? "#7e8792" : "#d47d00";
    const rect = svgElement("rect", {x:b.x, y:b.y, width:b.w, height:b.h, stroke:color, fill:color,
      class:active ? "other-item muted" : "other-item", "data-item-id":row.id});
    if (row.status === "rejected") rect.setAttribute("stroke-dasharray", "4 4");
    rect.onclick = () => { if (!drawingBox) select(row.id); };
    const title = svgElement("title", {});
    title.textContent = `${rowName(row)} · ${t(row.status)}`;
    rect.append(title);
    svg.append(rect);
  }
  if (active) {
    // The white halo and magenta outline stay visible across dense black strokes.
    for (const [className, stroke] of [["selection-halo", "#ffffff"], ["active", SELECTED_COLOR]]) {
      svg.append(svgElement("rect", {x:active.x, y:active.y, width:active.w, height:active.h,
        stroke, fill:SELECTED_COLOR, class:className, "data-item-id":selected}));
    }
  }
}
function field(id, label, value, type = "text") {
  return `<label>${esc(t(label))}<input id="${id}" type="${type}" value="${esc(value)}"></label>`;
}
function options(id, label, values, value) {
  return `<label>${esc(t(label))}<select id="${id}">${values.map(v => `<option value="${v}" ${v === value ? "selected" : ""}>${esc(t(v))}</option>`).join("")}</select></label>`;
}
function renderInspector() {
  const row = draft || current();
  if (!row) { $("inspector").innerHTML = `<p>${t(drawingBox ? "drawHint" : "selectHint")}</p>`; return; }
  const legend = tab === "legends", obj = legend ? row.entry : row.detection;
  let html = `<h3>${t(legend ? "entry" : "symbol")}</h3>`;
  if (row.reason && !reasonKeys[row.reason] && language === "zh-CN") html += `<small>${t("sourceEvidence")}</small>`;
  html += `<p>${esc(translatedReason(row.reason) || t(row.status || "newEntry"))}</p>`;
  if (legend) {
    const type = legendType(obj);
    html += `<span class="origin-badge">${esc(legendTypeText(obj))}</span>`;
    if (type !== "drawing") html += `<p>${t(type === "text" ? "sourceDefinition" : "libraryDefinition")}</p>`;
    if (obj.image_b64) html += `<img alt="${t("sourceCrop")}" src="data:image/png;base64,${esc(obj.image_b64)}">`;
  }
  html += field("editLabel", legend ? "name" : "tag", obj.label);
  const symbolKinds = obj.kind === "raster_symbol" ? ["raster_symbol","equipment","instrument","opc"] : ["equipment","instrument","opc"];
  html += options("editKind", "kind", legend ? ["equipment","instrument","line","valve","connector","other"] : symbolKinds, obj.kind);
  if (!legend && row.source_observation) html += `<p>${esc(t("rasterReviewHint"))}</p>`;
  const interpretation = row.source_observation?.legend_interpretation;
  if (!legend && interpretation) html += `<p><strong>${esc(t("legendSuggestion"))}</strong> ${esc(interpretation.reason)}</p>`;
  if (legend) html += field("editClass", "symbolClass", obj.symbol_class) + `<label>${t("description")}<textarea id="editDescription">${esc(obj.description)}</textarea></label>`;
  else for (const key of CLASS_FIELDS) html += field(key, key, obj.attributes?.[key] || "");
  const b = legend ? obj.source_bbox : obj.bbox;
  const location = rowLocation(row);
  if (b && location) {
    const p = state.pages.find(p => p.page_index === location.pageIndex);
    const context = contextBounds(b, p), url = cropUrl(location.pageIndex, context);
    // Matching highlight in the enlarged detail keeps the selected footprint unambiguous.
    html += `<a target="_blank" rel="noopener" href="${url}" class="detail-link"><svg class="source-detail" viewBox="0 0 ${context.w} ${context.h}" role="img" aria-label="${t("sourceDetail")}"><image href="${url}" width="${context.w}" height="${context.h}"/><rect x="${b.x-context.x}" y="${b.y-context.y}" width="${b.w}" height="${b.h}"/></svg></a><small>${t("sourceDetail")}</small>`;
  }
  if (b) html += `<div class="box-fields">${BOX_FIELDS.map(key => field(`box_${key}`, key, b[key], "number")).join("")}</div><small>${t("coordinates")}</small>`;
  html += `<div><button id="save" class="primary">${t("save")}</button><button id="reject">${t("reject")}</button><button id="pending">${t("keepPending")}</button></div>`;
  $("inspector").innerHTML = html;
  function save(status) {
    const updated = structuredClone(obj);
    updated.label = $("editLabel").value.trim(); updated.kind = $("editKind").value;
    if (legend) { updated.symbol_class = $("editClass").value.trim(); updated.description = $("editDescription").value; }
    else {
      updated.attributes ||= {};
      for (const key of CLASS_FIELDS) {
        delete updated.attributes[key];
        if ($(key).value.trim()) updated.attributes[key] = $(key).value.trim();
      }
    }
    if (b) updated[legend ? "source_bbox" : "bbox"] = Object.fromEntries(BOX_FIELDS.map(key => [key, Number($("box_" + key).value)]));
    act({action:legend ? "save_legend" : "save_symbol", id:row.id, status, [legend ? "entry" : "detection"]:updated});
  }
  $("save").onclick = () => save("confirmed");
  $("reject").onclick = () => save("rejected");
  $("pending").onclick = () => save("pending");
}
function point(event) {
  const p = new DOMPoint(event.clientX, event.clientY).matrixTransform($("drawing").getScreenCTM().inverse());
  return {x:Math.round(Math.max(0, Math.min(page().width, p.x))), y:Math.round(Math.max(0, Math.min(page().height, p.y)))};
}
function cancelDrawing() { drag = null; drawingBox = false; $("drawing").style.cursor = ""; $("draftBox")?.remove(); }
$("drawing").onpointerdown = event => {
  if (!drawingBox) return;
  event.preventDefault(); drag = point(event); $("drawing").setPointerCapture(event.pointerId);
};
$("drawing").onpointermove = event => {
  if (!drag) return;
  const p = point(event); $("draftBox")?.remove();
  $("drawing").append(svgElement("rect", {id:"draftBox", x:Math.min(drag.x,p.x), y:Math.min(drag.y,p.y), width:Math.abs(drag.x-p.x), height:Math.abs(drag.y-p.y), fill:SELECTED_COLOR, stroke:SELECTED_COLOR}));
};
$("drawing").onpointerup = event => {
  if (!drag) return;
  const p = point(event), bbox = {x:Math.min(drag.x,p.x), y:Math.min(drag.y,p.y), w:Math.abs(drag.x-p.x), h:Math.abs(drag.y-p.y)};
  cancelDrawing();
  if (!bbox.w || !bbox.h) return;
  draft = {detection:{id:"new", page_index:page().page_index, tile_id:"manual-review", kind:"equipment", label:"", bbox, confidence:"high", attributes:{}, source_text_ids:[]}};
  renderInspector();
};
$("drawing").onpointercancel = cancelDrawing;
$("add").onclick = () => {
  selected = null; draft = null; cancelDrawing();
  if (tab === "legends") draft = {entry:{label:"", kind:"other", symbol_class:"", source:"customer_override"}};
  else { if (page().role !== "pid") return; drawingBox = true; $("drawing").style.cursor = "crosshair"; }
  render();
};
$("bulk").onclick = () => {
  const ids = rows().filter(row => row.status === "pending" && !row.stale && (tab === "legends" || row.origin === "detected")).map(row => row.id);
  if (ids.length && confirm(t(tab === "legends" ? "confirmLegendQuestion" : "confirmDetectedQuestion", {count:ids.length}))) act({action:tab === "legends" ? "confirm_legends" : "confirm_detected", ids});
};
$("rejectCandidates").onclick = () => {
  const ids = rows().filter(row => row.status === "pending" && ["native_candidate","proposal"].includes(row.origin)).map(row => row.id);
  if (ids.length && confirm(t("rejectQuestion", {count:ids.length}))) act({action:"reject_candidates", ids});
};
$("coverage").onchange = () => act({action:"coverage", page_index:page().page_index, checked:$("coverage").checked});
$("undo").onclick = () => act({action:"undo"});
$("build").onclick = () => { location.href = "/?" + new URLSearchParams({reviewed_run:runDir, reviewed_revision:state.revision}); };
$("buildDraft").onclick = () => { location.href = "/?" + new URLSearchParams({reviewed_run:runDir, reviewed_revision:state.revision, draft:"true"}); };
for (const [id, name] of [["legendTab","legends"],["symbolTab","symbols"]]) $(id).onclick = () => {
  tab = name; selected = null; draft = null; cancelDrawing(); render();
};
$("page").onchange = () => { selected = null; draft = null; zoom = 1; cancelDrawing(); render(); };
for (const id of ["filter","search","legendType"]) $(id).addEventListener(id === "search" ? "input" : "change", render);
$("zoomIn").onclick = () => setZoom(zoom * 1.4);
$("zoomOut").onclick = () => setZoom(zoom / 1.4);
$("fit").onclick = () => { zoom = 1; renderDrawing(); $("viewport").scrollTo(0,0); };
$("focus").onclick = focusSelected;
$("autoFocus").onchange = () => { saveSetting("diagex.detection.autoFocus", String($("autoFocus").checked)); if ($("autoFocus").checked) focusSelected(); };
$("languageSwitch").onchange = () => {
  // Preserve unsaved fields and current selection when changing UI language.
  const values = [...$("inspector").querySelectorAll("input,select,textarea")].map(input => [input.id,input.value]);
  language = $("languageSwitch").value;
  saveSetting("diagex.web.language", language); saveSetting("diagex.review.language", language);
  applyLanguage();
  if (state) { renderPages(); render(); for (const [id,value] of values) if ($(id)) $(id).value = value; }
};
async function load() {
  try {
    state = await api(`/api/detection-review?${new URLSearchParams({run_dir:runDir})}`);
    $("error").classList.add("hidden"); renderPages(); render();
  } catch (e) { error(e); }
}
$("reload").onclick = () => { draft = null; load(); };
let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (state) { if (current() && $("autoFocus").checked) focusSelected(); else renderDrawing(); } }, 100);
});
applyLanguage();
load();
