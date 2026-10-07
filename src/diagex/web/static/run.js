"use strict";
const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
const run = params.get("run_dir") || "";
const NS = "http://www.w3.org/2000/svg";
let data, selected = null, zoom = 1, request = 0;
const endpoint = (path, extra = {}) => path + "?" + new URLSearchParams({run_dir: run, ...extra});
const el = (tag, text, cls) => { const n = document.createElement(tag); if (text != null) n.textContent = text; if (cls) n.className = cls; return n; };
const boxOf = row => row.bbox_global || row.bbox || row.source_bbox;
const pageOf = row => row.page_index ?? row.source_page_index;
const displayValue = value => ({"Model classified native glyph":"模型已识别原生图形","classification contradicts native glyph geometry":"识别类型与图形特征冲突，需核查","Model proposal has no selected native glyph":"模型发现额外图形，尚未通过定位校验","Model did not select or assess this native candidate":"模型尚未判定此候选","Conflicting candidate decisions":"模型对同一候选给出了冲突结论",equipment:"设备",instrument:"仪表",opc:"跨图连接符",valve:"阀门",check:"止回阀",check_valve:"止回阀",candidate:"候选符号",high:"高",medium:"中",low:"低",selected:"已识别",uncertain:"不确定",unresolved:"尚未确定",unreviewed:"尚未判定",rejected:"已排除",reject_annotation:"文字或注释",pid:"工艺管道及仪表图",legend:"图例页",completed:"已完成",paused:"已暂停",pending:"待处理",open_inline_valve:"开放式管内符号",round_symbol:"圆形符号",valve_body:"阀门形状"}[value] || value);
const labelOf = row => typeof row === "string" ? row : row.label || displayValue(row.symbol_class) || displayValue(row.reason) || row.message || row.type || displayValue(row.kind) || row.id || "历史记录";
const currentPage = () => data?.pages.find(p => p.page_index === Number($("page").value));
const rowKey = (category, index) => `${category}:${index}`;
function subtitle(row) {
  if (typeof row !== "object" || !row) return "";
  return [row.candidate_only ? "候选" : displayValue(row.kind || row.line_type),
    pageOf(row) != null ? `第 ${pageOf(row) + 1} 页` : "",
    displayValue(row.saved_status), row.id].filter(Boolean).join(" · ");
}
function renderList() {
  if (!data) return;
  const category = $("category").value, q = $("search").value.trim().toLowerCase();
  const rows = data[category] || [], container = $("items");
  container.replaceChildren(); let count = 0;
  rows.forEach((row, index) => {
    if (q && !JSON.stringify(row, (k,v) => k === "image_b64" ? undefined : v).toLowerCase().includes(q)) return;
    count++;
    const button = el("button", null, "result" + (row.candidate_only ? " candidate" : "") + (selected === rowKey(category,index) ? " selected" : ""));
    button.append(el("strong", labelOf(row)), el("small", subtitle(row)));
    const origin = data.evidence_origins?.[category]?.[index];
    if (origin) button.append(el('small',origin.summary,'evidence-origin'));
    button.addEventListener("click", () => select(category, index)); container.append(button);
  });
  $("count").textContent = `共 ${rows.length} 项，当前显示 ${count} 项 · 所有页`;
  if (!count) container.append(el("p", "无保存结果", "muted"));
}
function inspect(category, index) {
  const row = data[category][index], panel = $("detail"); panel.replaceChildren(el("h3", labelOf(row)));
  if (typeof row !== "object" || !row) return;
  if (row.image_b64) { const img = el("img"); img.src = "data:image/png;base64," + row.image_b64; img.alt = "已保存的图例图像"; panel.append(img); }
  const dl = el("dl");
  for (const [label,value] of [["ID",row.id],["类型",row.kind || row.symbol_class || row.line_type], ["置信度",row.confidence], ["历史状态",row.saved_status], ["说明",row.description || row.reason || row.saved_reason], ["原始依据",row.source_quote]]) {
    if (value != null && value !== "") dl.append(el("dt",label),el("dd",typeof value === "object" ? JSON.stringify(value) : String(displayValue(value))));
  }
  const explanation = row.attributes?.recognition_evidence || row.attributes?.raster_vlm_reason;
  if (explanation) dl.append(el("dt","识别说明（模型原文）"),el("dd",explanation));
  const diagnostic = data.diagnostics?.[category]?.[index];
  if (diagnostic) {
    dl.append(el("dt","识别状态"),el("dd",diagnostic.label));
    if (diagnostic.reference_status) dl.append(el("dt","参考使用情况"),el("dd",diagnostic.reference_status));
  }
  panel.append(dl);
  const origin = data.evidence_origins?.[category]?.[index];
  if (origin) {
    const section = el('section',null,'evidence-panel');
    section.append(el('h4','识别依据'));
    for (const source of origin.sources) {
      section.append(el('strong',source.label));
      if (source.reference_id) {
        const link = el('a',source.reference_id + (source.reference_version ? ` · v${source.reference_version}` : ''));
        link.href='/knowledge#'+encodeURIComponent(source.reference_id); link.target='_blank'; link.rel='noopener'; section.append(link);
      }
      if (source.variant_id) section.append(el('p',`画法： ${source.variant_id}`));
      if (source.entry_label) section.append(el('p',source.entry_label));
      if (source.legend_entry_id) section.append(el('small',`Legend entry / 图例条目: ${source.legend_entry_id}`));
      if (source.page_index != null) section.append(el('p',`Legend page / 图例页: ${source.page_index+1}`));
      if (source.evidence) section.append(el('p',source.evidence));
    }
    if (origin.supplied_sources.length) section.append(el('p','提供的参考: '+origin.supplied_sources.join(', ')));
    if (origin.citation_errors.length) section.append(el('p','未采纳的引用: '+origin.citation_errors.join(', ')));
    section.append(el('small',origin.note)); panel.append(section);
  }
  if (row.candidate_only) panel.append(el("p","未确定的候选项。"));
  if (category === "edges") {
    if (row.cross_sheet) panel.append(el("p","跨页连接：分别查看两端。"));
    for (const [field,label] of [["from_node","起点"],["to_node","终点"]]) {
      const i = data.nodes.findIndex(n => n.id === row[field]);
      if (i >= 0) { const b = el("button",`${label}: ${data.nodes[i].label || row[field]}`); b.addEventListener("click",()=>select("nodes",i)); panel.append(b); }
    }
  }
  const raw = el("details"), pre = el("pre",JSON.stringify(row,(k,v)=> k === "image_b64" ? "[illustration shown above]" : v,2));
  raw.append(el("summary","保存的依据"),pre); panel.append(raw);
}
function select(category,index) {
  selected = rowKey(category,index); $("category").value = category;
  const row = data[category][index];
  let page = pageOf(row);
  if (page == null && category === "edges") page = pageOf(data.nodes.find(n=>n.id === row.from_node) || {});
  if (page != null && data.pages.some(p=>p.page_index === page) && Number($("page").value) !== page) { $("page").value = String(page); renderPage(); }
  else renderOverlays();
  renderList(); inspect(category,index);
  const target = Array.from($("overlays").children).find(n => n.dataset.key === selected);
  if (target) target.scrollIntoView({block:"nearest",inline:"nearest"});
}
function overlay(tag, attrs, category,index) {
  const n = document.createElementNS(NS, tag);
  for (const [k,v] of Object.entries(attrs)) n.setAttribute(k,String(v));
  n.dataset.key = rowKey(category,index);
  if (n.dataset.key === selected) n.classList.add("selected-overlay");
  const title = document.createElementNS(NS,"title"); title.textContent = labelOf(data[category][index]); n.append(title);
  n.addEventListener("click",()=>select(category,index)); $("overlays").append(n);
}
function renderOverlays() {
  $("overlays").replaceChildren(); const page = currentPage();
  // A PDF's point dimensions are not necessarily the extraction's pixel frame.
  if (!page || page.geometry_inferred) return;
  if ($("connections").checked) data.edges.forEach((row,index)=>{
    const from = data.nodes.find(n=>n.id === row.from_node), to = data.nodes.find(n=>n.id === row.to_node);
    if (!from || !to || pageOf(from) !== page.page_index || pageOf(to) !== page.page_index || row.cross_sheet) return;
    if (Array.isArray(row.polyline_global) && row.polyline_global.length > 1) overlay("polyline",{points:row.polyline_global.map(p=>p.join(",")).join(" "),class:"connection"},"edges",index);
  });
  if ($("symbols").checked) data.nodes.forEach((row,index)=>{
    const b = boxOf(row); if (pageOf(row) !== page.page_index || !b || b.w <= 0 || b.h <= 0) return;
    overlay("rect",{x:b.x,y:b.y,width:b.w,height:b.h,class:"symbol" + (row.candidate_only ? " candidate" : "")},"nodes",index);
  });
  if (selected?.startsWith("legend:")) {
    const index = Number(selected.split(":")[1]), row = data.legend[index], b = boxOf(row);
    if (b && pageOf(row) === page.page_index) overlay("rect",{x:b.x,y:b.y,width:b.w,height:b.h,class:"symbol"},"legend",index);
  }
}
function sizeCanvas() {
  const p = currentPage(); if (!p) return;
  const width = Math.max(200,$("viewport").clientWidth-26) * zoom;
  $("canvas").setAttribute("width",width); $("canvas").setAttribute("height",width*p.height/p.width);
}
function renderPage() {
  const p = currentPage();
  const source = document.createElementNS(NS,"image"); source.id = "source";
  $("source").replaceWith(source);
  if (!p) { $("canvas").setAttribute("hidden", ""); $("pageNotice").textContent = "无页面信息，请查看结果列表。"; return; }
  $("canvas").removeAttribute("hidden");
  $("canvas").setAttribute("viewBox",`0 0 ${p.width} ${p.height}`);
  $("source").setAttribute("width",p.width); $("source").setAttribute("height",p.height);
  $("source").setAttribute("preserveAspectRatio","none");
  const notes = [];
  if (data.source_available) source.setAttribute("href",endpoint("/api/run-page",{page:p.page_index}));
  else notes.push("原图不可用，仅显示保存的标记。");
  if (p.geometry_inferred) notes.push("提取坐标系不可用，已隐藏标记。");
  const note = notes.join(" ");
  source.addEventListener("load",()=>{if($("source") === source) $("pageNotice").textContent = note;});
  source.addEventListener("error",()=>{if($("source") === source) $("pageNotice").textContent = "无法加载原图，仍可查看保存的结果。";});
  $("pageNotice").textContent = data.source_available ? "加载图纸… " + note : note;
  sizeCanvas(); renderOverlays();
}
async function load(version = "extraction") {
  const ticket = ++request; $("version").disabled = true;
  try {
    const response = await fetch(endpoint("/api/run-view",{version})); const value = await response.json();
    if (!response.ok) throw new Error(value.error || "无法加载历史结果");
    if (ticket !== request) return;
    data = value; selected = null;
    $("title").textContent = data.name;
    document.title = data.name + " · DiagEx";
    $("summary").textContent = [data.summary.model,data.summary.engine,data.summary.quality_status,`${data.nodes.length} 个符号／候选 · ${data.edges.length} 条连接 · ${data.legend.length} 条图例 · ${data.findings.length} 项待核查`].filter(Boolean).join(" | ");
    if (data.knowledge_source_names !== null && data.knowledge_source_names !== undefined) $("summary").textContent += '参考来源: ' + (data.knowledge_source_names.join(', ') || '未选择');
    const notes = [];
    if (version === "saved") notes.push("历史修改按原样显示，未经重新验证。");
    if (!data.source_available) notes.push("未找到原图，仍可查看保存的结果。");
    else if (data.source_verified !== true) notes.push("原图可用，但缺少历史哈希，无法核验。");
    $("notice").textContent = notes.join(" ");
    $("version").replaceChildren(...data.versions.map(v=>{const o=el("option",v.label);o.value=v.id;return o;})); $("version").value = version;
    $("page").replaceChildren(...data.pages.map(p=>{const o=el("option",`第 ${p.page_index+1} 页${p.role ? " · " + displayValue(p.role) : ""}`);o.value=String(p.page_index);return o;}));
    $("page").disabled = !data.pages.length;
    const initial = data.pages.find(p=>p.role === "pid") || data.pages[0]; if (initial) $("page").value = String(initial.page_index);
    $("downloads").replaceChildren(...data.downloads.map(name=>{const a=el("a",name);a.href=endpoint("/api/run-artifact",{name});return a;}));
    $("category").value = data.nodes.length ? "nodes" : data.legend.length ? "legend" : "findings";
    $("detail").textContent = "选择结果查看依据。";
    $("search").value = ""; renderList(); renderPage();
    history.replaceState(null,"",endpoint("/runs/view",{version}));
  } catch(error) { $("notice").textContent = error.message; $("title").textContent = "无法打开结果"; }
  finally { if(ticket === request) $("version").disabled = false; }
}
$("version").addEventListener("change",()=>load($("version").value));
$("category").addEventListener("change",renderList); $("search").addEventListener("input",renderList);
$("page").addEventListener("change",renderPage);
$("symbols").addEventListener("change",renderOverlays); $("connections").addEventListener("change",renderOverlays);
$("zoomIn").addEventListener("click",()=>{zoom=Math.min(8,zoom*1.5);sizeCanvas();});
$("zoomOut").addEventListener("click",()=>{zoom=Math.max(.25,zoom/1.5);sizeCanvas();});
$("fit").addEventListener("click",()=>{zoom=1;sizeCanvas();});
window.addEventListener("resize",sizeCanvas);
load(params.get("version") || "extraction");
