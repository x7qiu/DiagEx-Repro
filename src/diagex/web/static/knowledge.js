(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const types = {symbol:'符号',line_style:'线型',assembly_pattern:'组合',scope_boundary:'范围界线',convention:'约定'};
  const categories = {instruments:'仪表',valves_actuators:'阀门与执行机构',measurement:'测量元件',lines_signals:'线型与信号',connectors:'连接符',equipment:'设备',piping:'管件'};
  let catalog, category = '', visibleCount = 40;
  const replacementLabels=new Map();
  const el = (tag, text, className) => { const n=document.createElement(tag); if(text!==undefined)n.textContent=text; if(className)n.className=className; return n; };
  const full = () => $('viewFilter').value === 'full';
  const letters = () => $('viewFilter').value === 'letters';
  const title = e => full() ? e.concept : e.catalog?.short_name || e.concept;
  const sourceName = e => catalog.documents.find(d=>d.id===e.source.document_id)?.publisher || (e.source.document_id?.startsWith('isa-')?'ISA 5.1:2009':e.source.document || '通用');
  const standards = e => e.applicability.standards || [];
  const standardKey = s => `${s.name} · ${s.edition || '版本未知'}`;
  const assetUrl = (e,a) => a ? `/api/context/asset?${new URLSearchParams({id:e.id,asset:a.id})}` : `/api/context/illustration?${new URLSearchParams({id:e.id})}`;
  const variants = e => full() ? e.assets.filter(a=>a.role==='variant' && !a.id.startsWith('catalog-')) : (e.catalog?.displayed_asset_ids || []).map(id=>e.assets.find(a=>a.id===id)).filter(Boolean);
  function details(parent, label) {const d=el('details');d.append(el('summary',label));parent.append(d);return d;}
  function picture(e,a,context=false) {
    const f=el('figure',undefined,`visual${context?' context-view':''}`), b=el('button'), img=el('img');
    const label=a?.label || e.concept, raster=Boolean(a?.web || catalog.documents.find(d=>d.id===e.source.document_id)?.scan_note); b.type='button'; b.setAttribute('aria-label',`放大 ${label}`);
    img.src=assetUrl(e,a); img.alt=label; img.loading='lazy'; b.append(img);
    b.onclick=()=>{ $('imageTitle').textContent=`${title(e)} · ${label}`; $('largeImage').src=img.src; $('largeImage').alt=label; $('largeImage').style.width=raster && a?`${Math.min(1000,Math.max(400,a.width*4))}px`:''; $('imageDialog').showModal(); };
    f.append(b,el('figcaption',`${label}${raster && a?` · ${a.width} × ${a.height} 像素（原图）`:''}${a?.quality==='ambiguous'?'扫描不清晰':''} · 点击放大`)); return f;
  }
  function section(parent,label,lines) {
    if(!lines?.length)return; parent.append(el('h3',label)); const ul=el('ul'); lines.forEach(s=>ul.append(el('li',s))); parent.append(ul);
  }
  const letterColumns = [
    ['measured_variable','Measured / initiating variable','被测／初始变量'],
    ['variable_modifier','Variable modifier','变量修饰词'],
    ['readout_function','Readout / passive function','读出／被动功能'],
    ['output_function','Output / active function','输出／主动功能'],
    ['function_modifier','Function modifier','功能修饰词'],
  ];
  function letterTable(parent,e) {
    const matrix=e.letter_matrix;
    if(!matrix)return;
    parent.append(el('p','按字母位置及列识读；— 表示原表空白，不等于供选用。','hint'));
    const noteDetails=el('details',undefined,'letter-notes');noteDetails.append(el('summary','解释注释'));
    const noteTargets=new Map();
    (matrix.notes || []).forEach(note=>{
      const d=el('div',undefined,'letter-note');d.id=`letter-note-${note.marker}`;d.tabIndex=-1;
      d.append(el('h4',`(${note.marker})${note.status==='unresolved'?'未查明':''}`),el('p',note.text));
      d.append(el('small',`PDF 页码 ${note.source.pdf_page} · ${note.source.passage || note.source.table || ''}`,'hint'));
      noteDetails.append(d);noteTargets.set(note.marker,d);
    });
    function footnotes(parent,refs) {
      (refs || []).forEach(marker=>{
        const target=noteTargets.get(marker) || noteTargets.get(marker.replace(/(?<=\d)[a-z]$/,''));
        const b=el('button',`(${marker})`,'letter-footnote');b.type='button';b.setAttribute('aria-label',`查看注释 ${marker}`);
        b.onclick=()=>{noteDetails.open=true;target?.scrollIntoView({block:'center'});target?.focus({preventScroll:true});};parent.append(b);
      });
    }
    const wrap=el('div',undefined,'letter-table-scroll');wrap.tabIndex=0;wrap.setAttribute('aria-label','仪表字母含义');
    const table=el('table',undefined,'letter-table');table.append(el('caption',`${e.catalog?.short_name || e.concept} · ${matrix.rows.length} letters / 字母`));
    const head=el('thead'), groups=el('tr'),letter=el('th','字母');letter.rowSpan=2;letter.scope='col';groups.append(letter);
    for(const [label,span,key] of [['首位字母',2,'first_letters'],['后继字母',3,'succeeding_letters']]){
      const th=el('th',label);th.colSpan=span;th.scope='colgroup';footnotes(th,matrix.header_footnotes?.[key]);groups.append(th);
    }
    const headings=el('tr');letterColumns.forEach(([key,en,zh])=>{const th=el('th');th.scope='col';th.append(el('span',zh),el('small',en));footnotes(th,matrix.header_footnotes?.[key]);headings.append(th);});
    head.append(groups,headings);table.append(head);
    const body=el('tbody');matrix.rows.forEach(row=>{
      const tr=el('tr'),th=el('th',row.letter);th.scope='row';tr.append(th);
      letterColumns.forEach(([key])=>{const cell=row.cells[key],td=el('td',undefined,`letter-cell ${cell.state}`);
        td.dataset.column=key;td.append(el('span',cell.state==='blank'?'—':cell.text || '无法辨认'));
        if(cell.state==='user_choice')td.append(el('small','需项目定义','cell-state'));
        if(cell.state==='unreadable')td.append(el('small','来源不清晰','cell-state'));
        footnotes(td,cell.footnotes);tr.append(td);
      });body.append(tr);
    });table.append(body);wrap.append(table);parent.append(wrap,noteDetails);
    const source=details(parent,'查看原表与注释');
    e.assets.filter(a=>a.role==='context' || a.role==='note').forEach(a=>source.append(picture(e,a,true)));
  }
  function show(e, replacement=null) {
    history.replaceState(null,'',`#${encodeURIComponent(e.id)}`);
    const panel=$('detail'); const heading=el('h2',title(e));heading.id='detailTitle';
    panel.replaceChildren(el('span',categories[e.catalog?.category] || types[e.reference_type],'badge'),heading);
    if(replacement){const notice=el('p',undefined,'hint');notice.append(document.createTextNode('已由 ISA 条目替代: '));replacement.replacement_ids.forEach((id,i)=>{const target=catalog.entries.find(r=>r.id===id);if(!target)return;if(i)notice.append(document.createTextNode(' · '));const a=el('a',target.catalog?.short_name || target.concept);a.href=`#${encodeURIComponent(id)}`;a.onclick=event=>{event.preventDefault();show(target);};notice.append(a);});panel.append(notice);}
    panel.append(el('p',full()?e.explanation:e.catalog?.interpretation || e.explanation,'definition'));
    letterTable(panel,e);
    const images=variants(e);
    if(images.length){const grid=el('div',undefined,'variant-grid');images.forEach(a=>grid.append(picture(e,a)));panel.append(grid);}
    else if(full() && e.source.image && !e.letter_matrix)panel.append(picture(e,null,true));
    if(e.text_slots.length){panel.append(el('h3','符号内的文字'));e.text_slots.forEach(s=>{const p=el('p',undefined,'slot');p.append(el('code',s.example_marker),document.createTextNode(` → ${s.meaning} (${s.location})`));panel.append(p);});panel.append(el('p','示例占位符不是图纸中的实际位号。','hint'));}
    const linked=(e.catalog?.supporting_reference_ids || []).map(id=>catalog.entries.find(r=>r.id===id)).filter(Boolean);
    if(linked.length){const d=details(panel,'识读说明');linked.forEach(r=>{d.append(el('h3',r.catalog?.short_name || r.concept),el('p',r.catalog?.interpretation || r.explanation));});}
    if(e.interpretation.length || e.exceptions.length){const d=details(panel,'限定与例外');section(d,'解释规则',e.interpretation);section(d,'例外',e.exceptions);}
    const evidence=details(panel,'来源与技术详情');
    evidence.append(el('p',`${e.id} · v${e.version}`,'identity'));
    if(e.aliases.length)evidence.append(el('p',e.aliases.join(' · ')));
    const mappings=catalog.mappings.filter(m=>m.reference_id===e.id && m.reference_version===e.version);
    const facts=el('dl',undefined,'facts');
    const scope=Object.entries(e.applicability).filter(([k,v])=>k!=='standards' && v && (!Array.isArray(v)||v.length)).map(([k,v])=>`${k}: ${Array.isArray(v)?v.join(', '):v}`);
    const rows=[['标准',standards(e).map(standardKey).join(', ')||'通用'],['背景',scope.join('; ')||'无附加限制'],['来源',[e.source.document,e.source.edition].filter(Boolean).join(' · ')||'未知'],['位置',[e.source.section,e.source.page&&`印刷页码 ${e.source.page}`,e.source.pdf_page&&`PDF 页码 ${e.source.pdf_page}`,e.source.table&&`表 ${e.source.table}`,e.source.item&&`条目 ${e.source.item}`].filter(Boolean).join(' · ')||'未知'],['参考类型',types[e.reference_type]],['知识用途',e.catalog?.role || 'Legacy'],['DEXPI 映射',mappings.length?mappings.map(m=>`${m.target_model} ${m.target_version}: ${m.target}`).join('; '):'尚未映射']];
    rows.forEach(([k,v])=>facts.append(el('dt',k),el('dd',v)));evidence.append(facts);
    const doc=catalog.documents.find(d=>d.id===e.source.document_id);
    if(doc?.scan_note)evidence.append(el('p',`扫描来源： ${doc.scan_note}`,'hint'));
    if(e.source.url && /^https?:\/\//.test(e.source.url)){const link=el('a','查看原始来源');link.href=e.source.url;link.target='_blank';link.rel='noopener noreferrer';evidence.append(link);}
    if(doc?.retrieved_at)evidence.append(el('p',`获取日期： ${doc.retrieved_at.slice(0,10)}`,'hint'));
    evidence.append(el('p',doc?.source_type==='web'?'网站补充参考，未经标准核实，图纸图例优先。':'需匹配声明的标准与版本，图纸明确图例优先。','hint'));
    const originals=details(evidence,'原始插图与注释');
    e.assets.filter(a=>a.role==='context').forEach(a=>originals.append(picture(e,a,true)));
    if(!full()){e.assets.filter(a=>a.role==='variant' && !a.id.startsWith('catalog-')).forEach(a=>originals.append(picture(e,a)));}
    if(e.source.passage)originals.append(el('pre',e.source.passage));
    if(e.table_data?.length)originals.append(el('pre',JSON.stringify(e.table_data,null,2)));
    e.notes.forEach(n=>originals.append(el('p',n.explanation),el('p',`${n.source.passage || ''} · PDF 页码 ${n.source.pdf_page || '?'}`,'hint')));
    e.assets.filter(a=>a.role==='note').forEach(a=>originals.append(picture(e,a,true)));
    if(full() && e.relations.length){const d=details(evidence,'关联条目');e.relations.forEach(r=>{const p=el('p'),a=el('a',r.reference_id);a.href=`#${encodeURIComponent(r.reference_id)}`;a.onclick=event=>{event.preventDefault();const target=catalog.entries.find(e=>e.id===r.reference_id);if(target)show(target);};p.append(a,document.createTextNode(` — ${r.explanation}`));d.append(p);});}
    details(evidence,'图像溯源').append(el('pre',JSON.stringify({source:e.source,assets:e.assets,mappings},null,2)));
    if(!$('detailDialog').open)$('detailDialog').showModal();
    $('detailDialog').scrollTop=0;
  }
  function renderCategories() {
    $('categories').replaceChildren();$('categories').hidden=full() || letters();$('typeLabel').hidden=!full();
    if(full() || letters())return;
    const available=new Set(catalog.entries.filter(e=>e.catalog?.role==='catalog' && (!$('sourceFilter').value || e.source.document_id===$('sourceFilter').value)).map(e=>e.catalog.category));
    for(const [value,label] of [['','全部符号'],...Object.entries(categories).filter(([k])=>available.has(k))]){
      const b=el('button',label);b.type='button';b.setAttribute('aria-pressed',String(category===value));b.onclick=()=>{category=value;render();};$('categories').append(b);
    }
  }
  function render(reset=true) {
    if(!catalog)return;if(reset)visibleCount=40;
    renderCategories();
    document.querySelector('.intro h1').textContent=letters()?'字母含义表':'符号目录';
    document.querySelector('.intro > p:last-child').textContent=letters()?'按字母位置识读，查看各标准定义及原文注释。':'查找符号，点击查看含义与其他画法。';
    const query=$('search').value.trim().toLocaleLowerCase();
    const matches=new Map();
    const entries=catalog.entries.filter(e=>{
      if (!(full() || (letters()?Boolean(e.letter_matrix):e.catalog?.role==='catalog')) || !(full() || letters() || !category || e.catalog.category===category) || !(!full() || !$('typeFilter').value || e.reference_type===$('typeFilter').value) || !(!$('sourceFilter').value || e.source.document_id===$('sourceFilter').value) || !(!$('standardFilter').value || standards(e).some(s=>standardKey(s)===$('standardFilter').value)))return false;
      const match=CatalogSearch.match(e,query,{expanded:$('searchDescriptions').checked,replacementLabels:replacementLabels.get(e.id)});
      if(!match)return false;matches.set(e.id,match);return true;
    });
    entries.sort((a,b)=>matches.get(b.id).score-matches.get(a.id).score || (full() || letters()?a.id.localeCompare(b.id,undefined,{numeric:true}):Object.keys(categories).indexOf(a.catalog.category)-Object.keys(categories).indexOf(b.catalog.category)||title(a).localeCompare(title(b),undefined,{numeric:true})));
    $('libraryStatus').textContent=`${entries.length} ${letters()?'字母表':full()?'参考条目':'符号'} · ${Math.min(visibleCount,entries.length)} 项已显示`;
    $('entries').replaceChildren();
    entries.slice(0,visibleCount).forEach(e=>{const b=el('button',undefined,'entry-card');b.type='button';b.dataset.id=e.id;const a=variants(e)[0];if(a || (full() || letters()) && e.source.image){const img=el('img');img.src=assetUrl(e,a);img.alt=title(e);img.loading='lazy';b.append(img);}else b.append(el('span','Aa','text-reference'));b.append(el('strong',title(e)),el('small',sourceName(e),'hint'));const kind=matches.get(e.id).kind;if(kind==='family' || kind==='description')b.append(el('small',kind==='family'?'类别匹配':'说明或来源注释匹配','match-reason'));b.onclick=()=>show(e);$('entries').append(b);});
    if(entries.length>visibleCount){const more=el('button','显示更多','load-more');more.type='button';more.onclick=()=>{visibleCount+=40;render(false);};$('entries').append(more);}
    if(!entries.length)$('entries').append(el('p','没有匹配条目','empty'));
  }
  for(const [dialog,button] of [['imageDialog','closeImage'],['detailDialog','closeDetail']]){
    $(button).onclick=()=>$(dialog).close();
    $(dialog).addEventListener('click',event=>{if(event.target===$(dialog))$(dialog).close();});
  }
  $('detailDialog').addEventListener('close',()=>history.replaceState(null,'',location.pathname+location.search));
  ['search','searchDescriptions','typeFilter','standardFilter','sourceFilter','viewFilter'].forEach(id=>$(id).addEventListener(id==='search'?'input':'change',()=>{if(id==='sourceFilter')category='';render();}));
  function openHash(){const id=decodeURIComponent(location.hash.slice(1));const replacement=(catalog.replacements || []).find(r=>r.reference_id===id);const entry=catalog.entries.find(e=>e.id===(replacement?.replacement_ids[0] || id));if(entry){if(entry.letter_matrix)$('viewFilter').value='letters';else if(entry.catalog?.role!=='catalog')$('viewFilter').value='full';render();show(entry,replacement);}}
  window.addEventListener('hashchange',()=>{if(catalog)openHash();});
  fetch('/api/context/library').then(r=>{if(!r.ok)throw new Error(`HTTP ${r.status}`);return r.json();}).then(data=>{
    catalog=data;(data.replacements || []).forEach(r=>r.replacement_ids.forEach(id=>replacementLabels.set(id,[...(replacementLabels.get(id) || []),r.source_label])));data.documents.forEach(d=>$('sourceFilter').add(new Option(d.title,d.id)));
    const params=new URLSearchParams(location.search);if(['letters','full','catalog'].includes(params.get('view')))$('viewFilter').value=params.get('view');
    const initialSource=params.get('source');if(initialSource)$('sourceFilter').value=initialSource;
    Object.entries(types).forEach(([value,label])=>$('typeFilter').add(new Option(label,value)));
    [...new Set(data.entries.flatMap(e=>standards(e).map(standardKey)))].sort().forEach(s=>$('standardFilter').add(new Option(s,s)));
    render();openHash();
  }).catch(error=>{$('libraryStatus').textContent=`无法加载： ${error.message}`;$('libraryStatus').classList.add('error');});
})();
