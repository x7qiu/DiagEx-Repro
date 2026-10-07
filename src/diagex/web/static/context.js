/* Context is confirmed once per project; editing a draft never changes a saved run. */
(() => {
  const fields = [
    ['industry', '行业'], ['unit_types', '装置类型 （用逗号分隔）'],
    ['company', '公司'], ['project', '项目'],
    ['standards', '明确采用标准 （每行一项：标准名称 | 版本）'],
    ['conventions', '项目约定 （每行一项）'],
  ];
  let profiles = [], confirmed = null, dirty = false, evidence = [];
  let library = null, referenceRevision = 0, referenceTimer;
  const error = (e) => { $('contextError').textContent = e.message; };
  const clearError = () => { $('contextError').textContent = ''; };
  const post = (url, body) => api(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  function markDirty() { dirty = true; $('contextStatus').textContent = '修改后请确认'; updateReferences(); }
  function referenceCard(entry, active) {
    const card = document.createElement('div'); card.className = 'context-reference';
    const asset = entry.assets?.find(a=>a.role === 'variant');
    if (asset || entry.source.image) {
      const image = document.createElement('img'); image.alt = entry.concept; image.loading = 'lazy';
      image.src = asset ? '/api/context/asset?' + new URLSearchParams({id:entry.id,asset:asset.id})
        : '/api/context/illustration?' + new URLSearchParams({id:entry.id});
      card.append(image);
    }
    const text = document.createElement('div'), link = document.createElement('a');
    link.href = '/knowledge#' + encodeURIComponent(entry.id); link.target = '_blank'; link.rel = 'noopener';
    link.textContent = entry.concept + ' ↗';
    const meaning = document.createElement('p'); meaning.textContent = entry.explanation;
    const scope = document.createElement('small');
    const standards = (entry.applicability.standards || []).map(s=>s.name+(s.edition ? ' · '+s.edition : '')).join(', ');
    scope.textContent = (active ? '可检索' : '当前未应用') + ' · ' + (standards || '背景参考');
    text.append(link, meaning, scope); card.append(text); return card;
  }
  function renderReferences(ids, status) {
    if (!library) return;
    const eligible = new Set(ids);
    const active = library.entries.filter(e=>eligible.has(e.id)), other = library.entries.filter(e=>!eligible.has(e.id));
    $('applicableReferenceStatus').textContent = status;
    $('applicableReferences').replaceChildren(...active.map(e=>referenceCard(e,true)));
    $('otherReferences').replaceChildren(...other.map(e=>referenceCard(e,false)));
    $('otherReferencesDetails').hidden = !other.length;
    $('otherReferencesSummary').textContent = `当前未应用（${other.length} 项，仅供浏览）`;
  }
  function updateReferences() {
    const revision = ++referenceRevision;
    clearTimeout(referenceTimer);
    const mode = $('knowledgeMode').value;
    $('knowledgeSourceStatus').textContent = mode === 'off' ? '参考知识已关闭，所选来源不使用' : $('knowledgeSources').querySelector('input:checked') ? '仅使用所选来源及共享通用说明' : '未选择知识库，使用图纸图例及局部依据';
    $('referencePreviewPageField').classList.toggle('hidden', mode !== 'profile');
    if (mode === 'off') {
      renderReferences([], '未启用参考知识，仍可浏览知识库。'); return;
    }
    if (mode === 'profile' && (!confirmed || dirty)) {
      renderReferences([], '请确认并保存项目背景后查看适用参考。'); return;
    }
    renderReferences([], '正在检查适用范围…');
    referenceTimer = setTimeout(async () => {
      try {
        const page = mode === 'profile' ? Number($('referencePreviewPage').value) : 1;
        if (!Number.isInteger(page) || page < 1) throw new Error('页码必须为正整数');
        const result = await post('/api/context/applicable', {...window.drawingContextPayload(), page});
        if (revision !== referenceRevision) return;
        renderReferences(result.reference_ids, `${result.reference_ids.length} 条适用参考${mode === 'profile' ? ` · 第 ${page} 页 · ${confirmed.name} v${confirmed.version}` : ' · 已选参考来源'}`);
      } catch (e) {
        if (revision === referenceRevision) renderReferences([], e.message);
      }
    }, 180);
  }
  for (const [field, label] of fields) {
    for (const prefix of ['projectContext_', 'sheetContext_']) {
      const wrapper = document.createElement('label'); wrapper.className = ['industry', 'unit_types', 'company', 'project'].includes(field) ? 'context-short' : 'wide';
      if (prefix === 'sheetContext_') {
        const check = document.createElement('input'); check.type = 'checkbox'; check.id = prefix + field + '_enabled';
        wrapper.append(check);
      }
      const span = document.createElement('span'); span.textContent = label;
      const input = document.createElement(['standards','conventions'].includes(field) ? 'textarea' : 'input');
      input.id = prefix + field; input.maxLength = 8000;
      if (prefix === 'projectContext_') input.addEventListener('input', markDirty);
      else {
        input.addEventListener('input', updateReferences);
        wrapper.querySelector('input[type=checkbox]').addEventListener('change', updateReferences);
      }
      wrapper.append(span, input);
      $(prefix === 'projectContext_' ? 'contextKnownFields' : 'contextOverrideFields').append(wrapper);
    }
  }
  function read(prefix, overrides = false) {
    const result = {};
    for (const [field] of fields) {
      if (overrides && !$(prefix + field + '_enabled').checked) continue;
      const value = $(prefix + field).value.trim();
      result[field] = field === 'unit_types' ? value.split(/[,，;；]/).map(v=>v.trim()).filter(Boolean)
        : field === 'conventions' ? value.split('\n').map(v=>v.trim()).filter(Boolean)
        : field === 'standards' ? value.split('\n').filter(v=>v.trim()).map(v=> {const [name, edition] = v.split('|').map(s=>s.trim()); return {name, edition:edition || null};})
        : value || null;
    }
    return result;
  }
  function write(field, value) {
    $('projectContext_' + field).value = field === 'standards' ? value.map(s=>s.name+(s.edition ? ' | '+s.edition : '')).join('\n')
      : Array.isArray(value) ? value.join(field === 'unit_types' ? ', ' : '\n') : value || '';
  }
  async function refresh(selected = '') {
    profiles = (await api('/api/context/profiles')).profiles;
    $('contextProfile').replaceChildren(new Option('新建', ''), ...profiles.map(p=>new Option(`${p.name} · v${p.version}`, p.id)));
    $('contextProfile').value = selected;
  }
  $('knowledgeMode').addEventListener('change', () => {
    $('contextProfileFields').classList.toggle('hidden', $('knowledgeMode').value !== 'profile');
    window.maybeSuggestDrawingContext?.();
    updateReferences();
  });
  $('contextProfile').addEventListener('change', () => {
    confirmed = profiles.find(p=>p.id === $('contextProfile').value) || null;
    $('contextName').value = confirmed?.name || '';
    for (const [field] of fields) write(field, confirmed?.context[field] || (['standards','unit_types','conventions'].includes(field) ? [] : null));
    for (const option of $('contextReferences').options) option.selected = confirmed?.reference_ids.includes(option.value) || false;
    evidence = confirmed?.confirmation_evidence || [];
    $('contextSuggestions').replaceChildren(); dirty = false;
    $('contextStatus').textContent = confirmed ? `v${confirmed.version} · 已确认` : '';
    updateReferences();
  });
  $('contextName').addEventListener('input', markDirty);
  $('contextReferences').addEventListener('change', markDirty);
  $('suggestContext').addEventListener('click', async () => {
    clearError();
    try {
      if (!state.upload) throw new Error('请先上传文档');
      const result = await post('/api/context/suggestions', {upload_id:state.upload.id});
      const root = $('contextSuggestions'); root.replaceChildren();
      const notice = document.createElement('p'); notice.textContent = result.notice; root.append(notice);
      const missing = read('projectContext_');
      for (const suggestion of result.suggestions) {
        if (missing[suggestion.field] && (!Array.isArray(missing[suggestion.field]) || missing[suggestion.field].length)) continue;
        const row = document.createElement('div');
        const quote = document.createElement('p'); quote.textContent = `${suggestion.source}，第 ${suggestion.page} 页: “${suggestion.passage}”`;
        const use = document.createElement('button'); use.type = 'button'; use.textContent = '采纳到草稿';
        use.addEventListener('click', () => {write(suggestion.field, suggestion.value); evidence.push(suggestion); markDirty(); use.disabled = true;});
        row.append(quote,use); root.append(row);
      }
    } catch(e) { error(e); }
  });
  $('confirmContext').addEventListener('click', async () => {
    clearError();
    try {
      const result = await post('/api/context/profiles', {
        ...(confirmed ? {id:confirmed.id, expected_version:confirmed.version} : {}),
        name:$('contextName').value.trim(), context:read('projectContext_'), confirmed:true,
        reference_ids:[...$('contextReferences').selectedOptions].map(o=>o.value), confirmation_evidence:evidence.filter(s=>JSON.stringify(read('projectContext_')[s.field]) === JSON.stringify(s.value)).map(s=>({...s,status:'confirmed'})),
      });
      confirmed = result.profile; dirty = false; await refresh(confirmed.id);
      $('contextStatus').textContent = `Saved & confirmed version ${confirmed.version} / 已保存确认`;
      updateReferences();
    } catch(e) { error(e); }
  });
  window.drawingContextPayload = () => {
    const mode = $('knowledgeMode').value;
    const knowledge_sources = [...$('knowledgeSources').querySelectorAll('input:checked')].map(input=>input.value);
    if (mode !== 'profile') return {knowledge_mode:mode, knowledge_sources};
    if (!confirmed || dirty) throw new Error('请先确认并保存项目背景');
    const override = read('sheetContext_', true), pages = [];
    const text = $('contextPages').value.trim();
    if (text) for (const part of text.split(/[,，]/)) {
      if (!/^\s*\d+(?:\s*-\s*\d+)?\s*$/.test(part)) throw new Error('Pages must be numbers or ranges, e.g. 1,3-5');
      const [a,b=a] = part.split('-').map(Number);
      if (a<1 || b<a || b-a>1000) throw new Error('Invalid page range');
      for(let n=a;n<=b;n++) pages.push(n);
    }
    return {knowledge_mode:mode, knowledge_sources, context_profile_id:confirmed.id, context_profile_version:confirmed.version,
      drawing_overrides:Object.keys(override).length ? [{pages:[...new Set(pages)], context:override}] : []};
  };
  const suggestedUploads = new Set();
  $('knowledgeSources').addEventListener('change', () => {
    if ($('knowledgeMode').value === 'off') { $('knowledgeMode').value = 'general'; $('knowledgeMode').dispatchEvent(new Event('change')); }
    else updateReferences();
  });
  $('contextPages').addEventListener('input', updateReferences);
  $('referencePreviewPage').addEventListener('input', updateReferences);
  window.maybeSuggestDrawingContext = () => {
    if ($('knowledgeMode').value !== 'profile' || confirmed || !state.upload || suggestedUploads.has(state.upload.id)) return;
    suggestedUploads.add(state.upload.id); $('suggestContext').click();
  };
  Promise.all([refresh(), api('/api/context/library').then(value => {
    library = value;
    $('contextReferences').replaceChildren(...library.entries.map(e=>new Option(e.concept,e.id)));
  })]).then(updateReferences).catch(e => {error(e); $('applicableReferenceStatus').textContent = '无法加载参考条目';});
})();
