/* Pure catalog search, shared by the browser and offline regression checks. */
((root, factory) => {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.CatalogSearch = api;
})(typeof window === 'undefined' ? null : window, () => {
  'use strict';
  const plurals = new Map(Object.entries({exchangers:'exchanger',pumps:'pump',valves:'valve',
    actuators:'actuator',instruments:'instrument',connectors:'connector',vessels:'vessel',
    signals:'signal',lines:'line',tubes:'tube',boilers:'boiler',condensers:'condenser',
    heaters:'heater',towers:'tower',filters:'filter',reactors:'reactor',motors:'motor'}));
  const stop = new Set(['and','or','the','a','an','of','with']);
  function words(value) {
    return (String(value || '').normalize('NFKC').toLowerCase().match(/[\p{L}\p{N}]+/gu) || [])
      .map(w=>plurals.get(w) || w).filter(w=>!stop.has(w));
  }
  function quality(field, query) {
    const tokens=words(field);
    if (!tokens.length) return 0;
    const text=tokens.join(' '), phrase=query.join(' ');
    if (text===phrase) return 30;
    // Word boundaries keep "gas" out of "gasoline"; CJK needs substring matching.
    const matches=(q,t,prefix=false)=>t===q || (/\p{Script=Han}/u.test(q) && t.includes(q))
      || (prefix && q.length>=4 && t.startsWith(q));
    if (query.every(q=>tokens.some(t=>matches(q,t)))) {
      return (` ${text} `).includes(` ${phrase} `)?20:10;
    }
    // Only the final word may be unfinished while the user types.
    return query.every((q,i)=>tokens.some(t=>matches(q,t,i===query.length-1)))?5:0;
  }
  function match(entry, input, options={}) {
    const raw=String(input || '').trim();
    if (!raw) return {score:0,kind:'all'};
    if (raw.toLowerCase()===entry.id.toLowerCase()) return {score:500,kind:'id'};
    const query=words(raw);
    if (!query.length) return null;
    const metadata=entry.catalog || {};
    const fields=[
      ['name',300,[metadata.short_name,entry.concept]],
      ['alias',250,[...(entry.aliases || []),...(options.replacementLabels || [])]],
      ['family',150,metadata.search_terms || []],
    ];
    if (options.expanded) fields.push(['description',50,[
      metadata.interpretation,entry.explanation,entry.id,
      ...(entry.tags || []),...(entry.interpretation || []),...(entry.exceptions || []),
      entry.source?.passage,entry.source?.section,entry.source?.document,
      ...(entry.notes || []).flatMap(n=>[n.explanation,n.source?.passage]),
      ...(entry.text_slots || []).flatMap(s=>[s.meaning,s.example_marker]),
    ]]);
    for (const [kind,base,values] of fields) {
      const best=Math.max(0,...values.map(v=>quality(v,query)));
      if (best) return {score:base+best,kind};
    }
    return null;
  }
  return {match};
});
