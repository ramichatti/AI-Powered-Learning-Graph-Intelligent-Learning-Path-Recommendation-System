(function () {
  const { api, fmt, esc, showError, setLoading } = JL;
  const ICONS = {
    briefcase: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="7" width="18" height="13" rx="2"/><path d="M9 7V5a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2M3 13h18"/></svg>',
    tag: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M20.6 13.4 13.4 20.6a2 2 0 0 1-2.8 0L3 13V3h10l7.6 7.6a2 2 0 0 1 0 2.8Z"/></svg>',
    spark: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M5.6 18.4l2.8-2.8M15.6 8.4l2.8-2.8"/></svg>',
    clusters: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="7" cy="8" r="3"/><circle cx="17" cy="7" r="2.5"/><circle cx="12" cy="17" r="3.5"/></svg>',
    building: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="4" y="3" width="16" height="18" rx="1.5"/><path d="M9 21v-4h6v4"/></svg>',
    globe: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/></svg>',
  };
  const kpi = (ic, label, value, foot) => `<div class="card kpi"><div class="label">${ICONS[ic]}${esc(label)}</div>
      <div class="value">${value}</div><div class="foot">${foot}</div></div>`;

  api("/api/summary").then((d) => {
    const a = d.analytics, t = a.totals;
    JL.$("#kpis").innerHTML =
      kpi("briefcase", "Job ads", fmt.int(t.postings), `cleaned from ${fmt.int(t.raw_postings)} raw ads`) +
      kpi("tag", "Job titles", t.job_categories, "categories the classifier knows") +
      kpi("spark", "Skills", fmt.int(t.distinct_skills), `about ${a.skills_per_posting.median} per ad`) +
      kpi("clusters", "Job families", d.clustering_metrics ? d.clustering_metrics.k : "–", "found automatically") +
      kpi("building", "Companies", fmt.compact(t.companies), "hiring") +
      kpi("globe", "Countries", t.countries, `${fmt.pct(t.remote_share)} of ads are remote`);

    const sk = a.top_skills.slice(0, 15);
    setLoading(JL.$("#c-skills"), false);
    Charts.hbar("#c-skills", sk.map((s) => ({ label: s.name, value: s.share, tip: [["Share of postings", fmt.pct(s.share)], ["Postings", fmt.int(s.count)], ["Category", s.category]] })),
      { format: (v) => fmt.pct(v, 0), axisFormat: d3.format(".0%"), xLabel: "% of ads" });

    setLoading(JL.$("#c-cats"), false);
    Charts.hbar("#c-cats", a.category_distribution.map((c) => ({ label: c.name, value: c.count, tip: [["Postings", fmt.int(c.count)], ["Share", fmt.pct(c.count / t.postings)]] })),
      { xLabel: "Ads", onClick: (c) => (location.href = "/analytics#cat=" + encodeURIComponent(c.label)) });

    setLoading(JL.$("#c-monthly"), false);
    Charts.line("#c-monthly", a.monthly.map((m) => ({ x: new Date(m.month + "-01T00:00:00"), y: m.count })), { time: true, yLabel: "Postings", yName: "Postings" });
    setLoading(JL.$("#c-countries"), false);
    Charts.hbar("#c-countries", a.top_countries.slice(0, 10).map((c) => ({ label: c.name, value: c.count })), { xLabel: "Postings" });

    const cl = JL.$("#c-clusters"); setLoading(cl, false);
    if (d.clusters) {
      cl.className = "grid " + (d.clusters.length % 3 === 0 ? "g-3" : "g-4"); cl.style.gap = "12px";
      cl.innerHTML = d.clusters.map((c) => `<a class="cluster-card" style="text-decoration:none;color:inherit" href="/clustering#c=${c.cluster}">
        <div class="h"><span style="width:10px;height:10px;border-radius:3px;display:inline-block;flex:none;background:${Charts.clusterColor(c.cluster)}"></span>${esc(c.top_categories[0] ? c.top_categories[0].name : "Family " + c.cluster)}<span class="sz">${fmt.int(c.size)} ads</span></div>
        <div class="l">${esc(c.label)}</div>
        <div class="chips">${c.top_skills.map((s) => `<span class="chip">${esc(s.name)}</span>`).join("")}</div></a>`).join("");
    } else showError(cl, { message: d.clusters_error, kind: "artifact" });

    const mc = JL.$("#model-cards"); setLoading(mc, false);
    const c = d.classification, cm = d.clustering_metrics, se = a.semantic_eval, sx = a.skill_extraction_eval;
    const row = (name, v, sub) => `<div style="padding:10px 0;border-bottom:1px solid var(--border)"><div style="display:flex;justify-content:space-between;gap:8px"><span style="font-weight:600">${name}</span><span class="num" style="font-weight:700">${v}</span></div><div class="hint">${sub}</div></div>`;
    mc.innerHTML = (c ? row("Classifier accuracy", fmt.pct(c.accuracy), `TF-IDF + LogReg (title + skills), macro-F1 ${fmt.num(c.f1_macro)}, ${fmt.int(c.test)} test ads`) +
      row("Skills-only classifier", fmt.pct(c.skills_only_accuracy), `macro-F1 ${fmt.num(c.skills_only_f1_macro)}: profession family, not seniority`) : "") +
      row("Semantic neighbours", fmt.pct(se["label_agreement@10_semantic"]), `same category among DistilBERT top-10 (random ${fmt.pct(se.random_baseline)})`) +
      (cm ? row("Cluster silhouette", fmt.num(cm.silhouette), `cosine, K=${cm.k}; NMI vs categories ${fmt.num(cm.nmi)}`) : "") +
      row("Skill extraction precision", fmt.pct(sx.precision_vs_tags), `on ${fmt.int(sx.extracted_mentions)} title mentions vs dataset tags`);
  }).catch((e) => { showError(JL.$("#dash-error"), e); JL.$$(".loading").forEach((n) => setLoading(n, false)); });
})();
