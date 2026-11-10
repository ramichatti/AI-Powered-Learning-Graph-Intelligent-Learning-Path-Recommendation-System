(function () {
  const { api, fmt, esc, showError, setLoading, $ } = JL;
  api("/api/analytics").then((d) => {
    JL.$$(".loading").forEach((n) => setLoading(n, false));
    const a = d.analytics, sbc = a.skills_by_category;
    // top skills with category filter
    const cats = [...new Set(a.top_skills.map((s) => s.category))];
    $("#skill-cat").innerHTML += cats.map((c) => `<option>${esc(c)}</option>`).join("");
    const drawTop = () => {
      const c = $("#skill-cat").value;
      const rows = a.top_skills.filter((s) => !c || s.category === c).slice(0, 25);
      Charts.hbar("#c-top", rows.map((s) => ({ label: s.name, value: s.share, tip: [["Share of postings", fmt.pct(s.share)], ["Postings", fmt.int(s.count)], ["Category", s.category]] })),
        { format: (v) => fmt.pct(v, 1), axisFormat: d3.format(".0%"), xLabel: "% of postings", rowHeight: 22 });
    };
    $("#skill-cat").addEventListener("change", drawTop); drawTop();

    // per-job top skills
    const jobs = sbc.jobs;
    $("#job-sel").innerHTML = jobs.map((j) => `<option>${esc(j)}</option>`).join("");
    const hash = decodeURIComponent((location.hash.match(/cat=([^&]+)/) || [])[1] || "");
    if (jobs.includes(hash)) $("#job-sel").value = hash;
    const drawJob = () => {
      const j = $("#job-sel").value;
      const rows = sbc.skills.map((s, i) => ({ label: s, value: sbc.share[j][i] })).sort((x, y) => y.value - x.value).slice(0, 15);
      Charts.hbar("#c-job", rows, { format: (v) => fmt.pct(v, 0), axisFormat: d3.format(".0%"), xLabel: `% of ${j} postings`, max: 1, rowHeight: 24, colorFn: () => JL.series(2) });
    };
    $("#job-sel").addEventListener("change", drawJob); drawJob();

    Charts.heatmap("#c-heat", sbc.skills, jobs, sbc.skills.map((s, i) => jobs.map((j) => sbc.share[j][i])),
      { colName: "Job category", valueName: "Share of postings", cellHeight: 25, maxRowLabel: 140 });

    const prof = d.clusters, skc = d.clustering.skills_by_cluster;
    Charts.heatmap("#c-skcl", skc.skills, prof.map((p) => `C${p.cluster}`), skc.skills.map((s, i) => skc.share.map((col) => col[i])),
      { colName: "Cluster", valueName: "Share of the cluster's ads", cellHeight: 24, maxRowLabel: 120 });

    const mix = a.skill_category_mix;
    Charts.heatmap("#c-mix", jobs, mix.categories, jobs.map((j) => mix.jobs[j]), { colName: "Skill category", valueName: "Share of skill mentions", cellHeight: 26, maxRowLabel: 160 });

    Charts.ranges("#c-sal", a.salary_by_category.map((s) => ({ label: s.name, ...s })), { xLabel: "USD per year" });
    Charts.hbar("#c-cat", a.category_distribution.map((c) => ({ label: c.name, value: c.count })), { xLabel: "Postings" });
    Charts.hbar("#c-csize", prof.map((p) => ({ label: `C${p.cluster} · ${p.label}`, value: p.size, color: Charts.clusterColor(p.cluster) })),
      { xLabel: "Ads", maxLabelWidth: 320, onClick: (r) => (location.href = "/clustering#c=" + r.label.match(/^C(\d+)/)[1]) });

    const models = d.classification.models;
    const short = (n) => n.replace("TF-IDF + LogReg ", "").replace(" - MAIN", " (main)").replace("Majority-class baseline", "Majority baseline");
    const mk = document.createElement("div"); mk.className = "mb"; JL.$("#c-metrics").before(mk);
    Charts.legend(mk, models.map((m, i) => ({ label: short(m.model), color: JL.series([3, 1, 0][i]) })));
    Charts.groupedHbar("#c-metrics", ["accuracy", "precision_macro", "recall_macro", "f1_macro"].map((k) => ({
      label: { accuracy: "Accuracy", precision_macro: "Precision (macro)", recall_macro: "Recall (macro)", f1_macro: "F1 (macro)" }[k],
      ...Object.fromEntries(models.map((m, i) => ["m" + i, m[k]])) })),
      models.map((m, i) => ({ key: "m" + i, name: short(m.model), slot: [3, 1, 0][i] })), { max: 1, format: (v) => v.toFixed(2), xLabel: "Score", rowHeight: 50 });
    $("#t-metrics").innerHTML = `<thead><tr><th>Model</th><th class="r">Accuracy</th><th class="r">Precision</th><th class="r">Recall</th><th class="r">F1 (macro)</th><th class="r">F1 (weighted)</th></tr></thead><tbody>` +
      models.map((m, i) => `<tr><td><span class="chip"><span class="sw" style="background:${JL.series([3, 1, 0][i])}"></span>${esc(short(m.model))}</span></td><td class="r">${fmt.num(m.accuracy)}</td><td class="r">${fmt.num(m.precision_macro)}</td><td class="r">${fmt.num(m.recall_macro)}</td><td class="r">${fmt.num(m.f1_macro)}</td><td class="r">${fmt.num(m.f1_weighted)}</td></tr>`).join("") + "</tbody>";

    const c = d.clustering;
    Charts.line("#c-elbow", c.k_range.map((k, i) => ({ x: k, y: c.inertia[i] })), { zero: false, xLabel: "K", yLabel: "Inertia", xName: "K", yName: "Inertia", markX: c.k_elbow, markLabel: `elbow K=${c.k_elbow}`, format: (v) => v.toFixed(1), yFormat: d3.format(".0f") });
    Charts.vbar("#c-sil", c.k_range.map((k, i) => ({ label: String(k), value: c.silhouette[i], color: k === c.k_selected ? JL.series(0) : JL.cssVar("--seq-1"), tipTitle: `K = ${k}` })),
      { format: (v) => v.toFixed(3), axisFormat: d3.format(".2f"), xLabel: "K", yLabel: "Silhouette", valueName: "Silhouette" });

    Charts.line("#c-month", a.monthly.map((m) => ({ x: new Date(m.month + "-01T00:00:00"), y: m.count })), { time: true, yLabel: "Postings", yName: "Postings" });
    Charts.hbar("#c-sched", a.schedule_types.map((s) => ({ label: s.name, value: s.count })), { xLabel: "Postings" });
    const t = a.totals;
    $("#facts").innerHTML = `<b>${fmt.pct(t.remote_share)}</b> of ads are remote. A typical ad lists <b>${a.skills_per_posting.median}</b> skills.
      Ads come from <b>${fmt.int(t.companies)}</b> companies in <b>${t.countries}</b> countries.`;
  }).catch((e) => { JL.$$(".loading").forEach((n) => setLoading(n, false)); showError($("#err"), e); });
})();
