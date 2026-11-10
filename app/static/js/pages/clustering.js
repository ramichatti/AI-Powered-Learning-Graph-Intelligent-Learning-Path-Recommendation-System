(function () {
  const { api, fmt, esc, showError, setLoading, $ } = JL;
  let D = null, selected = null, page = 1, query = "";

  api("/api/clusters").then((d) => {
    D = d; D.labels = Object.fromEntries(d.profiles.map((p) => [p.cluster, p.label]));
    setLoading($("#scatter"), false);
    $("#kpis").innerHTML = [
      ["Clusters (K)", d.k, `selected by silhouette; elbow at K=${d.k_elbow}`],
      ["Silhouette (cosine)", fmt.num(d.final_silhouette), "higher = better separated"],
      ["NMI vs categories", fmt.num(d.nmi), "0 = unrelated, 1 = identical"],
      ["ARI vs categories", fmt.num(d.ari), "chance-corrected agreement"],
      ["Embedded ads", fmt.int(d.n_total), `${fmt.int(d.n_points)} plotted on the map`],
    ].map(([l, v, f]) => `<div class="card kpi"><div class="label">${l}</div><div class="value">${v}</div><div class="foot">${f}</div></div>`).join("");
    $("#sc-note").textContent = "This flat map is a simplification: families that overlap here can still be well separated in the full model.";
    $("#sil-sub").textContent = d.selection_rule;
    drawScatter(); drawCards();
    Charts.line("#elbow", d.k_range.map((k, i) => ({ x: k, y: d.inertia[i] })), { zero: false, xLabel: "Number of clusters K", yLabel: "Inertia", xName: "K", yName: "Inertia", markX: d.k_elbow, markLabel: `elbow K=${d.k_elbow}`, format: (v) => v.toFixed(1), yFormat: d3.format(".0f") });
    Charts.vbar("#sil", d.k_range.map((k, i) => ({ label: String(k), value: d.silhouette[i], color: k === d.k ? JL.series(0) : JL.cssVar("--seq-1"), tipTitle: `K = ${k}` })),
      { format: (v) => v.toFixed(3), axisFormat: d3.format(".2f"), xLabel: "Number of clusters K", yLabel: "Silhouette", valueName: "Silhouette" });
    drawComposition();
    const m = location.hash.match(/c=(\d+)/); select(m ? +m[1] : 0);
  }).catch((e) => { setLoading($("#scatter"), false); showError($("#err"), e); });

  function drawScatter() {
    Charts.legend("#legend", D.profiles.map((p) => ({ label: `Family ${p.cluster} · ${Charts.truncate(p.label, 200, 12.5)}`, title: p.label, color: Charts.clusterColor(p.cluster),
      symbol: Charts.clusterSymbol(p.cluster), on: selected === p.cluster, dim: selected != null && selected !== p.cluster })),
      { onClick: (it, i) => select(D.profiles[i].cluster === selected ? null : D.profiles[i].cluster) });
    Charts.scatter("#scatter", D, { height: 540, selected, onSelect: (c) => select(c === selected ? null : c),
      xLabel: `PC1 (${fmt.pct(D.explained_variance[0])})`, yLabel: `PC2 (${fmt.pct(D.explained_variance[1])})` });
  }
  function drawCards() {
    $("#cards").innerHTML = D.profiles.map((p) => `<div class="cluster-card ${selected === p.cluster ? "on" : ""}" data-c="${p.cluster}">
      <div class="h"><svg width="14" height="14" viewBox="-7 -7 14 14"><path d="${d3.symbol().type(Charts.clusterSymbol(p.cluster)).size(80)()}" fill="${Charts.clusterColor(p.cluster)}"/></svg>Family ${p.cluster}<span class="sz">${fmt.int(p.size)} ads</span></div>
      <div class="l">${esc(p.label)}</div><div class="chips">${p.top_skills.slice(0, 5).map((s) => `<span class="chip">${esc(s.name)} ${fmt.pct(s.share, 0)}</span>`).join("")}</div></div>`).join("");
    JL.$$(".cluster-card").forEach((n) => n.addEventListener("click", () => select(+n.dataset.c)));
  }
  function drawComposition() {
    const cats = [...new Set(D.profiles.flatMap((p) => p.top_categories.map((c) => c.name)))].sort();
    Charts.heatmap("#ct", D.profiles.map((p) => `C${p.cluster} · ${p.label}`), cats,
      D.profiles.map((p) => cats.map((c) => (p.top_categories.find((x) => x.name === c) || { share: 0 }).share)),
      { max: 1, colName: "Category", valueName: "Share of cluster", maxRowLabel: 330, cellFormat: (v) => (v >= 0.005 ? Math.round(v * 100) : "") });
  }
  $("#all").addEventListener("click", () => select(null));
  $("#reset").addEventListener("click", () => { const el = $("#scatter"); el._resetZoom && el._resetZoom(); });

  function select(c) {
    selected = c; page = 1; query = "";
    history.replaceState(null, "", c == null ? location.pathname : `#c=${c}`);
    drawScatter(); drawCards();
    if (c == null) { $("#d-title").textContent = "All families"; $("#d-sub").textContent = "Select a family to see its ads and skills"; $("#detail").innerHTML = ""; return; }
    loadDetail();
  }
  async function loadDetail() {
    const c = selected; setLoading($("#detail-card"), true);
    try {
      const d = await api(`/api/clusters/${c}?page=${page}&q=${encodeURIComponent(query)}`);
      const p = d.profile;
      $("#d-title").innerHTML = `Family ${c} <span class="badge">${fmt.int(p.size)} ads</span>`;
      $("#d-sub").textContent = p.label;
      $("#detail").innerHTML = `<div class="grid g-2" style="gap:16px">
          <div><div class="hint" style="font-weight:600;margin-bottom:4px">Most frequent skills</div><div id="d-skills"></div></div>
          <div><div class="hint" style="font-weight:600;margin-bottom:4px">Job titles</div><div id="d-cats"></div>
            <div class="hint" style="font-weight:600;margin:10px 0 6px">Skills that set it apart</div>
            <div class="chips">${p.distinctive_skills.map((s) => `<span class="chip accent">${esc(s.name)} ×${s.lift.toFixed(1)}</span>`).join("") || "–"}</div>
            <div class="hint" style="font-weight:600;margin:10px 0 6px">Most common ad titles</div>
            <div class="chips">${p.top_titles.map((t) => `<span class="chip">${esc(t.name)} (${t.count})</span>`).join("")}</div></div></div>
        <div class="field-row mt"><input type="search" id="d-q" placeholder="Filter these ads by title or skill…" value="${esc(query)}" style="max-width:340px">
          <span class="hint">${fmt.int(d.n_matching)} ads</span></div>
        <div class="table-wrap mt-s"><table class="data"><thead><tr><th>Title</th><th>Company</th><th>Category</th><th>Skills</th></tr></thead><tbody>
        ${d.ads.map((a) => `<tr><td>${esc(a.title)}</td><td>${esc(a.company)}</td><td><span class="badge">${esc(a.category)}</span></td><td class="hint">${esc(a.skills.slice(0, 8).join(", "))}${a.skills.length > 8 ? "…" : ""}</td></tr>`).join("") || `<tr><td colspan="4">${JL.empty("No ad matches the filter.")}</td></tr>`}
        </tbody></table></div>
        <div class="pager"><button class="btn btn-ghost btn-sm" id="prev" ${d.page <= 1 ? "disabled" : ""}>Previous</button>
          <span>Page ${d.page} / ${d.pages}</span><button class="btn btn-ghost btn-sm" id="next" ${d.page >= d.pages ? "disabled" : ""}>Next</button></div>`;
      Charts.hbar("#d-skills", p.top_skills.map((s) => ({ label: s.name, value: s.share })), { format: (v) => fmt.pct(v, 0), axisFormat: d3.format(".0%"), colorFn: () => Charts.clusterColor(c), max: 1, rowHeight: 23 });
      Charts.hbar("#d-cats", p.top_categories.map((s) => ({ label: s.name, value: s.share })), { format: (v) => fmt.pct(v, 0), axisFormat: d3.format(".0%"), max: 1, rowHeight: 23, colorFn: () => JL.cssVar("--seq-3") });
      $("#prev").onclick = () => { page--; loadDetail(); };
      $("#next").onclick = () => { page++; loadDetail(); };
      if (query) { const inp = $("#d-q"); inp.focus(); inp.setSelectionRange(query.length, query.length); }
      let t; $("#d-q").addEventListener("input", (e) => { clearTimeout(t); t = setTimeout(() => { query = e.target.value; page = 1; loadDetail(); }, 350); });
    } catch (e) { showError($("#detail"), e); }
    finally { setLoading($("#detail-card"), false); }
  }
})();
