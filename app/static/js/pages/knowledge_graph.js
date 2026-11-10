(function () {
  const { api, fmt, esc, showError, setLoading, $, $$, cssVar, series } = JL;
  if (window.cytoscape && window.cytoscapeFcose) cytoscape.use(cytoscapeFcose);
  const TYPE = { job: { name: "Job category", shape: "round-rectangle", slot: 0 }, company: { name: "Company", shape: "diamond", slot: 1 }, skill: { name: "Skill", shape: "ellipse", slot: 2 } };
  const EDGE = { offers: { name: "company offers job", style: "solid" }, requires: { name: "job requires skill", style: "solid" },
                 uses: { name: "company uses skill", style: "dashed" }, "co-occurs": { name: "skill co-occurs with skill", style: "dotted" } };
  let cy = null, current = null, colorMode = "type", hidden = new Set(), mode = "overview";

  // ------------------------------------------------------------------ styles
  function nodeColor(n) {
    if (colorMode === "community" && n.data("community") >= 0) return series(n.data("community"));
    return series(TYPE[n.data("type")].slot);
  }
  function sizeOf(ele, minS, maxS, all) {
    const ofType = all.filter((n) => n.data("type") === ele.data("type"));
    const mx = Math.max(...ofType.map((n) => n.data("count")), 1);
    return minS + (maxS - minS) * Math.sqrt(ele.data("count") / mx);
  }
  function style() {
    return [
      { selector: "node", style: {
          "background-color": (n) => nodeColor(n), shape: (n) => TYPE[n.data("type")].shape,
          width: (n) => (n.data("type") === "job" ? Math.max(n.data("_s"), n.data("_lw")) : n.data("_s")),
          height: (n) => (n.data("type") === "job" ? 30 + n.data("_s") * 0.12 : n.data("_s")),
          "border-width": 2, "border-color": cssVar("--surface"),
          label: "data(label)", "font-size": (n) => (n.data("type") === "job" ? 12 : 10.5), "font-weight": (n) => (n.data("type") === "job" ? 700 : 500),
          "font-family": "Inter, Segoe UI, system-ui, sans-serif", color: cssVar("--text"),
          "text-valign": (n) => (n.data("type") === "job" ? "center" : "bottom"), "text-margin-y": (n) => (n.data("type") === "job" ? 0 : 4),
          "text-wrap": "ellipsis", "text-max-width": (n) => (n.data("type") === "job" ? 220 : 120),
          "text-background-color": cssVar("--surface"), "text-background-opacity": (n) => (n.data("type") === "job" ? 0 : 0.85),
          "text-background-padding": 2, "text-background-shape": "roundrectangle", "min-zoomed-font-size": 5,
          "z-index": (n) => (n.data("type") === "job" ? 30 : 20) } },
      { selector: 'node[type = "job"]', style: { color: "#ffffff", "text-outline-width": 0 } },
      { selector: "edge", style: {
          width: "data(_w)", "line-color": cssVar("--border-strong"), opacity: 0.55, "curve-style": "bezier",
          "line-style": (e) => (EDGE[e.data("type")] || EDGE.requires).style, "z-index": 1 } },
      { selector: 'edge[type = "requires"]', style: { "line-color": series(2), opacity: 0.4 } },
      { selector: 'edge[type = "offers"]', style: { "line-color": series(1), opacity: 0.4 } },
      { selector: 'edge[type = "co-occurs"]', style: { "line-color": series(6), opacity: 0.6 } },
      { selector: ".faded", style: { opacity: 0.08, "text-opacity": 0 } },
      { selector: "edge.hl", style: { opacity: 0.95, width: (e) => Math.max(2, e.data("_w") * 1.4), label: "data(label)", "font-size": 9.5, color: cssVar("--text-2"),
          "text-background-color": cssVar("--surface"), "text-background-opacity": 1, "text-background-padding": 2, "z-index": 50 } },
      { selector: "node.hl", style: { "border-color": cssVar("--text"), "border-width": 3, "z-index": 60 } },
      { selector: ".hidden", style: { display: "none" } },
    ];
  }

  function prepare(elements) {
    const nodes = elements.nodes, edges = elements.edges;
    const maxW = {};
    edges.forEach((e) => { const t = e.data.type; maxW[t] = Math.max(maxW[t] || 0, e.data.weight); });
    edges.forEach((e) => { e.data._w = 0.8 + 4.2 * Math.sqrt(e.data.weight / (maxW[e.data.type] || 1)); });
    const byType = {};
    nodes.forEach((n) => { (byType[n.data.type] = byType[n.data.type] || []).push(n.data.count); });
    const range = { job: [60, 110], company: [18, 44], skill: [14, 44] };
    nodes.forEach((n) => {
      const mx = Math.max(...byType[n.data.type]); const [a, b] = range[n.data.type];
      n.data._s = a + (b - a) * Math.sqrt(n.data.count / mx);
      n.data._lw = Charts.textWidth(n.data.label, 12.5) * 1.08 + 22;   // job boxes always fit their label
    });
    return elements;
  }

  function render(data, opts = {}) {
    current = data;
    const els = prepare(JSON.parse(JSON.stringify(data.elements)));
    if (cy) cy.destroy();
    cy = cytoscape({ container: $("#cy"), elements: els, style: style(), wheelSensitivity: 0.25, minZoom: 0.15, maxZoom: 3.5 });
    applyHidden();
    runLayout(opts.animate);
    bind();
    updateOverlay(data.stats);
    updateCommunities(data.stats);
    legend();
  }
  // Job categories are pinned on a ring (ordered by professional family) so the big job boxes can
  // never overlap; companies and skills are then placed by the force-directed fcose layout.
  const JOB_ORDER = ["Business Analyst", "Data Analyst", "Senior Data Analyst", "Data Scientist", "Senior Data Scientist",
    "Machine Learning Engineer", "Software Engineer", "Cloud Engineer", "Senior Data Engineer", "Data Engineer"];
  function jobConstraints() {
    const jobs = cy.nodes('[type = "job"]:visible').sort((a, b) => JOB_ORDER.indexOf(a.data("label")) - JOB_ORDER.indexOf(b.data("label")));
    if (jobs.length === 1) return [{ nodeId: jobs[0].id(), position: { x: 0, y: 0 } }];
    if (jobs.length < 3) return undefined;
    const R = Math.max(200, jobs.length * 30);
    return jobs.map((n, i) => {
      const a = -Math.PI / 2 + (2 * Math.PI * i) / jobs.length;
      return { nodeId: n.id(), position: { x: R * 1.6 * Math.cos(a), y: R * Math.sin(a) } };
    });
  }
  function runLayout(animate) {
    const n = cy.nodes(":visible").length;
    const fixed = jobConstraints();
    cy.layout({ name: window.cytoscapeFcose ? "fcose" : "cose", quality: "proof", randomize: true, animate: !!animate, animationDuration: 500,
      nodeDimensionsIncludeLabels: true, packComponents: false, fixedNodeConstraint: fixed,
      nodeRepulsion: () => 16000, idealEdgeLength: (e) => (e.data("type") === "requires" ? 110 : 85),
      edgeElasticity: () => 0.3, nodeSeparation: 90, gravity: 0.25, gravityRange: 3.0, numIter: 5000,
      fit: true, padding: n > 60 ? 18 : 32, eles: cy.elements(":visible") }).run();
  }

  function bind() {
    cy.on("mouseover", "node", (ev) => {
      const n = ev.target, nb = n.closedNeighborhood();
      cy.elements().addClass("faded"); nb.removeClass("faded"); nb.edges().addClass("hl"); n.addClass("hl");
      const d = n.data();
      JL.showTip(JL.tipRows(d.label, [["Type", TYPE[d.type].name], [d.type === "skill" ? "Postings listing it" : "Postings", fmt.int(d.count)],
        ["Connections shown", d.degree ?? n.degree()], ...(d.category ? [["Skill category", d.category]] : []),
        ...(d.community >= 0 ? [["Community", "#" + d.community]] : [])]), ev.originalEvent);
      showNode(n);
    });
    cy.on("mousemove", "node", (ev) => JL.moveTip(ev.originalEvent));
    cy.on("mouseout", "node", () => { cy.elements().removeClass("faded hl"); JL.hideTip(); });
    cy.on("mouseover", "edge", (ev) => {
      const e = ev.target; e.addClass("hl");
      JL.showTip(JL.tipRows(`${e.source().data("label")} → ${e.target().data("label")}`, [["Relation", (EDGE[e.data("type")] || {}).name || e.data("type")],
        ["Weight", e.data("type") === "requires" || e.data("type") === "co-occurs" ? fmt.pct(e.data("weight")) : fmt.int(e.data("weight")) + " postings"]]), ev.originalEvent);
    });
    cy.on("mouseout", "edge", (ev) => { ev.target.removeClass("hl"); JL.hideTip(); });
    cy.on("tap", "node", (ev) => { JL.hideTip(); explore(ev.target.data("type"), ev.target.data("label")); });
  }

  function showNode(n) {
    const d = n.data();
    const nb = n.neighborhood("node").sort((a, b) => b.data("count") - a.data("count"));
    $("#node-sub").textContent = `${TYPE[d.type].name} · ${fmt.int(d.count)} postings`;
    $("#node-info").innerHTML = `<div style="font-weight:700;font-size:15px;color:var(--text)">${esc(d.label)}</div>
      ${d.category ? `<div class="hint">${esc(d.category)}</div>` : ""}
      <div class="mt-s">Connected to ${nb.length} nodes shown:</div>
      <div class="chips mt-s">${nb.slice(0, 18).map((m) => `<span class="chip"><span class="sw" style="background:${nodeColor(m)}"></span>${esc(m.data("label"))}</span>`).join("")}</div>
      <div class="hint mt-s">Click the node to open its full neighbourhood.</div>`;
  }

  function updateOverlay(st) {
    const back = mode !== "overview" ? `<span class="chip chip-btn" id="back" style="border-color:var(--accent);color:var(--accent-ink)">← Back to overview</span>` : "";
    $("#overlay").innerHTML = back + `<span class="chip">${st.nodes} nodes</span><span class="chip">${st.edges} edges</span>` +
      (st.communities ? `<span class="chip">${st.communities} communities${st.modularity != null ? ` · Q=${st.modularity.toFixed(2)}` : ""}</span>` : "") +
      (mode !== "overview" ? `<span class="chip accent">${esc(mode)}</span>` : "");
    const b = $("#back"); b && b.addEventListener("click", () => { mode = "overview"; load(); });
  }
  function updateCommunities(st) {
    const ids = [...new Set(current.elements.nodes.map((n) => n.data.community))].filter((c) => c >= 0).sort((a, b) => a - b);
    $("#community").innerHTML = `<option value="">All communities</option>` + ids.map((c) => {
      const mem = current.elements.nodes.filter((n) => n.data.community === c).sort((a, b) => b.data.count - a.data.count).slice(0, 3).map((n) => n.data.label);
      return `<option value="${c}">Community ${c}: ${esc(mem.join(", "))}</option>`; }).join("");
  }
  function legend() {
    const shapeSvg = (t) => {
      const c = series(TYPE[t].slot);
      if (t === "job") return `<svg width="18" height="12"><rect x="1" y="1" width="16" height="10" rx="3" fill="${c}"/></svg>`;
      if (t === "company") return `<svg width="18" height="14"><path d="M9 1 L16 7 L9 13 L2 7 Z" fill="${c}"/></svg>`;
      return `<svg width="18" height="14"><circle cx="9" cy="7" r="6" fill="${c}"/></svg>`;
    };
    let html = "";
    if (colorMode === "type") html += Object.keys(TYPE).map((t) => `<div class="row">${shapeSvg(t)}${TYPE[t].name}</div>`).join("");
    else {
      const ids = [...new Set(cy.nodes().map((n) => n.data("community")))].filter((c) => c >= 0).sort((a, b) => a - b);
      html += `<div class="row" style="font-weight:600">Shape = type · colour = community</div>` + ids.slice(0, 8).map((c) => `<div class="row"><svg width="14" height="14"><circle cx="7" cy="7" r="6" fill="${series(c)}"/></svg>Community ${c}</div>`).join("");
    }
    const types = [...new Set(cy.edges().map((e) => e.data("type")))];
    html += "<hr>" + types.map((t) => { const col = t === "requires" ? series(2) : t === "offers" ? series(1) : t === "co-occurs" ? series(6) : cssVar("--border-strong");
      const da = EDGE[t].style === "dashed" ? 'stroke-dasharray="4 3"' : EDGE[t].style === "dotted" ? 'stroke-dasharray="1.5 3"' : "";
      return `<div class="row"><svg width="22" height="8"><line x1="1" y1="4" x2="21" y2="4" stroke="${col}" stroke-width="2.5" ${da}/></svg>${EDGE[t].name}</div>`; }).join("");
    html += `<hr><div class="row hint">Node size = number of postings</div>`;
    $("#g-legend").innerHTML = html;
  }

  function applyHidden() {
    if (!cy) return;
    cy.nodes().forEach((n) => n.toggleClass("hidden", hidden.has(n.data("type"))));
    const comm = $("#community").value;
    if (comm !== "") cy.nodes().forEach((n) => { if (n.data("community") !== +comm) n.addClass("hidden"); });
  }

  // ------------------------------------------------------------------ data loading
  function params() {
    const jobs = $$("#jobs input:checked").map((o) => o.value);
    const p = new URLSearchParams({ companies: $("#n-comp").value, skills: $("#n-skill").value, min_share: $("#min-share").value / 100, company_skills: 2 });
    if (jobs.length) p.set("jobs", jobs.join("|"));
    if ($("#cluster").value !== "") p.set("cluster", $("#cluster").value);
    return p;
  }
  async function load() {
    setLoading($("#graph-wrap"), true);
    try { mode = "overview"; render(await api("/api/graph?" + params())); }
    catch (e) { showError($("#overlay"), e); }
    finally { setLoading($("#graph-wrap"), false); }
  }
  async function explore(type, label) {
    setLoading($("#graph-wrap"), true);
    try {
      const d = await api(`/api/graph/neighborhood?type=${type}&label=${encodeURIComponent(label)}`);
      mode = `${TYPE[type].name}: ${label}`; render(d, { animate: true });
    } catch (e) { JL.toast(e.message); }
    finally { setLoading($("#graph-wrap"), false); }
  }

  // ------------------------------------------------------------------ controls
  ["n-comp", "n-skill", "min-share"].forEach((id) => $("#" + id).addEventListener("input", (e) => {
    $(`#${id}-v`).textContent = id === "min-share" ? e.target.value + "%" : e.target.value; }));
  $("#apply").addEventListener("click", load);
  $$("#type-toggles input").forEach((cb) => cb.addEventListener("change", () => {
    cb.checked ? hidden.delete(cb.value) : hidden.add(cb.value); applyHidden(); runLayout(true); }));
  $("#community").addEventListener("change", () => { applyHidden(); runLayout(true); });
  $("#cluster").addEventListener("change", load);
  $$("#color-mode button").forEach((b) => b.addEventListener("click", () => {
    $$("#color-mode button").forEach((x) => x.classList.toggle("on", x === b)); colorMode = b.dataset.m;
    if (cy) { cy.style(style()); legend(); } }));
  $("#z-in").addEventListener("click", () => cy && cy.zoom({ level: cy.zoom() * 1.25, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } }));
  $("#z-out").addEventListener("click", () => cy && cy.zoom({ level: cy.zoom() / 1.25, renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 } }));
  $("#z-fit").addEventListener("click", () => cy && cy.fit(cy.elements(":visible"), 30));
  $("#relayout").addEventListener("click", () => cy && runLayout(true));
  document.addEventListener("themechange", () => { if (cy) { cy.style(style()); legend(); } });

  let st;
  $("#node-search").addEventListener("input", (e) => {
    clearTimeout(st); const q = e.target.value.trim();
    if (q.length < 2) { $("#suggest").style.display = "none"; return; }
    st = setTimeout(async () => {
      try {
        const res = await api("/api/graph/search?q=" + encodeURIComponent(q));
        $("#suggest").innerHTML = res.length ? res.map((r, i) => `<div class="chip-btn" data-i="${i}" style="padding:7px 9px;border-radius:7px;cursor:pointer;display:flex;gap:8px;align-items:center">
          <span class="badge">${TYPE[r.type].name}</span>${esc(r.label)}${r.count ? `<span class="hint" style="margin-left:auto">${fmt.int(r.count)}</span>` : ""}</div>`).join("") : `<div class="hint" style="padding:8px">No match</div>`;
        $("#suggest").style.display = "block";
        $$("#suggest [data-i]").forEach((n) => n.addEventListener("click", () => { const r = res[+n.dataset.i]; $("#suggest").style.display = "none"; $("#node-search").value = r.label; explore(r.type, r.label); }));
      } catch (err) { /* ignore */ }
    }, 220);
  });
  document.addEventListener("click", (e) => { if (!$("#suggest").contains(e.target) && e.target !== $("#node-search")) $("#suggest").style.display = "none"; });

  // ------------------------------------------------------------------ Q&A
  const QA_EX = ["Which companies offer jobs requiring Python and AWS?", "What skills does a Data Engineer need?", "Which jobs require Spark and Kafka?",
    "Which companies hire data scientists in France?", "Which skills should a data analyst who knows Excel and SQL learn next?", "What is the salary of a senior data scientist with PyTorch?",
    "Where are Snowflake jobs?"];
  $("#qa-ex").innerHTML = QA_EX.map((q, i) => `<span class="chip chip-btn" data-i="${i}">${esc(q)}</span>`).join("");
  $("#qa-ex").addEventListener("click", (e) => { const i = e.target.dataset.i; if (i != null) { $("#question").value = QA_EX[i]; ask(); } });
  $("#ask").addEventListener("click", ask);
  $("#question").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } });
  async function ask() {
    const q = $("#question").value;
    if (!q.trim()) { showError($("#qa-out"), { message: "Please type a question.", kind: "input" }); return; }
    setLoading($("#qa-out"), true);
    try {
      const r = await api("/api/qa", { json: { question: q } });
      const p = r.parsed;
      const ents = [["intent", p.intent], ...p.jobs.map((j) => ["job", j]), ...p.skills.map((s) => ["skill", s]), ...(p.country ? [["country", p.country]] : [])];
      $("#qa-out").innerHTML = `<div class="qa-answer">${esc(r.answer)}</div>
        <div class="parsed">Understood as: ${ents.map(([k, v]) => `<span class="chip"><b>${k}</b>&nbsp;${esc(v)}</span>`).join("")}</div>
        ${r.table && r.table.length ? `<div class="table-wrap mt-s"><table class="data qa-table"><thead><tr>${r.columns.map((c, i) => `<th class="${i ? "r" : ""}">${esc(c)}</th>`).join("")}</tr></thead>
          <tbody>${r.table.slice(0, 10).map((row) => `<tr>${row.map((v, i) => `<td class="${i ? "r" : ""}">${esc(typeof v === "number" ? fmt.int(v) : v)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>` : ""}
        ${r.graph && r.graph.stats.nodes ? `<button class="btn btn-ghost btn-sm mt-s" id="show-ans">Show the answer subgraph</button>` : ""}`;
      const b = $("#show-ans"); b && b.addEventListener("click", () => { mode = "Answer subgraph"; render(r.graph, { animate: true }); $("#graph-wrap").scrollIntoView({ behavior: "smooth", block: "center" }); });
      if (r.graph && r.graph.stats.nodes) { mode = "Answer subgraph"; render(r.graph, { animate: true }); }
    } catch (e) { showError($("#qa-out"), e); }
    finally { setLoading($("#qa-out"), false); }
  }

  // ------------------------------------------------------------------ init
  Promise.all([api("/api/graph?companies=20&skills=30&min_share=0.15&company_skills=2"), api("/api/clusters").catch(() => null)]).then(([g, cl]) => {
    $("#jobs").innerHTML = g.jobs.map((j) => `<label><input type="checkbox" value="${esc(j)}">${esc(j)}</label>`).join("");
    $("#jobs").addEventListener("change", () => {
      const sel = $$("#jobs input:checked").map((i) => i.value);
      $("#jobs-sum").textContent = sel.length ? (sel.length <= 2 ? sel.join(", ") : `${sel.length} categories`) : "All categories";
    });
    document.addEventListener("click", (e) => { if (!$("#jobs-dd").contains(e.target)) $("#jobs-dd").open = false; });
    if (cl) $("#cluster").innerHTML += cl.profiles.map((p) => `<option value="${p.cluster}">K-Means cluster ${p.cluster}: ${esc(Charts.truncate(p.label, 230, 12))}</option>`).join("");
    render(g);
  }).catch((e) => showError($("#overlay"), e));
})();
