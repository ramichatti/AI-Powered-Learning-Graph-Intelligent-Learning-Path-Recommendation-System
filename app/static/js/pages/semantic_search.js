(function () {
  const { api, fmt, esc, showError, setLoading, $ } = JL;
  const q = $("#q"), msg = $("#msg");
  const EX = ["building deep neural networks for image recognition", "dashboards and KPI reports for business stakeholders",
    "real-time event streaming infrastructure", "python, sql, airflow, snowflake, dbt", "cloud infrastructure automation with terraform"];
  $("#examples").innerHTML = '<span class="hint" style="align-self:center">Try:</span>' + EX.map((e, i) => `<span class="chip chip-btn" data-i="${i}">${esc(e)}</span>`).join("");
  $("#examples").addEventListener("click", (e) => { const i = e.target.dataset.i; if (i != null) { q.value = EX[i]; run(); } });
  $("#go").addEventListener("click", run);
  q.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run(); });
  let clusterData = null;
  api("/api/classification/info").then((m) => { $("#cat").innerHTML += m.labels.map((l) => `<option>${esc(l)}</option>`).join(""); }).catch(() => {});
  api("/api/clusters").then((d) => (clusterData = d)).catch(() => {});

  const PIN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 21s-7-6.2-7-11a7 7 0 1 1 14 0c0 4.8-7 11-7 11Z"/><circle cx="12" cy="10" r="2.5"/></svg>';
  const BLD = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="4" y="3" width="16" height="18" rx="1.5"/><path d="M9 21v-4h6v4"/></svg>';

  async function run() {
    if (!q.value.trim()) { showError(msg, { message: "Please describe a job or your skills first.", kind: "input" }); return; }
    msg.innerHTML = ""; setLoading($("#res-card"), true);
    $("#res-sub").textContent = "Searching…";
    try {
      const r = await api("/api/search", { json: { query: q.value, k: +$("#k").value, category: $("#cat").value } });
      render(r);
    } catch (e) { showError($("#results"), e); $("#res-sub").textContent = "Search failed"; }
    finally { setLoading($("#res-card"), false); }
  }

  function render(r) {
    $("#res-sub").textContent = `Top ${r.results.length} of ${fmt.int(r.n_candidates)} ads` +
      (r.query_skills.length ? ` · skills recognised: ${r.query_skills.join(", ")}` : "");
    if (!r.results.length) { $("#results").innerHTML = JL.empty("No ad matches the selected filter."); return; }
    const max = r.results[0].similarity, min = Math.min(...r.results.map((x) => x.similarity));
    $("#results").innerHTML = r.results.map((x, i) => {
      const w = max === min ? 100 : 35 + 65 * (x.similarity - min) / (max - min);
      const matched = new Set(x.matched_skills);
      const skills = x.skills.slice().sort((a, b) => matched.has(b) - matched.has(a));
      return `<div class="result"><div><div class="title"><span class="rank">${i + 1}</span>${esc(x.title)}</div>
        <div class="meta"><span>${BLD}${esc(x.company)}</span><span>${PIN}${esc(x.location)}</span><span class="badge">${esc(x.category)}</span>
        ${x.salary ? `<span>${fmt.usd(x.salary)}/yr</span>` : ""}</div></div>
        <div class="score"><div class="hint">match</div><div class="v">${x.similarity.toFixed(3)}</div><div class="meter"><span style="width:${w}%"></span></div></div>
        <div class="chips">${skills.slice(0, 14).map((s) => `<span class="chip ${matched.has(s) ? "match" : ""}">${esc(s)}</span>`).join("")}${skills.length > 14 ? `<span class="hint">+${skills.length - 14}</span>` : ""}</div></div>`;
    }).join("");
    if (clusterData && r.query_xy) {
      $("#where-card").style.display = "";
      $("#where-txt").innerHTML = `Closest job family: <b>${esc(r.query_cluster_label)}</b> (red star)`;
      const labels = Object.fromEntries(clusterData.profiles.map((p) => [p.cluster, p.label]));
      Charts.scatter("#mini", { points: clusterData.points.filter((_, i) => i % 3 === 0), centroids: clusterData.centroids, labels },
        { height: 300, query: r.query_xy, selected: r.query_cluster });
    }
  }
})();
