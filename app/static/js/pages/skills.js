(function () {
  const { api, fmt, esc, showError, setLoading, withButton, $ } = JL;
  const ta = $("#text"), msg = $("#msg");
  const CATS = ["Programming Languages", "BI & Visualization Tools", "Data Analysis & Office Tools", "Machine Learning Frameworks",
    "Data Engineering & Big Data", "Databases", "Cloud Platforms", "DevOps & Engineering Tools", "Web & App Frameworks",
    "Collaboration Tools", "Data & AI Concepts"];
  const EXAMPLE = "We are hiring a Senior Analytics Engineer to own our modern data stack. You will model data with dbt on Snowflake, " +
    "orchestrate pipelines with Apache Airflow and maintain ingestion from PostgreSQL and MongoDB. Dashboards are built in Looker and Power BI. " +
    "Strong SQL and Python (pandas, scikit-learn) required; experience with AWS (S3, Lambda), Docker and Git. " +
    "Knowledge of R or Scala is a plus. You will collaborate with our R&D team using Jira and Confluence.";

  api("/api/skills/dictionary").then((d) => { $("#n-skills").textContent = d.n_skills; $("#n-alias").textContent = d.n_aliases; }).catch(() => {});
  $("#example").addEventListener("click", () => { ta.value = EXAMPLE; run(); });
  $("#clear").addEventListener("click", () => { ta.value = ""; msg.innerHTML = ""; $("#details").style.display = "none"; $("#hl").innerHTML = ""; $("#cats").innerHTML = JL.empty("No text analysed yet."); });
  $("#random").addEventListener("click", (ev) => withButton(ev.currentTarget, async () => {
    try { const r = await api("/api/random-ad"); ta.value = r.text; run(); } catch (e) { showError(msg, e); }
  }));
  $("#go").addEventListener("click", run);
  ta.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run(); });

  async function run() {
    if (!ta.value.trim()) { showError(msg, { message: "Please paste a job description first.", kind: "input" }); return; }
    msg.innerHTML = ""; setLoading($("#hl-card"), true);
    try { render(await api("/api/skills", { json: { text: ta.value } })); }
    catch (e) { showError(msg, e); }
    finally { setLoading($("#hl-card"), false); }
  }

  function highlight(text, skills) {
    const spans = skills.flatMap((s) => s.spans.map(([a, b]) => ({ a, b, s: s.skill, c: s.category }))).sort((x, y) => x.a - y.a);
    let html = "", pos = 0;
    spans.forEach((sp) => { if (sp.a < pos) return; html += esc(text.slice(pos, sp.a)) + `<mark title="${esc(sp.s)} · ${esc(sp.c)}">${esc(text.slice(sp.a, sp.b))}</mark>`; pos = sp.b; });
    return html + esc(text.slice(pos));
  }

  function render(r) {
    $("#hl-sub").textContent = `${r.n_skills} skills in ${r.n_categories} categories`;
    if (!r.n_skills) {
      $("#cats").innerHTML = JL.empty("No known skill found in this text.");
      $("#hl").innerHTML = ""; $("#details").style.display = "none"; return;
    }
    const order = CATS.filter((c) => r.by_category[c]);
    $("#cats").innerHTML = order.map((c) => `<div style="margin-bottom:10px"><div class="hint" style="font-weight:600;margin-bottom:5px">${esc(c)} <span class="badge">${r.by_category[c].length}</span></div>
      <div class="chips">${r.by_category[c].map((s) => `<span class="chip accent">${esc(s)}</span>`).join("")}</div></div>`).join("");
    $("#hl").innerHTML = `<div class="hl-text">${highlight(r.text, r.skills)}</div>`;
    $("#details").style.display = "";
    const withShare = r.skills.filter((s) => s.dataset_share != null).sort((a, b) => b.dataset_share - a.dataset_share);
    if (withShare.length) {
      Charts.hbar("#freq", withShare.map((s) => ({ label: s.skill, value: s.dataset_share,
        tip: [["Share of postings", fmt.pct(s.dataset_share, 2)], ["Postings", fmt.int(s.dataset_count)], ["Category", s.category]] })),
        { format: (v) => fmt.pct(v, v < 0.01 ? 2 : 1), axisFormat: d3.format(".0%"), xLabel: "% of all ads" });
    } else $("#freq").innerHTML = JL.empty("These skills are concepts that are not tagged in the dataset.");
    if (r.recommended.length) {
      Charts.hbar("#recs", r.recommended.map((x) => ({ label: x.skill, value: x.cooccurrence,
        tip: [["Average co-occurrence with detected skills", fmt.pct(x.cooccurrence)], ["Lift vs. overall frequency", "×" + x.lift.toFixed(2)], ["PPR × log-lift score", x.score.toFixed(4)], ["Category", x.category]] })),
        { format: (v) => fmt.pct(v, 0), axisFormat: d3.format(".0%"), xLabel: "How often they appear together", colorFn: () => JL.series(2) });
    } else $("#recs").innerHTML = JL.empty("No suggestion for these skills.");
    $("#tbl").innerHTML = `<thead><tr><th>Skill</th><th>Category</th><th class="r">Mentions</th><th class="r">% of ads</th><th>Most requested by</th><th>Similar skills</th></tr></thead><tbody>` +
      r.skills.map((s) => `<tr><td><b>${esc(s.skill)}</b></td><td>${esc(s.category)}</td><td class="r">${s.mentions}</td>
        <td class="r">${s.dataset_share != null ? fmt.pct(s.dataset_share, 2) : "–"}</td>
        <td>${s.top_job ? `${esc(s.top_job)} <span class="hint">(${fmt.pct(s.top_job_share)} of its ads)</span>` : "–"}</td>
        <td><div class="chips">${s.related_semantic.map((x) => `<span class="chip" title="cosine ${x.similarity.toFixed(2)}">${esc(x.skill)}</span>`).join("")}</div></td></tr>`).join("") + "</tbody>";
  }
})();
