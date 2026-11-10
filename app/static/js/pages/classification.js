(function () {
  const { api, fmt, esc, showError, setLoading, withButton, $ } = JL;
  const ta = $("#text"), out = $("#result"), msg = $("#form-msg");
  const EXAMPLES = [
    ["Streaming data engineer", "Data Engineer – we build real-time pipelines with Kafka, Spark Structured Streaming and Airflow on AWS (S3, Glue, Redshift). Python and SQL required, Terraform is a plus."],
    ["BI reporting", "Reporting analyst: design Power BI and Tableau dashboards for finance stakeholders, write SQL queries, automate Excel reports with VBA."],
    ["Deep learning", "Machine Learning Engineer to deploy deep learning models (PyTorch, TensorFlow) in production with Docker, Kubernetes and MLflow on GCP."],
    ["Lead scientist", "Lead Data Scientist – own experimentation and causal inference, Python, R, scikit-learn, Spark; mentor a team of 5."],
  ];
  $("#examples").innerHTML = '<span class="hint" style="align-self:center">Examples:</span>' +
    EXAMPLES.map((e, i) => `<span class="chip chip-btn" data-i="${i}">${esc(e[0])}</span>`).join("");
  $("#examples").addEventListener("click", (ev) => { const i = ev.target.dataset.i; if (i != null) { ta.value = EXAMPLES[i][1]; upd(); predict(); } });
  const upd = () => ($("#count").textContent = `${fmt.int(ta.value.length)} characters`);
  ta.addEventListener("input", upd);
  $("#clear").addEventListener("click", () => { ta.value = ""; upd(); msg.innerHTML = ""; out.innerHTML = JL.empty("Paste a job ad or pick an example, then press Classify."); });
  $("#random").addEventListener("click", (ev) => withButton(ev.currentTarget, async () => {
    try { const r = await api("/api/random-ad"); ta.value = r.text; upd(); msg.innerHTML = JL.alertBox(`Real dataset record — true category: ${r.category} (${r.company})`, "info"); }
    catch (e) { showError(msg, e); }
  }));
  ta.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) predict(); });
  $("#predict").addEventListener("click", predict);

  async function predict() {
    const text = ta.value;
    if (!text.trim()) { showError(msg, { message: "Please paste a job ad first.", kind: "input" }); ta.focus(); return; }
    if (!msg.querySelector(".alert-info")) msg.innerHTML = "";
    setLoading($("#result-card"), true);
    try {
      const r = await api("/api/classify", { json: { text } });
      render(r);
    } catch (e) { showError(out, e); }
    finally { setLoading($("#result-card"), false); }
  }

  function render(r) {
    const warn = r.warning ? JL.alertBox(r.warning, "warn") : "";
    const so = r.skills_only ? `<div class="hint mt">Looking only at the skills, the model would say <b>${esc(r.skills_only.prediction)}</b> (${fmt.pct(r.skills_only.confidence)}).</div>` :
      "";
    out.innerHTML = `${warn}
      <div class="pred-hero ${warn ? "mt-s" : ""}"><div><div class="lbl">Job title</div><div class="val">${esc(r.prediction)}</div></div>
        <div class="conf"><div class="lbl">Confidence</div><div class="val num">${fmt.pct(r.confidence)}</div></div></div>
      <div class="mt"><div class="hint" style="margin-bottom:4px">How likely each title is</div><div id="proba"></div></div>
      <div class="grid g-2 mt-s">
        <div><div class="hint" style="margin-bottom:6px">Words that decided it</div><div class="chips">${r.top_terms.map((t) => `<span class="chip accent" title="contribution ${t.weight.toFixed(3)}">${esc(t.term)}</span>`).join("") || '<span class="hint">–</span>'}</div></div>
        <div><div class="hint" style="margin-bottom:6px">Skills found (${r.skills.length})</div><div class="chips">${r.skills.map((s) => `<span class="chip">${esc(s)}</span>`).join("") || '<span class="hint">none</span>'}</div></div>
      </div>${so}`;
    Charts.hbar("#proba", r.probabilities.slice(0, 6).map((p, i) => ({ label: p.label, value: p.p, color: i === 0 ? JL.cssVar("--series-1") : JL.cssVar("--seq-2") })),
      { max: 1, format: (v) => fmt.pct(v), axisFormat: d3.format(".0%"), valueName: "Probability", rowHeight: 24 });
  }

  api("/api/classification/info").then((m) => {
    setLoading($("#cm"), false); setLoading($("#f1"), false);
    const vals = m.confusion_matrix.map((row) => { const s = row.reduce((a, b) => a + b, 0); return row.map((v) => v / s); });
    Charts.heatmap("#cm", m.labels, m.labels, vals, { max: 1, colName: "Predicted", valueName: "Share of true class", cellHeight: 30, maxRowLabel: 175,
      cellFormat: (v) => (v >= 0.005 ? Math.round(v * 100) : "") });
    Charts.legend("#f1-legend", [{ label: "Title + skills (main)", color: JL.series(0) }, { label: "Skills only", color: JL.series(1) }]);
    const rows = m.per_class.slice().sort((a, b) => b.f1 - a.f1).map((p) => ({ label: p.label, main: p.f1, sk: p.f1_skills_only }));
    Charts.groupedHbar("#f1", rows, [{ key: "main", name: "Title + skills" }, { key: "sk", name: "Skills only" }], { max: 1, format: (v) => v.toFixed(2), xLabel: "F1-score" });
    $("#model-meta").innerHTML = `<b>Main model:</b> accuracy ${fmt.pct(m.accuracy)}, macro-F1 ${fmt.num(m.f1_macro)}, macro precision ${fmt.num(m.precision_macro)}, macro recall ${fmt.num(m.recall_macro)}.
      Trained on ${fmt.int(m.train)} ads, evaluated on ${fmt.int(m.test)} held-out ads (stratified split), C = ${m.best_C}.
      Most frequent confusion: <b>${esc(m.top_confusions[0].true)}</b> → ${esc(m.top_confusions[0].pred)} (${fmt.int(m.top_confusions[0].count)} ads).`;
  }).catch((e) => { setLoading($("#cm"), false); setLoading($("#f1"), false); showError($("#cm"), e); });
})();
