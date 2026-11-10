/* JobLens — shared front-end utilities (API calls, loading states, errors, theme, tooltip). */
(function () {
  "use strict";
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const fmt = {
    int: (v) => (v == null ? "–" : Number(v).toLocaleString("en-US", { maximumFractionDigits: 0 })),
    pct: (v, d = 1) => (v == null ? "–" : (v * 100).toFixed(d) + "%"),
    num: (v, d = 3) => (v == null ? "–" : Number(v).toFixed(d)),
    compact: (v) => (v == null ? "–" : Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(v)),
    usd: (v) => (v == null ? "–" : "$" + Math.round(v).toLocaleString("en-US")),
  };

  async function api(url, options = {}) {
    const opts = { headers: { Accept: "application/json" }, ...options };
    if (opts.json !== undefined) {
      opts.method = opts.method || "POST";
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(opts.json);
      delete opts.json;
    }
    let res;
    try {
      res = await fetch(url, opts);
    } catch (e) {
      throw new Error("Cannot reach the server. Is the Flask app still running?");
    }
    let data = null;
    try { data = await res.json(); } catch (e) { /* non-JSON */ }
    if (!res.ok) {
      const err = new Error((data && data.error) || `Request failed (HTTP ${res.status}).`);
      err.kind = data && data.kind; err.status = res.status;
      throw err;
    }
    return data;
  }

  const ICON_ALERT = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16h.01"/></svg>';
  const ICON_EMPTY = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>';

  function alertBox(msg, kind = "error") {
    return `<div class="alert alert-${kind}" role="alert">${ICON_ALERT}<div>${esc(msg)}</div></div>`;
  }
  function showError(el, err) {
    if (!el) return;
    const msg = typeof err === "string" ? err : err.message;
    const hint = err && err.kind === "artifact" ? " Run the notebook to (re)generate the models." : "";
    el.innerHTML = alertBox(msg + (hint && !msg.includes("notebook") ? hint : ""), err && err.kind === "input" ? "warn" : "error");
  }
  function empty(msg) { return `<div class="empty">${ICON_EMPTY}${esc(msg)}</div>`; }

  function setLoading(el, on) { if (el) el.classList.toggle("loading", !!on); }
  async function withButton(btn, fn) {
    const html = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Working…';
    try { return await fn(); } finally { btn.disabled = false; btn.innerHTML = html; }
  }

  function toast(msg, kind = "error") {
    const t = $("#toast");
    t.innerHTML = alertBox(msg, kind);
    clearTimeout(t._h); t._h = setTimeout(() => (t.innerHTML = ""), 5000);
  }

  // ---------------------------------------------------------------- tooltip
  const tip = () => $("#tooltip");
  function showTip(html, ev) {
    const t = tip(); t.innerHTML = html; t.style.opacity = 1; moveTip(ev);
  }
  function moveTip(ev) {
    const t = tip(); if (!ev) return;
    const pad = 14, w = t.offsetWidth, h = t.offsetHeight;
    let x = ev.clientX + pad, y = ev.clientY + pad;
    if (x + w > window.innerWidth - 8) x = ev.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = ev.clientY - h - pad;
    t.style.left = Math.max(8, x) + "px"; t.style.top = Math.max(8, y) + "px";
  }
  function hideTip() { tip().style.opacity = 0; }
  function tipRows(title, rows) {
    return `<div class="tt-title">${esc(title)}</div>` +
      rows.map(([k, v]) => `<div class="tt-row"><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join("");
  }

  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const series = (i) => cssVar(`--series-${(i % 8) + 1}`);

  // ---------------------------------------------------------------- theme / nav / status
  function initChrome() {
    const btn = $("#theme-btn");
    btn && btn.addEventListener("click", () => {
      const cur = document.documentElement.dataset.theme ||
        (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
      const next = cur === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("joblens-theme", next); } catch (e) {}
      document.dispatchEvent(new CustomEvent("themechange"));
    });
    const mb = $("#menu-btn");
    mb && mb.addEventListener("click", (e) => { e.stopPropagation(); $("#sidebar").classList.toggle("open"); });
    document.addEventListener("click", (e) => {
      const sb = $("#sidebar"); if (sb.classList.contains("open") && !sb.contains(e.target)) sb.classList.remove("open");
    });
    api("/api/health").then((h) => {
      const missing = Object.entries(h.files).filter(([, ok]) => !ok).map(([k]) => k);
      const ok = missing.length === 0 && h.distilbert_dir !== null;
      $("#status-dot").classList.add(ok ? "ok" : "bad");
      $("#status-text").textContent = ok ? "All models ready" :
        `Missing: ${missing.concat(h.distilbert_dir === null ? ["DistilBERT weights"] : []).join(", ")}`;
    }).catch(() => { $("#status-dot").classList.add("bad"); $("#status-text").textContent = "Server unreachable"; });
  }
  document.addEventListener("DOMContentLoaded", initChrome);

  window.JL = { $, $$, esc, fmt, api, showError, alertBox, empty, setLoading, withButton, toast,
                showTip, moveTip, hideTip, tipRows, cssVar, series };
})();
