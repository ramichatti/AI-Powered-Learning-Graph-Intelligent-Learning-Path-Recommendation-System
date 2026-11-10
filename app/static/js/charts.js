/* JobLens — D3 chart components.
   Every chart: titled by its card, labelled axes, recessive grid, direct value labels
   where useful, hover tooltip, responsive (re-renders on resize / theme change). */
(function () {
  "use strict";
  const { esc, showTip, moveTip, hideTip, tipRows, cssVar, series, fmt } = window.JL;
  const registry = new Map();

  const ctx = document.createElement("canvas").getContext("2d");
  function textWidth(t, size = 12) { ctx.font = `${size}px Inter, Segoe UI, system-ui, sans-serif`; return ctx.measureText(t).width; }
  function truncate(t, maxW, size = 12) {
    t = String(t); if (textWidth(t, size) <= maxW) return t;
    while (t.length > 1 && textWidth(t + "…", size) > maxW) t = t.slice(0, -1);
    return t + "…";
  }

  function mount(el, render) {
    if (typeof el === "string") el = document.querySelector(el);
    if (!el) return;
    el.classList.add("chart");
    const entry = { render, width: 0 };
    registry.set(el, entry);
    const draw = () => {
      const w = el.clientWidth; if (!w) return;
      entry.width = w; el.innerHTML = ""; render(el, w);
    };
    entry.draw = draw;
    if (!el._ro) {
      el._ro = new ResizeObserver(() => { if (Math.abs(el.clientWidth - entry.width) > 4) registry.get(el).draw(); });
      el._ro.observe(el);
    }
    draw();
  }
  document.addEventListener("themechange", () => registry.forEach((e) => e.draw && e.draw()));

  const svgEl = (el, w, h) => d3.select(el).append("svg").attr("width", w).attr("height", h)
    .attr("viewBox", `0 0 ${w} ${h}`).attr("role", "img");

  // ------------------------------------------------------------------ horizontal bars
  function hbar(el, data, o = {}) {
    mount(el, (node, W) => {
      const rowH = o.rowHeight || 26, fs = 12;
      const labels = data.map((d) => d.label + (d.sub ? "" : ""));
      const maxLab = Math.min(o.maxLabelWidth || 230, W * 0.42);
      const labW = Math.min(maxLab, d3.max(labels, (l) => textWidth(l, fs)) + 10);
      const m = { t: 6, r: 56, b: o.xLabel ? 40 : 24, l: labW + 8 };
      const H = m.t + m.b + data.length * rowH;
      const svg = svgEl(node, W, H);
      const x = d3.scaleLinear().domain([0, o.max || d3.max(data, (d) => d.value) || 1]).nice().range([m.l, W - m.r]);
      const y = d3.scaleBand().domain(data.map((d, i) => i)).range([m.t, H - m.b]).padding(0.28);
      svg.append("g").attr("class", "gridline").attr("transform", `translate(0,${H - m.b})`)
        .call(d3.axisBottom(x).ticks(Math.max(2, Math.floor((W - m.l) / 90))).tickSize(-(H - m.b - m.t)).tickFormat(""));
      svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`)
        .call(d3.axisBottom(x).ticks(Math.max(2, Math.floor((W - m.l) / 90))).tickFormat(o.axisFormat || d3.format("~s")).tickSizeOuter(0));
      if (o.xLabel) svg.append("text").attr("class", "axis-title").attr("x", (m.l + W - m.r) / 2).attr("y", H - 6)
        .attr("text-anchor", "middle").text(o.xLabel);
      const g = svg.append("g").selectAll("g").data(data).join("g")
        .style("cursor", o.onClick ? "pointer" : "default");
      g.append("rect").attr("x", x(0)).attr("y", (d, i) => y(i)).attr("height", y.bandwidth())
        .attr("width", (d) => Math.max(0, x(d.value) - x(0))).attr("rx", 4)
        .attr("fill", (d, i) => d.color || (o.colorFn ? o.colorFn(d, i) : cssVar("--series-1")))
        .attr("opacity", (d) => (o.highlight && d.label !== o.highlight ? 0.45 : 1));
      g.append("rect").attr("x", 0).attr("y", (d, i) => y(i) - 3).attr("width", W).attr("height", y.bandwidth() + 6)
        .attr("fill", "transparent")
        .on("mousemove", (ev, d) => showTip(tipRows(d.label, d.tip || [[o.valueName || "Value", (o.format || fmt.int)(d.value)]]), ev))
        .on("mouseleave", hideTip)
        .on("click", (ev, d) => o.onClick && o.onClick(d));
      g.append("text").attr("class", "cat-label").attr("x", m.l - 8).attr("y", (d, i) => y(i) + y.bandwidth() / 2)
        .attr("dy", "0.35em").attr("text-anchor", "end").text((d) => truncate(d.label, labW - 6, fs))
        .append("title").text((d) => d.label);
      g.append("text").attr("class", "value-label").attr("x", (d) => x(d.value) + 6)
        .attr("y", (d, i) => y(i) + y.bandwidth() / 2).attr("dy", "0.35em").text((d) => (o.format || fmt.int)(d.value));
    });
  }

  // ------------------------------------------------------------------ grouped horizontal bars (2-3 series)
  function groupedHbar(el, rows, keys, o = {}) {
    mount(el, (node, W) => {
      const fs = 12, rowH = o.rowHeight || 34;
      const labW = Math.min(220, d3.max(rows, (r) => textWidth(r.label, fs)) + 10);
      const m = { t: 8, r: 48, b: 40, l: labW + 8 };
      const H = m.t + m.b + rows.length * rowH;
      const svg = svgEl(node, W, H);
      const x = d3.scaleLinear().domain([0, o.max || 1]).range([m.l, W - m.r]);
      const y = d3.scaleBand().domain(rows.map((r) => r.label)).range([m.t, H - m.b]).padding(0.2);
      const yi = d3.scaleBand().domain(keys.map((k) => k.key)).range([0, y.bandwidth()]).padding(0.12);
      svg.append("g").attr("class", "gridline").attr("transform", `translate(0,${H - m.b})`)
        .call(d3.axisBottom(x).ticks(5).tickSize(-(H - m.b - m.t)).tickFormat(""));
      svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`)
        .call(d3.axisBottom(x).ticks(5).tickFormat(o.axisFormat || d3.format(".1f")).tickSizeOuter(0));
      if (o.xLabel) svg.append("text").attr("class", "axis-title").attr("x", (m.l + W - m.r) / 2).attr("y", H - 6).attr("text-anchor", "middle").text(o.xLabel);
      const g = svg.append("g").selectAll("g").data(rows).join("g").attr("transform", (r) => `translate(0,${y(r.label)})`);
      keys.forEach((k, ki) => {
        g.append("rect").attr("x", x(0)).attr("y", yi(k.key)).attr("height", yi.bandwidth())
          .attr("width", (r) => Math.max(0, x(r[k.key]) - x(0))).attr("rx", 3).attr("fill", series(k.slot ?? ki))
          .on("mousemove", (ev, r) => showTip(tipRows(r.label, keys.map((kk) => [kk.name, (o.format || fmt.num)(r[kk.key])])), ev))
          .on("mouseleave", hideTip);
        g.append("text").attr("class", "value-label").attr("x", (r) => x(r[k.key]) + 5).attr("y", yi(k.key) + yi.bandwidth() / 2)
          .attr("dy", "0.35em").style("font-size", "10px").text((r) => (o.format || fmt.num)(r[k.key]));
      });
      g.append("text").attr("class", "cat-label").attr("x", m.l - 8).attr("y", y.bandwidth() / 2).attr("dy", "0.35em")
        .attr("text-anchor", "end").text((r) => truncate(r.label, labW - 6, fs));
    });
  }

  // ------------------------------------------------------------------ vertical bars
  function vbar(el, data, o = {}) {
    mount(el, (node, W) => {
      const H = o.height || 260, m = { t: 22, r: 12, b: o.xLabel ? 46 : 30, l: 52 };
      const svg = svgEl(node, W, H);
      const x = d3.scaleBand().domain(data.map((d) => d.label)).range([m.l, W - m.r]).padding(0.22);
      const y = d3.scaleLinear().domain([o.min ?? 0, (o.max || d3.max(data, (d) => d.value)) * 1.12]).nice().range([H - m.b, m.t]);
      svg.append("g").attr("class", "gridline").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(W - m.l - m.r)).tickFormat(""));
      svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(o.axisFormat || d3.format("~s")).tickSizeOuter(0));
      svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).tickSizeOuter(0));
      if (o.xLabel) svg.append("text").attr("class", "axis-title").attr("x", (m.l + W - m.r) / 2).attr("y", H - 8).attr("text-anchor", "middle").text(o.xLabel);
      if (o.yLabel) svg.append("text").attr("class", "axis-title").attr("transform", `translate(13,${(m.t + H - m.b) / 2}) rotate(-90)`).attr("text-anchor", "middle").text(o.yLabel);
      const bars = svg.append("g").selectAll("g").data(data).join("g");
      bars.append("rect").attr("x", (d) => x(d.label)).attr("width", x.bandwidth()).attr("y", (d) => y(d.value))
        .attr("height", (d) => Math.max(0, y(o.min ?? 0) - y(d.value))).attr("rx", 4)
        .attr("fill", (d) => d.color || cssVar("--series-1"))
        .on("mousemove", (ev, d) => showTip(tipRows(d.tipTitle || d.label, d.tip || [[o.valueName || "Value", (o.format || fmt.int)(d.value)]]), ev))
        .on("mouseleave", hideTip);
      const showAll = x.bandwidth() > 34;
      bars.filter((d) => showAll || d.color === cssVar("--series-1")).append("text").attr("class", "value-label").attr("text-anchor", "middle")
        .attr("x", (d) => x(d.label) + x.bandwidth() / 2).attr("y", (d) => y(d.value) - 6).text((d) => (o.format || fmt.int)(d.value));
    });
  }

  // ------------------------------------------------------------------ line with crosshair
  function line(el, pts, o = {}) {
    mount(el, (node, W) => {
      const H = o.height || 260, m = { t: 18, r: 18, b: o.xLabel ? 46 : 30, l: 60 };
      const svg = svgEl(node, W, H);
      const x = (o.time ? d3.scaleTime().domain(d3.extent(pts, (p) => p.x)) : d3.scaleLinear().domain(d3.extent(pts, (p) => p.x))).range([m.l, W - m.r]);
      const lo = o.zero === false ? d3.min(pts, (p) => p.y) * 0.97 : 0;
      const y = d3.scaleLinear().domain([lo, d3.max(pts, (p) => p.y) * 1.08]).nice().range([H - m.b, m.t]);
      svg.append("g").attr("class", "gridline").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(W - m.l - m.r)).tickFormat(""));
      svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(o.yFormat || d3.format("~s")).tickSizeOuter(0));
      const xa = o.time ? d3.axisBottom(x).ticks(d3.timeMonth.every(W < 500 ? 2 : 1)).tickFormat(d3.timeFormat("%b")) :
        d3.axisBottom(x).tickValues(pts.map((p) => p.x)).tickFormat(d3.format("d"));
      svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(xa.tickSizeOuter(0));
      if (o.xLabel) svg.append("text").attr("class", "axis-title").attr("x", (m.l + W - m.r) / 2).attr("y", H - 8).attr("text-anchor", "middle").text(o.xLabel);
      if (o.yLabel) svg.append("text").attr("class", "axis-title").attr("transform", `translate(14,${(m.t + H - m.b) / 2}) rotate(-90)`).attr("text-anchor", "middle").text(o.yLabel);
      if (o.markX != null) {
        svg.append("line").attr("x1", x(o.markX)).attr("x2", x(o.markX)).attr("y1", m.t).attr("y2", H - m.b)
          .attr("stroke", cssVar("--text-3")).attr("stroke-dasharray", "4 4");
        svg.append("text").attr("class", "value-label").attr("x", x(o.markX) + 6).attr("y", m.t + 10).text(o.markLabel || "");
      }
      svg.append("path").datum(pts).attr("fill", "none").attr("stroke", cssVar("--series-1")).attr("stroke-width", 2)
        .attr("d", d3.line().x((p) => x(p.x)).y((p) => y(p.y)).curve(d3.curveMonotoneX));
      svg.append("g").selectAll("circle").data(pts).join("circle").attr("cx", (p) => x(p.x)).attr("cy", (p) => y(p.y))
        .attr("r", 4).attr("fill", cssVar("--series-1")).attr("stroke", cssVar("--surface")).attr("stroke-width", 2);
      const cross = svg.append("line").attr("y1", m.t).attr("y2", H - m.b).attr("stroke", cssVar("--border-strong")).style("opacity", 0);
      const dot = svg.append("circle").attr("r", 6).attr("fill", "none").attr("stroke", cssVar("--series-1")).attr("stroke-width", 2).style("opacity", 0);
      svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", W - m.l - m.r).attr("height", H - m.t - m.b).attr("fill", "transparent")
        .on("mousemove", (ev) => {
          const [mx] = d3.pointer(ev);
          const p = pts.reduce((a, b) => (Math.abs(x(b.x) - mx) < Math.abs(x(a.x) - mx) ? b : a));
          cross.attr("x1", x(p.x)).attr("x2", x(p.x)).style("opacity", 1);
          dot.attr("cx", x(p.x)).attr("cy", y(p.y)).style("opacity", 1);
          showTip(tipRows(o.time ? d3.timeFormat("%B %Y")(p.x) : `${o.xName || "x"} = ${p.x}`, [[o.yName || "Value", (o.format || fmt.int)(p.y)]]), ev);
        })
        .on("mouseleave", () => { cross.style("opacity", 0); dot.style("opacity", 0); hideTip(); });
    });
  }

  // ------------------------------------------------------------------ heatmap
  function heatmap(el, rows, cols, values, o = {}) {
    mount(el, (node, W) => {
      const fs = 11.5;
      const labW = Math.min(o.maxRowLabel || 210, d3.max(rows, (r) => textWidth(r, fs)) + 12);
      const colLabMax = Math.min(150, d3.max(cols, (c) => textWidth(c, fs)));
      const rot = (W - labW) / cols.length < colLabMax + 8;
      const m = { t: 8, r: 10, b: rot ? colLabMax * 0.72 + 20 : 34, l: labW };
      const cellH = o.cellHeight || 26;
      const H = m.t + m.b + rows.length * cellH;
      const svg = svgEl(node, W, H);
      const x = d3.scaleBand().domain(cols).range([m.l, W - m.r]).padding(0.04);
      const y = d3.scaleBand().domain(rows).range([m.t, H - m.b]).padding(0.06);
      const vmax = o.max ?? d3.max(values.flat());
      const ramp = [0, 1, 2, 3, 4, 5, 6, 7].map((i) => cssVar(`--seq-${i}`));
      const color = d3.scaleQuantize().domain([0, vmax || 1]).range(ramp);
      const cells = [];
      rows.forEach((r, i) => cols.forEach((c, j) => cells.push({ r, c, v: values[i][j] })));
      const g = svg.append("g").selectAll("g").data(cells).join("g");
      g.append("rect").attr("x", (d) => x(d.c)).attr("y", (d) => y(d.r)).attr("width", x.bandwidth()).attr("height", y.bandwidth())
        .attr("rx", 3).attr("fill", (d) => color(d.v))
        .on("mousemove", (ev, d) => showTip(tipRows(`${d.r}`, [[o.colName || "Column", d.c], [o.valueName || "Value", (o.format || fmt.pct)(d.v)]]), ev))
        .on("mouseleave", hideTip);
      if (x.bandwidth() > 26 && o.labels !== false) g.append("text").attr("x", (d) => x(d.c) + x.bandwidth() / 2).attr("y", (d) => y(d.r) + y.bandwidth() / 2)
        .attr("dy", "0.35em").attr("text-anchor", "middle").style("font-size", "10.5px").style("pointer-events", "none")
        .style("fill", (d) => (color.range().indexOf(color(d.v)) >= 4 ? "#ffffff" : cssVar("--text")))
        .text((d) => (o.cellFormat || ((v) => Math.round(v * 100)))(d.v));
      svg.append("g").selectAll("text").data(rows).join("text").attr("class", "cat-label").style("font-size", fs + "px")
        .attr("x", m.l - 8).attr("y", (r) => y(r) + y.bandwidth() / 2).attr("dy", "0.35em").attr("text-anchor", "end")
        .text((r) => truncate(r, labW - 12, fs)).append("title").text((r) => r);
      const ct = svg.append("g").selectAll("text").data(cols).join("text").attr("class", "cat-label").style("font-size", fs + "px");
      if (rot) ct.attr("transform", (c) => `translate(${x(c) + x.bandwidth() / 2},${H - m.b + 10}) rotate(-38)`).attr("text-anchor", "end").attr("dy", "0.35em")
        .text((c) => truncate(c, 140, fs));
      else ct.attr("x", (c) => x(c) + x.bandwidth() / 2).attr("y", H - m.b + 18).attr("text-anchor", "middle").text((c) => truncate(c, x.bandwidth() - 4, fs));
      ct.append("title").text((c) => c);
    });
  }

  // ------------------------------------------------------------------ salary ranges (IQR + median)
  function ranges(el, data, o = {}) {
    mount(el, (node, W) => {
      const fs = 12, rowH = 30;
      const labW = Math.min(200, d3.max(data, (d) => textWidth(d.label, fs)) + 10);
      const m = { t: 8, r: 70, b: 40, l: labW + 8 };
      const H = m.t + m.b + data.length * rowH;
      const svg = svgEl(node, W, H);
      const x = d3.scaleLinear().domain([o.min ?? d3.min(data, (d) => d.q1) * 0.9, d3.max(data, (d) => d.q3) * 1.04]).nice().range([m.l, W - m.r]);
      const y = d3.scaleBand().domain(data.map((d) => d.label)).range([m.t, H - m.b]).padding(0.4);
      svg.append("g").attr("class", "gridline").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(6).tickSize(-(H - m.b - m.t)).tickFormat(""));
      svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(6).tickFormat((v) => "$" + d3.format("~s")(v)).tickSizeOuter(0));
      if (o.xLabel) svg.append("text").attr("class", "axis-title").attr("x", (m.l + W - m.r) / 2).attr("y", H - 6).attr("text-anchor", "middle").text(o.xLabel);
      const g = svg.append("g").selectAll("g").data(data).join("g")
        .on("mousemove", (ev, d) => showTip(tipRows(d.label, [["Median", fmt.usd(d.median)], ["25th–75th pct", `${fmt.usd(d.q1)} – ${fmt.usd(d.q3)}`], ["Postings with salary", fmt.int(d.n)]]), ev))
        .on("mouseleave", hideTip);
      g.append("rect").attr("x", (d) => x(d.q1)).attr("width", (d) => x(d.q3) - x(d.q1)).attr("y", (d) => y(d.label)).attr("height", y.bandwidth())
        .attr("rx", 4).attr("fill", cssVar("--seq-2"));
      g.append("line").attr("x1", (d) => x(d.median)).attr("x2", (d) => x(d.median)).attr("y1", (d) => y(d.label) - 3).attr("y2", (d) => y(d.label) + y.bandwidth() + 3)
        .attr("stroke", cssVar("--series-1")).attr("stroke-width", 3).attr("stroke-linecap", "round");
      g.append("text").attr("class", "cat-label").attr("x", m.l - 8).attr("y", (d) => y(d.label) + y.bandwidth() / 2).attr("dy", "0.35em").attr("text-anchor", "end").text((d) => truncate(d.label, labW - 6, fs));
      g.append("text").attr("class", "value-label").attr("x", (d) => x(d.q3) + 6).attr("y", (d) => y(d.label) + y.bandwidth() / 2).attr("dy", "0.35em").text((d) => "$" + d3.format(".3~s")(d.median));
    });
  }

  // ------------------------------------------------------------------ cluster scatter (zoom, pan, select)
  const SYMBOLS = [d3.symbolCircle, d3.symbolSquare, d3.symbolTriangle, d3.symbolDiamond, d3.symbolCross,
                   d3.symbolStar, d3.symbolWye, d3.symbolTriangle2, d3.symbolSquare2, d3.symbolDiamond2];
  function clusterColor(c) { return series(c); }
  function clusterSymbol(c) { return SYMBOLS[c % SYMBOLS.length]; }

  function scatter(el, data, o = {}) {
    mount(el, (node, W) => {
      const H = o.height || 520, m = { t: 12, r: 12, b: 44, l: 52 };
      const svg = svgEl(node, W, H);
      const pts = data.points;
      const xe = d3.extent(pts, (p) => p.x), ye = d3.extent(pts, (p) => p.y);
      const x0 = d3.scaleLinear().domain([xe[0] - 0.03, xe[1] + 0.03]).range([m.l, W - m.r]);
      const y0 = d3.scaleLinear().domain([ye[0] - 0.03, ye[1] + 0.03]).range([H - m.b, m.t]);
      let x = x0, y = y0;
      svg.append("defs").append("clipPath").attr("id", "clip-sc").append("rect").attr("x", m.l).attr("y", m.t).attr("width", W - m.l - m.r).attr("height", H - m.t - m.b);
      const gx = svg.append("g").attr("class", "axis").attr("transform", `translate(0,${H - m.b})`);
      const gy = svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`);
      svg.append("text").attr("class", "axis-title").attr("x", (m.l + W - m.r) / 2).attr("y", H - 8).attr("text-anchor", "middle").text(o.xLabel || "PC1");
      svg.append("text").attr("class", "axis-title").attr("transform", `translate(13,${(m.t + H - m.b) / 2}) rotate(-90)`).attr("text-anchor", "middle").text(o.yLabel || "PC2");
      const plot = svg.append("g").attr("clip-path", "url(#clip-sc)");
      const sel = o.selected;
      const sym = {}; for (let c = 0; c < 20; c++) sym[c] = d3.symbol().type(clusterSymbol(c)).size(30)();
      const dots = plot.append("g").selectAll("path").data(pts).join("path").attr("d", (p) => sym[p.c])
        .attr("fill", (p) => clusterColor(p.c)).attr("fill-opacity", (p) => (sel == null || p.c === sel ? 0.55 : 0.06));
      const cents = plot.append("g").selectAll("g").data(data.centroids.map((c, i) => ({ c: i, x: c[0], y: c[1] }))).join("g")
        .style("cursor", "pointer").on("click", (ev, d) => o.onSelect && o.onSelect(d.c));
      cents.append("circle").attr("r", 13).attr("fill", cssVar("--surface")).attr("stroke", (d) => (sel === d.c ? cssVar("--text") : cssVar("--text-2"))).attr("stroke-width", (d) => (sel === d.c ? 3 : 1.6));
      cents.append("text").attr("text-anchor", "middle").attr("dy", "0.35em").style("font-size", "11px").style("font-weight", 700).style("fill", cssVar("--text")).text((d) => d.c);
      let qmark = null;
      if (o.query) {
        qmark = plot.append("g");
        qmark.append("path").attr("d", d3.symbol().type(d3.symbolStar).size(420)()).attr("fill", cssVar("--series-8")).attr("stroke", cssVar("--surface")).attr("stroke-width", 2);
      }
      function place() {
        gx.call(d3.axisBottom(x).ticks(6).tickSizeOuter(0)); gy.call(d3.axisLeft(y).ticks(6).tickSizeOuter(0));
        dots.attr("transform", (p) => `translate(${x(p.x)},${y(p.y)})`);
        cents.attr("transform", (d) => `translate(${x(d.x)},${y(d.y)})`);
        if (qmark) qmark.attr("transform", `translate(${x(o.query[0])},${y(o.query[1])})`);
      }
      place();
      const delaunay = d3.Delaunay.from(pts, (p) => x0(p.x), (p) => y0(p.y));
      let transform = d3.zoomIdentity;
      const zoom = d3.zoom().scaleExtent([1, 14]).translateExtent([[m.l, m.t], [W - m.r, H - m.b]]).extent([[m.l, m.t], [W - m.r, H - m.b]])
        .on("zoom", (ev) => { transform = ev.transform; x = transform.rescaleX(x0); y = transform.rescaleY(y0); place(); });
      const overlay = svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", W - m.l - m.r).attr("height", H - m.t - m.b)
        .attr("fill", "transparent").style("cursor", "grab").lower();
      svg.call(zoom);
      svg.on("mousemove", (ev) => {
        const [mx, my] = d3.pointer(ev);
        const [ix, iy] = transform.invert([mx, my]);
        const i = delaunay.find(ix, iy); const p = pts[i];
        if (!p) return;
        const dx = x(p.x) - mx, dy = y(p.y) - my;
        if (dx * dx + dy * dy > 120) { hideTip(); return; }
        const lab = (data.labels && data.labels[p.c]) || "";
        showTip(tipRows(p.t, [["Cluster", `C${p.c} · ${lab}`], ["Category", p.j]]), ev);
      }).on("mouseleave", hideTip);
      svg.on("dblclick.zoom", null);
      node._resetZoom = () => svg.transition().duration(300).call(zoom.transform, d3.zoomIdentity);
    });
  }

  function legend(el, items, o = {}) {
    if (typeof el === "string") el = document.querySelector(el);
    el.className = "legend";
    el.innerHTML = items.map((it, i) => {
      const s = it.symbol ? `<svg width="14" height="14" viewBox="-7 -7 14 14"><path d="${d3.symbol().type(it.symbol).size(70)()}" fill="${it.color}"/></svg>` :
        `<span class="sw" style="display:inline-block;width:12px;height:12px;border-radius:3px;background:${it.color}"></span>`;
      const cls = ["item", o.onClick ? "clickable" : "", it.on ? "on" : "", it.dim ? "dim" : ""].join(" ");
      return `<span class="${cls}" data-i="${i}" title="${esc(it.title || it.label)}">${s}${esc(it.label)}</span>`;
    }).join("");
    if (o.onClick) el.querySelectorAll(".item").forEach((n) => n.addEventListener("click", () => o.onClick(items[+n.dataset.i], +n.dataset.i)));
  }

  window.Charts = { hbar, groupedHbar, vbar, line, heatmap, ranges, scatter, legend, clusterColor, clusterSymbol, truncate, textWidth, mount };
})();
