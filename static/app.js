"use strict";

// ---------- helpers ----------
const $ = (sel, el = document) => el.querySelector(sel);
const app = $("#app");
const api = (path) => fetch(path).then((r) => r.json());
const pct = (x, d = 1) => (x == null ? "—" : `${x.toFixed(d)}%`);
const num = (x, d = 2) => (x == null ? "—" : Number(x).toFixed(d));
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const signed = (x, d = 1) => (x == null ? "—" : `${x >= 0 ? "+" : ""}${x.toFixed(d)}`);
const cls = (x) => (x > 0 ? "pos" : x < 0 ? "neg" : "");

function fmtTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function setMeta(meta) {
  if (!meta) return;
  $("#model-version").textContent = meta.model_version || "";
  let txt = "model: " + (meta.last_prediction_at ? fmtTime(meta.last_prediction_at) : "—");
  if (meta.wc_matches_in_ratings > 0)
    txt += ` · ratings incl. ${meta.wc_matches_in_ratings} WC result${meta.wc_matches_in_ratings > 1 ? "s" : ""}`;
  $("#freshness").textContent = txt;
}

const DISCLAIMER =
  `<div class="disclaimer">Probabilities are model estimates, not guarantees. Recommendations use pre-match data and
   may change as lineups and prices move. Positive expected value does not ensure short-term profit. Use bankroll
   limits and do not chase losses.</div>`;

// ---------- tiny SVG chart kit ----------
function svgLine(series, { w = 720, h = 240, pad = 34, yLabel = "" } = {}) {
  // series: [{name,color,points:[{x:Number,y:Number}]}]
  const xs = series.flatMap((s) => s.points.map((p) => p.x));
  const ys = series.flatMap((s) => s.points.map((p) => p.y));
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const sx = (x) => pad + ((x - xmin) / (xmax - xmin || 1)) * (w - 2 * pad);
  const sy = (y) => h - pad - ((y - ymin) / (ymax - ymin || 1)) * (h - 2 * pad);
  let g = "";
  for (let i = 0; i <= 4; i++) {
    const yy = pad + (i / 4) * (h - 2 * pad);
    const val = ymax - (i / 4) * (ymax - ymin);
    g += `<line class="grid-line" x1="${pad}" y1="${yy}" x2="${w - pad}" y2="${yy}"/>`;
    g += `<text class="axis-text" x="4" y="${yy + 3}">${val.toFixed(2)}</text>`;
  }
  const paths = series.map((s) => {
    const d = s.points.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(" ");
    return `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2"/>`;
  }).join("");
  return `<svg viewBox="0 0 ${w} ${h}" width="100%">${g}${paths}
    <text class="axis-text" x="${pad}" y="${h - 6}">${yLabel}</text></svg>`;
}

function svgBars(bars, { w = 720, h = 220, pad = 30 } = {}) {
  const max = Math.max(...bars.map((b) => b.value), 1);
  const bw = (w - 2 * pad) / bars.length;
  let out = "";
  bars.forEach((b, i) => {
    const bh = (b.value / max) * (h - 2 * pad);
    const x = pad + i * bw, y = h - pad - bh;
    out += `<rect x="${x + 3}" y="${y}" width="${bw - 6}" height="${bh}" fill="${b.color || "var(--accent)"}" rx="2"/>`;
    out += `<text class="axis-text" x="${x + bw / 2}" y="${h - pad + 12}" text-anchor="middle">${esc(b.label)}</text>`;
    out += `<text class="axis-text" x="${x + bw / 2}" y="${y - 4}" text-anchor="middle">${b.value}</text>`;
  });
  return `<svg viewBox="0 0 ${w} ${h}" width="100%">${out}</svg>`;
}

function svgReliability(rows, { w = 360, h = 360, pad = 36 } = {}) {
  const sx = (x) => pad + x * (w - 2 * pad);
  const sy = (y) => h - pad - y * (h - 2 * pad);
  let g = "";
  for (let i = 0; i <= 5; i++) {
    const v = i / 5;
    g += `<line class="grid-line" x1="${sx(0)}" y1="${sy(v)}" x2="${sx(1)}" y2="${sy(v)}"/>`;
    g += `<text class="axis-text" x="6" y="${sy(v) + 3}">${v.toFixed(1)}</text>`;
    g += `<text class="axis-text" x="${sx(v)}" y="${h - 8}" text-anchor="middle">${v.toFixed(1)}</text>`;
  }
  const diag = `<line class="diag" x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(1)}"/>`;
  const pts = rows.map((r) => {
    const rad = 3 + Math.min(8, r.count);
    return `<circle cx="${sx(r.predicted_mean)}" cy="${sy(r.observed_freq)}" r="${rad}"
      fill="rgba(79,157,255,.5)" stroke="var(--accent)"/>`;
  }).join("");
  return `<svg viewBox="0 0 ${w} ${h}" width="100%">${g}${diag}${pts}
    <text class="axis-text" x="${w / 2}" y="${h - 22}" text-anchor="middle">predicted</text></svg>`;
}

// ---------- Dashboard ----------
async function renderDashboard() {
  app.innerHTML = `<div class="loading">Loading matches…</div>`;
  const data = await api("/api/v1/matches");
  setMeta(data.meta);
  const matches = data.matches;

  const valueCards = matches
    .filter((m) => m.top_recommendation && ["Bet", "Lean"].includes(m.top_recommendation.status))
    .sort((a, b) => (b.best_edge_pct || 0) - (a.best_edge_pct || 0))
    .slice(0, 6);

  const lineMoves = matches
    .filter((m) => m.line_movement_summary)
    .map((m) => ({ m, mv: Math.abs(m.line_movement_summary.delta_pp) }))
    .sort((a, b) => b.mv - a.mv).slice(0, 5);

  const confCounts = { High: 0, Medium: 0, Low: 0 };
  matches.forEach((m) => { if (m.confidence_band) confCounts[m.confidence_band]++; });

  app.innerHTML = `
    ${DISCLAIMER}
    <div class="filters">
      <label>Stage <select id="f-stage"><option value="">All</option>
        <option>Group</option><option>Round of 32</option><option>Round of 16</option>
        <option>Quarterfinal</option><option>Semifinal</option><option>Final</option></select></label>
      <label>Status <select id="f-status"><option value="">All</option>
        <option>Bet</option><option>Lean</option><option>Pass</option></select></label>
      <label>Market <select id="f-market"><option value="">All</option>
        <option>1X2</option><option>O/U</option><option>BTTS</option></select></label>
    </div>

    <div class="grid cards-row" id="value-cards">${valueCards.map(valueCardHTML).join("") ||
      `<div class="muted">No active value flags.</div>`}</div>

    <div class="grid two-col" style="margin-top:16px">
      <div class="panel">
        <h2>Upcoming & recent matches</h2>
        <table id="match-table"><thead><tr>
          <th>Kickoff</th><th>Match</th><th>Stage</th><th>Top pick</th>
          <th class="num">Edge</th><th class="num">EV</th><th>Conf.</th><th>Line</th>
        </tr></thead><tbody></tbody></table>
      </div>
      <div>
        <div class="panel"><h2>Biggest line moves</h2>
          ${lineMoves.map(({ m }) => {
            const s = m.line_movement_summary;
            return `<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid var(--border)">
              <span>${esc(m.home_team)} v ${esc(m.away_team)}</span>
              <span class="${cls(s.delta_pp)}">${signed(s.delta_pp)} pp ${esc(s.direction)}</span></div>`;
          }).join("") || `<div class="muted">No movement data.</div>`}
        </div>
        <div class="panel" style="margin-top:16px"><h2>Market confidence</h2>
          ${svgBars(Object.entries(confCounts).map(([k, v]) => ({
            label: k, value: v, color: k === "High" ? "#4f9dff" : k === "Medium" ? "#d4a017" : "#e5534b"
          })), { h: 180 })}
        </div>
      </div>
    </div>`;

  const tbody = $("#match-table tbody");
  function applyFilters() {
    const st = $("#f-stage").value, status = $("#f-status").value, mk = $("#f-market").value;
    const rows = matches.filter((m) =>
      (!st || m.stage === st) &&
      (!status || (m.top_recommendation && m.top_recommendation.status === status)) &&
      (!mk || (m.top_recommendation && m.top_recommendation.market_type === mk)));
    tbody.innerHTML = rows.map(matchRowHTML).join("");
    tbody.querySelectorAll("tr").forEach((tr) =>
      tr.addEventListener("click", () => (location.hash = `#/match/${tr.dataset.id}`)));
  }
  ["f-stage", "f-status", "f-market"].forEach((id) => $("#" + id).addEventListener("change", applyFilters));
  applyFilters();
  $("#value-cards").querySelectorAll("[data-id]").forEach((c) =>
    c.addEventListener("click", () => (location.hash = `#/match/${c.dataset.id}`)));
}

function valueCardHTML(m) {
  const r = m.top_recommendation;
  return `<div class="value-card" data-id="${m.match_id}" style="cursor:pointer">
    <div class="teams">${esc(m.home_team)} v ${esc(m.away_team)}</div>
    <div class="muted" style="font-size:12px">${esc(m.stage)} · ${fmtTime(m.kickoff_utc)}</div>
    <div class="sel"><span class="badge ${r.status}">${r.status}</span> ${esc(r.market_type)} · ${esc(r.selection)}</div>
    <div class="stats">
      <span>Edge <b>${num(m.best_edge_pct, 1)}pp</b></span>
      <span>EV <b class="${cls(m.best_ev_pct)}">${pct(m.best_ev_pct)}</b></span>
      <span>@ <b>${num(r.offered_odds)}</b> ${esc(r.bookmaker)}</span>
    </div></div>`;
}

function matchRowHTML(m) {
  const r = m.top_recommendation;
  return `<tr data-id="${m.match_id}">
    <td>${m.status === "Final" ? "<span class='muted'>FT " + (m.score || "") + "</span>" : fmtTime(m.kickoff_utc)}</td>
    <td>${esc(m.home_team)} <span class="muted">v</span> ${esc(m.away_team)}</td>
    <td class="muted">${esc(m.stage)}</td>
    <td>${r ? `<span class="badge ${r.status}">${r.status}</span> ${esc(r.market_type)}/${esc(r.selection)}` : "—"}</td>
    <td class="num">${num(m.best_edge_pct, 1)}</td>
    <td class="num ${cls(m.best_ev_pct)}">${pct(m.best_ev_pct)}</td>
    <td>${m.confidence_band ? `<span class="badge ${m.confidence_band}">${m.confidence_band}</span>` : "—"}</td>
    <td class="muted">${m.line_movement_summary ? esc(m.line_movement_summary.direction) : "—"}</td>
  </tr>`;
}

// ---------- Recommendations ----------
async function renderRecommendations() {
  app.innerHTML = `<div class="loading">Loading recommendations…</div>`;
  const data = await api("/api/v1/recommendations");
  setMeta(data.meta);
  app.innerHTML = `${DISCLAIMER}
    <div class="filters">
      <label>Status <select id="rf-status"><option value="">All</option>
        <option value="bet">Bet</option><option value="lean">Lean</option><option value="pass">Pass</option></select></label>
      <label>Min edge (pp) <input id="rf-edge" type="number" step="0.5" style="width:80px"/></label>
      <label>Min EV (%) <input id="rf-ev" type="number" step="0.5" style="width:80px"/></label>
      <label>Market <select id="rf-market"><option value="">All</option>
        <option>1X2</option><option>O/U</option><option>BTTS</option></select></label>
    </div>
    <div class="panel"><table id="rec-table"><thead><tr>
      <th>Match</th><th>Market</th><th>Selection</th><th>Book</th>
      <th class="num">Odds</th><th class="num">Fair</th><th class="num">Model</th>
      <th class="num">Mkt</th><th class="num">Edge</th><th class="num">EV</th>
      <th>Status</th><th>Conf.</th><th class="num">Stake</th>
    </tr></thead><tbody></tbody></table></div>`;

  async function load() {
    const p = new URLSearchParams();
    if ($("#rf-status").value) p.set("status", $("#rf-status").value);
    if ($("#rf-edge").value) p.set("min_edge", $("#rf-edge").value);
    if ($("#rf-ev").value) p.set("min_ev", $("#rf-ev").value);
    if ($("#rf-market").value) p.set("market_type", $("#rf-market").value);
    const d = await api("/api/v1/recommendations?" + p.toString());
    $("#rec-table tbody").innerHTML = d.recommendations.map((r) => `
      <tr data-id="${r.match_id}">
        <td>${esc(r.home)} v ${esc(r.away)}</td>
        <td class="muted">${esc(r.market_type)}</td><td>${esc(r.selection)}</td>
        <td class="muted">${esc(r.bookmaker)}</td>
        <td class="num">${num(r.offered_odds)}</td>
        <td class="num muted">${num(r.fair_odds)}</td>
        <td class="num">${pct(r.model_prob * 100, 0)}</td>
        <td class="num muted">${pct(r.market_prob_novig * 100, 0)}</td>
        <td class="num">${num(r.edge_pct_points, 1)}</td>
        <td class="num ${cls(r.expected_value_pct)}">${pct(r.expected_value_pct)}</td>
        <td><span class="badge ${r.recommendation_status}">${r.recommendation_status}</span></td>
        <td><span class="badge ${r.confidence_band}">${r.confidence_band}</span></td>
        <td class="num">${num(r.stake_fraction, 2)}</td>
      </tr>`).join("") || `<tr><td colspan="13" class="muted">No recommendations match.</td></tr>`;
    $("#rec-table tbody").querySelectorAll("tr[data-id]").forEach((tr) =>
      tr.addEventListener("click", () => (location.hash = `#/match/${tr.dataset.id}`)));
  }
  ["rf-status", "rf-edge", "rf-ev", "rf-market"].forEach((id) => $("#" + id).addEventListener("input", load));
  load();
}

// ---------- Match detail ----------
function fmtStat(v) {
  if (v == null) return "—";
  const n = Number(v);
  return Number.isInteger(n) ? n : parseFloat(n.toFixed(2));
}

function resultPanelHTML(d) {
  const r = d.result, h = d.home_team.team_name, a = d.away_team.team_name;
  const sh = (r.stats && r.stats.home) || {}, sa = (r.stats && r.stats.away) || {};
  const goalIcon = (t) => (t === "penalty" ? "⚽(P)" : t === "own_goal" ? "⚽(OG)" : "⚽");
  const scorers = (side) =>
    r.scorers.filter((s) => s.side === side)
      .map((s) => `<div>${goalIcon(s.type)} ${esc(s.player || "?")} <span class="muted">${esc(s.minute || "")}</span></div>`)
      .join("") || `<div class="muted">—</div>`;

  const rows = [["Shots", "shots"], ["On target", "shots_on_target"],
    ["Possession %", "possession_pct"], ["Corners", "corners"], ["Fouls", "fouls"],
    ["Offsides", "offsides"], ["xG (est.)", "xg"]];
  const cmp = rows.map(([label, key]) => {
    const hv = sh[key], av = sa[key];
    if (hv == null && av == null) return "";
    const tot = (hv || 0) + (av || 0) || 1;
    return `<div class="cmp">
      <span class="cmp-v">${fmtStat(hv)}</span>
      <span class="cmp-l">${label}</span>
      <span class="cmp-v">${fmtStat(av)}</span>
      <div class="cmp-bar"><span style="width:${((hv || 0) / tot) * 100}%"></span></div>
    </div>`;
  }).join("");

  return `<div class="panel" style="margin-top:16px">
    <h2>Full-time result &amp; statistics</h2>
    <div class="ft-score"><span>${esc(h)}</span><b>${r.home_goals} – ${r.away_goals}</b><span>${esc(a)}</span></div>
    <div class="ft-scorers">
      <div><h3>Goals</h3>${scorers("home")}</div>
      <div style="text-align:right"><h3>Goals</h3>${scorers("away")}</div>
    </div>
    ${r.stats ? `<h3>Match statistics</h3>${cmp}
      <div class="muted" style="font-size:11px;margin-top:6px">xG is a shot-based estimate (ESPN provides no official xG).</div>` : ""}
    ${r.cards.length ? `<h3>Cards</h3><div class="muted" style="font-size:12px">${
      r.cards.map((c) => `${c.type === "red" ? "🟥" : "🟨"} ${esc(c.player || "?")} ${esc(c.minute || "")}`).join(" · ")}</div>` : ""}
  </div>`;
}

async function renderMatch(id) {
  app.innerHTML = `<div class="loading">Loading match…</div>`;
  const d = await api(`/api/v1/matches/${id}`);
  if (d.error) { app.innerHTML = `<div class="loading">${esc(d.error)}</div>`; return; }
  setMeta(d.meta);
  const mo = d.model || {};
  const ph = (mo.prob_home_win || 0) * 100, pd = (mo.prob_draw || 0) * 100, pa = (mo.prob_away_win || 0) * 100;

  const fairRows = d.fair_vs_market.sort((a, b) =>
    (a.market_type + a.selection).localeCompare(b.market_type + b.selection));

  // line movement chart — 1X2 implied for the primary bookmaker (book-agnostic)
  const series = [];
  const colorFor = { Home: "#4f9dff", Draw: "#6b7686", Away: "#f0a35e" };
  const lcp = d.line_chart_points || {};
  const books = [...new Set(Object.keys(lcp).map((k) => k.split(" ").slice(0, -1).join(" ")))];
  const primaryBook = books[0] || "";
  Object.entries(lcp).forEach(([key, pts]) => {
    const sel = key.split(" ").slice(-1)[0];
    const book = key.split(" ").slice(0, -1).join(" ");
    if (book !== primaryBook) return;
    series.push({
      name: key, color: colorFor[sel] || "#999",
      points: pts.map((p) => ({ x: new Date(p.t).getTime(), y: p.implied })),
    });
  });

  const feats = d.features.filter((f) => f.feature_value != null);
  const featGroups = {};
  feats.forEach((f) => { (featGroups[f.feature_group] ||= []).push(f); });

  app.innerHTML = `
    <a class="back" href="#/dashboard">← Dashboard</a>
    <div class="panel">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px">
        <h2 style="margin:0;font-size:18px;text-transform:none;color:var(--text)">
          ${esc(d.home_team.team_name)} <span class="muted">vs</span> ${esc(d.away_team.team_name)}</h2>
        <span class="muted">${esc(d.match.stage)}${d.match.group_name ? " · Group " + d.match.group_name : ""}
          · ${fmtTime(d.match.kickoff_utc)} · ${esc(d.match.venue)}, ${esc(d.match.city)} · ${esc(d.match.weather || "")}
          ${d.match.score ? " · <b>FT " + d.match.score + "</b>" : ""}</span>
      </div>
      ${d.risk_flags.map((f) => `<div class="flag" style="margin-top:10px">⚠ ${esc(f)}</div>`).join("")}
    </div>

    ${d.result ? resultPanelHTML(d) : ""}

    <div class="grid two-col" style="margin-top:16px">
      <div class="panel">
        <h2>1X2 model probabilities</h2>
        <div class="prob-bar">
          <div class="h" style="width:${ph}%">${ph.toFixed(0)}%</div>
          <div class="d" style="width:${pd}%">${pd.toFixed(0)}%</div>
          <div class="a" style="width:${pa}%">${pa.toFixed(0)}%</div>
        </div>
        <div class="legend"><span><i style="background:#4f9dff"></i>${esc(d.home_team.team_name)} win</span>
          <span><i style="background:#6b7686"></i>Draw</span>
          <span><i style="background:#f0a35e"></i>${esc(d.away_team.team_name)} win</span></div>

        <h3>Expected goals & key markets</h3>
        <div class="xg">
          <div><div class="v">${num(mo.lambda_home)}</div><div class="lbl">xG ${esc(d.home_team.team_name)}</div></div>
          <div class="muted">–</div>
          <div><div class="v">${num(mo.lambda_away)}</div><div class="lbl">xG ${esc(d.away_team.team_name)}</div></div>
        </div>
        <dl class="kv">
          <dt>Over 1.5</dt><dd>${pct((mo.prob_over_15||0)*100)}</dd>
          <dt>Over 2.5</dt><dd>${pct((mo.prob_over_25||0)*100)}</dd>
          <dt>Over 3.5</dt><dd>${pct((mo.prob_over_35||0)*100)}</dd>
          <dt>BTTS yes</dt><dd>${pct((mo.prob_btts_yes||0)*100)}</dd>
          <dt>Confidence</dt><dd>${num((mo.confidence_score||0)*100,0)}%</dd>
        </dl>
      </div>

      <div class="panel">
        <h2>Strength comparison</h2>
        <dl class="kv">
          <dt>Elo ${esc(d.home_team.team_name)}</dt><dd>${num(d.strength_comparison.elo_home,0)}</dd>
          <dt>Elo ${esc(d.away_team.team_name)}</dt><dd>${num(d.strength_comparison.elo_away,0)}</dd>
          <dt>Elo difference</dt><dd>${signed(d.strength_comparison.elo_diff,0)}</dd>
          <dt>FIFA rank</dt><dd>#${d.strength_comparison.rank_home} / #${d.strength_comparison.rank_away}</dd>
          <dt>Rest days</dt><dd>${d.match.rest_days_home} / ${d.match.rest_days_away}</dd>
        </dl>
      </div>
    </div>

    <div class="grid two-col" style="margin-top:16px">
      <div class="panel">
        <h2>Fair odds vs best market price</h2>
        <table><thead><tr><th>Market</th><th>Selection</th>
          <th class="num">Best odds</th><th>Book</th></tr></thead><tbody>
          ${fairRows.map((r) => `<tr><td class="muted">${esc(r.market_type)}</td>
            <td>${esc(r.selection)}</td><td class="num">${num(r.decimal_odds)}</td>
            <td class="muted">${esc(r.bookmaker)}</td></tr>`).join("")}
        </tbody></table>
      </div>
      <div class="panel">
        <h2>Recommendations</h2>
        <table><thead><tr><th>Mkt</th><th>Sel</th><th class="num">Edge</th>
          <th class="num">EV</th><th>Status</th></tr></thead><tbody>
          ${d.recommendations.map((r) => `<tr>
            <td class="muted">${esc(r.market_type)}</td><td>${esc(r.selection)}</td>
            <td class="num">${num(r.edge_pct_points,1)}</td>
            <td class="num ${cls(r.expected_value_pct)}">${pct(r.expected_value_pct)}</td>
            <td><span class="badge ${r.recommendation_status}">${r.recommendation_status}</span></td>
          </tr>`).join("")}
        </tbody></table>
      </div>
    </div>

    <div class="panel" style="margin-top:16px">
      <h2>Market movement — 1X2 implied probability${primaryBook ? " (" + esc(primaryBook) + ")" : ""}</h2>
      ${series.length ? svgLine(series, { yLabel: "implied probability" }) : '<div class="muted">No line data.</div>'}
      <div class="legend"><span><i style="background:#4f9dff"></i>Home</span>
        <span><i style="background:#6b7686"></i>Draw</span><span><i style="background:#f0a35e"></i>Away</span></div>
    </div>

    <div class="panel" style="margin-top:16px">
      <h2>Feature contributions (explainability)</h2>
      <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(220px,1fr))">
        ${Object.entries(featGroups).map(([grp, items]) => `<div>
          <h3 style="text-transform:capitalize">${esc(grp)}</h3>
          ${items.map((f) => `<div style="display:flex;justify-content:space-between;font-size:12px;padding:2px 0">
            <span class="muted">${esc(f.feature_name)}</span><b>${num(f.feature_value,2)}</b></div>`).join("")}
        </div>`).join("")}
      </div>
    </div>`;
}

// ---------- Performance lab (recommendation results from the bet log) ----------
const resultBadge = (s) => s === "Win" ? '<span class="badge Bet">✓ Win</span>'
  : s === "Loss" ? '<span class="badge Low">✗ Loss</span>'
  : '<span class="badge Pass">— Void</span>';

function recBreakdown(obj) {
  return `<table><thead><tr><th>Group</th><th class="num">Picks</th><th class="num">Settled</th>
    <th class="num">Win%</th><th class="num">ROI%</th><th class="num">Profit</th><th class="num">Avg edge</th>
    </tr></thead><tbody>${Object.entries(obj).map(([k, v]) => `<tr>
      <td>${esc(k)}</td><td class="num">${v.logged}</td><td class="num">${v.settled}</td>
      <td class="num">${v.settled ? pct(v.win_rate*100,0) : "—"}</td>
      <td class="num ${v.settled ? cls(v.roi_pct) : ""}">${v.settled ? pct(v.roi_pct) : "—"}</td>
      <td class="num ${cls(v.profit_units)}">${signed(v.profit_units,2)}</td>
      <td class="num">${num(v.avg_edge_pct,1)}</td></tr>`).join("")}</tbody></table>`;
}

async function renderPerformance() {
  app.innerHTML = `<div class="loading">Loading recommendation results…</div>`;
  const rs = await api("/api/v1/recommendation-stats");
  setMeta(rs.meta);
  const roi = (rs.roi_curve || []).map((p) => ({ x: p.n, y: p.cum }));
  const rel = rs.reliability || [];

  const note = rs.logged === 0
    ? `<div class="disclaimer">No recommendations logged yet — run <b>↻ Refresh data</b> while matches are still upcoming so picks get recorded.</div>`
    : rs.settled === 0
    ? `<div class="disclaimer">${rs.pending} pick(s) logged and awaiting results. Picks are recorded when first flagged (pre-kickoff) and graded once the match finishes — hit <b>↻ Refresh data</b> after more matches play.</div>`
    : "";

  app.innerHTML = `${DISCLAIMER}
    <div class="panel"><h2>Recommendation results</h2>
      ${note}
      <div class="stat-grid">
        ${stat("Logged picks", rs.logged)}
        ${stat("Settled", rs.settled)}
        ${stat("Pending", rs.pending)}
        ${stat("Wins", rs.wins)}
        ${stat("Losses", rs.losses)}
        ${stat("Win rate", rs.settled ? pct(rs.win_rate*100,1) : "—")}
        ${stat("ROI", rs.settled ? pct(rs.roi_pct,1) : "—", rs.roi_pct)}
        ${stat("Profit (units)", signed(rs.profit_units,2), rs.profit_units)}
        ${stat("Avg edge", num(rs.avg_edge_pct,1)+"pp")}
      </div>
      <div style="margin-top:12px">
        <button class="btn" id="exp-json">Export JSON</button>
        <button class="btn" id="exp-csv">Export CSV</button>
      </div>
    </div>

    <div class="grid two-col" style="margin-top:16px">
      <div class="panel"><h2>By recommendation type</h2>${recBreakdown(rs.by_status)}</div>
      <div class="panel"><h2>By market</h2>${recBreakdown(rs.by_market)}</div>
    </div>

    <div class="grid two-col" style="margin-top:16px">
      <div class="panel"><h2>Cumulative P/L (settled picks)</h2>
        ${roi.length ? svgLine([{ name: "pnl", color: "#4f9dff", points: roi }], { yLabel: "units" }) : '<div class="muted">No settled picks yet.</div>'}
      </div>
      <div class="panel"><h2>Model reliability (settled picks)</h2>
        ${rel.length ? svgReliability(rel) : '<div class="muted">Not enough settled picks yet.</div>'}
        <div class="muted" style="font-size:12px">Predicted vs actual win rate. Brier ${rs.brier ?? "—"} · log-loss ${rs.log_loss ?? "—"}.</div>
      </div>
    </div>

    <div class="panel" style="margin-top:16px"><h2>Recently graded picks</h2>
      ${rs.recent.length ? `<table><thead><tr><th>Match</th><th>Market</th><th>Pick</th>
        <th class="num">Odds</th><th>Result</th><th class="num">P/L</th></tr></thead><tbody>
        ${rs.recent.map((r) => `<tr><td>${esc(r.home)} v ${esc(r.away)}</td>
          <td class="muted">${esc(r.market_type)}</td><td>${esc(r.selection)}</td>
          <td class="num">${num(r.offered_odds)}</td><td>${resultBadge(r.result_status)}</td>
          <td class="num ${cls(r.pnl_units)}">${signed(r.pnl_units,2)}</td></tr>`).join("")}
        </tbody></table>` : '<div class="muted">No graded picks yet.</div>'}
    </div>

    <div class="panel" style="margin-top:16px"><h2>Open picks (awaiting result)</h2>
      ${rs.upcoming.length ? `<table><thead><tr><th>Match</th><th>Market</th><th>Pick</th>
        <th class="num">Odds</th><th class="num">Edge</th><th>Type</th><th>Kickoff</th></tr></thead><tbody>
        ${rs.upcoming.map((r) => `<tr><td>${esc(r.home)} v ${esc(r.away)}</td>
          <td class="muted">${esc(r.market_type)}</td><td>${esc(r.selection)}</td>
          <td class="num">${num(r.offered_odds)}</td><td class="num">${num(r.edge_pct,1)}</td>
          <td><span class="badge ${r.status}">${r.status}</span></td>
          <td class="muted">${fmtTime(r.kickoff_utc)}</td></tr>`).join("")}
        </tbody></table>` : '<div class="muted">No open picks.</div>'}
    </div>`;

  $("#exp-json").addEventListener("click", () => download("recommendation_stats.json", JSON.stringify(rs, null, 2), "application/json"));
  $("#exp-csv").addEventListener("click", () => download("recommendations_by_market.csv", recCsv(rs.by_market), "text/csv"));
}

function recCsv(byMarket) {
  const head = "market,logged,settled,win_rate,roi_pct,profit_units,avg_edge_pct";
  const rows = Object.entries(byMarket).map(([k, v]) =>
    [k, v.logged, v.settled, v.win_rate.toFixed(4), v.roi_pct.toFixed(2),
     v.profit_units.toFixed(2), v.avg_edge_pct.toFixed(2)].join(","));
  return [head, ...rows].join("\n");
}

// ---------- Refresh button ----------
async function doRefresh() {
  const btn = $("#refresh-btn");
  if (!btn || btn.disabled) return;
  btn.disabled = true;
  const orig = btn.textContent;
  btn.textContent = "↻ Refreshing…";
  const fr = $("#freshness");
  const fOrig = fr ? fr.textContent : "";
  if (fr) fr.textContent = "Fetching latest data… (~30s)";
  try {
    const res = await fetch("/api/v1/refresh", { method: "POST" }).then((r) => r.json());
    if (res.error) { alert("Refresh failed: " + res.error); if (fr) fr.textContent = fOrig; }
    else { router(); }  // re-render current view with fresh data + meta
  } catch (e) {
    alert("Refresh failed: " + e);
    if (fr) fr.textContent = fOrig;
  } finally {
    btn.disabled = false;
    btn.textContent = orig;
  }
}
window.addEventListener("DOMContentLoaded", () => {
  const b = $("#refresh-btn");
  if (b) b.addEventListener("click", doRefresh);
});

function stat(label, value, color) {
  const c = color == null ? "" : cls(color);
  return `<div><div class="metric-big ${c}">${value}</div><div class="metric-lbl">${label}</div></div>`;
}

function histogram(values, lo, hi, bins) {
  const out = Array.from({ length: bins }, (_, i) => {
    const a = lo + (i / bins) * (hi - lo), b = lo + ((i + 1) / bins) * (hi - lo);
    return { label: `${a.toFixed(0)}`, value: 0, color: a >= 0 ? "#2ea44f" : "#e5534b" };
  });
  values.forEach((v) => {
    let idx = Math.floor(((v - lo) / (hi - lo)) * bins);
    idx = Math.max(0, Math.min(bins - 1, idx));
    out[idx].value++;
  });
  return out;
}

function toCsv(byMarket) {
  const head = "market,total_bets,win_rate,roi_pct,avg_edge_pct,avg_clv_pct";
  const rows = Object.entries(byMarket).map(([k, v]) =>
    [k, v.total_bets, v.win_rate.toFixed(4), v.roi_pct.toFixed(2), v.avg_edge_pct.toFixed(2), v.avg_clv_pct.toFixed(2)].join(","));
  return [head, ...rows].join("\n");
}

function download(name, content, type) {
  const blob = new Blob([content], { type });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = name; a.click();
  URL.revokeObjectURL(a.href);
}

// ---------- router ----------
function router() {
  const hash = location.hash || "#/dashboard";
  const [, route, arg] = hash.split("/");
  document.querySelectorAll("nav a").forEach((a) =>
    a.classList.toggle("active", a.dataset.route === route));
  if (route === "match" && arg) return renderMatch(arg);
  if (route === "recommendations") return renderRecommendations();
  if (route === "performance") return renderPerformance();
  return renderDashboard();
}
window.addEventListener("hashchange", router);
window.addEventListener("DOMContentLoaded", router);
