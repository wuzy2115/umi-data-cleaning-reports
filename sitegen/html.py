# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Hand-written HTML template for the S8 review site (spec §10). No templating
engine dependency — a handful of section kinds cover every page build.py needs:
heading, free text/HTML, table, plotly figure, media wall, stat tiles. Plotly
is embedded via CDN + inline `Plotly.newPlot(...)`; nothing here talks to a
Plotly Python package (figures are plain dicts the caller builds by hand).

Visual design: an "instrument ledger" skin — graphite-ink sticky topbar with a
copper signal accent, warm paper background, monospace-forward data typography
(mono kickers/nav/numerals, tabular-nums tables), KPI stat tiles, and plotly
figures themed through a non-destructive template merge. All tokens live as
CSS custom properties on `:root`. Sections are grouped into cards by heading
(`{"h": ...}` starts a new card); this is purely a rendering-time grouping and
does not require build.py to change its flat `sections` lists. Pages stay
fully self-contained: system font stacks only, single plotly CDN script tag,
relative asset paths.
"""
from __future__ import annotations

import html as _html
import json
from pathlib import Path

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

FONT_STACK = (
    '-apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, '
    '"PingFang SC", "Microsoft YaHei", sans-serif'
)

MONO_STACK = (
    'ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, '
    '"Liberation Mono", "PingFang SC", "Microsoft YaHei", monospace'
)

# Categorical colorway matched to the CSS design tokens (copper first so
# single-trace histograms pick up the site accent).
_PLOTLY_COLORWAY = ["#b45309", "#31708e", "#5f7a3a", "#8a4f7d", "#50555e", "#c98a2e"]

_PLOTLY_AXIS = {
    "gridcolor": "#ebe8df",
    "linecolor": "#d3cfc2",
    "zerolinecolor": "#d3cfc2",
    "ticks": "outside",
    "tickcolor": "#d3cfc2",
}

# Shallow-merge base for every Plotly figure's layout: font/paper/plot bg +
# tighter margins at the top level, plus a `template` carrying axis/colorway
# defaults (plotly applies template values underneath the caller's layout).
# `Object.assign(DEFAULT, layout)` in the emitted JS means any key the
# caller's `layout` dict already sets wins — this only fills in gaps, never
# clobbers explicit layout choices.
_DEFAULT_PLOTLY_LAYOUT = {
    "font": {"family": FONT_STACK, "size": 12, "color": "#42454c"},
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "margin": {"t": 44, "r": 20},
    "template": {
        "layout": {
            "colorway": _PLOTLY_COLORWAY,
            "xaxis": _PLOTLY_AXIS,
            "yaxis": _PLOTLY_AXIS,
            "title": {"font": {"family": FONT_STACK, "size": 13, "color": "#565a63"}, "x": 0},
            "legend": {"font": {"size": 11}, "bgcolor": "rgba(0,0,0,0)"},
        }
    },
}
_DEFAULT_PLOTLY_LAYOUT_JSON = json.dumps(_DEFAULT_PLOTLY_LAYOUT)

_CSS_TEMPLATE = """
<style>
  :root {
    --bg: #f4f3ee;
    --surface: #fffefb;
    --surface-2: #f7f5ee;
    --border: #e5e2d8;
    --border-strong: #d3cfc2;
    --ink: #1d1f23;
    --ink-2: #565a63;
    --ink-3: #8b8e96;
    --accent: #b45309;
    --accent-deep: #8a3f06;
    --accent-soft: #f6ead9;
    --keep: #1a7f42;
    --drop: #b42318;
    --head-bg: #17191d;
    --head-ink: #ecebe6;
    --head-muted: #9a9ca3;
    --radius: 8px;
    --shadow: 0 1px 2px rgba(29, 27, 20, 0.04), 0 1px 5px rgba(29, 27, 20, 0.05);
    --shadow-lift: 0 6px 18px rgba(29, 27, 20, 0.09);
    --page-w: 1180px;
    --sans: %%SANS%%;
    --mono: %%MONO%%;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    font-family: var(--sans);
    font-size: 14.5px;
    line-height: 1.6;
    color: var(--ink);
    background: var(--bg);
    -webkit-font-smoothing: antialiased;
  }
  a { color: var(--accent); }
  a:hover { color: var(--accent-deep); }
  code, pre { font-family: var(--mono); }
  pre {
    font-size: 0.76rem; line-height: 1.6; color: var(--ink-2);
    background: var(--surface-2); border: 1px solid var(--border); border-radius: 6px;
    padding: 0.9rem 1.05rem; overflow-x: auto;
  }
  ol, ul { padding-left: 1.4rem; }
  li { margin: 0.15rem 0; }
  .muted { color: var(--ink-2); }

  /* ---- topbar --------------------------------------------------------- */
  .site-header {
    position: sticky; top: 0; z-index: 50;
    background: var(--head-bg);
  }
  .site-header::before {
    content: ""; display: block; height: 2px;
    background: linear-gradient(90deg, var(--accent) 0%, #d97a2b 45%, rgba(217, 122, 43, 0) 100%);
  }
  .header-inner {
    max-width: var(--page-w); margin: 0 auto; height: 54px;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 1.25rem; gap: 1.25rem;
  }
  .site-title {
    font-family: var(--mono); font-weight: 700; font-size: 0.82rem;
    letter-spacing: 0.06em; color: var(--head-ink); text-decoration: none;
    white-space: nowrap; display: flex; align-items: center; gap: 0.6rem;
  }
  .site-title::before {
    content: ""; width: 9px; height: 9px; flex: none;
    background: var(--accent); border-radius: 2px;
    box-shadow: 0 0 0 3px rgba(180, 83, 9, 0.28);
  }
  .site-title:hover { color: #ffffff; }
  .top-nav { display: flex; gap: 2px; overflow-x: auto; scrollbar-width: none; }
  .top-nav::-webkit-scrollbar { display: none; }
  .top-nav a {
    font-family: var(--mono); font-size: 0.67rem; letter-spacing: 0.09em;
    text-transform: uppercase; color: var(--head-muted); text-decoration: none;
    white-space: nowrap; padding: 0.45rem 0.6rem; border-radius: 5px;
    transition: color 0.12s ease, background-color 0.12s ease;
  }
  .top-nav a:hover { color: var(--head-ink); background: rgba(255, 255, 255, 0.07); }
  .top-nav a.active { color: #f0b476; background: rgba(180, 83, 9, 0.20); }

  /* ---- page frame ------------------------------------------------------ */
  main.container { max-width: var(--page-w); margin: 0 auto; padding: 2rem 1.25rem 3rem; }
  .page-head { margin: 0.25rem 0 1.5rem; }
  .kicker {
    font-family: var(--mono); font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.16em; text-transform: uppercase; color: var(--accent);
    margin-bottom: 0.35rem;
  }
  h1.page-title { font-size: 1.45rem; font-weight: 700; letter-spacing: -0.01em; margin: 0; }
  .site-footer {
    max-width: var(--page-w); margin: 0 auto; padding: 0 1.25rem 2.5rem;
    font-family: var(--mono); font-size: 0.63rem; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--ink-3);
  }

  /* ---- cards ----------------------------------------------------------- */
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); box-shadow: var(--shadow);
    padding: 1.15rem 1.35rem 1.25rem; margin-bottom: 1rem;
  }
  .card h2 {
    font-family: var(--mono); font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.11em; text-transform: uppercase; color: var(--ink-2);
    margin: 0 0 0.95rem; display: flex; align-items: baseline; gap: 0.6rem;
  }
  .card h2::before {
    content: ""; width: 0.55rem; height: 2px; flex: none;
    background: var(--accent); align-self: center;
  }
  .card > *:last-child { margin-bottom: 0; }

  /* ---- stat tiles ------------------------------------------------------ */
  .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(155px, 1fr)); gap: 10px; }
  .stat {
    position: relative; overflow: hidden;
    background: var(--surface-2); border: 1px solid var(--border); border-radius: 6px;
    padding: 0.8rem 0.95rem 0.75rem;
  }
  .stat::after {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--border-strong);
  }
  .stat-keep::after { background: var(--keep); }
  .stat-drop::after { background: var(--drop); }
  .stat-accent::after { background: var(--accent); }
  .stat-label {
    font-family: var(--mono); font-size: 0.62rem; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--ink-3); margin-bottom: 0.3rem;
  }
  .stat-value {
    font-family: var(--mono); font-size: 1.45rem; font-weight: 700;
    letter-spacing: -0.02em; line-height: 1.15; color: var(--ink);
    font-variant-numeric: tabular-nums;
  }
  .stat-keep .stat-value { color: var(--keep); }
  .stat-drop .stat-value { color: var(--drop); }
  .stat-accent .stat-value { color: var(--accent-deep); }
  .stat-sub { font-family: var(--mono); font-size: 0.68rem; color: var(--ink-2); margin-top: 0.28rem; }

  /* ---- tables ---------------------------------------------------------- */
  .table-wrap { width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch; }
  table { border-collapse: collapse; width: 100%; font-size: 0.82rem; font-variant-numeric: tabular-nums; }
  thead th {
    font-family: var(--mono); font-size: 0.65rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-3);
    text-align: left; padding: 0.45rem 0.7rem;
    border-bottom: 1px solid var(--border-strong);
    cursor: pointer; user-select: none; white-space: nowrap;
  }
  thead th::after { content: "\\2195"; margin-left: 0.4em; opacity: 0.35; }
  thead th[data-asc="true"]::after { content: "\\2191"; opacity: 1; color: var(--accent); }
  thead th[data-asc="false"]::after { content: "\\2193"; opacity: 1; color: var(--accent); }
  thead th:hover { color: var(--accent); }
  tbody td { padding: 0.42rem 0.7rem; border-bottom: 1px solid var(--border); }
  tbody tr:last-child td { border-bottom: none; }
  tbody tr:hover td { background: var(--surface-2); }
  td[data-num] { text-align: right; font-family: var(--mono); font-size: 0.78rem; }
  td[data-bool] { font-family: var(--mono); font-size: 0.75rem; }
  td[data-bool="true"] { color: var(--keep); }
  td[data-bool="false"] { color: var(--drop); }
  td[data-bool]::before {
    content: ""; display: inline-block; width: 6px; height: 6px; border-radius: 50%;
    margin-right: 0.45em; vertical-align: 1px; background: currentColor;
  }

  /* ---- plotly ---------------------------------------------------------- */
  .plotly-fig { width: 100%; height: 420px; }

  /* ---- media walls ----------------------------------------------------- */
  .media-wall { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 12px; }
  .media-item {
    position: relative; background: var(--surface);
    border: 1px solid var(--border); border-radius: 8px; padding: 6px;
    transition: transform 0.12s ease, box-shadow 0.12s ease, border-color 0.12s ease;
  }
  .media-item:hover { transform: translateY(-2px); box-shadow: var(--shadow-lift); border-color: var(--border-strong); }
  .media-item video, .media-item img {
    width: 100%; aspect-ratio: 16 / 9; object-fit: cover; background: #101113;
    border-radius: 5px; display: block;
  }
  .media-item.rep { border-left: 3px solid var(--keep); }
  .media-item.member { border-left: 3px solid var(--drop); }
  .media-item .badge {
    position: absolute; top: 12px; left: 12px;
    font-family: var(--mono); font-size: 0.58rem; font-weight: 700;
    letter-spacing: 0.09em; padding: 0.16rem 0.42rem; border-radius: 3px;
  }
  .media-item.rep .badge { background: var(--keep); color: #ffffff; }
  .media-item.member .badge { background: var(--drop); color: #ffffff; }
  .caption {
    font-family: var(--mono); font-size: 0.66rem; line-height: 1.5;
    color: var(--ink-2); word-break: break-all; margin-top: 0.4rem; padding: 0 2px 2px;
  }

  /* ---- index landing grid ---------------------------------------------- */
  .landing-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
    gap: 12px; counter-reset: pagecard;
  }
  .landing-card {
    position: relative; display: block;
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 0.95rem 1.05rem 0.9rem; text-decoration: none; color: inherit;
    transition: transform 0.12s ease, box-shadow 0.12s ease, border-color 0.12s ease;
  }
  .landing-card::before {
    counter-increment: pagecard; content: counter(pagecard, decimal-leading-zero);
    display: block; font-family: var(--mono); font-size: 0.62rem;
    letter-spacing: 0.1em; color: var(--ink-3); margin-bottom: 0.45rem;
  }
  .landing-card:hover { transform: translateY(-2px); border-color: var(--accent); box-shadow: var(--shadow-lift); }
  .landing-card .t { font-weight: 600; font-size: 0.92rem; color: var(--ink); margin-bottom: 0.25rem; }
  .landing-card .t::after { content: " \\2192"; color: var(--accent); opacity: 0; transition: opacity 0.12s ease; }
  .landing-card:hover .t { color: var(--accent-deep); }
  .landing-card:hover .t::after { opacity: 1; }
  .landing-card .d { font-size: 0.76rem; color: var(--ink-2); line-height: 1.5; }

  /* ---- responsive ------------------------------------------------------ */
  @media (max-width: 768px) {
    .header-inner { height: auto; flex-direction: column; align-items: flex-start; gap: 0.25rem; padding: 0.55rem 1rem 0.6rem; }
    .top-nav { width: 100%; }
    main.container { padding: 1.25rem 0.85rem 2.5rem; }
    .card { padding: 0.95rem 1rem 1rem; }
    .stat-grid { grid-template-columns: repeat(2, 1fr); }
    .stat-value { font-size: 1.2rem; }
    .media-wall { grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); }
    .plotly-fig { height: 320px; }
  }
  @media (prefers-reduced-motion: reduce) {
    * { transition: none !important; }
  }
</style>
"""

_CSS = _CSS_TEMPLATE.replace("%%SANS%%", FONT_STACK).replace("%%MONO%%", MONO_STACK)

_SORT_JS = """
<script>
function sortTable(th) {
  var table = th.closest("table");
  var idx = Array.prototype.indexOf.call(th.parentNode.children, th);
  var tbody = table.querySelector("tbody");
  var rows = Array.prototype.slice.call(tbody.querySelectorAll("tr"));
  var asc = th.dataset.asc !== "true";
  rows.sort(function (a, b) {
    var av = a.children[idx].innerText, bv = b.children[idx].innerText;
    var an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) { return asc ? an - bn : bn - an; }
    return asc ? av.localeCompare(bv) : bv.localeCompare(av);
  });
  Array.prototype.forEach.call(th.parentNode.children, function (o) {
    if (o !== th) { delete o.dataset.asc; }
  });
  th.dataset.asc = asc;
  rows.forEach(function (r) { tbody.appendChild(r); });
}
</script>
"""


def _render_table(rows: list[dict]) -> str:
    if not rows:
        return '<p class="muted"><em>(no rows)</em></p>'
    cols = list(rows[0].keys())
    thead = "".join(f'<th onclick="sortTable(this)">{_html.escape(str(c))}</th>' for c in cols)
    body = []
    for r in rows:
        tds = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, bool):
                attrs = f' data-bool="{"true" if v else "false"}"'
            elif isinstance(v, (int, float)):
                attrs = ' data-num="true"'
            else:
                attrs = ""
            tds.append(f"<td{attrs}>{_html.escape(str(v))}</td>")
        body.append(f"<tr>{''.join(tds)}</tr>")
    table = f"<table><thead><tr>{thead}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    return f'<div class="table-wrap">{table}</div>'


def _render_stats(items: list[dict]) -> str:
    """KPI tile strip: `{"label", "value", "sub"?, "tone"?}` per tile, where
    `tone` in {"keep", "drop", "accent"} colors the value + top edge."""
    tiles = []
    for it in items:
        label = _html.escape(str(it.get("label", "")))
        value = _html.escape(str(it.get("value", "")))
        tone = str(it.get("tone", ""))
        cls = f" stat-{tone}" if tone in ("keep", "drop", "accent") else ""
        sub = str(it.get("sub", "") or "")
        sub_html = f'<div class="stat-sub">{_html.escape(sub)}</div>' if sub else ""
        tiles.append(
            f'<div class="stat{cls}"><div class="stat-label">{label}</div>'
            f'<div class="stat-value">{value}</div>{sub_html}</div>'
        )
    return f'<div class="stat-grid">{"".join(tiles)}</div>'


def _render_media(items: list[dict]) -> str:
    cells = []
    video_ext = (".mp4", ".webm", ".mov", ".m4v")
    for it in items:
        src = str(it.get("src", ""))
        caption = _html.escape(str(it.get("caption", "")))
        cls = str(it.get("cls", ""))
        if src.lower().endswith(video_ext):
            tag = f'<video controls preload="none" src="{_html.escape(src)}"></video>'
        else:
            tag = f'<img loading="lazy" src="{_html.escape(src)}">'
        badge = ""
        if cls == "rep":
            badge = '<span class="badge">REP</span>'
        elif cls == "member":
            badge = '<span class="badge">MEMBER</span>'
        cells.append(f'<div class="media-item {cls}">{badge}{tag}<div class="caption">{caption}</div></div>')
    return f'<div class="media-wall">{"".join(cells)}</div>'


class _PlotlyCounter:
    """Per-page counter so `<div id="plotly-N">` ids restart at 1 for every call
    to `page()` — deterministic output, no cross-page growth."""

    def __init__(self) -> None:
        self.n = 0

    def next_id(self) -> str:
        self.n += 1
        return f"plotly-{self.n}"


def _render_plotly(fig: dict, counter: _PlotlyCounter) -> str:
    div_id = counter.next_id()
    data = json.dumps(fig.get("data", []))
    layout = json.dumps(fig.get("layout", {}))
    return (
        f'<div id="{div_id}" class="plotly-fig"></div>'
        "<script>(function(){"
        f"var _l = Object.assign({_DEFAULT_PLOTLY_LAYOUT_JSON}, {layout});"
        f'Plotly.newPlot("{div_id}", {data}, _l, {{responsive: true}});'
        "})();</script>"
    )


def _render_section(sec: dict, counter: _PlotlyCounter) -> str:
    if "h" in sec:
        return f"<h2>{_html.escape(str(sec['h']))}</h2>"
    if "text" in sec:
        return f"<div>{sec['text']}</div>"
    if "raw" in sec:
        # Emitted verbatim — no wrapper div, unlike {"text"}. For markup that must
        # come out exactly as given: `frozen_page` re-renders an already-published
        # page through this, so a rebuild reproduces it byte for byte.
        return sec["raw"]
    if "table" in sec:
        return _render_table(sec["table"])
    if "stats" in sec:
        return _render_stats(sec["stats"])
    if "plotly" in sec:
        return _render_plotly(sec["plotly"], counter)
    if "media" in sec:
        return _render_media(sec["media"])
    return ""


def _group_sections(sections: list[dict]) -> list[dict]:
    """Group a flat section list into card groups: each `{"h": ...}` section
    starts a new card, all following non-heading sections belong to it (any
    leading content before the first heading gets its own headless card).
    Purely a rendering-time grouping — build.py's section lists are untouched."""
    groups: list[dict] = []
    cur: dict | None = None
    for s in sections:
        if "h" in s:
            if cur is not None:
                groups.append(cur)
            cur = {"h": s["h"], "items": []}
        else:
            if cur is None:
                cur = {"h": None, "items": []}
            cur["items"].append(s)
    if cur is not None:
        groups.append(cur)
    return groups


def _render_group(g: dict, counter: _PlotlyCounter) -> str:
    body = "\n".join(_render_section(s, counter) for s in g["items"])
    if g["h"] is None:
        return f'<div class="card">{body}</div>' if body else ""
    return f'<div class="card"><h2>{_html.escape(str(g["h"]))}</h2>{body}</div>'


def _render_nav(nav: list[tuple[str, str]] | None, current: str | None) -> str:
    if not nav:
        return ""
    links = []
    for href, label in nav:
        cls = ' class="active"' if href == current else ""
        links.append(f'<a href="{_html.escape(href)}"{cls}>{_html.escape(label)}</a>')
    return f'<nav class="top-nav">{"".join(links)}</nav>'


def _render_page_head(title: str) -> str:
    """Page heading block. Titles of the form "<name> — <descriptor>" split
    into a small mono kicker (name) above the descriptor headline; anything
    else renders as a plain headline."""
    kicker, sep, rest = title.partition(" — ")
    if sep and rest:
        return (
            '<div class="page-head">'
            f'<div class="kicker">{_html.escape(kicker)}</div>'
            f'<h1 class="page-title">{_html.escape(rest)}</h1>'
            "</div>"
        )
    return f'<div class="page-head"><h1 class="page-title">{_html.escape(title)}</h1></div>'


def page(
    title: str,
    sections: list[dict],
    nav: list[tuple[str, str]] | None = None,
    current: str | None = None,
    site_name: str | None = None,
) -> str:
    """Render a self-contained HTML page string from `sections`.

    Section kinds: `{"h": str}` heading, `{"text": str}` free HTML/text (wrapped in
    a div), `{"raw": str}` markup emitted verbatim, `{"table": list[dict]}` (keys of
    the first row become the header row), `{"stats": list[dict]}` KPI tiles
    (`{"label", "value", "sub"?, "tone"?}`), `{"plotly": fig_dict}` (embedded via CDN
    + inline `Plotly.newPlot`), `{"media": [{"src", "caption"}]}` (video vs image
    chosen by extension).

    `nav`/`current`/`site_name` are optional: when given, `nav` (a list of
    `(href, label)` pairs) renders as a sticky top nav bar with `current`'s
    entry highlighted, and `site_name` is the header brand link, pointing at
    the first nav entry (the site's home page). All three default to
    no-header-nav for backward compatibility with callers that only pass
    `title`/`sections`.
    """
    counter = _PlotlyCounter()
    groups = _group_sections(sections)
    body = "\n".join(_render_group(g, counter) for g in groups)
    nav_html = _render_nav(nav, current)
    brand = _html.escape(site_name) if site_name else _html.escape(title)
    home = _html.escape(nav[0][0]) if nav else "index.html"
    header = (
        '<header class="site-header"><div class="header-inner">'
        f'<a class="site-title" href="{home}">{brand}</a>'
        f"{nav_html}"
        "</div></header>"
    )
    footer = f'<footer class="site-footer">{brand} · S8 cleaning review · static report</footer>'
    return (
        "<!doctype html>\n"
        '<html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_html.escape(title)}</title>"
        f'<script src="{PLOTLY_CDN}"></script>'
        f"{_CSS}</head>"
        f'<body>{header}<main class="container">'
        f"{_render_page_head(title)}"
        f"{body}</main>{footer}{_SORT_JS}</body></html>"
    )


def write_page(
    out_dir: Path,
    name: str,
    title: str,
    sections: list[dict],
    nav: list[tuple[str, str]] | None = None,
    current: str | None = None,
    site_name: str | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    p.write_text(page(title, sections, nav=nav, current=current, site_name=site_name), encoding="utf-8")
    return p
