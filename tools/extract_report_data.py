# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Reverse-extract `data/report.json` out of the rendered `report.html`.

Provenance note — read this before trusting the JSON. The report page was
originally rendered by an upstream (cosmos) builder from a parquet cleaning
manifest. That manifest no longer exists on any machine we control, so the
rendered page is now the *only* surviving copy of the data. This script reads
that page back into structured JSON so `site/report_page.py` can re-render it
in-tree; the JSON, not the parquet, is henceforth the source of truth.

Consequences of extracting from the rendered page rather than the manifest:

* Stat-tile values are **pre-formatted display strings** ("200,551", "389.2 h",
  "34.7% of total"). The underlying floats are gone — the thousands separators
  and the rounding are baked in. Editing a tile means editing the string.
* Table cells keep their type (int / float / bool / str) because the renderer
  tagged them `data-num` / `data-bool`; that much survives the round trip.
* Only the numbers that were *shown* survive. Anything the upstream builder
  computed but did not print is unrecoverable.

Structure of the page (11 `div.card` blocks, keyed by their `<h2>`): five plain
content cards, two plotly figures, three "Top dup cluster" cards, and two cards
whose body is raw HTML. The dup-cluster cards are not uniform — see
`_CLUSTER_*` below.

Usage:
    python3 tools/extract_report_data.py \
        --html umi400h/v0_study/report.html --out data/report.json
"""
from __future__ import annotations

import argparse
import html as _html
import json
import re
from html.parser import HTMLParser
from pathlib import Path

# Headings are the join key between the rendered page and the schema. Matching
# on them (rather than on card order) means a reordered page still extracts
# correctly, and a renamed heading fails loudly instead of silently mis-keying.
H_HEADLINE = "概况 Headline"
H_BEFORE_AFTER = "清洗前后对照 Before / After"
H_KEEP_RATE = "任务分布偏移 Keep rate by skill"
H_FUNNEL = "Funnel (dropped per filter, priority order)"
H_DROP_REASON = "drop_reason distribution"
H_DEDUP = "去重 Dedup"
H_SPOT_CHECK = "误杀抽检 Manual spot-check (fill in after human review)"
H_PIPELINE = "Pipeline (S1 -> S9)"

_CLUSTER_RE = re.compile(r"^Top dup cluster (\d+) \(size=(\d+)\)$")

# The three dup-cluster cards render differently: the top cluster got a media
# wall (rep video + member stills were exported for it), the other two fell back
# to a bare segment table because no media was exported. Both shapes are carried
# in the JSON under a `kind` discriminator so the builder can reproduce each.
_CLUSTER_MEDIA = "media"
_CLUSTER_TABLE = "table"


def _typed(text: str, attrs: dict[str, str]):
    """Recover a cell's Python type from the render-time markers the table
    renderer left behind (`data-num` / `data-bool`); anything else is a str."""
    if "data-bool" in attrs:
        return attrs["data-bool"] == "true"
    if "data-num" in attrs:
        return float(text) if re.search(r"[.eE]", text) else int(text)
    return text


class _CardParser(HTMLParser):
    """Pulls the widgets out of a single card's inner HTML.

    Emits into `self.blocks` one entry per widget in document order, so a card
    that mixes (say) a heading and a table stays ordered. Only the widget kinds
    this page actually uses are handled: stat tiles, tables, media walls.
    Plotly figures and raw-HTML bodies are picked up separately (see
    `_plotly_figure` / `_raw_div`) because their content is not markup the
    parser can meaningfully walk.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[dict] = []
        self._stats: list[dict] | None = None
        self._stat: dict | None = None
        self._rows: list[dict] | None = None
        self._cols: list[str] = []
        self._row: list | None = None
        self._media: list[dict] | None = None
        self._item: dict | None = None
        self._sink: str | None = None  # which field the next handle_data() feeds
        self._cell_attrs: dict[str, str] = {}
        self._buf: list[str] = []

    # -- helpers ---------------------------------------------------------
    def _flush(self) -> str:
        s = "".join(self._buf).strip()
        self._buf = []
        return s

    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = {k: (v or "") for k, v in attrs_list}
        cls = attrs.get("class", "").split()

        if "stat-grid" in cls:
            self._stats = []
        elif "stat" in cls and self._stats is not None:
            tone = next((c[5:] for c in cls if c.startswith("stat-")), "")
            self._stat = {"label": "", "value": ""}
            if tone:
                self._stat["tone"] = tone
        elif self._stat is not None and cls and cls[0] in ("stat-label", "stat-value", "stat-sub"):
            self._sink = cls[0]
            self._buf = []

        elif tag == "table":
            self._rows, self._cols = [], []
        elif tag == "th" and self._rows is not None:
            self._sink, self._buf = "th", []
        elif tag == "tr" and self._rows is not None and self._cols:
            self._row = []
        elif tag == "td" and self._row is not None:
            self._sink, self._buf, self._cell_attrs = "td", [], attrs

        elif "media-wall" in cls:
            self._media = []
        elif "media-item" in cls:
            self._item = {"src": "", "caption": "", "cls": next((c for c in cls if c != "media-item"), "")}
        elif tag in ("video", "img") and self._item is not None:
            self._item["src"] = attrs.get("src", "")
        elif "caption" in cls and self._item is not None:
            self._sink, self._buf = "caption", []

    def handle_data(self, data: str) -> None:
        if self._sink:
            self._buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._sink == "th" and tag == "th":
            self._cols.append(self._flush())
        elif self._sink == "td" and tag == "td":
            self._row.append(_typed(self._flush(), self._cell_attrs))
        elif self._sink == "caption" and tag == "div":
            self._item["caption"] = self._flush()
        elif self._sink in ("stat-label", "stat-value", "stat-sub") and tag == "div":
            self._stat[self._sink.replace("stat-", "")] = self._flush()
        else:
            # Closing tags for the containers themselves.
            if tag == "tr" and self._row is not None:
                self._rows.append(dict(zip(self._cols, self._row)))
                self._row = None
            elif tag == "table" and self._rows is not None:
                self.blocks.append({"table": self._rows})
                self._rows = None
            elif tag == "div":
                if self._item is not None and self._item["src"]:
                    self._media.append(self._item)
                    self._item = None
                elif self._stat is not None and self._stat["value"]:
                    self._stats.append(self._stat)
                    self._stat = None
                elif self._media is not None:
                    self.blocks.append({"media": self._media})
                    self._media = None
                elif self._stats is not None:
                    self.blocks.append({"stats": self._stats})
                    self._stats = None
            return
        self._sink = None


def _cards(page: str) -> list[tuple[str, str]]:
    """Split the page body into (heading, inner_html) per `div.card`."""
    out = []
    for body in re.findall(r'<div class="card">(.*?)(?=<div class="card">|</main>)', page, re.S):
        m = re.search(r"<h2>(.*?)</h2>", body, re.S)
        out.append((_html.unescape(m.group(1)) if m else "", body))
    return out


def _widgets(card: str) -> list[dict]:
    p = _CardParser()
    p.feed(card)
    p.close()
    return p.blocks


def _stats(card: str) -> list[dict]:
    return next(b["stats"] for b in _widgets(card))


def _table(card: str) -> list[dict]:
    return next(b["table"] for b in _widgets(card))


def _media(card: str) -> list[dict]:
    return next(b["media"] for b in _widgets(card))


def _raw_div(card: str) -> str:
    """Inner HTML of a card whose body is a single raw-HTML `<div>` — i.e. a
    section the original builder passed through as `{"text": ...}`.

    `card` still carries the `</div>` that closes the card itself, so the body's
    own closing tag is the *second-to-last* one: match both and keep neither."""
    m = re.search(r"</h2><div>(.*)</div></div>\s*$", card, re.S)
    return m.group(1).strip()


def _json_after(text: str, marker: str, pos: int = 0) -> tuple[object, int]:
    """Decode the JSON value that begins just after `marker` (skipping the
    separator whitespace), returning it with the offset just past its end.
    `raw_decode` walks the value properly, so nested braces and escaped quotes
    inside the figure JSON are handled — a brace-counting regex would not be."""
    i = text.index(marker, pos) + len(marker)
    while text[i] in " \t":
        i += 1
    return json.JSONDecoder().raw_decode(text, i)


def _plotly_figure(card: str) -> dict:
    """Lift a figure's `data` / `layout` straight out of its inline
    `Plotly.newPlot(...)` call, which the renderer emits inside the card:

        var _l = Object.assign(<renderer defaults>, <the builder's layout>);
        Plotly.newPlot("<id>", <data>, _l, {responsive: true});

    The builder's own layout is the *second* argument to `Object.assign`; the
    first is the design-system default that `site/html.py` re-applies at render
    time, and which therefore must not be baked into the JSON.
    """
    _defaults, end = _json_after(card, "Object.assign(")
    layout, end = _json_after(card, ",", end)
    data, _ = _json_after(card, ",", card.index("Plotly.newPlot(", end))
    return {"data": data, "layout": layout}


def extract(page: str) -> dict:
    cards = dict(_cards(page))
    title = _html.unescape(re.search(r"<title>(.*?)</title>", page).group(1))

    clusters = []
    for heading, card in _cards(page):
        m = _CLUSTER_RE.match(heading)
        if not m:
            continue
        cid, size = int(m.group(1)), int(m.group(2))
        if "media-wall" in card:
            clusters.append({"cid": cid, "size": size, "kind": _CLUSTER_MEDIA, "items": _media(card)})
        else:
            clusters.append({"cid": cid, "size": size, "kind": _CLUSTER_TABLE, "rows": _table(card)})

    spot = _raw_div(cards[H_SPOT_CHECK])
    return {
        "title": title,
        "dataset": title.split(" — ")[0],
        "headline": _stats(cards[H_HEADLINE]),
        "before_after": _table(cards[H_BEFORE_AFTER]),
        "keep_rate_by_skill": _plotly_figure(cards[H_KEEP_RATE]),
        "funnel": _table(cards[H_FUNNEL]),
        "drop_reason": _plotly_figure(cards[H_DROP_REASON]),
        "dedup": _stats(cards[H_DEDUP]),
        "dup_clusters": clusters,
        # The spot-check table ships empty on purpose (a human fills it in), so
        # only its column names survive; the renderer's table widget would print
        # "(no rows)" for an empty list, hence the original emitted raw HTML.
        "spot_check_columns": re.findall(r"<th>(.*?)</th>", spot),
        "pipeline_html": _raw_div(cards[H_PIPELINE]),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--html", type=Path, default=Path("umi400h/v0_study/report.html"))
    ap.add_argument("--out", type=Path, default=Path("data/report.json"))
    args = ap.parse_args(argv)

    data = extract(args.html.read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    walls = sum(c["kind"] == _CLUSTER_MEDIA for c in data["dup_clusters"])
    print(
        f"[done] {args.out}: {len(data['headline'])} headline tiles, "
        f"{len(data['before_after'])} before/after rows, {len(data['funnel'])} funnel rows, "
        f"{len(data['dedup'])} dedup tiles, {len(data['dup_clusters'])} dup clusters "
        f"({walls} media wall, {len(data['dup_clusters']) - walls} table)"
    )


if __name__ == "__main__":
    main()
