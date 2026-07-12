# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""`frozen_page` re-renders a published page from itself, so the only thing that can
go wrong is the parse: a card body that comes back wrong silently corrupts the page.
The contract these tests pin down is that rendering is a fixpoint."""

import pytest

from sitegen import html
from sitegen.frozen_page import _split_cards, build_page, parse_page

NAV = [("index.html", "Overview"), ("report.html", "Report")]


def _render(sections, title="umi400h — probe", current="index.html"):
    return html.page(title, sections, nav=NAV, current=current, site_name="umi400h")


def test_round_trip_is_a_fixpoint(tmp_path):
    """Re-rendering a page built from ordinary sections must reproduce it byte for
    byte — the property `build.py --check` relies on for the frozen run."""
    original = _render([
        {"h": "概况 Headline"},
        {"stats": [{"label": "segments", "value": "200,551", "tone": "keep"}]},
        {"h": "Pipeline (S1 -> S9)"},          # escaped on render, must not double-escape
        {"text": "<ol><li>S1 &amp; S2</li></ol>"},
        {"h": "Curve"},
        {"plotly": {"data": [{"y": [1, 2]}], "layout": {}}},
    ])
    src = tmp_path / "index.html"
    src.write_text(original)

    build_page(src, tmp_path / "out", NAV, "index.html")
    assert (tmp_path / "out" / "index.html").read_text() == original


def test_round_trip_survives_a_second_pass(tmp_path):
    """Rebuilding in place, repeatedly, must not drift (no re-escaping, no lost divs)."""
    src = tmp_path / "index.html"
    src.write_text(_render([{"h": "去重 Dedup"}, {"table": [{"cid": 1, "keep": False}]}]))
    once = src.read_text()

    for _ in range(2):
        build_page(src, tmp_path, NAV, "index.html")
    assert src.read_text() == once


def test_parse_page_recovers_title_and_cards():
    title, sections = parse_page(_render([
        {"h": "A"}, {"text": "body-a"},
        {"h": "B"}, {"text": "body-b"},
    ]))
    assert title == "umi400h — probe"
    assert [s.get("h") for s in sections if "h" in s] == ["A", "B"]
    assert [s["raw"] for s in sections if "raw" in s] == ["<div>body-a</div>", "<div>body-b</div>"]


def test_split_cards_is_depth_aware():
    """A card body full of nested divs (media walls, stat grids) must not truncate at
    the first </div> — that was the failure mode this splitter exists to avoid."""
    cards = _split_cards(
        '<div class="card"><h2>one</h2><div class="stat-grid"><div class="stat">x</div></div></div>'
        '<div class="card">two</div>'
    )
    assert cards == [
        '<h2>one</h2><div class="stat-grid"><div class="stat">x</div></div>',
        "two",
    ]


def test_unbalanced_card_is_an_error_not_a_silent_truncation():
    with pytest.raises(ValueError):
        _split_cards('<div class="card"><div>never closed</div>')


def test_page_without_figures_keeps_its_plotly_script_out(tmp_path):
    """A figure-less page ships without the plotly CDN <script> (3 MB nobody needs).
    Re-rendering must not hand it back."""
    src = tmp_path / "index.html"
    src.write_text(_render([{"h": "no figs"}, {"text": "hi"}]).replace(
        f'<script src="{html.PLOTLY_CDN}"></script>', "", 1))

    build_page(src, tmp_path / "out", NAV, "index.html")
    assert html.PLOTLY_CDN not in (tmp_path / "out" / "index.html").read_text()
