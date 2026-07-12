# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Build the `report` tab — the S8 cleaning-campaign summary.

The **corpus-level** review page: what the S7 verdict did to the dataset as a
whole (headline counts, before/after, per-skill keep rate, the drop funnel and
its reason split) plus a look at what dedup actually threw away — the top
duplicate clusters, shown as media so a human can eyeball whether the segments
really were duplicates.

Input is `data/report.json`, which `tools/extract_report_data.py` reverse-extracted
from the previously rendered page: the upstream parquet cleaning manifest is gone,
so that JSON is now the source of truth. One consequence is visible here — the stat
tiles carry pre-formatted display strings ("200,551", "34.7% of total"), so this
module does no number formatting of its own.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import html as report_html
from .nav import NAV, SITE_NAME

TITLE = "umi400h — 对外汇报 review report"

# Headings are page chrome rather than data: they are the same for any re-run of
# the cleaning campaign, so they live here and not in report.json.
H_HEADLINE = "概况 Headline"
H_BEFORE_AFTER = "清洗前后对照 Before / After"
H_KEEP_RATE = "任务分布偏移 Keep rate by skill"
H_FUNNEL = "Funnel (dropped per filter, priority order)"
H_DROP_REASON = "drop_reason distribution"
H_DEDUP = "去重 Dedup"
H_SPOT_CHECK = "误杀抽检 Manual spot-check (fill in after human review)"
H_PIPELINE = "Pipeline (S1 -> S9)"


def _spot_check_table(columns: list[str]) -> str:
    """The spot-check table ships with a header and no rows — a human fills it in
    after reviewing. It is emitted as raw HTML rather than as a `{"table": ...}`
    section because the table widget renders an empty row list as "(no rows)",
    which would throw away the column names the reviewer is supposed to fill."""
    head = "".join(f"<th>{c}</th>" for c in columns)
    return f"<table><thead><tr>{head}</tr></thead><tbody></tbody></table>"


def _cluster_sections(cluster: dict) -> list[dict]:
    """One duplicate cluster. Media was only ever exported for the largest
    cluster, which renders as a media wall (rep video + member stills); the
    others carry no media and fall back to a plain segment table."""
    secs: list[dict] = [{"h": f"Top dup cluster {cluster['cid']} (size={cluster['size']})"}]
    if cluster["kind"] == "media":
        secs.append({"media": cluster["items"]})
    else:
        secs.append({"table": cluster["rows"]})
    return secs


def build_sections(data: dict) -> list[dict]:
    sections: list[dict] = [
        {"h": H_HEADLINE},
        {"stats": data["headline"]},
        {"h": H_BEFORE_AFTER},
        {"table": data["before_after"]},
        {"h": H_KEEP_RATE},
        {"plotly": data["keep_rate_by_skill"]},
        {"h": H_FUNNEL},
        {"table": data["funnel"]},
        {"h": H_DROP_REASON},
        {"plotly": data["drop_reason"]},
        {"h": H_DEDUP},
        {"stats": data["dedup"]},
    ]
    for cluster in data["dup_clusters"]:
        sections += _cluster_sections(cluster)
    sections += [
        {"h": H_SPOT_CHECK},
        {"text": _spot_check_table(data["spot_check_columns"])},
        {"h": H_PIPELINE},
        {"text": data["pipeline_html"]},
    ]
    return sections


def build_page(data_path: Path, site_dir: Path) -> Path:
    """Render `report.html` into `site_dir`. This is the entry point `build.py` calls."""
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    return report_html.write_page(
        site_dir, "report.html", data.get("title", TITLE), build_sections(data),
        nav=NAV, current="report.html", site_name=SITE_NAME,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=Path("data/report.json"))
    ap.add_argument("--site-dir", type=Path, default=Path("umi400h/v0_study"))
    args = ap.parse_args(argv)
    print(f"[done] wrote {build_page(args.data, args.site_dir)}")


if __name__ == "__main__":
    main()
