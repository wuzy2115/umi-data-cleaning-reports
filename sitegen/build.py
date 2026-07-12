# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Rebuild the whole site: every page in `nav.NAV`, from the snapshots in `data/`.

    python -m sitegen.build                       # rebuild all pages in place
    python -m sitegen.build --site-dir /tmp/out   # render elsewhere (media not copied)
    python -m sitegen.build --check               # rebuild to a temp dir, diff, don't write

`--check` is the regression guard: it proves the committed HTML is still exactly what
the current generator + data produce, so a page nobody rebuilt can't silently drift.
"""
from __future__ import annotations

import argparse
import difflib
import json
import tempfile
from pathlib import Path

from . import report_page, vio_filter_page, yaw_review_page
from .nav import NAV

DATA = Path("data")
SITE_DIR = Path("umi400h/v0_study")


def build_all(data: Path, site_dir: Path, prune_old_media: bool = False) -> list[Path]:
    """Render every page into `site_dir`, returning the paths written, nav order."""
    written = {
        "report.html": lambda: report_page.build_page(data / "report.json", site_dir),
        "vio_filter.html": lambda: vio_filter_page.build_page(
            data / "vio_review",
            json.loads((data / "calib/abs_coord_400h.json").read_text()),
            json.loads((data / "calib/abs_coord_2000h.json").read_text()),
            site_dir,
            episode_table=json.loads((data / "calib/episodes_400h.json").read_text()),
            prune_old_media=prune_old_media,
        ),
        "yaw_review.html": lambda: yaw_review_page.build_page(data / "yaw_review", site_dir),
    }
    missing = [href for href, _ in NAV if href not in written]
    if missing:
        raise SystemExit(f"nav lists pages with no builder: {missing}")
    return [written[href]() for href, _ in NAV]


def check(data: Path, site_dir: Path) -> int:
    """Rebuild into a temp dir and diff against the committed pages. 0 = in sync."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        # the page builders skip episodes whose clips are missing — point them at the
        # real media/ so `--check` compares content, not media availability
        (tmp_dir / "media").symlink_to((site_dir / "media").resolve())
        build_all(data, tmp_dir)

        stale = 0
        for href, _ in NAV:
            fresh, committed = (tmp_dir / href).read_text(), (site_dir / href).read_text()
            if fresh == committed:
                print(f"[ok]    {href}")
                continue
            stale += 1
            delta = sum(1 for line in difflib.unified_diff(
                committed.splitlines(), fresh.splitlines(), n=0) if line[:1] in "+-")
            print(f"[STALE] {href} — {delta} changed lines; rebuild with `python -m sitegen.build`")
    return stale


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--site-dir", type=Path, default=SITE_DIR)
    ap.add_argument("--check", action="store_true",
                    help="don't write: rebuild to a temp dir and report pages that drifted")
    ap.add_argument("--prune-old-media", action="store_true",
                    help="delete media/vio_* files the rebuilt pages no longer reference")
    args = ap.parse_args(argv)

    if args.check:
        raise SystemExit(check(args.data, args.site_dir))
    for out in build_all(args.data, args.site_dir, prune_old_media=args.prune_old_media):
        print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
