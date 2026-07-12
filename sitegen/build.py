# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Rebuild the site: every page of every run in `nav.RUNS`.

    python -m sitegen.build                    # rebuild every page in place
    python -m sitegen.build --run v0_study     # one run
    python -m sitegen.build --check            # rebuild to a temp dir, diff, don't write

`--check` is the regression guard: it proves the committed HTML is still exactly what
the current generator + data produce, so a page nobody rebuilt can't silently drift.

`v0_study` is rebuilt from the snapshots under `data/`. `v1a_dedup_probe` is frozen —
its upstream numbers are gone, so each page is re-rendered from itself (see
`frozen_page`); that keeps it on the shared nav + design system without a data source.
"""
from __future__ import annotations

import argparse
import difflib
import json
import tempfile
from pathlib import Path

from . import frozen_page, report_page, vio_filter_page, yaw_review_page
from .nav import RUNS

DATA = Path("data")
SITE_ROOT = Path("umi400h")
FROZEN_RUNS = ("v1a_dedup_probe",)


def _v0_builders(data: Path, site_dir: Path, prune_old_media: bool):
    return {
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


def build_run(run: str, data: Path, site_root: Path, out_dir: Path | None = None,
              prune_old_media: bool = False) -> list[Path]:
    """Render every page of `run`, in nav order, returning the paths written."""
    dst = Path(out_dir) if out_dir is not None else site_root / run
    if run in FROZEN_RUNS:
        return frozen_page.build_run(run, site_root, out_dir=dst)

    builders = _v0_builders(data, dst, prune_old_media)
    missing = [href for href, _ in RUNS[run] if href not in builders]
    if missing:
        raise SystemExit(f"{run}: nav lists pages with no builder: {missing}")
    return [builders[href]() for href, _ in RUNS[run]]


def check(run: str, data: Path, site_root: Path) -> int:
    """Rebuild `run` into a temp dir and diff against the committed pages. 0 = in sync."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        # the page builders skip episodes whose clips are missing — point them at the
        # real media/ so this compares content, not media availability
        media = site_root / run / "media"
        if media.is_dir():
            (out / "media").symlink_to(media.resolve())
        build_run(run, data, site_root, out_dir=out)

        stale = 0
        for href, _ in RUNS[run]:
            fresh = (out / href).read_text()
            committed = (site_root / run / href).read_text()
            if fresh == committed:
                print(f"[ok]    {run}/{href}")
                continue
            stale += 1
            delta = sum(1 for line in difflib.unified_diff(
                committed.splitlines(), fresh.splitlines(), n=0) if line[:1] in "+-")
            print(f"[STALE] {run}/{href} — {delta} changed lines; rebuild with `python -m sitegen.build`")
    return stale


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", choices=sorted(RUNS), default=None,
                    help="rebuild only this run (repeatable); default: every run")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--site-root", type=Path, default=SITE_ROOT)
    ap.add_argument("--check", action="store_true",
                    help="don't write: rebuild to a temp dir and report pages that drifted")
    ap.add_argument("--prune-old-media", action="store_true",
                    help="delete media/vio_* files the rebuilt pages no longer reference")
    args = ap.parse_args(argv)
    runs = args.run or sorted(RUNS)

    if args.check:
        raise SystemExit(sum(check(r, args.data, args.site_root) for r in runs))
    for run in runs:
        for out in build_run(run, args.data, args.site_root,
                             prune_old_media=args.prune_old_media):
            print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
