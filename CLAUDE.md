# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

The UMI data-cleaning review site **and** the generator that builds it. Self-contained: the page
builders read snapshots under `data/` and write HTML into `umi400h/<run>/`. Nothing here imports the
cleaning pipeline (`cosmos`) — the numbers each page needs were snapshotted into `data/` when the
pages were built, so a page can be edited and rebuilt without the dataset or the pipeline's parquet
workdir (which no longer exists on this machine).

```
sitegen/          the generator (pure python; numpy for pose curves, everything else stdlib)
  nav.py          THE page set — RUNS[run] is one entry per tab; RUNS[run][0] is that run's home
  html.py         design system: page(title, sections, nav, current, site_name) -> html str
  lazy_video.py   viewport-prefetching <video> tags (clips load as you scroll, autoplay in view)
  report_page.py  v0_study/report.html     <- data/report.json
  vio_filter_page.py  v0_study/vio_filter.html <- data/vio_review/ + data/calib/
  yaw_review_page.py  v0_study/yaw_review.html <- data/yaw_review/
  frozen_page.py  v1a_dedup_probe/*.html   <- itself (see "Two kinds of run")
  build.py        rebuild every page of every run; --check diffs against what's committed
data/             committed input snapshots (~17 MB; the clips live in the site's media/, not here)
tools/            one-off provenance scripts (e.g. how data/report.json was extracted)
umi400h/<run>/    the built site: *.html + media/ (the published artifact, committed)
index.html        root redirect -> the home page of nav.DEFAULT_RUN
```

## Commands

```bash
python3 -m sitegen.build                  # rebuild every page of every run, in place
python3 -m sitegen.build --run v0_study   # one run
python3 -m sitegen.build --check          # rebuild to a temp dir and diff; exit != 0 = a page drifted
python3 -m sitegen.report_page            # rebuild one page (same for vio_filter_page / yaw_review_page)
python3 -m http.server 8000               # review at localhost:8000/umi400h/v0_study/

uv run --no-project --with pytest --with numpy python -m pytest tests/ -q   # 58 tests, ~1s
uv run --no-project --with pytest --with numpy python -m pytest tests/test_vio_filter_page.py -q -k curve   # single test
```

**Run `--check` after editing a builder.** It is the regression guard: it proves the committed HTML
is exactly what the current generator + data produce, so a page nobody rebuilt cannot silently drift
from its source. Commit the rebuilt HTML together with the builder change.

## Two kinds of run

- **`v0_study` is live**: each page is rebuilt from a data snapshot under `data/`, so its numbers
  can be recomputed — edit the builder or the JSON and rebuild.
- **`v1a_dedup_probe` is frozen**: the cleaning run behind it is gone, so `frozen_page` re-renders
  each page *from the published page itself* — it parses the cards back out and feeds them through
  `html.page()` as verbatim `{"raw"}` sections. That keeps the run on the shared nav + design system
  (a CSS or nav change reaches it) and keeps it inside this repo's build, but its content is
  preserved, not recomputed: changing a number there means editing markup. Re-rendering an unchanged
  page is a fixpoint — that is exactly what `--check` asserts, and what `tests/test_frozen_page.py`
  pins down.

## How the pieces fit

- **Sections, not markup.** A builder returns a list of section dicts (`{"h"}`, `{"text"}`, `{"raw"}`,
  `{"table"}`, `{"stats"}`, `{"plotly"}`, `{"media"}`) and `html.page()` renders them into cards.
  Reach for raw HTML in a `{"text": ...}` section only when the shared components genuinely don't
  fit — `vio_filter_page._episode_block` does this deliberately (the `.media-item` thumbnail grid
  crops the 4:3 ego2 video; it needs uncropped side-by-side cells).
- **Adding/removing a tab** is `sitegen/nav.py` plus a builder module. Every page of a run renders
  that run's nav, and `build.py` refuses to run if a run lists a page with no builder.
- **Media is not in `data/`.** The clips (740 files) are already published under
  `umi400h/v0_study/media/` and committed. `data/vio_review/` carries only `manifest.json` +
  `pose.npz` (the curves), `data/yaw_review/` only `manifest.json`. A builder renders an episode when
  its clips are found in `media/` — or, when you point `--review-root` at a *fresh* episode-review run
  that still carries them, it imports them into `media/` first.

## Media is committed raw to git

No git-lfs. `.git` is ~900 MB and the working tree ~2.7 GB. Adding a report run means adding
hundreds of jpg/png/mp4 blobs permanently — keep clips small (they are transcoded h264/yuv420p so
browsers can play them) and use `python3 -m sitegen.build --prune-old-media` to drop clips the
rebuilt pages no longer reference.

## Conventions

- Commit subject: `<page>: <what changed>` (e.g. `vio_filter: yaw-deviation histogram`), or
  `report: <run> — <summary>` for a whole new run.
- `README.md` carries one link per report run — update it when a run is added.
- Feature work happens in `.worktrees/<branch>` (git worktrees on the same repo).

## History note

Page generation used to live in `cosmos` (`cosmos_framework/.../cleaning/report/build.py` +
`experiments/umi400h/bench/build_*_page.py`), and regenerating the site from there would rewrite
these pages. That coupling is gone — `sitegen/` is now the only thing that writes this site. If you
find yourself reaching into cosmos to change a page, you're in the wrong repo.
