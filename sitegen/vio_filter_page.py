# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Build the `vio_filter` tab.

The **episode-level** review page: a human scans ~50 whole episodes (ego2 video +
3D world-frame trajectory + xy/z threshold curves) to calibrate an episode-level
VIO drift threshold that is auto-applied at training init. Inputs are the episode
review snapshot in `data/vio_review/` (manifest.json + per-episode `pose.npz`; the
matching `ego2_4x.mp4` / `traj3d.png` already live in the site's `media/`) plus the
two abs-coord calibration jsons and the full-population episode table under
`data/calib/`.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from . import html as report_html
from .lazy_video import LAZY_VIDEO_SCRIPT, video_tag
from .nav import NAV, SITE_NAME

# The 8 fine sub-2m bands (experiments/umi400h/bench/vio_episode_review.py's
# _FINE_BAND_ORDER) render after the 5 coarse bands. Listed explicitly (rather
# than relying on manifest order) so page ordering is deterministic regardless
# of the order bands happen to appear in manifest.json.
_FINE_BANDS = ("z_1p5_2", "z_1_1p5", "z_0p5_1", "z_0_0p5",
              "xy_1p5_2", "xy_1_1p5", "xy_0p5_1", "xy_0_0p5")
_BAND_ORDER = ("extreme", "high", "mid", "control", "z_only") + _FINE_BANDS
_BAND_CRITERION = {
    "extreme": "xy_max > 20 m", "high": "10 < xy_max ≤ 20 m",
    "mid": "5 < xy_max ≤ 10 m", "control": "2 < xy_max ≤ 5 m(对照)",
    "z_only": "z_max > 3 m 且 xy_max ≤ 5 m",
    "z_1p5_2": "1.5 < z_max ≤ 2 m 且 xy_max ≤ 2(全审,总体仅~6条)",
    "z_1_1p5": "1 < z_max ≤ 1.5 m 且 xy_max ≤ 2",
    "z_0p5_1": "0.5 < z_max ≤ 1 m 且 xy_max ≤ 2",
    "z_0_0p5": "z_max ≤ 0.5 m 且 xy_max ≤ 2",
    "xy_1p5_2": "1.5 < xy_max ≤ 2 m 且 z_max ≤ 1.5",
    "xy_1_1p5": "1 < xy_max ≤ 1.5 m 且 z_max ≤ 1.5",
    "xy_0p5_1": "0.5 < xy_max ≤ 1 m 且 z_max ≤ 1.5",
    "xy_0_0p5": "xy_max ≤ 0.5 m 且 z_max ≤ 1.5",
}
_DROP_TONE_BANDS = {"extreme", "high", "z_only"}

_STREAMS = ("head", "wrist1", "wrist2")


def episode_curve_fig(pose, kind: str, thresholds: list[float],
                       max_points: int = 600) -> dict:
    """Plotly fig dict: per-stream (head/wrist1/wrist2) drift quantity over the
    whole episode, decimated to <= max_points points/trace, one dashed
    threshold hline + `τ = {v}` annotation per candidate threshold. `kind`
    picks the quantity: "xy" -> ‖p_xy‖ per stream, "z" -> |z| per stream."""
    t = np.asarray(pose["t"])
    stride = max(1, int(np.ceil(t.size / max_points)))
    data = []
    for name in _STREAMS:
        p = np.asarray(pose[name])
        y = np.linalg.norm(p[:, :2], axis=1) if kind == "xy" else np.abs(p[:, 2])
        data.append({"type": "scatter", "mode": "lines", "name": name,
                     "x": [round(float(v), 2) for v in t[::stride]],
                     "y": [round(float(v), 4) for v in y[::stride]]})
    ylab = "‖p_xy‖ (m)" if kind == "xy" else "|z| (m)"
    return {"data": data, "layout": {
        "height": 260, "margin": {"l": 50, "r": 10, "t": 10, "b": 35},
        "xaxis": {"title": {"text": "t (s)"}}, "yaxis": {"title": {"text": ylab}},
        "shapes": [{"type": "line", "xref": "paper", "x0": 0, "x1": 1,
                    "y0": thr, "y1": thr, "line": {"dash": "dash", "width": 1.5}}
                   for thr in thresholds],
        "annotations": [{"xref": "paper", "x": 1, "y": thr, "text": f"τ = {thr}",
                          "showarrow": False, "xanchor": "right",
                          "yanchor": "bottom", "font": {"size": 11}}
                         for thr in thresholds],
    }}


def _calib_row(name: str, c: dict) -> dict:
    xy_q = c.get("ep_xy_max_q", {})
    z_q = c.get("ep_z_max_q", {})
    yaw_q = c.get("ep_yaw_max_q", {})
    return {
        "dataset": name,
        "n_episodes": c.get("n_episodes", "—"),
        "ep |p_xy| q50/q90/q99/max": "/".join(
            str(xy_q.get(k, "—")) for k in ("0.5", "0.9", "0.99", "1.0")),
        "ep |z| q50/q90/q99/max": "/".join(
            str(z_q.get(k, "—")) for k in ("0.5", "0.9", "0.99", "1.0")),
        "ep |Δyaw| q50/q90/q99/max": "/".join(
            str(yaw_q.get(k, "—")) for k in ("0.5", "0.9", "0.99", "1.0")),
        "episodes >10m xy frac": c.get("ep_xy_exceed_frac", {}).get("10.0", "—"),
        "episodes >3m z frac": c.get("ep_z_exceed_frac", {}).get("3.0", "—"),
    }


_XY_POP_EDGES = [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 5), (5, 10), (10, 20), (20, None)]
_XY_POP_LABELS = ["(0, 0.5]", "(0.5, 1]", "(1, 1.5]", "(1.5, 2]", "(2, 5]",
                  "(5, 10]", "(10, 20]", "> 20"]
_Z_POP_EDGES = [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, None)]
_Z_POP_LABELS = ["(0, 0.5]", "(0.5, 1]", "(1, 1.5]", "(1.5, 2]", "(2, 3]", "> 3"]


def _count_in_band(values: list[float], lo: float, hi: float | None) -> int:
    if hi is None:
        return sum(1 for v in values if v > lo)
    return sum(1 for v in values if lo < v <= hi)


def population_band_rows(table: list[dict]) -> list[dict]:
    """Full-population (not just the sampled review set) xy/z band ladder,
    for the `episode 分布(全量)` table: how many of ALL episodes fall in each
    xy_max_m bucket, and (restricted to xy_max_m<=2, since that's the fine-band
    review's scope) each z_max_m bucket. Fractions are percent of the whole
    table (both axes share the same denominator so the two halves stay
    directly comparable), rounded to 3 decimals. Pure/unit-testable -- no I/O."""
    total = len(table)
    xy_vals = [float(r["xy_max_m"]) for r in table]
    z_vals = [float(r["z_max_m"]) for r in table if float(r["xy_max_m"]) <= 2]
    rows: list[dict] = []
    for (lo, hi), label in zip(_XY_POP_EDGES, _XY_POP_LABELS):
        cnt = _count_in_band(xy_vals, lo, hi)
        rows.append({"axis": "xy", "band": label, "episodes": cnt,
                     "fraction": round(100 * cnt / total, 3) if total else 0.0})
    for (lo, hi), label in zip(_Z_POP_EDGES, _Z_POP_LABELS):
        cnt = _count_in_band(z_vals, lo, hi)
        rows.append({"axis": "z", "band": label, "episodes": cnt,
                     "fraction": round(100 * cnt / total, 3) if total else 0.0})
    return rows


_HIST_UPPER = {"xy": 3.0, "z": 2.0, "yaw": 360.0}
_HIST_THRESHOLDS = {"xy": [2.0], "z": [1.5, 2.0], "yaw": []}
# The two histograms deliberately use different percent denominators (xy: all
# episodes; z: the xy<=2 subset it's restricted to -- see population_hist_fig's
# docstring) so each bar's own bins sum to 100%. That's a natural histogram
# convention, but it makes the z fig's percent NOT directly comparable to
# population_band_rows' full-population percent below it. Spell the
# denominator out in the hover text itself so a reviewer never has to guess
# which population a given percent is drawn from. yaw shares xy's convention
# (denominator = all episodes) since it isn't restricted to any subset.
_HIST_PERCENT_LABEL = {"xy": "% of 全部 episodes", "z": "% of xy≤2 子集",
                       "yaw": "% of 全部 episodes"}
_HIST_UNIT = {"xy": "m", "z": "m", "yaw": "°"}
_HIST_XLABEL = {"xy": "xy_max (m)", "z": "z_max (m, xy_max≤2 subset)",
               "yaw": "yaw_max (deg, unwrapped |Δyaw| vs. episode start)"}


def _bin_index(v: float, bin_m: float, n_bins: int) -> int:
    # +epsilon guards against float bin-edge jitter (e.g. 0.3/0.1 landing on
    # 2.9999999999999996 and flooring one bin short of where the value
    # actually belongs). Scaled to bin_m (rather than a flat constant) so the
    # guarantee holds for arbitrary bin sizes, not just the 0.1 default;
    # 1e-6 relative jitter is far above float noise but far below half a bin.
    eps = bin_m * 1e-6
    idx = int((v + eps) // bin_m)
    return min(max(idx, 0), n_bins)


def population_hist_fig(table: list[dict], kind: str, bin_m: float = 0.1) -> dict:
    """Fine-grained histogram of the full population's xy_max_m (`kind="xy"`,
    0.1 m default bins covering [0, 3)), z_max_m restricted to xy_max_m<=2
    (`kind="z"`, 0.1 m bins covering [0, 2), same isolation convention as
    `population_band_rows`), or yaw_max_deg (`kind="yaw"`, 10-degree bins
    covering [0, 360); rows missing the `yaw_max_deg` key -- old-format
    tables -- are silently skipped, for backward compat) -- makes the
    distribution SHAPE visible where `population_band_rows`'s coarse bands
    wash it out. Counts are pre-aggregated here in python: the ~10878-row raw
    table never reaches the returned fig dict, only a handful of per-bin
    counts do.

    y-axis is log-scaled: the dominant 0.5-1.0 m mass (~1000+ episodes) would
    otherwise flatten the 1-5-count tail bins (which is exactly the region
    the candidate thresholds live in) to invisible slivers on a linear scale.
    Zero-count bins are safe on a log axis for a bar chart -- Plotly just
    draws them as zero-height bars, it doesn't error like a log(0) would."""
    upper = _HIST_UPPER[kind]
    n_bins = round(upper / bin_m)
    edges = [(round(i * bin_m, 4), round((i + 1) * bin_m, 4)) for i in range(n_bins)]
    labels = [f"{lo:.1f}–{hi:.1f}" for lo, hi in edges] + [f">{upper:g}"]

    if kind == "xy":
        vals = [float(r["xy_max_m"]) for r in table]
    elif kind == "z":
        vals = [float(r["z_max_m"]) for r in table if float(r["xy_max_m"]) <= 2]
    else:  # yaw
        vals = [float(r["yaw_max_deg"]) for r in table if "yaw_max_deg" in r]

    counts = [0] * (n_bins + 1)
    for v in vals:
        counts[_bin_index(v, bin_m, n_bins)] += 1

    total = len(vals)
    fractions = [round(100 * c / total, 3) if total else 0.0 for c in counts]

    unit = _HIST_UNIT[kind]
    data = [{
        "type": "bar", "x": labels, "y": counts, "customdata": fractions,
        "hovertemplate": ("%{x} " + unit + "<br>episodes: %{y}<br>%{customdata:.2f}"
                          + _HIST_PERCENT_LABEL[kind] + "<extra></extra>"),
    }]

    # Threshold lines mark bin BOUNDARIES, not bin centers: on a category
    # x-axis, category i sits at integer position i, so the boundary between
    # the bin ending at `thr` and the next one is at position
    # (thr/bin_m - 1) + 0.5 == thr/bin_m - 0.5. yaw has no candidate
    # thresholds (_HIST_THRESHOLDS["yaw"] == []) so this loop is a no-op there.
    shapes, annotations = [], []
    for thr in _HIST_THRESHOLDS[kind]:
        idx = thr / bin_m - 0.5
        shapes.append({"type": "line", "xref": "x", "x0": idx, "x1": idx,
                       "yref": "paper", "y0": 0, "y1": 1,
                       "line": {"dash": "dash", "width": 1.5}})
        annotations.append({"xref": "x", "x": idx, "yref": "paper", "y": 1,
                            "text": f"τ = {thr}", "showarrow": False,
                            "xanchor": "left", "yanchor": "top", "font": {"size": 11}})

    xlab = _HIST_XLABEL[kind]
    return {"data": data, "layout": {
        "height": 280, "margin": {"l": 50, "r": 10, "t": 10, "b": 70},
        "bargap": 0.05,
        "xaxis": {"title": {"text": xlab}, "tickangle": -60},
        "yaxis": {"title": {"text": "episodes"}, "type": "log"},
        "shapes": shapes, "annotations": annotations,
    }}


def _calib_sections(calib_400h: dict, calib_2000h: dict) -> list[dict]:
    rows = [_calib_row("400h", calib_400h), _calib_row("2000h", calib_2000h)]
    worst = calib_2000h.get("worst20_xy", [])[:10]
    return [
        {"h": "阈值校准 — episode 级绝对坐标分布"},
        {"table": rows},
        {"h": "worst-10 episodes by xy_max(2000h)"},
        {"table": worst},
    ]


def _band_stats(bands: dict) -> list[dict]:
    tiles = []
    for band, count in bands.items():
        tone = "drop" if band in _DROP_TONE_BANDS else ""
        tile = {"label": band, "value": count, "sub": _BAND_CRITERION.get(band, "")}
        if tone:
            tile["tone"] = tone
        tiles.append(tile)
    return tiles


def _episode_media_paths(review_root: Path, ep: dict) -> tuple[Path, Path, Path]:
    d = Path(review_root) / ep["dir"]
    return d / "ego2_4x.mp4", d / "traj3d.png", d / "pose.npz"


def _episode_source_media(review_root: Path, ep: dict) -> tuple[Path, Path] | None:
    """`(mp4, png)` inside the review root, iff both are there. A full episode-review
    run has them; `data/vio_review/` — the snapshot committed here — carries only the
    manifest + pose.npz, because the clips are already published under the site's
    `media/`. Present only when importing episodes from a fresh review run."""
    video, img, _pose = _episode_media_paths(review_root, ep)
    return (video, img) if video.exists() and img.exists() else None


def _episode_published_media(media_dir: Path | None, ep: dict) -> bool:
    """Are this episode's clips already sitting in the site's `media/`?"""
    if media_dir is None:
        return False
    mp4_name, png_name = _media_names(ep)
    return (media_dir / mp4_name).exists() and (media_dir / png_name).exists()


def _episode_pose(review_root: Path, ep: dict, media_dir: Path | None = None) -> Path | None:
    """Return `pose.npz` for `ep` iff the episode is renderable — pose curves plus
    both clips, the latter either already published in `media/` or importable from
    the review root. Single source of truth for "is this episode complete", shared by
    `build_page_sections` (skip it) and `main()`'s copy loop (don't copy media the
    page won't reference), so the two can never disagree."""
    _video, _img, pose_path = _episode_media_paths(review_root, ep)
    if not pose_path.exists():
        return None
    if _episode_published_media(media_dir, ep) or _episode_source_media(review_root, ep):
        return pose_path
    return None


def _media_names(ep: dict) -> tuple[str, str]:
    """Derive the (mp4, traj3d png) filenames used for an episode's copied/
    referenced media. Single place shared by `_episode_block` (page markup)
    and `main()`'s copy loop so the two names can never drift apart."""
    s, e = ep["shard"], ep["episode"]
    return f"vio_ep_s{s}_e{e}.mp4", f"vio_ep_s{s}_e{e}_traj3d.png"


def _episode_block(ep: dict, pose_path: Path, media_rel: str) -> list[dict]:
    s, e = ep["shard"], ep["episode"]
    mp4_name, png_name = _media_names(ep)
    # Quotable episode ID gets a small monospace badge (this is the string the
    # reviewer types back verbatim); xy_max/z_max sit beside it in tabular-nums
    # so the two numbers that actually gate the band decision are the first
    # thing the eye lands on — the reviewer never has to hunt for them.
    headline = (
        '<div style="display:flex;align-items:baseline;flex-wrap:wrap;'
        'gap:0.75rem;margin-bottom:0.65rem">'
        f'<span style="font-family:var(--mono);font-weight:700;font-size:0.98rem;'
        f'color:var(--ink);background:var(--accent-soft);'
        f'border:1px solid var(--border-strong);border-radius:5px;'
        f'padding:0.2rem 0.6rem;letter-spacing:0.01em">'
        f'<b>shard{s} ep{e}</b></span>'
        f'<span style="font-family:var(--mono);font-size:0.82rem;'
        f'color:var(--ink-2);font-variant-numeric:tabular-nums">'
        f'xy_max {ep["xy_max_m"]} m / z_max {ep["z_max_m"]} m</span>'
        '</div>'
    )
    # Custom pair block instead of the shared {"media"} wall: the site's
    # .media-item CSS is a thumbnail grid (16:9 object-fit:cover, ~200px
    # cells) that crops the 4:3 ego2 video and shrinks the traj plot.
    # object-fit:contain + two wide flex columns show the evidence uncropped
    # — this exact lesson was learned building the previous (window-level)
    # version of this page.
    cell = ('<div style="flex:1 1 380px;min-width:320px">{tag}'
            '<div class="caption">{cap}</div></div>')
    cells = [
        cell.format(
            tag=video_tag(
                f"{media_rel}/{mp4_name}",
                # height:auto — the tag's width/height attrs then reserve the
                # correct box before the lazy src attaches (no layout jump).
                style=("width:100%;height:auto;max-height:460px;"
                       "object-fit:contain;background:#101113;"
                       "border-radius:6px;display:block")),
            cap="ego2 视频(4× speed,进入视口自动播放)"),
        cell.format(
            tag=(f'<img loading="lazy" src="{media_rel}/{png_name}" '
                 'style="width:100%;max-height:460px;object-fit:contain;'
                 'background:#101113;border-radius:6px;display:block">'),
            cap="world 系 3D 轨迹(head + 双腕)"),
    ]
    media_row = ('<div style="display:flex;flex-wrap:wrap;gap:14px;'
                 f'align-items:flex-start">{"".join(cells)}</div>')
    # Fine sub-2m bands need fine-grained hlines (the coarse 5/10/20 & 1.5/2/3
    # ladders are meaningless at this scale); coarse bands keep the original
    # xy ladder, but z is tightened from [3] to [1.5, 2, 3] -- z>3 alone is
    # uselessly loose given human height, per the review that motivated fine.
    if ep["band"] in _FINE_BANDS:
        xy_thresholds, z_thresholds = [0.5, 1, 1.5, 2], [0.5, 1, 1.5, 2]
    else:
        xy_thresholds, z_thresholds = [5, 10, 20], [1.5, 2, 3]
    with np.load(pose_path) as pose:
        fig_xy = episode_curve_fig(pose, "xy", xy_thresholds)
        fig_z = episode_curve_fig(pose, "z", z_thresholds)
    return [
        {"text": headline},
        {"text": media_row},
        {"plotly": fig_xy},
        {"plotly": fig_z},
    ]


def build_page_sections(review_root: Path, calib_400h: dict, calib_2000h: dict,
                         media_rel: str = "media",
                         episode_table: list[dict] | None = None,
                         media_dir: Path | None = None) -> list[dict]:
    review_root = Path(review_root)
    manifest = json.loads((review_root / "manifest.json").read_text())
    bands_meta = manifest["meta"]["bands"]

    sections: list[dict] = [
        {"h": "VIO episode 审查 — 阈值校准"},
        {"text": (
            "VIO 世界系漂移会污染 head/hand 位移统计与绝对坐标标签;背景与逐"
            'episode 漂移证据见 <a href="https://github.com/wuzy2115/'
            'vio-drift-report">vio-drift-report</a>。本页按 xy_max/z_max 分档'
            "抽样~50个完整 episode(视频 + 3D 轨迹 + xy/z 阈值曲线),人工审查后"
            "校准的 episode 级漂移阈值将在训练 init 时自动应用于剔除。"
        )},
        {"stats": _band_stats(bands_meta)},
    ]
    sections += _calib_sections(calib_400h, calib_2000h)
    if episode_table is not None:
        sections.append({"h": "episode 分布(全量)"})
        sections.append({"plotly": population_hist_fig(episode_table, "xy")})
        sections.append({"plotly": population_hist_fig(episode_table, "z")})
        sections.append({"text": (
            "直方图百分比 = xy≤2 子集内占比；下表百分比 = 全量占比"
        )})
        # yaw_max_deg is a newer field; older episode tables won't have it on
        # any row. Require it on EVERY row (one scan writes the whole table, so
        # production is all-or-nothing): a partially-annotated table would render
        # a histogram whose denominator (annotated rows) contradicts the
        # "% of 全部 episodes" hover label — omit the section instead.
        if episode_table and all("yaw_max_deg" in r for r in episode_table):
            sections.append({"plotly": population_hist_fig(episode_table, "yaw", bin_m=10.0)})
            sections.append({"text": "头部航向角最大偏转 |Δyaw|(unwrapped,可>360°)"})
        sections.append({"table": population_band_rows(episode_table)})

    by_band: dict[str, list[dict]] = {b: [] for b in _BAND_ORDER}
    for ep in manifest["episodes"]:
        by_band.setdefault(ep["band"], []).append(ep)

    skipped: list[str] = []
    band_order = list(_BAND_ORDER) + [b for b in by_band if b not in _BAND_ORDER]
    for band in band_order:
        eps = by_band.get(band)
        if not eps:
            continue
        sections.append({"h": f"{band} — {_BAND_CRITERION.get(band, '')}"})
        for ep in eps:
            pose_path = _episode_pose(review_root, ep, media_dir)
            if pose_path is None:
                skipped.append(ep["dir"])
                continue
            sections += _episode_block(ep, pose_path, media_rel)

    if skipped:
        sections.append({"text": f"媒体缺失,skipped:{', '.join(skipped)}"})
    # Last section: the script runs at parse time and only sees the <video>
    # tags emitted before it, so it must trail every episode block.
    sections.append({"text": LAZY_VIDEO_SCRIPT})
    return sections


def build_page(review_root: Path, calib_400h: dict, calib_2000h: dict, site_dir: Path,
               episode_table: list[dict] | None = None, prune_old_media: bool = False) -> Path:
    """Render `<site_dir>/vio_filter.html` and return its path. Clips are imported
    from `review_root` only when it carries them (a fresh episode-review run);
    otherwise the ones already under `<site_dir>/media/` are reused as-is."""
    media_dir = site_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((Path(review_root) / "manifest.json").read_text())
    referenced: set[str] = set()
    for ep in manifest["episodes"]:
        if _episode_pose(review_root, ep, media_dir) is None:
            continue
        mp4_name, png_name = _media_names(ep)
        referenced |= {mp4_name, png_name}
        source = _episode_source_media(review_root, ep)
        if source is not None:
            for src, name in zip(source, (mp4_name, png_name), strict=True):
                shutil.copy2(src, media_dir / name)

    if prune_old_media:
        for p in sorted(media_dir.glob("vio_*")):
            if p.name not in referenced:
                p.unlink()
                print(f"[prune] removed {p}")

    sections = build_page_sections(review_root, calib_400h, calib_2000h,
                                   episode_table=episode_table, media_dir=media_dir)
    out = site_dir / "vio_filter.html"
    out.write_text(report_html.page(
        f"{SITE_NAME} — VIO episode review", sections, nav=NAV,
        current="vio_filter.html", site_name=SITE_NAME))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-root", type=Path, default=Path("data/vio_review"))
    ap.add_argument("--calib-400h", type=Path, default=Path("data/calib/abs_coord_400h.json"))
    ap.add_argument("--calib-2000h", type=Path, default=Path("data/calib/abs_coord_2000h.json"))
    ap.add_argument("--episode-table", type=Path, default=Path("data/calib/episodes_400h.json"),
                    help="per-episode abs table json ([{shard, episode, xy_max_m, z_max_m, "
                         "yaw_max_deg}]); drives the full-population histograms + band table")
    ap.add_argument("--site-dir", type=Path, default=Path("umi400h/v0_study"))
    ap.add_argument("--prune-old-media", action="store_true",
                    help="delete media/vio_* files the rebuilt page no longer references")
    args = ap.parse_args(argv)

    out = build_page(
        args.review_root,
        json.loads(args.calib_400h.read_text()),
        json.loads(args.calib_2000h.read_text()),
        args.site_dir,
        episode_table=json.loads(args.episode_table.read_text()) if args.episode_table else None,
        prune_old_media=args.prune_old_media,
    )
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
