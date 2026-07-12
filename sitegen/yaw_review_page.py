# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Build the static yaw-threshold episode review page.

Input is the review snapshot in `data/yaw_review/` (manifest.json); the per-episode
`ego2_4x.mp4` / `yaw_curve.png` already live in the site's `media/` and are only
re-imported when the manifest points at a fresh review run that still carries them.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import shutil
from pathlib import Path

from . import html as report_html
from .lazy_video import LAZY_VIDEO_SCRIPT, video_tag
from .nav import NAV, SITE_NAME

# (name, low_deg, high_deg, sampling quota) — mirrors the sampler that produced the
# review manifest; the page groups episodes into exactly these bands.
YAW_BANDS: tuple[tuple[str, float, float | None, int], ...] = (
    ("0-90", 0.0, 90.0, 8),
    ("90-135", 90.0, 135.0, 8),
    ("135-160", 135.0, 160.0, 12),
    ("160-180", 160.0, 180.0, 16),
    ("180-200", 180.0, 200.0, 16),
    ("200-240", 200.0, 240.0, 12),
    ("240-360", 240.0, 360.0, 10),
    ("360-720", 360.0, 720.0, 8),
    (">720", 720.0, None, 6),
)

_OPEN_BANDS = {"160-180", "180-200"}
_BAND_LABELS = {
    "0-90": "0–90°",
    "90-135": "90–135°",
    "135-160": "135–160°",
    "160-180": "160–180°",
    "180-200": "180–200°",
    "200-240": "200–240°",
    "240-360": "240–360°",
    "360-720": "360–720°",
    ">720": ">720°",
}

_PAGE_CSS = """
<style>
  .yaw-console {
    border-top: 3px solid #d97706; padding-top: .85rem;
  }
  .yaw-console p { margin: 0; max-width: 82ch; color: var(--ink-2); }
  .yaw-method {
    margin-top: .65rem; font-family: var(--mono); font-size: .7rem;
    line-height: 1.6; color: var(--ink-3);
  }
  .yaw-threshold-rail {
    position: relative; height: 42px; margin: 1rem 0 .45rem;
    border-top: 5px solid #e7b45f;
  }
  .yaw-threshold-rail::before {
    content: ""; position: absolute; left: 50%; top: -10px;
    width: 3px; height: 26px; background: #b45309;
    box-shadow: 0 0 0 4px rgba(180, 83, 9, .14);
  }
  .yaw-threshold-rail::after {
    content: "180° candidate threshold"; position: absolute; left: 50%; top: 18px;
    transform: translateX(-50%); white-space: nowrap;
    font-family: var(--mono); font-size: .66rem; font-weight: 700;
    letter-spacing: .06em; text-transform: uppercase; color: var(--accent-deep);
  }
  .yaw-band {
    background: var(--surface); border: 1px solid var(--border);
    border-left: 3px solid var(--border-strong); margin: 0 0 .65rem;
  }
  .yaw-band.threshold-adjacent { border-left-color: #d97706; }
  .yaw-band summary {
    cursor: pointer; list-style-position: outside; padding: .72rem .9rem;
    font-family: var(--mono); font-variant-numeric: tabular-nums;
  }
  .yaw-band summary::marker { color: var(--accent); }
  .yaw-band-title { font-size: .82rem; font-weight: 750; color: var(--ink); }
  .yaw-band-count {
    float: right; font-size: .66rem; letter-spacing: .07em;
    text-transform: uppercase; color: var(--ink-3);
  }
  .yaw-evidence-list { border-top: 1px solid var(--border); }
  .yaw-evidence {
    padding: .8rem .9rem 1rem; border-bottom: 1px solid var(--border);
  }
  .yaw-evidence:last-child { border-bottom: 0; }
  .yaw-evidence-head {
    display: flex; align-items: baseline; flex-wrap: wrap; gap: .45rem .8rem;
    margin-bottom: .65rem;
  }
  .yaw-id {
    font-family: var(--mono); font-size: .78rem; font-weight: 750;
    color: var(--ink); background: var(--accent-soft);
    border: 1px solid #dfbd91; padding: .14rem .5rem;
  }
  .yaw-metrics {
    display: flex; flex-wrap: wrap; gap: .35rem .8rem;
    font-family: var(--mono); font-size: .68rem; color: var(--ink-2);
    font-variant-numeric: tabular-nums;
  }
  .yaw-metrics b { color: var(--accent-deep); font-weight: 750; }
  .yaw-media {
    display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: .7rem;
  }
  .yaw-media figure { min-width: 0; margin: 0; }
  /* height:auto lets the clip's width/height attributes supply the aspect
     ratio, so the box is the right size before the src is ever attached —
     otherwise a lazily-attached clip grows from 150px and shoves the page. */
  .yaw-media video, .yaw-media img {
    display: block; width: 100%; height: auto; max-height: 430px;
    object-fit: contain; background: #101113; border: 1px solid #2b2d31;
  }
  .yaw-media figcaption {
    margin-top: .28rem; font-family: var(--mono); font-size: .61rem;
    letter-spacing: .05em; text-transform: uppercase; color: var(--ink-3);
  }
  .yaw-empty, .yaw-skipped {
    padding: .75rem .9rem; font-family: var(--mono); font-size: .68rem;
    color: var(--ink-3);
  }
  .yaw-skipped {
    margin-top: .8rem; border: 1px dashed var(--border-strong);
    background: var(--surface-2); overflow-wrap: anywhere;
  }
  @media (max-width: 720px) {
    .yaw-media { grid-template-columns: 1fr; }
    .yaw-band-count { float: none; display: block; margin-top: .15rem; }
    .yaw-threshold-rail::after { font-size: .58rem; }
  }
</style>
"""


def _episode_identity(episode: dict) -> tuple[int, int]:
    identity = []
    for field in ("shard", "episode"):
        value = episode.get(field)
        if type(value) is not int or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
        identity.append(value)
    return identity[0], identity[1]


def _episode_metrics(episode: dict) -> tuple[float, float, float, float]:
    metrics = []
    for field in ("yaw_max_deg", "xy_max_m", "z_max_m", "duration_s"):
        value = episode.get(field)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{field} must be a finite number")
        metrics.append(float(value))
    return metrics[0], metrics[1], metrics[2], metrics[3]


def _episode_media(review_root: Path, episode: dict) -> tuple[Path, Path] | None:
    """`(mp4, png)` inside the review root, iff both are there. Present on a fresh
    episode-review run; absent in the `data/yaw_review/` snapshot committed here,
    whose clips are already published under the site's `media/` — see
    `_episode_published_media`."""
    try:
        _episode_identity(episode)
    except ValueError:
        return None
    root = Path(review_root).resolve()
    directory = (root / episode["dir"]).resolve()
    if not directory.is_relative_to(root):
        return None
    video = directory / "ego2_4x.mp4"
    curve = directory / "yaw_curve.png"
    if video.is_file() and curve.is_file():
        return video, curve
    return None


def _episode_published_media(media_dir: Path | None, episode: dict) -> bool:
    """Are this episode's clips already sitting in the site's `media/`?"""
    if media_dir is None:
        return False
    try:
        names = _media_names(episode)
    except ValueError:
        return False
    return all((media_dir / name).is_file() for name in names)


def _episode_renderable(review_root: Path, episode: dict, media_dir: Path | None) -> bool:
    """Media available either way — already published, or importable from the review
    root. Shared by `build_page_sections` (skip the episode) and `build_page`'s copy
    loop (never copy media the page won't reference)."""
    return (_episode_published_media(media_dir, episode)
            or _episode_media(review_root, episode) is not None)


def _media_names(episode: dict) -> tuple[str, str]:
    shard, episode_id = _episode_identity(episode)
    stem = f"yaw_ep_s{shard}_e{episode_id}"
    return f"{stem}.mp4", f"{stem}_yaw.png"


def _fmt(value: object) -> str:
    return html.escape(str(value))


def _episode_markup(episode: dict, media_rel: str) -> str:
    mp4_name, png_name = _media_names(episode)
    yaw, xy, z, duration = _episode_metrics(episode)
    media = html.escape(media_rel.rstrip("/"))
    shard, episode_id = _episode_identity(episode)
    return (
        '<article class="yaw-evidence">'
        '<div class="yaw-evidence-head">'
        f'<span class="yaw-id">shard{shard} ep{episode_id}</span>'
        '<span class="yaw-metrics">'
        f'<span><b>yaw {_fmt(yaw)}°</b></span>'
        f'<span>xy {_fmt(xy)} m</span>'
        f'<span>z {_fmt(z)} m</span>'
        f'<span>duration {_fmt(duration)} s</span>'
        "</span></div>"
        '<div class="yaw-media">'
        "<figure>"
        f"{video_tag(f'{media}/{mp4_name}')}"
        "<figcaption>ego2 · full episode · 4× · autoplay in view</figcaption>"
        "</figure><figure>"
        f'<img loading="lazy" src="{media}/{png_name}" '
        f'alt="Unwrapped yaw curve for shard{shard} episode {episode_id}">'
        "<figcaption>unwrapped Δyaw · ±180° guides</figcaption>"
        "</figure></div></article>"
    )


def _intro_markup(selected: int, complete: int) -> str:
    return (
        _PAGE_CSS
        + '<div class="yaw-console">'
        "<p><strong>Threshold calibration console.</strong> Review full-episode "
        "ego2 motion beside unwrapped head-yaw evidence. The amber 180° boundary "
        "is a candidate decision point, not an automatic filter verdict.</p>"
        '<div class="yaw-method">'
        f"umi400h · selected {selected} · complete {complete} · "
        "metric max |unwrap(yawₜ) − yaw₀| · "
        "labels: walking/turning · stationary turn · VIO drift · VIO jump · unclear"
        "</div>"
        '<div class="yaw-threshold-rail" aria-label="Candidate threshold at 180 degrees">'
        "</div></div>"
    )


def build_page_sections(
    review_root: Path, media_rel: str = "media", media_dir: Path | None = None
) -> list[dict]:
    """Build ordered static-page sections from a review manifest."""
    review_root = Path(review_root)
    manifest = json.loads((review_root / "manifest.json").read_text())
    by_band = {name: [] for name, _low, _high, _quota in YAW_BANDS}
    skipped: list[str] = []
    complete_count = 0

    for episode in manifest["episodes"]:
        band = episode["band"]
        if band not in by_band:
            continue
        try:
            _episode_metrics(episode)
        except ValueError:
            skipped.append(str(episode["dir"]))
            continue
        if not _episode_renderable(review_root, episode, media_dir):
            skipped.append(str(episode["dir"]))
            continue
        by_band[band].append(episode)
        complete_count += 1

    sections: list[dict] = [
        {"h": "Yaw Review — 180° threshold calibration"},
        {"text": _intro_markup(len(manifest["episodes"]), complete_count)},
    ]
    for name, _low, _high, quota in YAW_BANDS:
        episodes = sorted(
            by_band[name],
            key=lambda ep: (
                float(ep["yaw_max_deg"]),
                int(ep["shard"]),
                int(ep["episode"]),
            ),
        )
        open_attr = " open" if name in _OPEN_BANDS else ""
        band_class = "yaw-band threshold-adjacent" if name in _OPEN_BANDS else "yaw-band"
        evidence = "".join(_episode_markup(ep, media_rel) for ep in episodes)
        if not evidence:
            evidence = '<div class="yaw-empty">No complete evidence in this interval.</div>'
        markup = (
            f'<details{open_attr} data-band="{name}" class="{band_class}">'
            "<summary>"
            f'<span class="yaw-band-title">{_BAND_LABELS[name]}</span>'
            f'<span class="yaw-band-count">{len(episodes)} complete / {quota} target</span>'
            "</summary>"
            f'<div class="yaw-evidence-list">{evidence}</div>'
            "</details>"
        )
        sections.append({"text": markup})

    if skipped:
        escaped_dirs = ", ".join(html.escape(directory) for directory in skipped)
        sections.append(
            {
                "text": (
                    '<div class="yaw-skipped"><strong>Incomplete media skipped:</strong> '
                    f"{escaped_dirs}</div>"
                )
            }
        )
    # Trails every band: the script binds the <video> tags parsed before it.
    # Clips inside a collapsed <details> stay unfetched (zero-size elements
    # never intersect) and start prefetching when the band is expanded.
    sections.append({"text": LAZY_VIDEO_SCRIPT})
    return sections


def build_page(review_root: Path, site_dir: Path) -> Path:
    """Render `<site_dir>/yaw_review.html` and return its path. Clips are imported
    from `review_root` only when it carries them; otherwise the ones already under
    `<site_dir>/media/` are reused as-is."""
    review_root, site_dir = Path(review_root), Path(site_dir)
    media_dir = site_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    site_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((review_root / "manifest.json").read_text())
    skipped: list[str] = []
    for episode in manifest["episodes"]:
        try:
            _episode_metrics(episode)
        except ValueError:
            skipped.append(str(episode["dir"]))
            continue
        source = _episode_media(review_root, episode)
        if source is None:
            if not _episode_published_media(media_dir, episode):
                skipped.append(str(episode["dir"]))
            continue
        for src, name in zip(source, _media_names(episode), strict=True):
            shutil.copy2(src, media_dir / name)

    sections = build_page_sections(review_root, media_dir=media_dir)
    output = site_dir / "yaw_review.html"
    page_html = report_html.page(
        f"{SITE_NAME} — Yaw Review",
        sections,
        nav=NAV,
        current="yaw_review.html",
        site_name=SITE_NAME,
    )
    # This page ships no plotly figures — drop the CDN <script> the shared design
    # system always emits rather than make every reader pay for a 3 MB library.
    plotly_tag = f'<script src="{report_html.PLOTLY_CDN}"></script>'
    output.write_text(page_html.replace(plotly_tag, "", 1))
    for directory in skipped:
        print(f"[skip] incomplete media: {directory}")
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", type=Path, default=Path("data/yaw_review"))
    parser.add_argument("--site-dir", type=Path, default=Path("umi400h/v0_study"))
    args = parser.parse_args(argv)

    print(f"[done] wrote {build_page(args.review_root, args.site_dir)}")


if __name__ == "__main__":
    main()
