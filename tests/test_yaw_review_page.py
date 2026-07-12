# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path

import pytest

from sitegen import html as report_html
from sitegen.yaw_review_page import (
    YAW_BANDS,
    _episode_media,
    _episode_published_media,
    _media_names,
    build_page_sections,
    main,
)


def _review_root(tmp_path: Path) -> tuple[Path, list[dict]]:
    root = tmp_path / "review"
    episodes = []
    for index, (band, _low, _high, _quota) in enumerate(YAW_BANDS):
        shard = index % 3
        episode = 100 + index
        media_dir = f"{band}/s{shard}_e{episode}"
        directory = root / media_dir
        directory.mkdir(parents=True)
        (directory / "ego2_4x.mp4").write_bytes(b"video")
        (directory / "yaw_curve.png").write_bytes(b"plot")
        episodes.append(
            {
                "band": band,
                "shard": shard,
                "episode": episode,
                "yaw_max_deg": 45.0 + 60.0 * index,
                "xy_max_m": 0.25 + index,
                "z_max_m": 0.1 + index / 10,
                "duration_s": 12.5 + index,
                "dir": media_dir,
            }
        )

    missing_dir = root / "180-200" / "s9_e999"
    missing_dir.mkdir(parents=True)
    (missing_dir / "ego2_4x.mp4").write_bytes(b"video only")
    episodes.append(
        {
            "band": "180-200",
            "shard": 9,
            "episode": 999,
            "yaw_max_deg": 190.0,
            "xy_max_m": 2.0,
            "z_max_m": 0.5,
            "duration_s": 8.0,
            "dir": "180-200/s9_e999",
        }
    )
    manifest = {
        "meta": {
            "bands": {
                name: sum(ep["band"] == name for ep in episodes)
                for name, _low, _high, _quota in YAW_BANDS
            }
        },
        "episodes": episodes,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, episodes


def _raw_html(sections: list[dict]) -> str:
    return "\n".join(section["text"] for section in sections if "text" in section)


def test_sections_have_nine_ordered_disclosures_and_only_boundary_bands_open(tmp_path):
    root, _episodes = _review_root(tmp_path)

    markup = _raw_html(build_page_sections(root))

    assert markup.count("<details") == 9
    positions = [markup.index(f'data-band="{name}"') for name, *_rest in YAW_BANDS]
    assert positions == sorted(positions)
    assert markup.count("<details open") == 2
    assert '<details open data-band="160-180"' in markup
    assert '<details open data-band="180-200"' in markup


def test_complete_sample_contains_metadata_and_lazy_deterministic_media(tmp_path):
    root, episodes = _review_root(tmp_path)
    sample = episodes[0]

    markup = _raw_html(build_page_sections(root, media_rel="assets"))

    assert "shard0 ep100" in markup
    assert "yaw 45.0°" in markup
    assert "xy 0.25 m" in markup
    assert "z 0.1 m" in markup
    assert "duration 12.5 s" in markup
    # The clip carries data-src, never src: the page must not fetch ~100 videos
    # on load. lazy_video.py's script attaches the real src near the viewport.
    assert 'data-src="assets/yaw_ep_s0_e100.mp4"' in markup
    assert 'src="assets/yaw_ep_s0_e100.mp4"' not in markup.replace("data-src=", "")
    assert (
        '<img loading="lazy" '
        'src="assets/yaw_ep_s0_e100_yaw.png"' in markup
    )
    assert sample["dir"] not in markup


def test_incomplete_media_is_skipped_and_reported(tmp_path):
    root, _episodes = _review_root(tmp_path)

    markup = _raw_html(build_page_sections(root))

    assert "180-200/s9_e999" in markup
    assert "shard9 ep999" not in markup
    assert "yaw_ep_s9_e999.mp4" not in markup


def test_episode_media_rejects_directory_outside_review_root(tmp_path):
    root = tmp_path / "review"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "ego2_4x.mp4").write_bytes(b"video")
    (outside / "yaw_curve.png").write_bytes(b"plot")
    episode = {"dir": "../outside", "shard": 7, "episode": 8}

    assert _episode_media(root, episode) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shard", True),
        ("shard", "7"),
        ("shard", -1),
        ("shard", 7.0),
        ("episode", False),
        ("episode", "8"),
        ("episode", -1),
        ("episode", 8.0),
    ],
)
def test_episode_identity_requires_strict_nonnegative_integers(
    tmp_path, field, value
):
    root, episodes = _review_root(tmp_path)
    episode = episodes[0]
    episode[field] = value

    assert _episode_media(root, episode) is None
    with pytest.raises(ValueError, match=field):
        _media_names(episode)


def test_malicious_identity_is_not_rendered_or_copied(tmp_path):
    root, episodes = _review_root(tmp_path)
    episodes[0]["shard"] = "../../outside"
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "meta": {"bands": {}},
                "episodes": episodes,
            }
        )
    )
    site = tmp_path / "site"

    markup = _raw_html(build_page_sections(root))
    main(["--review-root", str(root), "--site-dir", str(site)])

    assert "../../outside" not in markup
    assert not any("outside" in path.name for path in site.rglob("*"))
    assert len(list((site / "media").iterdir())) == 16


@pytest.mark.parametrize(
    "field", ["yaw_max_deg", "xy_max_m", "z_max_m", "duration_s"]
)
def test_non_numeric_metric_values_are_not_rendered(tmp_path, field):
    root, episodes = _review_root(tmp_path)
    injection = '<img src=x onerror="alert(1)">'
    episodes[0][field] = injection
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "meta": {"bands": {}},
                "episodes": episodes,
            }
        )
    )

    markup = _raw_html(build_page_sections(root))

    assert injection not in markup
    assert "shard0 ep100" not in markup


def test_main_writes_page_and_copies_complete_pairs(tmp_path):
    root, episodes = _review_root(tmp_path)
    site = tmp_path / "site"

    argv = ["--review-root", str(root), "--site-dir", str(site)]
    main(argv)
    main(argv)   # rebuilding is idempotent

    page = (site / "yaw_review.html").read_text()
    assert "Yaw Review" in page
    assert 'href="yaw_review.html" class="active"' in page
    assert page.count('href="yaw_review.html"') == 1
    # no plotly figures on this page — the shared CDN <script> must be stripped
    assert f'<script src="{report_html.PLOTLY_CDN}"></script>' not in page

    complete = episodes[:-1]
    expected_names = {
        name
        for ep in complete
        for name in (
            f"yaw_ep_s{ep['shard']}_e{ep['episode']}.mp4",
            f"yaw_ep_s{ep['shard']}_e{ep['episode']}_yaw.png",
        )
    }
    assert {path.name for path in (site / "media").iterdir()} == expected_names


def test_episode_renders_from_published_media(tmp_path):
    """`data/yaw_review/` holds only the manifest — the clips are already published
    under the site's `media/`, and an episode must still render off them."""
    root, episodes = _review_root(tmp_path)
    episode = episodes[0]
    for name in ("ego2_4x.mp4", "yaw_curve.png"):
        (root / episode["dir"] / name).unlink()
    assert _episode_media(root, episode) is None            # gone from the review root

    media_dir = tmp_path / "media"
    media_dir.mkdir()
    for name in _media_names(episode):
        (media_dir / name).write_bytes(b"\x00")
    assert _episode_published_media(media_dir, episode) is True

    markup = _raw_html(build_page_sections(root, media_dir=media_dir))
    assert f"shard{episode['shard']} ep{episode['episode']}" in markup


def test_lazy_video_script_is_emitted_once_after_every_clip(tmp_path):
    # The script binds `video[data-src]` at parse time, so a clip emitted after
    # it would never load — and a second copy would double every observer.
    root, _episodes = _review_root(tmp_path)

    markup = _raw_html(build_page_sections(root))

    binder = 'querySelectorAll("video[data-src]")'
    assert markup.count(binder) == 1
    assert markup.rindex("data-src=") < markup.index(binder)
