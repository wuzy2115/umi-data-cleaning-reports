# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path

import numpy as np

from sitegen.vio_filter_page import (
    _BAND_CRITERION, _BAND_ORDER, _bin_index, _calib_row, _episode_block,
    _episode_pose, build_page_sections, episode_curve_fig,
    population_band_rows, population_hist_fig,
)

CALIB = {"n_episodes": 100,
         "ep_xy_max_q": {"0.5": 0.7, "0.9": 1.7, "0.99": 4.2, "1.0": 261.0},
         "ep_z_max_q": {"0.5": 0.6, "0.9": 1.1, "0.99": 1.4, "1.0": 24.8},
         "ep_xy_exceed_frac": {"10.0": 0.00285},
         "ep_z_exceed_frac": {"3.0": 0.00193},
         "worst20_xy": [{"xy_max_m": 261.0, "shard": 1, "episode": 3441}]}


def _mini_review(tmp_path):
    root = tmp_path / "review"
    n = 2000
    for band, s, e in (("extreme", 0, 1892), ("control", 1, 9)):
        d = root / band / f"s{s}_e{e}"
        d.mkdir(parents=True)
        np.savez(d / "pose.npz", t=np.arange(n, dtype=np.float32) / 30.0,
                 head=np.linspace(0, 30, 3 * n).reshape(n, 3).astype(np.float32),
                 wrist1=np.zeros((n, 3), np.float32),
                 wrist2=np.zeros((n, 3), np.float32))
        (d / "ego2_4x.mp4").write_bytes(b"\x00")
        (d / "traj3d.png").write_bytes(b"\x00")
    # a third episode with missing media -> listed as skipped, no crash
    d = root / "high" / "s0_e7"
    d.mkdir(parents=True)
    manifest = {
        "meta": {"root": "/r", "lowdim_version": "v2", "seed": 0,
                 "bands": {"extreme": 1, "high": 1, "control": 1}},
        "episodes": [
            {"band": "extreme", "shard": 0, "episode": 1892, "xy_max_m": 219.5,
             "z_max_m": 4.1, "dir": "extreme/s0_e1892"},
            {"band": "high", "shard": 0, "episode": 7, "xy_max_m": 12.0,
             "z_max_m": 1.0, "dir": "high/s0_e7"},
            {"band": "control", "shard": 1, "episode": 9, "xy_max_m": 3.0,
             "z_max_m": 0.5, "dir": "control/s1_e9"},
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_episode_curve_fig_decimation_and_thresholds(tmp_path):
    root = _mini_review(tmp_path)
    pose = np.load(root / "extreme" / "s0_e1892" / "pose.npz")
    fig = episode_curve_fig(pose, "xy", [5, 10, 20], max_points=600)
    assert {tr["name"] for tr in fig["data"]} == {"head", "wrist1", "wrist2"}
    assert all(len(tr["y"]) <= 600 for tr in fig["data"])   # decimated
    assert {s["y0"] for s in fig["layout"]["shapes"]} == {5, 10, 20}
    figz = episode_curve_fig(pose, "z", [3])
    assert {s["y0"] for s in figz["layout"]["shapes"]} == {3}


def test_build_page_sections_content(tmp_path):
    root = _mini_review(tmp_path)
    sections = build_page_sections(root, CALIB, CALIB, media_rel="media")
    text = json.dumps(sections, default=str)
    assert "shard0 ep1892" in text                    # quotable headline ID
    assert "xy_max 219.5" in text
    assert "vio_ep_s0_e1892.mp4" in text              # media naming contract
    assert "vio-drift-report" in text                 # background link
    assert any("stats" in s for s in sections)        # band tiles
    assert any("plotly" in s for s in sections)       # threshold curves
    assert "s0_e7" in text                             # missing-media episode listed as skipped
    assert "shard0 ep7" not in text.replace("skipped", "")  # ...but gets no evidence block


def test_episode_renders_from_published_media(tmp_path):
    """The `data/vio_review/` snapshot carries no clips — they are already published
    under the site's `media/`. An episode whose review dir holds only pose.npz must
    still render when its clips are found there."""
    root = _mini_review(tmp_path)
    ep = {"band": "extreme", "shard": 0, "episode": 1892, "xy_max_m": 219.5,
          "z_max_m": 4.1, "dir": "extreme/s0_e1892"}
    for name in ("ego2_4x.mp4", "traj3d.png"):
        (root / ep["dir"] / name).unlink()
    assert _episode_pose(root, ep) is None                       # no clips anywhere

    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "vio_ep_s0_e1892.mp4").write_bytes(b"\x00")
    (media_dir / "vio_ep_s0_e1892_traj3d.png").write_bytes(b"\x00")
    assert _episode_pose(root, ep, media_dir) == root / ep["dir"] / "pose.npz"

    sections = build_page_sections(root, CALIB, CALIB, media_dir=media_dir)
    assert "shard0 ep1892" in json.dumps(sections, default=str)


def test_episode_without_pose_is_skipped(tmp_path):
    """Published clips alone aren't enough — the threshold curves need pose.npz."""
    root = _mini_review(tmp_path)
    ep = {"band": "high", "shard": 0, "episode": 7, "xy_max_m": 12.0,
          "z_max_m": 1.0, "dir": "high/s0_e7"}
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "vio_ep_s0_e7.mp4").write_bytes(b"\x00")
    (media_dir / "vio_ep_s0_e7_traj3d.png").write_bytes(b"\x00")
    assert _episode_pose(root, ep, media_dir) is None


def test_band_criterion_covers_all_bands_no_empty_sub():
    assert len(_BAND_ORDER) == 13
    for band in _BAND_ORDER:
        assert _BAND_CRITERION.get(band, "").strip() != ""


def test_population_band_rows_counts_and_fractions():
    table = [
        {"shard": 0, "episode": 0, "xy_max_m": 0.3, "z_max_m": 0.2},   # xy (0,0.5]  z (0,0.5]
        {"shard": 0, "episode": 1, "xy_max_m": 0.7, "z_max_m": 1.2},   # xy (0.5,1]  z (1,1.5]
        {"shard": 0, "episode": 2, "xy_max_m": 1.3, "z_max_m": 0.4},   # xy (1,1.5]  z (0,0.5]
        {"shard": 0, "episode": 3, "xy_max_m": 1.8, "z_max_m": 1.8},   # xy (1.5,2]  z (1.5,2]
        {"shard": 0, "episode": 4, "xy_max_m": 3.0, "z_max_m": 0.1},   # xy (2,5]    z excluded (xy>2)
        {"shard": 0, "episode": 5, "xy_max_m": 7.0, "z_max_m": 2.5},   # xy (5,10]   z excluded (xy>2)
        {"shard": 0, "episode": 6, "xy_max_m": 15.0, "z_max_m": 0.5},  # xy (10,20]  z excluded (xy>2)
        {"shard": 0, "episode": 7, "xy_max_m": 25.0, "z_max_m": 5.0},  # xy > 20     z excluded (xy>2)
    ]
    rows = population_band_rows(table)
    by_key = {(r["axis"], r["band"]): r for r in rows}

    for label in ("(0, 0.5]", "(0.5, 1]", "(1, 1.5]", "(1.5, 2]",
                  "(2, 5]", "(5, 10]", "(10, 20]", "> 20"):
        row = by_key[("xy", label)]
        assert row["episodes"] == 1
        assert row["fraction"] == 12.5

    assert by_key[("z", "(0, 0.5]")]["episodes"] == 2     # 0.2 and 0.4, both xy<=2
    assert by_key[("z", "(0, 0.5]")]["fraction"] == 25.0
    assert by_key[("z", "(0.5, 1]")]["episodes"] == 0
    assert by_key[("z", "(1, 1.5]")]["episodes"] == 1     # 1.2
    assert by_key[("z", "(1.5, 2]")]["episodes"] == 1     # 1.8
    assert by_key[("z", "(2, 3]")]["episodes"] == 0       # 2.5's episode has xy=7>2, excluded
    assert by_key[("z", "> 3")]["episodes"] == 0          # 5.0's episode has xy=25>2, excluded


def test_population_band_rows_empty_table():
    rows = population_band_rows([])
    assert all(r["episodes"] == 0 and r["fraction"] == 0.0 for r in rows)


def _write_pose(path, n=60):
    np.savez(path, t=np.arange(n, dtype=np.float32) / 30.0,
             head=np.linspace(0, 3, 3 * n).reshape(n, 3).astype(np.float32),
             wrist1=np.zeros((n, 3), np.float32), wrist2=np.zeros((n, 3), np.float32))


def test_episode_block_fine_band_gets_fine_thresholds(tmp_path):
    pose_path = tmp_path / "pose.npz"
    _write_pose(pose_path)
    ep = {"band": "z_1p5_2", "shard": 2, "episode": 100, "xy_max_m": 0.8, "z_max_m": 1.7}
    blocks = _episode_block(ep, pose_path, "media")
    fig_xy = next(b["plotly"] for b in blocks if "plotly" in b and
                  b["plotly"]["layout"]["yaxis"]["title"]["text"] == "‖p_xy‖ (m)")
    fig_z = next(b["plotly"] for b in blocks if "plotly" in b and
                 b["plotly"]["layout"]["yaxis"]["title"]["text"] == "|z| (m)")
    assert {s["y0"] for s in fig_xy["layout"]["shapes"]} == {0.5, 1, 1.5, 2}
    assert {s["y0"] for s in fig_z["layout"]["shapes"]} == {0.5, 1, 1.5, 2}


def test_episode_block_coarse_band_gets_coarse_thresholds(tmp_path):
    pose_path = tmp_path / "pose.npz"
    _write_pose(pose_path)
    ep = {"band": "high", "shard": 1, "episode": 3, "xy_max_m": 12.0, "z_max_m": 1.0}
    blocks = _episode_block(ep, pose_path, "media")
    fig_xy = next(b["plotly"] for b in blocks if "plotly" in b and
                  b["plotly"]["layout"]["yaxis"]["title"]["text"] == "‖p_xy‖ (m)")
    fig_z = next(b["plotly"] for b in blocks if "plotly" in b and
                 b["plotly"]["layout"]["yaxis"]["title"]["text"] == "|z| (m)")
    assert {s["y0"] for s in fig_xy["layout"]["shapes"]} == {5, 10, 20}
    assert {s["y0"] for s in fig_z["layout"]["shapes"]} == {1.5, 2, 3}


def test_build_page_sections_with_episode_table(tmp_path):
    root = _mini_review(tmp_path)
    ep_table = [{"shard": 0, "episode": 1, "xy_max_m": 0.3, "z_max_m": 0.2}]
    sections = build_page_sections(root, CALIB, CALIB, media_rel="media",
                                   episode_table=ep_table)
    assert any(s.get("h") == "episode 分布(全量)" for s in sections)
    pop_table = next(s["table"] for s in sections
                     if "table" in s and s["table"] and "axis" in s["table"][0])
    assert any(r["band"] == "(0, 0.5]" and r["axis"] == "xy" and r["episodes"] == 1
               for r in pop_table)


def test_build_page_sections_without_episode_table_omits_population_section(tmp_path):
    root = _mini_review(tmp_path)
    sections = build_page_sections(root, CALIB, CALIB, media_rel="media")
    assert not any(s.get("h") == "episode 分布(全量)" for s in sections)


def test_video_and_img_but_no_pose_is_gated_from_page_and_copy(tmp_path):
    # Regression: an episode with video+img but no pose.npz must be skipped by
    # build_page_sections AND must be gated out of build_page()'s copy loop via the
    # same completeness predicate — the two must never disagree.
    root = tmp_path / "review"
    d = root / "high" / "s2_e5"
    d.mkdir(parents=True)
    (d / "ego2_4x.mp4").write_bytes(b"\x00")
    (d / "traj3d.png").write_bytes(b"\x00")
    ep = {"band": "high", "shard": 2, "episode": 5, "xy_max_m": 12.0,
          "z_max_m": 1.0, "dir": "high/s2_e5"}
    manifest = {
        "meta": {"root": "/r", "lowdim_version": "v2", "seed": 0,
                 "bands": {"high": 1}},
        "episodes": [ep],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))

    # unit-level: the shared completeness predicate must say "incomplete"
    assert _episode_pose(root, ep) is None

    sections = build_page_sections(root, CALIB, CALIB, media_rel="media")
    text = json.dumps(sections, default=str)
    assert "s2_e5" in text                                   # listed as skipped
    assert "shard2 ep5" not in text.replace("skipped", "")   # no evidence block


def test_population_hist_fig_bin_counts_hand_checked():
    # z-kind: upper=2.0, so 2.5 lands in the overflow bucket. All rows keep
    # xy_max_m<=2 so none is excluded by the z isolation filter.
    table = [
        {"shard": 0, "episode": 0, "xy_max_m": 0.5, "z_max_m": 0.05},
        {"shard": 0, "episode": 1, "xy_max_m": 0.5, "z_max_m": 0.14},
        {"shard": 0, "episode": 2, "xy_max_m": 0.5, "z_max_m": 0.15},
        {"shard": 0, "episode": 3, "xy_max_m": 0.5, "z_max_m": 2.5},
    ]
    fig = population_hist_fig(table, "z", bin_m=0.1)
    bar = fig["data"][0]
    counts = dict(zip(bar["x"], bar["y"]))
    assert counts["0.0–0.1"] == 1     # 0.05
    assert counts["0.1–0.2"] == 2     # 0.14, 0.15
    assert counts[">2"] == 1          # 2.5
    assert sum(bar["y"]) == 4


def test_population_hist_fig_z_excludes_high_xy():
    table = [
        {"shard": 0, "episode": 0, "xy_max_m": 0.5, "z_max_m": 0.55},
        {"shard": 0, "episode": 1, "xy_max_m": 5.0, "z_max_m": 0.55},  # excluded, xy>2
    ]
    fig = population_hist_fig(table, "z")
    bar = fig["data"][0]
    counts = dict(zip(bar["x"], bar["y"]))
    assert counts["0.5–0.6"] == 1
    assert sum(bar["y"]) == 1


def test_population_hist_fig_overflow_bucket_present_and_last():
    table = [{"shard": 0, "episode": 0, "xy_max_m": 10.0, "z_max_m": 0.1}]
    fig_xy = population_hist_fig(table, "xy")
    bar_xy = fig_xy["data"][0]
    assert bar_xy["x"][-1] == ">3"
    assert bar_xy["y"][-1] == 1

    table_z = [{"shard": 0, "episode": 0, "xy_max_m": 0.1, "z_max_m": 5.0}]
    fig_z = population_hist_fig(table_z, "z")
    bar_z = fig_z["data"][0]
    assert bar_z["x"][-1] == ">2"
    assert bar_z["y"][-1] == 1


def test_population_hist_fig_thresholds_xy_and_z():
    table = [{"shard": 0, "episode": 0, "xy_max_m": 0.1, "z_max_m": 0.1}]
    fig_xy = population_hist_fig(table, "xy")
    assert len(fig_xy["layout"]["shapes"]) == 1
    assert [a["text"] for a in fig_xy["layout"]["annotations"]] == ["τ = 2.0"]

    fig_z = population_hist_fig(table, "z")
    assert len(fig_z["layout"]["shapes"]) == 2
    texts = {a["text"] for a in fig_z["layout"]["annotations"]}
    assert texts == {"τ = 1.5", "τ = 2.0"}


def test_population_hist_fig_hover_has_count_and_percent():
    table = [{"shard": 0, "episode": i, "xy_max_m": 0.05, "z_max_m": 0.05} for i in range(4)]
    fig = population_hist_fig(table, "xy")
    bar = fig["data"][0]
    assert "hovertemplate" in bar
    assert "%{customdata" in bar["hovertemplate"] and "%{y}" in bar["hovertemplate"]
    assert bar["customdata"][0] == 100.0   # all 4 rows fall in bin 0


def test_build_page_sections_episode_table_adds_two_histograms_before_table(tmp_path):
    root = _mini_review(tmp_path)
    ep_table = [{"shard": 0, "episode": 1, "xy_max_m": 0.3, "z_max_m": 0.2}]
    sections_without = build_page_sections(root, CALIB, CALIB, media_rel="media")
    sections_with = build_page_sections(root, CALIB, CALIB, media_rel="media",
                                        episode_table=ep_table)
    plotly_without = sum(1 for s in sections_without if "plotly" in s)
    plotly_with = sum(1 for s in sections_with if "plotly" in s)
    assert plotly_with == plotly_without + 2

    idx_h = next(i for i, s in enumerate(sections_with) if s.get("h") == "episode 分布(全量)")
    idx_table = next(i for i, s in enumerate(sections_with)
                     if "table" in s and s["table"] and "axis" in s["table"][0])
    between = [s for s in sections_with[idx_h + 1:idx_table] if "plotly" in s]
    assert len(between) == 2


# ---------------------------------------------------------------------------
# Percent-denominator labeling (xy hist = % of all episodes, z hist = % of the
# xy<=2 subset; population_band_rows below is always % of all episodes).
# ---------------------------------------------------------------------------

def test_population_hist_fig_hovertemplate_labels_denominator():
    table = [{"shard": 0, "episode": 0, "xy_max_m": 0.1, "z_max_m": 0.1}]
    fig_xy = population_hist_fig(table, "xy")
    fig_z = population_hist_fig(table, "z")
    assert "全部 episodes" in fig_xy["data"][0]["hovertemplate"]
    assert "xy≤2 子集" in fig_z["data"][0]["hovertemplate"]
    # cross-check: the labels aren't accidentally swapped
    assert "xy≤2 子集" not in fig_xy["data"][0]["hovertemplate"]
    assert "全部 episodes" not in fig_z["data"][0]["hovertemplate"]


def test_population_hist_fig_z_percent_vs_population_band_rows_percent():
    # 80/100 episodes have xy_max_m<=2 (the z-histogram's denominator), all
    # sharing one z bin -> the z histogram reports 100% for that bin, while
    # population_band_rows (denominator = all 100 episodes) reports 80% for
    # the same z bucket. This is the deliberate mismatch finding 1 requires
    # to be labeled, pinned numerically so a future edit can't silently
    # collapse the two conventions into one.
    table = ([{"shard": 0, "episode": i, "xy_max_m": 0.5, "z_max_m": 0.25}
              for i in range(80)]
             + [{"shard": 0, "episode": 80 + i, "xy_max_m": 5.0, "z_max_m": 0.25}
                for i in range(20)])
    fig_z = population_hist_fig(table, "z", bin_m=0.1)
    bar = fig_z["data"][0]
    counts = dict(zip(bar["x"], bar["y"]))
    fractions = dict(zip(bar["x"], bar["customdata"]))
    assert counts["0.2–0.3"] == 80
    assert fractions["0.2–0.3"] == 100.0

    pop_rows = population_band_rows(table)
    z_row = next(r for r in pop_rows if r["axis"] == "z" and r["band"] == "(0, 0.5]")
    assert z_row["episodes"] == 80
    assert z_row["fraction"] == 80.0


def test_build_page_sections_has_denominator_caption_before_population_table(tmp_path):
    root = _mini_review(tmp_path)
    ep_table = [{"shard": 0, "episode": 1, "xy_max_m": 0.3, "z_max_m": 0.2}]
    sections = build_page_sections(root, CALIB, CALIB, media_rel="media",
                                   episode_table=ep_table)
    idx_table = next(i for i, s in enumerate(sections)
                     if "table" in s and s["table"] and "axis" in s["table"][0])
    # the caption sits somewhere before the population table and names both
    # denominators so a reviewer comparing hist% to table% isn't misled
    caption = next((s["text"] for s in sections[:idx_table] if "text" in s
                    and "子集内占比" in s.get("text", "")), None)
    assert caption is not None
    assert "全量占比" in caption


# ---------------------------------------------------------------------------
# Boundary-float regression tests (0.1-grid values the epsilon exists for).
# ---------------------------------------------------------------------------

def test_bin_index_boundary_floats_no_off_by_one():
    n_bins_xy = 30
    assert _bin_index(0.1, 0.1, n_bins_xy) == 1     # -> label "0.1–0.2"
    assert _bin_index(0.2, 0.1, n_bins_xy) == 2     # -> label "0.2–0.3"
    assert _bin_index(0.3, 0.1, n_bins_xy) == 3     # -> label "0.3–0.4" (classic 2.9999996 trap)
    assert _bin_index(2.9999, 0.1, n_bins_xy) == 29
    assert _bin_index(3.0, 0.1, n_bins_xy) == n_bins_xy   # overflow, clamped

    n_bins_z = 20
    assert _bin_index(2.0, 0.1, n_bins_z) == n_bins_z     # overflow, clamped


def test_population_hist_fig_boundary_floats_land_in_expected_bins():
    table = [{"shard": 0, "episode": i, "xy_max_m": v, "z_max_m": 0.0}
             for i, v in enumerate([0.1, 0.2, 0.3, 2.0, 2.9999, 3.0])]
    fig_xy = population_hist_fig(table, "xy", bin_m=0.1)
    counts = dict(zip(fig_xy["data"][0]["x"], fig_xy["data"][0]["y"]))
    assert counts["0.1–0.2"] == 1
    assert counts["0.2–0.3"] == 1
    assert counts["0.3–0.4"] == 1
    assert counts[">3"] == 1        # the 3.0 value overflows, doesn't land in "2.9–3.0"

    table_z = [{"shard": 0, "episode": 0, "xy_max_m": 0.1, "z_max_m": 2.0}]
    fig_z = population_hist_fig(table_z, "z", bin_m=0.1)
    assert fig_z["data"][0]["x"][-1] == ">2"
    assert fig_z["data"][0]["y"][-1] == 1


# ---------------------------------------------------------------------------
# Epsilon scaled to bin_m (finding 3): the guarantee must hold for arbitrary
# bin sizes, not just the 0.1 default, while bin_m=0.1 behavior is unchanged.
# ---------------------------------------------------------------------------

def test_bin_index_epsilon_scales_with_bin_m():
    # bin_m=0.1 regression pin: identical to pre-fix behavior.
    assert _bin_index(0.3, 0.1, 30) == 3

    # A flat epsilon sized for the 0.1 default (~1e-9) is 10x LARGER than a
    # 1e-10 bin_m: added to v it overshoots into the wrong bin entirely. The
    # bin_m-scaled epsilon (bin_m * 1e-6) stays a small fraction of the bin
    # regardless of how fine the grid is.
    bin_m = 1e-10
    v = bin_m * 2.4   # belongs in bin 2
    assert _bin_index(v, bin_m, 100) == 2


# ---------------------------------------------------------------------------
# yaw histogram (kind="yaw"): 10-degree bins over [0, 360) + ">360" overflow,
# no threshold vlines, rows missing yaw_max_deg skipped (old-format tables).
# ---------------------------------------------------------------------------

def test_population_hist_fig_yaw_bins_hand_checked():
    table = [
        {"shard": 0, "episode": 0, "xy_max_m": 0.1, "z_max_m": 0.1, "yaw_max_deg": 5},
        {"shard": 0, "episode": 1, "xy_max_m": 0.1, "z_max_m": 0.1, "yaw_max_deg": 15},
        {"shard": 0, "episode": 2, "xy_max_m": 0.1, "z_max_m": 0.1, "yaw_max_deg": 15.01},
        {"shard": 0, "episode": 3, "xy_max_m": 0.1, "z_max_m": 0.1, "yaw_max_deg": 359.9},
        {"shard": 0, "episode": 4, "xy_max_m": 0.1, "z_max_m": 0.1, "yaw_max_deg": 360.0},
        {"shard": 0, "episode": 5, "xy_max_m": 0.1, "z_max_m": 0.1, "yaw_max_deg": 500},
    ]
    fig = population_hist_fig(table, "yaw", bin_m=10.0)
    bar = fig["data"][0]
    counts = dict(zip(bar["x"], bar["y"]))
    assert counts["0.0–10.0"] == 1      # 5
    assert counts["10.0–20.0"] == 2     # 15, 15.01
    assert counts["350.0–360.0"] == 1   # 359.9
    assert counts[">360"] == 2          # 360.0 (overflow, not "350.0-360.0") and 500
    assert sum(bar["y"]) == 6
    assert fig["layout"]["shapes"] == []       # no threshold vlines for yaw
    assert fig["layout"]["annotations"] == []
    assert "全部 episodes" in bar["hovertemplate"]
    assert "°" in bar["hovertemplate"]
    assert fig["layout"]["xaxis"]["title"]["text"] == \
        "yaw_max (deg, unwrapped |Δyaw| vs. episode start)"


def test_population_hist_fig_yaw_skips_rows_missing_key():
    table = [
        {"shard": 0, "episode": 0, "xy_max_m": 0.1, "z_max_m": 0.1, "yaw_max_deg": 5},
        {"shard": 0, "episode": 1, "xy_max_m": 0.1, "z_max_m": 0.1},   # old-format row
    ]
    fig = population_hist_fig(table, "yaw", bin_m=10.0)
    assert sum(fig["data"][0]["y"]) == 1
    assert fig["data"][0]["customdata"][0] == 100.0   # denominator = present rows only


def test_build_page_sections_yaw_histogram_added_when_present(tmp_path):
    root = _mini_review(tmp_path)
    ep_table = [
        {"shard": 0, "episode": 1, "xy_max_m": 0.3, "z_max_m": 0.2, "yaw_max_deg": 20.0},
        {"shard": 0, "episode": 2, "xy_max_m": 0.4, "z_max_m": 0.3, "yaw_max_deg": 400.0},
    ]
    sections = build_page_sections(root, CALIB, CALIB, media_rel="media",
                                   episode_table=ep_table)
    text = json.dumps(sections, default=str, ensure_ascii=False)
    assert "头部航向角最大偏转" in text
    assert "unwrapped,可>360°" in text

    idx_h = next(i for i, s in enumerate(sections) if s.get("h") == "episode 分布(全量)")
    idx_table = next(i for i, s in enumerate(sections)
                     if "table" in s and s["table"] and "axis" in s["table"][0])
    pop_plotly = [s["plotly"] for s in sections[idx_h + 1:idx_table] if "plotly" in s]
    assert len(pop_plotly) == 3   # xy, z, yaw
    assert pop_plotly[2]["layout"]["xaxis"]["title"]["text"].startswith("yaw_max")

    # caption stays adjacent to the xy/z pair; yaw histogram + its own caption
    # come after it, before the population_band_rows table.
    caption_idx = next(i for i, s in enumerate(sections) if s.get("text", "").startswith(
        "直方图百分比"))
    yaw_h_idx = next(i for i, s in enumerate(sections) if "plotly" in s and
                     s["plotly"]["layout"]["xaxis"]["title"]["text"].startswith("yaw_max"))
    assert caption_idx < yaw_h_idx < idx_table


def test_build_page_sections_yaw_omitted_on_partial_coverage(tmp_path):
    # A mixed-format table (only SOME rows carry yaw_max_deg) must NOT render
    # the yaw histogram: its denominator (annotated rows only) would contradict
    # the "% of 全部 episodes" hover label. Production tables are written by a
    # single scan and are all-or-nothing, so omission is the honest fallback.
    root = _mini_review(tmp_path)
    ep_table = [
        {"shard": 0, "episode": 1, "xy_max_m": 0.3, "z_max_m": 0.2, "yaw_max_deg": 20.0},
        {"shard": 0, "episode": 2, "xy_max_m": 0.4, "z_max_m": 0.3},
    ]
    sections = build_page_sections(root, CALIB, CALIB, media_rel="media",
                                   episode_table=ep_table)
    assert "头部航向角最大偏转" not in json.dumps(sections, default=str, ensure_ascii=False)


def test_build_page_sections_yaw_omitted_when_no_row_has_key(tmp_path):
    # Regression: an old-format episode table (no yaw_max_deg on any row)
    # must produce identical sections to before this field existed.
    root = _mini_review(tmp_path)
    ep_table = [{"shard": 0, "episode": 1, "xy_max_m": 0.3, "z_max_m": 0.2}]
    sections = build_page_sections(root, CALIB, CALIB, media_rel="media",
                                   episode_table=ep_table)
    assert "头部航向角最大偏转" not in json.dumps(sections, default=str, ensure_ascii=False)

    idx_h = next(i for i, s in enumerate(sections) if s.get("h") == "episode 分布(全量)")
    idx_table = next(i for i, s in enumerate(sections)
                     if "table" in s and s["table"] and "axis" in s["table"][0])
    pop_plotly = [s for s in sections[idx_h + 1:idx_table] if "plotly" in s]
    assert len(pop_plotly) == 2   # unchanged: xy + z only


# ---------------------------------------------------------------------------
# _calib_row: ep |Δyaw| q50/q90/q99/max column.
# ---------------------------------------------------------------------------

def test_calib_row_yaw_quantile_column():
    c = {"ep_yaw_max_q": {"0.5": 12.0, "0.9": 45.0, "0.99": 210.0,
                          "0.999": 300.0, "1.0": 720.0}}
    row = _calib_row("test", c)
    assert row["ep |Δyaw| q50/q90/q99/max"] == "12.0/45.0/210.0/720.0"


def test_calib_row_yaw_quantile_fallback_dash():
    row = _calib_row("test", {})
    assert row["ep |Δyaw| q50/q90/q99/max"] == "—/—/—/—"


def test_clips_defer_their_fetch_and_the_script_trails_them(tmp_path):
    # ~70 clips at 1.6 Mbps: none may carry a bare `src` (the page would fetch
    # them all), and the binding script must come after the last one, since it
    # only sees the video tags parsed before it.
    root = _mini_review(tmp_path)

    markup = "\n".join(
        s["text"] for s in build_page_sections(root, CALIB, CALIB) if "text" in s
    )

    assert 'data-src="media/vio_ep_s0_e1892.mp4"' in markup
    assert 'src="media/vio_ep_s0_e1892.mp4"' not in markup.replace("data-src=", "")
    binder = 'querySelectorAll("video[data-src]")'
    assert markup.count(binder) == 1
    assert markup.rindex("data-src=") < markup.index(binder)
