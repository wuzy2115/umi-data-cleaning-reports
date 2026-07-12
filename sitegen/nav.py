# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The page set of every report run — the single source of truth for the top nav.

One entry per tab, in nav order. `RUNS[run][0]` is that run's home page: `html.page()`
points the header brand link at it, and the repo-root `index.html` redirects to the
home page of `DEFAULT_RUN`.

Adding or removing a tab is an edit here plus a builder module; `build.py` refuses to
run if a run lists a page nothing knows how to build.
"""
from __future__ import annotations

SITE_NAME = "umi400h"
DEFAULT_RUN = "v0_study"

RUNS: dict[str, list[tuple[str, str]]] = {
    # The live run: rebuilt from the snapshots under data/ by the page builders.
    "v0_study": [
        ("report.html", "Report"),
        ("vio_filter.html", "VIO Filter"),
        ("yaw_review.html", "Yaw Review"),
    ],
    # A finished probe, kept as a frozen artifact: its upstream numbers are gone, so
    # `frozen_page` re-renders each page from the published page itself. The pages stay
    # editable and stay on the current design system; their content can't be recomputed.
    "v1a_dedup_probe": [
        ("index.html", "Overview"),
        ("report.html", "Report"),
        ("funnel.html", "Funnel"),
        ("motion.html", "Motion"),
        ("gripper.html", "Gripper"),
        ("dedup_threshold.html", "Dedup τ"),
        ("dedup_clusters.html", "Dedup Clusters"),
        ("samples.html", "Samples"),
        ("provenance.html", "Provenance"),
    ],
}

# The v0_study builders render this nav.
NAV = RUNS[DEFAULT_RUN]
