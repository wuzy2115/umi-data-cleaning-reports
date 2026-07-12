# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The site's page set — the single source of truth for the top nav.

Every page builder renders with this nav, so adding/removing a tab is a one-line
change here plus a builder module. `NAV[0]` is the site home: `html.page()` points
the header brand link at it, and the repo-root `index.html` redirects to it.
"""
from __future__ import annotations

NAV: list[tuple[str, str]] = [
    ("report.html", "Report"),
    ("vio_filter.html", "VIO Filter"),
    ("yaw_review.html", "Yaw Review"),
]

SITE_NAME = "umi400h"
