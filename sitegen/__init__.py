# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Generator for the umi-data-cleaning-reports site.

Self-contained: the page builders read the data snapshots under `data/` and write
HTML into `umi400h/<run>/`. No dependency on the cleaning pipeline (cosmos) — the
numbers each page needs were snapshotted into `data/` when the pages were first
built. `python -m sitegen.build` rebuilds every page.
"""
