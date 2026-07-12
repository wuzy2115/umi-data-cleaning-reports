# UMI data-cleaning reports

- [umi400h / v0_study](umi400h/v0_study/report.html) — threshold study. Tabs: **Report** (before/after
  + funnel + dedup highlights) · **VIO Filter** (episode-level drift review) · **Yaw Review**
  (180° threshold calibration)
- [umi400h / v1a_dedup_probe](umi400h/v1a_dedup_probe/index.html) — strategy-a dedup probe
  (tau_vis .988, tau_act .065, motion off)

## Rebuilding

The site is generated from the snapshots in `data/` by `sitegen/` — no dependency on the cleaning
pipeline:

```bash
python3 -m sitegen.build          # rebuild every page of v0_study
python3 -m sitegen.build --check  # verify the committed HTML matches generator + data
```

`v1a_dedup_probe` predates the generator and is kept as a frozen artifact. See `CLAUDE.md` for the
architecture and the test commands.
