# GRAPE + CRAB QOC

Code and notebooks for quantum optimal control experiments (GRAPE/CRAB/Adam).

- Python modules live under `src/`.
- Reproducible notebooks live under `notebooks/`.
- Tracked documentation lives under `docs/`.

## Baseline data

Deterministic GRAPE baselines are built in memory through `src.baselines.build_grape_baseline` using the `GrapeBaselineConfig` dataclass. Notebooks supply the configuration parameters directly and tests build their own defaults via the same API.

When you need to materialise the arrays on disk (e.g., to snapshot an experiment), call `src.baselines.write_baseline(arrays, metadata, out_dir)` after constructing them in Python. No `.npz` files are required for the default workflow.

## Documentation

- PDF: docs/GRAPE_CRAB_ADAM_Guide.pdf
- Text: docs/GRAPE_CRAB_ADAM_Guide.md (also mirrored as .txt)
