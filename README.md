# GRAPE + CRAB QOC

Code and notebooks for quantum optimal control experiments (GRAPE/CRAB/Adam).

- Python modules live under `src/`.
- Reproducible notebooks live under `notebooks/`.
- Tracked documentation lives under `docs/`.

## Baseline data

A reference CRAB baseline is stored in `data/baselines/_baseline_crab/arrays.npz` with metadata alongside it. Notebooks and tests load it through `src.qoc_common_crab.load_baseline_crab()`.

To regenerate the dataset, run:

```
python scripts/refresh_crab_baseline.py
```

Optional arguments (`--Nt`, `--T`, `--seed`, `--out`) mirror the notebook defaults.

## Documentation

- PDF: docs/GRAPE_CRAB_ADAM_Guide.pdf
- Text: docs/GRAPE_CRAB_ADAM_Guide.md (also mirrored as .txt)

