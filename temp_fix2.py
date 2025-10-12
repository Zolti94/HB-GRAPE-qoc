import nbformat
from pathlib import Path

paths = [
    (Path("notebooks/10_StepSize_Methods.ipynb"), 0),
    (Path("notebooks/11_Basis_Capacity_Sweep.ipynb"), 0),
    (Path("notebooks/12_costfunction_types.ipynb"), 0),
]

markdown = """# Notebook Execution Guide

1. Activate the repo context (first code cell) – ensures `.venv` and `src/` imports.
2. Stage 1 (baseline/optimization) – run once per configuration.
3. Stage 2 (analysis/plots) – rerun freely after Stage 1 completes.
4. Stage 3 (exports or optional studies) – execute as needed.

Keep outputs tidy and re-run `Stage 1` if parameters change.
"""

for path, idx in paths:
    nb = nbformat.read(path.open("r", encoding="utf-8"), as_version=4)
    nb.cells[idx].source = markdown
    nbformat.write(nb, path.open("w", encoding="utf-8"))
