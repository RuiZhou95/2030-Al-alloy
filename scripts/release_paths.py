"""Repository path configuration.

Set ALLOY_PRIVATE_ROOT to a private directory containing a data/ subdirectory.
Set ALLOY_OUTPUT_ROOT to the desired analysis-output directory.
No data are distributed with this repository.
"""
from __future__ import annotations
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = Path(os.environ.get("ALLOY_PRIVATE_ROOT", REPO_ROOT / "private_data")).resolve()
OUTPUT_ROOT = Path(os.environ.get("ALLOY_OUTPUT_ROOT", REPO_ROOT / "outputs")).resolve()
SCIENTIFIC_OUTPUT = OUTPUT_ROOT / "scientific"
PAPER_OUTPUT = OUTPUT_ROOT / "paper"

for directory in (
    SCIENTIFIC_OUTPUT / "data",
    SCIENTIFIC_OUTPUT / "audit",
    PAPER_OUTPUT / "data",
    PAPER_OUTPUT / "data_materials_v2",
    PAPER_OUTPUT / "audit",
):
    directory.mkdir(parents=True, exist_ok=True)
