# 2030-Al-alloy

Data and analysis code for **Machine Learning for Cross-Grade Aluminium Alloy Design through Composition, Processing and Strength–Ductility Relationships**.

## Release scope

This release includes the unchanged processed modelling dataset (284 records), four author-confirmed new-trial records, the historical RF parameters/predictions, and numerical source tables supporting the main and supplementary analyses. Code is licensed under **MIT**; files in `data/` are licensed under **CC BY 4.0**. See `LICENSE`, `LICENSE-DATA`, and `data/README.md`.

**Computational reproducibility and experimental provenance are different.** The authors confirmed on 2026-10-13 that the four trial records came from independent post-design fabrication. Design recipes, fabrication logs and raw testing records were not provided for this release and are not independently verified by the code audit. An available historical preprocessing routine also selects these same four IDs using measured outcomes. Its logic and the observed overlap are retained and documented in `PROVENANCE.md`; they must not be hidden or treated as proof of prospective selection. The supported reproduction starts from the released fixed modelling/trial tables and never reselects trial records.

## Three distinct computational stages

| Stage | Actual source / entry | Scope |
|---|---|---|
| Historical RF | `historical/02_train_models.py`, `historical/07_verification.py`; replay with `scripts/reproduce_reported_rf.py` | Direct YS/UTS/EL predictions reported for the four records |
| Supplementary analysis | `scripts/02_nested_validation.py` through `08_candidate_generation.py` | Grouped validation, YS/Δσ/EL models, reconstructed UTS and route-constrained candidate analysis; not interchangeable with the historical RF |
| Materials tables and final figures | `scripts/15_materials_source_analysis.py`, `build_paper_source_tables.py`, `09_descriptor_screening.py`, `10_descriptor_roles.py`, `11_make_main_figures.py`, `14_make_current_supplement.py` | Tables, descriptors and figure rendering used in the manuscript revisions |

`CODE_PROVENANCE.json` maps computational files to actual working-source filenames and hashes. New wrappers are explicitly identified as release/reproduction utilities, not historical experiment-selection code. Old file names containing `prospective` or `freeze` describe stored analysis artifacts, not independently authenticated experiment dates.

## Environment

The verified historical environment uses Python 3.11, NumPy 1.24.3, pandas 1.5.3, scikit-learn 1.3.0 and XGBoost 1.3.3. See `requirements.txt` / `environment.yml`. Other versions may not reproduce tree-model outputs exactly.

```bash
python -m venv .venv
# Activate the environment using the command for your operating system.
python -m pip install -r requirements.txt
```

Inputs default to this repository's `data/` directory. `ALLOY_PRIVATE_ROOT` remains a backward-compatible optional override for a directory containing `data/`. Outputs default to `outputs/`; set `ALLOY_OUTPUT_ROOT` to a new directory for each independent run. Reference inputs are never overwritten.

## Verified replay and recomputation

Run from the repository root:

```bash
python scripts/prepare_reproduction.py
python scripts/verify_features.py
python scripts/reproduce_reported_rf.py --include-cv
python scripts/build_paper_source_tables.py
python scripts/09_descriptor_screening.py
python scripts/10_descriptor_roles.py
python scripts/verify_recomputed_tables.py
python scripts/11_make_main_figures.py
python scripts/14_make_current_supplement.py
python -m pytest -q
```

`prepare_reproduction.py` verifies checksums and stages **published numerical reference outputs**. It does not retrain nested models. The RF command refits the actual historical RF calculation, checks all 12 predictions and optionally recomputes the original 10-fold error scales. The table and descriptor steps recompute their quantities from released records; the figure steps render actual released/recomputed quantities. The distinction between replay and fresh model training is deliberate.

## Fresh supplementary model fitting

Use a separate empty output directory. The computational order is:

```bash
python scripts/00_data_quality_audit.py
python scripts/02_nested_validation.py --output-tag nested_v1_full
python scripts/03_summarize_validation.py
python scripts/04_fit_models.py
python scripts/05_external_evaluation.py
python scripts/05b_support_distance.py
python scripts/06_interpretability.py
python scripts/08_candidate_generation.py
```

Full nested validation, interpretation and candidate generation can be expensive. Their released reference tables permit scrutiny without rerunning those computations. The release audit records exactly which stages were executed; do not interpret successful figure replay or syntax tests as independent full retraining. Historic symbolic-regression exploration is not needed for the current paper's main numerical results and is not part of the quick replay.

## Data and citation

See `data/README.md`, `data/field_dictionary.csv`, `data/manifest.json`, and `data/CITATION.cff`. Numerical records use their original precision and original categorical labels, with an English route dictionary. `verification_holdout.csv` is retained as a historical filename; its role is defined in `data/split_provenance.json`.

Suggested availability wording:

> The processed modelling and author-confirmed trial datasets, numerical source tables, and analysis code are available at https://github.com/RuiZhou95/2030-Al-alloy. Data are licensed under CC BY 4.0 and code under the MIT License. The release documents the distinction between the historical random-forest calculation, supplementary analyses, and figure reproduction, together with the limits of the available experimental provenance records.

This repository does not contain manuscripts, generated figures, private credentials or fitted-model binaries. No DOI has been assigned by this release. Cite the tagged version and commit; a DOI-bearing archive can be added separately.
