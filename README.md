# 2030-Al-alloy

Analysis code accompanying a study of cross-grade composition-process-property
relationships and model-guided experimental design in 2024, 5083, 6082 and
7075 aluminium alloys.

## Scope

The repository contains the analysis pipeline used for:

- data-quality and grade-identity diagnostics;
- grouped and nested model validation;
- final model fitting and prospective trial evaluation;
- applicability-distance analysis;
- model interpretation and symbolic-regression auditing;
- route-constrained candidate generation;
- systematic descriptor generation and stability screening; and
- manuscript and supplementary figure generation.

Data, trained-model binaries, generated results, figures and manuscript files
are not distributed in this repository.

## Repository layout

    config/              Analysis settings and input-data schema
    scripts/             Numbered analysis and figure-generation scripts
    tests/               Release-integrity tests
    environment.yml      Conda environment specification
    requirements.txt     Pinned Python dependencies

## Environment

The analyses were run with Python 3.11.4. A matching environment can be
created with either:

    conda env create -f environment.yml
    conda activate al-alloy-2030

or:

    python -m venv .venv
    python -m pip install -r requirements.txt

## Private input data

Set ALLOY_PRIVATE_ROOT to a private directory containing a data subdirectory.
The required primary inputs are:

- 01_processed_data.csv
- 02_engineered_features.csv
- verification_holdout.csv
- variance_decomposition.csv
- symbolic-regression audit files described in config/data_schema.yaml

The experimental data used in the study are available from the corresponding
author upon request.

## Output location

Set ALLOY_OUTPUT_ROOT to a writable directory. If it is not set, outputs are
written under outputs/ in the repository. Generated outputs are excluded by
.gitignore.

Example:

    export ALLOY_PRIVATE_ROOT=/path/to/private/project
    export ALLOY_OUTPUT_ROOT=/path/to/analysis/output

## Analysis order

Run the numbered scripts from the repository root. The principal dependency
order is:

    python scripts/00_data_quality_audit.py
    python scripts/02_nested_validation.py
    python scripts/03_summarize_validation.py
    python scripts/04_fit_models.py
    python scripts/05_external_evaluation.py
    python scripts/05b_support_distance.py
    python scripts/06_interpretability.py
    python scripts/07_symbolic_regression_audit.py
    python scripts/08_candidate_generation.py
    python scripts/09_descriptor_screening.py
    python scripts/10_descriptor_roles.py
    python scripts/11_make_main_figures.py
    python scripts/12_make_supplementary_figures.py
    python scripts/13_make_trial_distance_figure.py

Some validation runs are computationally intensive. Command-line options for
the nested-validation stage can be inspected with:

    python scripts/02_nested_validation.py --help

## Release checks

Run:

    python -m pytest

The tests verify syntax, release boundaries, absence of project-specific
absolute paths and common credential patterns.

## Citation

Please cite the associated article and the archived code release. Citation
metadata are provided in CITATION.cff.

## License

The code is released under the MIT License. See LICENSE.
