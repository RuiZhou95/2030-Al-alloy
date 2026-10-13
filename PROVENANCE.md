# Provenance and verification boundaries

## Author-confirmed experimental role

On 2026-10-13 the authors confirmed that `al_150`, `al_191`, `al_243` and `al_48` represent independently fabricated new trials after model design. The authors did not supply the corresponding design recipes, fabrication logs or raw test records for this release. This is an author attestation, not a chronology established by this repository.

## Historical selection routine

The preserved `historical/00_data_overview.py::select_verification_samples` computes global min-max scales from measured YS, UTS and EL, takes each record's minimum normalized response, and selects the largest score within each grade. When applied to the combined released 284+4 records, it selects the same four IDs; its scores match the stored `balance_score` within floating-point rounding.

This observable fact does not authenticate the physical trial chronology and is not erased by the authors' confirmation. The relation between this historical routine and the physical design/fabrication sequence cannot be resolved without the unavailable records. Do not claim that a successful calculation replay verifies prospective candidate selection. Do not run this routine to recreate a supposedly blind prospective split.

The published split is fixed. `balance_score` is preserved for audit, never used as a predictor or used to reselect trial rows by the supported reproduction commands. No property value, sample ID, or stored split membership was changed to resolve this issue.

The workbook-import stage recorded an earlier exclusion of `al_227` and a historical correction of strength-column labels. This release distributes the processed analysis inputs, not the unavailable underlying workbook revision history or instrument records. The archival comments remain visible and are not current claims about independent experimental timing.

## Actual code and release adaptations

- `historical/01_feature_screen.py`, `02_train_models.py`, `05_multiobjective.py` and `07_verification.py` are preserved working-source copies. Their historical comments and exploratory figure routines are not rewritten as the current manuscript methods.
- In `historical/00_data_overview.py`, only the machine-specific workbook path is replaced by the `ALLOY_SOURCE_WORKBOOK` environment variable. The source hash and the adaptation are recorded.
- `scripts/15_materials_source_analysis.py` restores an omitted actual source-analysis stage. Its root paths are made portable and its historical snapshot input is redirected to the byte-identical released trial table.
- The published scientific scripts came from the actual supplementary-analysis implementation. Root paths and internal script filenames were adapted for public portability. Function-level differences are listed in `CODE_PROVENANCE.json` and checked in the release audit.
- The final main-figure script is the actual final plotting source, not a generic replacement example. Its plotting and scientific functions are retained.
- `prepare_reproduction.py`, `reproduce_reported_rf.py` and `build_paper_source_tables.py` are new, explicitly labelled release wrappers. They stage reference tables or call/extract existing calculations; they do not invent the original design history.

## Do not conflate three model roles

The historical RF directly predicts YS, UTS and EL. It uses stored hyperparameters and the original preprocessing and error-scale calculation. Its 10-fold error scale is reproduced as historical practice, not upgraded to nested, batch-independent uncertainty.

The later supplementary models use YS, Δσ and EL, reconstruct UTS, and supply the 48,000-candidate route-preserving analysis. They are not the same model as the historical RF. A matching figure or matching error statistic is not evidence that this later candidate grid produced the four physical trial specimens.

Published reference outputs are data for replay and inspection. Staging them is not fresh training. Numerical tests distinguish RF refitting, descriptor/table recalculation, source-output equality and figure generation. The tests do not establish causal metallurgy, independent production batches, or experimental chronology.
