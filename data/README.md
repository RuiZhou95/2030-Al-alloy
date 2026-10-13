# Released study data

These are processed study records and numerical analysis/source tables, not raw instrument curves or a complete fabrication-log archive. Licence: CC BY 4.0 (`../LICENSE-DATA`). Software has the separate MIT licence.

## Primary files

| File | Records | Meaning |
|---|---:|---|
| `01_processed_data.csv` | 284 | Fixed modelling records; 2024/5083/6082/7075 counts 64/32/89/99 |
| `02_engineered_features.csv` | 284 | Historical 27-feature numeric matrix, metadata and YS/UTS/EL responses |
| `verification_holdout.csv` | 4 | Fixed author-confirmed trial records; historical filename retained |
| `results.json` | three target result groups | Historical six-model search results and RF hyperparameters |
| `verification_predictions.csv` | 12 target-record rows | Historical prediction results and original CV error scales |
| `variance_decomposition.csv` | target summaries | Historical aggregate decomposition used for comparison |

`reference/scientific/` holds published grouped-validation, explanation and candidate-analysis reference outputs. `reference/paper/` holds materials summaries and RF paper-source tables. `reference/descriptors/` holds descriptor screening reference outputs. All are numerical data, not generated figure files. `manifest.json` records file hashes, source stages, shapes and roles.

## Variables, units and missingness

- Element columns ending `_wt_pct` are weight percent. YS and UTS are MPa; elongation is percent. Absolute elongation differences and RMSE therefore have units of percentage points.
- Temperature columns use degrees Celsius, treatment time uses hours, reduction uses percent and specimen thickness uses mm. `field_dictionary.csv` records field names, units where specified, and coverage. Fields with no values are retained from the original general-purpose schema and do not represent measurements made in this study.
- CSV empty fields remain missing. The historical feature-engineering code converts non-numeric/missing numeric inputs to zero and drops constant features. Consequently zero in an engineered field must not automatically be interpreted as an independently measured physical zero. See the preserved implementation before reuse.
- Original Chinese categorical labels are preserved to avoid changing model encodings. `route_dictionary.csv` provides English descriptions. Numeric grade labels and `sample_id` are identifiers, not responses.
- The original `balance_score` is an outcome-based historical diagnostic, retained transparently in the four-record file. It is not an input feature and is not used to define or update the fixed split by supported replay commands.

## Provenance and experimental scope

The authors confirm that the four trial records are independently fabricated post-design materials. Design recipes, fabrication logs and raw testing records have not been deposited. Available historical code also selects the same four rows by measured-property scores. These facts and their unresolved chronology are separately documented in `../PROVENANCE.md` and `split_provenance.json`. Data release and successful prediction replay do not authenticate prospective selection or physical experimental chronology.

The modelling and trial tables have disjoint IDs. No response values, IDs or split memberships were changed for this release. The historical workbook importer documents exclusion of `al_227`; the released modelling table is already processed. This release does not silently reintroduce or remove records.

## Reproduction

Start with the commands in `../README.md`. Numerical reference tables make the figure workflow usable without refitting every model. Such replay is explicitly distinct from RF refitting, descriptor/statistics recalculation, and full nested-model retraining. `../REPRODUCIBILITY.md` records what the release audit actually executed.

## Attribution

Cite the four dataset authors, dataset title, version 1.1.0, GitHub repository URL and tagged commit. See `CITATION.cff`. No DOI is claimed. If you transform the data, identify your changes and cite the original release.
