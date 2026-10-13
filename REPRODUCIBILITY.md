# Release verification record

Audit date: 2026-10-13. Release: 1.1.0. This record concerns computational reproduction, not independent experimental authentication.

## Executed and passed

| Check | Result |
|---|---|
| Fixed modelling / trial inputs | 284 / 4 rows; disjoint IDs; original primary CSV bytes unchanged |
| Modelling grade counts | 2024=64, 5083=32, 6082=89, 7075=99 |
| Historical engineered matrix | All 284 rows and 27 features recalculated and matched |
| Historical RF fits | All 12 predictions matched the paper source table within 6×10⁻¹⁴ |
| Historical 10-fold RF error scales | Recomputed and matched the stored scales |
| Materials / descriptor tables | 12 key tables recalculated and compared cell by cell with tolerance 1×10⁻⁹ relative / 1×10⁻¹⁰ absolute |
| Descriptor screening | 118 descriptors; original 1000-resample procedure rerun |
| Candidate reference data | 48,000 rows; 46,705 satisfy the released support and feasibility flags |
| Final figures | Seven main and seven supplementary PNGs regenerated and byte-identical to retained final working-source PNGs; figures are not distributed |
| Release tests | 11 tests passed, covering manifests, identities, physical inequalities, key numeric results, provenance disclosure, compilation and release/security boundaries |
| Clean-clone run | A fresh clone with no pre-existing outputs passed the manifest/tests, historical features and RF prediction replay, materials/descriptor recalculation, 12-table comparison and all 14 figure-generation steps |
| Data security scan | No private absolute paths, email-address strings, private-key headers or common access-token patterns in released data files |

The RF replay RMSE values are 16.766003187839008 MPa (YS), 15.980032142896633 MPa (UTS), and 1.5418439435891311 percentage points (EL). They round to 16.77, 15.98 and 1.54. Historical CV-RMSE scales are 28.064686358910976 MPa, 25.54117720368965 MPa and 1.7412082653759184 percentage points, respectively.

## Release repairs

1. Added the actual historical training, feature-engineering, multi-objective and verification sources, plus stored historical search parameters.
2. Restored the omitted materials-source-table computation stage and portable wrappers.
3. Replaced the public audit's hard-coded `outcome_based_selector=False` with inspection of the preserved historical source, matching the actual audit logic.
4. Restored the historical-RF versus later-model comparison output omitted during earlier public packaging.
5. Corrected stale Figure 4 panel-count metadata from four to two. No plotted value or layout changed.
6. Changed the default data path from an absent private-data directory to the released dataset, and updated schema, licensing, manifests and tests to admit reviewed data while still excluding figures, manuscripts and model binaries.

## Not established by these tests

- The full original nested model search, SHAP/permutation jobs and 48,000-candidate predictions were not all retrained from scratch during this release audit. Their retained numerical outputs are explicitly identified as reference data, not newly recomputed model results.
- The paper's historical RF and the later supplementary XGB/GPR analyses are distinct workflows. Their predictions must not be substituted for one another.
- Trial design recipes, fabrication logs and raw tensile-test records are not deposited. The authors confirm the independent post-design trial role; the available historical outcome-selection routine and exact ID/score overlap remain disclosed. Independent experimental chronology is not established.
- No tests claim to establish production-batch independence, causal microstructural mechanisms, or a statistical guarantee of prospective improvement.
- Historical preprocessing is reproduced faithfully, including fitting its scaler before the reported RF error-scale calculation. Reproduction is not an endorsement that this historical procedure is equivalent to a fully nested or batch-independent validation design.

See `PROVENANCE.md`, `CODE_PROVENANCE.json`, and `data/split_provenance.json` before interpreting the scope of a successful run.
