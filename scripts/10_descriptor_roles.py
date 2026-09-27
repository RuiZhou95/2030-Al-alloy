#!/usr/bin/env python3
"""Assign evidence roles after the systematic descriptor screen.

The full 118-descriptor library is retained as a sensitivity audit.  Only
literature-grounded raw, aggregate and phase-stoichiometric descriptors are
eligible for material interpretation in the main text.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


from release_paths import PAPER_OUTPUT as OUT
DATA = OUT / "data_materials_v2"
AUDIT = OUT / "audit"
GRADES = ["2024", "5083", "6082", "7075"]
PRIMARY = [
    "wt_Mg",
    "phasecap_S_Al2CuMg",
    "phasecap_beta_Mg2Si",
    "phaseimbalance_beta",
    "phasecap_eta_MgZn2",
    "phaseimbalance_eta",
    "dispersoid_MnCrZr_wt",
    "impurity_FeSi_wt",
    "eta_total_ZnMgCu_wt",
    "total_solute_wt",
]


def eta2_by_grade(values: pd.Series, grade: pd.Series) -> float:
    x = values.to_numpy(float)
    grand = float(np.mean(x))
    total = float(np.sum((x - grand) ** 2))
    if total <= 1e-15:
        return 0.0
    between = 0.0
    for g in GRADES:
        z = x[grade.to_numpy(str) == g]
        between += len(z) * (float(np.mean(z)) - grand) ** 2
    return between / total


def main() -> None:
    matrix = pd.read_csv(DATA / "descriptor_candidate_matrix_284.csv", encoding="utf-8-sig")
    metadata = pd.read_csv(DATA / "descriptor_library.csv", encoding="utf-8-sig")
    stats = pd.read_csv(DATA / "descriptor_stability_all.csv", encoding="utf-8-sig")
    matrix["grade"] = matrix.grade.astype(str)
    stats["grade"] = stats.grade.astype(str)
    assert len(matrix) == 284
    assert set(PRIMARY).issubset(matrix.columns)

    eta_rows = []
    for name in metadata.name:
        eta_rows.append({
            "descriptor": name,
            "grade_eta2": eta2_by_grade(matrix[name], matrix.grade),
            "family": metadata.set_index("name").loc[name, "family"],
            "label": metadata.set_index("name").loc[name, "label"],
        })
    eta = pd.DataFrame(eta_rows).sort_values("grade_eta2", ascending=False)
    eta.to_csv(DATA / "descriptor_grade_separation.csv", index=False, encoding="utf-8-sig")

    med = matrix.groupby("grade")[PRIMARY].median().reindex(GRADES)
    scaled = med.copy()
    for col in PRIMARY:
        lo, hi = float(med[col].min()), float(med[col].max())
        scaled[col] = 0.0 if hi <= lo else (med[col] - lo) / (hi - lo)

    rows = []
    eta_map = eta.set_index("descriptor").grade_eta2
    meta_map = metadata.set_index("name")
    for grade in GRADES:
        for name in PRIMARY:
            q = stats[(stats.grade == grade) & (stats.descriptor == name)]
            q = q.sort_values("stability_score", ascending=False)
            r = q.iloc[0]
            rows.append({
                "grade": grade,
                "descriptor": name,
                "label": meta_map.loc[name, "label"],
                "family": meta_map.loc[name, "family"],
                "median_raw": float(med.loc[grade, name]),
                "branch_scaled_0_1": float(scaled.loc[grade, name]),
                "grade_eta2": float(eta_map[name]),
                "best_target": r.target,
                "best_target_short": r.target_short,
                "rho": float(r.rho) if np.isfinite(r.rho) else np.nan,
                "sign_stability": float(r.sign_stability)
                if np.isfinite(r.sign_stability) else np.nan,
                "within_grade_score": float(r.stability_score),
                "within_grade_pass": bool(r.passes_threshold),
                "branch_coordinate": bool(eta_map[name] >= 0.50),
            })
    panel = pd.DataFrame(rows)
    panel.to_csv(DATA / "descriptor_primary_panel.csv", index=False, encoding="utf-8-sig")

    role = (
        panel.groupby(["descriptor", "label", "family", "grade_eta2"], as_index=False)
        .agg(n_within_grade_pass=("within_grade_pass", "sum"),
             max_within_grade_score=("within_grade_score", "max"),
             branch_coordinate=("branch_coordinate", "max"))
        .sort_values(["branch_coordinate", "n_within_grade_pass",
                      "max_within_grade_score", "grade_eta2"], ascending=False)
    )
    role["main_text_role"] = np.select(
        [
            role.branch_coordinate & (role.n_within_grade_pass > 0),
            role.branch_coordinate,
            role.n_within_grade_pass > 0,
        ],
        [
            "branch coordinate plus within-grade association",
            "branch coordinate",
            "within-grade association",
        ],
        default="context/control descriptor",
    )
    role.to_csv(DATA / "descriptor_role_assignment.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# Descriptor evidence-role assignment",
        "",
        "The 118-candidate systematic library is a completeness and sensitivity audit.",
        "Exploratory pair sums, products and balances are not promoted to material",
        "mechanisms from correlation alone. Main-text interpretation is restricted to",
        "predeclared raw, aggregate and literature-grounded phase-stoichiometric terms.",
        "",
        "A branch coordinate has grade eta-squared >= 0.50. A within-grade association",
        "retains the original |rho| >= 0.25 and bootstrap sign-stability >= 0.80 rule.",
        "",
        "## Primary descriptors",
    ]
    for r in role.itertuples():
        lines.append(
            f"- {r.label}: grade eta2={r.grade_eta2:.3f}; "
            f"within-grade passes={int(r.n_within_grade_pass)}; "
            f"role={r.main_text_role}."
        )
    lines += [
        "",
        "## Reporting rule",
        "Between-grade separation and within-grade response are reported separately.",
        "A high grade-separation score is not described as a within-grade mechanism.",
        "A stable exploratory correlation is not described as causal without a",
        "literature-grounded physical role and process-consistent evidence.",
    ]
    (AUDIT / "descriptor_role_assignment.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    manifest = {
        "systematic_library_size": int(len(metadata)),
        "primary_physics_descriptors": int(len(PRIMARY)),
        "trial_samples_read": 0,
        "frozen_predictions_changed": False,
    }
    (DATA / "descriptor_role_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(role.to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
