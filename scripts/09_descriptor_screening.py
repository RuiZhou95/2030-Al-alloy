#!/usr/bin/env python3
"""Build and screen a systematic composition-descriptor library.

Only the 284 modelling samples are read.  The four post-model trial samples are
not opened by this script.  Screening is descriptive and branch-resolved; it
does not refit or alter the frozen trial-prediction model.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


from release_paths import PRIVATE_ROOT as AL, PAPER_OUTPUT as OUT, SCIENTIFIC_OUTPUT as OLD
DATA = OUT / "data_materials_v2"
AUDIT = OUT / "audit"
GRADES = ["2024", "5083", "6082", "7075"]
TARGETS = {
    "yield_strength_mpa": "YS",
    "delta_strength_mpa": "DeltaSigma",
    "elongation_pct": "EL",
}
ATOMIC_WEIGHT = {
    "Al": 26.9815385, "Ti": 47.867, "Fe": 55.845, "Ni": 58.6934,
    "Cr": 51.9961, "Mn": 54.938044, "Si": 28.085, "Cu": 63.546,
    "Zn": 65.38, "Mg": 24.305, "V": 50.9415, "Zr": 91.224,
}
PAIR_ELEMENTS = ["Cu", "Mg", "Si", "Zn", "Mn", "Cr", "Zr", "Fe"]
N_BOOT = 1000
SEED = 20260922


@dataclass(frozen=True)
class Descriptor:
    name: str
    label: str
    family: str
    formula: str
    physical_role: str


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 4 or np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return np.nan
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    return float(np.corrcoef(rx, ry)[0, 1])


def add_candidate(store: dict[str, np.ndarray], meta: list[Descriptor],
                  name: str, values, label: str, family: str,
                  formula: str, role: str) -> None:
    arr = np.asarray(values, dtype=float)
    if name in store:
        raise ValueError(f"duplicate descriptor: {name}")
    if not np.isfinite(arr).all():
        raise ValueError(f"non-finite descriptor: {name}")
    store[name] = arr
    meta.append(Descriptor(name, label, family, formula, role))


def build_library(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    store: dict[str, np.ndarray] = {}
    meta: list[Descriptor] = []
    mol: dict[str, np.ndarray] = {}

    for el, aw in ATOMIC_WEIGHT.items():
        col = f"{el}_wt_pct"
        if col not in raw:
            continue
        wt = raw[col].fillna(0).to_numpy(float)
        mol[el] = 10.0 * wt / aw
        if np.unique(wt).size > 1:
            add_candidate(store, meta, f"wt_{el}", wt, f"{el} (wt.%)",
                          "raw composition", col,
                          "Measured elemental mass fraction retained in the full model.")
            add_candidate(store, meta, f"mol_{el}", mol[el], f"n({el})",
                          "molar composition", f"10*{el}_wt_pct/M_{el}",
                          "Elemental amount per kilogram of alloy.")

    non_al = [el for el in ATOMIC_WEIGHT if el != "Al" and f"{el}_wt_pct" in raw]
    total_solute = sum(raw[f"{el}_wt_pct"].fillna(0).to_numpy(float) for el in non_al)
    add_candidate(store, meta, "total_solute_wt", total_solute, "Total solute",
                  "aggregate load", "sum(non-Al wt.%)",
                  "Total measured non-aluminium solute load.")
    add_candidate(store, meta, "dispersoid_MnCrZr_wt",
                  raw["Mn_wt_pct"].to_numpy(float) + raw["Cr_wt_pct"].to_numpy(float)
                  + raw["Zr_wt_pct"].to_numpy(float),
                  "Mn+Cr+Zr", "aggregate load", "Mn+Cr+Zr (wt.%)",
                  "Elements associated with dispersoid-forming additions.")
    add_candidate(store, meta, "impurity_FeSi_wt",
                  raw["Fe_wt_pct"].to_numpy(float) + raw["Si_wt_pct"].to_numpy(float),
                  "Fe+Si", "aggregate load", "Fe+Si (wt.%)",
                  "Combined Fe-Si load relevant to constituent particles.")
    add_candidate(store, meta, "eta_total_ZnMgCu_wt",
                  raw["Zn_wt_pct"].to_numpy(float) + raw["Mg_wt_pct"].to_numpy(float)
                  + raw["Cu_wt_pct"].to_numpy(float),
                  "Zn+Mg+Cu", "aggregate load", "Zn+Mg+Cu (wt.%)",
                  "Total major solute load of the 7xxx branch.")

    for i, a in enumerate(PAIR_ELEMENTS):
        for b in PAIR_ELEMENTS[i + 1:]:
            na, nb = mol[a], mol[b]
            denom = na + nb
            balance = np.divide(np.minimum(na, nb), denom,
                                out=np.zeros_like(denom), where=denom > 0)
            add_candidate(store, meta, f"pair_sum_{a}_{b}", denom,
                          f"n({a})+n({b})", "systematic pair sum",
                          f"n_{a}+n_{b}",
                          "Systematically generated pairwise molar load.")
            add_candidate(store, meta, f"pair_product_{a}_{b}", na * nb,
                          f"n({a})x n({b})", "systematic pair product",
                          f"n_{a}*n_{b}",
                          "Systematically generated pairwise co-presence term.")
            add_candidate(store, meta, f"pair_balance_{a}_{b}", balance,
                          f"balance({a},{b})", "systematic pair balance",
                          f"min(n_{a},n_{b})/(n_{a}+n_{b})",
                          "Dimensionless pair-balance coordinate.")

    ncu, nmg, nsi, nzn = mol["Cu"], mol["Mg"], mol["Si"], mol["Zn"]
    s_cap = np.minimum(ncu, nmg)
    beta_cap = np.minimum(nmg / 2.0, nsi)
    eta_cap = np.minimum(nmg, nzn / 2.0)
    add_candidate(store, meta, "phasecap_S_Al2CuMg", s_cap, "S capacity",
                  "phase-stoichiometric capacity", "min(n_Cu,n_Mg)",
                  "Limiting-reagent coordinate for Al2CuMg-type stoichiometry.")
    add_candidate(store, meta, "phaseimbalance_S", np.abs(ncu - nmg), "S imbalance",
                  "phase-stoichiometric excess", "|n_Cu-n_Mg|",
                  "Departure from a 1:1 Cu:Mg molar balance.")
    add_candidate(store, meta, "phasecap_beta_Mg2Si", beta_cap, "Mg2Si capacity",
                  "phase-stoichiometric capacity", "min(n_Mg/2,n_Si)",
                  "Limiting-reagent coordinate for Mg2Si-type stoichiometry.")
    add_candidate(store, meta, "phaseimbalance_beta", np.abs(nmg / 2.0 - nsi),
                  "Mg2Si imbalance", "phase-stoichiometric excess",
                  "|n_Mg/2-n_Si|", "Departure from Mg2Si molar balance.")
    add_candidate(store, meta, "phasecap_eta_MgZn2", eta_cap, "MgZn2 capacity",
                  "phase-stoichiometric capacity", "min(n_Mg,n_Zn/2)",
                  "Limiting-reagent coordinate for MgZn2-type stoichiometry.")
    add_candidate(store, meta, "phaseimbalance_eta", np.abs(nmg - nzn / 2.0),
                  "MgZn2 imbalance", "phase-stoichiometric excess",
                  "|n_Mg-n_Zn/2|", "Departure from MgZn2 molar balance.")

    identifiers = raw[["sample_id", "alloy_or_coating_name"]].rename(
        columns={"alloy_or_coating_name": "grade"}
    ).reset_index(drop=True)
    matrix = pd.concat([identifiers, pd.DataFrame(store)], axis=1)
    metadata = pd.DataFrame([d.__dict__ for d in meta])
    return matrix, metadata


def bootstrap_stability(x: np.ndarray, y: np.ndarray, rho: float,
                        rng: np.random.Generator) -> tuple[float, float, float]:
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    vals = np.empty(N_BOOT, dtype=float)
    n = len(x)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        xs, ys = rx[idx], ry[idx]
        if np.std(xs) <= 1e-12 or np.std(ys) <= 1e-12:
            vals[i] = np.nan
        else:
            vals[i] = np.corrcoef(xs, ys)[0, 1]
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return np.nan, np.nan, np.nan
    sign = 1.0 if rho >= 0 else -1.0
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)),
            float(np.mean(np.sign(vals) == sign)))


def screen(matrix: pd.DataFrame, metadata: pd.DataFrame,
           raw: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    base = raw[["sample_id", "yield_strength_mpa", "ultimate_tensile_strength_mpa",
                "elongation_pct"]].copy()
    base["delta_strength_mpa"] = (
        base["ultimate_tensile_strength_mpa"] - base["yield_strength_mpa"]
    )
    merged = matrix.merge(base, on="sample_id", validate="one_to_one")
    merged = merged.merge(groups[["sample_id", "condition_group"]],
                          on="sample_id", validate="one_to_one")
    feature_names = metadata["name"].tolist()
    grouped = merged.groupby(["grade", "condition_group"], as_index=False)[
        feature_names + list(TARGETS)
    ].mean()
    rng = np.random.default_rng(SEED)
    rows = []
    for grade in GRADES:
        g = grouped[grouped.grade == grade].reset_index(drop=True)
        for name in feature_names:
            x = g[name].to_numpy(float)
            n_unique = int(np.unique(np.round(x, 12)).size)
            if n_unique < 5 or np.std(x) <= 1e-12:
                for target, short in TARGETS.items():
                    rows.append({
                        "grade": grade, "descriptor": name, "target": target,
                        "target_short": short, "n_condition_groups": len(g),
                        "n_unique": n_unique, "rho": np.nan, "ci_low": np.nan,
                        "ci_high": np.nan, "sign_stability": np.nan,
                        "stability_score": 0.0, "passes_threshold": False,
                    })
                continue
            for target, short in TARGETS.items():
                y = g[target].to_numpy(float)
                rho = safe_corr(x, y)
                if not np.isfinite(rho) or abs(rho) < 0.15:
                    lo = hi = stab = np.nan
                else:
                    lo, hi, stab = bootstrap_stability(x, y, rho, rng)
                score = abs(rho) * (stab if np.isfinite(stab) else 0.0)
                passes = bool(abs(rho) >= 0.25 and np.isfinite(stab) and stab >= 0.80)
                rows.append({
                    "grade": grade, "descriptor": name, "target": target,
                    "target_short": short, "n_condition_groups": len(g),
                    "n_unique": n_unique, "rho": rho, "ci_low": lo,
                    "ci_high": hi, "sign_stability": stab,
                    "stability_score": score, "passes_threshold": passes,
                })
    return pd.DataFrame(rows)


def select_nonredundant(matrix: pd.DataFrame, stats: pd.DataFrame,
                        metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for grade in GRADES:
        s = stats[stats.grade == grade].copy()
        idx = s.groupby("descriptor").stability_score.idxmax()
        best = s.loc[idx].sort_values(
            ["passes_threshold", "stability_score"], ascending=[False, False]
        )
        values = matrix[matrix.grade == grade].set_index("sample_id")
        kept: list[str] = []
        for r in best.itertuples():
            if r.n_unique < 5:
                continue
            if any(abs(values[r.descriptor].corr(values[k], method="spearman")) >= 0.95
                   for k in kept):
                continue
            kept.append(r.descriptor)
            rows.append({
                "grade": grade, "rank": len(kept), "descriptor": r.descriptor,
                "best_target": r.target, "target_short": r.target_short,
                "rho": r.rho, "sign_stability": r.sign_stability,
                "stability_score": r.stability_score,
                "passes_threshold": bool(r.passes_threshold),
            })
            if len(kept) == 10:
                break
    selected = pd.DataFrame(rows).merge(
        metadata, left_on="descriptor", right_on="name", how="left", validate="many_to_one"
    ).drop(columns="name")
    global_rank = (
        selected.groupby(["descriptor", "label", "family"], as_index=False)
        .agg(n_grades=("grade", "nunique"),
             mean_score=("stability_score", "mean"),
             max_score=("stability_score", "max"),
             n_threshold=("passes_threshold", "sum"))
        .sort_values(["n_threshold", "n_grades", "max_score", "mean_score"],
                     ascending=False)
    )
    display = global_rank.head(10).copy()
    return selected, display


def write_audit(metadata: pd.DataFrame, stats: pd.DataFrame,
                selected: pd.DataFrame, display: pd.DataFrame,
                n_groups: dict[str, int]) -> None:
    lines = [
        "# Composition descriptor library and stability audit",
        "",
        "- Input: 284 modelling samples only.",
        "- Post-model trial samples read: 0.",
        f"- Candidate descriptors generated: {len(metadata)}.",
        f"- Bootstrap resamples per auditable grade-descriptor-target relation: {N_BOOT}.",
        "- Screening unit: unique continuous condition group within grade.",
        "- Pass rule: |Spearman rho| >= 0.25 and bootstrap sign stability >= 0.80.",
        "- Redundancy rule: within-grade |Spearman rho| >= 0.95 keeps the higher-ranked descriptor.",
        "- Purpose: interpretable coordinate screening; frozen trial predictions are unchanged.",
        "",
        "## Condition groups",
    ]
    lines += [f"- {g}: {n_groups[g]}" for g in GRADES]
    lines += ["", "## Library families"]
    for family, n in metadata.groupby("family").size().sort_values(ascending=False).items():
        lines.append(f"- {family}: {n}")
    lines += ["", "## Non-redundant top descriptors by grade"]
    for grade in GRADES:
        lines.append(f"",)
        lines.append(f"### {grade}")
        sub = selected[selected.grade == grade]
        for r in sub.itertuples():
            mark = "pass" if r.passes_threshold else "ranked"
            lines.append(
                f"- {r.rank}. {r.label} ({r.family}); {r.target_short}, "
                f"rho={r.rho:.3f}, sign stability={r.sign_stability:.3f} [{mark}]"
            )
    lines += ["", "## Main-figure display set"]
    for r in display.itertuples():
        lines.append(
            f"- {r.label}: grades={r.n_grades}, threshold passes={r.n_threshold}, "
            f"max score={r.max_score:.3f}"
        )
    lines += [
        "", "## Interpretation boundary",
        "The library is systematic within the declared generation rules, not a claim that",
        "all possible thermodynamic or microstructural state variables were observed.",
        "Raw elemental fields remain in the full modelling feature space.",
    ]
    (AUDIT / "descriptor_stability_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(AL / "data/01_processed_data.csv", encoding="utf-8-sig")
    groups = pd.read_csv(OLD / "data/condition_groups.csv", encoding="utf-8-sig")
    groups["alloy_or_coating_name"] = groups["alloy_or_coating_name"].astype(str)
    assert len(raw) == 284 and raw.sample_id.is_unique
    assert len(groups) == 284 and groups.sample_id.is_unique
    assert set(raw.alloy_or_coating_name.astype(str)) == set(GRADES)
    raw["alloy_or_coating_name"] = raw.alloy_or_coating_name.astype(str)
    matrix, metadata = build_library(raw)
    stats = screen(matrix, metadata, raw, groups)
    selected, display = select_nonredundant(matrix, stats, metadata)
    n_groups = groups.groupby("alloy_or_coating_name").condition_group.nunique().astype(int).to_dict()
    matrix.to_csv(DATA / "descriptor_candidate_matrix_284.csv",
                  index=False, encoding="utf-8-sig")
    metadata.to_csv(DATA / "descriptor_library.csv", index=False, encoding="utf-8-sig")
    stats.to_csv(DATA / "descriptor_stability_all.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(DATA / "descriptor_selected_by_grade.csv",
                    index=False, encoding="utf-8-sig")
    display.to_csv(DATA / "descriptor_display_set.csv",
                   index=False, encoding="utf-8-sig")
    write_audit(metadata, stats, selected, display, n_groups)
    manifest = {
        "input_modelling_samples": len(raw),
        "post_model_trial_samples_read": 0,
        "candidate_descriptors": len(metadata),
        "bootstrap_resamples": N_BOOT,
        "grades": GRADES,
        "condition_groups": n_groups,
        "selection_threshold": {"abs_spearman_rho": 0.25, "sign_stability": 0.80},
        "frozen_trial_predictions_changed": False,
    }
    (DATA / "descriptor_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
