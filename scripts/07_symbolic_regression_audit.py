#!/usr/bin/env python3
"""Stability and boundary audit for the original symbolic-regression formulas."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


from release_paths import PRIVATE_ROOT as AL, SCIENTIFIC_OUTPUT as REV
DATA = REV / "data"
AUDIT = REV / "audit"
TARGETS = ["yield_strength_mpa", "ultimate_tensile_strength_mpa"]
RNG = np.random.RandomState(42)


def structure(formula: str) -> str:
    return re.sub(r"(?<![A-Za-z_])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", "C", str(formula))


def variables(formula: str, features: list[str]) -> list[str]:
    found = sorted({int(x) for x in re.findall(r"x(\d+)", str(formula))})
    return [features[i] if i < len(features) else f"x{i}" for i in found]


def eval_formula(text: str, frame: pd.DataFrame) -> np.ndarray:
    env = {c: frame[c].to_numpy(float) for c in frame.columns}
    env.update({"square": np.square, "cube": lambda x: np.power(x, 3), "sqrt": np.sqrt})
    with np.errstate(all="ignore"):
        return np.asarray(eval(text, {"__builtins__": {}}, env), dtype=float)


def main():
    setting_rows, feature_rows, boundary_rows = [], [], []
    decisions = []
    for target in TARGETS:
        scan = pd.read_csv(AL / f"data/sr_{target}_scan_all.csv", encoding="utf-8-sig")
        eligible = scan[(scan["complexity"] >= 8) & (scan["complexity"] <= 16)].copy()
        selected = eligible.sort_values("score", ascending=False).groupby("setting", as_index=False).first()
        features = [x.strip() for x in (AL / f"data/sr_{target}_features.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
        selected["structure"] = selected["formula"].map(structure)
        selected["variables"] = selected["formula"].map(lambda x: ";".join(variables(x, features)))
        selected["target"] = target
        setting_rows.append(selected[["target", "setting", "complexity", "loss", "score", "formula", "structure", "variables"]])
        for feature in features:
            feature_rows.append({
                "target": target, "feature": feature,
                "selected_setting_frequency": float(selected["variables"].str.split(";").map(lambda xs: feature in xs).mean()),
                "n_settings": len(selected),
            })

        counts = selected["structure"].value_counts()
        max_frequency = float(counts.iloc[0] / len(selected))
        unique_fraction = float(selected["structure"].nunique() / len(selected))

        best = json.loads((AL / f"data/sr_{target}_best.json").read_text(encoding="utf-8"))
        xtr = pd.read_csv(AL / f"data/sr_{target}_X_tr.csv", header=None, names=features)
        xte = pd.read_csv(AL / f"data/sr_{target}_X_te.csv", header=None, names=features)
        observed = pd.concat([xtr, xte], ignore_index=True)
        low = observed.min(axis=0); high = observed.max(axis=0); width = high - low
        n = 100000
        expanded = pd.DataFrame({
            c: RNG.uniform(low[c] - 0.1 * width[c], high[c] + 0.1 * width[c], n)
            for c in observed.columns
        })
        obs_pred = eval_formula(best["formula_text"], observed)
        ext_pred = eval_formula(best["formula_text"], expanded)
        finite = np.isfinite(ext_pred)
        obs_span = float(np.nanmax(obs_pred) - np.nanmin(obs_pred))
        extreme = finite & (
            (ext_pred < np.nanmin(obs_pred) - 2 * obs_span)
            | (ext_pred > np.nanmax(obs_pred) + 2 * obs_span)
        )
        boundary_rows.append({
            "target": target, "formula": best["formula_text"],
            "n_settings": len(selected), "unique_structures": int(selected["structure"].nunique()),
            "most_common_structure_frequency": max_frequency,
            "unique_structure_fraction": unique_fraction,
            "expanded_samples": n,
            "nonfinite_fraction_expanded_10pct": float((~finite).mean()),
            "extreme_fraction_of_finite": float(extreme.sum() / max(finite.sum(), 1)),
            "observed_prediction_min": float(np.nanmin(obs_pred)),
            "observed_prediction_max": float(np.nanmax(obs_pred)),
            "group_aware_refit_validation_available": False,
        })
        decisions.append({
            "target": target,
            "decision": "supplementary_only",
            "reason": "No condition-group refit validation; formula structure is selected under a random split and contains unrestricted nonlinear operators.",
        })

    settings = pd.concat(setting_rows, ignore_index=True)
    features = pd.DataFrame(feature_rows)
    boundary = pd.DataFrame(boundary_rows)
    decision = pd.DataFrame(decisions)
    settings.to_csv(DATA / "symbolic_regression_setting_stability.csv", index=False, encoding="utf-8-sig")
    features.to_csv(DATA / "symbolic_regression_feature_frequency.csv", index=False, encoding="utf-8-sig")
    boundary.to_csv(DATA / "symbolic_regression_boundary_audit.csv", index=False, encoding="utf-8-sig")
    decision.to_csv(DATA / "symbolic_regression_decision.csv", index=False, encoding="utf-8-sig")

    lines = []
    for r in boundary.itertuples():
        lines.append(
            f"- {r.target}: {r.n_settings}组设置产生{r.unique_structures}种选定结构；"
            f"最常见结构频率{r.most_common_structure_frequency:.1%}；"
            f"观测域外扩10%时非有限值比例{r.nonfinite_fraction_expanded_10pct:.2%}，"
            f"极端外推比例{r.extreme_fraction_of_finite:.2%}。"
        )
    report = """# 符号回归正文去留审查

## 稳定性与边界

""" + "\n".join(lines) + """

## 决定

符号回归移至补充材料，不再作为正文核心贡献。决定的充分条件是：现有公式及其结构选择没有在重复条件分组的外层折中重新搜索和验证；原随机拆分测试分数不能证明跨条件稳定性。正文只保留其作为探索性压缩表达的说明，并明确标准化变量、适用范围和禁止外推。

该决定不依赖4个实验样本表现，也不删除原始扫描结果。
"""
    (AUDIT / "10_symbolic_regression_decision.md").write_text(report, encoding="utf-8")
    print(report)
    print(boundary.to_string(index=False))


if __name__ == "__main__":
    main()
