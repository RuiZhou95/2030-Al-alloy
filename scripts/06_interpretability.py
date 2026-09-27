#!/usr/bin/env python3
"""Conditional SHAP, held-out permutation importance and within-grade ALE."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


from release_paths import PRIVATE_ROOT as AL, SCIENTIFIC_OUTPUT as REV
DATA = REV / "data"
AUDIT = REV / "audit"
RNG = np.random.RandomState(42)


def core_module():
    path = Path(__file__).resolve().parent / "02_nested_validation.py"
    spec = importlib.util.spec_from_file_location("nested_core", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clean_names(names):
    return [str(x).replace("num__", "").replace("cat__", "") for x in names]


def xgb_fold_shap(core, df, spec, target, params):
    rows = []
    model_params = {k.replace("model__", ""): v for k, v in params.items()}
    cols = spec["numeric"] + spec["categorical"]
    for fold, (tr, te) in enumerate(core.outer_splits(df, "condition"), 1):
        pipe = Pipeline([
            ("prep", core.preprocessor(spec)),
            ("model", XGBRegressor(
                objective="reg:squarederror", tree_method="hist", random_state=42 + fold,
                n_jobs=4, verbosity=0, **model_params,
            )),
        ])
        pipe.fit(df.iloc[tr][cols], df.iloc[tr][target])
        xte = pipe.named_steps["prep"].transform(df.iloc[te][cols])
        names = clean_names(pipe.named_steps["prep"].get_feature_names_out())
        values = np.asarray(shap.TreeExplainer(pipe.named_steps["model"]).shap_values(xte))
        for local_i, global_i in enumerate(te):
            for j, name in enumerate(names):
                rows.append({
                    "sample_id": df.iloc[global_i]["sample_id"], "grade": df.iloc[global_i]["grade"],
                    "fold": fold, "target": target, "feature": name,
                    "shap_value": float(values[local_i, j]), "abs_shap": float(abs(values[local_i, j])),
                })
    return pd.DataFrame(rows)


def group_bootstrap_rank(long_df, group_map, value_col, n_boot=1000):
    out = []
    for target, sub in long_df.groupby("target"):
        wide = sub.pivot_table(index="sample_id", columns="feature", values=value_col, aggfunc="mean")
        wide["condition_group"] = wide.index.map(group_map)
        features = [c for c in wide.columns if c != "condition_group"]
        group_values = wide.groupby("condition_group")[features].mean()
        top_counts = pd.Series(0.0, index=features)
        ranks = {f: [] for f in features}
        for _ in range(n_boot):
            take = RNG.randint(0, len(group_values), len(group_values))
            imp = group_values.iloc[take].mean(axis=0).sort_values(ascending=False)
            top_counts.loc[imp.index[:5]] += 1
            rank = imp.rank(ascending=False, method="average")
            for f in features:
                ranks[f].append(float(rank[f]))
        mean_imp = group_values.mean(axis=0)
        for f in features:
            out.append({
                "target": target, "feature": f, "mean_importance": float(mean_imp[f]),
                "top5_frequency": float(top_counts[f] / n_boot),
                "mean_rank": float(np.mean(ranks[f])),
                "rank_ci_low": float(np.quantile(ranks[f], 0.025)),
                "rank_ci_high": float(np.quantile(ranks[f], 0.975)),
                "bootstrap_unit": "condition_group", "n_bootstrap": n_boot,
            })
    return pd.DataFrame(out)


def gpr_fold_permutation(core, df, spec, target):
    cols = spec["numeric"] + spec["categorical"]
    rows = []
    for fold, (tr, te) in enumerate(core.outer_splits(df, "condition"), 1):
        estimator = core.model_search("gpr", spec, core.inner_splits(df.iloc[tr].reset_index(drop=True), "condition"), 42 + fold)
        estimator.fit(df.iloc[tr][cols], df.iloc[tr][target])
        base_pred = np.asarray(estimator.predict(df.iloc[te][cols])).reshape(-1)
        base_rmse = float(np.sqrt(mean_squared_error(df.iloc[te][target], base_pred)))
        test = df.iloc[te][cols + ["grade"]].copy().reset_index(drop=True)
        for feature in cols:
            for repeat in range(10):
                perm = test[cols].copy()
                for _, idx in test.groupby("grade").groups.items():
                    vals = perm.loc[idx, feature].to_numpy(copy=True)
                    RNG.shuffle(vals)
                    perm.loc[idx, feature] = vals
                pred = np.asarray(estimator.predict(perm)).reshape(-1)
                rmse = float(np.sqrt(mean_squared_error(df.iloc[te][target], pred)))
                rows.append({
                    "fold": fold, "target": target, "feature": feature, "repeat": repeat,
                    "baseline_rmse": base_rmse, "permuted_rmse": rmse,
                    "importance_delta_rmse": rmse - base_rmse,
                })
    return pd.DataFrame(rows)


def ale_for_feature(model, data, cols, feature, grade, bins=8):
    sub = data[data["grade"] == grade].copy()
    vals = sub[feature].astype(float)
    if vals.nunique() < 5:
        return []
    edges = np.unique(np.quantile(vals, np.linspace(0, 1, bins + 1)))
    if len(edges) < 4:
        return []
    which = np.clip(np.digitize(vals, edges[1:-1], right=True), 0, len(edges) - 2)
    effects, counts = [], []
    for k in range(len(edges) - 1):
        mask = which == k
        n = int(mask.sum())
        counts.append(n)
        if n == 0:
            effects.append(0.0)
            continue
        lo = sub.loc[mask, cols].copy(); hi = lo.copy()
        lo[feature] = edges[k]; hi[feature] = edges[k + 1]
        diff = np.asarray(model.predict(hi)).reshape(-1) - np.asarray(model.predict(lo)).reshape(-1)
        effects.append(float(np.mean(diff)))
    accumulated = np.cumsum(effects)
    centered = accumulated - np.average(accumulated, weights=np.maximum(counts, 1))
    return [{
        "grade": grade, "feature": feature, "bin": k + 1,
        "bin_center": float((edges[k] + edges[k + 1]) / 2),
        "ale": float(centered[k]), "n_bin": int(counts[k]),
        "range_low": float(edges[k]), "range_high": float(edges[k + 1]),
    } for k in range(len(effects))]


def main():
    core = core_module()
    df, feature_sets = core.load_data()
    config = json.loads((DATA / "final_model_config.json").read_text(encoding="utf-8"))
    config_by_target = {x["target"]: x for x in config["models"]}
    group_map = pd.read_csv(DATA / "condition_groups.csv", encoding="utf-8-sig").set_index("sample_id")["condition_group"]

    shap_parts = []
    for target in ["yield_strength_mpa", "delta_strength_mpa"]:
        cfg = config_by_target[target]
        shap_parts.append(xgb_fold_shap(
            core, df, feature_sets[cfg["feature_set"]], target, cfg["best_params_full_284_group_cv"]
        ))
    shap_long = pd.concat(shap_parts, ignore_index=True)
    shap_long.to_csv(DATA / "conditional_shap_oof.csv", index=False, encoding="utf-8-sig")
    shap_summary = shap_long.groupby(["target", "grade", "feature"], as_index=False).agg(
        mean_abs_shap=("abs_shap", "mean"), mean_signed_shap=("shap_value", "mean"), n=("sample_id", "nunique")
    )
    shap_summary["rank_within_grade"] = shap_summary.groupby(["target", "grade"])["mean_abs_shap"].rank(ascending=False, method="min")
    shap_summary.to_csv(DATA / "conditional_shap_by_grade.csv", index=False, encoding="utf-8-sig")
    shap_stability = group_bootstrap_rank(shap_long, group_map, "abs_shap")
    shap_stability.to_csv(DATA / "conditional_shap_stability.csv", index=False, encoding="utf-8-sig")

    el_cfg = config_by_target["elongation_pct"]
    perm = gpr_fold_permutation(core, df, feature_sets[el_cfg["feature_set"]], "elongation_pct")
    perm.to_csv(DATA / "elongation_gpr_grouped_permutation.csv", index=False, encoding="utf-8-sig")
    perm_summary = perm.groupby(["target", "feature"], as_index=False).agg(
        mean_delta_rmse=("importance_delta_rmse", "mean"),
        sd_delta_rmse=("importance_delta_rmse", "std"),
        positive_fraction=("importance_delta_rmse", lambda x: float((x > 0).mean())),
    ).sort_values("mean_delta_rmse", ascending=False)
    perm_summary.to_csv(DATA / "elongation_gpr_permutation_summary.csv", index=False, encoding="utf-8-sig")

    # ALE uses frozen full-data models and remains within each grade's observed range.
    ale_rows = []
    for target in ["yield_strength_mpa", "delta_strength_mpa", "elongation_pct"]:
        cfg = config_by_target[target]
        model = joblib.load(REV / cfg["model_file"])
        cols = cfg["numeric_features"] + cfg["categorical_features"]
        if target in ["yield_strength_mpa", "delta_strength_mpa"]:
            top = shap_stability[shap_stability["target"] == target].sort_values(
                ["top5_frequency", "mean_importance"], ascending=False
            )["feature"].tolist()
        else:
            top = perm_summary[perm_summary["mean_delta_rmse"] > 0]["feature"].tolist()
        top_numeric = [x for x in top if x in cfg["numeric_features"]][:4]
        for feature in top_numeric:
            for grade in sorted(df["grade"].unique()):
                part = ale_for_feature(model, df, cols, feature, grade)
                for item in part:
                    item["target"] = target
                    ale_rows.append(item)
    ale = pd.DataFrame(ale_rows)
    ale.to_csv(DATA / "conditional_ale_by_grade.csv", index=False, encoding="utf-8-sig")

    top_lines = []
    for target in ["yield_strength_mpa", "delta_strength_mpa"]:
        top = shap_stability[shap_stability["target"] == target].sort_values(
            ["top5_frequency", "mean_importance"], ascending=False
        ).head(5)
        top_lines.append(f"- {target}: " + "；".join(
            f"{r.feature}（Top5频率{r.top5_frequency:.0%}）" for r in top.itertuples()
        ))
    top_el = perm_summary.head(5)
    top_lines.append("- elongation_pct: " + "；".join(
        f"{r.feature}（ΔRMSE={r.mean_delta_rmse:.3f}）" for r in top_el.itertuples()
    ))
    report = """# 条件解释与稳定性审计

## 方法

- YS与Δσ：在重复条件分组的每个外层测试折上计算XGBoost SHAP，随后按牌号汇总；排序稳定性以条件组为单位bootstrap 1000次。
- EL：在每个外层测试折内按牌号置换单一特征，记录GPR的组外RMSE增量。
- ALE：只在每个牌号自身的观测范围内计算；牌号内唯一值少于5的特征不画响应曲线。

## 稳定特征

""" + "\n".join(top_lines) + """

## 解释边界

- 最终强度模型使用工艺/路线特征，说明当前数据的强度预测主要依赖已见工艺状态，而非跨牌号成分外推。
- 路线类别与牌号高度混杂；其SHAP贡献不能解释为独立工艺因果效应。
- ALE与SHAP均是冻结模型在现有支撑域内的响应，不替代析出相、晶粒、织构或缺陷的实验机制证据。
"""
    (AUDIT / "09_interpretability_findings.md").write_text(report, encoding="utf-8")
    print(report)
    print("ALE rows", len(ale), "SHAP rows", len(shap_long), "permutation rows", len(perm))


if __name__ == "__main__":
    main()
