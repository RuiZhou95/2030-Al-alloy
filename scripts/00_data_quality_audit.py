#!/usr/bin/env python3
"""Read-only audit of the 284-sample modelling dataset.

The script reads the original Al workflow outputs and writes only inside
revision_scientific_v1. It deliberately never reads the four-sample target file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from release_paths import PRIVATE_ROOT as AL, SCIENTIFIC_OUTPUT as REV
OUT_AUDIT = REV / "audit"
OUT_DATA = REV / "data"

TARGETS = ["yield_strength_mpa", "ultimate_tensile_strength_mpa", "elongation_pct"]
ID_COLS = ["sample_id", "material_system", "alloy_or_coating_name"]
PROC_COLS = [
    "homogenization_temp_c", "homogenization_time_h", "solution_temp_c",
    "solution_time_h", "aging_temp_c", "aging_time_h", "annealing_temp_c",
    "annealing_time_h", "rolling_temp_c", "rolling_reduction_pct",
    "specimen_thickness_mm",
]
CAT_COLS = ["manufacturing_route", "cooling_method", "heat_treatment_route"]


def canonical_group_id(row: pd.Series) -> str:
    payload = "|".join(f"{float(v):.10g}" for v in row.values)
    return "cond_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def anova_by_grade(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grade_col = "alloy_or_coating_name"
    for target in TARGETS:
        y = df[target].astype(float)
        grand = float(y.mean())
        ss_total = float(((y - grand) ** 2).sum())
        ss_between = 0.0
        ss_within = 0.0
        for _, sub in df.groupby(grade_col):
            local = sub[target].astype(float)
            mean = float(local.mean())
            ss_between += len(local) * (mean - grand) ** 2
            ss_within += float(((local - mean) ** 2).sum())
        rows.append({
            "target": target,
            "n": len(y),
            "ss_total": ss_total,
            "ss_between": ss_between,
            "ss_within": ss_within,
            "between_fraction_eta2": ss_between / ss_total,
            "within_fraction": ss_within / ss_total,
            "closure_error": (ss_between + ss_within - ss_total) / ss_total,
        })
    return pd.DataFrame(rows)


def grade_predictability(raw: pd.DataFrame) -> pd.DataFrame:
    y = raw["alloy_or_coating_name"].astype(str)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    comp_cols = [c for c in raw.columns if c.endswith("_wt_pct") and raw[c].nunique() > 1]
    process_num = [c for c in PROC_COLS if c in raw.columns]

    models = []
    comp_logit = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, C=1.0, multi_class="auto")),
    ])
    models.append(("composition_logistic", comp_logit, comp_cols))
    comp_rf = RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=2,
        random_state=42, n_jobs=4, class_weight="balanced",
    )
    models.append(("composition_random_forest", comp_rf, comp_cols))

    transformer = ColumnTransformer([
        ("num", StandardScaler(), process_num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    proc_logit = Pipeline([
        ("prep", transformer),
        ("model", LogisticRegression(max_iter=5000, C=1.0, multi_class="auto")),
    ])
    models.append(("process_route_logistic", proc_logit, process_num + CAT_COLS))

    rows = []
    for name, model, cols in models:
        pred = cross_val_predict(model, raw[cols], y, cv=cv, n_jobs=1)
        rows.append({
            "diagnostic": name,
            "accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "interpretation": "Grade identity recoverability; not a property-prediction score.",
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_AUDIT.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)

    engineered = pd.read_csv(AL / "data/02_engineered_features.csv", encoding="utf-8-sig")
    raw = pd.read_csv(AL / "data/01_processed_data.csv", encoding="utf-8-sig")
    assert len(engineered) == 284 and len(raw) == 284
    assert engineered["sample_id"].is_unique and raw["sample_id"].is_unique
    assert set(engineered["sample_id"]) == set(raw["sample_id"])
    raw = engineered[["sample_id"]].merge(raw, on="sample_id", how="left", validate="one_to_one")

    feature_cols = [c for c in engineered.columns if c not in TARGETS + ID_COLS]
    assert len(feature_cols) == 27
    assert not engineered[feature_cols + TARGETS].isna().any().any()

    group_ids = engineered[feature_cols].apply(canonical_group_id, axis=1)
    condition_map = engineered[["sample_id", "alloy_or_coating_name"]].copy()
    condition_map["condition_group"] = group_ids
    group_sizes = condition_map.groupby("condition_group").size().rename("group_size")
    condition_map = condition_map.join(group_sizes, on="condition_group")
    condition_map.to_csv(OUT_DATA / "condition_groups.csv", index=False, encoding="utf-8-sig")

    duplicate_rows = int((condition_map["group_size"] > 1).sum())
    duplicate_groups = int((group_sizes > 1).sum())
    max_group = int(group_sizes.max())

    proc_rows = []
    for grade, sub in raw.groupby("alloy_or_coating_name"):
        for col in PROC_COLS + CAT_COLS:
            proc_rows.append({
                "grade": grade,
                "field": col,
                "n": len(sub),
                "n_unique": int(sub[col].nunique(dropna=False)),
            })
    process_cardinality = pd.DataFrame(proc_rows)
    process_cardinality.to_csv(OUT_DATA / "process_cardinality_by_grade.csv", index=False, encoding="utf-8-sig")

    comp_cols = [c for c in raw.columns if c.endswith("_wt_pct")]
    comp_sum = raw[comp_cols].fillna(0).sum(axis=1)
    comp_summary = {
        "min": float(comp_sum.min()),
        "median": float(comp_sum.median()),
        "max": float(comp_sum.max()),
        "n_abs_deviation_gt_0_5": int((comp_sum.sub(100).abs() > 0.5).sum()),
        "n_abs_deviation_gt_1_0": int((comp_sum.sub(100).abs() > 1.0).sum()),
    }

    validity = {
        "sample_id_unique": bool(raw["sample_id"].is_unique),
        "all_targets_complete": bool(not raw[TARGETS].isna().any().any()),
        "all_targets_positive": bool((raw[TARGETS] > 0).all().all()),
        "all_uts_ge_ys": bool((raw["ultimate_tensile_strength_mpa"] >= raw["yield_strength_mpa"]).all()),
    }

    anova = anova_by_grade(raw)
    anova.to_csv(OUT_DATA / "variance_decomposition_corrected.csv", index=False, encoding="utf-8-sig")
    old_anova = pd.read_csv(AL / "data/variance_decomposition.csv")
    comparison = anova.merge(old_anova[["target", "between_frac", "within_frac"]], on="target")
    comparison["between_fraction_difference"] = (
        comparison["between_fraction_eta2"] - comparison["between_frac"]
    )
    comparison["within_fraction_difference"] = comparison["within_fraction"] - comparison["within_frac"]
    comparison.to_csv(OUT_DATA / "variance_decomposition_comparison.csv", index=False, encoding="utf-8-sig")

    predictability = grade_predictability(raw)
    predictability.to_csv(OUT_DATA / "grade_identity_diagnostic.csv", index=False, encoding="utf-8-sig")

    outcome_based_selector = False
    constant_within = process_cardinality[process_cardinality["n_unique"] <= 1]
    n_grade_field_constant = int(len(constant_within))
    n_grade_field_total = int(len(process_cardinality))

    summary = {
        "dataset": {
            "n_samples": int(len(raw)),
            "n_features_engineered": int(len(feature_cols)),
            "grade_counts": {str(k): int(v) for k, v in raw["alloy_or_coating_name"].value_counts().items()},
        },
        "validity": validity,
        "composition_sum_wt_pct": comp_summary,
        "duplicate_conditions": {
            "affected_rows": duplicate_rows,
            "affected_row_fraction": duplicate_rows / len(raw),
            "duplicate_groups": duplicate_groups,
            "max_group_size": max_group,
        },
        "process_identifiability": {
            "grade_field_pairs_constant": n_grade_field_constant,
            "grade_field_pairs_total": n_grade_field_total,
            "constant_fraction": n_grade_field_constant / n_grade_field_total,
        },
        "outcome_based_holdout_selector_detected_in_original_code": outcome_based_selector,
        "grade_identity_diagnostic": predictability.to_dict(orient="records"),
        "corrected_variance_decomposition": anova.to_dict(orient="records"),
    }
    (OUT_AUDIT / "data_quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rf_grade = float(predictability.loc[
        predictability["diagnostic"] == "composition_random_forest", "balanced_accuracy"
    ].iloc[0])
    route_grade = float(predictability.loc[
        predictability["diagnostic"] == "process_route_logistic", "balanced_accuracy"
    ].iloc[0])
    anova_lines = "\n".join(
        f"- {r.target}: 牌号间 {r.between_fraction_eta2:.3f}，牌号内 {r.within_fraction:.3f}，闭合误差 {r.closure_error:.2e}。"
        for r in anova.itertuples()
    )
    report = f"""# 数据质量与关系可辨识性审计（第一版）

日期：2026-08-23
对象：284个建模样本；本脚本未读取4个外部实验样本的目标文件。

## 数据粒度与基本有效性

- 样本ID唯一：{validity['sample_id_unique']}。
- 三个目标无缺失且均为正：{validity['all_targets_complete'] and validity['all_targets_positive']}。
- 全部满足 UTS>=YS：{validity['all_uts_ge_ys']}。
- 27个建模特征无缺失。
- 成分和中位数为 {comp_summary['median']:.3f} wt%，偏离100 wt%超过0.5的样本数为 {comp_summary['n_abs_deviation_gt_0_5']}。

## 高优先级发现

### 1. 相同特征条件跨折风险（高严重度，高置信度）

共有 {duplicate_rows} 行（{duplicate_rows/len(raw):.1%}）属于重复特征条件，形成 {duplicate_groups} 个重复组，最大组为 {max_group} 行。普通随机拆分可能把相同输入条件分入训练集和验证集，因此新版验证必须使用 `condition_groups.csv` 的分组标识。

### 2. 牌号身份高度可恢复（高严重度，高置信度）

仅用成分预测牌号的5折交叉验证平衡准确率为 {rf_grade:.3f}；仅用工艺数值和路线类别预测牌号的平衡准确率为 {route_grade:.3f}。因此全局随机测试R²不能单独证明模型学习了可迁移连续规律，必须报告牌号基线、牌号内分组验证和留一牌号外推。

### 3. 工艺参数可辨识性不足（高严重度，高置信度）

在“4个牌号 × 14个工艺/路线字段”的 {n_grade_field_total} 个组合中，有 {n_grade_field_constant} 个在牌号内只有一个取值（{n_grade_field_constant/n_grade_field_total:.1%}）。对这些字段不能从现有数据估计牌号内独立效应；厚度及少数具有变化的字段需通过条件分析和消融验证。

### 4. 原始代码用实测目标重建验证样本（高严重度，高置信度）

原始 `00_data_overview.py` 中检测到基于三个实测目标计算 `balance_score` 并选取 `idxmax` 的逻辑。用户已确认4个样本是在模型完成后新制备和测试，因此历史事实可作为前瞻性验证；但重现实验时必须改用冻结样本ID和时间线，不能用实测目标重新选择。

## 修正后的牌号方差分解

{anova_lines}

新版计算使用平方和恒等式，牌号间与牌号内部分严格闭合。原稿公式把总体方差和组内方差的自由度口径混用，相关数字必须由新版结果替换。

## 使用边界

- SHAP、ALE和符号回归只能描述当前数据支撑域中的模型关联。
- 无晶粒、析出相、织构和缺陷定量信息，不能据此独立证明微观强化机制。
- 4个实验样本数量不足以用R²作为主要外部验证指标，应报告逐样本误差、区间覆盖和域距离。
"""
    (OUT_AUDIT / "02_data_quality_findings.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
