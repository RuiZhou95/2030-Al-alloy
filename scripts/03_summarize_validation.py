#!/usr/bin/env python3
"""Validate, summarize and freeze the nested-CV results before external evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


from release_paths import PRIVATE_ROOT as AL, SCIENTIFIC_OUTPUT as REV
DATA = REV / "data"
AUDIT = REV / "audit"
TAG = "nested_v1_full"
TARGETS = ["yield_strength_mpa", "delta_strength_mpa", "elongation_pct"]
MODEL_ORDER = {"elasticnet": 0, "pls": 1, "gpr": 2, "xgb": 3}
FEATURE_ORDER = {"composition": 0, "process": 1, "joint": 2}


def score(y, pred):
    return {
        "r2": float(r2_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
    }


def safe_csv(df: pd.DataFrame, path: Path):
    if path.exists():
        existing = pd.read_csv(path, encoding="utf-8-sig")
        pd.testing.assert_frame_equal(
            existing.reset_index(drop=True), df.reset_index(drop=True),
            check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-10,
        )
        return
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_text(text: str, path: Path):
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Existing file differs: {path}")
        return
    path.write_text(text, encoding="utf-8")


def choose_models(pooled: pd.DataFrame) -> pd.DataFrame:
    eligible = pooled[
        (pooled["scheme"] == "condition")
        & pooled["model"].isin(MODEL_ORDER)
        & pooled["target"].isin(TARGETS)
    ].copy()
    rows = []
    for target, sub in eligible.groupby("target"):
        best_rmse = float(sub["rmse"].min())
        tied = sub[sub["rmse"] <= best_rmse * 1.02].copy()
        tied["model_order"] = tied["model"].map(MODEL_ORDER)
        tied["feature_order"] = tied["feature_set"].map(FEATURE_ORDER)
        tied = tied.sort_values(["model_order", "feature_order", "rmse"])
        chosen = tied.iloc[0].to_dict()
        chosen["best_observed_rmse"] = best_rmse
        chosen["within_2pct_of_best"] = True
        chosen["selection_rule"] = "condition pooled RMSE; 2% tie then model simplicity and feature simplicity"
        rows.append(chosen)
    return pd.DataFrame(rows)


def bootstrap_selected(pred: pd.DataFrame, selected: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    group_map = groups.set_index("sample_id")["condition_group"]
    rng = np.random.RandomState(42)
    rows = []
    candidates = []
    for r in selected.itertuples():
        candidates.append((r.target, r.feature_set, r.model))
    for target in TARGETS:
        for baseline in ["global_mean", "grade_mean", "route_mean"]:
            candidates.append((target, "metadata_baseline", baseline))
    for target, feature_set, model in candidates:
        sub = pred[
            (pred["scheme"] == "condition")
            & (pred["target"] == target)
            & (pred["feature_set"] == feature_set)
            & (pred["model"] == model)
        ].copy()
        sub["condition_group"] = sub["sample_id"].map(group_map)
        codes, unique = pd.factorize(sub["condition_group"], sort=True)
        y = sub["observed"].to_numpy(float)
        p = sub["predicted"].to_numpy(float)
        n_groups = len(unique)
        stats = {"r2": [], "mae": [], "rmse": []}
        for _ in range(1000):
            group_counts = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups))
            w = group_counts[codes].astype(float)
            denom = w.sum()
            err = y - p
            mae = float(np.sum(w * np.abs(err)) / denom)
            rmse = float(np.sqrt(np.sum(w * err ** 2) / denom))
            ybar = float(np.sum(w * y) / denom)
            ss_tot = float(np.sum(w * (y - ybar) ** 2))
            r2 = float(1.0 - np.sum(w * err ** 2) / ss_tot) if ss_tot > 0 else np.nan
            stats["r2"].append(r2)
            stats["mae"].append(mae)
            stats["rmse"].append(rmse)
        point = score(sub["observed"], sub["predicted"])
        for metric in ["r2", "mae", "rmse"]:
            rows.append({
                "target": target, "feature_set": feature_set, "model": model,
                "metric": metric, "estimate": point[metric],
                "ci_low": float(np.quantile(stats[metric], 0.025)),
                "ci_high": float(np.quantile(stats[metric], 0.975)),
                "bootstrap_unit": "condition_group", "n_bootstrap": 1000,
            })
    return pd.DataFrame(rows)


def main():
    fold = pd.read_csv(DATA / f"{TAG}_fold_metrics.csv", encoding="utf-8-sig")
    pred = pd.read_csv(DATA / f"{TAG}_oof_predictions.csv", encoding="utf-8-sig")
    pooled = pd.read_csv(DATA / f"{TAG}_pooled_metrics.csv", encoding="utf-8-sig")
    config = json.loads((DATA / f"{TAG}_run_config.json").read_text(encoding="utf-8"))
    groups = pd.read_csv(DATA / "condition_groups.csv", encoding="utf-8-sig")
    assert config["data_rows"] == 284
    assert config["external_experiment_targets_read"] is False

    combo = ["sample_id", "scheme", "feature_set", "model", "target"]
    if pred.duplicated(combo).any():
        raise AssertionError("Duplicate out-of-fold prediction detected")
    coverage = pred.groupby(["scheme", "feature_set", "model", "target"])["sample_id"].nunique()
    if not (coverage == 284).all():
        raise AssertionError(coverage[coverage != 284].to_string())

    # Independent pooled-metric recomputation.
    checks = []
    for key, sub in pred.groupby(["scheme", "feature_set", "model", "target"]):
        recomputed = score(sub["observed"], sub["predicted"])
        saved = pooled[
            (pooled["scheme"] == key[0]) & (pooled["feature_set"] == key[1])
            & (pooled["model"] == key[2]) & (pooled["target"] == key[3])
        ]
        if len(saved) != 1:
            raise AssertionError(f"Missing pooled metric for {key}")
        checks.append(max(abs(recomputed[m] - float(saved.iloc[0][m])) for m in recomputed))
    if max(checks) > 1e-10:
        raise AssertionError(f"Metric mismatch {max(checks)}")

    selected = choose_models(pooled)
    safe_csv(selected, DATA / "model_freeze_selection.csv")

    ladder_rows = []
    for chosen in selected.itertuples():
        sub = pooled[
            (pooled["target"] == chosen.target)
            & (pooled["feature_set"] == chosen.feature_set)
            & (pooled["model"] == chosen.model)
        ].copy()
        ladder_rows.append(sub)
    ladder = pd.concat(ladder_rows, ignore_index=True)
    safe_csv(ladder, DATA / "selected_model_validation_ladder.csv")

    uncertainty_rows = []
    gpr = pred[pred["model"] == "gpr"].dropna(subset=["predictive_std"])
    for key, sub in gpr.groupby(["scheme", "feature_set", "target"]):
        lo = sub["predicted"] - 1.96 * sub["predictive_std"]
        hi = sub["predicted"] + 1.96 * sub["predictive_std"]
        uncertainty_rows.append({
            "scheme": key[0], "feature_set": key[1], "target": key[2],
            "nominal": 0.95,
            "coverage": float(((sub["observed"] >= lo) & (sub["observed"] <= hi)).mean()),
            "mean_width": float((hi - lo).mean()),
            "rmse": score(sub["observed"], sub["predicted"])["rmse"],
        })
    uncertainty = pd.DataFrame(uncertainty_rows)
    safe_csv(uncertainty, DATA / "gpr_uncertainty_calibration.csv")

    ci = bootstrap_selected(pred, selected, groups)
    safe_csv(ci, DATA / "condition_group_bootstrap_ci.csv")

    hashes = {}
    for path in [
        Path(__file__).resolve().parent / "02_nested_validation.py",
        DATA / f"{TAG}_run_config.json",
        DATA / f"{TAG}_fold_metrics.csv",
        DATA / f"{TAG}_oof_predictions.csv",
        DATA / f"{TAG}_pooled_metrics.csv",
    ]:
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    freeze = {
        "status": "FROZEN_BEFORE_EXTERNAL_EVALUATION",
        "data_rows": 284,
        "external_targets_used_for_selection": False,
        "selection": selected.to_dict(orient="records"),
        "hashes": hashes,
        "metric_recompute_max_abs_difference": max(checks),
    }
    safe_text(json.dumps(freeze, ensure_ascii=False, indent=2), AUDIT / "model_freeze_manifest.json")

    lines = []
    for r in selected.itertuples():
        lines.append(
            f"- {r.target}: {r.model} / {r.feature_set}，condition组外RMSE={r.rmse:.3f}，"
            f"MAE={r.mae:.3f}，R²={r.r2:.3f}。"
        )
    report = """# 嵌套验证与模型冻结报告

## 验证完整性

- 每个“验证方案×特征集×模型×目标”均覆盖284个唯一建模样本。
- 相同样本在同一组合下仅有一个严格组外预测。
- 保存指标已由组外预测独立重算，最大绝对差见冻结清单。
- 模型选择过程未读取4个外部实验样本目标。

## 按预注册规则选定

""" + "\n".join(lines) + """

## 解释边界

- condition是模型选择主口径；random只作乐观偏差对照。
- route和leave_grade用于界定外推边界，不用较弱的外推结果反向更换主口径。
- GPR区间是否可直接采用，由保存的覆盖率和相对RMSE门槛决定；不满足时使用组外残差conformal区间。
"""
    safe_text(report, AUDIT / "05_nested_validation_freeze.md")
    print(report)
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
