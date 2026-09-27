#!/usr/bin/env python3
"""One-time external evaluation after verifying the frozen-model manifest."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import pairwise_distances


from release_paths import PRIVATE_ROOT as AL, SCIENTIFIC_OUTPUT as REV
DATA = REV / "data"
AUDIT = REV / "audit"


def verify_manifest():
    lines = (AUDIT / "final_model_freeze.sha256").read_text(encoding="utf-8").splitlines()
    for line in lines:
        expected, rel = line.split("  ", 1)
        path = REV / rel
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"Frozen artifact changed: {rel}")


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    h = df.copy()
    h["Zn_x_Mg"] = h["Zn_wt_pct"] * h["Mg_wt_pct"]
    h["Cu_x_Mg"] = h["Cu_wt_pct"] * h["Mg_wt_pct"]
    h["Mg_x_Si"] = h["Mg_wt_pct"] * h["Si_wt_pct"]
    comp = [c for c in h.columns if c.endswith("_wt_pct") and c != "Al_wt_pct"]
    h["total_alloy_wt"] = h[comp].fillna(0).sum(axis=1)
    return h


def safe_csv(df: pd.DataFrame, path: Path):
    if path.exists():
        raise FileExistsError(path)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_text(text: str, path: Path):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text, encoding="utf-8")


def support_ratio(estimator, train: pd.DataFrame, test: pd.DataFrame, cols: list[str]):
    pipeline = getattr(estimator, "best_estimator_", estimator)
    prep = pipeline.named_steps["prep"]
    a = np.asarray(prep.transform(train[cols]), dtype=float)
    b = np.asarray(prep.transform(test[cols]), dtype=float)
    within = pairwise_distances(a, a)
    np.fill_diagonal(within, np.inf)
    reference = float(np.median(np.min(within, axis=1)))
    nearest = np.min(pairwise_distances(b, a), axis=1)
    ratio = nearest / reference if reference > 0 else np.full(len(test), np.inf)
    return nearest, ratio, reference


def main():
    verify_manifest()
    config = json.loads((DATA / "final_model_config.json").read_text(encoding="utf-8"))
    calibration = pd.read_csv(DATA / "conformal_group_calibration.csv", encoding="utf-8-sig")
    train_eng = pd.read_csv(AL / "data/02_engineered_features.csv", encoding="utf-8-sig")
    train_raw = pd.read_csv(AL / "data/01_processed_data.csv", encoding="utf-8-sig")
    train = train_eng.merge(
        train_raw[["sample_id", "manufacturing_route", "cooling_method", "heat_treatment_route"]],
        on="sample_id", validate="one_to_one"
    )
    external_raw = pd.read_csv(
        AL / "data/verification_holdout.csv", encoding="utf-8-sig"
    )
    external = engineer(external_raw)
    assert len(train) == 284 and len(external) == 4

    rows = []
    predictions = {}
    support = {}
    native_std = {}
    for cfg in config["models"]:
        target = cfg["target"]
        model = joblib.load(REV / cfg["model_file"])
        cols = cfg["numeric_features"] + cfg["categorical_features"]
        pred = np.asarray(model.predict(external[cols]), dtype=float).reshape(-1)
        predictions[target] = pred
        native_std[target] = np.full(len(external), np.nan)
        if cfg["model"] == "gpr":
            pipeline = getattr(model, "best_estimator_", model)
            transformed = pipeline.named_steps["prep"].transform(external[cols])
            pred, std = pipeline.named_steps["model"].predict(transformed, return_std=True)
            predictions[target] = np.asarray(pred).reshape(-1)
            native_std[target] = np.asarray(std).reshape(-1)
        nearest, ratio, ref = support_ratio(model, train, external, cols)
        support[target] = (nearest, ratio, ref)

    # Physical reconstruction.
    predictions["delta_strength_mpa"] = np.maximum(0.0, predictions["delta_strength_mpa"])
    predictions["ultimate_tensile_strength_mpa"] = (
        predictions["yield_strength_mpa"] + predictions["delta_strength_mpa"]
    )

    half = dict(zip(calibration["target"], calibration["symmetric_half_width"]))
    half["ultimate_tensile_strength_mpa"] = (
        half["yield_strength_mpa"] + half["delta_strength_mpa"]
    )
    measured = {
        "yield_strength_mpa": external["yield_strength_mpa"].to_numpy(float),
        "delta_strength_mpa": (
            external["ultimate_tensile_strength_mpa"] - external["yield_strength_mpa"]
        ).to_numpy(float),
        "ultimate_tensile_strength_mpa": external["ultimate_tensile_strength_mpa"].to_numpy(float),
        "elongation_pct": external["elongation_pct"].to_numpy(float),
    }
    for target in ["yield_strength_mpa", "delta_strength_mpa", "ultimate_tensile_strength_mpa", "elongation_pct"]:
        for i, sid in enumerate(external["sample_id"]):
            pred = float(predictions[target][i])
            obs = float(measured[target][i])
            source_target = target if target != "ultimate_tensile_strength_mpa" else "yield_strength_mpa"
            nearest, ratio, ref = support[source_target]
            std = native_std.get(target, np.full(len(external), np.nan))[i]
            rows.append({
                "sample_id": sid,
                "grade": str(external.loc[i, "alloy_or_coating_name"]),
                "target": target,
                "measured": obs,
                "predicted": pred,
                "error": pred - obs,
                "absolute_error": abs(pred - obs),
                "relative_error_pct": 100.0 * (pred - obs) / obs if obs != 0 else np.nan,
                "interval_half_width": float(half[target]),
                "interval_low": pred - float(half[target]),
                "interval_high": pred + float(half[target]),
                "interval_covered": bool(abs(pred - obs) <= float(half[target])),
                "native_gpr_std": float(std) if np.isfinite(std) else np.nan,
                "nearest_training_distance": float(nearest[i]),
                "support_distance_ratio": float(ratio[i]),
                "training_nn_distance_median": float(ref),
            })
    detail = pd.DataFrame(rows)
    safe_csv(detail, DATA / "external_experiment_predictions.csv")

    agg = []
    for target, sub in detail.groupby("target", sort=False):
        agg.append({
            "target": target, "n": len(sub),
            "mae": float(mean_absolute_error(sub["measured"], sub["predicted"])),
            "rmse": float(math.sqrt(mean_squared_error(sub["measured"], sub["predicted"]))),
            "exploratory_r2_n4": float(r2_score(sub["measured"], sub["predicted"])),
            "interval_coverage": float(sub["interval_covered"].mean()),
            "max_absolute_relative_error_pct": float(sub["relative_error_pct"].abs().max()),
            "max_support_distance_ratio": float(sub["support_distance_ratio"].max()),
        })
    aggregate = pd.DataFrame(agg)
    safe_csv(aggregate, DATA / "external_experiment_summary.csv")


    lines = []
    for r in aggregate.itertuples():
        lines.append(
            f"- {r.target}: MAE={r.mae:.2f}，RMSE={r.rmse:.2f}，"
            f"4/4区间覆盖率={r.interval_coverage:.0%}，最大绝对相对误差={r.max_absolute_relative_error_pct:.1f}%。"
        )
    report = """# 四样本外部实验复算

## 证据顺序

本脚本首先验证最终模型和校准文件的SHA-256，随后才读取四样本实测目标。改进模型、特征集、超参数和区间半宽均在284个建模样本上冻结，四样本未参与选择或校准。

## 汇总结果

""" + "\n".join(lines) + """

## 表述规则

- 原始锁定模型结果保留为前瞻性实验验证。
- 本次结果称为“冻结分析后的外部实验复算”。
- 由于每个目标仅4个样本，R²只保存在数据表中作为探索性数值，正文以逐样本误差、区间覆盖和支撑域距离为主。
- UTS由YS与非负Δσ相加，模型结构保证UTS>=YS。
"""
    safe_text(report, AUDIT / "06_external_experiment_evaluation.md")
    print(report)
    print(aggregate.to_string(index=False))
    print(detail[["sample_id", "grade", "target", "measured", "predicted", "relative_error_pct", "interval_covered", "support_distance_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
