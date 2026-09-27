#!/usr/bin/env python3
"""Correct support-distance normalization while preserving v1 outputs."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


from release_paths import PRIVATE_ROOT as AL, SCIENTIFIC_OUTPUT as REV
DATA = REV / "data"
AUDIT = REV / "audit"


def engineer(df):
    h = df.copy()
    h["Zn_x_Mg"] = h["Zn_wt_pct"] * h["Mg_wt_pct"]
    h["Cu_x_Mg"] = h["Cu_wt_pct"] * h["Mg_wt_pct"]
    h["Mg_x_Si"] = h["Mg_wt_pct"] * h["Si_wt_pct"]
    comp = [c for c in h.columns if c.endswith("_wt_pct") and c != "Al_wt_pct"]
    h["total_alloy_wt"] = h[comp].fillna(0).sum(axis=1)
    return h


def main():
    config = json.loads((DATA / "final_model_config.json").read_text(encoding="utf-8"))
    eng = pd.read_csv(AL / "data/02_engineered_features.csv", encoding="utf-8-sig")
    raw = pd.read_csv(AL / "data/01_processed_data.csv", encoding="utf-8-sig")
    train = eng.merge(raw[["sample_id", "manufacturing_route", "cooling_method", "heat_treatment_route"]], on="sample_id", validate="one_to_one")
    external = engineer(pd.read_csv(AL / "data/verification_holdout.csv", encoding="utf-8-sig"))

    support = {}
    support_rows = []
    for cfg in config["models"]:
        target = cfg["target"]
        model = joblib.load(REV / cfg["model_file"])
        pipeline = getattr(model, "best_estimator_", model)
        prep = pipeline.named_steps["prep"]
        cols = cfg["numeric_features"] + cfg["categorical_features"]
        a = np.asarray(prep.transform(train[cols]), dtype=float)
        b = np.asarray(prep.transform(external[cols]), dtype=float)
        within = pairwise_distances(a, a)
        np.fill_diagonal(within, np.inf)
        nn = np.min(within, axis=1)
        positive = nn[np.isfinite(nn) & (nn > 1e-12)]
        if len(positive) == 0:
            raise AssertionError(f"No positive training distance for {target}")
        reference = float(np.median(positive))
        nearest = np.min(pairwise_distances(b, a), axis=1)
        ratio = nearest / reference
        support[target] = (nearest, ratio, reference)
        for i, sid in enumerate(external["sample_id"]):
            support_rows.append({
                "sample_id": sid, "support_model_target": target,
                "nearest_training_distance": float(nearest[i]),
                "positive_training_nn_median": reference,
                "support_distance_ratio": float(ratio[i]),
                "normalization": "median of strictly positive training nearest-neighbour distances",
            })
    support_df = pd.DataFrame(support_rows)
    support_df.to_csv(DATA / "external_support_distance_v2.csv", index=False, encoding="utf-8-sig")

    detail = pd.read_csv(DATA / "external_experiment_predictions.csv", encoding="utf-8-sig")
    source = detail["target"].replace({"ultimate_tensile_strength_mpa": "yield_strength_mpa"})
    detail["support_model_target"] = source
    lookup = support_df.set_index(["sample_id", "support_model_target"])
    for idx, row in detail.iterrows():
        item = lookup.loc[(row["sample_id"], row["support_model_target"])]
        detail.loc[idx, "nearest_training_distance"] = item["nearest_training_distance"]
        detail.loc[idx, "training_nn_distance_median"] = item["positive_training_nn_median"]
        detail.loc[idx, "support_distance_ratio"] = item["support_distance_ratio"]
    detail.to_csv(DATA / "external_experiment_predictions_v2.csv", index=False, encoding="utf-8-sig")

    summary = pd.read_csv(DATA / "external_experiment_summary.csv", encoding="utf-8-sig")
    for idx, row in summary.iterrows():
        sub = detail[detail["target"] == row["target"]]
        summary.loc[idx, "max_support_distance_ratio"] = sub["support_distance_ratio"].max()
    summary.to_csv(DATA / "external_experiment_summary_v2.csv", index=False, encoding="utf-8-sig")

    report = """# 外部样本支撑域距离校正

首版支撑域比值使用全部训练样本最近邻距离的中位数。由于工艺/路线特征存在大量完全重复条件，该中位数为0，造成YS、Δσ和UTS的比值为无穷。预测值、误差和区间覆盖不受影响。

校正版保留首版文件，新增v2文件，并使用“严格大于0的训练最近邻距离中位数”归一化。论文和后续分析只引用v2支撑域结果。
"""
    (AUDIT / "07_external_support_correction.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False))
    print(support_df.to_string(index=False))


if __name__ == "__main__":
    main()
