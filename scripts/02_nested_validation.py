#!/usr/bin/env python3
"""Leakage-controlled validation on the 284 modelling samples only.

The external four-sample experiment file is intentionally not referenced.
All preprocessing and tuning occur inside the outer training fold.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    LeaveOneGroupOut,
    RandomizedSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


from release_paths import PRIVATE_ROOT as AL, SCIENTIFIC_OUTPUT as REV
RANDOM_STATE = 42

RAW_TARGETS = ["yield_strength_mpa", "ultimate_tensile_strength_mpa", "elongation_pct"]
MODEL_TARGETS = ["yield_strength_mpa", "delta_strength_mpa", "elongation_pct"]
CAT_COLS = ["manufacturing_route", "cooling_method", "heat_treatment_route"]
PROC_COLS = [
    "homogenization_temp_c", "homogenization_time_h", "solution_temp_c",
    "solution_time_h", "aging_temp_c", "aging_time_h", "annealing_temp_c",
    "annealing_time_h", "rolling_temp_c", "rolling_reduction_pct",
    "specimen_thickness_mm",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--schemes", nargs="+", default=["random", "condition", "route", "leave_grade"])
    p.add_argument("--feature-sets", nargs="+", default=["composition", "process", "joint"])
    p.add_argument("--models", nargs="+", default=["elasticnet", "pls", "xgb", "gpr"])
    p.add_argument("--targets", nargs="+", default=MODEL_TARGETS)
    p.add_argument("--max-outer-folds", type=int, default=0)
    p.add_argument("--output-tag", default="nested_v1")
    return p.parse_args()


def load_data() -> tuple[pd.DataFrame, dict]:
    eng = pd.read_csv(AL / "data/02_engineered_features.csv", encoding="utf-8-sig")
    raw = pd.read_csv(AL / "data/01_processed_data.csv", encoding="utf-8-sig")
    groups = pd.read_csv(REV / "data/condition_groups.csv", encoding="utf-8-sig")
    assert len(eng) == len(raw) == len(groups) == 284
    assert eng["sample_id"].is_unique and raw["sample_id"].is_unique

    meta = raw[["sample_id"] + CAT_COLS].copy()
    df = eng.merge(meta, on="sample_id", how="left", validate="one_to_one")
    df = df.merge(groups[["sample_id", "condition_group"]], on="sample_id", validate="one_to_one")
    df["grade"] = df["alloy_or_coating_name"].astype(str)
    df["route_group"] = df[CAT_COLS].astype(str).agg("|".join, axis=1)
    df["delta_strength_mpa"] = (
        df["ultimate_tensile_strength_mpa"] - df["yield_strength_mpa"]
    )
    assert (df["delta_strength_mpa"] >= 0).all()

    all_numeric = [
        c for c in eng.columns
        if c not in RAW_TARGETS + ["sample_id", "material_system", "alloy_or_coating_name"]
    ]
    comp = [c for c in all_numeric if c.endswith("_wt_pct")]
    comp += [c for c in ["Zn_x_Mg", "Cu_x_Mg", "Mg_x_Si", "total_alloy_wt"] if c in all_numeric]
    feature_sets = {
        "composition": {"numeric": comp, "categorical": []},
        "process": {"numeric": PROC_COLS, "categorical": CAT_COLS},
        "joint": {"numeric": all_numeric, "categorical": CAT_COLS},
    }
    return df, feature_sets


def outer_splits(df: pd.DataFrame, scheme: str):
    index = np.arange(len(df))
    if scheme == "random":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        splits = cv.split(index, df["grade"])
    elif scheme == "condition":
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        splits = cv.split(index, df["grade"], df["condition_group"])
    elif scheme == "route":
        cv = GroupKFold(n_splits=5)
        splits = cv.split(index, groups=df["route_group"])
    elif scheme == "leave_grade":
        cv = LeaveOneGroupOut()
        splits = cv.split(index, groups=df["grade"])
    else:
        raise ValueError(f"Unknown validation scheme: {scheme}")
    return list(splits)


def inner_splits(train: pd.DataFrame, scheme: str):
    index = np.arange(len(train))
    if scheme == "random":
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE + 1)
        return list(cv.split(index, train["grade"]))
    if scheme == "condition":
        cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE + 1)
        return list(cv.split(index, train["grade"], train["condition_group"]))
    if scheme == "route":
        n = min(4, train["route_group"].nunique())
        cv = GroupKFold(n_splits=n)
        return list(cv.split(index, groups=train["route_group"]))
    # For leave-one-grade-out, tune without holding out another complete grade;
    # condition groups remain intact and grade proportions are preserved.
    n = min(3, train["grade"].nunique())
    cv = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=RANDOM_STATE + 1)
    return list(cv.split(index, train["grade"], train["condition_group"]))


def preprocessor(spec: dict) -> ColumnTransformer:
    transformers = []
    if spec["numeric"]:
        transformers.append(("num", StandardScaler(), spec["numeric"]))
    if spec["categorical"]:
        transformers.append((
            "cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), spec["categorical"]
        ))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def model_search(name: str, spec: dict, inner, seed: int):
    prep = preprocessor(spec)
    if name == "elasticnet":
        pipe = Pipeline([
            ("prep", prep),
            ("model", ElasticNet(
                max_iter=100000, tol=1e-3, random_state=seed, selection="cyclic"
            )),
        ])
        grid = {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            "model__l1_ratio": [0.1, 0.25, 0.5, 0.75, 0.95],
        }
        return GridSearchCV(pipe, grid, cv=inner, scoring="neg_root_mean_squared_error", n_jobs=4)
    if name == "pls":
        pipe = Pipeline([
            ("prep", prep),
            ("model", PLSRegression(scale=False, max_iter=2000)),
        ])
        grid = {"model__n_components": [2, 4, 6, 8]}
        return GridSearchCV(pipe, grid, cv=inner, scoring="neg_root_mean_squared_error", n_jobs=4)
    if name == "xgb":
        pipe = Pipeline([
            ("prep", prep),
            ("model", XGBRegressor(
                objective="reg:squarederror", tree_method="hist", random_state=seed,
                n_jobs=2, verbosity=0,
            )),
        ])
        dist = {
            "model__n_estimators": [150, 300, 500, 800],
            "model__max_depth": [2, 3, 4, 5],
            "model__learning_rate": [0.02, 0.04, 0.07, 0.1],
            "model__min_child_weight": [1, 3, 6],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__reg_lambda": [1.0, 5.0, 20.0],
            "model__reg_alpha": [0.0, 0.1, 1.0],
        }
        return RandomizedSearchCV(
            pipe, dist, n_iter=18, cv=inner, scoring="neg_root_mean_squared_error",
            random_state=seed, n_jobs=4,
        )
    if name == "gpr":
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2))
            * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=1.5)
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e2))
        )
        return Pipeline([
            ("prep", prep),
            ("model", GaussianProcessRegressor(
                kernel=kernel, normalize_y=True, n_restarts_optimizer=2,
                random_state=seed, alpha=1e-8,
            )),
        ])
    raise ValueError(name)


def predict_1d(estimator, X: pd.DataFrame) -> np.ndarray:
    pred = np.asarray(estimator.predict(X), dtype=float)
    return pred.reshape(-1)


def scores(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "r2": float(r2_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(math.sqrt(mean_squared_error(y, pred))),
    }


def baseline_predictions(train: pd.DataFrame, test: pd.DataFrame, target: str, key: str) -> np.ndarray:
    global_mean = float(train[target].mean())
    if key == "global_mean":
        return np.full(len(test), global_mean)
    group_col = "grade" if key == "grade_mean" else "route_group"
    means = train.groupby(group_col)[target].mean()
    return test[group_col].map(means).fillna(global_mean).to_numpy(float)


def safe_csv(df: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    df.to_csv(path, index=False, encoding="utf-8-sig")


def safe_json(obj: dict, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = REV / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    df, feature_sets = load_data()
    for value in args.schemes:
        if value not in {"random", "condition", "route", "leave_grade"}:
            raise ValueError(value)
    for value in args.feature_sets:
        if value not in feature_sets:
            raise ValueError(value)
    for value in args.models:
        if value not in {"elasticnet", "pls", "xgb", "gpr"}:
            raise ValueError(value)
    for value in args.targets:
        if value not in MODEL_TARGETS:
            raise ValueError(value)

    metric_rows = []
    pred_rows = []
    for scheme in args.schemes:
        splits = outer_splits(df, scheme)
        if args.max_outer_folds > 0:
            splits = splits[: args.max_outer_folds]
        for fold, (tr_idx, te_idx) in enumerate(splits, start=1):
            train = df.iloc[tr_idx].reset_index(drop=True)
            test = df.iloc[te_idx].reset_index(drop=True)
            assert set(train["sample_id"]).isdisjoint(set(test["sample_id"]))
            if scheme == "condition":
                assert set(train["condition_group"]).isdisjoint(set(test["condition_group"]))
            if scheme == "route":
                assert set(train["route_group"]).isdisjoint(set(test["route_group"]))
            if scheme == "leave_grade":
                assert set(train["grade"]).isdisjoint(set(test["grade"]))

            for target in args.targets:
                y_train = train[target].to_numpy(float)
                y_test = test[target].to_numpy(float)
                for baseline in ["global_mean", "grade_mean", "route_mean"]:
                    pred = baseline_predictions(train, test, target, baseline)
                    row = {
                        "scheme": scheme, "fold": fold, "feature_set": "metadata_baseline",
                        "model": baseline, "target": target,
                        "n_train": len(train), "n_test": len(test), "best_params": "{}",
                    }
                    row.update(scores(y_test, pred))
                    metric_rows.append(row)
                    for i, value in enumerate(pred):
                        pred_rows.append({
                            "sample_id": test.loc[i, "sample_id"], "grade": test.loc[i, "grade"],
                            "route_group": test.loc[i, "route_group"], "scheme": scheme,
                            "fold": fold, "feature_set": "metadata_baseline", "model": baseline,
                            "target": target, "observed": y_test[i], "predicted": float(value),
                            "predictive_std": np.nan,
                        })

                for feature_name in args.feature_sets:
                    spec = feature_sets[feature_name]
                    cols = spec["numeric"] + spec["categorical"]
                    inner = inner_splits(train, scheme)
                    for model_name in args.models:
                        seed = RANDOM_STATE + fold
                        estimator = model_search(model_name, spec, inner, seed)
                        estimator.fit(train[cols], y_train)
                        pred = predict_1d(estimator, test[cols])
                        pred_std = np.full(len(test), np.nan)
                        if model_name == "gpr":
                            transformed = estimator.named_steps["prep"].transform(test[cols])
                            pred, pred_std = estimator.named_steps["model"].predict(
                                transformed, return_std=True
                            )
                            pred = np.asarray(pred).reshape(-1)
                            pred_std = np.asarray(pred_std).reshape(-1)
                        best_params = getattr(estimator, "best_params_", {})
                        row = {
                            "scheme": scheme, "fold": fold, "feature_set": feature_name,
                            "model": model_name, "target": target,
                            "n_train": len(train), "n_test": len(test),
                            "best_params": json.dumps(best_params, sort_keys=True, default=str),
                        }
                        row.update(scores(y_test, pred))
                        metric_rows.append(row)
                        for i, value in enumerate(pred):
                            pred_rows.append({
                                "sample_id": test.loc[i, "sample_id"], "grade": test.loc[i, "grade"],
                                "route_group": test.loc[i, "route_group"], "scheme": scheme,
                                "fold": fold, "feature_set": feature_name, "model": model_name,
                                "target": target, "observed": y_test[i], "predicted": float(value),
                                "predictive_std": float(pred_std[i]) if np.isfinite(pred_std[i]) else np.nan,
                            })
                        print(scheme, fold, target, feature_name, model_name, row["r2"], flush=True)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(pred_rows)
    pooled_rows = []
    keys = ["scheme", "feature_set", "model", "target"]
    for key, sub in predictions.groupby(keys, sort=True):
        item = dict(zip(keys, key))
        item.update(scores(sub["observed"].to_numpy(), sub["predicted"].to_numpy()))
        item["n_predictions"] = len(sub)
        item["n_unique_samples"] = sub["sample_id"].nunique()
        pooled_rows.append(item)

    # Reconstruct UTS from independently cross-fitted YS and non-negative delta.
    pivot = predictions.pivot_table(
        index=["sample_id", "grade", "route_group", "scheme", "fold", "feature_set", "model"],
        columns="target", values=["observed", "predicted"], aggfunc="first",
    )
    if {"yield_strength_mpa", "delta_strength_mpa"}.issubset(set(predictions["target"])):
        pivot = pivot.dropna(subset=[
            ("observed", "yield_strength_mpa"), ("observed", "delta_strength_mpa"),
            ("predicted", "yield_strength_mpa"), ("predicted", "delta_strength_mpa"),
        ])
        for key, sub in pivot.groupby(level=[3, 5, 6], sort=True):
            obs = sub[("observed", "yield_strength_mpa")].to_numpy() + sub[("observed", "delta_strength_mpa")].to_numpy()
            pred = sub[("predicted", "yield_strength_mpa")].to_numpy() + np.maximum(
                0.0, sub[("predicted", "delta_strength_mpa")].to_numpy()
            )
            item = {"scheme": key[0], "feature_set": key[1], "model": key[2],
                    "target": "ultimate_tensile_strength_mpa_derived"}
            item.update(scores(obs, pred))
            item["n_predictions"] = len(sub)
            item["n_unique_samples"] = sub.reset_index()["sample_id"].nunique()
            pooled_rows.append(item)

    pooled = pd.DataFrame(pooled_rows)
    prefix = args.output_tag
    safe_csv(metrics, out_dir / f"{prefix}_fold_metrics.csv")
    safe_csv(predictions, out_dir / f"{prefix}_oof_predictions.csv")
    safe_csv(pooled, out_dir / f"{prefix}_pooled_metrics.csv")
    safe_json({
        "data_rows": len(df), "random_state": RANDOM_STATE,
        "schemes": args.schemes, "feature_sets": args.feature_sets,
        "models": args.models, "targets": args.targets,
        "max_outer_folds": args.max_outer_folds,
        "external_experiment_targets_read": False,
    }, out_dir / f"{prefix}_run_config.json")
    print(pooled.sort_values(["target", "scheme", "rmse"]).to_string(index=False))


if __name__ == "__main__":
    main()
