#!/usr/bin/env python3
"""Fit and freeze final 284-sample models without reading external targets."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


from release_paths import PRIVATE_ROOT as AL, SCIENTIFIC_OUTPUT as REV
DATA = REV / "data"
AUDIT = REV / "audit"


def load_core():
    path = Path(__file__).resolve().parent / "02_nested_validation.py"
    spec = importlib.util.spec_from_file_location("nested_validation_core", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def finite_group_quantile(values: np.ndarray, coverage: float = 0.95) -> tuple[float, float]:
    n = len(values)
    level = min(1.0, math.ceil((n + 1) * coverage) / n)
    return float(np.quantile(values, level, method="higher")), level


def safe_text(text: str, path: Path):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text, encoding="utf-8")


def main():
    core = load_core()
    df, feature_sets = core.load_data()
    selected = pd.read_csv(DATA / "model_freeze_selection.csv", encoding="utf-8-sig")
    oof = pd.read_csv(DATA / "nested_v1_full_oof_predictions.csv", encoding="utf-8-sig")
    group_map = pd.read_csv(DATA / "condition_groups.csv", encoding="utf-8-sig").set_index("sample_id")["condition_group"]
    model_dir = DATA / "frozen_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    if any(model_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty {model_dir}")

    configs = []
    calibration_rows = []
    for row in selected.itertuples():
        target = row.target
        model_name = row.model
        feature_name = row.feature_set
        spec = feature_sets[feature_name]
        cols = spec["numeric"] + spec["categorical"]
        inner = core.inner_splits(df, "condition")
        estimator = core.model_search(model_name, spec, inner, core.RANDOM_STATE)
        estimator.fit(df[cols], df[target].to_numpy(float))
        best_params = getattr(estimator, "best_params_", {})
        model_path = model_dir / f"{target}_{model_name}_{feature_name}.joblib"
        joblib.dump(estimator, model_path, compress=3)

        residual = oof[
            (oof["scheme"] == "condition")
            & (oof["target"] == target)
            & (oof["model"] == model_name)
            & (oof["feature_set"] == feature_name)
        ][["sample_id", "observed", "predicted"]].copy()
        if residual["sample_id"].nunique() != 284:
            raise AssertionError(f"Incomplete OOF residuals for {target}")
        residual["condition_group"] = residual["sample_id"].map(group_map)
        residual["abs_error"] = (residual["observed"] - residual["predicted"]).abs()
        group_max = residual.groupby("condition_group")["abs_error"].max().to_numpy()
        q95, qlevel = finite_group_quantile(group_max, 0.95)
        calibration_rows.append({
            "target": target, "model": model_name, "feature_set": feature_name,
            "n_oof_samples": len(residual), "n_condition_groups": len(group_max),
            "method": "finite-sample condition-group max absolute OOF residual",
            "nominal_coverage": 0.95, "quantile_level": qlevel,
            "symmetric_half_width": q95,
        })
        configs.append({
            "target": target, "model": model_name, "feature_set": feature_name,
            "numeric_features": spec["numeric"], "categorical_features": spec["categorical"],
            "best_params_full_284_group_cv": best_params,
            "model_file": str(model_path.relative_to(REV)),
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "external_targets_used": False,
        })

    calibration = pd.DataFrame(calibration_rows)
    cal_path = DATA / "conformal_group_calibration.csv"
    if cal_path.exists():
        raise FileExistsError(cal_path)
    calibration.to_csv(cal_path, index=False, encoding="utf-8-sig")

    config_path = DATA / "final_model_config.json"
    safe_text(json.dumps({
        "status": "FINAL_MODELS_FROZEN_BEFORE_EXTERNAL_EVALUATION",
        "n_training_samples": 284,
        "response_parameterization": ["yield_strength_mpa", "delta_strength_mpa", "elongation_pct"],
        "uts_reconstruction": "YS + max(0, delta)",
        "models": configs,
        "external_targets_used": False,
    }, ensure_ascii=False, indent=2), config_path)

    files = [config_path, cal_path, DATA / "model_freeze_selection.csv"] + sorted(model_dir.glob("*.joblib"))
    manifest = []
    for path in files:
        manifest.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(REV)}")
    safe_text("\n".join(manifest) + "\n", AUDIT / "final_model_freeze.sha256")
    print(json.dumps(configs, ensure_ascii=False, indent=2, default=str))
    print(calibration.to_string(index=False))


if __name__ == "__main__":
    main()
