#!/usr/bin/env python3
"""Route-preserving, support-constrained robust Pareto mapping."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


from release_paths import PRIVATE_ROOT as AL, SCIENTIFIC_OUTPUT as REV
DATA = REV / "data"
AUDIT = REV / "audit"
RNG = np.random.RandomState(42)
N_PER_ROUTE = 8000


def core_module():
    path = Path(__file__).resolve().parent / "02_nested_validation.py"
    spec = importlib.util.spec_from_file_location("nested_core", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pareto_2d(x, y):
    order = np.argsort(-x, kind="mergesort")
    keep, best_y = [], -np.inf
    for idx in order:
        if y[idx] > best_y:
            keep.append(idx)
            best_y = y[idx]
    return np.array(keep, dtype=int)


def transformed_distance(prep, train, candidates, cols, chunk=4000):
    a = np.asarray(prep.transform(train[cols]), dtype=float)
    within = pairwise_distances(a, a)
    np.fill_diagonal(within, np.inf)
    nn = np.min(within, axis=1)
    positive = nn[np.isfinite(nn) & (nn > 1e-12)]
    threshold = float(np.quantile(positive, 0.95))
    median_positive = float(np.median(positive))
    out = []
    for start in range(0, len(candidates), chunk):
        b = np.asarray(prep.transform(candidates.iloc[start:start + chunk][cols]), dtype=float)
        out.append(np.min(pairwise_distances(b, a), axis=1))
    return np.concatenate(out), threshold, median_positive


def main():
    core = core_module()
    train, _ = core.load_data()
    cfg = json.loads((DATA / "final_model_config.json").read_text(encoding="utf-8"))
    cfg = {x["target"]: x for x in cfg["models"]}
    calibration = pd.read_csv(DATA / "conformal_group_calibration.csv", encoding="utf-8-sig").set_index("target")
    models = {t: joblib.load(REV / cfg[t]["model_file"]) for t in cfg}

    comp_cols = [c for c in cfg["elongation_pct"]["numeric_features"] if c.endswith("_wt_pct")]
    all_cols = sorted(set(sum([
        x["numeric_features"] + x["categorical_features"] for x in cfg.values()
    ], [])))
    base_cols = ["sample_id", "grade", "route_group"] + all_cols
    base_cols = list(dict.fromkeys(base_cols))

    parts = []
    for route, sub in train.groupby("route_group", sort=True):
        sub = sub.reset_index(drop=True)
        if len(sub) < 2:
            continue
        a = RNG.randint(0, len(sub), N_PER_ROUTE)
        b = RNG.randint(0, len(sub), N_PER_ROUTE)
        lam = RNG.uniform(0, 1, N_PER_ROUTE)
        cand = sub.iloc[a][base_cols].copy().reset_index(drop=True)
        va = sub.iloc[a][comp_cols].to_numpy(float)
        vb = sub.iloc[b][comp_cols].to_numpy(float)
        cand.loc[:, comp_cols] = lam[:, None] * va + (1 - lam)[:, None] * vb
        cand["Zn_x_Mg"] = cand["Zn_wt_pct"] * cand["Mg_wt_pct"]
        cand["Cu_x_Mg"] = cand["Cu_wt_pct"] * cand["Mg_wt_pct"]
        cand["Mg_x_Si"] = cand["Mg_wt_pct"] * cand["Si_wt_pct"]
        cand["total_alloy_wt"] = cand[[c for c in comp_cols if c != "Al_wt_pct"]].sum(axis=1)
        cand["anchor_sample_id"] = sub.iloc[a]["sample_id"].to_numpy()
        cand["composition_partner_id"] = sub.iloc[b]["sample_id"].to_numpy()
        cand["mix_fraction_anchor"] = lam
        parts.append(cand)
    candidates = pd.concat(parts, ignore_index=True)
    candidates.insert(0, "candidate_id", [f"cand_{i:06d}" for i in range(len(candidates))])

    pred = {}
    for target, model in models.items():
        cols = cfg[target]["numeric_features"] + cfg[target]["categorical_features"]
        pred[target] = np.asarray(model.predict(candidates[cols]), dtype=float).reshape(-1)
    pred["delta_strength_mpa"] = np.maximum(0.0, pred["delta_strength_mpa"])
    pred["ultimate_tensile_strength_mpa"] = pred["yield_strength_mpa"] + pred["delta_strength_mpa"]

    el_model = models["elongation_pct"]
    el_pipe = getattr(el_model, "best_estimator_", el_model)
    el_cols = cfg["elongation_pct"]["numeric_features"] + cfg["elongation_pct"]["categorical_features"]
    el_trans = el_pipe.named_steps["prep"].transform(candidates[el_cols])
    el_mean, el_std = el_pipe.named_steps["model"].predict(el_trans, return_std=True)
    pred["elongation_pct"] = np.asarray(el_mean).reshape(-1)
    el_std = np.asarray(el_std).reshape(-1)

    distance, threshold, median_positive = transformed_distance(
        el_pipe.named_steps["prep"], train, candidates, el_cols
    )
    supported = distance <= threshold
    qys = float(calibration.loc["yield_strength_mpa", "symmetric_half_width"])
    qdelta = float(calibration.loc["delta_strength_mpa", "symmetric_half_width"])
    qel = float(calibration.loc["elongation_pct", "symmetric_half_width"])
    el_penalty = np.maximum(qel, 1.96 * el_std)

    candidates["pred_ys"] = pred["yield_strength_mpa"]
    candidates["pred_delta"] = pred["delta_strength_mpa"]
    candidates["pred_uts"] = pred["ultimate_tensile_strength_mpa"]
    candidates["pred_el"] = pred["elongation_pct"]
    candidates["lcb_ys"] = candidates["pred_ys"] - qys
    candidates["lcb_delta"] = np.maximum(0.0, candidates["pred_delta"] - qdelta)
    candidates["lcb_uts"] = candidates["lcb_ys"] + candidates["lcb_delta"]
    candidates["gpr_el_std"] = el_std
    candidates["el_uncertainty_penalty"] = el_penalty
    candidates["lcb_el"] = candidates["pred_el"] - el_penalty
    candidates["support_distance"] = distance
    candidates["support_distance_ratio"] = distance / median_positive
    candidates["supported"] = supported
    candidates["physically_feasible"] = (
        (candidates["pred_uts"] >= candidates["pred_ys"])
        & (candidates["lcb_el"] > 0)
    )
    feasible = candidates[candidates["supported"] & candidates["physically_feasible"]].copy()

    iy = pareto_2d(feasible["lcb_ys"].to_numpy(), feasible["lcb_el"].to_numpy())
    iu = pareto_2d(feasible["lcb_uts"].to_numpy(), feasible["lcb_el"].to_numpy())
    front_y = feasible.iloc[iy].sort_values("lcb_ys").copy()
    front_u = feasible.iloc[iu].sort_values("lcb_uts").copy()

    representatives = []
    for grade, sub in feasible.groupby("grade"):
        vals = sub[["lcb_ys", "lcb_uts", "lcb_el"]].copy()
        norm = (vals - vals.min()) / (vals.max() - vals.min()).replace(0, 1)
        score = norm.min(axis=1)
        chosen = sub.loc[[score.idxmax()]].copy()
        chosen["representative_reason"] = "maximin of grade-local robust objectives"
        representatives.append(chosen)
    rep = pd.concat(representatives, ignore_index=True)

    candidates.to_csv(DATA / "robust_route_candidates.csv", index=False, encoding="utf-8-sig")
    front_y.to_csv(DATA / "robust_pareto_ys_el.csv", index=False, encoding="utf-8-sig")
    front_u.to_csv(DATA / "robust_pareto_uts_el.csv", index=False, encoding="utf-8-sig")
    rep.to_csv(DATA / "robust_representatives_by_grade.csv", index=False, encoding="utf-8-sig")
    limits = {
        "candidate_generation": "observed process vector + within-grade-and-route convex composition interpolation",
        "n_candidates": len(candidates), "n_supported_feasible": len(feasible),
        "route_groups": int(candidates["route_group"].nunique()),
        "support_space": "frozen elongation GPR joint preprocessor space",
        "support_threshold_training_positive_nn_p95": threshold,
        "support_reference_positive_nn_median": median_positive,
        "uncertainty": {"q_ys": qys, "q_delta": qdelta, "q_el": qel, "el_uses_max_of_conformal_and_1.96_gpr_std": True},
        "front_sizes": {"ys_el": len(front_y), "uts_el": len(front_u)},
    }
    (AUDIT / "robust_pareto_protocol.json").write_text(json.dumps(limits, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# 支撑域约束的稳健Pareto审计

## 候选构造

- 共生成 {len(candidates)} 个候选，覆盖 {candidates['route_group'].nunique()} 个已有路线组合。
- 每个候选完整继承一条已观测工艺向量，只在同牌号、同路线内对两条已观测成分作凸组合。
- EL联合特征空间采用训练样本严格非零最近邻距离的95百分位作为支撑阈值；通过支撑域与物理约束的候选为 {len(feasible)} 个。

## 稳健目标

- YS和Δσ使用条件组外残差校准半宽；UTS由YS与非负Δσ重构。
- EL惩罚取组外校准半宽与1.96倍GPR标准差的较大者。
- YS-EL稳健前沿含 {len(front_y)} 点，UTS-EL稳健前沿含 {len(front_u)} 点。

## 科学边界

冻结的YS和Δσ模型只使用工艺/路线输入，因此成分凸组合不会改变强度预测。该结果是已见牌号和路线内的稳健性能权衡映射，不构成新成分的逆向设计证据。正文不得声称当前数据已识别可跨牌号迁移的成分强化函数。
"""
    (AUDIT / "11_robust_pareto_findings.md").write_text(report, encoding="utf-8")
    print(report)
    print(rep[["grade", "route_group", "pred_ys", "pred_uts", "pred_el", "lcb_ys", "lcb_uts", "lcb_el", "support_distance_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
