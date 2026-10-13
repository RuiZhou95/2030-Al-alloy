#!/usr/bin/env python3
"""Mechanism-oriented analysis and figures for the Al materials-mainline rewrite.

The script reads the frozen 284-sample modelling set and the four author-confirmed
post-model trial samples. It does not alter any historical artifact.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


from release_paths import PRIVATE_ROOT as AL, SCIENTIFIC_OUTPUT as OLD, PAPER_OUTPUT as OUT
DATA = OUT / "data"
FIG = OUT / "figure"
QA = OUT / "qa"

GRADES = ["2024", "5083", "6082", "7075"]
COLORS = {"2024": "#C98A16", "5083": "#2F6B9A", "6082": "#788A32", "7075": "#B5537D"}
MARKERS = {"2024": "o", "5083": "s", "6082": "^", "7075": "D"}
INK = "#262626"
GREY = "#777777"
LIGHT = "#D9DEE3"
PALE = "#F5F7F8"
BLUE = "#2F6B9A"
GOLD = "#C98A16"
OLIVE = "#788A32"

MASS = {"Mg": 24.305, "Si": 28.085, "Cu": 63.546, "Zn": 65.38}


def setup() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.6,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 9.3,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def panel(ax, label: str, title: str = "") -> None:
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    if title:
        ax.set_title(title, loc="left", pad=8)


def save(fig, name: str) -> None:
    fig.savefig(FIG / name, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(AL / "data/01_processed_data.csv", encoding="utf-8-sig")
    external = pd.read_csv(AL / "data/verification_holdout.csv", encoding="utf-8-sig")
    pred_hist = pd.read_csv(AL / "data/verification_predictions.csv", encoding="utf-8-sig")
    groups = pd.read_csv(OLD / "data/condition_groups.csv", encoding="utf-8-sig")
    assert len(train) == 284 and len(external) == 4
    assert set(train["sample_id"]).isdisjoint(set(external["sample_id"]))
    assert set(external["alloy_or_coating_name"].astype(str)) == set(GRADES)
    return train, external, pred_hist, groups


def engineer(df: pd.DataFrame, role: str) -> pd.DataFrame:
    x = df.copy()
    x["grade"] = x["alloy_or_coating_name"].astype(str)
    x["role"] = role
    x["delta_strength_mpa"] = x["ultimate_tensile_strength_mpa"] - x["yield_strength_mpa"]
    x["delta_over_ys"] = x["delta_strength_mpa"] / x["yield_strength_mpa"]
    x["ys_over_uts"] = x["yield_strength_mpa"] / x["ultimate_tensile_strength_mpa"]
    x["ys_el_product"] = x["yield_strength_mpa"] * x["elongation_pct"]
    red = pd.to_numeric(x["rolling_reduction_pct"], errors="coerce") / 100.0
    x["rolling_true_strain"] = -np.log(np.clip(1.0 - red, 1e-6, None))

    n_mg = 10.0 * pd.to_numeric(x["Mg_wt_pct"], errors="coerce").fillna(0) / MASS["Mg"]
    n_si = 10.0 * pd.to_numeric(x["Si_wt_pct"], errors="coerce").fillna(0) / MASS["Si"]
    n_cu = 10.0 * pd.to_numeric(x["Cu_wt_pct"], errors="coerce").fillna(0) / MASS["Cu"]
    n_zn = 10.0 * pd.to_numeric(x["Zn_wt_pct"], errors="coerce").fillna(0) / MASS["Zn"]
    x["mg_nominal_load_wt_pct"] = pd.to_numeric(x["Mg_wt_pct"], errors="coerce")
    x["s_nominal_capacity_mol_kg"] = np.minimum(n_cu, n_mg)
    x["beta_nominal_capacity_mol_kg"] = np.minimum(n_mg / 2.0, n_si)
    x["eta_nominal_capacity_mol_kg"] = np.minimum(n_mg, n_zn / 2.0)
    x["dispersoid_element_load_wt_pct"] = (
        pd.to_numeric(x["Mn_wt_pct"], errors="coerce").fillna(0)
        + pd.to_numeric(x["Cr_wt_pct"], errors="coerce").fillna(0)
    )
    return x


def mechanism_summary(train: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "mg_nominal_load_wt_pct",
        "s_nominal_capacity_mol_kg",
        "beta_nominal_capacity_mol_kg",
        "eta_nominal_capacity_mol_kg",
        "dispersoid_element_load_wt_pct",
    ]
    summary = train.groupby("grade")[cols].median().loc[GRADES]
    summary.index.name = "grade"
    summary.reset_index().to_csv(DATA / "mechanism_descriptor_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.RandomState, reps: int = 2000) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    if len(values) < 3:
        return float(np.mean(values)), np.nan, np.nan
    idx = rng.randint(0, len(values), size=(reps, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def state_label(row: pd.Series) -> str:
    g = str(row["grade"])
    if g == "2024":
        return f"Solution {row['solution_time_h']:g} h"
    if g == "5083":
        return "Cold-roll + anneal" if "退火" in str(row["manufacturing_route"]) else "Hot-roll route"
    if g == "6082":
        return f"Solution {row['solution_time_h']:g} h"
    return f"Reduction {row['rolling_reduction_pct']:g}%"


def state_response_summary(train: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    x = train.merge(groups[["sample_id", "condition_group"]], on="sample_id", how="left", validate="one_to_one")
    x["state"] = x.apply(state_label, axis=1)
    targets = ["yield_strength_mpa", "delta_strength_mpa", "elongation_pct"]
    for g in GRADES:
        m = x["grade"] == g
        for target in targets:
            mu = x.loc[m, target].mean(); sd = x.loc[m, target].std(ddof=0)
            x.loc[m, target + "_z"] = (x.loc[m, target] - mu) / sd if sd > 0 else 0.0
    rng = np.random.RandomState(42)
    rows = []
    for (g, state), sub in x.groupby(["grade", "state"], sort=False):
        for target in targets:
            agg = sub.groupby("condition_group", as_index=False)[target + "_z"].mean()
            mean, lo, hi = bootstrap_mean_ci(agg[target + "_z"].to_numpy(), rng)
            rows.append({
                "grade": g, "state": state, "target": target,
                "n_samples": len(sub), "n_condition_groups": len(agg),
                "mean_z": mean, "ci_low": lo, "ci_high": hi,
                "raw_mean": sub[target].mean(), "raw_sd": sub[target].std(ddof=1),
            })
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "state_response_summary.csv", index=False, encoding="utf-8-sig")
    x.to_csv(DATA / "materials_engineered_training.csv", index=False, encoding="utf-8-sig")
    return out


def nondominated(df: pd.DataFrame, strength: str) -> pd.DataFrame:
    arr = df[["elongation_pct", strength]].to_numpy(float)
    keep = []
    for i, p in enumerate(arr):
        dominates = ((arr[:, 0] >= p[0]) & (arr[:, 1] >= p[1]) & ((arr[:, 0] > p[0]) | (arr[:, 1] > p[1]))).any()
        keep.append(not dominates)
    return df.loc[np.asarray(keep)].sort_values("elongation_pct")


def percentile(values: pd.Series, value: float) -> float:
    a = values.to_numpy(float)
    return float((a <= value).mean())


def design_validation(train: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in GRADES:
        tr = train[train["grade"] == g]
        ex = ext[ext["grade"] == g].iloc[0]
        rec = {"grade": g, "sample_id": ex["sample_id"]}
        pcts = []
        for target in ["yield_strength_mpa", "ultimate_tensile_strength_mpa", "elongation_pct", "delta_strength_mpa"]:
            p = percentile(tr[target], float(ex[target])); rec[target + "_percentile"] = p
            if target != "delta_strength_mpa": pcts.append(p)
        rec["balanced_property_percentile"] = min(pcts)
        for strength in ["yield_strength_mpa", "ultimate_tensile_strength_mpa"]:
            front = nondominated(tr, strength)
            sx = max(tr["elongation_pct"].max() - tr["elongation_pct"].min(), 1e-9)
            sy = max(tr[strength].max() - tr[strength].min(), 1e-9)
            d = np.sqrt(
                ((front["elongation_pct"] - ex["elongation_pct"]) / sx) ** 2
                + ((front[strength] - ex[strength]) / sy) ** 2
            ).min()
            rec[("ys" if strength.startswith("yield") else "uts") + "_front_distance"] = float(d)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "cross_grade_design_validation.csv", index=False, encoding="utf-8-sig")
    return out


def historical_rf_validation(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred[["sample_id", "target", "measured", "pred_RandomForest", "cv_rmse"]].copy()
    out = out.rename(columns={"pred_RandomForest": "predicted"})
    out["grade"] = out["sample_id"].map({"al_150": "2024", "al_191": "5083", "al_243": "6082", "al_48": "7075"})
    out["error"] = out["predicted"] - out["measured"]
    out["relative_error_pct"] = 100 * out["error"] / out["measured"]
    out["interval_low"] = out["predicted"] - 1.96 * out["cv_rmse"]
    out["interval_high"] = out["predicted"] + 1.96 * out["cv_rmse"]
    out["covered"] = (out["measured"] >= out["interval_low"]) & (out["measured"] <= out["interval_high"])
    rmse = out.groupby("target").apply(
        lambda d: float(np.sqrt(np.mean((d["predicted"] - d["measured"]) ** 2)))
    )
    out["four_sample_rmse"] = out["target"].map(rmse)
    out.to_csv(DATA / "prospective_cross_grade_validation.csv", index=False, encoding="utf-8-sig")
    assert out["covered"].all()
    return out


def fig1_route() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    boxes = [
        (0.02, 0.58, 0.20, "Nominal chemistry", "Mg solid solution\nCu–Mg / Mg–Si / Zn–Mg–Cu\nstoichiometric proxies", GOLD),
        (0.27, 0.58, 0.20, "Mechanism branches", "5083: solution + work hardening\n2024: GPB / S′\n6082: clusters / β″\n7075: GP / η′", BLUE),
        (0.52, 0.58, 0.20, "Process-state proxies", "Solution time | route\nrolling strain | thickness\nheat-treatment record", OLIVE),
        (0.77, 0.58, 0.20, "Latent microstructure", "Supersaturation | precipitates\ndislocations | recovery\ngrain / texture / boundaries", "#777777"),
        (0.64, 0.12, 0.24, "Strength–ductility response", "YS: flow initiation\nΔσ: post-yield load increment\nEL: localization + damage", "#9B6A2F"),
        (0.36, 0.12, 0.22, "Branch-bounded design", "Select mechanism family\nselect supported process state\nquantify uncertainty", BLUE),
        (0.08, 0.12, 0.22, "Cross-grade trial validation", "4 frozen designs\nnew melting + processing + tests\n2024 | 5083 | 6082 | 7075", "#B5537D"),
    ]
    for x, y, w, title, body, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, 0.27, boxstyle="round,pad=0.012,rounding_size=0.018",
                                    fc=PALE, ec=color, lw=1.5))
        ax.text(x + 0.015, y + 0.225, title, fontweight="bold", fontsize=9.2, color=INK, va="center")
        ax.text(x + 0.015, y + 0.175, body, fontsize=8.1, color=INK, va="top", linespacing=1.35)
    arrows = [((0.22, 0.715), (0.27, 0.715)), ((0.47, 0.715), (0.52, 0.715)),
              ((0.72, 0.715), (0.77, 0.715)), ((0.89, 0.58), (0.79, 0.39)),
              ((0.64, 0.255), (0.58, 0.255)), ((0.36, 0.255), (0.30, 0.255))]
    for a, b in arrows:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.3, color=GREY))
    ax.text(0.50, 0.02, "Observed fields → mechanism-informed proxies → bounded rules → post-model experiments",
            ha="center", va="bottom", fontsize=8.6, color=GREY)
    fig.tight_layout(pad=0.4)
    save(fig, "fig1_materials_route.png")


def fig2_branches(train: pd.DataFrame, ext: pd.DataFrame, desc: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.75), gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})
    ax = axes[0]
    raw = desc.to_numpy(float)
    lo = raw.min(axis=0); span = np.where(raw.max(axis=0) > lo, raw.max(axis=0) - lo, 1.0)
    norm = (raw - lo) / span
    im = ax.imshow(norm, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    labels = ["Mg load", "S capacity", "β capacity", "η capacity", "Mn+Cr load"]
    ax.set_xticks(np.arange(len(labels)), labels, rotation=48, ha="right", fontsize=7.6)
    ax.set_yticks(np.arange(4), GRADES)
    for i in range(4):
        for j in range(len(labels)):
            fmt = f"{raw[i,j]:.2f}" if raw[i,j] >= 0.1 else f"{raw[i,j]:.3f}"
            ax.text(j, i, fmt, ha="center", va="center", fontsize=6.8,
                    color="white" if norm[i,j] > 0.58 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Column-normalized level", fontsize=7.5)
    panel(ax, "(a)", "Nominal chemistry fingerprints")

    for k, (ax, y, ylabel, title) in enumerate([
        (axes[1], "yield_strength_mpa", "YS (MPa)", "Strength–ductility branches"),
        (axes[2], "delta_over_ys", "Δσ / YS", "Post-yield reserve–ductility"),
    ]):
        for g in GRADES:
            tr = train[train["grade"] == g]; ex = ext[ext["grade"] == g]
            ax.scatter(tr["elongation_pct"], tr[y], s=16, alpha=0.38, color=COLORS[g], marker=MARKERS[g],
                       edgecolor="none", rasterized=True, label=g)
            ax.scatter(ex["elongation_pct"], ex[y], s=100, color=COLORS[g], marker="*", edgecolor=INK, lw=0.7, zorder=5)
        ax.set_xlabel("EL (%)"); ax.set_ylabel(ylabel); ax.grid(color="#ECECEC", lw=0.6)
        panel(ax, f"({chr(98+k)})", title)
    axes[1].legend(title="Grade", ncol=2, fontsize=7.5, loc="best")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig2_chemistry_property_branches.png")


def sorted_states(g: str, states: list[str]) -> list[str]:
    def key(s: str):
        nums = [float(x) for x in s.replace("%", "").split() if x.replace(".", "", 1).isdigit()]
        return nums[0] if nums else (0 if "Hot" in s else 1)
    return sorted(states, key=key)


def fig3_states(summary: pd.DataFrame) -> None:
    targets = ["yield_strength_mpa", "delta_strength_mpa", "elongation_pct"]
    tlabel = {"yield_strength_mpa": "YS", "delta_strength_mpa": "Δσ", "elongation_pct": "EL"}
    tcolor = {"yield_strength_mpa": BLUE, "delta_strength_mpa": GOLD, "elongation_pct": OLIVE}
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.1))
    for ax, g, lab in zip(axes.ravel(), GRADES, ["(a)", "(b)", "(c)", "(d)"]):
        sub = summary[summary["grade"] == g]
        states = sorted_states(g, list(sub["state"].unique())); x = np.arange(len(states))
        offsets = [-0.18, 0.0, 0.18]
        for off, target in zip(offsets, targets):
            s = sub[sub["target"] == target].set_index("state").reindex(states)
            mean = s["mean_z"].to_numpy(float); lo = s["ci_low"].to_numpy(float); hi = s["ci_high"].to_numpy(float)
            valid = np.isfinite(lo) & np.isfinite(hi)
            ax.scatter(x + off, mean, s=34, color=tcolor[target], label=tlabel[target], zorder=3)
            if valid.any():
                ax.errorbar((x + off)[valid], mean[valid], yerr=np.vstack([mean[valid]-lo[valid], hi[valid]-mean[valid]]),
                            fmt="none", ecolor=tcolor[target], lw=1.0, capsize=2, zorder=2)
        ax.axhline(0, color=GREY, lw=0.8, ls="--")
        ax.set_xticks(x, states, rotation=28, ha="right", fontsize=7.5)
        ax.set_ylabel("Within-grade standardized response")
        ax.grid(axis="y", color="#ECECEC", lw=0.6)
        panel(ax, lab, g)
        counts = sub[sub["target"] == targets[0]].set_index("state").reindex(states)["n_condition_groups"].fillna(0).astype(int)
        for xi, n in zip(x, counts):
            ax.text(xi, ax.get_ylim()[0], f"n={n}", ha="center", va="bottom", fontsize=6.7, color=GREY)
    axes[0,0].legend(ncol=3, fontsize=7.5, loc="upper right")
    fig.tight_layout(h_pad=1.1, w_pad=1.0)
    save(fig, "fig3_branch_state_responses.png")


def fig4_deformation(train: pd.DataFrame, ext: pd.DataFrame) -> None:
    specs = [
        ("yield_strength_mpa", "delta_strength_mpa", "YS (MPa)", "Δσ (MPa)", "Yield barrier vs post-yield increment"),
        ("elongation_pct", "delta_over_ys", "EL (%)", "Δσ / YS", "Normalized reserve vs ductility"),
        ("elongation_pct", "ys_over_uts", "EL (%)", "YS / UTS", "Yield ratio vs ductility"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.9, 3.65))
    for k, (ax, (xv, yv, xl, yl, title)) in enumerate(zip(axes, specs)):
        for g in GRADES:
            tr=train[train["grade"]==g]; ex=ext[ext["grade"]==g]
            ax.scatter(tr[xv],tr[yv],s=15,alpha=0.36,color=COLORS[g],marker=MARKERS[g],edgecolor="none",rasterized=True,label=g)
            ax.scatter(ex[xv],ex[yv],s=95,color=COLORS[g],marker="*",edgecolor=INK,lw=0.7,zorder=5)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(color="#ECECEC",lw=0.6)
        panel(ax,f"({chr(97+k)})",title)
    axes[0].legend(title="Grade",fontsize=7.5,ncol=2,loc="best")
    fig.tight_layout(w_pad=1.0)
    save(fig,"fig4_yield_hardening_ductility.png")


def fig5_design_space(train: pd.DataFrame, ext: pd.DataFrame, validation: pd.DataFrame) -> None:
    fig,axes=plt.subplots(1,3,figsize=(11.0,3.65),gridspec_kw={"width_ratios":[1.0,1.0,0.75]})
    for k,(ax,strength,label) in enumerate([(axes[0],"yield_strength_mpa","YS (MPa)"),(axes[1],"ultimate_tensile_strength_mpa","UTS (MPa)")]):
        for g in GRADES:
            tr=train[train["grade"]==g]; fr=nondominated(tr,strength); ex=ext[ext["grade"]==g]
            ax.scatter(tr["elongation_pct"],tr[strength],s=12,color=COLORS[g],alpha=0.22,edgecolor="none",rasterized=True)
            ax.plot(fr["elongation_pct"],fr[strength],color=COLORS[g],lw=1.4,marker=MARKERS[g],ms=3,label=g)
            ax.scatter(ex["elongation_pct"],ex[strength],marker="*",s=95,color=COLORS[g],edgecolor=INK,lw=0.7,zorder=5)
        ax.set_xlabel("EL (%)");ax.set_ylabel(label);ax.grid(color="#ECECEC",lw=0.6)
        panel(ax,f"({chr(97+k)})",label.split()[0]+"–EL observed envelopes")
    axes[0].legend(title="Grade",fontsize=7.3,ncol=2,loc="best")
    ax=axes[2]; y=np.arange(len(GRADES)); w=0.34
    v=validation.set_index("grade").loc[GRADES]
    ax.barh(y-w/2,v["ys_front_distance"],height=w,color=BLUE,label="YS–EL")
    ax.barh(y+w/2,v["uts_front_distance"],height=w,color=GOLD,label="UTS–EL")
    ax.set_yticks(y,GRADES);ax.invert_yaxis();ax.set_xlabel("Normalized distance to\nwithin-grade observed front")
    ax.grid(axis="x",color="#ECECEC",lw=0.6);ax.legend(fontsize=7.3,loc="best")
    panel(ax,"(c)","Trial-sample front distance")
    fig.tight_layout(w_pad=1.0)
    save(fig,"fig5_cross_grade_design_space.png")


def fig6_validation(pred: pd.DataFrame) -> None:
    specs=[("yield_strength_mpa","YS (MPa)"),("ultimate_tensile_strength_mpa","UTS (MPa)"),("elongation_pct","EL (%)")]
    fig,axes=plt.subplots(2,2,figsize=(10.4,6.0)); flat=axes.ravel()
    for k,(target,title) in enumerate(specs):
        ax=flat[k]; sub=pred[pred["target"]==target].set_index("grade").loc[GRADES].reset_index(); y=np.arange(4)
        err=np.vstack([sub["predicted"]-sub["interval_low"],sub["interval_high"]-sub["predicted"]])
        ax.errorbar(sub["predicted"],y,xerr=err,fmt="o",color=BLUE,ecolor="#A9C5DA",elinewidth=4,capsize=3,ms=5,label="Frozen RF ±1.96 CV-RMSE")
        ax.scatter(sub["measured"],y,marker="*",s=90,color=INK,zorder=5,label="Measured")
        ax.set_yticks(y,[f"{g} ({s})" for g,s in zip(sub["grade"],sub["sample_id"])]);ax.invert_yaxis();ax.set_xlabel(title)
        ax.grid(axis="x",color="#ECECEC",lw=0.6);panel(ax,f"({chr(97+k)})",title)
    flat[0].legend(fontsize=7.2,loc="best")
    ax=flat[3]; piv=pred.pivot(index="grade",columns="target",values="error").loc[GRADES,["yield_strength_mpa","ultimate_tensile_strength_mpa","elongation_pct"]]
    scales=pd.Series({"yield_strength_mpa":pred[pred.target=="yield_strength_mpa"].cv_rmse.iloc[0],"ultimate_tensile_strength_mpa":pred[pred.target=="ultimate_tensile_strength_mpa"].cv_rmse.iloc[0],"elongation_pct":pred[pred.target=="elongation_pct"].cv_rmse.iloc[0]})
    z=piv/scales; lim=max(1.0,float(np.abs(z.to_numpy()).max())); im=ax.imshow(z,cmap="PuOr",vmin=-lim,vmax=lim,aspect="auto")
    ax.set_xticks(np.arange(3),["YS","UTS","EL"]);ax.set_yticks(np.arange(4),GRADES)
    for i in range(4):
        for j in range(3): ax.text(j,i,f"{z.iloc[i,j]:+.2f}",ha="center",va="center",fontsize=7.5,color=INK)
    cb=fig.colorbar(im,ax=ax,fraction=0.045,pad=0.03);cb.set_label("(Predicted − measured) / CV-RMSE",fontsize=7.2)
    panel(ax,"(d)","Cross-grade residual fingerprints")
    fig.tight_layout(h_pad=1.0,w_pad=1.0)
    save(fig,"fig6_cross_grade_trial_validation.png")


def fig7_guidance(validation: pd.DataFrame) -> None:
    fig,ax=plt.subplots(figsize=(11.2,5.0));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
    cols=[0.02,0.16,0.39,0.61,0.78,0.98]
    headers=["Grade","Dominant branch","Observable state knobs","Supported design use","Trial evidence"]
    for i,h in enumerate(headers):
        ax.text((cols[i]+cols[i+1])/2,0.94,h,ha="center",va="center",fontweight="bold",fontsize=8.6)
    rows={
        "2024":["Al–Cu–Mg / GPB–S′","Solution time; reduction; thickness","Balance strength and post-yield increment\ninside observed Al–Cu–Mg branch","334 / 473 MPa; 17.9%"],
        "5083":["Mg solution + work hardening","Complete route; thickness","Select hot-/cold-worked–annealed state\nwithout separating route-confounded effects","271 / 347 MPa; 17.5%"],
        "6082":["Mg–Si clusters / β″","Solution time; reduction; thickness","Select supported solution–deformation state;\naging optimization is not identifiable","318 / 346 MPa; 13.0%"],
        "7075":["Zn–Mg–Cu / GP–η′","Reduction; thickness","Select high-strength product state;\naging and quench changes are not identifiable","499 / 568 MPa; 14.3%"],
    }
    ys=[0.78,0.58,0.38,0.18];h=0.16
    for y,g in zip(ys,GRADES):
        ax.add_patch(FancyBboxPatch((0.015,y-h/2),0.97,h,boxstyle="round,pad=0.005,rounding_size=0.008",fc="#FAFAFA",ec=COLORS[g],lw=1.1))
        ax.text((cols[0]+cols[1])/2,y,g,ha="center",va="center",fontweight="bold",color=COLORS[g],fontsize=9)
        vals=rows[g]
        for i,val in enumerate(vals,1): ax.text((cols[i]+cols[i+1])/2,y,val,ha="center",va="center",fontsize=7.4,linespacing=1.25)
    ax.text(0.5,0.035,"Rule scope: four known mechanism branches and observed process support; not zero-shot prediction of an unseen fifth grade.",ha="center",va="bottom",fontsize=8,color=GREY)
    fig.tight_layout(pad=0.4)
    save(fig,"fig7_bounded_design_guidance.png")


def write_process_coverage(train: pd.DataFrame) -> None:
    fields=["homogenization_temp_c","homogenization_time_h","solution_temp_c","solution_time_h","aging_temp_c","aging_time_h","annealing_temp_c","annealing_time_h","rolling_temp_c","rolling_reduction_pct","specimen_thickness_mm","manufacturing_route","cooling_method","heat_treatment_route"]
    rows=[]
    for g in GRADES:
        sub=train[train["grade"]==g]
        for f in fields:
            vals=sub[f].dropna(); rows.append({"grade":g,"field":f,"n":len(vals),"n_unique":vals.nunique(),"min":vals.min() if pd.api.types.is_numeric_dtype(vals) else "","max":vals.max() if pd.api.types.is_numeric_dtype(vals) else ""})
    pd.DataFrame(rows).to_csv(DATA/"process_state_coverage.csv",index=False,encoding="utf-8-sig")


def main() -> None:
    setup()
    raw_train, raw_ext, pred_hist, groups = read_inputs()
    train=engineer(raw_train,"modelling");ext=engineer(raw_ext,"post-model trial")
    assert (train["delta_strength_mpa"]>=0).all() and (ext["delta_strength_mpa"]>=0).all()
    desc=mechanism_summary(train)
    summary=state_response_summary(train,groups)
    validation=design_validation(train,ext)
    pred=historical_rf_validation(pred_hist)
    ext.to_csv(DATA/"materials_engineered_trial_samples.csv",index=False,encoding="utf-8-sig")
    write_process_coverage(train)

    fig1_route();fig2_branches(train,ext,desc);fig3_states(summary);fig4_deformation(train,ext)
    fig5_design_space(train,ext,validation);fig6_validation(pred);fig7_guidance(validation)

    required=[f"fig{i}_" for i in range(1,8)]
    files=sorted(p.name for p in FIG.glob("*.png"))
    assert all(any(name.startswith(prefix) for name in files) for prefix in required)
    report={
        "status":"PASS","n_modelling":len(train),"n_trial":len(ext),"grades":GRADES,
        "all_trial_intervals_covered":bool(pred["covered"].all()),
        "four_sample_rmse":pred.groupby("target")["four_sample_rmse"].first().to_dict(),
        "figures":files,
        "claims_scope":"cross-grade design across four known strengthening branches; no unseen-grade zero-shot claim",
    }
    (QA/"materials_mainline_analysis_validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
