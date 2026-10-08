#!/usr/bin/env python3
"""Build the v2 Supplementary figures without duplicating main-text panels."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from release_paths import PRIVATE_ROOT as AL, PAPER_OUTPUT as OUT, SCIENTIFIC_OUTPUT as OLD
FIG = OUT / "figures/supporting"
DATA = OUT / "data_materials_v2"
GRADES = ["2024", "5083", "6082", "7075"]
C = {"2024": "#C58E12", "5083": "#2E6B9A", "6082": "#78872B", "7075": "#AA5578"}
INK = "#2B2B2B"
MM = 1 / 25.4

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7.2, "axes.labelsize": 7.2, "axes.titlesize": 7.6,
    "xtick.labelsize": 6.4, "ytick.labelsize": 6.2,
    "axes.spines.right": False, "axes.spines.top": False,
    "svg.fonttype": "none", "pdf.fonttype": 42,
})


def label(ax, text):
    ax.text(-0.14, 1.05, text, transform=ax.transAxes, fontsize=8.2,
            fontweight="bold", ha="left", va="bottom")


def contrast_text_color(rgba):
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "white" if luminance < 0.52 else INK


def save(fig, stem):
    from matplotlib.text import Text
    for text in fig.findobj(Text):
        value = text.get_text()
        for before, after in [("UTS - YS", "Δσ"), ("UTS-YS", "Δσ"), ("DeltaSigma", "Δσ"),
                              ("Al2CuMg", "Al₂CuMg"), ("Mg2Si", "Mg₂Si"), ("MgZn2", "MgZn₂")]:
            value = value.replace(before, after)
        text.set_text(value)
    FIG.mkdir(parents=True, exist_ok=True)
    base = FIG / stem
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pad_inches=0.02, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)


def fig_s1():
    selected = pd.read_csv(DATA / "descriptor_selected_by_grade.csv", encoding="utf-8-sig")
    selected["grade"] = selected.grade.astype(str)
    fig, axes = plt.subplots(2, 2, figsize=(145 * MM, 104 * MM))
    for ax, grade, letter in zip(axes.ravel(), GRADES, "abcd"):
        q = selected[selected.grade == grade].sort_values("stability_score").tail(10)
        y = np.arange(len(q))
        colors = [C[grade] if x else "#BDBDBD" for x in q.passes_threshold]
        ax.barh(y, q.stability_score, color=colors, height=0.58)
        labs = [x if len(x) <= 23 else x[:21] + "..." for x in q.label]
        ax.set_yticks(y, labs)
        ax.set_xlim(0, max(0.55, float(q.stability_score.max()) * 1.12))
        ax.set_xlabel("abs(rho) x sign stability")
        ax.set_title(grade, loc="left", color=C[grade], fontweight="bold")
        label(ax, letter)
        for yi, r in enumerate(q.itertuples()):
            ax.text(r.stability_score + 0.008, yi, r.target_short,
                    va="center", fontsize=5.5)
    fig.subplots_adjust(left=0.24, right=0.99, top=0.95, bottom=0.09,
                        hspace=0.40, wspace=0.55)
    save(fig, "figS1_descriptor_stability_full")


def fig_s5():
    card = pd.read_csv(OLD / "data/process_cardinality_by_grade.csv", encoding="utf-8-sig")
    card["grade"] = card.grade.astype(str)
    fields = [
        "homogenization_temp_c", "homogenization_time_h", "solution_temp_c",
        "solution_time_h", "aging_temp_c", "aging_time_h", "annealing_temp_c",
        "annealing_time_h", "rolling_temp_c", "rolling_reduction_pct",
        "specimen_thickness_mm", "manufacturing_route", "cooling_method",
        "heat_treatment_route",
    ]
    short = ["Homog. T", "Homog. t", "Solution T", "Solution t", "Age T", "Age t",
             "Anneal T", "Anneal t", "Rolling T", "Reduction", "Thickness",
             "Route", "Cooling", "Heat-treatment"]
    pivot = card.pivot(index="grade", columns="field", values="n_unique").reindex(GRADES)
    arr = pivot[fields].to_numpy(float)
    assert np.isfinite(arr).all() and (arr >= 1).all()
    log_arr = np.log2(arr)
    fig, ax = plt.subplots(figsize=(145 * MM, 58 * MM))
    im = ax.imshow(log_arr, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(fields)), short, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(4), GRADES)
    ax.set_xlabel("Process or route field")
    for i in range(4):
        for j in range(len(fields)):
            ax.text(j, i, f"{int(arr[i, j])}", ha="center", va="center", fontsize=5.6,
                    color=contrast_text_color(im.cmap(im.norm(log_arr[i, j]))))
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("log2(unique values)")
    fig.subplots_adjust(left=0.08, right=0.96, top=0.95, bottom=0.24)
    save(fig, "figS2_process_field_coverage")



def fig_s4():
    df = pd.read_csv(OLD / "data/nested_v1_full_pooled_metrics.csv", encoding="utf-8-sig")
    df = df[(df.scheme == "condition") & df.target.isin(
        ["yield_strength_mpa", "delta_strength_mpa", "elongation_pct"]
    )]
    rows = []
    for target in df.target.unique():
        sub = df[df.target == target]
        for feature_set in ["composition", "process", "joint"]:
            best = sub[(sub.feature_set == feature_set) &
                       sub.model.isin(["elasticnet", "pls", "xgb", "gpr"])].sort_values("rmse").iloc[0]
            rows.append([target, feature_set.title(), best.rmse, "model"])
        for model, label_name in [("grade_mean", "Grade mean"), ("route_mean", "Route mean")]:
            base = sub[(sub.feature_set == "metadata_baseline") & (sub.model == model)].iloc[0]
            rows.append([target, label_name, base.rmse, "baseline"])
    plot = pd.DataFrame(rows, columns=["target", "label", "rmse", "kind"])
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(145 * MM, 58 * MM))
    specs = [
        ("yield_strength_mpa", "YS (MPa)"),
        ("delta_strength_mpa", "UTS - YS (MPa)"),
        ("elongation_pct", "EL (%)"),
    ]
    order = ["Composition", "Process", "Joint", "Grade mean", "Route mean"]
    for ax, (target, title), letter in zip(axes, specs, "abc"):
        sub = plot[plot.target == target].set_index("label").loc[order]
        y = np.arange(5)
        for i, (_, row) in enumerate(sub.iterrows()):
            face = "#2E6B9A" if row.kind == "model" else "white"
            edge = "#2E6B9A" if row.kind == "model" else "#737373"
            ax.scatter(row.rmse, i, s=22, facecolor=face, edgecolor=edge, lw=0.8)
            ax.text(row.rmse, i - 0.18, f"{row.rmse:.2f}", ha="center",
                    va="bottom", fontsize=5.6)
        ax.set_yticks(y, order)
        ax.invert_yaxis()
        ax.set_xlim(0, sub.rmse.max() * 1.12)
        ax.set_xlabel("Grouped-CV RMSE")
        ax.set_title(title, loc="left", fontweight="bold")
        label(ax, letter)
        if letter != "a":
            ax.tick_params(axis="y", labelleft=False)
    fig.subplots_adjust(left=0.18, right=0.99, bottom=0.18, top=0.84, wspace=0.36)
    save(fig, "figS4_feature_increment")


def fig_s5_explanation():
    shap = pd.read_csv(OLD / "data/conditional_shap_stability.csv", encoding="utf-8-sig")
    perm = pd.read_csv(OLD / "data/elongation_gpr_permutation_summary.csv", encoding="utf-8-sig")
    names = {
        "aging_time_h": "Aging time", "specimen_thickness_mm": "Thickness",
        "homogenization_time_h": "Homog. time",
        "rolling_reduction_pct": "Reduction",
        "solution_time_h": "Solution time", "homogenization_temp_c": "Homog. T",
        "aging_temp_c": "Aging T", "solution_temp_c": "Solution T",
        "annealing_temp_c": "Anneal T", "annealing_time_h": "Anneal time",
        "rolling_temp_c": "Rolling T", "Zr_wt_pct": "Zr",
    }
    fig, axes = plt.subplots(1, 3, figsize=(145 * MM, 64 * MM))
    for ax, target, title, letter in [
        (axes[0], "yield_strength_mpa", "YS grouped SHAP", "a"),
        (axes[1], "delta_strength_mpa", "UTS - YS grouped SHAP", "b"),
    ]:
        q = shap[shap.target == target].sort_values(
            ["top5_frequency", "mean_importance"], ascending=False
        ).head(5).sort_values("mean_importance")
        y = np.arange(len(q))
        ax.barh(y, q.mean_importance, color="#2E6B9A", height=0.62)
        ax.set_yticks(y, [names.get(x, x) for x in q.feature])
        ax.set_xlabel("Mean absolute SHAP (MPa)")
        ax.set_xlim(0, q.mean_importance.max() * 1.35)
        for i, row in enumerate(q.itertuples()):
            ax.text(row.mean_importance + q.mean_importance.max() * 0.025, i,
                    f"{row.top5_frequency:.0%}", va="center", fontsize=5.6)
        ax.set_title(title, loc="left", fontweight="bold")
        label(ax, letter)
    ax = axes[2]
    q = perm[perm.mean_delta_rmse > 0].sort_values(
        "mean_delta_rmse", ascending=False
    ).head(5).sort_values("mean_delta_rmse")
    y = np.arange(len(q))
    ax.barh(y, q.mean_delta_rmse, xerr=q.sd_delta_rmse, color="#6A93B5",
            height=0.62, error_kw={"ecolor": "#737373", "elinewidth": 0.7, "capsize": 1.5})
    ax.set_yticks(y, [names.get(x, x) for x in q.feature])
    ax.set_xlabel("Permutation RMSE increase (%)")
    xmax = float((q.mean_delta_rmse + q.sd_delta_rmse).max())
    ax.set_xlim(min(-0.03, float((q.mean_delta_rmse - q.sd_delta_rmse).min()) * 1.08),
                xmax * 1.42)
    for i, row in enumerate(q.itertuples()):
        ax.text(row.mean_delta_rmse + row.sd_delta_rmse + xmax * 0.03, i,
                f"{row.positive_fraction:.0%}", va="center", fontsize=5.6)
    ax.set_title("EL grouped permutation", loc="left", fontweight="bold")
    label(ax, "c")
    fig.subplots_adjust(left=0.18, right=0.99, bottom=0.18, top=0.84, wspace=0.68)
    save(fig, "figS5_explanation_stability")

def load_script(name):
    import importlib.util
    import sys
    path = Path(__file__).resolve().parent / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    fig_s1()
    fig_s5()
    fig_s4()
    fig_s5_explanation()
    old = load_script("12_make_supplementary_figures.py")
    old.SUPP = FIG
    old.setup()
    original_save = old.save_bundle
    def renamed_save(fig, directory, stem):
        original_save(fig, directory, stem.replace("figS2_validation_hierarchy", "figS3_validation_hierarchy"))
    old.save_bundle = renamed_save
    old.supp_fig2()
    trial = load_script("13_make_trial_distance_figure.py")
    trial.FIG = FIG
    trial.main()
    main_figs = load_script("11_make_main_figures.py")
    values = main_figs.load()
    main_figs.FIG = FIG
    main_figs.thickness_figure(values[0], values[8])


if __name__ == "__main__":
    main()
