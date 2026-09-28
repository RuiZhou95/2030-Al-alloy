#!/usr/bin/env python3
"""Rebuild all manuscript and Supplementary figures under the nature-figure contract.

Python/matplotlib is the exclusive visual backend. Every figure is exported as
editable SVG/PDF plus TIFF and PNG previews at its final 145-mm document width.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


from release_paths import PRIVATE_ROOT as AL, PAPER_OUTPUT as OUT, SCIENTIFIC_OUTPUT as OLD
V1 = OUT
MAIN = OUT / "figure/main"
SUPP = OUT / "figures/supplementary"
DATA = OUT / "data"
QA = OUT / "qa"

GRADES = ["2024", "5083", "6082", "7075"]
GRADE_COLOR = {"2024": "#C79424", "5083": "#356C9A", "6082": "#7B8738", "7075": "#A75B78"}
GRADE_LIGHT = {"2024": "#EAD7A8", "5083": "#B9D0E1", "6082": "#CBD2A8", "7075": "#DEC0CC"}
GRADE_MARKER = {"2024": "o", "5083": "s", "6082": "^", "7075": "D"}
INK = "#2B2B2B"
GREY = "#737373"
LIGHT_GREY = "#D8D8D8"
VERY_LIGHT = "#F3F3F3"
BLUE = "#356C9A"
GOLD = "#C79424"
OLIVE = "#7B8738"

MM = 1 / 25.4
FIG_WIDTH_MM = 145


def setup() -> None:
    for p in [MAIN, SUPP, DATA, QA]:
        p.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.2,
        "axes.labelsize": 7.2,
        "axes.titlesize": 7.6,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.4,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": INK,
    })


def fig_mm(width_mm: float, height_mm: float):
    scale = FIG_WIDTH_MM / width_mm
    return plt.figure(figsize=(FIG_WIDTH_MM * MM, height_mm * scale * MM))


def label_panel(ax, letter: str) -> None:
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=8.2,
            fontweight="bold", ha="left", va="bottom")


def contrast_text_color(rgba):
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "white" if luminance < 0.52 else INK


def save_bundle(fig, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / stem
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.02,
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def read_data():
    train = pd.read_csv(AL / "data/01_processed_data.csv", encoding="utf-8-sig")
    ext = pd.read_csv(AL / "data/verification_holdout.csv", encoding="utf-8-sig")
    for d, role in [(train, "modelling"), (ext, "post-model trial")]:
        d["grade"] = d["alloy_or_coating_name"].astype(str)
        d["role"] = role
        d["delta_strength_mpa"] = d["ultimate_tensile_strength_mpa"] - d["yield_strength_mpa"]
        d["delta_over_ys"] = d["delta_strength_mpa"] / d["yield_strength_mpa"]
    desc = pd.read_csv(V1 / "data/mechanism_descriptor_summary.csv", encoding="utf-8-sig")
    desc["grade"] = desc["grade"].astype(str)
    state = pd.read_csv(V1 / "data/state_response_summary.csv", encoding="utf-8-sig")
    state["grade"] = state["grade"].astype(str)
    design = pd.read_csv(V1 / "data/cross_grade_design_validation.csv", encoding="utf-8-sig")
    design["grade"] = design["grade"].astype(str)
    pred = pd.read_csv(V1 / "data/prospective_cross_grade_validation.csv", encoding="utf-8-sig")
    pred["grade"] = pred["grade"].astype(str)
    assert len(train) == 284 and len(ext) == 4
    assert set(ext["grade"]) == set(GRADES)
    return train, ext, desc, state, design, pred


def nondominated(df: pd.DataFrame, strength: str) -> pd.DataFrame:
    arr = df[["elongation_pct", strength]].to_numpy(float)
    keep = np.ones(len(arr), dtype=bool)
    for i, p in enumerate(arr):
        keep[i] = not np.any((arr[:, 0] >= p[0]) & (arr[:, 1] >= p[1]) &
                             ((arr[:, 0] > p[0]) | (arr[:, 1] > p[1])))
    return df.loc[keep].sort_values("elongation_pct")


def source_tables(train, ext, desc, state, design, pred) -> None:
    desc.to_csv(DATA / "fig2a_nominal_chemistry.csv", index=False, encoding="utf-8-sig")
    pd.concat([train, ext], ignore_index=True)[[
        "sample_id", "grade", "role", "yield_strength_mpa", "ultimate_tensile_strength_mpa",
        "delta_strength_mpa", "elongation_pct"
    ]].to_csv(DATA / "fig2b_branch_points.csv", index=False, encoding="utf-8-sig")
    state.to_csv(DATA / "fig3_state_responses.csv", index=False, encoding="utf-8-sig")
    design.to_csv(DATA / "fig4_trial_percentiles.csv", index=False, encoding="utf-8-sig")
    pred.to_csv(DATA / "fig5_trial_predictions.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "n_modelling": len(train), "n_trial": len(ext), "grades": GRADES,
        "exclusion": "none; all 284 modelling and 4 post-model trial observations retained",
        "backend": "Python/matplotlib",
        "final_width_mm": 145,
    }
    (DATA / "figure_source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main_fig1() -> None:
    fig = fig_mm(145, 70)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.10)
    ax = fig.add_subplot(gs[0, 0]); ax.set_axis_off(); label_panel(ax, "a")
    rows = [
        ("2024", "Cu–Mg  →  GPB / S′", "higher YS; broad Δσ"),
        ("5083", "Mg solute + dislocations\n→ recovery / recrystallization", "moderate YS; high EL"),
        ("6082", "Mg–Si  →  clusters / β″", "high yield ratio"),
        ("7075", "Zn–Mg–Cu  →  GP / η′", "high YS; bounded Δσ"),
    ]
    yvals = [0.82, 0.61, 0.40, 0.19]
    colx = [0.02, 0.23, 0.72]
    for x, t in zip(colx, ["Grade", "Mechanism branch", "Macroscopic response"]):
        ax.text(x, 0.96, t, fontsize=6.4, fontweight="bold", ha="left", va="center")
    for (g, branch, response), y in zip(rows, yvals):
        ax.add_patch(FancyBboxPatch((0.0, y-0.075), 0.98, 0.145,
                                    boxstyle="round,pad=0.008,rounding_size=0.012",
                                    fc="#FAFAFA", ec=GRADE_COLOR[g], lw=1.0))
        ax.text(colx[0]+0.01, y, g, color=GRADE_COLOR[g], fontweight="bold", va="center")
        ax.text(colx[1], y, branch, va="center", color=GREY, fontsize=6.2, linespacing=1.15)
        ax.text(colx[2], y, response, va="center", fontsize=6.2)
        ax.add_patch(FancyArrowPatch((0.64, y), (0.70, y), arrowstyle="-|>", mutation_scale=8,
                                     color=LIGHT_GREY, lw=0.8))
    ax = fig.add_subplot(gs[0, 1]); ax.set_axis_off(); label_panel(ax, "b")
    steps = [
        ("284 modelling samples", "branch-specific chemistry and process states"),
        ("Within-branch evidence", "grouped state contrasts and uncertainty"),
        ("Bounded candidate selection", "observed support; no cross-grade interpolation"),
        ("4 post-model trials", "new melting, processing and tensile tests"),
    ]
    ys = [0.84, 0.61, 0.38, 0.15]
    for i, ((title, body), y) in enumerate(zip(steps, ys)):
        fc = VERY_LIGHT if i < 3 else "#F7EDF1"
        ec = GREY if i < 3 else GRADE_COLOR["7075"]
        ax.add_patch(FancyBboxPatch((0.05, y-0.08), 0.90, 0.16,
                                    boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ec, lw=0.9))
        ax.text(0.09, y+0.025, title, fontweight="bold", va="center")
        ax.text(0.09, y-0.035, body, fontsize=6.1, color=GREY, va="center")
        if i < len(steps)-1:
            ax.add_patch(FancyArrowPatch((0.50, y-0.09), (0.50, ys[i+1]+0.09),
                                         arrowstyle="-|>", mutation_scale=8, color=GREY, lw=0.8))
    fig.subplots_adjust(bottom=0.13)
    fig.text(0.50, 0.018,
             "Latent microstructures are hypotheses, not direct measurements.\n"
             "Model and design rules were fixed before trial testing.",
             ha="center", va="bottom", fontsize=5.6, color=GREY,
             linespacing=1.15)
    save_bundle(fig, MAIN, "fig1_mechanism_design_logic")


def main_fig2(train, ext, desc) -> None:
    fig = fig_mm(145, 72)
    gs = fig.add_gridspec(2, 2, width_ratios=[0.82, 1.55], height_ratios=[1.15, 0.85],
                          wspace=0.30, hspace=0.36)
    ax = fig.add_subplot(gs[:, 0]); label_panel(ax, "a")
    cols = ["mg_nominal_load_wt_pct", "s_nominal_capacity_mol_kg", "beta_nominal_capacity_mol_kg",
            "eta_nominal_capacity_mol_kg", "dispersoid_element_load_wt_pct"]
    labels = ["Mg", "S", "β", "η", "Mn+Cr"]
    raw = desc.set_index("grade").loc[GRADES, cols].to_numpy(float)
    span = np.where(raw.max(0) > raw.min(0), raw.max(0)-raw.min(0), 1)
    norm = (raw - raw.min(0)) / span
    cmap = mpl.colormaps["RdBu_r"]
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(5), labels); ax.set_yticks(range(4), GRADES)
    ax.tick_params(length=0)
    for i in range(4):
        for j in range(5):
            value = raw[i, j]
            txt = f"{value:.2f}" if value >= 0.1 else f"{value:.3f}"
            rgba = cmap(norm[i, j]); lum = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.8,
                    color="white" if lum < 0.48 else INK)
    ax.set_xlabel("Nominal load or capacity")
    ax.spines[:].set_visible(False)

    ax = fig.add_subplot(gs[0, 1]); label_panel(ax, "b")
    for g in GRADES:
        tr = train[train.grade == g]; ex = ext[ext.grade == g]
        sizes = 7 + 1.25 * tr["elongation_pct"].to_numpy(float)
        ax.scatter(tr["yield_strength_mpa"], tr["delta_strength_mpa"], s=sizes,
                   color=GRADE_COLOR[g], alpha=0.34, marker=GRADE_MARKER[g], edgecolor="none", rasterized=True)
        ax.scatter(ex["yield_strength_mpa"], ex["delta_strength_mpa"], s=50,
                   color=GRADE_COLOR[g], marker="*", edgecolor=INK, linewidth=0.7, zorder=5)
        mx = tr["yield_strength_mpa"].median(); my = tr["delta_strength_mpa"].median()
        ax.text(mx, my, g, color=GRADE_COLOR[g], fontsize=6.4, fontweight="bold",
                ha="center", va="center")
    ax.set_xlabel("YS (MPa)", labelpad=-1); ax.set_ylabel("Δσ (MPa)")
    ax.text(0.98, 0.96, "Point size = EL", transform=ax.transAxes, ha="right", va="top", fontsize=5.8, color=GREY)

    ax = fig.add_subplot(gs[1, 1]); label_panel(ax, "c")
    properties = [("yield_strength_mpa", "YS"), ("delta_strength_mpa", "Δσ"), ("elongation_pct", "EL")]
    pooled = train[[p for p, _ in properties]]
    mu = pooled.mean(); sd = pooled.std(ddof=0)
    offsets = np.linspace(-0.24, 0.24, 4)
    y = np.arange(3)
    for off, g in zip(offsets, GRADES):
        med = train[train.grade == g][[p for p, _ in properties]].median()
        z = ((med - mu) / sd).to_numpy(float)
        ax.scatter(z, y+off, s=18, color=GRADE_COLOR[g], marker=GRADE_MARKER[g], label=g)
    ax.axvline(0, color=GREY, lw=0.7, ls="--")
    ax.set_yticks(y, [lab for _, lab in properties]); ax.invert_yaxis()
    ax.set_xlabel("Standardized grade median")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.48),
              columnspacing=0.7, handletextpad=0.25)
    fig.subplots_adjust(bottom=0.18, hspace=0.56)
    save_bundle(fig, MAIN, "fig2_chemistry_mechanical_branches")


def state_order(g: str, states: list[str]) -> list[str]:
    def num(s):
        vals = [float(x.replace("%", "")) for x in s.split() if x.replace("%", "").replace(".", "", 1).isdigit()]
        if vals:
            return vals[0]
        return 0 if "Hot" in s else 1
    return sorted(states, key=num)


def main_fig3(state) -> None:
    fig = fig_mm(145, 96)
    axes = fig.subplots(2, 2, sharex=True)
    target_order = ["yield_strength_mpa", "delta_strength_mpa", "elongation_pct"]
    target_label = {"yield_strength_mpa": "YS", "delta_strength_mpa": "Δσ", "elongation_pct": "EL"}
    target_marker = {"yield_strength_mpa": "o", "delta_strength_mpa": "s", "elongation_pct": "^"}
    target_color = {"yield_strength_mpa": BLUE, "delta_strength_mpa": GOLD, "elongation_pct": OLIVE}
    x_min = float(np.nanmin(state.ci_low)) - 0.15; x_max = float(np.nanmax(state.ci_high)) + 0.15
    for ax, g, letter in zip(axes.ravel(), GRADES, list("abcd")):
        sub = state[state.grade == g]
        states = state_order(g, list(sub.state.unique()))
        y = np.arange(len(states)); offsets = [-0.18, 0, 0.18]
        for off, target in zip(offsets, target_order):
            s = sub[sub.target == target].set_index("state").reindex(states)
            mean = s.mean_z.to_numpy(float); lo = s.ci_low.to_numpy(float); hi = s.ci_high.to_numpy(float)
            ax.errorbar(mean, y+off, xerr=np.vstack([mean-lo, hi-mean]), fmt=target_marker[target],
                        color=target_color[target], ecolor=target_color[target], ms=3.5,
                        elinewidth=0.8, capsize=1.5, label=target_label[target])
        ax.axvline(0, color=GREY, lw=0.7, ls="--")
        counts = sub[sub.target == "yield_strength_mpa"].set_index("state").reindex(states).n_condition_groups.astype(int)
        labels = [f"{s.replace('Solution ', 'Sol. ').replace('Reduction ', 'Red. ')}  n={n}" for s, n in zip(states, counts)]
        ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlim(x_min, x_max)
        ax.set_title(g, loc="left", color=GRADE_COLOR[g], fontweight="bold")
        label_panel(ax, letter)
    axes[1, 0].set_xlabel("Within-grade standardized response")
    axes[1, 1].set_xlabel("Within-grade standardized response")
    handles = [Line2D([0], [0], marker=target_marker[t], color=target_color[t], lw=0, label=target_label[t], ms=4)
               for t in target_order]
    fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.53, 1.0),
               columnspacing=1.0, handletextpad=0.3)
    fig.subplots_adjust(left=0.18, right=0.99, bottom=0.10, top=0.92,
                        hspace=0.34, wspace=0.58)
    save_bundle(fig, MAIN, "fig3_within_branch_state_responses")


def main_fig4(train, ext, design) -> None:
    fig = fig_mm(145, 62)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 0.88], wspace=0.34)
    for k, (strength, ylabel) in enumerate([("yield_strength_mpa", "YS (MPa)"),
                                            ("ultimate_tensile_strength_mpa", "UTS (MPa)")]):
        ax = fig.add_subplot(gs[0, k]); label_panel(ax, "ab"[k])
        for g in GRADES:
            tr = train[train.grade == g]; ex = ext[ext.grade == g]
            fr = nondominated(tr, strength)
            ax.scatter(tr.elongation_pct, tr[strength], s=7, color=GRADE_LIGHT[g], alpha=0.60,
                       edgecolor="none", rasterized=True)
            # Keep the observed non-dominated states discrete: a connecting line
            # would imply an experimentally reachable continuous process path.
            ax.scatter(fr.elongation_pct, fr[strength], s=18,
                       marker=GRADE_MARKER[g], facecolor="white",
                       edgecolor=GRADE_COLOR[g], linewidth=0.9, zorder=4)
            ax.scatter(ex.elongation_pct, ex[strength], s=38, marker="*", color=GRADE_COLOR[g],
                       edgecolor=INK, linewidth=0.6, zorder=5)
        ax.set_xlabel("EL (%)"); ax.set_ylabel(ylabel)
    ax = fig.add_subplot(gs[0, 2]); label_panel(ax, "c")
    prop = [("yield_strength_mpa_percentile", "YS", "o"),
            ("ultimate_tensile_strength_mpa_percentile", "UTS", "s"),
            ("elongation_pct_percentile", "EL", "^")]
    y = np.arange(4); offsets = [-0.16, 0, 0.16]
    d = design.set_index("grade").loc[GRADES]
    for off, (col, lab, marker) in zip(offsets, prop):
        ax.scatter(100*d[col], y+off, s=18, marker=marker, color=INK, facecolor="white", label=lab)
    ax.set_yticks(y, GRADES); ax.invert_yaxis(); ax.set_xlim(0, 105)
    ax.set_xlabel("Within-grade percentile")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), columnspacing=0.7, handletextpad=0.2)
    save_bundle(fig, MAIN, "fig4_branch_bounded_design_space")


def main_fig5(pred) -> None:
    fig = fig_mm(145, 92)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.85], hspace=0.44, wspace=0.55)
    specs = [("yield_strength_mpa", "YS (MPa)"), ("ultimate_tensile_strength_mpa", "UTS (MPa)"),
             ("elongation_pct", "EL (%)")]
    for k, (target, label) in enumerate(specs):
        ax = fig.add_subplot(gs[0, 2*k:2*k+2]); label_panel(ax, "abc"[k])
        sub = pred[pred.target == target].set_index("grade").loc[GRADES].reset_index()
        lo = min(sub.measured.min(), sub.interval_low.min()); hi = max(sub.measured.max(), sub.interval_high.max())
        pad = 0.04*(hi-lo); ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], ls="--", lw=0.8, color=GREY)
        for r in sub.itertuples():
            yerr = np.array([[r.predicted-r.interval_low], [r.interval_high-r.predicted]])
            ax.errorbar(r.measured, r.predicted, yerr=yerr, fmt=GRADE_MARKER[str(r.grade)],
                        color=GRADE_COLOR[str(r.grade)], ecolor=GRADE_LIGHT[str(r.grade)],
                        ms=4, elinewidth=1.2, capsize=1.8)
        rmse = sub.four_sample_rmse.iloc[0]
        ax.text(0.04, 0.95, f"RMSE = {rmse:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=6.2)
        ax.set_xlabel(f"Measured {label}"); ax.set_ylabel(f"Predicted {label}")
        ax.set_xlim(lo-pad, hi+pad); ax.set_ylim(lo-pad, hi+pad)

    p_rel = pred.pivot(index="grade", columns="target", values="relative_error_pct").loc[GRADES, [x[0] for x in specs]]
    p_z = pred.pivot(index="grade", columns="target", values="error").loc[GRADES, [x[0] for x in specs]]
    scale = pred.groupby("target").cv_rmse.first().reindex([x[0] for x in specs])
    p_z = p_z / scale
    for ax, matrix, letter, title, lim, fmt in [
        (fig.add_subplot(gs[1, 0:3]), p_rel, "d", "Relative error (%)", max(20, np.abs(p_rel.to_numpy()).max()), "{:+.1f}"),
        (fig.add_subplot(gs[1, 3:6]), p_z, "e", "Residual / CV-RMSE", max(1.5, np.abs(p_z.to_numpy()).max()), "{:+.2f}"),
    ]:
        label_panel(ax, letter)
        cmap = mpl.colormaps["RdBu_r"]
        im = ax.imshow(matrix, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(3), [x[1].split()[0] for x in specs]); ax.set_yticks(range(4), GRADES)
        ax.set_title(title, loc="left", fontsize=7.0)
        ax.tick_params(length=0)
        for i in range(4):
            for j in range(3):
                ax.text(j, i, fmt.format(matrix.iloc[i, j]), ha="center", va="center", fontsize=6.0,
                        color=contrast_text_color(im.cmap(im.norm(matrix.iloc[i, j]))))
        ax.spines[:].set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    save_bundle(fig, MAIN, "fig5_cross_grade_trial_validation")


def supp_fig1() -> None:
    identity = pd.read_csv(OLD / "data/grade_identity_diagnostic.csv", encoding="utf-8-sig")
    var = pd.read_csv(OLD / "data/variance_decomposition_corrected.csv", encoding="utf-8-sig")
    card = pd.read_csv(OLD / "data/process_cardinality_by_grade.csv", encoding="utf-8-sig")
    card["grade"] = card["grade"].astype(str)
    fig = fig_mm(145, 62); gs = fig.add_gridspec(1, 3, width_ratios=[0.82, 0.82, 1.55], wspace=0.46)
    ax = fig.add_subplot(gs[0, 0]); label_panel(ax, "a")
    labs = ["Composition\nlogistic", "Composition\nRF", "Process + route\nlogistic"]
    vals = identity.balanced_accuracy.to_numpy(float); y = np.arange(3)
    ax.barh(y, vals, color=[BLUE, "#6A93B5", BLUE], height=0.55)
    ax.set_yticks(y, labs); ax.invert_yaxis(); ax.set_xlim(0, 1.05); ax.set_xlabel("Balanced accuracy")
    for yi, v in zip(y, vals): ax.text(v-0.03, yi, f"{v:.2f}", ha="right", va="center", color="white", fontsize=6)
    ax = fig.add_subplot(gs[0, 1]); label_panel(ax, "b")
    order = ["yield_strength_mpa", "ultimate_tensile_strength_mpa", "elongation_pct"]
    vv = var.set_index("target").loc[order]
    x = np.arange(3); between = vv.between_fraction_eta2.to_numpy(); within = vv.within_fraction.to_numpy()
    ax.bar(x, between, color=BLUE, width=0.58, label="Between grade")
    ax.bar(x, within, bottom=between, color="#C7D9E8", width=0.58, label="Within grade")
    ax.set_xticks(x, ["YS", "UTS", "EL"]); ax.set_ylim(0, 1); ax.set_ylabel("Variance fraction")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=1)
    ax = fig.add_subplot(gs[0, 2]); label_panel(ax, "c")
    fields = ["homogenization_time_h", "solution_time_h", "aging_time_h", "annealing_time_h",
              "rolling_reduction_pct", "specimen_thickness_mm", "manufacturing_route", "cooling_method"]
    flab = ["Homog. t", "Solution t", "Aging t", "Anneal t", "Reduction", "Thickness", "Route", "Cooling"]
    p = card.pivot(index="grade", columns="field", values="n_unique").loc[GRADES, fields]
    raw = p.to_numpy(float); im = ax.imshow(np.log2(raw), cmap="RdBu_r", vmin=0, vmax=np.log2(raw.max()), aspect="auto")
    ax.set_xticks(range(len(fields)), flab, rotation=42, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(4), GRADES); ax.tick_params(length=0)
    for i in range(4):
        for j in range(len(fields)):
            ax.text(j, i, str(int(raw[i, j])), ha="center", va="center", fontsize=5.6,
                    color=contrast_text_color(im.cmap(im.norm(np.log2(raw[i, j])))))
    ax.spines[:].set_visible(False); fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    save_bundle(fig, SUPP, "figS1_identifiability_support")


def supp_fig2() -> None:
    df = pd.read_csv(OLD / "data/selected_model_validation_ladder.csv", encoding="utf-8-sig")
    fig = fig_mm(145, 55); axes = fig.subplots(1, 3, sharex=True, sharey=True)
    specs = [("yield_strength_mpa", "YS"), ("delta_strength_mpa", "Δσ"), ("elongation_pct", "EL")]
    schemes = ["random", "condition", "route", "leave_grade"]
    labs = ["Random split", "Grouped conditions", "Leave route out", "Leave grade out"]
    for ax, (target, title), letter in zip(axes, specs, "abc"):
        sub = df[df.target == target].set_index("scheme").loc[schemes]
        y = np.arange(4); ax.axvline(0, color=GREY, lw=0.7, ls="--")
        ax.plot(sub.r2, y, color=LIGHT_GREY, lw=0.8)
        for i, (scheme, value) in enumerate(zip(schemes, sub.r2)):
            face = BLUE if scheme in ["random", "condition"] else "white"
            ax.scatter(value, i, s=22, facecolor=face, edgecolor=BLUE, lw=0.8)
            ax.text(value+0.10, i, f"{value:.2f}", va="center", fontsize=5.8)
        ax.set_yticks(y, labs); ax.invert_yaxis(); ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Out-of-fold R²"); label_panel(ax, letter); ax.set_xlim(-2.7, 1.15)
        if letter != "a":
            ax.tick_params(axis="y", labelleft=False)
    fig.subplots_adjust(left=0.18, right=0.99, bottom=0.18, top=0.88, wspace=0.26)
    save_bundle(fig, SUPP, "figS2_validation_hierarchy")


def supp_fig3() -> None:
    df = pd.read_csv(OLD / "data/nested_v1_full_pooled_metrics.csv", encoding="utf-8-sig")
    df = df[(df.scheme == "condition") & df.target.isin(["yield_strength_mpa", "delta_strength_mpa", "elongation_pct"])]
    rows=[]
    for target in df.target.unique():
        sub=df[df.target==target]
        for fs in ["composition","process","joint"]:
            best=sub[(sub.feature_set==fs)&sub.model.isin(["elasticnet","pls","xgb","gpr"])].sort_values("rmse").iloc[0]
            rows.append([target,fs.title(),best.rmse,"model"])
        for model,label in [("grade_mean","Grade mean"),("route_mean","Route mean")]:
            r=sub[(sub.feature_set=="metadata_baseline")&(sub.model==model)].iloc[0]
            rows.append([target,label,r.rmse,"baseline"])
    plot=pd.DataFrame(rows,columns=["target","label","rmse","kind"])
    fig=fig_mm(145,55);axes=fig.subplots(1,3,sharey=True)
    specs=[("yield_strength_mpa","YS (MPa)"),("delta_strength_mpa","Δσ (MPa)"),("elongation_pct","EL (percentage points)")]
    order=["Composition","Process","Joint","Grade mean","Route mean"]
    for ax,(target,title),letter in zip(axes,specs,"abc"):
        sub=plot[plot.target==target].set_index("label").loc[order]; y=np.arange(5)
        for i,(lab,r) in enumerate(sub.iterrows()):
            face=BLUE if r.kind=="model" else "white"; edge=BLUE if r.kind=="model" else GREY
            ax.scatter(r.rmse,i,s=22,facecolor=face,edgecolor=edge,lw=0.8)
            ax.text(r.rmse,i-0.24,f"{r.rmse:.2f}",ha="center",va="bottom",fontsize=5.8)
        ax.set_yticks(y,order);ax.invert_yaxis();ax.set_xlim(0,sub.rmse.max()*1.10)
        ax.set_xlabel("Grouped-CV RMSE");ax.set_title(title,loc="left",fontweight="bold");label_panel(ax,letter)
        if letter != "a":
            ax.tick_params(axis="y", labelleft=False)
    fig.subplots_adjust(left=0.18,right=0.99,bottom=0.18,top=0.88,wspace=0.30)
    save_bundle(fig,SUPP,"figS3_feature_increment")


def supp_fig4() -> None:
    df=pd.read_csv(OLD/"data/conditional_ale_by_grade.csv",encoding="utf-8-sig");df["grade"]=df.grade.astype(str)
    fig=fig_mm(145,58);axes=fig.subplots(1,3)
    specs=[("yield_strength_mpa","YS"),("delta_strength_mpa","Δσ"),("elongation_pct","EL")]
    for ax,(target,title),letter in zip(axes,specs,"abc"):
        sub=df[(df.target==target)&(df.feature=="specimen_thickness_mm")]
        for g in GRADES:
            q=sub[sub.grade==g].sort_values("bin_center")
            ax.plot(q.bin_center,q.ale,color=GRADE_COLOR[g],marker=GRADE_MARKER[g],ms=2.5,lw=0.9,label=g)
        ax.axhline(0,color=GREY,lw=0.7,ls="--");ax.set_xlabel("Thickness (mm)");ax.set_ylabel("Centered ALE")
        ax.set_title(title,loc="left",fontweight="bold");label_panel(ax,letter)
    axes[-1].legend(ncol=1,loc="best")
    save_bundle(fig,SUPP,"figS4_thickness_conditional_effects")


def supp_fig5() -> None:
    shap=pd.read_csv(OLD/"data/conditional_shap_stability.csv",encoding="utf-8-sig")
    perm=pd.read_csv(OLD/"data/elongation_gpr_permutation_summary.csv",encoding="utf-8-sig")
    fmap={"aging_time_h":"Aging time","specimen_thickness_mm":"Thickness","homogenization_time_h":"Homogenization time",
          "rolling_reduction_pct":"Rolling reduction","solution_time_h":"Solution time","homogenization_temp_c":"Homogenization T",
          "aging_temp_c":"Aging T","solution_temp_c":"Solution T","annealing_temp_c":"Annealing T","annealing_time_h":"Annealing time",
          "rolling_temp_c":"Rolling T","Zr_wt_pct":"Zr"}
    fig=fig_mm(145,62);axes=fig.subplots(1,3)
    for ax,target,title,letter in [(axes[0],"yield_strength_mpa","YS grouped SHAP","a"),(axes[1],"delta_strength_mpa","Δσ grouped SHAP","b")]:
        s=shap[shap.target==target].sort_values(["top5_frequency","mean_importance"],ascending=False).head(5).sort_values("mean_importance")
        y=np.arange(len(s));ax.barh(y,s.mean_importance,color=BLUE,height=0.62)
        ax.set_yticks(y,[fmap.get(x,x) for x in s.feature]);ax.set_xlabel("Mean |SHAP| (MPa)")
        ax.set_xlim(0, s.mean_importance.max()*1.28)
        for i,r in enumerate(s.itertuples()):ax.text(r.mean_importance,i,f" {r.top5_frequency:.0%}",va="center",fontsize=5.6)
        ax.set_title(title,loc="left",fontweight="bold");label_panel(ax,letter)
    ax=axes[2];s=perm[perm.mean_delta_rmse>0].sort_values("mean_delta_rmse",ascending=False).head(5).sort_values("mean_delta_rmse")
    y=np.arange(len(s));ax.barh(y,s.mean_delta_rmse,xerr=s.sd_delta_rmse,color="#6A93B5",height=0.62,
                               error_kw={"ecolor":GREY,"elinewidth":0.7,"capsize":1.5})
    ax.set_yticks(y,[fmap.get(x,x) for x in s.feature]);ax.set_xlabel("Permutation ΔRMSE")
    ax.set_xlim(min(-0.03, float((s.mean_delta_rmse-s.sd_delta_rmse).min())*1.08),
                float((s.mean_delta_rmse+s.sd_delta_rmse).max())*1.30)
    margin = float((s.mean_delta_rmse+s.sd_delta_rmse).max()) * 0.025
    for i,r in enumerate(s.itertuples()):
        ax.text(r.mean_delta_rmse+r.sd_delta_rmse+margin, i, f"{r.positive_fraction:.0%}",
                va="center", ha="left", fontsize=5.6)
    ax.set_title("EL grouped permutation",loc="left",fontweight="bold");label_panel(ax,"c")
    fig.subplots_adjust(left=0.18,right=0.99,bottom=0.16,top=0.88,wspace=0.90)
    save_bundle(fig,SUPP,"figS5_explanation_stability")


def supp_fig6(pred) -> None:
    cand=pd.read_csv(OLD/"data/robust_route_candidates.csv",encoding="utf-8-sig")
    cand=cand[cand.supported & cand.physically_feasible]
    fy=pd.read_csv(OLD/"data/robust_pareto_ys_el.csv",encoding="utf-8-sig")
    fu=pd.read_csv(OLD/"data/robust_pareto_uts_el.csv",encoding="utf-8-sig")
    fig=fig_mm(145,60);axes=fig.subplots(1,2)
    for ax,strength,front,ylabel,letter in [(axes[0],"lcb_ys",fy,"YS lower bound (MPa)","a"),(axes[1],"lcb_uts",fu,"UTS lower bound (MPa)","b")]:
        ax.scatter(cand.lcb_el,cand[strength],s=2,color=LIGHT_GREY,alpha=0.32,edgecolor="none",rasterized=True)
        ax.scatter(front.lcb_el,front[strength],s=10,color=BLUE,edgecolor="none")
        target="yield_strength_mpa" if strength=="lcb_ys" else "ultimate_tensile_strength_mpa"
        es=pred[pred.target==target].set_index("sample_id"); ee=pred[pred.target=="elongation_pct"].set_index("sample_id")
        ids=sorted(set(es.index)&set(ee.index)); x=[ee.loc[i,"interval_low"] for i in ids]; y=[es.loc[i,"interval_low"] for i in ids]
        ax.scatter(x,y,s=28,marker="*",color=INK,zorder=5)
        for sid,xx,yy in zip(ids,x,y):ax.text(xx+0.15,yy,sid,fontsize=5.4,va="center")
        ax.set_xlabel("EL lower bound (%)");ax.set_ylabel(ylabel);label_panel(ax,letter)
    handles=[Line2D([0],[0],marker="o",lw=0,color=LIGHT_GREY,label="Supported candidates",ms=3),
             Line2D([0],[0],marker="o",lw=0,color=BLUE,label="Robust front",ms=3),
             Line2D([0],[0],marker="*",lw=0,color=INK,label="Trial lower bounds",ms=5)]
    axes[1].legend(handles=handles,loc="best")
    save_bundle(fig,SUPP,"figS6_support_domain_sensitivity")


def main() -> None:
    setup()
    train, ext, desc, state, design, pred = read_data()
    source_tables(train, ext, desc, state, design, pred)
    supp_fig1()
    supp_fig2()
    supp_fig3()
    supp_fig4()
    supp_fig5()
    supp_fig6(pred)
    print({"supplementary_figures": 6, "backend": "python"})


if __name__=="__main__":
    main()
