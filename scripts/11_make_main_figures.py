#!/usr/bin/env python3
"""Rebuild the seven main figures around a materials-science evidence chain.

Python/matplotlib is the exclusive backend.  All 284 modelling observations and
all four post-model trial observations are retained; no aesthetic sampling is
used.  The descriptor stability analysis is read from the v2 audit outputs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter, MaxNLocator


from release_paths import PRIVATE_ROOT as AL, PAPER_OUTPUT as OUT, SCIENTIFIC_OUTPUT as OLD
FIG = OUT / "figures/main"
DATA = OUT / "data_materials_v2"
GRADES = ["2024", "5083", "6082", "7075"]
C = {"2024": "#C58E12", "5083": "#2E6B9A", "6082": "#78872B", "7075": "#AA5578"}
L = {"2024": "#E9D6A7", "5083": "#B9D1E2", "6082": "#CDD4A8", "7075": "#DFC0CD"}
M = {"2024": "o", "5083": "s", "6082": "^", "7075": "D"}
INK = "#2B2B2B"
GREY = "#737373"
LIGHT = "#D6D6D6"
BLUE = "#2E6B9A"
MM = 1 / 25.4
FIG_WIDTH_MM = 145

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7.2,
    "axes.labelsize": 7.2,
    "axes.titlesize": 7.6,
    "xtick.labelsize": 6.4,
    "ytick.labelsize": 6.4,
    "legend.fontsize": 5.8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.edgecolor": INK,
})


def figmm(height_mm: float):
    return plt.figure(figsize=(FIG_WIDTH_MM * MM, height_mm * MM))


def label(ax, letter: str):
    ax.text(-0.12, 1.05, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=8.2, ha="left", va="bottom", zorder=20)


def save(fig, stem: str):
    FIG.mkdir(parents=True, exist_ok=True)
    base = FIG / stem
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pad_inches=0.02, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)


def load():
    tr = pd.read_csv(AL / "data/01_processed_data.csv", encoding="utf-8-sig")
    ex = pd.read_csv(AL / "data/verification_holdout.csv", encoding="utf-8-sig")
    for d in (tr, ex):
        d["grade"] = d.alloy_or_coating_name.astype(str)
        d["delta_strength_mpa"] = d.ultimate_tensile_strength_mpa - d.yield_strength_mpa
    state = pd.read_csv(OUT / "data/fig3_state_responses.csv", encoding="utf-8-sig")
    state["grade"] = state.grade.astype(str)
    pred = pd.read_csv(OUT / "data/fig5_trial_predictions.csv", encoding="utf-8-sig")
    pred["grade"] = pred.grade.astype(str)
    primary = pd.read_csv(DATA / "descriptor_primary_panel.csv", encoding="utf-8-sig")
    primary["grade"] = primary.grade.astype(str)
    library = pd.read_csv(DATA / "descriptor_library.csv", encoding="utf-8-sig")
    identity = pd.read_csv(OLD / "data/grade_identity_diagnostic.csv", encoding="utf-8-sig")
    variance = pd.read_csv(OLD / "data/variance_decomposition_corrected.csv", encoding="utf-8-sig")
    ale = pd.read_csv(OLD / "data/conditional_ale_by_grade.csv", encoding="utf-8-sig")
    ale["grade"] = ale.grade.astype(str)
    candidates = pd.read_csv(OLD / "data/robust_route_candidates.csv", encoding="utf-8-sig")
    candidates["grade"] = candidates.grade.astype(str)
    pareto_ys = pd.read_csv(OLD / "data/robust_pareto_ys_el.csv", encoding="utf-8-sig")
    pareto_uts = pd.read_csv(OLD / "data/robust_pareto_uts_el.csv", encoding="utf-8-sig")
    for d in (pareto_ys, pareto_uts):
        d["grade"] = d.grade.astype(str)
    assert len(tr) == 284 and len(ex) == 4 and len(pred) == 12
    return tr, ex, state, pred, primary, library, identity, variance, ale, candidates, pareto_ys, pareto_uts


def normalized_rows(frame: pd.DataFrame) -> np.ndarray:
    arr = frame.to_numpy(float)
    lo = np.nanmin(arr, axis=0)
    hi = np.nanmax(arr, axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return (arr - lo) / span


def fig1(tr, ex, identity, variance):
    fig = figmm(102)
    gs = fig.add_gridspec(2, 6, width_ratios=[0.95, 0.95, 1.25, 1.25, 1.10, 1.10],
                          hspace=0.58, wspace=0.72)
    raw = tr.groupby("grade")[["Mg_wt_pct", "Cu_wt_pct", "Si_wt_pct", "Zn_wt_pct"]].median().reindex(GRADES)
    raw["Mn+Cr"] = tr.assign(MnCr=tr.Mn_wt_pct + tr.Cr_wt_pct).groupby("grade").MnCr.median().reindex(GRADES)
    ax = fig.add_subplot(gs[:, 0:2]); label(ax, "a")
    im = ax.imshow(normalized_rows(raw), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    labs = ["Mg", "Cu", "Si", "Zn", "Mn+Cr"]
    ax.set_xticks(range(5), labs, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(4), GRADES)
    for i in range(4):
        for j in range(5):
            v = raw.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.7,
                    color="white" if im.cmap(im.norm(normalized_rows(raw)[i, j]))[0] < 0.45 else INK)
    ax.set_xlabel("Median composition (wt.%)")
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)

    ax = fig.add_subplot(gs[0, 3:6]); label(ax, "b")
    label_offsets = {"2024": (-12, 16), "5083": (-4, 16), "6082": (10, 20), "7075": (12, 14)}
    for g in GRADES:
        q = tr[tr.grade == g]
        e = ex[ex.grade == g]
        sizes = 9 + 1.25 * q.elongation_pct.to_numpy(float)
        ax.scatter(q.yield_strength_mpa, q.delta_strength_mpa, s=sizes,
                   color=C[g], alpha=0.48, marker=M[g], edgecolor="none",
                   rasterized=True, zorder=2)
        ax.scatter(e.yield_strength_mpa, e.delta_strength_mpa, s=62,
                   color=C[g], marker="*", edgecolor=INK, linewidth=0.7, zorder=8)
        x, y = q.yield_strength_mpa.median(), q.delta_strength_mpa.median()
        dx, dy = label_offsets[g]
        ann = ax.annotate(g, (x, y), xytext=(dx, dy), textcoords="offset points",
                          color=C[g], weight="bold", fontsize=6.8,
                          ha="center", va="center", zorder=12)
        ann.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])
    ax.set_xlabel("YS (MPa)")
    ax.set_ylabel("UTS - YS (MPa)")
    ax.text(0.98, 0.96, "Point size = EL", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.8, color=GREY, zorder=15)

    ax = fig.add_subplot(gs[1, 2:4]); label(ax, "c")
    props = [("yield_strength_mpa", "YS"), ("delta_strength_mpa", "UTS-YS"), ("elongation_pct", "EL")]
    pooled = tr[[p for p, _ in props]]
    mu, sd = pooled.mean(), pooled.std(ddof=0)
    y = np.arange(3)
    for off, g in zip(np.linspace(-0.24, 0.24, 4), GRADES):
        med = tr[tr.grade == g][[p for p, _ in props]].median()
        z = ((med - mu) / sd).to_numpy(float)
        ax.scatter(z, y + off, s=19, color=C[g], marker=M[g], label=g)
    ax.axvline(0, color=GREY, lw=0.7, ls="--")
    ax.set_yticks(y, [x[1] for x in props])
    ax.invert_yaxis()
    ax.set_xlabel("Standardized grade median")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.27),
              columnspacing=0.6, handletextpad=0.2)

    ax = fig.add_subplot(gs[1, 4:6]); label(ax, "d")
    order = ["yield_strength_mpa", "ultimate_tensile_strength_mpa", "elongation_pct"]
    v = variance.set_index("target").loc[order]
    x = np.arange(3)
    between = v.between_fraction_eta2.to_numpy(float)
    within = v.within_fraction.to_numpy(float)
    ax.bar(x, between, color=BLUE, width=0.58, label="Between grade")
    ax.bar(x, within, bottom=between, color="#C7D9E8", width=0.58, label="Within grade")
    ax.set_xticks(x, ["YS", "UTS", "EL"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Variance fraction")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2, columnspacing=0.7)
    for xi, b in zip(x, between):
        ax.text(xi, b / 2, f"{b:.2f}", color="white", ha="center", va="center", fontsize=5.8)

    fig.subplots_adjust(left=0.10, right=0.995, top=0.96, bottom=0.14)
    save(fig, "fig1_hierarchical_material_space")
    raw.reset_index().to_csv(DATA / "fig1_raw_composition_medians.csv", index=False, encoding="utf-8-sig")


def fig2(primary, library):
    fig = figmm(110)
    gs = fig.add_gridspec(2, 2, width_ratios=[0.82, 1.35],
                          height_ratios=[1.0, 1.25], hspace=0.42, wspace=0.34)

    ax = fig.add_subplot(gs[0, 0]); label(ax, "a")
    fam = library.groupby("family").size()
    counts = pd.Series({
        "Raw + molar": int(fam.get("raw composition", 0) + fam.get("molar composition", 0)),
        "Systematic pairs": int(fam.filter(like="systematic pair").sum()),
        "Phase stoichiometry": int(fam.filter(like="phase-stoichiometric").sum()),
        "Aggregate loads": int(fam.get("aggregate load", 0)),
    })
    y = np.arange(len(counts))
    ax.barh(y, counts.values, color=[BLUE, "#8AA9C2", "#C58E12", "#78872B"], height=0.58)
    ax.set_yticks(y, counts.index)
    ax.invert_yaxis()
    ax.set_xlabel("Candidate descriptors")
    for yi, v in zip(y, counts.values):
        ax.text(v + 1, yi, str(v), va="center", fontsize=6)
    ax.set_xlim(0, max(counts.values) * 1.20)

    descriptors = [
        ("wt_Mg", "Mg"),
        ("phasecap_S_Al2CuMg", "Al2CuMg cap."),
        ("phasecap_beta_Mg2Si", "Mg2Si cap."),
        ("phasecap_eta_MgZn2", "MgZn2 cap."),
        ("dispersoid_MnCrZr_wt", "Mn+Cr+Zr"),
        ("impurity_FeSi_wt", "Fe+Si"),
    ]
    ax = fig.add_subplot(gs[0, 1]); label(ax, "b")
    piv = primary.pivot(index="grade", columns="descriptor", values="branch_scaled_0_1").reindex(GRADES)
    arr = piv[[x[0] for x in descriptors]].to_numpy(float)
    im = ax.imshow(arr, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(descriptors)), [x[1] for x in descriptors],
                  rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(4), GRADES)
    ax.set_title("Between-grade composition indicators", loc="left")
    for i in range(4):
        for j in range(len(descriptors)):
            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=5.6,
                    color="white" if arr[i, j] > 0.58 else INK)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)

    ax = fig.add_subplot(gs[1, 0]); label(ax, "c")
    score = primary.pivot(index="grade", columns="descriptor", values="within_grade_score").reindex(GRADES)
    ss = score[[x[0] for x in descriptors]].to_numpy(float)
    im2 = ax.imshow(ss, cmap="YlGnBu", vmin=0, vmax=max(0.5, float(np.nanmax(ss))), aspect="auto")
    passes = primary.pivot(index="grade", columns="descriptor", values="within_grade_pass").reindex(GRADES)
    pp = passes[[x[0] for x in descriptors]].to_numpy(bool)
    ax.set_xticks(range(len(descriptors)), [x[1] for x in descriptors],
                  rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(4), GRADES)
    ax.set_title("Within-grade stability score", loc="left", fontsize=7.2)
    for i in range(4):
        for j in range(len(descriptors)):
            txt = f"{ss[i, j]:.2f}" + ("*" if pp[i, j] else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.4,
                    color="white" if ss[i, j] > 0.32 else INK)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)

    ax = fig.add_subplot(gs[1, 1]); ax.axis("off"); label(ax, "d")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.12, 1.12)
    steps = [
        ("284 modelling samples", "full elemental + process fields"),
        ("Comparative evidence", "between-grade indicators / within-grade conditions"),
        ("Candidate selection", "target + data distance + manufacturability"),
        ("4 alloy trials", "new melting, processing and tensile tests"),
    ]
    ys = [0.96, 0.64, 0.32, 0.00]
    for i, ((title, body), y0) in enumerate(zip(steps, ys)):
        ax.add_patch(FancyBboxPatch((0.06, y0 - 0.09), 0.88, 0.18,
                                    boxstyle="round,pad=0.01,rounding_size=0.015",
                                    fc="#F4F4F4" if i < 3 else "#F7EDF1",
                                    ec=GREY if i < 3 else C["7075"], lw=0.9))
        ax.text(0.10, y0 + 0.025, title, weight="bold", va="center", fontsize=6.2)
        ax.text(0.10, y0 - 0.040, body, color=GREY, va="center", fontsize=5.6)
        if i < 3:
            ax.add_patch(FancyArrowPatch((0.50, y0 - 0.105), (0.50, ys[i + 1] + 0.105),
                                         arrowstyle="-|>", mutation_scale=8, color=GREY, lw=0.8))
    fig.subplots_adjust(left=0.12, right=0.99, top=0.95, bottom=0.14)
    save(fig, "fig2_descriptor_screen_and_design_chain")


def branch_descriptor(d, grade):
    aw = {"Cu": 63.546, "Mg": 24.305, "Si": 28.085, "Zn": 65.38}
    if grade == "2024":
        return np.minimum(10 * d.Cu_wt_pct / aw["Cu"], 10 * d.Mg_wt_pct / aw["Mg"]), "Al2CuMg capacity (mol kg⁻¹)"
    if grade == "5083":
        return d.Mg_wt_pct, "Mg (wt.%)"
    if grade == "6082":
        return np.minimum(10 * d.Mg_wt_pct / (2 * aw["Mg"]), 10 * d.Si_wt_pct / aw["Si"]), "Mg2Si capacity (mol kg⁻¹)"
    return np.minimum(10 * d.Mg_wt_pct / aw["Mg"], 10 * d.Zn_wt_pct / (2 * aw["Zn"])), "MgZn2 capacity (mol kg⁻¹)"


def state_series(d, grade):
    if grade in ("2024", "6082"):
        return d.solution_time_h.map(lambda x: f"Solution {x:g} h")
    if grade == "5083":
        return pd.Series(np.where(d.annealing_temp_c.fillna(0) > 0,
                                  "Cold-roll + anneal", "Hot-roll route"), index=d.index)
    return d.rolling_reduction_pct.map(lambda x: f"Reduction {x:g}%")


def fig3(tr, ex):
    fig = figmm(92)
    axes = fig.subplots(2, 2)
    cmap = mpl.colormaps["cividis"]
    norm = mpl.colors.Normalize(tr.elongation_pct.min(), tr.elongation_pct.max())
    markers = ["o", "s", "^", "D", "P", "X"]
    out = []
    for ax, grade, letter in zip(axes.ravel(), GRADES, "abcd"):
        q = tr[tr.grade == grade].copy()
        x, xlab = branch_descriptor(q, grade)
        q["descriptor"] = x
        q["state"] = state_series(q, grade)
        states = list(dict.fromkeys(q.state.astype(str)))
        for marker, state in zip(markers, states):
            z = q[q.state == state]
            ax.scatter(z.descriptor, z.yield_strength_mpa, c=z.elongation_pct,
                       cmap=cmap, norm=norm, s=16, marker=marker, alpha=0.72,
                       edgecolor="none", label=state, rasterized=True)
        e = ex[ex.grade == grade].copy()
        xe, _ = branch_descriptor(e, grade)
        ax.scatter(xe, e.yield_strength_mpa, c=e.elongation_pct, cmap=cmap,
                   norm=norm, s=60, marker="*", edgecolor=INK, lw=0.7, zorder=8)
        rho_ys = q.descriptor.corr(q.yield_strength_mpa, method="spearman")
        rho_el = q.descriptor.corr(q.elongation_pct, method="spearman")
        ax.text(0.98, 0.96, f"rho(YS)={rho_ys:+.2f}\nrho(EL)={rho_el:+.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=5.6,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
        ax.set_xlabel(xlab)
        ax.set_ylabel("YS (MPa)")
        ax.set_title(grade, loc="left", color=C[grade], weight="bold")
        label(ax, letter)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.3f}".rstrip("0").rstrip(".")))
        legend_loc = "lower right" if grade in ("2024", "5083") else "best"
        leg = ax.legend(loc=legend_loc, fontsize=5.0, handletextpad=0.2, labelspacing=0.22, frameon=True, facecolor="white", framealpha=0.82, edgecolor="#D9D9D9")
        leg.set_zorder(15)
        out.append(q[["sample_id", "grade", "state", "descriptor",
                      "yield_strength_mpa", "elongation_pct", "delta_strength_mpa"]])
    fig.subplots_adjust(left=0.12, right=0.87, bottom=0.10, top=0.94, wspace=0.38, hspace=0.40)
    cax = fig.add_axes([0.90, 0.20, 0.018, 0.60])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("EL (%)")
    save(fig, "fig3_branch_chemistry_state_property")
    pd.concat(out).to_csv(DATA / "fig3_branch_points_v2.csv", index=False, encoding="utf-8-sig")


def binned(q, xcol, ycol, n=6):
    z = q[[xcol, ycol]].copy()
    z["bin"] = pd.qcut(z[xcol], q=min(n, z[xcol].nunique()), duplicates="drop")
    return z.groupby("bin", observed=True).agg(
        x=(xcol, "median"), y=(ycol, "median"), n=(ycol, "size")
    ).reset_index(drop=True)


def fig4(tr, ale):
    fig = figmm(100)
    axes = fig.subplots(2, 3)
    specs = [
        ("yield_strength_mpa", "YS (MPa)"),
        ("delta_strength_mpa", "UTS - YS (MPa)"),
        ("elongation_pct", "EL (%)"),
    ]
    for ax, (target, ylab), letter in zip(axes[0], specs, "abc"):
        for grade in GRADES:
            q = tr[tr.grade == grade]
            ax.scatter(q.specimen_thickness_mm, q[target], s=9, color=C[grade],
                       alpha=0.55, edgecolor="none", rasterized=True)
            b = binned(q, "specimen_thickness_mm", target)
            ax.plot(b.x, b.y, color=C[grade], marker=M[grade], ms=3.0, lw=1.0, label=grade)
            ymin, ymax = ax.get_ylim()
            ax.vlines(q.specimen_thickness_mm, ymin, ymin + (ymax - ymin) * 0.018,
                      color=C[grade], alpha=0.22, lw=0.4)
        ax.set_xlabel("Thickness (mm)")
        ax.set_ylabel(ylab)
        label(ax, letter)
    for ax, (target, ylab), letter in zip(axes[1], specs, "def"):
        q = ale[(ale.feature == "specimen_thickness_mm") & (ale.target == target)]
        for grade in GRADES:
            z = q[q.grade == grade].sort_values("bin_center")
            ax.plot(z.bin_center, z.ale, marker=M[grade], color=C[grade],
                    lw=1.0, ms=3.0, label=grade)
        ax.axhline(0, color=GREY, lw=0.7, ls="--")
        ax.set_xlabel("Thickness (mm)")
        ax.set_ylabel(f"ALE of {ylab}")
        label(ax, letter)
    handles = [Line2D([0], [0], color=C[g], marker=M[g], lw=1, label=g, ms=3.5) for g in GRADES]
    fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.54, 0.995),
               columnspacing=0.9, handletextpad=0.3)
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.10, top=0.91, wspace=0.55, hspace=0.46)
    save(fig, "fig4_thickness_raw_and_ale")


def nondominated(df, strength):
    arr = df[["elongation_pct", strength]].to_numpy(float)
    keep = []
    for i, p in enumerate(arr):
        keep.append(not np.any((arr[:, 0] >= p[0]) & (arr[:, 1] >= p[1]) &
                               ((arr[:, 0] > p[0]) | (arr[:, 1] > p[1]))))
    return df.loc[keep]


def trial_rank_counts(tr, ex):
    rows = []
    targets = ["yield_strength_mpa", "ultimate_tensile_strength_mpa", "elongation_pct"]
    for grade in GRADES:
        base = tr[tr.grade == grade]
        trial = ex[ex.grade == grade].iloc[0]
        for target in targets:
            count = int((base[target] <= trial[target]).sum())
            rows.append({
                "grade": grade, "sample_id": trial.sample_id, "target": target,
                "n_at_or_below": count, "n_modelling": len(base),
                "fraction_at_or_below": count / len(base),
            })
    return pd.DataFrame(rows)


def nondominated_fast(df, xcol, ycol):
    """Return an exact 2D upper-right skyline without quadratic comparisons."""
    s = df.sort_values([xcol, ycol], ascending=[False, False]).copy()
    previous = s[ycol].cummax().shift(fill_value=-np.inf)
    keep = s[ycol] > previous
    return s.loc[keep].sort_values(xcol)


def fig5(tr, ex, candidates, pareto_ys, pareto_uts):
    fig = figmm(108)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.05], hspace=0.48, wspace=0.82)
    for k, (strength, ylab) in enumerate([
        ("yield_strength_mpa", "YS (MPa)"),
        ("ultimate_tensile_strength_mpa", "UTS (MPa)"),
    ]):
        ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
        label(ax, "ab"[k])
        for grade in GRADES:
            q = tr[tr.grade == grade]
            fr = nondominated(q, strength)
            e = ex[ex.grade == grade]
            ax.scatter(q.elongation_pct, q[strength], s=9, color=C[grade],
                       alpha=0.65, edgecolor="none", rasterized=True)
            ax.scatter(fr.elongation_pct, fr[strength], s=22, marker=M[grade],
                       facecolor="white", edgecolor=C[grade], lw=0.9, zorder=5)
            ax.scatter(e.elongation_pct, e[strength], s=48, marker="*",
                       color=C[grade], edgecolor=INK, lw=0.6, zorder=8)
        ax.set_xlabel("EL (%)")
        ax.set_ylabel(ylab)

    ranks = trial_rank_counts(tr, ex)
    ranks.to_csv(DATA / "trial_within_grade_rank_counts.csv", index=False, encoding="utf-8-sig")
    ax = fig.add_subplot(gs[0, 4:6]); label(ax, "c")
    order = ["yield_strength_mpa", "ultimate_tensile_strength_mpa", "elongation_pct"]
    matrix = ranks.pivot(index="grade", columns="target", values="fraction_at_or_below").reindex(GRADES)[order]
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3), ["YS", "UTS", "EL"])
    ax.set_yticks(range(4), GRADES)
    ax.set_title("Trial position within grade", loc="left", fontsize=7.1)
    lookup = ranks.set_index(["grade", "target"])
    for i, grade in enumerate(GRADES):
        for j, target in enumerate(order):
            r = lookup.loc[(grade, target)]
            ax.text(j, i, f"{int(r.n_at_or_below)}/{int(r.n_modelling)}",
                    ha="center", va="center", fontsize=6.0,
                    color="white" if matrix.iloc[i, j] > 0.65 else INK)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)

    all_supported = candidates[candidates.supported & candidates.physically_feasible]
    for ax, strength, pareto, letter, ylab in [
        (fig.add_subplot(gs[1, 0:3]), "lcb_ys", pareto_ys, "d", "Lower-bound YS (MPa)"),
        (fig.add_subplot(gs[1, 3:6]), "lcb_uts", pareto_uts, "e", "Lower-bound UTS (MPa)"),
    ]:
        label(ax, letter)
        for grade in GRADES:
            q = all_supported[all_supported.grade == grade]
            p = nondominated_fast(q, "lcb_el", strength)
            ax.scatter(q.lcb_el, q[strength], s=3, color=L[grade], alpha=0.10,
                       edgecolor="none", rasterized=True)
            ax.scatter(p.lcb_el, p[strength], s=17, facecolor="white",
                       edgecolor=C[grade], marker=M[grade], lw=0.8, zorder=5)
        ax.set_xlabel("Lower-bound EL (%)")
        ax.set_ylabel(ylab)
    handles = [Line2D([0], [0], marker=M[g], color=C[g], lw=0, label=g, ms=4) for g in GRADES]
    fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.53, 0.995),
               columnspacing=0.8, handletextpad=0.25)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.09, top=0.91)
    save(fig, "fig5_observed_and_supported_design_fronts")


def raw_state(d, grade):
    return state_series(d, grade)


def order_states(grade, values):
    if grade == "5083":
        return [x for x in ["Hot-roll route", "Cold-roll + anneal"] if x in values]
    import re
    def num(s):
        m = re.search(r"\d+(?:\.\d+)?", s)
        return float(m.group()) if m else 0
    return sorted(values, key=num)


def fig6(tr, state):
    fig = figmm(100)
    axes = fig.subplots(2, 2, sharex=True)
    targets = [
        ("yield_strength_mpa", "YS", "o", "#2E6B9A"),
        ("delta_strength_mpa", "UTS-YS", "s", "#C58E12"),
        ("elongation_pct", "EL", "^", "#78872B"),
    ]
    for ax, grade, letter in zip(axes.ravel(), GRADES, "abcd"):
        q = tr[tr.grade == grade].copy()
        q["state"] = raw_state(q, grade)
        states = order_states(grade, q.state.unique().tolist())
        y = np.arange(len(states))
        for off, (target, lab, marker, color) in zip([-.20, 0, .20], targets):
            mu, sd = q[target].mean(), q[target].std(ddof=0)
            q["z"] = (q[target] - mu) / sd
            for j, s in enumerate(states):
                z = q[q.state == s].z.to_numpy()
                jitter = np.linspace(-.025, .025, len(z)) if len(z) > 1 else np.zeros(len(z))
                ax.scatter(z, j + off + jitter, s=7, color=color, alpha=0.36,
                           edgecolor="none", rasterized=True)
            ss = state[(state.grade == grade) & (state.target == target)].set_index("state").reindex(states)
            mean = ss.mean_z.to_numpy()
            lo = ss.ci_low.to_numpy()
            hi = ss.ci_high.to_numpy()
            ax.errorbar(mean, y + off, xerr=np.vstack([mean - lo, hi - mean]),
                        fmt=marker, color=color, ecolor=color, ms=4.0,
                        elinewidth=0.8, capsize=1.5, label=lab, zorder=5)
        counts = state[(state.grade == grade) & (state.target == "yield_strength_mpa")].set_index("state").reindex(states).n_condition_groups.astype(int)
        shown = [s.replace("Solution ", "Sol. ").replace("Reduction ", "Red. ") for s in states]
        ax.set_yticks(y, [f"{s}  n={n}" for s, n in zip(shown, counts)])
        ax.invert_yaxis()
        ax.axvline(0, color=GREY, lw=0.7, ls="--")
        ax.set_xlim(-3.2, 3.2)
        ax.set_title(grade, loc="left", color=C[grade], weight="bold")
        label(ax, letter)
    handles = [Line2D([0], [0], marker=x[2], color=x[3], lw=0, label=x[1], ms=4) for x in targets]
    fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.53, 0.995),
               columnspacing=1.0, handletextpad=0.3)
    fig.text(0.58, 0.025, "Within-grade standardized response",
             ha="center", va="center", fontsize=7.2)
    fig.subplots_adjust(left=0.20, right=0.99, bottom=0.12, top=0.91, hspace=0.36, wspace=0.62)
    save(fig, "fig6_process_state_partition")


def fig7(pred):
    fig = figmm(96)
    gs = fig.add_gridspec(2, 6, height_ratios=[1, .86], hspace=.58, wspace=.92)
    specs = [
        ("yield_strength_mpa", "YS (MPa)", "YS", "MPa"),
        ("ultimate_tensile_strength_mpa", "UTS (MPa)", "UTS", "MPa"),
        ("elongation_pct", "EL (%)", "EL", "%"),
    ]
    for k, (target, labeltxt, title, unit) in enumerate(specs):
        ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
        label(ax, "abc"[k])
        q = pred[pred.target == target].set_index("grade").loc[GRADES].reset_index()
        for r in q.itertuples():
            ax.errorbar(r.measured, r.predicted,
                        yerr=[[r.predicted - r.interval_low], [r.interval_high - r.predicted]],
                        fmt=M[r.grade], color=C[r.grade], ecolor=L[r.grade],
                        ms=4.2, elinewidth=1.2, capsize=2)
        lo = min(q.measured.min(), q.interval_low.min())
        hi = max(q.measured.max(), q.interval_high.max())
        ax.plot([lo, hi], [lo, hi], ls="--", color=GREY, lw=.8)
        ax.set_xlabel(f"Measured {labeltxt}")
        ax.set_title(title, loc="center", weight="bold")
        ax.text(.04, .94, f"RMSE = {q.four_sample_rmse.iloc[0]:.2f} {unit}",
                transform=ax.transAxes, ha="left", va="top", fontsize=5.8)
    fig.text(.015, .73, "Frozen prediction", rotation=90, rotation_mode="anchor",
             va="center", ha="center", fontsize=7.2)

    for i, (col, title, lim, letter) in enumerate([
        ("relative_error_pct", "Relative error (%)", 20, "d"),
        ("error", "Residual / CV-RMSE", 1.6, "e"),
    ]):
        ax = fig.add_subplot(gs[1, 3 * i:3 * i + 3])
        label(ax, letter)
        matrix = np.zeros((4, 3))
        for gi, grade in enumerate(GRADES):
            for ti, (target, _, _, _) in enumerate(specs):
                r = pred[(pred.grade == grade) & (pred.target == target)].iloc[0]
                matrix[gi, ti] = r.relative_error_pct if col == "relative_error_pct" else r.error / r.cv_rmse
        im = ax.imshow(matrix, cmap="PuOr", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(3), [x[2] for x in specs])
        ax.set_yticks(range(4), GRADES)
        ax.set_title(title, loc="left", fontsize=7)
        for y in range(4):
            for x in range(3):
                ax.text(x, y, f"{matrix[y, x]:+.1f}" if col == "relative_error_pct" else f"{matrix[y, x]:+.2f}",
                        ha="center", va="center", fontsize=5.7)
        fig.colorbar(im, ax=ax, fraction=.035, pad=.025)
    fig.subplots_adjust(left=.09, right=.99, bottom=.08, top=.94)
    save(fig, "fig7_trial_feedback")


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    (tr, ex, state, pred, primary, library, identity, variance, ale,
     candidates, pareto_ys, pareto_uts) = load()
    fig1(tr, ex, identity, variance)
    fig2(primary, library)
    fig3(tr, ex)
    fig4(tr, ale)
    fig5(tr, ex, candidates, pareto_ys, pareto_uts)
    fig6(tr, state)
    fig7(pred)
    manifest = {
        "backend": "Python/matplotlib",
        "n_modelling": len(tr),
        "n_trials": len(ex),
        "main_figures": 7,
        "panel_counts": {"fig1": 4, "fig2": 4, "fig3": 4, "fig4": 6,
                         "fig5": 5, "fig6": 4, "fig7": 5},
        "formats": ["svg", "pdf", "tiff", "png"],
        "exclusions": "none",
        "final_width_mm": FIG_WIDTH_MM,
    }
    (DATA / "figure_manifest_v2.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
