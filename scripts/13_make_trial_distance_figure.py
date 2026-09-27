#!/usr/bin/env python3
"""Rebuild Supplementary Fig. S6 with trial labels above all data layers."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


from release_paths import PRIVATE_ROOT as AL, PAPER_OUTPUT as OUT, SCIENTIFIC_OUTPUT as OLD
FIG = OUT / "figures/supplementary"
DATA = OUT / "data"
MM = 1 / 25.4
INK = "#2B2B2B"
BLUE = "#356C9A"
LIGHT = "#D8D8D8"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7.2,
    "axes.labelsize": 7.2,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 5.8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def save(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    base = FIG / stem
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pad_inches=0.02, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)


def panel_label(ax, text):
    ax.text(-0.12, 1.05, text, transform=ax.transAxes, fontweight="bold",
            fontsize=8.2, ha="left", va="bottom", zorder=30)


def main():
    cand = pd.read_csv(OLD / "data/robust_route_candidates.csv", encoding="utf-8-sig")
    cand = cand[cand.supported & cand.physically_feasible]
    front_ys = pd.read_csv(OLD / "data/robust_pareto_ys_el.csv", encoding="utf-8-sig")
    front_uts = pd.read_csv(OLD / "data/robust_pareto_uts_el.csv", encoding="utf-8-sig")
    pred = pd.read_csv(DATA / "fig5_trial_predictions.csv", encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(145 * MM, 62 * MM))
    offsets = {
        "al_150": (5, 8),
        "al_191": (5, -9),
        "al_243": (5, 8),
        "al_48": (5, 9),
    }
    for ax, strength, front, ylabel, letter in [
        (axes[0], "lcb_ys", front_ys, "YS lower bound (MPa)", "a"),
        (axes[1], "lcb_uts", front_uts, "UTS lower bound (MPa)", "b"),
    ]:
        ax.scatter(cand.lcb_el, cand[strength], s=2, color=LIGHT, alpha=0.32,
                   edgecolor="none", rasterized=True, zorder=1)
        ax.scatter(front.lcb_el, front[strength], s=10, color=BLUE,
                   edgecolor="none", zorder=4)
        target = "yield_strength_mpa" if strength == "lcb_ys" else "ultimate_tensile_strength_mpa"
        strength_rows = pred[pred.target == target].set_index("sample_id")
        el_rows = pred[pred.target == "elongation_pct"].set_index("sample_id")
        ids = sorted(set(strength_rows.index) & set(el_rows.index))
        xs = [el_rows.loc[sid, "interval_low"] for sid in ids]
        ys = [strength_rows.loc[sid, "interval_low"] for sid in ids]
        ax.scatter(xs, ys, s=30, marker="*", color=INK, zorder=16)
        for sid, x, y in zip(ids, xs, ys):
            dx, dy = offsets[sid]
            ax.annotate(
                sid, (x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=5.6, va="center", ha="left", zorder=25,
                bbox={"facecolor": "white", "alpha": 0.82,
                      "edgecolor": "none", "pad": 0.5},
            )
        ax.set_xlabel("EL lower bound (%)")
        ax.set_ylabel(ylabel)
        panel_label(ax, letter)
    handles = [
        Line2D([0], [0], marker="o", lw=0, color=LIGHT,
               label="Retained candidates", ms=3),
        Line2D([0], [0], marker="o", lw=0, color=BLUE,
               label="Non-dominated combinations", ms=3),
        Line2D([0], [0], marker="*", lw=0, color=INK,
               label="Trial lower bounds", ms=5),
    ]
    axes[1].legend(handles=handles, loc="best", frameon=True,
                   facecolor="white", framealpha=0.82, edgecolor="#D9D9D9")
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.17, top=0.93, wspace=0.36)
    save(fig, "figS6_candidate_distance_filter")


if __name__ == "__main__":
    main()
