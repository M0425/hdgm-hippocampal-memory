"""Figure 5C redraw: condition-wise seed point-range matrix, without lines."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


DATA = Path(r"C:\Users\马明哲\Desktop\建模\pythonProject32\hdgm_difficulty_v2_1_frozen_outputs_confirmatory\difficulty_v2_1_multiseed_results.json")
OUT = Path(__file__).with_name("figure5C_conditionwise_seed_matrix.png")

INK, MUTED, GRID = "#14213D", "#718096", "#D9E3EF"
HDGM, TEM, STATELESS = "#0072B2", "#475569", "#94A3B8"
ALIAS_TINT, NOISE_TINT = "#FDECEF", "#E9F8F2"


def load():
    with DATA.open(encoding="utf-8") as handle:
        return json.load(handle)["records"]


def values(records, condition, model):
    return np.array([record["conditions"][condition]["behavior"][model] for record in records])


def point_range(ax, x, vals, color, marker):
    """Raw seed observations plus mean and two-sided 95% t interval."""
    vals = np.asarray(vals)
    mean = vals.mean()
    ci = 2.262 * vals.std(ddof=1) / np.sqrt(len(vals))
    # Deterministic symmetric offsets retain every seed without implying order.
    jitter = np.linspace(-.065, .065, len(vals))
    ax.scatter(x + jitter, vals, s=18, marker=marker, color=color, alpha=.42,
               linewidths=.35, edgecolors="white", zorder=3)
    ax.vlines(x, mean - ci, mean + ci, color=color, lw=1.5, zorder=4)
    ax.hlines([mean - ci, mean + ci], x - .043, x + .043, color=color, lw=1.1, zorder=4)
    ax.scatter(x, mean, s=53, marker=marker, facecolor="white", edgecolor=color,
               linewidth=1.9, zorder=5)


def main():
    records = load()
    conditions = [
        ("alias_0.000", "0", "alias"),
        ("alias_0.250", ".25", "alias"),
        ("alias_0.500", ".50", "alias"),
        ("alias_0.750", ".75", "alias"),
        ("noise_0.020", ".02", "noise"),
        ("noise_0.050", ".05", "noise"),
        ("noise_0.080", ".08", "noise"),
        ("noise_0.120", ".12", "noise"),
    ]
    models = [("hdgm", "HDGM", HDGM, "o", -.22),
              ("tem", "TEM", TEM, "s", 0),
              ("stateless", "stateless", STATELESS, "X", .22)]
    xs = [0, 1, 2, 3, 5, 6, 7, 8]

    fig, ax = plt.subplots(figsize=(11.8, 5.45), facecolor="white")
    fig.subplots_adjust(left=.085, right=.985, bottom=.19, top=.84)
    ax.set_facecolor("#FCFDFE")
    ax.axvspan(-.55, 3.55, color=ALIAS_TINT, zorder=0)
    ax.axvspan(4.45, 8.55, color=NOISE_TINT, zorder=0)
    ax.axhspan(.45, .51, color="#E2E8F0", alpha=.70, zorder=0)
    ax.axhline(.5, color="#94A3B8", lw=1, ls=(0, (3, 3)), zorder=1)
    for x in xs:
        ax.axvline(x, color="white", lw=1.0, zorder=1)
    ax.grid(axis="y", color=GRID, lw=.8, zorder=0)

    for (key, _, _), x in zip(conditions, xs):
        for model, _, color, marker, offset in models:
            point_range(ax, x + offset, values(records, key, model), color, marker)

    ax.text(1.5, 1.004, "STRUCTURAL ALIASING", ha="center", va="top", color="#C23B55",
            fontsize=9.2, weight="bold")
    ax.text(6.5, 1.004, "OBSERVATION NOISE", ha="center", va="top", color="#008D6B",
            fontsize=9.2, weight="bold")
    ax.text(1.5, .535, "choice collapses toward chance", ha="center", color="#C23B55",
            fontsize=8.1, weight="bold")
    ax.text(6.5, .670, "choice remains preserved", ha="center", color="#008D6B",
            fontsize=8.1, weight="bold")
    ax.text(8.47, .505, "chance", ha="right", va="bottom", color=MUTED, fontsize=7.4)

    ax.set_xlim(-.55, 8.55); ax.set_ylim(.45, 1.01)
    ax.set_xticks(xs, [label for _, label, _ in conditions])
    ax.set_ylabel("Choice accuracy", color=INK, fontsize=11)
    ax.set_xlabel(r"Perturbation magnitude: corruption $\alpha$                                      noise $\sigma$",
                  color=INK, fontsize=10, labelpad=10)
    ax.tick_params(colors="#526273", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#94A8C0")

    handles = [Line2D([0], [0], marker=m, color="none", markerfacecolor="white",
                      markeredgecolor=c, markeredgewidth=1.6, markersize=7, label=name)
               for _, name, c, m, _ in models]
    ax.legend(handles=handles, frameon=False, ncol=3, fontsize=8.4,
              loc="upper center", bbox_to_anchor=(.5, 1.11), columnspacing=1.2)
    fig.text(.055, .93, "C", fontsize=22, weight="bold", color=INK)
    fig.text(.085, .052, "Dots: individual seeds (n = 10); open symbols: condition mean; whiskers: 95% CI. No between-condition interpolation.",
             fontsize=7.6, color=MUTED)
    fig.savefig(OUT, dpi=360, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
