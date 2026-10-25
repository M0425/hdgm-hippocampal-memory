"""Data-faithful population-bifurcation portrait for Figure 5B.

Uses the frozen V2.1 confirmatory JSON. All nodes and filaments are observed
seed-by-condition values; basin fills and violet halos are explicitly marked
as explanatory visual encodings rather than fitted dynamical landscapes.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.lines import Line2D


DATA = Path(r"C:\Users\马明哲\Desktop\建模\pythonProject32\hdgm_difficulty_v2_1_frozen_outputs_confirmatory\difficulty_v2_1_multiseed_results.json")
OUT = Path(__file__).with_name("figure5B_hippocampal_population_bifurcation.png")
INK, MUTED, GRID = "#14213D", "#718096", "#D9E3EF"
ALIAS, NOISE, TEM, VIOLET = "#D1495B", "#009E73", "#64748B", "#6E59A5"


def load_data():
    with DATA.open(encoding="utf-8") as handle:
        records = json.load(handle)["records"]
    names = ["alias_0.000", "alias_0.250", "alias_0.500", "alias_0.750",
             "noise_0.020", "noise_0.050", "noise_0.080", "noise_0.120"]
    out = {}
    for name in names:
        rows = []
        for record in records:
            c = record["conditions"][name]
            rows.append({
                "seed": record["seed"],
                "x": c["representation"]["hdgm"]["reference_transfer_probe"],
                "y": c["behavior"]["hdgm"],
                "tem_x": c["representation"]["tem"]["reference_transfer_probe"],
                "tem_y": c["behavior"]["tem"],
                "stability": c["representation"]["hdgm"]["population_split_half_stability"],
            })
        out[name] = rows
    return out


def xy(rows, tem=False):
    return (np.array([r["tem_x" if tem else "x"] for r in rows]),
            np.array([r["tem_y" if tem else "y"] for r in rows]))


def mean_xy(rows, tem=False):
    x, y = xy(rows, tem)
    return np.array([x.mean(), y.mean()])


def confidence_size(rows):
    """Relative 95% uncertainty magnitude for an in-node packet envelope."""
    x, y = xy(rows)
    cov = np.cov(np.vstack([x, y])) / len(x)
    return np.sqrt(np.trace(cov) * 5.991)


def packet(ax, rows, color, label, halo=True, offset=(5, 7)):
    mean = mean_xy(rows)
    uncertainty = confidence_size(rows)
    # The violet halo encodes the measured history-decoding coordinate. The
    # confidence envelope is deliberately a glyph (relative magnitude) rather
    # than a spatial ellipse, which would obscure the bifurcation geometry.
    if halo:
        ax.scatter(*mean, s=330 + 1950 * (mean[0] - .47) / .36, color=VIOLET, alpha=.12,
                   linewidths=0, zorder=3)
    envelope_size = 160 + 3900 * np.clip(uncertainty, 0, .13)
    ax.scatter(*mean, s=envelope_size, facecolor="none", edgecolor=color,
               linewidth=.9, alpha=.34, zorder=4)
    outer_size = 88 + 310 * (mean[1] - .46) / .22
    ax.scatter(*mean, s=outer_size, facecolor="white", edgecolor=color,
               linewidth=2.2, zorder=6)
    # Ten seed marks are a compact population-packet glyph; their actual
    # x/y values remain visible in the seed filaments and marginal ridges.
    theta = np.linspace(0, 2 * np.pi, 10, endpoint=False) + .20
    radius_x, radius_y = .0032, .0023
    ax.scatter(mean[0] + radius_x * np.cos(theta), mean[1] + radius_y * np.sin(theta),
               s=8.5, color=color, alpha=.78, linewidths=0, zorder=7)
    ax.scatter(*mean, s=15, color=color, zorder=7)
    if label:
        ax.annotate(label, xy=mean, xytext=offset, textcoords="offset points",
                    color=color, fontsize=7.4, weight="bold", zorder=8)
    return mean


def width(rows):
    # Wide trunks correspond to higher observed split-half stability.
    stability = np.mean([r["stability"] for r in rows])
    return 1.8 + 4.2 * np.clip(stability, 0, 1)


def seed_filaments(ax, data, order, color):
    for index in range(10):
        pts = np.array([[data[key][index]["x"], data[key][index]["y"]] for key in order])
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=.68, alpha=.13, zorder=1)


def mean_branch(ax, data, order, color):
    pts = np.array([mean_xy(data[key]) for key in order])
    for start, end, key in zip(pts[:-1], pts[1:], order[1:]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color,
                lw=width(data[key]), alpha=.84, solid_capstyle="round", zorder=4)
    arrow = FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", color=color,
                            mutation_scale=12, lw=0, zorder=5)
    ax.add_patch(arrow)


def tem_branch(ax, data, order):
    pts = np.array([mean_xy(data[key], tem=True) for key in order])
    ax.plot(pts[:, 0], pts[:, 1], color=TEM, lw=1.45, ls=(0, (3, 2)), zorder=3)
    ax.scatter(pts[:, 0], pts[:, 1], marker="s", s=28, facecolor="white",
               edgecolor=TEM, lw=1.1, zorder=5)


def ridge(ax, values, color, vertical=False):
    values = np.asarray(values)
    grid = np.linspace(.46, .84, 300) if not vertical else np.linspace(.45, .74, 300)
    # Deterministic Gaussian mixture; it displays samples without interpolating conditions.
    bw = .010 if not vertical else .007
    density = np.exp(-.5 * ((grid[:, None] - values[None, :]) / bw) ** 2).sum(axis=1)
    density /= density.max() or 1
    if vertical:
        ax.fill_betweenx(grid, 0, density, color=color, alpha=.27, lw=0)
        ax.plot(density, grid, color=color, lw=.8, alpha=.8)
    else:
        ax.fill_between(grid, 0, density, color=color, alpha=.27, lw=0)
        ax.plot(grid, density, color=color, lw=.8, alpha=.8)


def main():
    data = load_data()
    alias_order = ["alias_0.000", "alias_0.250", "alias_0.500", "alias_0.750"]
    noise_order = ["noise_0.050", "noise_0.080", "noise_0.120"]
    all_alias = [v for key in alias_order for v in xy(data[key])[0]]
    all_noise = [v for key in ["noise_0.020"] + noise_order for v in xy(data[key])[0]]
    all_alias_y = [v for key in alias_order for v in xy(data[key])[1]]
    all_noise_y = [v for key in ["noise_0.020"] + noise_order for v in xy(data[key])[1]]

    fig = plt.figure(figsize=(12.1, 7.05), facecolor="white")
    ax = fig.add_axes([.085, .180, .655, .650])
    ax_bottom = fig.add_axes([.085, .098, .655, .040], sharex=ax)
    ax_right = fig.add_axes([.762, .180, .054, .650], sharey=ax)

    ax.set_xlim(.47, .83); ax.set_ylim(.46, .72)
    ax.set_facecolor("#FCFDFE")
    ax.grid(color=GRID, lw=.75, zorder=0)
    ax.axvline(.5, color="#AAB7C5", lw=.9, ls=(0, (3, 3)), zorder=1)
    ax.axhline(.5, color="#AAB7C5", lw=.9, ls=(0, (3, 3)), zorder=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#94A8C0"); ax.spines["bottom"].set_color("#94A8C0")
    ax.tick_params(colors="#526273", labelsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("Behavioral accuracy", fontsize=11, color=INK, labelpad=7)

    # Interpretive, deliberately non-fitted basins.
    ax.add_patch(Ellipse((.504, .508), .095, .070, facecolor="#CBD5E1", edgecolor="#A8B4C4",
                         lw=.9, alpha=.27, zorder=0))
    ax.add_patch(Ellipse((.710, .645), .205, .068, facecolor="#A7F3D0", edgecolor=NOISE,
                         lw=.95, alpha=.18, zorder=0))
    ax.text(.505, .473, "chance / collapsed\nbasin", ha="center", va="center", fontsize=7.6, color="#64748B")
    ax.text(.705, .690, "accessible-memory basin", ha="center", fontsize=8.1, color=NOISE, weight="bold")
    ax.text(.827, .467, "interpretive regions;\nnot fitted potentials", ha="right", va="bottom",
            fontsize=6.4, color=MUTED, style="italic")

    seed_filaments(ax, data, alias_order, ALIAS)
    seed_filaments(ax, data, noise_order, NOISE)
    mean_branch(ax, data, alias_order, ALIAS)
    mean_branch(ax, data, noise_order, NOISE)
    tem_branch(ax, data, alias_order)

    # Conditions are empirical nodes only: no interpolated nodes are drawn.
    alias_labels = ["", r"$\alpha=.25$", r"$\alpha=.50$", r"$\alpha=.75$"]
    noise_labels = ["", r"$\sigma=.08$", r"$\sigma=.12$"]
    packets = {}
    alias_offsets = [(5, 7), (5, 7), (5, 7), (5, 7)]
    noise_offsets = [(5, 7), (8, 17), (-25, 14)]
    for key, label, offset in zip(alias_order, alias_labels, alias_offsets):
        packets[key] = packet(ax, data[key], ALIAS, label, offset=offset)
    for key, label, offset in zip(noise_order, noise_labels, noise_offsets):
        packets[key] = packet(ax, data[key], NOISE, label, offset=offset)
    low_noise = packet(ax, data["noise_0.020"], NOISE, r"$\sigma=.02$", halo=False, offset=(15, -7))
    base = packets["alias_0.000"]
    ax.plot([low_noise[0], base[0]], [low_noise[1], base[1]], color=NOISE, lw=1.0, alpha=.40, zorder=2)
    ax.annotate("shared memory\nstate", xy=base, xytext=(26, 28), textcoords="offset points",
                color=INK, fontsize=8.7, weight="bold",
                arrowprops=dict(arrowstyle="-", color=INK, lw=.8))
    ax.text(.592, .556, "alias branch →\nchance attractor", color=ALIAS, fontsize=8.2,
            weight="bold", ha="center", va="center")
    ax.text(.700, .610, "noise branch →\nstable memory manifold", color=NOISE, fontsize=8.2,
            weight="bold", ha="center", va="center")

    legend = [
        Line2D([0], [0], color=ALIAS, lw=3, label="alias branch"),
        Line2D([0], [0], color=NOISE, lw=3, label="observation-noise branch"),
        Line2D([0], [0], color=TEM, lw=1.4, ls=(0, (3, 2)), marker="s", markerfacecolor="white", label="TEM"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=VIOLET, alpha=.35,
               markersize=10, label="violet halo: history-access strength"),
    ]
    ax.legend(handles=legend, loc="upper left", frameon=False, ncol=2, fontsize=7.8,
              handlelength=2.0, columnspacing=1.4, bbox_to_anchor=(.0, 1.015))
    ax.text(0, 1.15, "B", transform=ax.transAxes, fontsize=23, weight="bold", color=INK)
    ax.text(0, 1.085, "Hippocampal population bifurcation portrait", transform=ax.transAxes,
            fontsize=12, weight="bold", color=INK)

    ridge(ax_bottom, all_alias, ALIAS)
    ridge(ax_bottom, all_noise, NOISE)
    ax_bottom.set_ylim(0, 1.15); ax_bottom.axis("off")
    ridge(ax_right, all_alias_y, ALIAS, vertical=True)
    ridge(ax_right, all_noise_y, NOISE, vertical=True)
    ax_right.set_xlim(0, 1.15); ax_right.axis("off")
    ax_right.text(1.18, .59, "seed distribution\nof behavior", fontsize=6.6, color=MUTED,
                  rotation=90, va="center")

    fig.text(.412, .063, "History decoding (frozen transfer probe)", ha="center", fontsize=11, color=INK)
    fig.text(.085, .019,
             "Packets: ten seed marks; center: mean; outer envelope: relative 95% mean-uncertainty magnitude. "
             "Raw seed trajectories and marginal distributions are shown separately. Trunk width: split-half stability.",
             fontsize=7.1, color=MUTED)
    fig.savefig(OUT, dpi=360, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
