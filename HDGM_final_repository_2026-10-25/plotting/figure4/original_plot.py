#!/usr/bin/env python3
"""Rebuild Figure 4 from the confirmatory position and bifurcation JSON files.

All quantitative panels use the ten confirmatory seeds stored in the JSON
records.  The figure deliberately distinguishes the position-ring assay from
the shared-corridor history assay; it does not imply that they are one task.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parent
PLACE_JSON = ROOT / "upload" / "86b854c5-fe30-4c5c-aef8-8e2355fbd77b.json"
BRANCH_JSON = ROOT / "upload" / "3cbcb19b-8142-4b55-8af5-2c2eff80e68b.json"
OUT_PNG = ROOT / "figure4_hdgm_positional_branch_organization.png"
OUT_PDF = ROOT / "figure4_hdgm_positional_branch_organization.pdf"
FISHER_IMAGE = ROOT / "assets" / "fisherman_sampan_ink_v2.png"
PINE_IMAGE = ROOT / "assets" / "pine_ink_v1.png"
REEDS_IMAGE = ROOT / "assets" / "reeds_ink_v1.png"
BRIDGE_IMAGE = ROOT / "assets" / "bridge_ink_v1.png"
FISH_TEAL_IMAGE = ROOT / "assets" / "fish_teal_ink_v1.png"
FISH_AMBER_IMAGE = ROOT / "assets" / "fish_amber_ink_v1.png"


# Color-blind-safe semantic palette: one hue, one scientific role.
HDGM = "#0072B2"
HDGM_DARK = "#005A8D"
HDGM_LIGHT = "#DCEEF7"
TEM = "#4B5563"
HISTORY_A = "#009E73"
HISTORY_A_DARK = "#007A59"
HISTORY_A_LIGHT = "#D9F0E8"
HISTORY_B = "#E69F00"
HISTORY_B_DARK = "#B97800"
HISTORY_B_LIGHT = "#FCECC8"
POSITIONAL = "#6A51A3"
POSITIONAL_DARK = "#4F3C7E"
POSITIONAL_LIGHT = "#E8E2F3"
CHANCE = "#9CA3AF"
MISSING = "#D1D5DB"

# Compatibility aliases used by model-centric panels.
BLUE = HDGM
BLUE_DARK = HDGM_DARK
BLUE_LIGHT = HDGM_LIGHT
AMBER = HISTORY_B
AMBER_LIGHT = HISTORY_B_LIGHT
TEAL = HISTORY_A
TEAL_DARK = HISTORY_A_DARK
SLATE = TEM
INK = "#1E293B"
MID = "#7C8DA8"
LIGHT = "#E2E5E9"
PALE = "#F5F8FC"
WHITE = "#FFFFFF"
RED = "#E76F51"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def bootstrap_mean_ci(x: np.ndarray, seed: int, n_boot: int = 12000) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise bootstrap CI across seeds."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975], axis=0)
    return lo.squeeze(), hi.squeeze()


def paired_dz(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(d.mean() / d.std(ddof=1))


def bootstrap_dz_ci(a: np.ndarray, b: np.ndarray, seed: int, n_boot: int = 20000) -> tuple[float, float]:
    """Bootstrap the mean standardized paired difference on a fixed SD scale.

    With n=10, recomputing the SD inside each resample makes dz intervals
    unstable when a resample contains nearly identical observations.  The
    confirmatory paired SD is therefore fixed before resampling.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    z = d / d.std(ddof=1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(z), size=(n_boot, len(z)))
    return tuple(np.quantile(z[idx].mean(axis=1), [0.025, 0.975]))


def style_axis(ax: mpl.axes.Axes, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MID)
    ax.spines[["left", "bottom"]].set_linewidth(0.8)
    ax.tick_params(colors=SLATE, labelsize=8.2, length=3, width=0.7)
    if grid_axis:
        ax.grid(axis=grid_axis, color=LIGHT, lw=0.7, alpha=0.75, zorder=0)


def panel_letter(fig: mpl.figure.Figure, ax: mpl.axes.Axes, letter: str, dx: float = 0.023, dy: float = 0.006) -> None:
    pos = ax.get_position()
    fig.text(pos.x0 - dx, pos.y1 + dy, letter, fontsize=19, fontweight="bold", color=INK,
             ha="left", va="top")


def narrative_header(ax: mpl.axes.Axes, text: str, color: str = INK) -> None:
    ax.text(0.0, 1.055, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10.2, fontweight="bold", color=color, clip_on=False)


def draw_task_definition(ax: mpl.axes.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    WATER = "#DCE8E4"
    WATER_DEEP = "#AABCB6"
    INK_WASH = "#4B5A56"
    PAPER = "#FAFBF8"
    STONE = "#98A19E"

    # Complete fisherman-and-sampan asset redrawn from the supplied reference.
    # It is deliberately loaded as RGBA so the ink linework sits naturally on
    # the pale scientific-figure cards without a rectangular pasted-in crop.
    def load_rgba(path: Path) -> np.ndarray:
        with Image.open(path) as source:
            return np.asarray(source.convert("RGBA"), dtype=float) / 255.0

    fisher_rgba = load_rgba(FISHER_IMAGE)
    pine_rgba = load_rgba(PINE_IMAGE)
    reeds_rgba = load_rgba(REEDS_IMAGE)
    bridge_rgba = load_rgba(BRIDGE_IMAGE)
    fish_teal_rgba = load_rgba(FISH_TEAL_IMAGE)
    fish_amber_rgba = load_rgba(FISH_AMBER_IMAGE)

    def draw_channel(xs: list[float] | np.ndarray, ys: list[float] | np.ndarray,
                     fill: str = WATER, width: float = 13.0,
                     edge: str = INK_WASH, z: int = 3) -> None:
        """A pale river bounded by restrained ink-brush edges."""
        ax.plot(xs, ys, color=edge, lw=width + 2.5, alpha=0.48,
                solid_capstyle="round", solid_joinstyle="round", zorder=z)
        ax.plot(xs, ys, color=fill, lw=width, alpha=0.98,
                solid_capstyle="round", solid_joinstyle="round", zorder=z + 1)

    def draw_sprite(sprite: np.ndarray, x0: float, y0: float, height: float,
                    z: int = 11, alpha: float = 1.0,
                    flip_x: bool = False) -> None:
        """Place a transparent ink sprite without distorting its proportions."""
        image = np.fliplr(sprite) if flip_x else sprite
        if alpha < 1.0:
            image = image.copy()
            image[..., 3] *= alpha
        image_aspect = image.shape[1] / image.shape[0]
        axes_ratio = ax.bbox.height / max(ax.bbox.width, 1.0)
        width = height * image_aspect * axes_ratio
        ax.imshow(image,
                  extent=[x0 - width / 2, x0 + width / 2,
                          y0 - height / 2, y0 + height / 2],
                  interpolation="lanczos", aspect="auto", zorder=z,
                  clip_on=True)

    def draw_fish(x0: float, y0: float, color: str, scale: float = 1.0,
                  direction: int = 1, alpha: float = 1.0) -> None:
        """Use detailed watercolor fish rather than ellipse-triangle symbols."""
        sprite = fish_teal_rgba if color in (HISTORY_A, HISTORY_A_DARK) else fish_amber_rgba
        draw_sprite(sprite, x0, y0, 0.028 * scale, z=12, alpha=alpha,
                    flip_x=direction < 0)

    def draw_ripples(x0: float, y0: float, scale: float = 1.0,
                     color: str = INK_WASH, z: int = 7) -> None:
        for dy, ww in [(0.0, 0.034), (-0.010, 0.024), (0.010, 0.020)]:
            ax.add_patch(Arc((x0, y0 + dy * scale), ww * scale, 0.013 * scale,
                             theta1=8, theta2=172, color=color, lw=0.65,
                             alpha=0.60, zorder=z))

    def draw_boat_fisher(x0: float, y0: float, scale: float = 1.0,
                         accent: str = POSITIONAL) -> None:
        """Place one complete, coherent ink fisherman-and-sampan agent."""
        image_h = 0.104 * scale
        image_aspect = fisher_rgba.shape[1] / fisher_rgba.shape[0]
        # Compensate for the panel's non-square axes so the human and boat are
        # not stretched when positioned in normalized data coordinates.
        axes_ratio = ax.bbox.height / max(ax.bbox.width, 1.0)
        image_w = image_h * image_aspect * axes_ratio
        ax.imshow(
            fisher_rgba,
            extent=[x0 - image_w / 2, x0 + image_w / 2,
                    y0 - image_h / 2, y0 + image_h / 2],
            interpolation="lanczos", aspect="auto", zorder=15, clip_on=True,
        )
        # A short chromatic keel mark links the story agent to the relevant
        # representational role without recolouring the ink illustration.
        ax.plot([x0 - image_w * 0.30, x0 + image_w * 0.30],
                [y0 - image_h * 0.34] * 2, color=accent, lw=0.8,
                alpha=0.72, solid_capstyle="round", zorder=14)

    def draw_bank_patch(x0: float, y0: float, width: float, height: float,
                        angle: float = 0.0) -> None:
        ax.add_patch(Ellipse((x0, y0), width, height, angle=angle,
                             fc="#F0EEE5", ec=WATER_DEEP, lw=0.45,
                             alpha=0.95, zorder=7))
        ax.add_patch(Arc((x0, y0 + 0.001), width * 0.82, height * 0.55,
                         angle=angle, theta1=8, theta2=172,
                         color=STONE, lw=0.5, alpha=0.65, zorder=8))

    def draw_lantern(x0: float, y0: float, scale: float = 1.0) -> None:
        ax.plot([x0, x0], [y0, y0 + 0.055 * scale], color=INK_WASH, lw=0.8, zorder=10)
        ax.plot([x0, x0 + 0.018 * scale], [y0 + 0.050 * scale] * 2,
                color=INK_WASH, lw=0.7, zorder=10)
        ax.add_patch(FancyBboxPatch((x0 + 0.010 * scale, y0 + 0.030 * scale),
                                    0.018 * scale, 0.023 * scale,
                                    boxstyle="round,pad=0.001,rounding_size=0.004",
                                    fc=POSITIONAL_LIGHT, ec=POSITIONAL,
                                    lw=0.55, zorder=11))

    def draw_seal(x0: float, y0: float, color: str, label: str) -> None:
        ax.add_patch(Circle((x0, y0), 0.020, fc=WHITE, ec=color, lw=1.2, zorder=12))
        ax.add_patch(Circle((x0, y0), 0.016, fc=color, ec="none", alpha=0.12, zorder=11))
        ax.text(x0, y0, label, ha="center", va="center", fontsize=7.0,
                color=color, fontweight="bold", zorder=13)

    def draw_water_window(x0: float, y0: float, width: float, height: float,
                          border: str, fish_color: str | None = None,
                          fish_pattern: tuple[tuple[float, float, int], ...] = ()) -> None:
        ax.add_patch(FancyBboxPatch((x0, y0), width, height,
                                   boxstyle="round,pad=0.006,rounding_size=0.014",
                                   fc="#F3F7F5", ec=border, lw=0.85, zorder=4))
        for frac, amp in [(0.30, 0.005), (0.52, -0.004), (0.74, 0.004)]:
            xx = np.linspace(x0 + 0.010, x0 + width - 0.010, 45)
            phase = np.linspace(0, 2.4 * np.pi, len(xx))
            yy = y0 + height * frac + amp * np.sin(phase)
            ax.plot(xx, yy, color=WATER_DEEP, lw=0.6, alpha=0.70, zorder=5)
        if fish_color is not None:
            for fx, fy, direction in fish_pattern:
                draw_fish(x0 + fx * width, y0 + fy * height,
                          fish_color, 0.56, direction, 0.94)

    # Pale rice-paper cards preserve the scientific layout while the internal
    # marks adopt a restrained modern ink language.
    for x0 in (0.02, 0.515):
        ax.add_patch(FancyBboxPatch((x0, 0.43), 0.465, 0.50,
                                   boxstyle="round,pad=0.012,rounding_size=0.025",
                                   facecolor=PAPER, edgecolor=LIGHT, linewidth=0.9))

    # POSITION: a closed experimental route sampled sequentially.  The boat is
    # the current state on one lap, not a decorative fisherman circling aimlessly.
    ax.text(0.055, 0.865, "POSITION", fontsize=7.4, fontweight="bold",
            color=POSITIONAL_DARK)
    ax.text(0.055, 0.815, "Where am I?", fontsize=8.5, fontweight="bold", color=INK)
    cx, cy, rx, ry = 0.255, 0.645, 0.145, 0.103
    theta = np.linspace(0, 2 * np.pi, 240)
    ring_x, ring_y = cx + rx * np.cos(theta), cy + ry * np.sin(theta)
    draw_channel(ring_x, ring_y, WATER, 13.0, INK_WASH, 3)
    # Twenty measured positions remain explicit, but quiet enough not to read
    # as another quantitative plot.
    sample_theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    for tt in sample_theta:
        ax.add_patch(Circle((cx + rx * np.cos(tt), cy + ry * np.sin(tt)),
                            0.0041, fc=WHITE, ec=POSITIONAL, lw=0.45,
                            alpha=0.82, zorder=7))
    # Fully illustrated, naturally grounded landmarks replace the former
    # programmatic line icons.  Their rock/mud bases make the bank explicit.
    draw_sprite(reeds_rgba, cx - 0.166, cy + 0.052, 0.105, z=12)
    draw_sprite(bridge_rgba, cx, cy + 0.102, 0.084, z=13)
    draw_sprite(pine_rgba, cx + 0.177, cy + 0.061, 0.132, z=12)
    draw_boat_fisher(cx - 0.005, cy - 0.112, 0.92, POSITIONAL)
    travel_theta = np.linspace(4.50, 5.52, 34)
    travel_x = cx + rx * np.cos(travel_theta)
    travel_y = cy + ry * np.sin(travel_theta)
    ax.plot(travel_x[:-1], travel_y[:-1], color=POSITIONAL, lw=1.45,
            solid_capstyle="round", zorder=9)
    ax.add_patch(FancyArrowPatch((travel_x[-2], travel_y[-2]),
                                 (travel_x[-1], travel_y[-1]),
                                 arrowstyle="-|>", mutation_scale=9,
                                 color=POSITIONAL, lw=1.45, zorder=10))
    # Three progressively stronger samples ahead of the boat make the ordered
    # traversal visible without placing time text on top of the illustration.
    for tt, alpha, radius in zip([5.05, 5.34, 5.63], [0.45, 0.70, 1.0],
                                 [0.0046, 0.0052, 0.0060]):
        px, py = cx + rx * np.cos(tt), cy + ry * np.sin(tt)
        ax.add_patch(Circle((px, py), radius, fc=POSITIONAL, ec=WHITE,
                            lw=0.6, alpha=alpha, zorder=17))
    ax.text(cx, 0.455, "closed route · 20 ordered samples",
            ha="center", fontsize=6.6, color=SLATE)

    # HISTORY: the same upstream view opens into two history-conditioned arms.
    ax.text(0.55, 0.865, "HISTORY", fontsize=7.4, fontweight="bold",
            color=HISTORY_A_DARK)
    ax.text(0.55, 0.815, "Which branch now?", fontsize=8.5,
            fontweight="bold", color=INK)
    y0 = 0.645
    junction = (0.80, y0)
    top_goal = (0.925, 0.755)
    bottom_goal = (0.925, 0.535)
    draw_channel([0.56, junction[0]], [y0, y0], WATER, 15.0, INK_WASH, 3)
    draw_channel([junction[0], top_goal[0]], [junction[1], top_goal[1]],
                 HISTORY_A_LIGHT, 15.0, INK_WASH, 3)
    draw_channel([junction[0], bottom_goal[0]], [junction[1], bottom_goal[1]],
                 HISTORY_B_LIGHT, 15.0, INK_WASH, 3)
    for xx in np.linspace(0.62, 0.76, 5):
        draw_ripples(xx, y0, 0.72, WATER_DEEP, 8)
    ax.add_patch(FancyArrowPatch((0.815, 0.660), (0.910, 0.744),
                                 arrowstyle="-|>", mutation_scale=12,
                                 color=HISTORY_A, lw=2.1, zorder=10))
    ax.add_patch(FancyArrowPatch((0.815, 0.630), (0.910, 0.546),
                                 arrowstyle="-|>", mutation_scale=12,
                                 color=HISTORY_B, lw=2.1, zorder=10))
    draw_boat_fisher(0.635, y0 + 0.010, 0.88, HDGM)
    draw_fish(0.930, 0.774, HISTORY_A, 0.78, 1)
    draw_fish(0.930, 0.516, HISTORY_B, 0.78, -1)
    draw_seal(0.958, 0.760, HISTORY_A, "A")
    draw_seal(0.958, 0.530, HISTORY_B, "B")
    # Previous-trip memory is carried by fish identity, not by a traffic cue.
    ax.add_patch(Circle((0.620, 0.700), 0.009, fc=WHITE, ec=LIGHT, lw=0.6, zorder=12))
    ax.add_patch(Circle((0.642, 0.718), 0.013, fc=WHITE, ec=LIGHT, lw=0.6, zorder=12))
    ax.add_patch(FancyBboxPatch((0.655, 0.704), 0.116, 0.071,
                               boxstyle="round,pad=0.008,rounding_size=0.018",
                               fc=WHITE, ec=LIGHT, lw=0.75, zorder=11))
    ax.text(0.713, 0.753, "remembered cue", ha="center", va="center",
            fontsize=6.1, color=SLATE, zorder=13)
    draw_fish(0.692, 0.722, HISTORY_A, 0.54, 1)
    draw_fish(0.738, 0.722, HISTORY_B, 0.54, -1)
    ax.text(0.745, 0.455, "same view · remembered cue selects branch",
            ha="center", fontsize=6.6, color=SLATE)

    # SAME RIVER VIEW: identical waves, different internally organized states.
    ax.add_patch(FancyBboxPatch((0.02, 0.055), 0.96, 0.30,
                               boxstyle="round,pad=0.012,rounding_size=0.025",
                               facecolor=PAPER, edgecolor=LIGHT, linewidth=0.9))
    ax.text(0.05, 0.305, "SAME RIVER VIEW", fontsize=7.0,
            fontweight="bold", color=SLATE)
    draw_water_window(0.070, 0.135, 0.115, 0.100, POSITIONAL)
    draw_ripples(0.128, 0.185, 1.25, POSITIONAL, 8)
    ax.text(0.127, 0.097, "$x_t$", ha="center", fontsize=7.1,
            color=POSITIONAL_DARK)
    ax.add_patch(FancyArrowPatch((0.205, 0.185), (0.392, 0.185),
                                 arrowstyle="-|>", mutation_scale=11,
                                 color=SLATE, lw=1.05))
    ax.text(0.298, 0.220, "remembered", ha="center", fontsize=6.3, color=SLATE)
    ax.text(0.298, 0.150, "history", ha="center", fontsize=6.3, color=SLATE)
    pattern_a = ((0.25, 0.68, 1), (0.55, 0.53, 1), (0.76, 0.72, 1), (0.36, 0.28, 1))
    pattern_b = ((0.25, 0.68, -1), (0.55, 0.53, -1), (0.76, 0.72, -1), (0.36, 0.28, -1))
    draw_water_window(0.430, 0.130, 0.135, 0.110, HISTORY_A,
                      HISTORY_A, pattern_a)
    draw_water_window(0.665, 0.130, 0.135, 0.110, HISTORY_B,
                      HISTORY_B, pattern_b)
    ax.text(0.497, 0.270, "state A", ha="center", fontsize=6.8,
            fontweight="bold", color=HISTORY_A_DARK)
    ax.text(0.732, 0.270, "state B", ha="center", fontsize=6.8,
            fontweight="bold", color=HISTORY_B_DARK)
    ax.text(0.890, 0.195, r"$p_t=g_t\odot f(x_{1:t})$", ha="center",
            va="center", fontsize=6.7, fontweight="bold", color=INK)
    ax.text(0.875, 0.108, "same view · different remembered histories",
            ha="center", fontsize=5.8, color=MID)

def draw_position_atlas(ax_hm: mpl.axes.Axes, ax_pair: mpl.axes.Axes,
                        place_records: list[dict]) -> None:
    maps = []
    block_centers = []
    for i, record in enumerate(place_records):
        m = np.asarray(record["primary_probes"]["hdgm"]["example_place_fields"]["normalized_rate_maps"], float)
        m = m[np.argsort(np.argmax(m, axis=1))]
        maps.append(m)
        block_centers.append(i * len(m) + (len(m) - 1) / 2)
    atlas = np.vstack(maps)
    cmap = mpl.colormaps["cividis"].copy()
    cmap.set_bad(MISSING)
    ax_hm.imshow(atlas, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    for i in range(1, len(place_records)):
        ax_hm.axhline(i * 12 - 0.5, color=WHITE, lw=0.75, alpha=0.95)
    ax_hm.set_xticks([0, 4, 9, 14, 19])
    ax_hm.set_xticklabels(["0", "4", "9", "14", "19"], fontsize=7.8, color=SLATE)
    ax_hm.set_xlabel("Ring position", fontsize=8.2, color=INK, labelpad=2)
    ax_hm.set_yticks(block_centers)
    ax_hm.set_yticklabels([str(r["seed"]) for r in place_records], fontsize=6.8, color=SLATE)
    ax_hm.set_ylabel("Seed · sorted units", fontsize=8.2, color=INK, labelpad=3)
    ax_hm.tick_params(length=0)
    for s in ax_hm.spines.values():
        s.set_color(LIGHT)
        s.set_linewidth(0.75)
    ax_hm.text(1.0, 1.055, "normalized activity", transform=ax_hm.transAxes,
               ha="right", va="bottom", fontsize=7.2, color=MID, clip_on=False)

    # The primary population endpoint: residual map stability against the structured null.
    actual = np.array([
        r["primary_probes"]["hdgm"]["sensory_residual_population_geometry"]["population_map_stability"]
        for r in place_records
    ])
    null = np.array([
        r["primary_probes"]["hdgm"]["sensory_residual_shuffled_population_geometry"]["population_map_stability"]
        for r in place_records
    ])
    jitter = np.linspace(-0.028, 0.028, len(actual))
    for i, (a, n) in enumerate(zip(actual, null)):
        ax_pair.plot([0 + jitter[i], 1 + jitter[i]], [n, a], color=CHANCE, alpha=0.42, lw=0.9, zorder=1)
        ax_pair.scatter(0 + jitter[i], n, s=17, facecolor=WHITE, edgecolor=TEM, lw=0.75, zorder=3)
        ax_pair.scatter(1 + jitter[i], a, s=20, facecolor=POSITIONAL, edgecolor=WHITE, lw=0.55, zorder=4)
    lo_n, hi_n = bootstrap_mean_ci(null, 401)
    lo_a, hi_a = bootstrap_mean_ci(actual, 402)
    for x, vals, lo, hi, col in [(0, null, lo_n, hi_n, TEM), (1, actual, lo_a, hi_a, POSITIONAL)]:
        ax_pair.errorbar(x, vals.mean(), yerr=[[vals.mean() - lo], [hi - vals.mean()]], fmt="D",
                         ms=5.0, mfc=col, mec=WHITE, mew=0.7, ecolor=col, elinewidth=1.3,
                         capsize=3, zorder=6)
    ax_pair.axhline(0, color=LIGHT, lw=0.8, zorder=0)
    ax_pair.set_xlim(-0.35, 1.35)
    ax_pair.set_ylim(-0.28, 1.02)
    ax_pair.set_xticks([0, 1])
    ax_pair.set_xticklabels(["Structured\nnull", "HDGM\nresidual"], fontsize=7.6)
    ax_pair.set_ylabel("Split-half map stability", fontsize=8.2, color=INK, labelpad=2)
    ax_pair.set_yticks([-0.2, 0.2, 0.6, 1.0])
    style_axis(ax_pair)
    ax_pair.text(0.5, 0.96, "$d_z=2.77$  ·  Holm $p=.0039$", transform=ax_pair.transAxes,
                 ha="center", va="top", fontsize=7.5, color=POSITIONAL_DARK,
                 bbox=dict(boxstyle="round,pad=0.18", fc=WHITE, ec=LIGHT, lw=0.6))


def draw_branch_trajectory(ax: mpl.axes.Axes, ax_kernel: mpl.axes.Axes,
                           branch_records: list[dict]) -> None:
    x = np.arange(18)
    hdgm = np.array([[r["probes"]["hdgm"]["outbound_position_curve"][str(i)] for i in x]
                     for r in branch_records], float)
    tem = np.array([[r["probes"]["tem"]["outbound_position_curve"][str(i)] for i in x]
                    for r in branch_records], float)
    hmean, tmean = hdgm.mean(axis=0), tem.mean(axis=0)
    hlo, hhi = bootstrap_mean_ci(hdgm, 501)
    tlo, thi = bootstrap_mean_ci(tem, 502)

    # Visual aperture is a transparent monotonic mapping of decoding above
    # chance, not a claim about Euclidean latent-state distance.
    aperture = lambda a: np.clip(2.0 * (np.asarray(a) - 0.5), 0.0, 1.0)
    ha = aperture(hdgm)
    ta = aperture(tem)
    hma, tma = aperture(hmean), aperture(tmean)
    hloa, hhia = aperture(hlo), aperture(hhi)
    tloa, thia = aperture(tlo), aperture(thi)

    # Main neural corridor: the two pale population sheets retain the dynamic
    # manifold language while matching the paper's light visual system.
    ax.set_facecolor("#FCFDFF")
    ax.axhspan(0, 1.08, color=HISTORY_A_LIGHT, alpha=0.24, zorder=0)
    ax.axhspan(-1.08, 0, color=AMBER_LIGHT, alpha=0.25, zorder=0)
    for i in x:
        ax.axvline(i, color=LIGHT, lw=0.45, alpha=0.65, zorder=0)
    ax.fill_between(x, 0, hma, color=HISTORY_A, alpha=0.11, zorder=1)
    ax.fill_between(x, -hma, 0, color=AMBER, alpha=0.12, zorder=1)
    ax.fill_between(x, hloa, hhia, color=HISTORY_A, alpha=0.16, zorder=2)
    ax.fill_between(x, -hhia, -hloa, color=AMBER, alpha=0.16, zorder=2)
    ax.fill_between(x, -thia, thia, color=SLATE, alpha=0.06, zorder=2)
    for row in ha:
        ax.plot(x, row, color=HISTORY_A, lw=0.72, alpha=0.17, zorder=3)
        ax.plot(x, -row, color=AMBER, lw=0.72, alpha=0.17, zorder=3)
    for row in ta:
        ax.plot(x, row, color=SLATE, lw=0.55, alpha=0.08, zorder=2)
        ax.plot(x, -row, color=SLATE, lw=0.55, alpha=0.08, zorder=2)
    glow_blue = [pe.Stroke(linewidth=5.0, foreground=HISTORY_A_LIGHT, alpha=0.95), pe.Normal()]
    glow_amber = [pe.Stroke(linewidth=5.0, foreground=AMBER_LIGHT, alpha=0.95), pe.Normal()]
    up, = ax.plot(x, hma, color=HISTORY_A_DARK, lw=2.35, zorder=6)
    dn, = ax.plot(x, -hma, color=HISTORY_B_DARK, lw=2.35, zorder=6)
    up.set_path_effects(glow_blue)
    dn.set_path_effects(glow_amber)

    # Directional arrowheads and wave packets turn the static envelope into a
    # trajectory unfolding along the corridor without changing the data map.
    for i0, i1 in ((2, 4), (7, 9), (12, 14)):
        ax.add_patch(FancyArrowPatch((x[i0], hma[i0]), (x[i1], hma[i1]),
                                    arrowstyle="-|>", mutation_scale=10,
                                    color=HISTORY_A, lw=1.45, alpha=0.90,
                                    connectionstyle="arc3,rad=0.02", zorder=8))
        ax.add_patch(FancyArrowPatch((x[i0], -hma[i0]), (x[i1], -hma[i1]),
                                    arrowstyle="-|>", mutation_scale=10,
                                    color=AMBER, lw=1.45, alpha=0.90,
                                    connectionstyle="arc3,rad=-0.02", zorder=8))
    wave_idx = np.array([4, 9, 14])
    wave_sizes = np.array([48, 72, 96])
    wave_up = ax.scatter(wave_idx, hma[wave_idx], s=wave_sizes, fc=WHITE,
                         ec=HISTORY_A, lw=1.25, alpha=1.0, zorder=9)
    wave_dn = ax.scatter(wave_idx, -hma[wave_idx], s=wave_sizes, fc=WHITE,
                         ec=AMBER, lw=1.25, alpha=1.0, zorder=9)
    wave_up.set_path_effects([pe.Stroke(linewidth=5.0, foreground=HISTORY_A_LIGHT, alpha=0.95), pe.Normal()])
    wave_dn.set_path_effects([pe.Stroke(linewidth=5.0, foreground=AMBER_LIGHT, alpha=0.95), pe.Normal()])

    # The centre rail is the identical physical corridor; neural branches open
    # around it without changing the observation sequence.
    ax.plot([0, 17], [0, 0], color=SLATE, lw=2.5, solid_capstyle="round", zorder=5)
    ax.scatter(x, np.zeros_like(x), s=23, fc=WHITE, ec=SLATE, lw=0.80, zorder=7)
    for i in x:
        ax.plot([i, i], [-hma[i], hma[i]], color=MID, lw=0.45, alpha=0.25, zorder=2)
    ax.add_patch(FancyArrowPatch((6.5, 0.11), (10.2, 0.11), arrowstyle="-|>",
                                mutation_scale=9, color=SLATE, lw=0.8, alpha=0.85, zorder=8))
    ax.text(8.35, 0.18, "trajectory direction", ha="center", va="bottom",
            fontsize=6.4, color=SLATE, zorder=8)

    ax.set_xlim(-0.35, 17.35)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xticks([])
    ax.set_yticks([0.82, 0, -0.82])
    ax.set_yticklabels(["history A", "same position", "history B"], fontsize=7.8)
    ax.tick_params(axis="y", colors=SLATE, length=0, pad=8)
    for tick, col in zip(ax.get_yticklabels(), [HISTORY_A_DARK, TEM, HISTORY_B_DARK]):
        tick.set_color(col)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(0.02, 0.55, "matched observation + position", transform=ax.transAxes,
            ha="left", va="center", fontsize=7.1, fontweight="bold", color=SLATE,
            bbox=dict(boxstyle="round,pad=0.18", fc=WHITE, ec=LIGHT,
                      lw=0.55, alpha=0.96), zorder=12)
    ax.text(0.985, 0.90, "A", transform=ax.transAxes, ha="right", va="center",
            fontsize=9.0, fontweight="bold", color=HISTORY_A_DARK)
    ax.text(0.985, 0.10, "B", transform=ax.transAxes, ha="right", va="center",
            fontsize=9.0, fontweight="bold", color=HISTORY_B_DARK)
    ax.text(0.015, 0.055, "visual aperture = 2 × (decoding − chance)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=6.9, color=MID,
            bbox=dict(boxstyle="round,pad=0.15", fc=WHITE, ec=LIGHT,
                      lw=0.50, alpha=0.94), zorder=12)
    ax.text(0.985, 0.055, "TEM remains collapsed at the centre rail", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.9, color=MID,
            bbox=dict(boxstyle="round,pad=0.15", fc=WHITE, ec=LIGHT,
                      lw=0.50, alpha=0.94), zorder=12)

    # Exact statistical kernel directly beneath the visual mapping.
    ax_kernel.set_facecolor("#FBFCFF")
    ax_kernel.fill_between(x, hlo, hhi, color=BLUE, alpha=0.16, zorder=1)
    ax_kernel.fill_between(x, tlo, thi, color=SLATE, alpha=0.11, zorder=1)
    ax_kernel.plot(x, hmean, color=BLUE, lw=2.0, marker="o", ms=2.8,
                   mfc=WHITE, mec=BLUE, mew=0.8, zorder=4)
    ax_kernel.plot(x, tmean, color=TEM, lw=1.25, ls=(0, (4, 2)), marker="s", ms=2.4,
                   mfc=WHITE, mec=TEM, mew=0.65, zorder=3)
    ax_kernel.axhline(0.5, color=CHANCE, lw=0.85, ls=(0, (4, 3)), alpha=0.9)
    ax_kernel.set_xlim(-0.35, 17.35)
    ax_kernel.set_ylim(0.47, 1.01)
    ax_kernel.set_xticks(np.arange(0, 18, 2))
    ax_kernel.set_yticks([0.5, 0.75, 1.0])
    ax_kernel.set_yticklabels([".50", ".75", "1.00"], fontsize=7.2)
    ax_kernel.set_xlabel("Matched position along the shared outbound corridor", fontsize=8.3,
                         color=INK, labelpad=3)
    ax_kernel.set_ylabel("Accuracy", fontsize=8.0, color=INK, labelpad=4)
    style_axis(ax_kernel)
    ax_kernel.grid(axis="y", color=LIGHT, lw=0.65, alpha=0.75)
    handles = [
        Line2D([0], [0], color=BLUE, lw=2.0, marker="o", mfc=WHITE, mec=BLUE, label="HDGM"),
        Line2D([0], [0], color=TEM, lw=1.25, ls=(0, (4, 2)), marker="s", mfc=WHITE, mec=TEM, label="TEM"),
        Line2D([0], [0], color=CHANCE, lw=0.9, ls=(0, (4, 3)), label="chance"),
    ]
    ax_kernel.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.015),
                     ncol=3, frameon=False, fontsize=7.0, handlelength=2.0,
                     columnspacing=1.0, borderaxespad=0)


def draw_layer_heatmap(ax: mpl.axes.Axes, branch_records: list[dict]) -> None:
    keys = ["mem", "g_hist", "predictive_state", "attractor_state", "associative_read",
            "feedback_state", "ec_gate_pattern", "ca3_gate_pattern", "ca1_gate_pattern"]
    labels = ["Memory", "EC history", "Predictive", "CA3 attractor", "Assoc. read",
              "CA1 feedback", "EC gate", "CA3 gate", "CA1 gate"]
    shared = np.array([[r["probes"]["hdgm"]["layerwise_full_history"][k]["probe_accuracy"]
                        for k in keys] for r in branch_records], float)
    choice = np.array([[r["probes"]["hdgm"]["choice_state_layerwise"][k]
                        for k in keys] for r in branch_records], float)
    mat = 100 * np.column_stack([shared.mean(axis=0), choice.mean(axis=0)])
    cmap = mpl.colormaps["cividis"].copy()
    cmap.set_bad(MISSING)
    ax.imshow(mat, aspect="auto", cmap=cmap, vmin=50, vmax=90, interpolation="nearest")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.6,
                    fontweight="bold", color=INK if v >= 68 else WHITE)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Shared\ncorridor", "Choice\nstate"], fontsize=7.7, color=INK)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", pad=5, length=0)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0, pad=0)
    for s in ax.spines.values():
        s.set_color(WHITE)
        s.set_linewidth(1.4)
    # D is a pure continuous-strength panel; categorical hues are deliberately
    # omitted so color represents only decoding magnitude.
    label_transform = ax.get_yaxis_transform()
    for i, label in enumerate(labels):
        ax.text(-0.018, i, label, transform=label_transform, ha="right", va="center",
                fontsize=7.25, color=SLATE, clip_on=False)
    ax.text(1.0, -0.07, "mean accuracy (%) · 10 seeds", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.2, color=MID, clip_on=False)


def draw_geometry_phase(ax: mpl.axes.Axes, branch_records: list[dict]) -> None:
    full_d = np.array([r["probes"]["hdgm"]["full_geometry"]["centroid_distance"] for r in branch_records])
    full_w = np.array([r["probes"]["hdgm"]["full_geometry"]["within_branch_dispersion"] for r in branch_records])
    reset_d = np.array([r["probes"]["hdgm"]["reset_geometry"]["centroid_distance"] for r in branch_records])
    reset_w = np.array([r["probes"]["hdgm"]["reset_geometry"]["within_branch_dispersion"] for r in branch_records])

    all_x = np.r_[full_w, reset_w]
    all_y = np.r_[full_d, reset_d]
    xmin, xmax = 10 ** (np.floor(np.log10(all_x.min())) - 0.1), 10 ** (np.ceil(np.log10(all_x.max())) + 0.1)
    ymin, ymax = 10 ** (np.floor(np.log10(all_y.min())) - 0.1), 10 ** (np.ceil(np.log10(all_y.max())) + 0.1)
    xx = np.logspace(np.log10(xmin), np.log10(xmax), 300)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # Ratio field: d_between / d_within.
    ax.fill_between(xx, xx, ymax, where=xx <= ymax, color="#EEF0F2", alpha=0.72, zorder=0)
    for ratio, label, alpha in [(0.1, "0.1", 0.45), (1, "1", 0.75), (10, "10", 0.45)]:
        yy = ratio * xx
        valid = (yy >= ymin) & (yy <= ymax)
        ax.plot(xx[valid], yy[valid], color=MID, lw=0.8 if ratio != 1 else 1.15,
                ls=(0, (3, 3)), alpha=alpha, zorder=1)
        if np.any(valid):
            k = np.flatnonzero(valid)[-1]
            ax.text(xx[k] / 1.07, yy[k] / 1.05, label, fontsize=6.8, color=MID,
                    ha="right", va="top", rotation=29)
    for i in range(len(branch_records)):
        arrow = FancyArrowPatch((reset_w[i], reset_d[i]), (full_w[i], full_d[i]),
                                arrowstyle="-|>", mutation_scale=8, lw=0.85,
                                color=BLUE, alpha=0.38, zorder=3,
                                connectionstyle="arc3,rad=0.035")
        ax.add_patch(arrow)
    ax.scatter(reset_w, reset_d, s=29, facecolor=WHITE, edgecolor=SLATE, lw=0.85,
               label="History reset", zorder=5)
    ax.scatter(full_w, full_d, s=38, facecolor=BLUE, edgecolor=WHITE, lw=0.65,
               label="Full history", zorder=6)
    ax.set_xlabel("Within-branch dispersion", fontsize=9.0, color=INK, labelpad=6)
    ax.set_ylabel("Opposite-history centroid distance", fontsize=9.0, color=INK, labelpad=6)
    style_axis(ax, grid_axis="")
    ax.grid(which="major", color=LIGHT, lw=0.65, alpha=0.65)
    ax.legend(loc="lower right", frameon=True, facecolor=WHITE, edgecolor=LIGHT,
              framealpha=0.96, fontsize=7.8, borderpad=0.45, handletextpad=0.5)
    ax.text(0.02, 0.96, "between > within", transform=ax.transAxes, ha="left", va="top",
            fontsize=7.4, color=BLUE_DARK, fontweight="bold")
    ax.text(0.985, 0.985, "dashed: separation ratio", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.1, color=MID)
    ax.text(0.02, 0.04, "arrows: reset → full history (same seed)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7.2, color=MID,
            bbox=dict(boxstyle="round,pad=0.20", fc=WHITE, ec=LIGHT, lw=0.55))


def draw_geometry_gain_wheel(ax: mpl.axes.Axes, branch_records: list[dict]) -> None:
    """Seed-resolved log-fold gain around a common no-change reference ring."""
    full = np.array([
        r["probes"]["hdgm"]["full_geometry"]["separation_ratio"]
        for r in branch_records
    ], float)
    reset = np.array([
        r["probes"]["hdgm"]["reset_geometry"]["separation_ratio"]
        for r in branch_records
    ], float)
    seeds = [str(r.get("seed", i + 1)) for i, r in enumerate(branch_records)]
    gain = np.log10(full / reset)
    theta = np.linspace(0, 2 * np.pi, len(full), endpoint=False)
    # A common 1x ring makes every paired change an explicit arm.  The single
    # extreme seed is capped and annotated rather than compressing the other
    # nine seeds into the origin.
    baseline = 0.30
    lower_gain, upper_gain = -0.27, 1.03
    end_r = baseline + np.clip(gain, lower_gain, upper_gain)
    max_r = baseline + upper_gain

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, max_r)
    # Alternating seed sectors act as a circular raster and keep the ten paired
    # trajectories visually separable without implying interpolation.
    width = 2 * np.pi / len(theta)
    for i, th in enumerate(theta):
        ax.bar(th, max_r, width=width * 0.94, bottom=0,
               color=MISSING if i % 2 == 0 else PALE,
               alpha=0.34, edgecolor=WHITE, linewidth=0.45, zorder=0)
        ax.plot([th, th], [0, max_r], color=LIGHT, lw=0.55, alpha=0.75, zorder=1)
        col = TEAL if gain[i] >= 0 else AMBER
        ax.annotate("", xy=(th, end_r[i]), xytext=(th, baseline),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.75,
                                    alpha=0.88, mutation_scale=9.0), zorder=4)
        ax.scatter([th], [end_r[i]], s=37, facecolor=col, edgecolor=WHITE,
                   lw=0.70, zorder=6)
    ax.scatter(theta, np.full_like(theta, baseline), s=20, facecolor=WHITE,
               edgecolor=SLATE, lw=0.75, zorder=5)

    ring_theta = np.linspace(0, 2 * np.pi, 361)
    ax.plot(ring_theta, np.full_like(ring_theta, baseline),
            color=SLATE, lw=1.05, ls=(0, (3, 2)), alpha=0.85, zorder=3)

    ax.set_xticks(theta)
    ax.set_xticklabels(seeds, fontsize=6.4, color=SLATE)
    ax.tick_params(axis="x", pad=1)
    rticks = baseline + np.array([0.0, 0.3, 0.6, 0.9])
    ax.set_yticks(rticks)
    ax.set_yticklabels(["1×", "2×", "4×", "8×"], fontsize=5.9, color=MID)
    # Put scale labels halfway between two seed spokes, away from the capped
    # seed-404 arm and its annotation.
    ax.set_rlabel_position(270)
    ax.grid(color=LIGHT, lw=0.55, alpha=0.78)
    ax.spines["polar"].set_color(LIGHT)
    ax.spines["polar"].set_linewidth(0.8)
    handles = [
        Line2D([0], [0], color=TEAL, lw=1.8, marker=">", markersize=4.8,
               label="gain"),
        Line2D([0], [0], color=AMBER, lw=1.8, marker="<", markersize=4.8,
               label="loss"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 0.98),
              ncol=1, frameon=True, facecolor=WHITE, edgecolor=LIGHT,
              framealpha=0.96, fontsize=6.5, borderpad=0.32,
              labelspacing=0.28, handletextpad=0.35)
    ax.text(0.5, -0.25, "$\\log_{10}(\\rho_{full}/\\rho_{reset})$ · outer arm = gain",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.5,
            color=MID, clip_on=False)
    i_extreme = int(np.argmax(gain))
    ax.annotate(f"{full[i_extreme] / reset[i_extreme]:.0f}×",
                xy=(theta[i_extreme], max_r - 0.02),
                xytext=(theta[i_extreme] + 0.17, max_r + 0.10),
                textcoords="data", ha="left", va="center", fontsize=6.5,
                fontweight="bold", color=TEAL_DARK,
                arrowprops=dict(arrowstyle="-", color=TEAL_DARK, lw=0.7),
                annotation_clip=False)


def draw_effect_forest(ax: mpl.axes.Axes, place_records: list[dict], branch_records: list[dict]) -> None:
    pos_actual = np.array([
        r["primary_probes"]["hdgm"]["sensory_residual_population_geometry"]["population_map_stability"]
        for r in place_records
    ])
    pos_null = np.array([
        r["primary_probes"]["hdgm"]["sensory_residual_shuffled_population_geometry"]["population_map_stability"]
        for r in place_records
    ])
    alias_actual = np.array([
        r["primary_probes"]["hdgm"]["sensory_residual_alias_population"]["alias_centroid_to_within_ratio"]
        for r in place_records
    ])
    alias_null = np.array([
        r["primary_probes"]["hdgm"]["sensory_residual_alias_shuffled"]["alias_centroid_to_within_ratio"]
        for r in place_records
    ])
    full = np.array([r["probes"]["hdgm"]["full_history_probe"] for r in branch_records])
    reset = np.array([r["probes"]["hdgm"]["history_reset_probe"] for r in branch_records])
    obs = np.array([r["probes"]["hdgm"]["observation_only_probe"] for r in branch_records])
    comparisons = [
        ("Residual map\nstability", pos_actual, pos_null, POSITIONAL, ".0039"),
        ("Matched-position\ngeometry", alias_actual, alias_null, POSITIONAL, ".0039"),
        ("History decode\nvs reset", full, reset, HISTORY_A, ".0039"),
        ("History decode\nvs observation", full, obs, HISTORY_A, ".0039"),
    ]
    ys = np.array([3.2, 2.2, 0.8, -0.2])
    ax.axhspan(1.48, 3.62, color=POSITIONAL_LIGHT, alpha=0.36, zorder=0)
    ax.axhspan(-0.58, 1.38, color=HISTORY_A_LIGHT, alpha=0.42, zorder=0)
    density_x = np.linspace(0.0, 4.55, 260)
    for i, ((label, a, b, col, ptxt), y) in enumerate(zip(comparisons, ys)):
        diff = np.asarray(a, float) - np.asarray(b, float)
        seed_z = diff / diff.std(ddof=1)
        # A compact ridge shows the full seed distribution; the forest marker
        # retains the confirmatory mean effect and bootstrap interval.
        bandwidth = max(0.20, 0.78 * seed_z.std(ddof=1) * len(seed_z) ** (-0.2))
        density = np.exp(-0.5 * ((density_x[:, None] - seed_z[None, :]) / bandwidth) ** 2).mean(axis=1)
        density /= max(density.max(), 1e-12)
        ridge = 0.17 * density
        ax.fill_between(density_x, y, y + ridge, color=col, alpha=0.18,
                        edgecolor=col, linewidth=0.55, zorder=1)
        jitter = 0.045 * np.sin(np.arange(len(seed_z)) * 2.15 + i)
        ax.scatter(seed_z, y - 0.085 + jitter, s=16, facecolor=WHITE,
                   edgecolor=col, lw=0.75, alpha=0.95, zorder=4)
        dz = paired_dz(a, b)
        lo, hi = bootstrap_dz_ci(a, b, 700 + i)
        ax.plot([lo, hi], [y, y], color=col, lw=2.15, solid_capstyle="round", zorder=5)
        ax.scatter([lo, hi], [y, y], s=13, color=col, zorder=6)
        ax.scatter(dz, y, s=57, marker="D", fc=col, ec=WHITE, lw=0.78, zorder=7)
        ax.text(5.10, y, f"p={ptxt}", ha="right", va="center", fontsize=6.8, color=SLATE,
                bbox=dict(boxstyle="round,pad=0.08", fc=WHITE, ec="none", alpha=0.92))
    ax.axvline(0, color=INK, lw=0.85, ls=(0, (4, 3)), zorder=1)
    ax.axhline(1.48, color=LIGHT, lw=0.8)
    ax.set_xlim(-0.45, 5.18)
    ax.set_ylim(-0.66, 3.72)
    ax.set_yticks(ys)
    ax.set_yticklabels([c[0] for c in comparisons], fontsize=7.45, color=INK)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlabel("Paired effect size  $d_z$", fontsize=8.7, color=INK, labelpad=5)
    style_axis(ax, grid_axis="x")
    ax.grid(axis="x", color=LIGHT, lw=0.7, alpha=0.75)
    ax.text(-0.42, 3.58, "POSITIONAL SCAFFOLD", fontsize=6.9, color=POSITIONAL_DARK,
            fontweight="bold", ha="left", va="top")
    ax.text(-0.42, 1.30, "HISTORY BRANCH", fontsize=6.9, color=HISTORY_A_DARK,
            fontweight="bold", ha="left", va="top")
    ax.text(1.0, 1.018, "○ seeds · density ridge · ◆ $d_z$ - 95% CI",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.25, color=MID,
            clip_on=False)


def build_figure() -> tuple[mpl.figure.Figure, dict[str, mpl.axes.Axes]]:
    place = load_json(PLACE_JSON)
    branch = load_json(BRANCH_JSON)
    place_records = place["records"]
    branch_records = branch["records"]

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelcolor": INK,
        "text.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": WHITE,
        "figure.facecolor": WHITE,
    })

    # The taller canvas is assigned almost entirely to the top task-definition
    # row; the data-rich middle and lower rows retain their physical height.
    fig = plt.figure(figsize=(17.0, 12.20), facecolor=WHITE)
    outer = fig.add_gridspec(
        3, 18, left=0.055, right=0.985, bottom=0.080, top=0.950,
        height_ratios=[4.85, 4.05, 2.85], hspace=0.06, wspace=0.95
    )
    ax_a = fig.add_subplot(outer[0, 0:6])

    # B keeps its previous physical height and top alignment; A alone uses the
    # additional lower part of the first row.
    bwrap = outer[0, 6:18].subgridspec(2, 1, height_ratios=[4.05, 0.80], hspace=0.0)
    bgrid = bwrap[0, 0].subgridspec(1, 12, wspace=0.90)
    ax_b1 = fig.add_subplot(bgrid[0, 0:8])
    ax_b2 = fig.add_subplot(bgrid[0, 9:12])

    cgrid = outer[1, 0:12].subgridspec(2, 1, height_ratios=[2.65, 1.0], hspace=0.16)
    ax_c = fig.add_subplot(cgrid[0, 0])
    ax_ck = fig.add_subplot(cgrid[1, 0])
    ax_d = fig.add_subplot(outer[1, 13:18])
    egrid = outer[2, 0:12].subgridspec(1, 12, wspace=0.72)
    ax_e = fig.add_subplot(egrid[0, 0:8])
    ax_ew = fig.add_subplot(egrid[0, 9:12], projection="polar")
    ax_f = fig.add_subplot(outer[2, 13:18])

    draw_task_definition(ax_a)
    draw_position_atlas(ax_b1, ax_b2, place_records)
    draw_branch_trajectory(ax_c, ax_ck, branch_records)
    draw_layer_heatmap(ax_d, branch_records)
    draw_geometry_phase(ax_e, branch_records)
    draw_geometry_gain_wheel(ax_ew, branch_records)
    draw_effect_forest(ax_f, place_records, branch_records)

    panel_letter(fig, ax_a, "A", dx=0.025)
    panel_letter(fig, ax_b1, "B", dx=0.025)
    panel_letter(fig, ax_c, "C", dx=0.025)
    panel_letter(fig, ax_d, "D", dx=0.050)
    panel_letter(fig, ax_e, "E", dx=0.025)
    panel_letter(fig, ax_f, "F", dx=0.025)

    axes = {"A": ax_a, "B1": ax_b1, "B2": ax_b2, "C1": ax_c, "C2": ax_ck,
            "D": ax_d, "E1": ax_e, "E2": ax_ew, "F": ax_f}
    return fig, axes


def main() -> None:
    fig, _ = build_figure()
    fig.savefig(OUT_PNG, dpi=360, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
