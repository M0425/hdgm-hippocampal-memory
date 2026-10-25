from pathlib import Path
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OP_FILE = DATA / "operation.json"
GATE_FILE = DATA / "gate.json"
CA1_FILE = DATA / "ca1.json"
OUT_PNG = ROOT / "figure3_refined_v2.png"
OUT_PDF = ROOT / "figure3_refined_v2.pdf"

NAVY = "#24324A"
MUTED = "#526077"
GRID = "#DCE4EE"
LIGHT = "#F4F7FA"
ENC = "#EA744C"
MAINT = "#6E81A7"
READ = "#16B7A5"
MODULE_COLORS = {"ec": "#315EF5", "ca3": "#F59E0B", "ca1": "#14B8A6"}
STAGE_COLORS = {"encoding": ENC, "maintenance": MAINT, "readout": READ}
MODULES = ["ec", "ca3", "ca1"]
STAGES = ["encoding", "maintenance", "readout"]
SEED_ORDER = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": .75,
    "axes.edgecolor": NAVY,
    "axes.labelcolor": NAVY,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def panel_letter(ax, letter, x=-0.08, y=1.06):
    # Letters use figure coordinates in main, independent of long axis labels.
    pass


def condition_order():
    return [(stage, module) for stage in STAGES for module in MODULES]


def operation_arrays(data):
    records = sorted(data["records"], key=lambda r: SEED_ORDER.index(r["seed"]))
    order = condition_order()
    normal = np.zeros((len(records), 9))
    long = np.zeros_like(normal)
    for i, record in enumerate(records):
        control_n = record["conditions"]["control"]["normal"]["accuracy"]
        control_l = record["conditions"]["control"]["long"]["accuracy"]
        for j, (stage, module) in enumerate(order):
            cond = record["conditions"][f"operation_cut__{module}__{stage}"]
            normal[i, j] = 100 * (control_n - cond["normal"]["accuracy"])
            long[i, j] = 100 * (control_l - cond["long"]["accuracy"])
    stats_n = data["statistics"]["operation_cut"]["condition_tests"]["normal_accuracy"]
    stats_l = data["statistics"]["operation_cut"]["condition_tests"]["long_accuracy"]
    dz_n, dz_l, sig_n, sig_l = [], [], [], []
    for stage, module in order:
        key = f"operation_cut__{module}__{stage}"
        dz_n.append(stats_n[key]["cohen_dz"])
        dz_l.append(stats_l[key]["cohen_dz"])
        sig_n.append(stats_n[key]["holm_p_across_9_conditions"] < 0.05)
        sig_l.append(stats_l[key]["holm_p_across_9_conditions"] < 0.05)
    return records, order, normal, long, np.array(dz_n), np.array(dz_l), np.array(sig_n), np.array(sig_l)


def effect_color(dz):
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "dz", ["#5874A8", "#EEF2F6", "#F8B57B", "#C94F34"])
    return cmap(colors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=3.1)(dz))


def draw_crown(ax, normal, long, dz_n, dz_l, sig_n, sig_l):
    order = condition_order()
    gap = np.deg2rad(3.5)
    group_gap = np.deg2rad(10)
    width = (2 * np.pi - 3 * group_gap - 6 * gap) / 9
    theta = []
    cursor = 0.0
    for stage_idx in range(3):
        for _ in range(3):
            theta.append(cursor + width / 2)
            cursor += width + gap
        cursor += group_gap - gap
    theta = np.asarray(theta)

    # Every condition owns two complete annular tracks. The pale scaffold remains
    # visible even when the measured effect is zero or negative.
    # A larger central aperture keeps EC→CA3→CA1 visually independent from
    # the inner normal-delay track after the panel is horizontally compressed.
    n0, track_h, track_gap = 0.98, 0.37, 0.07
    l0 = n0 + track_h + track_gap
    max_loss = 30.0
    means_n, means_l = normal.mean(0), long.mean(0)

    for j, ((stage, module), th) in enumerate(zip(order, theta)):
        for base in (n0, l0):
            ax.bar(th, track_h, width=width, bottom=base, color=LIGHT,
                   edgecolor="#E0E7EF", lw=0.75, zorder=1)
        for base, mean, dz, significant in (
            (n0, means_n[j], dz_n[j], sig_n[j]),
            (l0, means_l[j], dz_l[j], sig_l[j]),
        ):
            height = max(0.035, min(abs(mean) / max_loss, 1.0) * track_h)
            ax.bar(th, height, width=width * 0.84, bottom=base,
                   color=effect_color(dz), edgecolor="#111827" if significant else "white",
                   lw=1.5 if significant else 0.65, zorder=3)
        # Module identity is fixed at the inner rim.
        ax.bar(th, 0.035, width=width * 0.72, bottom=n0 - 0.075,
               color=MODULE_COLORS[module], linewidth=0, zorder=5)
        # Real seed-level evidence: normal in the inner track, long in the outer.
        local_theta = th + np.linspace(-0.34, 0.34, normal.shape[0]) * width
        rn = n0 + np.clip(normal[:, j], 0, max_loss) / max_loss * track_h
        rl = l0 + np.clip(long[:, j], 0, max_loss) / max_loss * track_h
        ax.scatter(local_theta, rn, s=6, facecolor="white", edgecolor=NAVY,
                   linewidth=0.35, zorder=6)
        ax.scatter(local_theta, rl, s=6, facecolor=NAVY, edgecolor="white",
                   linewidth=0.25, zorder=6)

    # Negative means are marked with an inward blue notch instead of disappearing.
    for j, mean in enumerate(means_n):
        if mean < 0:
            ax.bar(theta[j], 0.045, width=width * 0.55, bottom=n0 - 0.055,
                   color="#5874A8", linewidth=0, zorder=7)
    for j, mean in enumerate(means_l):
        if mean < 0:
            ax.bar(theta[j], 0.045, width=width * 0.55, bottom=l0 - 0.055,
                   color="#5874A8", linewidth=0, zorder=7)

    # Place condition labels outside the evidence tracks; never cover seeds.
    for stage, x, y, rotation in [("encoding", .78, .97, 0),
                                  ("maintenance", -.005, .53, 90),
                                  ("readout", .72, .02, 0)]:
        ax.text(x, y, stage.title(), transform=ax.transAxes,
                color=STAGE_COLORS[stage], fontsize=11, fontweight="bold",
                ha="center", va="center", rotation=rotation, clip_on=False)

    # Center circuit.
    xs = [0.36, 0.50, 0.64]
    for x, module in zip(xs, MODULES):
        circ = Circle((x, 0.50), 0.058, transform=ax.transAxes, fc="white",
                      ec=MODULE_COLORS[module], lw=1.8)
        ax.add_artist(circ)
        ax.text(x, 0.50, module.upper(), transform=ax.transAxes, ha="center", va="center",
                fontsize=9, fontweight="bold", color=MODULE_COLORS[module])
    for x1, x2 in zip(xs[:-1], xs[1:]):
        ax.add_artist(FancyArrowPatch((x1 + .060, .50), (x2 - .060, .50),
                                     transform=ax.transAxes, arrowstyle="-|>",
                                     mutation_scale=7, lw=.9, color="#91A0B5"))

    handles = [mpl.lines.Line2D([], [], marker="o", linestyle="none", mfc=fc,
                               mec=NAVY, ms=4, label=label)
               for fc, label in [("white", "Normal (inner)"), (NAVY, "Long (outer)")]]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.50, -.015),
              ncol=2, frameon=False, fontsize=9, handletextpad=.3, columnspacing=1)
    ax.set_ylim(0, 2.12)
    ax.set_axis_off()
    panel_letter(ax, "A", x=-0.04, y=1.04)


def draw_barcode(ax, normal, long, sig_n):
    order = condition_order()
    sensitivity = normal[:, :3].mean(1)
    idx = np.argsort(-sensitivity)
    matrix = normal[idx]
    seeds = np.array(SEED_ORDER)[idx]
    corr = np.array([np.corrcoef(normal[i], long[i])[0, 1] for i in idx])
    vmax = max(25, np.nanmax(np.abs(matrix)))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "loss", ["#5874A8", "#F4F7FA", "#F8B57B", "#C94F34"])
    norm = colors.TwoSlopeNorm(vmin=-6, vcenter=0, vmax=vmax)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(range(9), [m.upper() for _, m in order], fontsize=9)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=3)
    ax.set_yticks(range(10), seeds, fontsize=9)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_ylabel("Seed", color=NAVY, labelpad=4)
    for x in [2.5, 5.5]:
        ax.axvline(x, color="white", lw=2.2)
    for y in np.arange(-.5, 10, 1):
        ax.axhline(y, color="white", lw=.55)
    for j, significant in enumerate(sig_n):
        if significant:
            ax.text(j, -1.05, "*", ha="center", va="center", fontsize=11,
                    color="#111827", fontweight="bold", clip_on=False)
    # Mean profile above the raster.
    for j, value in enumerate(normal.mean(0)):
        ax.text(j, 10.1, f"{value:.1f}", ha="center", va="center", fontsize=9,
                color=NAVY, fontweight="bold", clip_on=False)
    ax.text(-0.65, 10.1, "Mean", ha="right", va="center", fontsize=9, color=MUTED)

    # Normal-long consistency strip stays inside the panel boundary.
    consistency_cmap = mpl.colormaps["GnBu"]
    for y, value in enumerate(corr):
        ax.add_patch(Rectangle((9.02, y - .5), .87, 1.0,
                               facecolor=consistency_cmap(np.clip(value, 0, 1)),
                               edgecolor="white", lw=.55, clip_on=False))
        ax.text(9.455, y, f"{value:.2f}", ha="center", va="center", fontsize=8.5,
                color="white" if value > .62 else NAVY, clip_on=False)
    ax.text(9.455, -.95, "r(N,L)", ha="center", va="center", fontsize=8.5,
            color=MUTED, clip_on=False)
    ax.set_xlim(-.5, 9.9)
    ax.set_ylim(9.5, -1.35)
    for center, stage in zip([1, 4, 7], STAGES):
        ax.text(center, 1.105, stage.title(), transform=ax.get_xaxis_transform(),
                ha="center", va="center",
                fontsize=10, fontweight="bold", color=STAGE_COLORS[stage], clip_on=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbax = ax.inset_axes([0.07, -0.20, 0.80, 0.035])
    cb = plt.colorbar(im, cax=cbax, orientation="horizontal", ticks=[-5, 0, 10, 20, 30])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9, length=0, colors=MUTED)
    cb.ax.set_title("Normal-delay accuracy loss (pp)", fontsize=10, color=MUTED, pad=5)


def ci95(values):
    values = np.asarray(values, float)
    return 1.96 * values.std(ddof=1) / np.sqrt(len(values))


def bootstrap_mean_ci(values, seed=0, n_boot=5000):
    values = np.asarray(values, float)
    if np.allclose(values, values[0]):
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def draw_gate_ablation(ax, data):
    records = sorted(data["records"], key=lambda r: SEED_ORDER.index(r["seed"]))
    groups = []
    for module in MODULES:
        zero, clamp = [], []
        for r in records:
            ctrl = r["conditions"]["control"]["normal"]["accuracy"]
            zero.append(100 * (ctrl - r["conditions"][f"zero__{module}__encoding"]["normal"]["accuracy"]))
            clamp.append(100 * (ctrl - r["conditions"][f"mean_clamp__{module}__encoding"]["normal"]["accuracy"]))
        groups.append((np.array(zero), np.array(clamp)))

    for idx, (module, (zero, clamp)) in enumerate(zip(MODULES, groups)):
        y = 2 - idx
        for vals, offset, color, label, marker in [(zero, .14, "#C86666", "Gate = 0", "o"),
                                           (clamp, -.14, "#7568B5", "Per-unit mean", "s")]:
            parts = ax.violinplot(vals, positions=[y + offset], vert=False, widths=.25,
                                  showmeans=False, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(.22); body.set_linewidth(.7)
            jitter = np.linspace(-.06, .06, len(vals))
            ax.scatter(vals, y + offset + jitter, s=17, marker=marker, facecolor="white", edgecolor=color,
                       linewidth=.8, zorder=4)
            mean, err = vals.mean(), ci95(vals)
            ax.errorbar(mean, y + offset, xerr=err, fmt=marker, ms=5.5, color=color,
                        mec="white", mew=.6, capsize=2.5, lw=1.2, zorder=5)
        ax.text(-.035, y, module.upper(), transform=ax.get_yaxis_transform(),
                color=MODULE_COLORS[module], ha="right", va="center",
                fontweight="bold", fontsize=11, clip_on=False)
    ax.axvline(0, color="#9CADC2", lw=.9, ls="--")
    # Preserve all observed values (the supplied script clipped the long tails).
    all_values = np.concatenate([v for pair in groups for v in pair])
    ax.set_xlim(min(-8, all_values.min() - 1), max(40.5, all_values.max() + 1))
    ax.set_xticks([-10, 0, 10, 20, 30, 40])
    ax.set_yticks([])
    ax.set_xlabel("Accuracy loss at encoding (pp)")
    ax.grid(axis="x", color=GRID, lw=.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_ylim(-.45, 2.85)
    handles = [mpl.lines.Line2D([0], [0], color="#C86666", marker="o", lw=1.2, label="Gate = 0"),
               mpl.lines.Line2D([0], [0], color="#7568B5", marker="s", lw=1.2, label="Per-unit mean")]
    ax.legend(handles=handles, frameon=False, loc="upper center",
              bbox_to_anchor=(.5, 1.09), ncol=2, fontsize=9,
              handlelength=1.3, columnspacing=1.2, borderaxespad=0)
    panel_letter(ax, "B", x=-0.09, y=1.08)


def draw_ca1_alignment(ax, data, raw_ax=None):
    records = sorted(data["records"], key=lambda r: SEED_ORDER.index(r["seed"]))
    tests = data["statistics"]["gate_trial_shuffle"]["condition_tests"]
    metrics = [
        ("Behavior", "normal_accuracy", 100, "pp"),
        ("History decode", "delay1_cueA_decoding", 100, "pp"),
        ("State distance", "delay1_state_distance", 1, "Δ"),
        ("Decision decode", "decision_pre_class_decoding", 100, "pp"),
    ]
    offsets = [-.20, 0.0, .20]
    base_y = [3, 2, 1, 0]
    stage_markers = ["o", "s", "^"]
    for divider in [.5, 1.5, 2.5]:
        ax.axhline(divider, color=GRID, lw=.7, zorder=0)
    ax.axvline(0, color="#9CADC2", lw=.9, ls="--", zorder=0)

    for metric_idx, ((label, metric, scale, unit), y0) in enumerate(zip(metrics, base_y)):
        for stage_idx, (stage, offset, marker) in enumerate(zip(STAGES, offsets, stage_markers)):
            key = f"gate_trial_shuffle__ca1__{stage}"
            diffs = []
            for record in records:
                control = record["conditions"]["control"]
                condition = record["conditions"][key]
                if metric == "normal_accuracy":
                    diff = control["normal"]["accuracy"] - condition["normal"]["accuracy"]
                else:
                    diff = control["representation"][metric] - condition["representation"][metric]
                diffs.append(diff * scale)
            diffs = np.asarray(diffs)
            sd = diffs.std(ddof=1)
            standardized = diffs / sd if sd > 1e-12 else np.zeros_like(diffs)
            stat = tests[metric][key]
            mean = stat["cohen_dz"]
            lo, hi = bootstrap_mean_ci(standardized, seed=100 + 10*metric_idx + stage_idx)
            y = y0 + offset
            jitter = np.linspace(-.035, .035, len(standardized))
            color = STAGE_COLORS[stage]
            ax.scatter(standardized, y + jitter, s=11, facecolor="white",
                       edgecolor=color, linewidth=.65, alpha=.85, zorder=2)
            significant = stat["holm_p_across_9_conditions"] < .05
            ax.errorbar(mean, y, xerr=[[mean-lo], [hi-mean]], fmt=marker,
                        ms=5.3, mfc=color, mec="#111827" if significant else "white",
                        mew=1.1 if significant else .7, ecolor=color, elinewidth=1.15,
                        capsize=2.4, zorder=4)
            raw = stat["mean_difference"] * scale
            zero_threshold = .005 if unit == "Δ" else .05
            if abs(raw) < zero_threshold:
                raw_text = "0"
            else:
                raw_text = f"{raw:+.2f}" if unit == "Δ" else f"{raw:+.1f}"
            label_ax = raw_ax if raw_ax is not None else ax
            raw_x = .45 if raw_ax is not None else 3.72
            label_ax.text(raw_x, y, raw_text, ha="center", va="center", color=color,
                    fontsize=9, fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", pad=.15), zorder=6)
            if significant:
                label_ax.text(.92 if raw_ax is not None else 4.08, y, "*", ha="left", va="center", color="#111827",
                        fontsize=10, fontweight="bold", zorder=6)

    ax.set_xlim(-2.15, 3.25 if raw_ax is not None else 4.28); ax.set_ylim(-.48, 3.48)
    ax.set_yticks(base_y, [m[0] for m in metrics], fontsize=10)
    ax.set_xticks([-2, -1, 0, 1, 2, 3])
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xlabel(r"Paired effect size $d_z$")
    ax.grid(axis="x", color=GRID, lw=.65)
    ax.spines[["top", "right", "left"]].set_visible(False)
    handles = [mpl.lines.Line2D([0], [0], marker=m, color=c, lw=1.2,
                                markerfacecolor=c, markeredgecolor="white",
                                label=s.title(), markersize=5)
               for s, c, m in zip(STAGES, [ENC, MAINT, READ], stage_markers)]
    ax.legend(handles=handles, frameon=False, loc="lower left",
              bbox_to_anchor=(0, 1.035), ncol=3, fontsize=9,
              columnspacing=.8, handletextpad=.3, borderaxespad=0)
    label_ax = raw_ax if raw_ax is not None else ax
    if raw_ax is not None:
        raw_ax.set_ylim(ax.get_ylim())
        raw_ax.set_xlim(0, 1)
        raw_ax.set_axis_off()
    label_ax.text(.45 if raw_ax is not None else 3.72, 3.50, "Raw Δ", ha="center", va="bottom", fontsize=9,
            color=MUTED, fontweight="bold")
    panel_letter(ax, "C", x=-0.12, y=1.09)


def draw_ca1_transformation(ax, data):
    records = sorted(data["records"], key=lambda r: SEED_ORDER.index(r["seed"]))
    pre = np.array([100 * r["full_representations"]["pre_ca1_state"]["class"] for r in records])
    out = np.array([100 * r["full_representations"]["output_state"]["class"] for r in records])
    for a, b in zip(pre, out):
        ax.plot([0, 1], [a, b], color="#B8C5D6", lw=.8, zorder=1)
        ax.scatter(0, a, s=20, facecolor="white", edgecolor="#667995", lw=.8, zorder=2)
        ax.scatter(1, b, s=20, facecolor="white", edgecolor=READ, lw=.9, zorder=2)
    means = [pre.mean(), out.mean()]
    errs = [ci95(pre), ci95(out)]
    ax.errorbar([0, 1], means, yerr=errs, fmt="o", ms=7, color=READ,
                ecolor=READ, capsize=3, lw=1.2, zorder=4)
    stat = data["statistics"]["secondary_CA1_output_integration"]["output_minus_pre_ca1_state"]
    ax.text(.5, 45.9, f"+{100*stat['mean_diff']:.1f} pp*", color=READ,
            fontweight="bold", ha="center", fontsize=12)
    ax.axhline(25, color="#9CADC2", lw=.8, ls="--")
    ax.text(.98, 25.4, "Chance", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=9, color=MUTED)
    ax.set_xlim(-.25, 1.25); ax.set_ylim(20, 47)
    ax.set_xticks([0, 1], ["Pre-CA1", "Output"])
    ax.set_ylabel("Class decoding (%)")
    ax.grid(axis="y", color=GRID, lw=.7)
    ax.spines[["top", "right"]].set_visible(False)
    panel_letter(ax, "D", x=-0.22, y=1.08)


def draw_controls(ax, data):
    records = sorted(data["records"], key=lambda r: SEED_ORDER.index(r["seed"]))
    stats = data["statistics"]["corrective_output_probe"]
    labels = ["Fixed open", "Calibrated mean", "Running mean", "Lagged"]
    conditions = ["fixed_open", "calibrated_mean", "causal_running_mean", "one_step_lagged"]
    keys = ["full_minus_fixed_open", "full_minus_calibrated_mean",
            "full_minus_causal_running_mean", "full_minus_one_step_lagged"]
    colors_ = [READ, READ, "#5078A4", READ]
    y_positions = [3, 2, 1, 0]
    for row, (y, label, condition, key, color) in enumerate(zip(y_positions, labels, conditions, keys, colors_)):
        s = stats[key]
        values = np.array([
            100 * (r["conditions"]["full"]["output_probe"]["class"] -
                   r["conditions"][condition]["output_probe"]["class"])
            for r in records
        ])
        violin = ax.violinplot(values, positions=[y], vert=False, widths=.52,
                               showmeans=False, showmedians=False, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(.16)
            body.set_linewidth(.7)
        jitter = np.linspace(-.13, .13, len(values))
        ax.scatter(values, y + jitter, s=15, facecolor="white", edgecolor=color,
                   linewidth=.75, alpha=.95, zorder=3)
        mean, lo, hi = 100*s["mean_diff"], 100*s["ci95_low"], 100*s["ci95_high"]
        ax.errorbar(mean, y, xerr=[[mean-lo], [hi-mean]], fmt="D", ms=5.2,
                    mfc=color, mec="white", mew=.7, color=color, ecolor=color,
                    capsize=2.5, lw=1.25, zorder=5)
        if s["holm_p"] < .05:
            ax.text(hi + .7, y, "*", ha="left", va="center", fontsize=11,
                    color="#111827", fontweight="bold")
    ax.axvline(0, color="#9CADC2", lw=.9, ls="--")
    ax.set_yticks(y_positions, labels, fontsize=10)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_ylim(-.5, 3.5); ax.set_xlim(-25, 17)
    ax.set_xlabel("Full − substitute decoding (pp)")
    ax.grid(axis="x", color=GRID, lw=.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    panel_letter(ax, "E", x=-0.16, y=1.09)


def main():
    op = load_json(OP_FILE)
    gate = load_json(GATE_FILE)
    ca1 = load_json(CA1_FILE)
    _, _, normal, long, dz_n, dz_l, sig_n, sig_l = operation_arrays(op)

    # Fixed review master: 14 x 7.8 in. Typography is enlarged in physical
    # points, not just via DPI. Do not shrink indiscriminately to one column.
    fig = plt.figure(figsize=(14, 7.8), facecolor="white")
    ax_a1 = fig.add_axes([.025, .565, .270, .370], projection="polar")
    ax_a2 = fig.add_axes([.335, .585, .365, .340])
    ax_b = fig.add_axes([.775, .585, .210, .340])
    ax_c = fig.add_axes([.105, .090, .275, .340])
    ax_c_raw = fig.add_axes([.398, .090, .038, .340])
    ax_d = fig.add_axes([.495, .090, .150, .340])
    ax_e = fig.add_axes([.785, .090, .200, .340])

    draw_crown(ax_a1, normal, long, dz_n, dz_l, sig_n, sig_l)
    draw_barcode(ax_a2, normal, long, sig_n)
    draw_gate_ablation(ax_b, gate)
    draw_ca1_alignment(ax_c, op, raw_ax=ax_c_raw)
    draw_ca1_transformation(ax_d, ca1)
    draw_controls(ax_e, ca1)

    for label, x, y in [("A", .020, .975), ("B", .738, .975),
                         ("C", .020, .477), ("D", .455, .477), ("E", .707, .477)]:
        fig.text(x, y, label, fontsize=19, weight="bold", color=NAVY, va="top")

    fig.savefig(OUT_PNG, dpi=300, facecolor="white")
    fig.savefig(OUT_PDF, dpi=300, facecolor="white")
    fig.savefig(ROOT / "figure3_refined_v2.svg", facecolor="white")
    fig.savefig(ROOT / "preview.png", dpi=145, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
