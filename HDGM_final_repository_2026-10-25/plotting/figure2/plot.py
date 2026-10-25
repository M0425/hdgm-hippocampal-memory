from __future__ import annotations

import csv
import json
import argparse
import hashlib
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle, Wedge
from matplotlib.path import Path as MplPath


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / 'data' if (HERE / 'data').exists() else HERE / 'figure2_refined_outputs' / 'data'
if not DEFAULT_DATA.exists():
    DEFAULT_DATA = Path(r'C:\Users\马明哲\Desktop\建模\pythonProject32')
DEFAULT_OUT = HERE
parser = argparse.ArgumentParser(description='Figure 2: source-data reconstruction, refined A and semantic palette.')
parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
parser.add_argument('--dpi', type=int, default=250)
args = parser.parse_args()
ROOT = args.data_root
DATA = ROOT / 'figure2_missing_outputs'
OUT = args.out_dir / 'figure2_E_rebalanced_v2.pdf'
PREVIEW = args.out_dir / 'figure2_E_rebalanced_v2.png'

INK = "#26364B"
MUTED = "#68778D"
GRID = "#D8E1E8"
TEM = "#318A98"
HDGM = "#A34E7A"
CUE_A = "#6D78B6"
CUE_B = "#D1A04C"
DECISION = "#609888"
PURPLE = "#8970AE"
WHITE = "#FFFFFF"

# NumPy equivalents of the original sklearn metrics; no additional dependency.
def confusion_matrix(y_true, y_pred, labels=None):
    labels = np.unique(np.concatenate([y_true, y_pred])) if labels is None else np.asarray(labels)
    lookup = {v:i for i,v in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for a,b in zip(y_true, y_pred):
        if a in lookup and b in lookup:
            cm[lookup[a], lookup[b]] += 1
    return cm

def f1_score(y_true, y_pred, average='macro'):
    assert average == 'macro'
    cm = confusion_matrix(y_true, y_pred)
    denom = cm.sum(0) + cm.sum(1)
    return np.divide(2*np.diag(cm), denom, out=np.zeros(len(cm), dtype=float), where=denom!=0).mean()

SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]
CHECKPOINTS = [10, 25, 50, 75, 100, 150]

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.3,
        "axes.labelsize": 7.4,
        "axes.titlesize": 8.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "savefig.facecolor": "white",
    }
)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def mean_ci(values):
    values = np.asarray(values, float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    half = 1.96 * values.std(ddof=1) / np.sqrt(len(values))
    return mean, mean - half, mean + half


def style_axis(ax, *, grid_axis="both"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, width=0.6, length=2.5)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.55, zorder=0)


def panel_letter(ax, letter, x=-0.10, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=18, fontweight="bold",
            color="#111827", ha="left", va="top", clip_on=False)


# ---------- data ----------
records = json.loads((ROOT / "fourclass_rapid_acquisition_v4_primary_results.json").read_text(encoding='utf-8'))["records"]
records = sorted(records, key=lambda r: r["seed"])

controls = read_csv(DATA / "figure2_history_controls.csv")
shortcuts = read_csv(DATA / "figure2_shortcut_diagnostics.csv")
delays = read_csv(DATA / "figure2_delay_transfer.csv")
pred = np.load(DATA / "figure2_endpoint_predictions.npz", allow_pickle=True)
assert [r['seed'] for r in records] == SEEDS
assert len(delays) == 120

control_lookup = {
    (int(r["seed"]), r["model"], r["condition"]): {
        "accuracy": float(r["accuracy"]),
        "loss": float(r["loss_vs_full_pp"]),
    }
    for r in controls
}
shortcut_lookup = {(int(r["seed"]), r["diagnostic"]): 100 * float(r["accuracy"]) for r in shortcuts}
delay_lookup = {(int(r["seed"]), r["model"], int(r["inserted_steps_per_delay"])): float(r["accuracy"]) for r in delays}


# ---------- canvas and asymmetric mosaic ----------
fig = plt.figure(figsize=(19.2, 12.2), facecolor="white")
outer = fig.add_gridspec(3, 100, left=0.049, right=0.974, top=0.941, bottom=0.08,
                        height_ratios=[1.0,1.05,1.15], hspace=0.39)


# ---------- A: fully editable experimental schematic ----------
axA = fig.add_subplot(outer[0, :34])
axA.set_axis_off()
axA.set_xlim(0, 1)
axA.set_ylim(0, 1)
panel_letter(axA, "A", x=-0.07, y=1.10)
axA.text(0, 1.015, "Controlled history evaluation", color=INK, fontweight="bold", fontsize=10)

def a_arrow(p, q, col=MUTED, lw=1.15, rad=0):
    axA.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9,
        connectionstyle=f"arc3,rad={rad}", color=col, lw=lw, zorder=3))

def a_box(x,y,w,h,col,fill=WHITE,lw=.9):
    axA.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.006,rounding_size=0.014",
        facecolor=fill,edgecolor=col,linewidth=lw))

def a_tile(x,y,col,pattern):
    # Layered cue tile and a distinct high-contrast pattern, not fabricated neural data.
    a_box(x+.006,y-.008,.064,.082,'#E0E5EB',fill='#EDF0F4',lw=.6)
    a_box(x,y,.064,.082,col)
    if pattern == 0:
        for i in range(3):
            axA.plot([x+.01+i*.015,x+.027+i*.015],[y+.012,y+.07],color=col,lw=.85)
    else:
        for i in range(2):
            for j in range(2):
                axA.plot(x+.020+i*.025,y+.023+j*.035,'o',ms=2.4,color=col)

axA.text(.16,.93,"Past cues",ha="center",fontsize=7.4,color=MUTED)
axA.text(.45,.93,"Memory",ha="center",fontsize=7.4,color=MUTED)
axA.text(.68,.93,"Same observation",ha="center",fontsize=7.4,color=INK)
axA.text(.93,.93,"Choice",ha="center",fontsize=7.4,color=MUTED)
for y,label,col,pat,choice in [(.78,"History 1",CUE_A,0,"1"),(.62,"History 2",CUE_B,1,"2")]:
    axA.text(0,y+.012,label,color=col,fontsize=7.3,fontweight='bold',va='center')
    a_tile(.145,y-.027,col,pat);a_tile(.234,y-.027,col,1-pat)
    a_arrow((.315,y+.012),(.378,y+.012),col)
    for j in range(3):
        axA.plot(.403+j*.035,y+.012,'o',ms=4+j*.35,mfc=col,mec=WHITE,mew=.5,zorder=4)
    axA.plot([.402,.474],[y+.012,y+.012],color=col,lw=1,zorder=2)
    a_arrow((.495,y+.012),(.583,y+.012),col)
    a_box(.59,y-.041,.17,.112,'#BAC6CE',fill='#F5F6F8')
    # Identical geometric sensory scene in BOTH rows.
    for xx in [.62,.70]:
        axA.add_patch(Rectangle((xx,y-.02),.025,.045,facecolor='#D8DFE5',edgecolor='#94A5B3',lw=.5))
        axA.plot([xx-.004,xx+.012,xx+.03],[y+.025,y+.052,y+.025],color='#94A5B3',lw=.7)
    axA.plot([.66,.672,.685],[y-.032,y+.004,y-.032],color='#B4BDC5',lw=.7)
    a_arrow((.774,y+.012),(.868,y+.012),col)
    axA.plot(.922,y+.012,marker='D' if choice=='2' else 'o',ms=13,color=col,mec=WHITE,mew=.8)
    axA.text(.922,y+.012,choice,fontsize=7,color=WHITE,ha='center',va='center',fontweight='bold')
axA.text(.48,.51,"Matched observation does not determine the target",fontsize=7.2,
         color=INK,ha='center',fontstyle='italic')
axA.plot([0,1],[.475,.475],color=GRID,lw=.6)
axA.text(0,.433,"Information controls",fontsize=8.0,fontweight='bold',color=INK)
# Four distinct frozen-model interventions, corresponding exactly to panel D.
for y,title,col,kind in [(.335,'Full history',CUE_A,0),(.232,'History reset',CUE_B,1),
                        (.129,'Decision-only',TEM,2),(.026,'Timing-only',DECISION,3)]:
    a_box(.006,y-.018,.42,.075,col,fill=WHITE,lw=.65)
    axA.text(.02,y+.02,title,fontsize=7.0,color=col,fontweight='bold',va='center')
    if kind < 2:
        axA.plot([.25,.30,.35],[y+.02]*3,color=col,lw=.9)
        axA.scatter([.25,.30,.35],[y+.02]*3,s=12,c=col,zorder=3)
        if kind==1: axA.text(.30,y+.02,'×',fontsize=13,color='#B66A63',ha='center',va='center',zorder=4)
    elif kind==2:
        axA.add_patch(Rectangle((.284,y+.002),.053,.04,facecolor='#DDE7EB',edgecolor=col,lw=.7))
        axA.plot([.29,.31,.33],[y+.006,y+.03,y+.006],color=col,lw=.8)
    else:
        axA.plot(.31,y+.02,'o',ms=10,mfc=WHITE,mec=col,mew=.9)
        axA.plot([.31,.31,.323],[y+.044,y+.02,y+.009],color=col,lw=.8)
    a_arrow((.44,y+.02),(.50,y+.02),col,lw=.8)
axA.plot([.50,.50],[.046,.355],color=MUTED,lw=.9)
a_arrow((.50,.202),(.58,.202),MUTED)
# Three offset translucent computational layers, entirely vector and editable.
for dx,dy in [(.025,.024),(.013,.012),(0,0)]:
    a_box(.585+dx,.084+dy,.22,.190,'#9CB2C3',fill='#F0F5F8',lw=.65)
for i in range(3):
    for j in range(3):
        for k in range(3):
            if i<2:
                axA.plot([.619+i*.069,.619+(i+1)*.069],[.143+j*.06,.143+k*.06],
                         color='#BCC9D2',lw=.4,zorder=2)
        axA.plot(.619+i*.069,.143+j*.06,'o',ms=4.6,mfc=[TEM,CUE_A,HDGM][i],mec=WHITE,mew=.45,zorder=4)
# Padlock rather than an animal, surgical apparatus, or invented measurement.
# The explicit 'Frozen model' label replaces the padlock to leave room for type.
axA.text(.70,.324,'Frozen model',fontsize=7.0,color=INK,ha='center',fontweight='bold')
a_arrow((.838,.203),(.888,.203),MUTED)
for i,col in enumerate([CUE_A,CUE_B,PURPLE,DECISION]):
    axA.add_patch(Rectangle((.895,.125+i*.04),.085,.031,facecolor=col,alpha=.22,edgecolor=col,lw=.6))
    axA.text(.937,.140+i*.04,str(i+1),ha='center',va='center',color=INK,fontsize=6.4)
axA.text(.937,.325,'Readout',ha='center',fontsize=7.4,color=INK)
axA.text(.5,-.045,'Shortcut nulls: separate observation-only / timing-only linear classifiers',
         ha='center',fontsize=6.8,color=MUTED)


# ---------- B: fixed-budget acquisition, first visual center ----------
gsB = outer[0, 37:].subgridspec(2, 4, height_ratios=[0.33, 0.67], width_ratios=[1, 1, 1, 0.82], hspace=0.33, wspace=0.46)
axBcurve = fig.add_subplot(gsB[0, :3])
panel_letter(axBcurve, "B", x=-0.07, y=1.23)
curve_mats = {}
for model, color, key in [("TEM", TEM, "tem_normal"), ("HDGM", HDGM, "hdgm_normal")]:
    mat = np.array([[r["checkpoint_curve"][str(cp)][key] for cp in CHECKPOINTS] for r in records])
    curve_mats[model] = mat
    means = mat.mean(0)
    lo = means - 1.96 * mat.std(0, ddof=1) / np.sqrt(len(mat))
    hi = means + 1.96 * mat.std(0, ddof=1) / np.sqrt(len(mat))
    axBcurve.fill_between(CHECKPOINTS, lo, hi, color=color, alpha=0.14, linewidth=0)
    axBcurve.plot(CHECKPOINTS, means, "o-", color=color, lw=1.8, ms=3.2, label=model)
axBcurve.axhline(0.25, color="#94A3B8", ls=(0, (3, 3)), lw=0.8)
axBcurve.text(10, 0.262, "25% chance", color=MUTED, fontsize=5.8, va="bottom")
axBcurve.set_ylim(0.20, 0.63)
axBcurve.set_xlim(8, 168)
axBcurve.set_ylabel("Mean accuracy")
axBcurve.set_xticks([])
for model, color in [("TEM", TEM), ("HDGM", HDGM)]:
    axBcurve.text(156, curve_mats[model].mean(0)[-1], model, color=color,
                  fontsize=6.1, fontweight="bold", va="center")
axBcurve.set_title("Fixed-budget acquisition", loc="left", fontweight="bold", color=INK, pad=4)
style_axis(axBcurve, grid_axis="y")

cmap = LinearSegmentedColormap.from_list("acc", ["#FAF7F4", "#F0D8DD", "#CB94AE", HDGM, "#583754"])
tem_cmap = LinearSegmentedColormap.from_list("tem_acc", ["#F5F9F8", "#DAE9E5", "#8FC3C4", TEM, "#244A69"])
axBh1 = fig.add_subplot(gsB[1, :3])
axBh1.set_axis_off()
subB = gsB[1, :3].subgridspec(2, 1, hspace=0.32)
for i, (model, color) in enumerate([("HDGM", HDGM), ("TEM", TEM)]):
    axh = fig.add_subplot(subB[i, 0])
    mat = curve_mats[model]
    axh.imshow(mat, cmap=cmap if model=='HDGM' else tem_cmap, vmin=0.20, vmax=0.85, aspect="auto", interpolation="nearest")
    axh.set_yticks(range(10), SEEDS)
    axh.set_xticks(range(6), CHECKPOINTS if i == 1 else [])
    if i == 1:
        axh.set_xlabel("Evaluation checkpoint (training updates)", labelpad=2)
    axh.set_title(model, loc="left", color=color, fontweight="bold", fontsize=7.4, pad=2)
    axh.tick_params(length=0, colors=MUTED, labelsize=5.7)
    axh.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    axh.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    axh.grid(which="minor", color=WHITE, linewidth=0.8)
    axh.tick_params(which="minor", bottom=False, left=False)
    for s in axh.spines.values():
        s.set_visible(False)
    # Numeric keys disclose the exact unchanged heatmap normalization.
    keyax=axh.inset_axes([.79,1.18,.17,.055])
    keyax.imshow(np.linspace(.20,.85,128)[None,:],aspect='auto',cmap=cmap if model=='HDGM' else tem_cmap,
                 vmin=.20,vmax=.85,extent=[.20,.85,0,1])
    keyax.set_xticks([]);keyax.set_yticks([])
    keyax.text(-.06,.5,'.20',transform=keyax.transAxes,ha='right',va='center',fontsize=7,color=MUTED)
    keyax.text(1.06,.5,'.85',transform=keyax.transAxes,ha='left',va='center',fontsize=7,color=MUTED)
    for sp in keyax.spines.values():sp.set_visible(False)
axBpair = fig.add_subplot(gsB[:, 3])
tem_auc = np.array([r["tem_early_aulc"] for r in records])
hdgm_auc = np.array([r["hdgm_early_aulc"] for r in records])
for a, b in zip(tem_auc, hdgm_auc):
    axBpair.plot([0, 1], [a, b], color="#CBD5E1", lw=0.7, zorder=1)
axBpair.scatter(np.zeros(10), tem_auc, s=16, facecolors=WHITE, edgecolors=TEM, linewidths=0.8, zorder=2)
axBpair.scatter(np.ones(10), hdgm_auc, s=16, facecolors=WHITE, edgecolors=HDGM, linewidths=0.8, zorder=2)
for x, vals, color in [(0, tem_auc, TEM), (1, hdgm_auc, HDGM)]:
    m, lo, hi = mean_ci(vals)
    axBpair.vlines(x, lo, hi, color=color, lw=1.5)
    axBpair.scatter(x, m, s=34, color=color, edgecolors=WHITE, linewidths=0.7, zorder=3)
axBpair.set_xlim(-0.45, 1.45)
axBpair.set_ylim(0.23, 0.66)
axBpair.set_xticks([0, 1], ["TEM", "HDGM"])
axBpair.set_ylabel("Normalized acquisition AUC")
axBpair.set_title("Cumulative acquisition", fontweight="bold", color=INK, pad=4)
axBpair.text(
    0.05,
    0.96,
    "ΔAUC = +9.8 pp\n95% CI [3.0, 16.5]",
    transform=axBpair.transAxes,
    ha="left",
    va="top",
    fontsize=5.5,
    color=MUTED,
    bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.88, "pad": 1.0},
)
style_axis(axBpair, grid_axis="y")


# ---------- C: endpoint joint-history structure, second visual center ----------
gsC = outer[1, :59].subgridspec(1, 3, width_ratios=[0.30, 0.43, 0.27], wspace=0.45)
axCperf = fig.add_subplot(gsC[0, 0])
panel_letter(axCperf, "C", x=-0.18, y=1.08)
full = pred["condition"] == "full_history"
perf = {m: {} for m in ("TEM", "HDGM")}
for model in ("TEM", "HDGM"):
    for seed in SEEDS:
        mask = full & (pred["model"] == model) & (pred["seed"] == seed)
        yt, yp = pred["y_true"][mask], pred["y_pred"][mask]
        perf[model][seed] = (float((yt == yp).mean()), float(f1_score(yt, yp, average="macro")))
for seed in SEEDS:
    a = perf["TEM"][seed]
    b = perf["HDGM"][seed]
    axCperf.plot([a[0], b[0]], [a[1], b[1]], color="#D5DEE6", lw=0.7, zorder=1)
for model, color in [("TEM", TEM), ("HDGM", HDGM)]:
    pts = np.array([perf[model][s] for s in SEEDS])
    axCperf.scatter(pts[:, 0], pts[:, 1], s=18, facecolors=WHITE, edgecolors=color, linewidths=0.9, zorder=2)
    mean = pts.mean(0)
    axCperf.scatter(*mean, s=30, color=color, edgecolors=WHITE, linewidths=0.7, zorder=3)
    axCperf.text(mean[0] + 0.012, mean[1] + (0.015 if model == "HDGM" else -0.025), model,
                 color=color, fontsize=6.3, fontweight="bold")
axCperf.plot([0.15, 0.70], [0.15, 0.70], color="#C7D2DC", lw=0.8, ls=(0, (3, 3)))
axCperf.set_xlim(0.17, 0.68)
axCperf.set_ylim(0.16, 0.72)
axCperf.set_xlabel("Held-out accuracy")
axCperf.set_ylabel("Held-out macro-F1")
axCperf.set_title("Endpoint performance", loc="left", fontweight="bold", color=INK)
style_axis(axCperf)

axCstruct = fig.add_subplot(gsC[0, 1])
axCstruct.set_aspect("equal")
axCstruct.set_xlim(-0.1, 2.1)
axCstruct.set_ylim(-0.1, 2.1)
axCstruct.axis("off")
axCstruct.set_title("Partial joint-history decision structure", fontweight="bold", color=INK, pad=2)
axCstruct.text(1.0, 2.03, "HDGM · ring = correct · edges = selected-error share (%)",
               fontsize=5.5, color=MUTED, ha="center")
mask_hdgm = full & (pred["model"] == "HDGM")
yt_h, yp_h = pred["y_true"][mask_hdgm], pred["y_pred"][mask_hdgm]
cm = confusion_matrix(yt_h, yp_h, labels=[0, 1, 2, 3])
class_acc = np.diag(cm) / cm.sum(1)
positions = {0: (0.35, 1.55), 1: (1.65, 1.55), 2: (0.35, 0.45), 3: (1.65, 0.45)}
pair_edges = [(0, 1, CUE_B), (0, 2, "#8AA3A5"), (1, 3, "#8AA3A5"), (2, 3, CUE_B)]
pair_vals = np.array([cm[i, j] + cm[j, i] for i, j, _ in pair_edges], float)
pair_pct = 100 * pair_vals / pair_vals.sum()
for (i, j, color), pct in zip(pair_edges, pair_pct):
    x1, y1 = positions[i]
    x2, y2 = positions[j]
    axCstruct.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                        connectionstyle="arc3,rad=0.08", color=color, alpha=0.75, lw=1.8))
    axCstruct.text((x1 + x2) / 2, (y1 + y2) / 2 + (0.12 if y1 == y2 else 0), f"{pct:.0f}",
                   color=color, fontsize=5.8, ha="center", fontweight="bold")
for c, (x, y) in positions.items():
    axCstruct.add_patch(Circle((x, y), 0.30, facecolor=WHITE, edgecolor="#D7DEE3", lw=5, zorder=3))
    axCstruct.add_patch(Wedge((x, y), 0.30, 90, 90 + 360 * class_acc[c], width=0.085,
                              facecolor=HDGM, edgecolor="none", zorder=4))
    a, b = divmod(c, 2)
    axCstruct.text(x, y + 0.035, f"A{a}B{b}", ha="center", va="center", fontsize=6.7, fontweight="bold", color=INK, zorder=5)
    axCstruct.text(x, y - 0.09, f"{100 * class_acc[c]:.0f}%", ha="center", va="center", fontsize=6.3, color=HDGM, zorder=5)

axCdiff = fig.add_subplot(gsC[0, 2])
class_diffs = []
for cls in range(4):
    vals = []
    for seed in SEEDS:
        model_acc = {}
        for model in ("TEM", "HDGM"):
            mask = full & (pred["model"] == model) & (pred["seed"] == seed) & (pred["y_true"] == cls)
            model_acc[model] = (pred["y_pred"][mask] == cls).mean()
        vals.append(100 * (model_acc["HDGM"] - model_acc["TEM"]))
    class_diffs.append(np.asarray(vals))
for cls, vals in enumerate(class_diffs):
    y = 3 - cls
    jitter = np.linspace(-0.07, 0.07, len(vals))
    axCdiff.scatter(vals, y + jitter, s=16, facecolors=WHITE, edgecolors=HDGM, linewidths=0.8)
    m, lo, hi = mean_ci(vals)
    axCdiff.hlines(y, lo, hi, color=HDGM, lw=1.5)
    axCdiff.scatter(m, y, marker="D", s=28, color=HDGM, edgecolors=WHITE, linewidths=0.6, zorder=3)
axCdiff.axvline(0, color="#9AA7B2", lw=0.8, ls=(0, (3, 3)))
axCdiff.set_yticks(range(4), ["A1B1", "A1B0", "A0B1", "A0B0"])
axCdiff.set_xlabel("Δ accuracy\n(HDGM - TEM; pp)")
axCdiff.set_title("Class-wise advantage", fontweight="bold", color=INK)
style_axis(axCdiff, grid_axis="x")


# ---------- D: narrow-cell history necessity control ----------
gsD = outer[1, 63:].subgridspec(2, 5, height_ratios=[0.43, 0.57], width_ratios=[1, 1, 1, 1, 0.92], hspace=0.48, wspace=0.55)
axDatlas = fig.add_subplot(gsD[0, :])
panel_letter(axDatlas, "D", x=-0.12, y=1.20)
conds = ["full_history", "complete_history_reset", "decision_observation_only", "timing_only"]
cond_short = ["Full-\nhistory", "History\nreset", "Decision-\nonly", "Timing-\nonly"]
atlas = np.zeros((10, 8))
for i, seed in enumerate(SEEDS):
    atlas[i, :4] = [control_lookup[(seed, "TEM", c)]["accuracy"] for c in conds]
    atlas[i, 4:] = [control_lookup[(seed, "HDGM", c)]["accuracy"] for c in conds]
axDatlas.imshow(atlas, cmap=cmap, vmin=0.22, vmax=0.65, aspect="auto", interpolation="nearest")
axDatlas.set_xticks(range(8), cond_short + cond_short)
axDatlas.xaxis.tick_top()
axDatlas.set_yticks(range(10), SEEDS)
axDatlas.tick_params(length=0, labelsize=5.0, colors=MUTED, pad=2)
axDatlas.set_xticks(np.arange(-0.5, 8, 1), minor=True)
axDatlas.set_yticks(np.arange(-0.5, 10, 1), minor=True)
axDatlas.grid(which="minor", color=WHITE, linewidth=0.75)
axDatlas.tick_params(which="minor", bottom=False, left=False)
axDatlas.axvline(3.5, color=WHITE, lw=3)
axDatlas.text(.24,1.28,"TEM",transform=axDatlas.transAxes,ha='center',color=TEM,fontweight='bold',fontsize=7)
axDatlas.text(.76,1.28,"HDGM",transform=axDatlas.transAxes,ha='center',color=HDGM,fontweight='bold',fontsize=7)
axDatlas.set_title("Seed-wise control atlas", loc="left", fontweight="bold", color=INK, pad=28)
axDatlas.text(1,1.46,'Accuracy: .22 (light) – .65 (dark)',transform=axDatlas.transAxes,
             ha='right',fontsize=6.2,color=MUTED)
for s in axDatlas.spines.values():
    s.set_visible(False)

axDloss = fig.add_subplot(gsD[1, :4])
base_y = [5, 3, 1]
rng = np.random.default_rng(20260831)
for idx, c in enumerate(conds[1:]):
    for model, off, color in [("TEM", 0.18, TEM), ("HDGM", -0.18, HDGM)]:
        vals = np.array([control_lookup[(s, model, c)]["loss"] for s in SEEDS])
        y = base_y[idx] + off
        axDloss.scatter(vals, y + rng.uniform(-0.04, 0.04, len(vals)), s=14, facecolors=WHITE,
                        edgecolors=color, linewidths=0.7)
        m, lo, hi = mean_ci(vals)
        axDloss.hlines(y, lo, hi, color=color, lw=1.4)
        axDloss.scatter(m, y, marker="D", s=25, color=color, edgecolors=WHITE, linewidths=0.5, zorder=3)
    axDloss.text(-2.1, base_y[idx] + 0.47, ["History reset", "Decision-only", "Timing-only"][idx],
                 fontsize=5.5, color=MUTED, fontweight="bold", ha="left")
axDloss.axvline(0, color="#98A5B1", lw=0.8, ls=(0, (3, 3)))
axDloss.set_xlim(-2.5, 42)
axDloss.set_ylim(0.5, 5.65)
axDloss.set_yticks([])
axDloss.set_xlabel("Accuracy loss (pp)")
axDloss.set_title("Performance loss under information controls", loc="left", fontweight="bold", color=INK)
axDloss.text(0.77, 1.03, "TEM", transform=axDloss.transAxes, color=TEM,
             fontsize=5.2, fontweight="bold", va="bottom")
axDloss.text(0.89, 1.03, "HDGM", transform=axDloss.transAxes, color=HDGM,
             fontsize=5.2, fontweight="bold", va="bottom")
style_axis(axDloss, grid_axis="x")

axDnull = fig.add_subplot(gsD[1, 4])
for y, key, color in [(1, "observation_only", PURPLE), (0, "timing_only", CUE_B)]:
    vals = np.array([shortcut_lookup[(s, key)] for s in SEEDS])
    axDnull.scatter(vals, y + rng.uniform(-0.04, 0.04, len(vals)), s=12, facecolors=WHITE,
                    edgecolors=color, linewidths=0.7)
    m, lo, hi = mean_ci(vals)
    axDnull.hlines(y, lo, hi, color=color, lw=1.2)
    axDnull.scatter(m, y, marker="D", s=22, color=color, edgecolors=WHITE, linewidths=0.5)
    axDnull.text(m, y + 0.18, f"{m:.1f}%", color=color, fontsize=5.3, ha="center", fontweight="bold")
axDnull.axvline(25, color="#98A5B1", lw=0.8, ls=(0, (3, 3)))
axDnull.set_xlim(22.3, 27.7)
axDnull.set_ylim(-0.35, 1.35)
axDnull.set_yticks([1, 0], ["Obs.", "Time"])
axDnull.set_xlabel("Accuracy (%)")
axDnull.set_title("Shortcut nulls", fontweight="bold", color=INK)
style_axis(axDnull, grid_axis="x")


# ---------- E: reconstruct the screenshot's radar + branch bundle ----------
# The missing original E layout cannot be recovered pixel-for-pixel. All values
# below come from the same seed-level CSV used by the original source.
gsE = outer[2, :].subgridspec(2, 6, height_ratios=[.20,.80],
    width_ratios=[1.5,1,1,1,1,1.45],hspace=.40,wspace=.72)
axEt = fig.add_subplot(gsE[0, :])
panel_letter(axEt, "E", x=-0.027, y=1.55)
axEt.axis('off');axEt.set_xlim(0,1);axEt.set_ylim(0,1)
axEt.text(0,1.02,'Frozen weights · n = 10 seeds',fontsize=7.4,color=MUTED)
segments=[(.0,.12,CUE_A,'Cue A'),(.12,.37,'#E8ECF3','Delay 1'),
          (.37,.49,CUE_B,'Cue B'),(.49,.74,'#E8ECF3','Delay 2'),(.74,.89,DECISION,'Decision')]
for x0,x1,col,label in segments:
    axEt.add_patch(Rectangle((x0,.36),x1-x0,.17,facecolor=col,edgecolor=GRID,lw=.45))
    axEt.text((x0+x1)/2,.71,label,ha='center',fontsize=7.6,fontweight='bold',color=INK)
for x in [.245,.615]:
    axEt.text(x,.12,'+ k matched repeats',color=PURPLE,fontsize=6.8,ha='center')
axEt.add_patch(FancyArrowPatch((.89,.445),(.995,.445),arrowstyle='-|>',mutation_scale=8,color=MUTED,lw=.8))

steps=sorted({int(r['inserted_steps_per_delay']) for r in delays})
delay_mats={m:np.array([[delay_lookup[(s,m,k)] for k in steps] for s in SEEDS]) for m in ['HDGM','TEM']}
assert steps==[0,4,8,16,32,64]
axEradar=fig.add_subplot(gsE[1,0],projection='polar')
ang=np.linspace(0,2*np.pi,6,endpoint=False)
axEradar.set_theta_offset(np.pi/2);axEradar.set_theta_direction(-1)
axEradar.set_ylim(0,.66);axEradar.set_xticks(ang,[str(k) for k in steps])
axEradar.set_yticks([.25,.50],['25%','50%']);axEradar.set_rlabel_position(20)
axEradar.tick_params(labelsize=7.1,pad=2,colors=MUTED)
axEradar.grid(color=GRID,lw=.55);axEradar.spines['polar'].set_visible(False)
delay_cols=['#DADFF3','#D8EAE6','#E5EBCF','#F4E6B9','#EFD5C8','#E8CFDF']
for a,col in zip(ang,delay_cols):
    axEradar.bar(a,.045,bottom=.60,width=2*np.pi/6-.03,color=col,edgecolor='none',zorder=0)
theta=np.linspace(0,2*np.pi,180)
axEradar.fill(theta,np.full_like(theta,.25),color='#E4E9EF',alpha=.35,zorder=0)
for m,col in [('HDGM',HDGM),('TEM',TEM)]:
    v=delay_mats[m].mean(0);aa=np.r_[ang,ang[0]];vv=np.r_[v,v[0]]
    axEradar.fill(aa,vv,color=col,alpha=.12)
    axEradar.plot(aa,vv,'o-',lw=1.55,ms=3.8,color=col,mfc=WHITE if m=='TEM' else col,mew=.85)
axEradar.text(.5,-.18,'k repeats per delay',transform=axEradar.transAxes,ha='center',fontsize=7.4,color=INK)
axEradar.text(.5,-.29,'Radius = mean accuracy',transform=axEradar.transAxes,ha='center',fontsize=6.8,color=MUTED)

axE=fig.add_subplot(gsE[1,1:5]);axE.axis('off');axE.set_xlim(0,1.15);axE.set_ylim(-.05,1.05)
axE.text(0,1.075,'Memory persistence across matched delays',fontsize=10,fontweight='bold',color=INK)
axE.text(.862,1.075,'k',color=INK,fontsize=8,ha='center')
axE.text(.94,1.075,'HDGM',color=HDGM,fontsize=8,fontweight='bold',ha='center')
axE.text(1.095,1.075,'TEM',color=TEM,fontsize=8,fontweight='bold',ha='center')
origin=(.045,.47)
for i,k in enumerate(steps):
    base=.92-i*.17
    # Each condition has a shared local baseline. Curves are schematic connectors.
    # Small vertical endpoint offsets use the SAME accuracy scale for both models.
    for m,col,off in [('HDGM',HDGM,.008),('TEM',TEM,-.008)]:
        vals=delay_mats[m][:,i];mean,lo,hi=mean_ci(vals)
        yy=base+.10*(mean-.35)+off
        vertices=[origin,(.28,.47),(.57,yy),(.83,yy)]
        path=MplPath(vertices,[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4])
        axE.add_patch(mpl.patches.PathPatch(path,facecolor='none',edgecolor=col,lw=2.0,alpha=.9))
        axE.plot(.83,yy,'o',ms=6.3,mfc=WHITE if m=='TEM' else col,mec=col,mew=1.0,zorder=4)
    axE.text(.862,base+.045,str(k),fontsize=7.5,color=INK,fontweight='bold',ha='center')
    axE.text(.94,base+.007,f"{delay_mats['HDGM'][:,i].mean()*100:.1f}%",ha='center',color=HDGM,fontsize=7.3)
    axE.text(1.095,base+.007,f"{delay_mats['TEM'][:,i].mean()*100:.1f}%",ha='center',color=TEM,fontsize=7.3)
axE.plot(*origin,'o',ms=7.5,color=CUE_A)
axE.text(origin[0],origin[1]-.10,'Cue',ha='center',fontsize=7.7,color=CUE_A,fontweight='bold')
axE.text(.57,-.16,'Curves = schematic conditions; terminal labels = mean accuracy',
         transform=axE.transAxes,ha='center',fontsize=6.8,color=MUTED)
axE.text(.57,-.26,'Not measured neural trajectories · no adaptation or retraining',
         transform=axE.transAxes,ha='center',fontsize=6.8,color=MUTED)

axEpair = fig.add_subplot(gsE[1, 5])
changes = {}
for model, color, x in [("HDGM", HDGM, 0), ("TEM", TEM, 1)]:
    vals = 100 * np.array([delay_lookup[(s, model, 64)] - delay_lookup[(s, model, 0)] for s in SEEDS])
    changes[model] = vals
    axEpair.scatter(np.full(10, x) + rng.uniform(-0.035, 0.035, 10), vals, s=16,
                    facecolors=WHITE, edgecolors=color, linewidths=0.8)
    m, lo, hi = mean_ci(vals)
    axEpair.vlines(x, lo, hi, color=color, lw=1.4)
    axEpair.scatter(x, m, s=30, color=color, edgecolors=WHITE, linewidths=0.6, zorder=3)
for h, t in zip(changes["HDGM"], changes["TEM"]):
    axEpair.plot([0, 1], [h, t], color="#D5DEE6", lw=0.65, zorder=0)
axEpair.axhline(0, color="#94A3B8", ls=(0, (3, 3)), lw=0.8)
axEpair.set_xlim(-0.4, 1.4)
axEpair.set_ylim(-28, 6)
axEpair.set_xticks([0, 1], ["HDGM", "TEM"])
axEpair.set_ylabel("Change from k=0 to 64 (pp)")
axEpair.set_title("Paired Δ accuracy (64 - 0)", fontweight="bold", color=INK)
axEpair.text(0.5, -0.16, r"$\Delta_{HDGM}-\Delta_{TEM}=+6.1$ pp" + "\nexact p = 0.023; Holm p = 0.070",
             transform=axEpair.transAxes, ha="center", va="top", fontsize=5.6, color=MUTED)
style_axis(axEpair, grid_axis="y")


# ---------- Minimal text / large-type presentation pass ----------
# This pass only modifies presentation artists; quantitative arrays and estimators
# above are unchanged. Explanatory prose is moved to the companion caption.
REMOVE_PREFIXES = (
    'Controlled history evaluation','Past cues','Memory',
    'Matched observation does not determine','Information controls',
    'Shortcut nulls: separate','HDGM · ring = correct',
    'Accuracy: .22','Frozen weights ·','Radius = mean accuracy',
    'Curves = schematic','Not measured neural trajectories',
)
all_axes = list(fig.axes)
for ax in list(fig.axes):
    all_axes.extend(ax.child_axes)
for ax in all_axes:
    for loc in ['left','center','right']:
        if ax.get_title(loc=loc) not in ('HDGM','TEM'):
            ax.set_title('',loc=loc)
    for artist in list(ax.texts):
        txt=artist.get_text()
        if txt.startswith(REMOVE_PREFIXES):
            artist.remove()
            continue
        if txt in ['A','B','C','D','E']:
            artist.set_fontsize(22)
        else:
            artist.set_fontsize(max(11.0,min(13.5,artist.get_fontsize()*1.6)))
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)
    ax.tick_params(axis='both',which='major',labelsize=11,pad=4)
    if ax.get_title(loc='left') in ('HDGM','TEM'):
        ax._left_title.set_fontsize(12)
    # Darker neutral text for readability without changing the colour palette.
    for txt in ax.get_xticklabels()+ax.get_yticklabels():
        txt.set_color('#4F5B68')

# One shared model key; remove unnecessary repeated free-floating labels.
from matplotlib.lines import Line2D
fig.legend([Line2D([],[],color=HDGM,marker='o',lw=2.0),
            Line2D([],[],color=TEM,marker='o',mfc=WHITE,lw=2.0)],
           ['HDGM','TEM'],loc='upper center',bbox_to_anchor=(.53,.998),
           ncol=2,frameon=False,fontsize=13,handlelength=2,columnspacing=2.0)
for ax in [axBcurve,axCperf,axDloss]:
    for artist in list(ax.texts):
        if artist.get_text() in ('HDGM','TEM'):
            artist.remove()

# Row identities are listed in the caption; the same ten seed rows are retained.
for ax in fig.axes:
    if len(ax.images) and ax.images[0].get_array().shape[0]==10:
        ax.set_yticks([])
        ax.set_ylabel('10 seeds',fontsize=10.5,labelpad=5,color=MUTED)
        ax.tick_params(axis='x',labelsize=10.0,length=0)
        if ax is not axDatlas:
            ax.set_xlabel('Training updates' if ax.get_xlabel() else '',fontsize=12)
            model_name=ax.get_title(loc='left')
            ax.set_title('',loc='left')
            ax.text(0,1.10,model_name,transform=ax.transAxes,va='bottom',
                    fontsize=12,fontweight='bold',color=HDGM if model_name=='HDGM' else TEM)
axDatlas.tick_params(axis='x',labelsize=9.5,pad=6)
for t in axDatlas.texts:
    if t.get_text() in ('HDGM','TEM'):
        t.set_position((t.get_position()[0],1.45))
# Replace the verbose atlas scale sentence by a small numerical colour key.
atlas_key=axDatlas.inset_axes([.74,-.30,.24,.065])
atlas_key.imshow(np.linspace(.22,.65,100)[None,:],aspect='auto',cmap=cmap,
                 vmin=.22,vmax=.65,extent=[.22,.65,0,1])
atlas_key.set_xticks([.22,.65],['.22','.65']);atlas_key.set_yticks([])
atlas_key.tick_params(labelsize=10,length=0,pad=2)
for sp in atlas_key.spines.values():sp.set_visible(False)
atlas_key.set_xlabel('')

# Short, necessary metric labels remain; interpretive headings do not.
axBpair.set_ylabel('Acquisition AUC',fontsize=12)
axCperf.set_xlabel('Accuracy',fontsize=12)
axCperf.set_ylabel('Macro-F1',fontsize=12)
axCdiff.set_xlabel('Δ accuracy (pp)\nHDGM − TEM',fontsize=11.5)
axDnull.set_xlabel('Accuracy\n(%)',fontsize=11)
axDnull.set_xticks([23,25,27])
axEpair.set_ylabel('Δ accuracy (pp)\nk = 64 − 0',fontsize=11.5)
axEpair.set_yticks([-25,-15,-5,0,5])
axDloss.set_ylim(.5,6.0)
for t in axDloss.texts:
    if t.get_text() in ('History reset','Decision-only','Timing-only'):
        t.set_fontsize(10)
        x,y=t.get_position();t.set_position((x,y+.12))
# Keep p/CI evidence but place it outside the plotted marks at readable size.
for t in axBpair.texts:
    if t.get_text().startswith('ΔAUC'):
        t.set_text('ΔAUC +9.8 pp\n95% CI [3.0, 16.5]')
        t.set_fontsize(10.0);t.set_position((.02,1.01));t.set_va('bottom')
for t in axEpair.texts:
    if 'Holm' in t.get_text():
        t.set_text('ΔΔ = +6.1 pp\np = .023; Holm p = .070')
        t.set_fontsize(10.0);t.set_position((.5,-.23))
for t in axE.texts:
    if t.get_text()=='Cue':t.set_text('')
for t in axA.texts:
    if t.get_text() in ['History 1','History 2','Same observation','Frozen model','Readout']:
        t.set_fontsize(10.5)
    if t.get_text() in ['1','2','3','4']:
        t.set_fontsize(10)
    if t.get_text()=='Frozen model':t.set_position((.70,.385))
    if t.get_text()=='Same observation':t.set_text('Same\nobservation');t.set_va('center')
    if t.get_text()=='Readout':t.set_text('4 choices')
    if t.get_text() in ['1','2','3','4'] and t.get_position()[0]>.93:
        t.set_visible(False)
for t in axCstruct.texts:
    t.set_fontsize(11.5)
    if len(t.get_text())==2 and t.get_text().isdigit():
        t.set_text(t.get_text()+'%')
axCstruct.text(.5,1.01,'HDGM',transform=axCstruct.transAxes,ha='center',
               fontsize=12,color=HDGM,fontweight='bold')
axEradar.tick_params(labelsize=11)
for t in axEradar.texts:
    if t.get_text()=='k repeats per delay':t.set_text('k repeats / delay');t.set_fontsize(11)
# Compact schematic key is necessary so E is not mistaken for measured dynamics.
axE.text(.03,1.055,'Schematic',transform=axE.transAxes,fontsize=11,color=MUTED)

# Reallocate only E's left and middle panels. Keep the timeline, paired plot,
# all plotted values and every A-D axes position unchanged.
radar_before = axEradar.get_position().frozen()
fan_before = axE.get_position().frozen()
pair_before = axEpair.get_position().frozen()
fan_shift = fan_before.width * .29
axE.set_position([fan_before.x0 + fan_shift, fan_before.y0,
                  fan_before.width - fan_shift, fan_before.height])
# Polar axes have a square physical aspect: width alone would not enlarge them.
# Use the existing whitespace above/below the old radar as well.
radar_height = min(radar_before.height * 1.40,
                   axEt.get_position().y0 - .022 - .052)
radar_width = radar_height * fig.get_figheight() / fig.get_figwidth()
radar_centre_x = (.049 + axE.get_position().x0 - .035) / 2
radar_y0 = max(.052, radar_before.y0 - .018)
axEradar.set_position([radar_centre_x-radar_width/2, radar_y0,
                       radar_width, radar_height])
for t in axEradar.texts:
    if t.get_text() == 'k repeats / delay':
        t.set_position((.5,-.20))
assert np.allclose(axEpair.get_position().bounds, pair_before.bounds)
print('E radar diameter scale:', radar_height/radar_before.height)
print('E schematic width scale:', axE.get_position().width/fan_before.width)

# Second layout pass: translate the entire schematic and its numerical table
# left without rescaling it. Expand the paired comparison into released space.
# Preserve the previous radar geometry, scale, values, and all A-D positions.
fan_compact = axE.get_position().frozen()
axE.set_position([fan_compact.x0-.075, fan_compact.y0,
                  fan_compact.width, fan_compact.height])
pair_width = pair_before.width * 1.50
pair_height = pair_before.height * 1.12
axEpair.set_position([pair_before.x1-pair_width,
                      pair_before.y0-.006, pair_width, pair_height])
axEpair.tick_params(labelsize=12)
axEpair.yaxis.label.set_fontsize(12)
for t in axEpair.texts:
    if 'Holm' in t.get_text():
        t.set_position((.5,-.20))
        t.set_fontsize(10.5)
print('E paired plot scale: width 1.50; height 1.12')

# Figure-level panel letters stay clear of all enlarged axis labels.
for ax in fig.axes:
    for t in list(ax.texts):
        if t.get_text() in ['A','B','C','D','E']:
            t.remove()
for letter,x,y in [('A',.026,axA.get_position().y1+.022),
                   ('B',.357,axA.get_position().y1+.022),
                   ('C',.026,axCperf.get_position().y1+.024),
                   ('D',.591,axCperf.get_position().y1+.024),
                   ('E',.026,axEt.get_position().y1+.024)]:
    fig.text(x,y,letter,fontsize=22,fontweight='bold',color='#111827',va='bottom')

OUT.parent.mkdir(parents=True, exist_ok=True)
PREVIEW.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
fig.savefig(PREVIEW, dpi=args.dpi)
fig.savefig(args.out_dir / 'figure2_E_rebalanced_v2.svg')
# Machine-readable evidence audit, including all arrays driving the quantitative panels.
audit = {
    'source_plot':'plot_figure2_mosaic.py',
    'data_root':str(ROOT), 'seeds':SEEDS, 'checkpoints':CHECKPOINTS,
    'uncertainty':'Original source estimator retained: mean +/- 1.96 * sample SD / sqrt(n); NOT bootstrap.',
    'auc_annotation':'Original source hard-coded annotation retained: +9.8 pp, CI [3.0,16.5].',
    'paired_p_annotation':'Original source annotation retained: exact p=.023; Holm p=.070.',
    'acquisition':{m:mat.tolist() for m,mat in curve_mats.items()},
    'acquisition_auc':{'TEM':tem_auc.tolist(),'HDGM':hdgm_auc.tolist()},
    'class_accuracy':class_acc.tolist(),'selected_edge_percent':pair_pct.tolist(),
    'class_differences':[v.tolist() for v in class_diffs],
    'control_atlas':atlas.tolist(),
    'delay_accuracy':{m:v.tolist() for m,v in delay_mats.items()},
    'paired_delay_changes':{m:v.tolist() for m,v in changes.items()},
    'e_reconstruction':'Radar and schematic branch bundle rebuilt from original screenshot and CSV; not original lost plotting code.',
}
assert round(100*(hdgm_auc-tem_auc).mean(),1)==9.8
assert np.allclose(100*class_acc,[36,51,54,49],atol=.6)
(args.out_dir/'figure2_data_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
plt.close(fig)
print(f"Saved: {OUT}")
