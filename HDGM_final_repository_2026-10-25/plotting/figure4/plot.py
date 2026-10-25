"""Figure 4 layout revision. Run this script; no artwork assets are required.

Quantitative functions are retained in original_plot.py for exact regression
testing. A is a vector task schematic, not an animal experiment or activation map.
"""
from pathlib import Path
import json
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import original_plot as source

ROOT = Path(__file__).resolve().parent
INK, MUTED, RULE = source.INK, "#536174", "#DFE5EC"
PURPLE, GREEN, GOLD = source.POSITIONAL, source.HISTORY_A, source.HISTORY_B


def arrow(ax, start, end, color, lw=1.4, **kwargs):
    p = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
                        color=color, lw=lw, **kwargs)
    ax.add_patch(p)
    return p


def task_schematic(ax):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(.235, .97, "Position assay", color=source.POSITIONAL_DARK,
            ha="center", va="top", fontsize=11, weight="bold")
    ax.text(.745, .97, "History assay", color=source.HISTORY_A_DARK,
            ha="center", va="top", fontsize=11, weight="bold")
    # A shallow track edge gives spatial structure without changing data scale:
    # this entire panel is a task diagram, not a quantitative perspective plot.
    t = np.linspace(0, 2*np.pi, 240)
    cx, cy, rx, ry = .235, .725, .184, .135
    x, y = cx + rx*np.cos(t), cy + ry*np.sin(t)
    ax.plot(x, y-.015, color="#B9C1CD", lw=15, solid_capstyle="round", zorder=1)
    ax.plot(x, y, color="#D3D9E4", lw=15, solid_capstyle="round", zorder=2)
    ax.plot(x, y, color="#F1F0F7", lw=12, solid_capstyle="round", zorder=3)
    ts = np.linspace(0, 2*np.pi, 20, endpoint=False)
    ax.scatter(cx + rx*np.cos(ts), cy+ry*np.sin(ts), s=10,
               facecolors="white", edgecolors=PURPLE, linewidths=.55, zorder=5)
    # Neutral agent marker, not a mouse or a human.
    ax.scatter([cx], [cy-ry], marker=">", s=75, c=PURPLE,
               edgecolors="white", linewidths=.8, zorder=7)
    tt = np.linspace(4.95, 5.75, 35)
    ax.plot(cx+rx*np.cos(tt), cy+ry*np.sin(tt), color=PURPLE, lw=1.8, zorder=6)
    arrow(ax, (cx+rx*np.cos(tt[-3]), cy+ry*np.sin(tt[-3])),
          (cx+rx*np.cos(tt[-1]), cy+ry*np.sin(tt[-1])), PURPLE, zorder=7)
    arrow(ax, (cx+rx*np.cos(tt[2]), cy+ry*np.sin(tt[2])),
          (cx+rx*np.cos(tt[0]), cy+ry*np.sin(tt[0])), PURPLE, zorder=7)
    ax.text(cx, cy, "20 positions", ha="center", va="center", fontsize=10, color=MUTED)
    ax.text(cx, .485, "Bidirectional transitions", ha="center", fontsize=9.5, color=MUTED)
    ax.plot([.47, .47], [.49, .91], color=RULE, lw=.8)

    # Two histories, one physically matched corridor, then two choice arms.
    # Histories are indicated by a circle and square as well as distinct hues.
    for hy, col, marker, lab in [(.84, GREEN, "o", "A"), (.59, GOLD, "s", "B")]:
        ax.scatter([.525], [hy], s=35, marker=marker, c=col, zorder=5)
        ax.text(.535, hy+.038, lab, ha="center", color=col, fontsize=10, weight="bold")
        arrow(ax, (.56, hy), (.66, .72), col, zorder=4)
    ax.plot([.65, .81], [.704, .704], lw=19, color="#BBC5D0", solid_capstyle="round", zorder=1)
    ax.plot([.65, .81], [.72, .72], lw=17, color="#EAF0F4", solid_capstyle="round", zorder=2)
    for ey, col, lab in [(.84, GREEN, "A"), (.59, GOLD, "B")]:
        ax.plot([.81, .955], [.705, ey-.015], color="#CAD2DB", lw=12,
                solid_capstyle="round", zorder=1)
        ax.plot([.81, .955], [.72, ey], color="#F1F4F7", lw=10,
                solid_capstyle="round", zorder=2)
        arrow(ax, (.82, .72), (.95, ey), col, zorder=4)
        ax.text(.985, ey, lab, ha="center", va="center", color=col, fontsize=10, weight="bold")
    for dy, col, ls in [(.009, GREEN, "-"), (-.009, GOLD, "--")]:
        ax.plot([.655, .815], [.72+dy]*2, color=col, lw=1.35, ls=ls, zorder=4)
    ax.scatter([.745], [.72], s=48, facecolor="white", edgecolor=INK, linewidth=1, zorder=6)
    ax.text(.745, .52, "Shared corridor", ha="center", fontsize=10, color=MUTED)
    # Local explanatory expansion; dashed connector is a callout, not a path.
    ax.plot([.745, .745], [.675, .575], color="#9BA7B5", lw=.8, ls=(0,(2,3)), zorder=0)
    ax.plot([.745, .745, .20], [.485, .455, .455], color="#9BA7B5", lw=.8, ls=(0,(2,3)), zorder=0)
    ax.plot([.20, .20], [.455, .365], color="#9BA7B5", lw=.8, ls=(0,(2,3)))
    ax.text(.70, .40, "Internal states (schematic)", ha="center", fontsize=9.5, color=MUTED)
    ax.add_patch(FancyBboxPatch((.10,.16), .20,.17, boxstyle="round,pad=.01,rounding_size=.018",
                               facecolor="#F4F6F8", edgecolor="#B4BFCC", lw=.8))
    # One identical observation symbol feeds both states.
    ax.text(.20,.245, r"$x_t$", ha="center", va="center", fontsize=17, color=INK)
    ax.text(.20,.10,"Matched input", ha="center", fontsize=10, color=MUTED)
    for sy, col, marker, label, levels in [(.29,GREEN,"o","A",[.25,.7,.45,.9,.3,.65]),
                                          (.10,GOLD,"s","B",[.75,.3,.85,.35,.7,.4])]:
        arrow(ax, (.32,.245), (.57,sy), col, lw=1.2)
        ax.text(.445, sy+(.046 if label=="A" else -.030), f"History {label}",
                ha="center", fontsize=9, color=col)
        ax.add_patch(FancyBboxPatch((.59,sy-.057),.24,.114,
                     boxstyle="round,pad=.008,rounding_size=.018", facecolor="white", edgecolor=col,lw=.9))
        # Authored glyphs encode different schematic states, not measured units.
        for i,v in enumerate(levels):
            ax.plot([.615+i*.037]*2,[sy-.025,sy-.025+.055*v],color=col,lw=2.8,alpha=.75)
        ax.scatter([.85],[sy],s=32,c=col,marker=marker)
        ax.text(.875,sy,f"State {label}",ha="left",va="center",fontsize=9,color=col)


def remove_text(ax, phrases):
    for text in list(ax.texts):
        if any(p in text.get_text() for p in phrases):
            text.remove()


def typography(ax, body=10, ticks=9.5):
    for text in ax.texts:
        text.set_fontsize(max(body, text.get_fontsize()))
        if text.get_color() == source.MID:
            text.set_color(MUTED)
    ax.tick_params(axis="both", labelsize=ticks)
    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_fontsize(11)
    legend=ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(9.5)


def build_figure():
    place=source.load_json(ROOT/"data"/"position.json")["records"]
    branch=source.load_json(ROOT/"data"/"branch.json")["records"]
    mpl.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
        "axes.labelcolor":INK,"text.color":INK,"xtick.color":MUTED,"ytick.color":MUTED,
        "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none",
        "axes.linewidth":.75,"figure.facecolor":"white","savefig.facecolor":"white"})
    fig=plt.figure(figsize=(14.5,10.7))
    bounds={"A":[.045,.675,.315,.285], "B1":[.420,.675,.340,.275],
            "B2":[.845,.675,.140,.275], "C1":[.075,.424,.58,.18],
            "C2":[.075,.312,.58,.075], "D":[.800,.312,.185,.292],
            "E1":[.075,.070,.365,.175], "E2":[.490,.068,.165,.183],
            "F":[.800,.070,.185,.175]}
    axes={k:fig.add_axes(v,projection="polar" if k=="E2" else None) for k,v in bounds.items()}
    task_schematic(axes["A"])
    source.draw_position_atlas(axes["B1"],axes["B2"],place)
    source.draw_branch_trajectory(axes["C1"],axes["C2"],branch)
    source.draw_layer_heatmap(axes["D"],branch)
    source.draw_geometry_phase(axes["E1"],branch)
    source.draw_geometry_gain_wheel(axes["E2"],branch)
    source.draw_effect_forest(axes["F"],place,branch)

    # Remove prose, never observations or inferential intervals.
    remove_text(axes["B1"],["normalized activity"])
    remove_text(axes["B2"],["Holm"])
    axes["B2"].text(.5,1.045,r"$d_z=2.77$ · $p_H=.0039$",transform=axes["B2"].transAxes,
                         ha="center",va="bottom",fontsize=9,color=source.POSITIONAL_DARK)
    b=axes["B1"]
    b.text(0,1.045,"Normalized activity",transform=b.transAxes,fontsize=10,va="bottom",color=MUTED)
    cbax=b.inset_axes([.70,1.040,.26,.027])
    cb=fig.colorbar(b.images[0],cax=cbax,orientation="horizontal",ticks=[0,1])
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=9,length=0,pad=2)
    remove_text(axes["C1"],["trajectory direction","matched observation","visual aperture","TEM remains"])
    for t in list(axes["C1"].texts):
        if t.get_text() in ("A","B"): t.remove()
    axes["C1"].text(.01,1.05,"Decoding-derived schematic",transform=axes["C1"].transAxes,
                        fontsize=10,color=MUTED,va="bottom")
    axes["C2"].set_xlabel("Matched corridor position")
    axes["C2"].set_ylabel("History decode")
    remove_text(axes["D"],["mean accuracy"])
    axes["D"].text(.5,-.04,"Decoding accuracy (%)",transform=axes["D"].transAxes,
                       ha="center",va="top",fontsize=10,color=MUTED)
    remove_text(axes["E1"],["between > within","dashed:","arrows:"])
    axes["E1"].set_ylabel("Between-history distance")
    axes["E1"].get_legend().set_frame_on(False)
    remove_text(axes["E2"],["outer arm"])
    axes["E2"].text(.5,-.15,r"$\rho_{full}/\rho_{reset}$ (log scale)",transform=axes["E2"].transAxes,
                        ha="center",va="top",fontsize=9.5,color=MUTED)
    lg=axes["E2"].get_legend(); lg.remove()
    # Signed gain is also encoded by inward/outward direction about the 1x ring.
    remove_text(axes["F"],["p=","POSITIONAL SCAFFOLD","HISTORY BRANCH","density ridge"])
    axes["F"].text(0,3.64,"Position",color=source.POSITIONAL_DARK,fontsize=9.5,va="top",weight="bold")
    axes["F"].text(0,1.33,"History",color=source.HISTORY_A_DARK,fontsize=9.5,va="top",weight="bold")
    axes["F"].text(1,1.08,r"Holm $p=.0039$ (all)",transform=axes["F"].transAxes,
                       fontsize=9.5,ha="right",va="bottom",color=MUTED)
    axes["F"].set_xlim(-.45,4.65)
    for k,a in axes.items():
        if k!="A": typography(a,body=9.5 if k in ("B2","E2","F") else 10,
                              ticks=8.5 if k=="E2" else 9.5)
    # Align panel letters by semantic rows rather than the label extents.
    for label,x,y in [("A",.025,.985),("B",.385,.985),("C",.025,.628),
                       ("D",.727,.628),("E",.025,.272),("F",.730,.272)]:
        fig.text(x,y,label,fontsize=20,weight="bold",va="top",color=INK)
    return fig,axes


def main():
    fig,axes=build_figure()
    fig.savefig(ROOT/"figure4_refined.png",dpi=300)
    fig.savefig(ROOT/"figure4_refined.pdf",dpi=300)
    fig.savefig(ROOT/"figure4_refined.svg",dpi=300)
    fig.savefig(ROOT/"preview.png",dpi=145)
    # The panel export is useful for inspecting the new task diagram alone.
    box=axes["A"].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(ROOT/"figure4_A_detail.png",dpi=300,bbox_inches=box.expanded(1.04,1.04))
    plt.close(fig)


if __name__=="__main__": main()
