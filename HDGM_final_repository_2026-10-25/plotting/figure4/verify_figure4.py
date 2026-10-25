"""Compare the final numerical drawing artists against the unmodified source."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import original_plot as old
import plot_figure4_refined as new

ROOT=Path(__file__).resolve().parent


def evidence(ax):
    arrays=[]
    for line in ax.lines:
        arrays.extend([np.asarray(line.get_xdata(),float),np.asarray(line.get_ydata(),float)])
    for c in ax.collections:
        if isinstance(c,matplotlib.collections.PathCollection):
            arrays.append(np.asarray(c.get_offsets(),float))
        elif isinstance(c,matplotlib.collections.LineCollection):
            arrays.extend([np.asarray(s,float) for s in c.get_segments()])
        elif isinstance(c,matplotlib.collections.PolyCollection):
            arrays.extend([p.vertices for p in c.get_paths()])
    arrays.extend([np.asarray(i.get_array(),float) for i in ax.images])
    for p in ax.patches:
        if hasattr(p,"_posA_posB") and p._posA_posB is not None:
            arrays.append(np.asarray(p._posA_posB,float))
    return arrays


def main():
    p=old.load_json(ROOT/"data"/"position.json")
    b=old.load_json(ROOT/"data"/"branch.json")
    assert len(p["records"])==len(b["records"])==10
    assert p["summary"]["config"]["track_len"]==20
    assert p["summary"]["config"]["p_backward"]>0
    assert b["summary"]["config"]["corridor_len"]==18
    fig,axes=new.build_figure()
    origfig=plt.figure(figsize=fig.get_size_inches())
    orig={k:origfig.add_axes(a.get_position(),projection="polar" if k=="E2" else None)
          for k,a in axes.items() if k!="A"}
    old.draw_position_atlas(orig["B1"],orig["B2"],p["records"])
    old.draw_branch_trajectory(orig["C1"],orig["C2"],b["records"])
    old.draw_layer_heatmap(orig["D"],b["records"])
    old.draw_geometry_phase(orig["E1"],b["records"])
    old.draw_geometry_gain_wheel(orig["E2"],b["records"])
    old.draw_effect_forest(orig["F"],p["records"],b["records"])
    checks={}
    for k in orig:
        a,c=evidence(orig[k]),evidence(axes[k])
        assert len(a)==len(c),(k,len(a),len(c))
        for aa,cc in zip(a,c):np.testing.assert_array_equal(aa,cc)
        checks[k]={"arrays_unchanged":len(a)}
    # Source p labels were constants; confirm them against the summary records.
    posstats=p["summary"]["primary_statistics_holm"]
    branchstats=b["summary"]["representation_primary_statistics_holm"]
    for stats in (posstats,branchstats):
        for s in stats.values(): assert round(s["p_holm"],4)==.0039
    dz=posstats["HDGM residual population stability vs structured null"]["dz"]
    assert round(dz,2)==2.77
    # The narrower F axis must still expose every observed seed value.
    for coll in axes["F"].collections:
        if isinstance(coll,matplotlib.collections.PathCollection):
            xx=np.asarray(coll.get_offsets())[:,0]
            assert np.all(xx>=axes["F"].get_xlim()[0]) and np.all(xx<=axes["F"].get_xlim()[1])
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    clipping=[]
    for ax in axes.values():
        # Check explicitly placed labels and axis labels; ticks are reviewed visually.
        for text in list(ax.texts)+[ax.xaxis.label,ax.yaxis.label]:
            if not text.get_visible() or not text.get_text():continue
            bb=text.get_window_extent(renderer)
            if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fig.bbox.width+1 or bb.y1 > fig.bbox.height+1:
                clipping.append(text.get_text())
    assert not clipping,clipping
    report={"passed":True,"panels":checks,"verified_seed_count":10,
            "verified_source_holm_p":.00390625,"verified_B_dz":dz,
            "outer_canvas_text_clipping":clipping,
            "source_sha256":{f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                             for f in (ROOT/"data").glob("*.json")}}
    (ROOT/"numerical_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    plt.close(fig);plt.close(origfig)


if __name__=="__main__":main()
