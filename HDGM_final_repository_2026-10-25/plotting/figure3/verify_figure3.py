"""Read-only scientific regression test; writes only a local QA report."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
ORIGINAL = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "original_plot.py"


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def evidence(ax):
    result = []
    for line in ax.lines:
        result.extend([np.asarray(line.get_xdata(), float),
                       np.asarray(line.get_ydata(), float)])
    for coll in ax.collections:
        if isinstance(coll, matplotlib.collections.PathCollection):
            result.append(np.asarray(coll.get_offsets(), float))
        elif isinstance(coll, matplotlib.collections.LineCollection):
            result.extend([np.asarray(s, float) for s in coll.get_segments()])
        elif isinstance(coll, matplotlib.collections.PolyCollection):
            result.extend([p.vertices for p in coll.get_paths()])
    result.extend([np.asarray(im.get_array(), float) for im in ax.images])
    return result


def main():
    old = module(ORIGINAL, "original_figure3")
    new = module(ROOT / "plot_figure3_refined.py", "refined_figure3")
    op, gate, ca1 = [new.load_json(ROOT / "data" / name)
                     for name in ("operation.json", "gate.json", "ca1.json")]
    arr_old, arr_new = old.operation_arrays(op), new.operation_arrays(op)
    for a, b in zip(arr_old[2:], arr_new[2:]):
        np.testing.assert_array_equal(a, b)
    normal, long, dn, dl, sn, sl = arr_new[2:]
    cases = [("draw_crown", (normal, long, dn, dl, sn, sl), "polar"),
             ("draw_barcode", (normal, long, sn), None),
             ("draw_gate_ablation", (gate,), None),
             ("draw_ca1_alignment", (op,), None),
             ("draw_ca1_transformation", (ca1,), None),
             ("draw_controls", (ca1,), None)]
    checked = {}
    for function, args, projection in cases:
        a = plt.figure().add_subplot(projection=projection)
        b = plt.figure().add_subplot(projection=projection)
        plt.figure(a.figure.number)
        getattr(old, function)(a, *args)
        plt.figure(b.figure.number)
        getattr(new, function)(b, *args)
        ea, eb = evidence(a), evidence(b)
        assert len(ea) == len(eb), (function, len(ea), len(eb))
        for x, y in zip(ea, eb):
            np.testing.assert_array_equal(x, y)
        checked[function] = {"numerical_artist_arrays": len(ea), "unchanged": True}
        plt.close(a.figure)
        plt.close(b.figure)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((ROOT / "data").glob("*.json"))}
    report = {"passed": True, "original_script": str(ORIGINAL),
              "source_sha256": hashes, "panels": checked,
              "A_mean_losses_pp": normal.mean(0).tolist(),
              "D_gain_pp": 100 * ca1["statistics"]["secondary_CA1_output_integration"]
                  ["output_minus_pre_ca1_state"]["mean_diff"]}
    (ROOT / "numerical_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
