# Running the archive

## Saved-result verification (no model training)
```sh
python tools/verify_package.py
python tools/recheck_statistics.py
```
## Plotting from saved results
```sh
python tools/plot.py 2
python tools/plot.py 3
python tools/plot.py 4
python tools/plot.py 5b
python tools/plot.py 5c
python tools/plot.py 5d
python tools/plot.py 5e
python tools/plot.py 5f
python tools/plot.py 6
```
These commands need Matplotlib/NumPy and other imports listed in requirements.txt.
Figure5 whole-layout command is deliberately omitted because the original capacity-boundary
script was not found. D/E/F commands are newly reconstructed, not recovered historical sources.

## Frozen gate-substitution evaluation (expensive; fits diagnostic probes)
From the package root, inspect options first:
```sh
python experiments/stage_gate_substitutions.py --help
python experiments/operation_cuts.py --help
python experiments/history_controls.py --help
```
For the V1 stage gate evaluation, supply experiments/four_context.py as --main-code and
weights/v4 as --checkpoint-dir; choose a new --output-dir. The source module documents
--quick and device options. Four-context checkpoints contain several internal snapshots.

## Training and remaining evaluations
See EXPERIMENT_MAP.csv and configs/source_defaults.json. These are the selected historical
entry scripts, not newly rewritten unified experiments. Several execute their configuration
directly under __main__ instead of accepting CLI flags. Review and adapt output/checkpoint
paths before execution. Preserve source defaults except for documented path/device changes.
Structural controls import the sibling 分叉.py and retain its original filename for that reason.
Never treat changes made for a debug run as the scientific training configuration.

This package does not claim a validated single-command end-to-end pipeline or exact
cross-platform numerical identity. No training or model evaluation was rerun. Saved-output statistics, D/E/F rendering
and explicitly recorded bootstrap reconstruction were executed.
