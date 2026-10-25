#!/usr/bin/env python3
"""HDGM V4 stage-specific causal perturbation experiment.

This is an evaluation-only experiment for the locked V4 primary checkpoints.
It does not retrain or select a model.  The script dynamically imports the
original main-experiment file, rebuilds its exact fixed held-out banks, and
changes EC/CA3/CA1 gates inside the recurrent forward dynamics.

Primary perturbation
--------------------
``mean_clamp`` replaces the selected gate by its phase-blind, per-unit mean
estimated on the independent analysis bank.  This removes stage-specific gate
modulation while preserving the gate's average magnitude and unit profile.

Secondary perturbation
----------------------
``zero`` clamps the selected gate to zero during the target stage.  Its meaning
is module-specific: EC input passage is removed, CA3 recurrent candidate
updates are suppressed, and CA1 gated amplification is removed.

Stages
------
encoding    : Cue A and Cue B samples.
maintenance : Delay 1 and Delay 2, excluding the terminal readout window.
readout     : the final ``readout_width`` recurrent samples ending at decision.

The default output files are:

* output/figure1d_data/hdgm_stage_specific_perturbation_v1.json
* output/figure1d_data/hdgm_stage_specific_perturbation_v1.csv
* output/figure1d_data/hdgm_stage_specific_perturbation_v1.npz

Example
-------
python HDGM_v4_stage_specific_perturbation.py \
  --main-code 主实验.py \
  --checkpoint-dir paper_rapid_v4_outputs/checkpoints \
  --device cuda

Run ``--quick`` first for a one-seed pipeline audit, then run without it for
the ten-seed confirmatory analysis.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

try:
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required for the frozen representation analyses. "
        "Install it with: pip install scikit-learn"
    ) from exc


FINAL_SEEDS = (11, 22, 33, 44, 55, 66, 77, 88, 99, 111)
MODULES = ("ec", "ca3", "ca1")
STAGES = ("encoding", "maintenance", "readout")
PERTURBATION_MODES = ("mean_clamp", "zero")
N_CLASSES = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Frozen-checkpoint 3-module x 3-stage HDGM perturbation audit."
    )
    parser.add_argument(
        "--main-code",
        type=Path,
        required=True,
        help="Path to the locked V4 main-experiment Python file.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing the ten primary_150 checkpoint .pt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/figure1d_data"),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in FINAL_SEEDS),
        help="Comma-separated seed list.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=",".join(PERTURBATION_MODES),
        help="Comma-separated subset of mean_clamp,zero.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=150,
        help="Locked model snapshot used for the analysis.",
    )
    parser.add_argument(
        "--readout-width",
        type=int,
        default=4,
        help="Number of recurrent samples in the terminal readout window.",
    )
    parser.add_argument(
        "--test-batches",
        type=int,
        default=None,
        help="Override the configured 50 fixed test batches.",
    )
    parser.add_argument(
        "--analysis-batches",
        type=int,
        default=None,
        help="Override the configured 5 fixed analysis batches.",
    )
    parser.add_argument(
        "--skip-representation",
        action="store_true",
        help="Export behavior only; skip history and decision decoding.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="One-seed, two-batch pipeline audit; not a paper result.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def choose_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_main_module(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    module_name = "hdgm_locked_v4_main"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import main experiment: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    required = (
        "ExperimentConfig",
        "SharedPathFourContextEnv",
        "HDGM",
        "build_fixed_batch_bank",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"Main experiment is missing required names: {missing}")
    return module


def trusted_torch_load(path, map_location="cpu"):
    """Load only checkpoints created by the user's locked experiment."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def discover_checkpoints(checkpoint_dir, requested_seeds):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(checkpoint_dir)
    found = {}
    for path in sorted(checkpoint_dir.glob("*.pt")):
        checkpoint = trusted_torch_load(path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            continue
        seed = checkpoint.get("seed")
        mode = checkpoint.get("mode", "")
        max_updates = checkpoint.get("max_updates")
        if seed is None or int(seed) not in requested_seeds:
            continue
        if mode and mode != "primary_150":
            continue
        if max_updates is not None and int(max_updates) < 150:
            continue
        if int(seed) in found:
            raise RuntimeError(f"Multiple matching checkpoints found for seed {seed}.")
        found[int(seed)] = (path, checkpoint)
    missing = sorted(set(requested_seeds) - set(found))
    if missing:
        raise FileNotFoundError(f"Missing primary checkpoint(s) for seeds: {missing}")
    return found


def config_from_checkpoint(main_module, checkpoint):
    raw = dict(checkpoint["config"])
    return main_module.ExperimentConfig(**raw)


def build_environment(main_module, cfg, checkpoint, seed):
    env = main_module.SharedPathFourContextEnv(
        track_len=cfg.track_len,
        obs_dim=cfg.obs_dim,
        cue_width=cfg.cue_width,
        cueA_start_range=cfg.cueA_start_range,
        delay1_range=cfg.delay1_range,
        delay2_range=cfg.delay2_range,
        n_cues=cfg.n_cues,
        alias_prob=cfg.alias_prob,
        noise_std=cfg.noise_std,
        seed=seed,
    )
    maps = checkpoint["environment_maps"]
    env.pos_map = np.asarray(maps["pos_map"], dtype=np.float32).copy()
    env.cueA_map = np.asarray(maps["cueA_map"], dtype=np.float32).copy()
    env.cueB_map = np.asarray(maps["cueB_map"], dtype=np.float32).copy()
    return env


def build_locked_banks(main_module, env, cfg, checkpoint, test_batches, analysis_batches):
    split = checkpoint["split_seeds"]
    normal = main_module.build_fixed_batch_bank(
        env,
        n_batches=test_batches,
        batch_size=cfg.test_batch_size,
        delay_mode="normal",
        seed=int(split["test_normal"]),
    )
    long = main_module.build_fixed_batch_bank(
        env,
        n_batches=test_batches,
        batch_size=cfg.test_batch_size,
        delay_mode="long",
        seed=int(split["test_long"]),
    )
    analysis = main_module.build_fixed_batch_bank(
        env,
        n_batches=analysis_batches,
        batch_size=cfg.analysis_batch_size,
        delay_mode="normal",
        seed=int(split["analysis"]),
    )
    return {"normal": normal, "long": long}, analysis


def make_intervenable_hdgm_class(main_module):
    class IntervenableHDGM(main_module.HDGM):
        """Exact V4 HDGM forward with test-only, in-dynamics gate control."""

        def set_gate_reference(self, gate_reference):
            self._gate_reference = {
                key: value.detach().clone().to(next(self.parameters()).device)
                for key, value in gate_reference.items()
            }

        @staticmethod
        def _stage_mask(phase_t, t, decision_positions, stage, readout_width):
            decision_positions = decision_positions.to(phase_t.device)
            readout_start = decision_positions - int(readout_width) + 1
            readout_mask = (t >= readout_start) & (t <= decision_positions)
            if stage == "encoding":
                return ((phase_t == 1) | (phase_t == 3)) & (~readout_mask)
            if stage == "maintenance":
                return ((phase_t == 2) | (phase_t == 4)) & (~readout_mask)
            if stage == "readout":
                return readout_mask
            raise ValueError(f"Unknown intervention stage: {stage}")

        def _apply_gate(self, gate, module_name, mask, intervention):
            if intervention is None or intervention.get("module") != module_name:
                return gate
            mode = intervention["mode"]
            if mode == "mean_clamp":
                if not hasattr(self, "_gate_reference"):
                    raise RuntimeError("Gate reference has not been calibrated.")
                replacement = self._gate_reference[module_name].view(1, -1).expand_as(gate)
            elif mode == "zero":
                replacement = torch.zeros_like(gate)
            elif mode == "open":
                replacement = torch.ones_like(gate)
            else:
                raise ValueError(f"Unknown intervention mode: {mode}")
            return torch.where(mask.unsqueeze(1), replacement, gate)

        def forward(
            self,
            x,
            a,
            cueA_zones=None,
            cueB_zones=None,
            decision_positions=None,
            return_details=False,
            skip_readout=False,
            phase=None,
            intervention=None,
            readout_width=4,
        ):
            if intervention is not None:
                if phase is None or decision_positions is None:
                    raise ValueError(
                        "Intervention scheduling requires phase and decision_positions."
                    )

            B, T, _ = x.shape
            g = torch.zeros(B, self.hidden_dim, device=x.device)
            M = torch.zeros(B, self.hidden_dim, self.hidden_dim, device=x.device)
            v = torch.zeros(B, self.fb_dim, device=x.device)
            decision_tensor = (
                torch.as_tensor(decision_positions, dtype=torch.long, device=x.device)
                if decision_positions is not None
                else None
            )

            pred_loss = 0.0
            mem_list, xhat_list, g_list = [], [], []
            ec_list, ca3_list, ca1_list = [], [], []
            write_strength_list, v_list = [], []
            pos_logits_list, phase_logits_list = [], []
            intervention_mask_list = []

            for t in range(T - 1):
                x_t = x[:, t]
                a_t = a[:, t]
                if intervention is None:
                    active_mask = torch.zeros(B, dtype=torch.bool, device=x.device)
                else:
                    active_mask = self._stage_mask(
                        phase[:, t],
                        t,
                        decision_tensor,
                        intervention["stage"],
                        readout_width,
                    )

                gate_ec = torch.sigmoid(
                    self.gate_ec(torch.cat([x_t, g, v], dim=1))
                )
                gate_ec = self._apply_gate(gate_ec, "ec", active_mask, intervention)
                x_mod = gate_ec * x_t
                x_emb = torch.tanh(self.fc_x(x_mod))

                a_emb = self.fc_a(self.action_emb(a_t))
                g_pred = torch.tanh(self.fc_g(g) + a_emb)

                h = x_emb
                ca3_steps = []
                for _ in range(5):
                    gate_ca3 = torch.sigmoid(
                        self.gate_ca3(torch.cat([h, g], dim=1))
                    )
                    gate_ca3 = self._apply_gate(
                        gate_ca3, "ca3", active_mask, intervention
                    )
                    recur = torch.tanh(
                        0.5 * h + torch.bmm(M, h.unsqueeze(2)).squeeze(2)
                    )
                    h = (1.0 - gate_ca3) * h + gate_ca3 * recur
                    ca3_steps.append(gate_ca3)

                p_t = h
                g = g_pred + self.alpha * p_t

                p_hat = torch.tanh(
                    0.5 * g + torch.bmm(M, g.unsqueeze(2)).squeeze(2)
                )
                gain_ca1 = torch.sigmoid(
                    self.gate_ca1(torch.cat([g, p_hat, v], dim=1))
                )
                gain_ca1 = self._apply_gate(
                    gain_ca1, "ca1", active_mask, intervention
                )
                p_out = torch.relu(
                    p_hat * (self.c1 + self.c2 * gain_ca1) - self.beta
                )

                x_hat = self.decoder(p_out)
                pos_logits = self.pos_head(p_out)
                phase_logits = self.phase_head(p_out)
                pred_loss = pred_loss + ((x_hat - x[:, t + 1]) ** 2).mean()

                v = self.rho * v + self.fb_proj(p_out)
                write_strength = gate_ec.mean(dim=1, keepdim=True)
                outer = torch.bmm(p_t.unsqueeze(2), p_t.unsqueeze(1))
                M = self.lambda_m * M + write_strength.unsqueeze(-1) * outer

                mem_list.append(p_out)
                xhat_list.append(x_hat)
                g_list.append(g)
                ec_list.append(gate_ec)
                ca3_list.append(torch.stack(ca3_steps, dim=1).mean(dim=1))
                ca1_list.append(gain_ca1)
                write_strength_list.append(write_strength.squeeze(1))
                v_list.append(v)
                pos_logits_list.append(pos_logits)
                phase_logits_list.append(phase_logits)
                intervention_mask_list.append(active_mask)

            mem = torch.stack(mem_list, dim=1)
            xhat = torch.stack(xhat_list, dim=1)
            g_hist = torch.stack(g_list, dim=1)
            ec_hist = torch.stack(ec_list, dim=1)
            ca3_hist = torch.stack(ca3_list, dim=1)
            ca1_hist = torch.stack(ca1_list, dim=1)
            write_hist = torch.stack(write_strength_list, dim=1)
            v_hist = torch.stack(v_list, dim=1)
            pos_logits_hist = torch.stack(pos_logits_list, dim=1)
            phase_logits_hist = torch.stack(phase_logits_list, dim=1)
            intervention_mask_hist = torch.stack(intervention_mask_list, dim=1)

            if skip_readout:
                logits = None
                readout_out = None
            elif return_details and hasattr(self.readout, "cueA_pool"):
                readout_out = self.readout(
                    mem,
                    cueA_zones,
                    cueB_zones,
                    decision_positions,
                    return_details=True,
                )
                logits = readout_out["logits"]
            else:
                logits = self.readout(
                    mem, cueA_zones, cueB_zones, decision_positions
                )
                readout_out = None

            cueA_aux_logits = None
            if cueA_zones is not None and self.cueA_aux_head is not None:
                cueA_reprs = [
                    mem[b, a0:a1, :].mean(dim=0)
                    for b, (a0, a1) in enumerate(cueA_zones)
                ]
                cueA_aux_logits = self.cueA_aux_head(torch.stack(cueA_reprs))

            cueB_aux_logits = None
            if cueB_zones is not None and self.cueB_aux_head is not None:
                cueB_reprs = [
                    mem[b, b0:b1, :].mean(dim=0)
                    for b, (b0, b1) in enumerate(cueB_zones)
                ]
                cueB_aux_logits = self.cueB_aux_head(torch.stack(cueB_reprs))

            if return_details:
                out = {
                    "pred_loss": pred_loss,
                    "logits": logits,
                    "mem": mem,
                    "xhat": xhat,
                    "g_hist": g_hist,
                    "ec_hist": ec_hist,
                    "ca3_hist": ca3_hist,
                    "ca1_hist": ca1_hist,
                    "write_hist": write_hist,
                    "v_hist": v_hist,
                    "pos_logits_hist": pos_logits_hist,
                    "phase_logits_hist": phase_logits_hist,
                    "cueA_aux_logits": cueA_aux_logits,
                    "cueB_aux_logits": cueB_aux_logits,
                    "intervention_mask_hist": intervention_mask_hist,
                }
                if readout_out is not None:
                    for key in (
                        "cueA_attn",
                        "cueB_attn",
                        "cueA_repr",
                        "cueB_repr",
                        "decision_repr",
                    ):
                        out[key] = readout_out[key]
                return out
            return pred_loss, logits

    return IntervenableHDGM


def instantiate_model(main_module, cfg, env, checkpoint, checkpoint_step, device):
    model_class = make_intervenable_hdgm_class(main_module)
    model = model_class(
        obs_dim=cfg.obs_dim,
        hidden_dim=96,
        track_len=env.track_len,
        action_dim=env.action_dim,
        n_classes=4,
        cfg=cfg,
    ).to(device)
    snapshots = checkpoint["snapshots"]
    if checkpoint_step not in snapshots:
        raise KeyError(
            f"Checkpoint step {checkpoint_step} is unavailable; "
            f"found {sorted(snapshots)}"
        )
    state = snapshots[checkpoint_step].get("hdgm")
    if state is None:
        state = snapshots[checkpoint_step].get("gate")
    if state is None:
        raise KeyError("Snapshot contains neither 'hdgm' nor 'gate' state.")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def unpack_batch(batch):
    return {
        "x": batch[0],
        "a": batch[1],
        "pos": batch[2],
        "phase": batch[3],
        "labels": batch[4],
        "cueA": batch[5],
        "cueB": batch[6],
        "cueA_zones": batch[7],
        "cueB_zones": batch[8],
        "decision_positions": batch[9],
        "delay1s": batch[10],
        "delay2s": batch[11],
    }


def forward_batch(model, batch, device, intervention, readout_width, skip_readout=False):
    item = unpack_batch(batch)
    return model(
        item["x"].to(device),
        item["a"].to(device),
        cueA_zones=None if skip_readout else item["cueA_zones"],
        cueB_zones=None if skip_readout else item["cueB_zones"],
        decision_positions=item["decision_positions"],
        return_details=True,
        skip_readout=skip_readout,
        phase=item["phase"].to(device),
        intervention=intervention,
        readout_width=readout_width,
    )


def validate_control_equivalence(
    main_module, cfg, env, intervenable_model, batch, device, readout_width
):
    """Verify that the added intervention interface changes nothing in control."""
    reference_model = main_module.HDGM(
        obs_dim=cfg.obs_dim,
        hidden_dim=96,
        track_len=env.track_len,
        action_dim=env.action_dim,
        n_classes=4,
        cfg=cfg,
    ).to(device)
    reference_model.load_state_dict(intervenable_model.state_dict(), strict=True)
    reference_model.eval()
    item = unpack_batch(batch)
    with torch.no_grad():
        reference = reference_model(
            item["x"].to(device),
            item["a"].to(device),
            cueA_zones=item["cueA_zones"],
            cueB_zones=item["cueB_zones"],
            decision_positions=item["decision_positions"],
            return_details=True,
        )
        candidate = forward_batch(
            intervenable_model,
            batch,
            device,
            intervention=None,
            readout_width=readout_width,
            skip_readout=False,
        )
    checked_keys = (
        "logits", "mem", "g_hist", "ec_hist", "ca3_hist", "ca1_hist",
        "write_hist", "v_hist",
    )
    errors = {
        key: float((reference[key] - candidate[key]).abs().max().item())
        for key in checked_keys
    }
    overall = max(errors.values())
    if overall > 1e-7:
        raise RuntimeError(
            f"Control-equivalence validation failed: max_abs_error={overall}"
        )
    del reference_model
    return {"max_abs_error": overall, "by_tensor": errors}


def calibrate_gate_reference(model, analysis_bank, device, readout_width):
    sums = {
        "ec": torch.zeros(model.obs_dim, dtype=torch.float64, device=device),
        "ca3": torch.zeros(model.hidden_dim, dtype=torch.float64, device=device),
        "ca1": torch.zeros(model.hidden_dim, dtype=torch.float64, device=device),
    }
    counts = {key: 0 for key in sums}
    key_map = {"ec": "ec_hist", "ca3": "ca3_hist", "ca1": "ca1_hist"}
    with torch.no_grad():
        for batch in analysis_bank:
            out = forward_batch(
                model, batch, device, intervention=None,
                readout_width=readout_width, skip_readout=True,
            )
            for module_name, history_key in key_map.items():
                history = out[history_key].double()
                sums[module_name] += history.sum(dim=(0, 1))
                counts[module_name] += history.shape[0] * history.shape[1]
    reference = {
        module_name: (sums[module_name] / counts[module_name]).float()
        for module_name in sums
    }
    model.set_gate_reference(reference)
    return {
        key: value.detach().cpu().numpy().tolist()
        for key, value in reference.items()
    }


def confusion_from_predictions(labels, predictions, n_classes=4):
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, predicted in zip(labels, predictions):
        matrix[int(truth), int(predicted)] += 1
    return matrix


def metrics_from_confusion(matrix):
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else float("nan")
    class_accuracy, f1_values = [], []
    for class_id in range(matrix.shape[0]):
        tp = float(matrix[class_id, class_id])
        fn = float(matrix[class_id, :].sum() - tp)
        fp = float(matrix[:, class_id].sum() - tp)
        class_total = tp + fn
        class_accuracy.append(float(tp / class_total) if class_total else None)
        denominator = 2.0 * tp + fp + fn
        f1_values.append(float(2.0 * tp / denominator) if denominator else 0.0)
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "class_accuracy": class_accuracy,
        "confusion_matrix": matrix.tolist(),
        "n_trials": total,
    }


def evaluate_behavior(model, bank, device, intervention, readout_width):
    labels_all, predictions_all = [], []
    intervened_steps = 0
    possible_steps = 0
    with torch.no_grad():
        for batch in bank:
            item = unpack_batch(batch)
            out = forward_batch(
                model, batch, device, intervention,
                readout_width=readout_width, skip_readout=False,
            )
            predictions = out["logits"].argmax(dim=1).detach().cpu().numpy()
            labels = item["labels"].detach().cpu().numpy()
            labels_all.append(labels)
            predictions_all.append(predictions)
            mask = out["intervention_mask_hist"]
            intervened_steps += int(mask.sum().item())
            possible_steps += int(mask.numel())
    labels_all = np.concatenate(labels_all)
    predictions_all = np.concatenate(predictions_all)
    matrix = confusion_from_predictions(labels_all, predictions_all, N_CLASSES)
    result = metrics_from_confusion(matrix)
    result["intervened_time_steps"] = intervened_steps
    result["intervened_steps_per_trial"] = float(
        intervened_steps / max(1, len(labels_all))
    )
    result["intervened_fraction"] = float(
        intervened_steps / max(1, possible_steps)
    )
    return result


def residualize_by_position(features, positions):
    features = np.asarray(features, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    residual = features.copy()
    for position in np.unique(positions):
        mask = positions == position
        residual[mask] -= residual[mask].mean(axis=0, keepdims=True)
    return residual


def stratified_ridge_accuracy(features, labels, seed):
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    classifier = make_pipeline(
        StandardScaler(),
        RidgeClassifier(alpha=1.0),
    )
    scores = cross_val_score(classifier, features, labels, cv=splitter, n_jobs=1)
    return float(scores.mean())


def standardized_binary_distance(features, labels):
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    scale = features.std(axis=0, ddof=0)
    scale[scale < 1e-6] = 1.0
    standardized = features / scale
    delta = standardized[labels == 1].mean(axis=0) - standardized[labels == 0].mean(axis=0)
    return float(np.linalg.norm(delta) / math.sqrt(features.shape[1]))


def evaluate_representations(model, bank, device, intervention, readout_width, seed):
    delay_states, delay_positions, delay_trial_ids = [], [], []
    cue_a_by_trial, class_by_trial, decision_states = [], [], []
    trial_offset = 0
    with torch.no_grad():
        for batch in bank:
            item = unpack_batch(batch)
            out = forward_batch(
                model, batch, device, intervention,
                readout_width=readout_width, skip_readout=True,
            )
            mem = out["mem"].detach().cpu().numpy()
            phase = item["phase"][:, :mem.shape[1]].numpy()
            pos = item["pos"][:, :mem.shape[1]].numpy()
            cue_a = item["cueA"].numpy()
            labels = item["labels"].numpy()
            for local_trial in range(mem.shape[0]):
                global_trial = trial_offset + local_trial
                indices = np.flatnonzero(phase[local_trial] == 2)
                if len(indices) == 0:
                    raise RuntimeError("A trial has no Delay-1 samples.")
                delay_states.append(mem[local_trial, indices])
                delay_positions.append(pos[local_trial, indices])
                delay_trial_ids.append(
                    np.full(len(indices), global_trial, dtype=np.int64)
                )
                decision_index = max(
                    0,
                    min(
                        int(item["decision_positions"][local_trial]) - 1,
                        mem.shape[1] - 1,
                    ),
                )
                decision_states.append(mem[local_trial, decision_index])
                cue_a_by_trial.append(int(cue_a[local_trial]))
                class_by_trial.append(int(labels[local_trial]))
            trial_offset += mem.shape[0]

    delay_states = np.concatenate(delay_states, axis=0)
    delay_positions = np.concatenate(delay_positions, axis=0)
    delay_trial_ids = np.concatenate(delay_trial_ids, axis=0)
    delay_residual = residualize_by_position(delay_states, delay_positions)
    n_trials = len(cue_a_by_trial)
    trial_features = np.zeros((n_trials, delay_residual.shape[1]), dtype=np.float64)
    for trial_id in range(n_trials):
        trial_features[trial_id] = delay_residual[delay_trial_ids == trial_id].mean(axis=0)

    cue_a_by_trial = np.asarray(cue_a_by_trial, dtype=np.int64)
    class_by_trial = np.asarray(class_by_trial, dtype=np.int64)
    decision_states = np.asarray(decision_states, dtype=np.float64)
    return {
        "delay1_cueA_decoding": stratified_ridge_accuracy(
            trial_features, cue_a_by_trial, seed + 101
        ),
        "delay1_state_distance": standardized_binary_distance(
            trial_features, cue_a_by_trial
        ),
        "decision_pre_class_decoding": stratified_ridge_accuracy(
            decision_states, class_by_trial, seed + 202
        ),
        "n_analysis_trials": int(n_trials),
    }


def intervention_specs(modes):
    yield {
        "condition": "control",
        "mode": "control",
        "module": None,
        "stage": None,
        "intervention": None,
    }
    for mode, module_name, stage in itertools.product(modes, MODULES, STAGES):
        yield {
            "condition": f"{mode}__{module_name}__{stage}",
            "mode": mode,
            "module": module_name,
            "stage": stage,
            "intervention": {
                "mode": mode,
                "module": module_name,
                "stage": stage,
            },
        }


def exact_sign_flip_test(differences):
    differences = np.asarray(differences, dtype=np.float64)
    observed = abs(float(differences.mean()))
    n = len(differences)
    exceed = 0
    total = 2 ** n
    for bits in range(total):
        signs = np.asarray(
            [1.0 if (bits >> index) & 1 else -1.0 for index in range(n)]
        )
        statistic = abs(float((differences * signs).mean()))
        if statistic >= observed - 1e-15:
            exceed += 1
    sd = float(differences.std(ddof=1)) if n > 1 else 0.0
    return {
        "mean_difference": float(differences.mean()),
        "p_two_sided_exact": float(exceed / total),
        "cohen_dz": float(differences.mean() / (sd + 1e-12)),
        "n": int(n),
    }


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def aggregate_statistics(records, modes):
    output = {}
    for mode in modes:
        condition_ids, tests = [], []
        for module_name, stage in itertools.product(MODULES, STAGES):
            condition = f"{mode}__{module_name}__{stage}"
            differences = [
                record["conditions"]["control"]["normal"]["accuracy"]
                - record["conditions"][condition]["normal"]["accuracy"]
                for record in records
            ]
            condition_ids.append(condition)
            tests.append(exact_sign_flip_test(differences))
        adjusted = holm_adjust(
            [test["p_two_sided_exact"] for test in tests]
        )
        output[mode] = {}
        for condition, test, corrected in zip(condition_ids, tests, adjusted):
            test["holm_p_across_9_conditions"] = float(corrected)
            output[mode][condition] = test
    return output


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def atomic_json_dump(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def write_csv(records, path):
    rows = []
    for record in records:
        control = record["conditions"]["control"]
        for condition_name, condition in record["conditions"].items():
            normal = condition["normal"]
            long_result = condition["long"]
            representation = condition.get("representation", {})
            rows.append({
                "seed": record["seed"],
                "condition": condition_name,
                "mode": condition["mode"],
                "module": condition["module"],
                "stage": condition["stage"],
                "normal_accuracy": normal["accuracy"],
                "normal_macro_f1": normal["macro_f1"],
                "long_accuracy": long_result["accuracy"],
                "long_macro_f1": long_result["macro_f1"],
                "normal_accuracy_drop": control["normal"]["accuracy"] - normal["accuracy"],
                "long_accuracy_drop": control["long"]["accuracy"] - long_result["accuracy"],
                "intervened_steps_per_trial_normal": normal["intervened_steps_per_trial"],
                "intervened_fraction_normal": normal["intervened_fraction"],
                "delay1_cueA_decoding": representation.get("delay1_cueA_decoding"),
                "delay1_state_distance": representation.get("delay1_state_distance"),
                "decision_pre_class_decoding": representation.get("decision_pre_class_decoding"),
            })
    fieldnames = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_npz(records, modes, path):
    seeds = np.asarray([record["seed"] for record in records], dtype=np.int64)
    shape = (len(records), len(modes), len(MODULES), len(STAGES))
    normal_drop = np.full(shape, np.nan, dtype=np.float64)
    long_drop = np.full(shape, np.nan, dtype=np.float64)
    macro_f1_drop = np.full(shape, np.nan, dtype=np.float64)
    history_decode_drop = np.full(shape, np.nan, dtype=np.float64)
    state_distance_drop = np.full(shape, np.nan, dtype=np.float64)
    decision_decode_drop = np.full(shape, np.nan, dtype=np.float64)
    step_count = np.full(shape, np.nan, dtype=np.float64)
    control_normal = np.empty(len(records), dtype=np.float64)
    control_long = np.empty(len(records), dtype=np.float64)

    for seed_index, record in enumerate(records):
        control = record["conditions"]["control"]
        control_normal[seed_index] = control["normal"]["accuracy"]
        control_long[seed_index] = control["long"]["accuracy"]
        control_rep = control.get("representation", {})
        for mode_index, mode in enumerate(modes):
            for module_index, module_name in enumerate(MODULES):
                for stage_index, stage in enumerate(STAGES):
                    condition = record["conditions"][
                        f"{mode}__{module_name}__{stage}"
                    ]
                    normal_drop[seed_index, mode_index, module_index, stage_index] = (
                        control["normal"]["accuracy"] - condition["normal"]["accuracy"]
                    )
                    long_drop[seed_index, mode_index, module_index, stage_index] = (
                        control["long"]["accuracy"] - condition["long"]["accuracy"]
                    )
                    macro_f1_drop[seed_index, mode_index, module_index, stage_index] = (
                        control["normal"]["macro_f1"] - condition["normal"]["macro_f1"]
                    )
                    step_count[seed_index, mode_index, module_index, stage_index] = (
                        condition["normal"]["intervened_steps_per_trial"]
                    )
                    rep = condition.get("representation", {})
                    if rep and control_rep:
                        history_decode_drop[seed_index, mode_index, module_index, stage_index] = (
                            control_rep["delay1_cueA_decoding"] - rep["delay1_cueA_decoding"]
                        )
                        state_distance_drop[seed_index, mode_index, module_index, stage_index] = (
                            control_rep["delay1_state_distance"] - rep["delay1_state_distance"]
                        )
                        decision_decode_drop[seed_index, mode_index, module_index, stage_index] = (
                            control_rep["decision_pre_class_decoding"]
                            - rep["decision_pre_class_decoding"]
                        )

    np.savez_compressed(
        path,
        seeds=seeds,
        modes=np.asarray(modes),
        modules=np.asarray(MODULES),
        stages=np.asarray(STAGES),
        control_normal_accuracy=control_normal,
        control_long_accuracy=control_long,
        normal_accuracy_drop=normal_drop,
        long_accuracy_drop=long_drop,
        normal_macro_f1_drop=macro_f1_drop,
        delay1_cueA_decoding_drop=history_decode_drop,
        delay1_state_distance_drop=state_distance_drop,
        decision_pre_class_decoding_drop=decision_decode_drop,
        intervened_steps_per_trial=step_count,
    )


def run_seed(
    seed,
    checkpoint_path,
    checkpoint,
    main_module,
    args,
    device,
    modes,
):
    print("\n" + "=" * 112)
    print(f"STAGE-SPECIFIC PERTURBATION | seed={seed} | device={device}")
    print("=" * 112)
    set_seed(seed)
    cfg = config_from_checkpoint(main_module, checkpoint)
    env = build_environment(main_module, cfg, checkpoint, seed)
    test_batches = 2 if args.quick else (args.test_batches or cfg.test_batches)
    analysis_batches = 2 if args.quick else (
        args.analysis_batches or cfg.analysis_batches
    )
    test_banks, analysis_bank = build_locked_banks(
        main_module,
        env,
        cfg,
        checkpoint,
        test_batches=test_batches,
        analysis_batches=analysis_batches,
    )
    model = instantiate_model(
        main_module,
        cfg,
        env,
        checkpoint,
        args.checkpoint_step,
        device,
    )
    equivalence = validate_control_equivalence(
        main_module,
        cfg,
        env,
        model,
        analysis_bank[0],
        device,
        args.readout_width,
    )
    print(
        "Control-equivalence max_abs_error="
        f"{equivalence['max_abs_error']:.3g}"
    )
    reference = calibrate_gate_reference(
        model, analysis_bank, device, args.readout_width
    )
    conditions = {}
    for index, spec in enumerate(intervention_specs(modes), start=1):
        print(
            f"[{index:02d}/{1 + 9 * len(modes):02d}] "
            f"{spec['condition']}",
            flush=True,
        )
        normal = evaluate_behavior(
            model,
            test_banks["normal"],
            device,
            spec["intervention"],
            args.readout_width,
        )
        long_result = evaluate_behavior(
            model,
            test_banks["long"],
            device,
            spec["intervention"],
            args.readout_width,
        )
        condition = {
            "mode": spec["mode"],
            "module": spec["module"],
            "stage": spec["stage"],
            "normal": normal,
            "long": long_result,
        }
        if not args.skip_representation:
            condition["representation"] = evaluate_representations(
                model,
                analysis_bank,
                device,
                spec["intervention"],
                args.readout_width,
                seed,
            )
        conditions[spec["condition"]] = condition
        print(
            f"    normal={normal['accuracy']:.4f}, "
            f"long={long_result['accuracy']:.4f}, "
            f"steps/trial={normal['intervened_steps_per_trial']:.2f}"
        )

    return {
        "seed": int(seed),
        "checkpoint_file": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": int(args.checkpoint_step),
        "config": asdict(cfg),
        "split_seeds": copy.deepcopy(checkpoint["split_seeds"]),
        "test_batches_per_condition": int(test_batches),
        "analysis_batches_per_condition": int(analysis_batches),
        "control_equivalence": equivalence,
        "gate_reference_per_unit": reference,
        "conditions": conditions,
    }


def main():
    args = parse_args()
    requested_seeds = tuple(
        int(value.strip()) for value in args.seeds.split(",") if value.strip()
    )
    modes = tuple(value.strip() for value in args.modes.split(",") if value.strip())
    invalid_modes = sorted(set(modes) - set(PERTURBATION_MODES))
    if invalid_modes:
        raise ValueError(f"Unsupported modes: {invalid_modes}")
    if args.quick:
        requested_seeds = requested_seeds[:1]
    if args.readout_width < 1:
        raise ValueError("readout_width must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = (
        "hdgm_stage_specific_perturbation_v1_quick_audit"
        if args.quick
        else "hdgm_stage_specific_perturbation_v1"
    )
    output_json = args.output_dir / f"{output_stem}.json"
    output_csv = args.output_dir / f"{output_stem}.csv"
    output_npz = args.output_dir / f"{output_stem}.npz"
    partial_json = args.output_dir / f"{output_stem}.partial.json"

    device = choose_device(args.device)
    main_module = load_main_module(args.main_code)
    main_module.device = device
    checkpoints = discover_checkpoints(args.checkpoint_dir, requested_seeds)

    protocol = {
        "experiment": "HDGM V4 frozen stage-specific perturbation v1",
        "main_code": str(args.main_code.resolve()),
        "main_code_sha256": sha256_file(args.main_code),
        "checkpoint_step": int(args.checkpoint_step),
        "seeds": list(requested_seeds),
        "modules": list(MODULES),
        "stages": list(STAGES),
        "modes": list(modes),
        "readout_width": int(args.readout_width),
        "stage_definition": {
            "encoding": "phase 1 or 3 (Cue A or Cue B)",
            "maintenance": "phase 2 or 4 excluding terminal readout window",
            "readout": "terminal readout_width samples ending at decision",
        },
        "mean_clamp_definition": (
            "phase-blind per-unit gate mean estimated on the independent "
            "fixed analysis bank"
        ),
        "zero_definition": "selected gate forced to zero inside forward dynamics",
        "weights_frozen": True,
        "no_retraining": True,
        "identical_fixed_banks_across_conditions": True,
        "phase_metadata_used_only_for_test_intervention_schedule": True,
        "quick_audit_not_for_paper": bool(args.quick),
    }

    records = []
    for seed in requested_seeds:
        path, checkpoint = checkpoints[seed]
        record = run_seed(
            seed,
            path,
            checkpoint,
            main_module,
            args,
            device,
            modes,
        )
        records.append(record)
        atomic_json_dump(
            {"protocol": protocol, "records": records}, partial_json
        )

    statistics = (
        aggregate_statistics(records, modes)
        if len(records) >= 2
        else {"note": "Quick audit has one seed; confirmatory statistics omitted."}
    )
    payload = {
        "protocol": protocol,
        "records": records,
        "statistics": statistics,
    }
    atomic_json_dump(payload, output_json)
    write_csv(records, output_csv)
    write_npz(records, modes, output_npz)
    partial_json.unlink(missing_ok=True)

    print("\n" + "#" * 112)
    print("EXPORT COMPLETE")
    print("#" * 112)
    print(output_json)
    print(output_csv)
    print(output_npz)
    if args.quick:
        print("WARNING: --quick output is a pipeline audit, not a paper result.")


if __name__ == "__main__":
    main()
