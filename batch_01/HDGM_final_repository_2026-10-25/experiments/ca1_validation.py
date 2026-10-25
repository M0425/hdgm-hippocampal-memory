"""
HDGM V6.2: independently calibrated causal CA1-like integration validation.

This standalone file preserves the complete V4 task/model implementation and
loads the exact frozen update-100 checkpoint for each seed. It corrects the V6
audit issue: no intervention may average, shuffle, or otherwise move future
test-trial gates into an earlier time point.

The active V6.2 main compares Full CA1 against fixed-open, a global unit-wise
mean estimated only from an independent unlabeled calibration bank, a causal
running mean, and a one-step-lagged schedule. Recurrent weights are never
updated. Cue windows are used only by the terminal event-aligned readout, while
labels and phase masks never enter recurrent dynamics or schedule generation.

Some legacy helpers remain above the V6.2 section solely for source-level
compatibility; the executable main calls only the v62_* locked pipeline.
"""

import copy
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =========================================================
# 0. Basic setup
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =========================================================
# 1. Experiment config (paper main version)
# =========================================================
@dataclass
class ExperimentConfig:
    # ===== core structure =====
    use_branch_readout: bool = True

    # ===== auxiliary heads off in thesis-paper main version =====
    use_cueA_aux_head: bool = False
    use_cueB_aux_head: bool = False
    use_phase_aux_head: bool = False

    # ===== keep only minimal mechanism losses =====
    use_ec_phase_rank_loss: bool = True
    use_position_aux_loss: bool = False
    use_cueB_margin_loss: bool = True
    use_cueB_peak_write_loss: bool = False

    # ===== gate loss weights =====
    gate_phase_weight: float = 0.00
    gate_cueB_margin_weight: float = 0.08
    gate_ec_phase_rank_weight: float = 0.12
    gate_pos_aux_weight: float = 0.00
    gate_cueA_aux_weight: float = 0.00
    gate_cueB_aux_weight: float = 0.00
    gate_phase_aux_weight: float = 0.00
    gate_cueB_peak_weight: float = 0.00

    # ===== TEM position aux off =====
    tem_pos_aux_weight: float = 0.00

    # ===== no gate-only class reweighting in paper main version =====
    gate_class_weights: tuple = (1.0, 1.0, 1.0, 1.0)

    # ===== training =====
    # V4 confirmatory protocol: the biological hypothesis concerns rapid
    # acquisition, not asymptotic task optimization. `epochs` is retained for
    # backward compatibility; the V4 runner uses `primary_updates`.
    epochs: int = 150
    batch_size: int = 32
    lr: float = 6e-4

    # ===== V4 rapid-acquisition protocol =====
    primary_updates: int = 150
    rapid_checkpoints: tuple = (10, 25, 50, 75, 100, 150)
    rapid_probe_checkpoints: tuple = (50, 100, 150)
    acquisition_thresholds: tuple = (0.50, 0.60)
    extended_updates: int = 500
    extended_checkpoints: tuple = (10, 25, 50, 75, 100, 150, 300, 500)

    # ===== validation monitoring =====
    val_batches: int = 10
    val_batch_size: int = 64
    val_score_alpha: float = 0.75   # score = alpha*normal + (1-alpha)*long
    val_interval: int = 10
    print_interval: int = 50

    # ===== fixed held-out evaluation =====
    # V4 validation is monitoring-only. Exact checkpoints are predeclared.
    # Test batches are generated once from an independent RNG stream and are
    # never used during training or model selection.
    test_batches: int = 50
    test_batch_size: int = 64
    analysis_batches: int = 5
    analysis_batch_size: int = 64
    split_seed_version: int = 4
    checkpoint_dir: str = "paper_rapid_v4_outputs/checkpoints"

    # ===== frozen decision-state probe =====
    probe_train_batches: int = 20
    probe_val_batches: int = 5
    probe_test_batches: int = 20
    probe_batch_size: int = 64
    probe_epochs: int = 200
    probe_lr: float = 1e-2
    probe_weight_decay: float = 1e-4

    # ===== environment =====
    track_len: int = 80
    obs_dim: int = 24
    cue_width: int = 4
    cueA_start_range: tuple = (3, 14)
    delay1_range: tuple = (8, 20)
    delay2_range: tuple = (10, 24)
    n_cues: int = 2
    alias_prob: float = 0.10
    noise_std: float = 0.08

    name: str = "fourclass_rapid_acquisition_v4"

    # ===== visualization =====
    save_dir: str = "paper_rapid_v4_outputs"
    representative_seed: int = 11


# =========================================================
# 2. Environment
# =========================================================
class SharedPathFourContextEnv:
    def __init__(
        self,
        track_len=80,
        obs_dim=24,
        cue_width=4,
        cueA_start_range=(3, 14),
        delay1_range=(8, 20),
        delay2_range=(10, 24),
        n_cues=2,
        alias_prob=0.10,
        noise_std=0.08,
        seed=None,
    ):
        self.track_len = track_len
        self.obs_dim = obs_dim
        self.cue_width = cue_width
        self.cueA_start_range = cueA_start_range
        self.delay1_range = delay1_range
        self.delay2_range = delay2_range
        self.n_cues = n_cues
        self.alias_prob = alias_prob
        self.noise_std = noise_std

        self.rng = np.random.RandomState(seed) if seed is not None else None

        self.pos_map = self._randn(track_len, obs_dim).astype(np.float32)
        for i in range(track_len):
            if self._rand() < alias_prob:
                j = self._randint(0, track_len)
                self.pos_map[i] = self.pos_map[j]

        self.cueA_map = self._randn(n_cues, obs_dim).astype(np.float32)
        self.cueB_map = self._randn(n_cues, obs_dim).astype(np.float32)

        self.action_dim = 1
        self.forward_action = 0

    def _randint(self, low, high=None):
        if self.rng is None:
            return np.random.randint(low, high)
        return self.rng.randint(low, high)

    def _randn(self, *shape):
        if self.rng is None:
            return np.random.randn(*shape)
        return self.rng.randn(*shape)

    def _rand(self):
        if self.rng is None:
            return np.random.rand()
        return self.rng.rand()

    def context_rule(self, cueA, cueB):
        return int(cueA * 2 + cueB)

    def _sample_valid_structure(self):
        max_tries = 500
        for _ in range(max_tries):
            a0 = self._randint(self.cueA_start_range[0], self.cueA_start_range[1] + 1)
            delay1 = self._randint(self.delay1_range[0], self.delay1_range[1] + 1)
            delay2 = self._randint(self.delay2_range[0], self.delay2_range[1] + 1)

            cueA_zone = (a0, a0 + self.cue_width)
            b0 = cueA_zone[1] + delay1
            cueB_zone = (b0, b0 + self.cue_width)
            decision_pos = cueB_zone[1] + delay2

            if decision_pos < self.track_len:
                return cueA_zone, cueB_zone, decision_pos, delay1, delay2

        raise RuntimeError("无法采样到合法 trial 结构，请增大 track_len 或缩小 delay range")

    def sample_trial(
        self,
        forced_cueA=None,
        forced_cueB=None,
        cueA_zone_override=None,
        cueB_zone_override=None,
        decision_pos_override=None,
        delay1_override=None,
        delay2_override=None,
        mask_decision_obs=False,
    ):
        cueA = self._randint(0, self.n_cues) if forced_cueA is None else int(forced_cueA)
        cueB = self._randint(0, self.n_cues) if forced_cueB is None else int(forced_cueB)

        if cueA_zone_override is None or cueB_zone_override is None or decision_pos_override is None:
            if delay1_override is None and delay2_override is None:
                cueA_zone, cueB_zone, decision_pos, delay1, delay2 = self._sample_valid_structure()
            else:
                a0 = self._randint(self.cueA_start_range[0], self.cueA_start_range[1] + 1)
                delay1 = int(delay1_override) if delay1_override is not None else self._randint(self.delay1_range[0], self.delay1_range[1] + 1)
                delay2 = int(delay2_override) if delay2_override is not None else self._randint(self.delay2_range[0], self.delay2_range[1] + 1)

                cueA_zone = (a0, a0 + self.cue_width)
                b0 = cueA_zone[1] + delay1
                cueB_zone = (b0, b0 + self.cue_width)
                decision_pos = cueB_zone[1] + delay2

                if decision_pos >= self.track_len:
                    cueA_zone, cueB_zone, decision_pos, delay1, delay2 = self._sample_valid_structure()
        else:
            cueA_zone = tuple(cueA_zone_override)
            cueB_zone = tuple(cueB_zone_override)
            decision_pos = int(decision_pos_override)
            delay1 = cueB_zone[0] - cueA_zone[1]
            delay2 = decision_pos - cueB_zone[1]

        assert cueA_zone[1] < cueB_zone[0]
        assert cueB_zone[1] < decision_pos
        assert decision_pos < self.track_len

        label = self.context_rule(cueA, cueB)

        xs, acts, pos, phase = [], [], [], []

        for t in range(self.track_len):
            x = self.pos_map[t].copy()

            in_A = cueA_zone[0] <= t < cueA_zone[1]
            in_B = cueB_zone[0] <= t < cueB_zone[1]
            in_decision = t == decision_pos

            if in_A:
                x = x + self.cueA_map[cueA]
                phase.append(1)
            elif in_B:
                x = x + self.cueB_map[cueB]
                phase.append(3)
            elif in_decision:
                phase.append(5)
            else:
                if t < cueA_zone[0]:
                    phase.append(0)
                elif cueA_zone[1] <= t < cueB_zone[0]:
                    phase.append(2)
                elif cueB_zone[1] <= t < decision_pos:
                    phase.append(4)
                else:
                    phase.append(0)

            if mask_decision_obs and in_decision:
                x = np.zeros_like(x, dtype=np.float32)

            x = x + self._randn(self.obs_dim).astype(np.float32) * self.noise_std

            xs.append(x)
            acts.append(self.forward_action)
            pos.append(t)

        return (
            np.array(xs, dtype=np.float32),
            np.array(acts, dtype=np.int64),
            np.array(pos, dtype=np.int64),
            np.array(phase, dtype=np.int64),
            int(label),
            int(cueA),
            int(cueB),
            tuple(cueA_zone),
            tuple(cueB_zone),
            int(decision_pos),
            int(delay1),
            int(delay2),
        )

    def sample_batch(
        self,
        batch_size=32,
        forced_cueAs=None,
        forced_cueBs=None,
        cueA_zones_override=None,
        cueB_zones_override=None,
        decision_pos_overrides=None,
        delay1_overrides=None,
        delay2_overrides=None,
        mask_decision_obs=False,
    ):
        xs, acts, poss, phases = [], [], [], []
        labels, cueAs, cueBs = [], [], []
        cueA_zones, cueB_zones, decision_positions = [], [], []
        delay1s, delay2s = [], []

        for i in range(batch_size):
            forced_cueA = None if forced_cueAs is None else forced_cueAs[i]
            forced_cueB = None if forced_cueBs is None else forced_cueBs[i]
            cueA_zone_override = None if cueA_zones_override is None else cueA_zones_override[i]
            cueB_zone_override = None if cueB_zones_override is None else cueB_zones_override[i]
            decision_override = None if decision_pos_overrides is None else decision_pos_overrides[i]
            delay1_override = None if delay1_overrides is None else delay1_overrides[i]
            delay2_override = None if delay2_overrides is None else delay2_overrides[i]

            x, a, p, ph, y, cueA, cueB, zA, zB, d, d1, d2 = self.sample_trial(
                forced_cueA=forced_cueA,
                forced_cueB=forced_cueB,
                cueA_zone_override=cueA_zone_override,
                cueB_zone_override=cueB_zone_override,
                decision_pos_override=decision_override,
                delay1_override=delay1_override,
                delay2_override=delay2_override,
                mask_decision_obs=mask_decision_obs,
            )

            xs.append(x)
            acts.append(a)
            poss.append(p)
            phases.append(ph)
            labels.append(y)
            cueAs.append(cueA)
            cueBs.append(cueB)
            cueA_zones.append(zA)
            cueB_zones.append(zB)
            decision_positions.append(d)
            delay1s.append(d1)
            delay2s.append(d2)

        return (
            torch.tensor(np.stack(xs), dtype=torch.float32),
            torch.tensor(np.stack(acts), dtype=torch.long),
            torch.tensor(np.stack(poss), dtype=torch.long),
            torch.tensor(np.stack(phases), dtype=torch.long),
            torch.tensor(np.array(labels), dtype=torch.long),
            torch.tensor(np.array(cueAs), dtype=torch.long),
            torch.tensor(np.array(cueBs), dtype=torch.long),
            cueA_zones,
            cueB_zones,
            decision_positions,
            torch.tensor(np.array(delay1s), dtype=torch.long),
            torch.tensor(np.array(delay2s), dtype=torch.long),
        )


# =========================================================
# 3. Attractor
# =========================================================
def attractor(M, h, steps=5):
    for _ in range(steps):
        h = torch.tanh(0.5 * h + torch.bmm(M, h.unsqueeze(2)).squeeze(2))
    return h


# =========================================================
# 4. Readouts
# =========================================================
class WindowAttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, window_feats):
        scores = self.score(window_feats).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        pooled = (weights.unsqueeze(-1) * window_feats).sum(dim=0)
        return pooled, weights


class BranchReadout(nn.Module):
    """Supervised event-aligned decoder.

    ``cueA_zones`` and ``cueB_zones`` are supplied only to this terminal
    decoder. They never enter the TEM/HDGM recurrent dynamics or gate
    equations. Consequently, the resulting classification score must be
    reported as event-aligned decoding accuracy, not as fully autonomous
    behavioral decision accuracy.
    """
    def __init__(self, hidden_dim, n_classes=4):
        super().__init__()
        self.cueA_pool = WindowAttentionPool(hidden_dim)
        self.cueB_pool = WindowAttentionPool(hidden_dim)

        self.cueA_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.cueB_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.decision_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear((hidden_dim // 2) * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, mem, cueA_zones, cueB_zones, decision_positions, return_details=False):
        reps_a = []
        reps_b = []
        reps_d = []
        cueA_attns = []
        cueB_attns = []

        B, Tm1, _ = mem.shape

        for b in range(B):
            a0, a1 = cueA_zones[b]
            b0, b1 = cueB_zones[b]
            d = max(0, min(decision_positions[b] - 1, Tm1 - 1))

            cueA_repr, cueA_w = self.cueA_pool(mem[b, a0:a1, :])
            cueB_repr, cueB_w = self.cueB_pool(mem[b, b0:b1, :])
            decision_repr = mem[b, d, :]

            reps_a.append(cueA_repr)
            reps_b.append(cueB_repr)
            reps_d.append(decision_repr)
            cueA_attns.append(cueA_w)
            cueB_attns.append(cueB_w)

        reps_a = torch.stack(reps_a, dim=0)
        reps_b = torch.stack(reps_b, dim=0)
        reps_d = torch.stack(reps_d, dim=0)

        fa = self.cueA_branch(reps_a)
        fb = self.cueB_branch(reps_b)
        fd = self.decision_branch(reps_d)

        fused = torch.cat([fa, fb, fd], dim=1)
        logits = self.fusion(fused)

        if return_details:
            return {
                "logits": logits,
                "cueA_attn": cueA_attns,
                "cueB_attn": cueB_attns,
                "cueA_repr": reps_a,
                "cueB_repr": reps_b,
                "decision_repr": reps_d,
            }
        return logits


class SimpleReadout(nn.Module):
    def __init__(self, hidden_dim, n_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, mem, cueA_zones, cueB_zones, decision_positions):
        reps = []
        B, Tm1, _ = mem.shape

        for b in range(B):
            a0, a1 = cueA_zones[b]
            b0, b1 = cueB_zones[b]
            d = max(0, min(decision_positions[b] - 1, Tm1 - 1))

            cueA_repr = mem[b, a0:a1, :].mean(dim=0)
            cueB_repr = mem[b, b0:b1, :].mean(dim=0)
            decision_repr = mem[b, d, :]

            reps.append(torch.cat([cueA_repr, cueB_repr, decision_repr], dim=0))

        reps = torch.stack(reps, dim=0)
        return self.net(reps)


# =========================================================
# 5. Models
# =========================================================
class TEMLite(nn.Module):
    def __init__(self, obs_dim, hidden_dim, track_len, action_dim=1, action_emb_dim=32, n_classes=4, cfg=None):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.track_len = track_len
        self.cfg = cfg if cfg is not None else ExperimentConfig()

        self.fc_x = nn.Linear(obs_dim, hidden_dim)
        self.action_emb = nn.Embedding(action_dim, action_emb_dim)
        self.fc_a = nn.Linear(action_emb_dim, hidden_dim)
        self.fc_g = nn.Linear(hidden_dim, hidden_dim)

        self.decoder = nn.Linear(hidden_dim, obs_dim)

        if self.cfg.use_branch_readout:
            self.readout = BranchReadout(hidden_dim, n_classes=n_classes)
        else:
            self.readout = SimpleReadout(hidden_dim, n_classes=n_classes)

        self.pos_head = nn.Linear(hidden_dim, track_len)
        self.phase_head = nn.Linear(hidden_dim, 6)

        self.lambda_m = 0.92
        self.alpha = 0.32

    def forward(
        self,
        x,
        a,
        cueA_zones=None,
        cueB_zones=None,
        decision_positions=None,
        return_details=False,
        skip_readout=False,
    ):
        B, T, _ = x.shape
        g = torch.zeros(B, self.hidden_dim, device=x.device)
        M = torch.zeros(B, self.hidden_dim, self.hidden_dim, device=x.device)

        pred_loss = 0.0
        mem_list, xhat_list, g_list, pos_logits_list, phase_logits_list = [], [], [], [], []

        for t in range(T - 1):
            x_t = x[:, t]
            a_t = a[:, t]

            x_emb = torch.tanh(self.fc_x(x_t))
            a_emb = self.fc_a(self.action_emb(a_t))
            g_pred = torch.tanh(self.fc_g(g) + a_emb)

            p_t = attractor(M, x_emb, steps=5)
            g = g_pred + self.alpha * p_t

            p_hat = torch.tanh(0.5 * g + torch.bmm(M, g.unsqueeze(2)).squeeze(2))
            p_out = p_hat

            x_hat = self.decoder(p_out)
            pos_logits = self.pos_head(p_out)
            phase_logits = self.phase_head(p_out)

            pred_loss = pred_loss + ((x_hat - x[:, t + 1]) ** 2).mean()

            outer = torch.bmm(p_t.unsqueeze(2), p_t.unsqueeze(1))
            M = self.lambda_m * M + outer

            mem_list.append(p_out)
            xhat_list.append(x_hat)
            g_list.append(g)
            pos_logits_list.append(pos_logits)
            phase_logits_list.append(phase_logits)

        mem = torch.stack(mem_list, dim=1)
        xhat = torch.stack(xhat_list, dim=1)
        g_hist = torch.stack(g_list, dim=1)
        pos_logits_hist = torch.stack(pos_logits_list, dim=1)
        phase_logits_hist = torch.stack(phase_logits_list, dim=1)

        if skip_readout:
            logits = None
            readout_out = None
        elif return_details and isinstance(self.readout, BranchReadout):
            readout_out = self.readout(mem, cueA_zones, cueB_zones, decision_positions, return_details=True)
            logits = readout_out["logits"]
        else:
            logits = self.readout(mem, cueA_zones, cueB_zones, decision_positions)
            readout_out = None

        if return_details:
            out = {
                "pred_loss": pred_loss,
                "logits": logits,
                "mem": mem,
                "xhat": xhat,
                "g_hist": g_hist,
                "pos_logits_hist": pos_logits_hist,
                "phase_logits_hist": phase_logits_hist,
            }
            if readout_out is not None:
                out["cueA_attn"] = readout_out["cueA_attn"]
                out["cueB_attn"] = readout_out["cueB_attn"]
                out["cueA_repr"] = readout_out["cueA_repr"]
                out["cueB_repr"] = readout_out["cueB_repr"]
                out["decision_repr"] = readout_out["decision_repr"]
            return out
        return pred_loss, logits


class HDGM(nn.Module):
    """Hippocampal Dynamic Gating Model used in the paper experiments."""
    def __init__(
        self,
        obs_dim,
        hidden_dim,
        track_len,
        action_dim=1,
        action_emb_dim=32,
        n_classes=4,
        fb_dim=None,
        c1=0.2,
        c2=1.0,
        beta=0.1,
        cfg=None
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.track_len = track_len
        self.fb_dim = hidden_dim if fb_dim is None else fb_dim
        self.cfg = cfg if cfg is not None else ExperimentConfig()

        self.fc_x = nn.Linear(obs_dim, hidden_dim)
        self.action_emb = nn.Embedding(action_dim, action_emb_dim)
        self.fc_a = nn.Linear(action_emb_dim, hidden_dim)
        self.fc_g = nn.Linear(hidden_dim, hidden_dim)

        self.decoder = nn.Linear(hidden_dim, obs_dim)

        self.gate_ec = nn.Linear(obs_dim + hidden_dim + self.fb_dim, obs_dim)
        self.gate_ca3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_ca1 = nn.Linear(hidden_dim * 2 + self.fb_dim, hidden_dim)
        self.fb_proj = nn.Linear(hidden_dim, self.fb_dim)

        if self.cfg.use_branch_readout:
            self.readout = BranchReadout(hidden_dim, n_classes=n_classes)
        else:
            self.readout = SimpleReadout(hidden_dim, n_classes=n_classes)

        self.pos_head = nn.Linear(hidden_dim, track_len)
        self.phase_head = nn.Linear(hidden_dim, 6)

        if self.cfg.use_cueA_aux_head:
            self.cueA_aux_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 2)
            )
        else:
            self.cueA_aux_head = None

        if self.cfg.use_cueB_aux_head:
            self.cueB_aux_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 2)
            )
        else:
            self.cueB_aux_head = None

        self.lambda_m = 0.92
        self.alpha = 0.32
        self.rho = 0.84

        self.c1 = c1
        self.c2 = c2
        self.beta = beta

    def forward(
        self,
        x,
        a,
        cueA_zones=None,
        cueB_zones=None,
        decision_positions=None,
        return_details=False,
        skip_readout=False,
        ca1_schedule=None,
    ):
        B, T, _ = x.shape
        g = torch.zeros(B, self.hidden_dim, device=x.device)
        M = torch.zeros(B, self.hidden_dim, self.hidden_dim, device=x.device)
        v = torch.zeros(B, self.fb_dim, device=x.device)

        pred_loss = 0.0
        mem_list, xhat_list, g_list = [], [], []
        ec_list, ca3_list, ca1_list, learned_ca1_list = [], [], [], []
        write_strength_list, v_list = [], []
        pos_logits_list, phase_logits_list = [], []
        phat_list = []

        if ca1_schedule is not None:
            expected = (B, T - 1, self.hidden_dim)
            if tuple(ca1_schedule.shape) != expected:
                raise ValueError(
                    f"ca1_schedule must have shape {expected}, got "
                    f"{tuple(ca1_schedule.shape)}"
                )
            ca1_schedule = ca1_schedule.to(device=x.device, dtype=x.dtype)

        for t in range(T - 1):
            x_t = x[:, t]
            a_t = a[:, t]

            gate_ec = torch.sigmoid(self.gate_ec(torch.cat([x_t, g, v], dim=1)))
            x_mod = gate_ec * x_t
            x_emb = torch.tanh(self.fc_x(x_mod))

            a_emb = self.fc_a(self.action_emb(a_t))
            g_pred = torch.tanh(self.fc_g(g) + a_emb)

            h = x_emb
            ca3_steps = []
            for _ in range(5):
                gate_ca3 = torch.sigmoid(self.gate_ca3(torch.cat([h, g], dim=1)))
                recur = torch.tanh(0.5 * h + torch.bmm(M, h.unsqueeze(2)).squeeze(2))
                h = (1.0 - gate_ca3) * h + gate_ca3 * recur
                ca3_steps.append(gate_ca3)

            p_t = h
            g = g_pred + self.alpha * p_t

            p_hat = torch.tanh(0.5 * g + torch.bmm(M, g.unsqueeze(2)).squeeze(2))
            learned_gain_ca1 = torch.sigmoid(
                self.gate_ca1(torch.cat([g, p_hat, v], dim=1))
            )
            gain_ca1 = (
                learned_gain_ca1
                if ca1_schedule is None
                else ca1_schedule[:, t]
            )
            p_out = torch.relu(p_hat * (self.c1 + self.c2 * gain_ca1) - self.beta)

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
            learned_ca1_list.append(learned_gain_ca1)
            write_strength_list.append(write_strength.squeeze(1))
            v_list.append(v)
            pos_logits_list.append(pos_logits)
            phase_logits_list.append(phase_logits)
            phat_list.append(p_hat)

        mem = torch.stack(mem_list, dim=1)
        xhat = torch.stack(xhat_list, dim=1)
        g_hist = torch.stack(g_list, dim=1)
        ec_hist = torch.stack(ec_list, dim=1)
        ca3_hist = torch.stack(ca3_list, dim=1)
        ca1_hist = torch.stack(ca1_list, dim=1)
        learned_ca1_hist = torch.stack(learned_ca1_list, dim=1)
        write_hist = torch.stack(write_strength_list, dim=1)
        v_hist = torch.stack(v_list, dim=1)
        phat_hist = torch.stack(phat_list, dim=1)
        pos_logits_hist = torch.stack(pos_logits_list, dim=1)
        phase_logits_hist = torch.stack(phase_logits_list, dim=1)

        if skip_readout:
            logits = None
            readout_out = None
        elif return_details and isinstance(self.readout, BranchReadout):
            readout_out = self.readout(mem, cueA_zones, cueB_zones, decision_positions, return_details=True)
            logits = readout_out["logits"]
        else:
            logits = self.readout(mem, cueA_zones, cueB_zones, decision_positions)
            readout_out = None

        cueA_aux_logits = None
        if cueA_zones is not None and self.cueA_aux_head is not None:
            cueA_reprs = []
            for b in range(B):
                a0, a1 = cueA_zones[b]
                cueA_reprs.append(mem[b, a0:a1, :].mean(dim=0))
            cueA_reprs = torch.stack(cueA_reprs, dim=0)
            cueA_aux_logits = self.cueA_aux_head(cueA_reprs)

        cueB_aux_logits = None
        if cueB_zones is not None and self.cueB_aux_head is not None:
            cueB_reprs = []
            for b in range(B):
                b0, b1 = cueB_zones[b]
                cueB_reprs.append(mem[b, b0:b1, :].mean(dim=0))
            cueB_reprs = torch.stack(cueB_reprs, dim=0)
            cueB_aux_logits = self.cueB_aux_head(cueB_reprs)

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
                "learned_ca1_hist": learned_ca1_hist,
                "write_hist": write_hist,
                "v_hist": v_hist,
                "phat_hist": phat_hist,
                "pos_logits_hist": pos_logits_hist,
                "phase_logits_hist": phase_logits_hist,
                "cueA_aux_logits": cueA_aux_logits,
                "cueB_aux_logits": cueB_aux_logits,
            }
            if readout_out is not None:
                out["cueA_attn"] = readout_out["cueA_attn"]
                out["cueB_attn"] = readout_out["cueB_attn"]
                out["cueA_repr"] = readout_out["cueA_repr"]
                out["cueB_repr"] = readout_out["cueB_repr"]
                out["decision_repr"] = readout_out["decision_repr"]
            return out
        return pred_loss, logits


# =========================================================
# 6. Basic metric helpers
# =========================================================
def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return float((preds == labels).float().mean().item())


def compute_per_class_accuracy(logits, labels, n_classes=4):
    preds = torch.argmax(logits, dim=1)
    res = {}
    for c in range(n_classes):
        mask = (labels == c)
        if mask.sum().item() == 0:
            res[c] = None
        else:
            res[c] = float((preds[mask] == labels[mask]).float().mean().item())
    return res


def compute_confusion_matrix(logits, labels, n_classes=4):
    preds = torch.argmax(logits, dim=1)
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for y, p in zip(labels.detach().cpu(), preds.detach().cpu()):
        cm[int(y), int(p)] += 1
    return cm.numpy()


def compute_similarity(mem):
    """Mean off-diagonal cosine similarity between state vectors.

    The thesis version returned ``mean(similarity_matrix) - 1``, which does
    not remove the diagonal correctly and makes the reported value hard to
    interpret. This implementation excludes all self-pairs explicitly while
    avoiding construction of a potentially very large N x N matrix.
    """
    if mem.ndim == 3:
        m = mem.reshape(-1, mem.shape[-1])
    else:
        m = mem
    if m.shape[0] < 2:
        return float("nan")
    norm = np.linalg.norm(m, axis=1, keepdims=True) + 1e-6
    m = m / norm
    n = m.shape[0]
    summed = m.sum(axis=0)
    off_diagonal_sum = float(np.dot(summed, summed) - np.sum(m * m))
    return off_diagonal_sum / float(n * (n - 1))


def compute_sparsity(mem, eps=1e-3):
    return float(np.mean(np.abs(mem) < eps))


def compute_temporal_state_change(mem):
    """Mean absolute one-step state change; lower means more stable."""
    if mem.ndim != 3:
        raise ValueError("compute_temporal_state_change expects [B, T, H]")
    return float(np.mean(np.abs(np.diff(mem, axis=1))))


def compute_stability(mem):
    """Backward-compatible alias; prefer ``compute_temporal_state_change``."""
    return compute_temporal_state_change(mem)


# =========================================================
# 7. Losses / analysis helpers
# =========================================================
def compute_cueB_margin_loss(write_hist, phase, margin=0.05):
    phase_trunc = phase[:, :-1].to(write_hist.device)
    cueB_mask = phase_trunc == 3
    baseline_mask = phase_trunc == 0
    delay2_mask = phase_trunc == 4

    losses = []
    for b in range(write_hist.shape[0]):
        wb = write_hist[b]
        cueB_vals = wb[cueB_mask[b]]
        baseline_vals = wb[baseline_mask[b]]
        delay2_vals = wb[delay2_mask[b]]

        if cueB_vals.numel() > 0 and baseline_vals.numel() > 0:
            losses.append(torch.relu(margin - (cueB_vals.mean() - baseline_vals.mean())))
        if cueB_vals.numel() > 0 and delay2_vals.numel() > 0:
            losses.append(torch.relu(margin - (cueB_vals.mean() - delay2_vals.mean())))

    if len(losses) == 0:
        return torch.tensor(0.0, device=write_hist.device)
    return torch.stack(losses).mean()


def compute_ec_phase_ranking_loss(write_hist, phase, margin=0.02):
    phase_trunc = phase[:, :-1].to(write_hist.device)
    losses = []

    for b in range(write_hist.shape[0]):
        wb = write_hist[b]
        ph = phase_trunc[b]

        cueA_vals = wb[ph == 1]
        cueB_vals = wb[ph == 3]
        baseline_vals = wb[ph == 0]
        delay1_vals = wb[ph == 2]
        delay2_vals = wb[ph == 4]
        decision_vals = wb[ph == 5]

        if cueA_vals.numel() > 0 and baseline_vals.numel() > 0:
            losses.append(torch.relu(margin - (cueA_vals.mean() - baseline_vals.mean())))
        if cueB_vals.numel() > 0 and baseline_vals.numel() > 0:
            losses.append(torch.relu(margin - (cueB_vals.mean() - baseline_vals.mean())))
        if cueA_vals.numel() > 0 and delay1_vals.numel() > 0:
            losses.append(torch.relu(margin - (cueA_vals.mean() - delay1_vals.mean())))
        if cueB_vals.numel() > 0 and delay2_vals.numel() > 0:
            losses.append(torch.relu(margin - (cueB_vals.mean() - delay2_vals.mean())))
        if cueA_vals.numel() > 0 and decision_vals.numel() > 0:
            losses.append(torch.relu(margin - (cueA_vals.mean() - decision_vals.mean())))
        if cueB_vals.numel() > 0 and decision_vals.numel() > 0:
            losses.append(torch.relu(margin - (cueB_vals.mean() - decision_vals.mean())))

    if len(losses) == 0:
        return torch.tensor(0.0, device=write_hist.device)

    return torch.stack(losses).mean()


def phase_gate_statistics(write_signal, phases):
    phase_name_map = {
        0: "baseline",
        1: "cueA",
        2: "delay1",
        3: "cueB",
        4: "delay2",
        5: "decision"
    }
    stats = {}
    for idx, name in phase_name_map.items():
        mask = phases == idx
        stats[f"{name}_mean"] = float(write_signal[mask].mean()) if mask.sum() > 0 else 0.0
    return stats


# =========================================================
# 8. Curriculum + batch helpers
# =========================================================
def sample_safe_delay_overrides(env, batch_size, mode="normal"):
    max_a0 = env.cueA_start_range[1]
    cue_w = env.cue_width
    max_total_delay = env.track_len - 1 - max_a0 - 2 * cue_w

    if max_total_delay < 4:
        raise ValueError("track_len 太短，无法容纳当前 cue 设置和 delay 设置")

    delay1s = []
    delay2s = []

    for _ in range(batch_size):
        if mode == "short":
            d1_low = max(2, env.delay1_range[0] - 4)
            d1_high = max(d1_low + 1, min(env.delay1_range[0] + 2, max_total_delay - 2) + 1)
            d1 = np.random.randint(d1_low, d1_high)

            remaining = max_total_delay - d1
            d2_low = max(2, env.delay2_range[0] - 5)
            d2_high = min(env.delay2_range[0] + 2, remaining) + 1
            d2 = max(2, remaining) if d2_high <= d2_low else np.random.randint(d2_low, d2_high)

        elif mode == "long":
            d1_low = min(env.delay1_range[1], max_total_delay - 2)
            d1_low = max(2, d1_low)
            d1_high = min(env.delay1_range[1] + 8, max_total_delay - 2) + 1
            d1 = d1_low if d1_high <= d1_low else np.random.randint(d1_low, d1_high)

            remaining = max_total_delay - d1
            d2_low = min(env.delay2_range[1], remaining)
            d2_low = max(2, d2_low)
            d2_high = min(env.delay2_range[1] + 10, remaining) + 1
            d2 = d2_low if d2_high <= d2_low else np.random.randint(d2_low, d2_high)

        else:
            d1_low = env.delay1_range[0]
            d1_high = min(env.delay1_range[1], max_total_delay - 2) + 1
            d1 = np.random.randint(d1_low, d1_high)

            remaining = max_total_delay - d1
            d2_low = env.delay2_range[0]
            d2_high = min(env.delay2_range[1], remaining) + 1
            d2 = max(2, remaining) if d2_high <= d2_low else np.random.randint(d2_low, d2_high)

        delay1s.append(int(d1))
        delay2s.append(int(d2))

    return np.array(delay1s), np.array(delay2s)


def build_balanced_batch(env, batch_size=32, mask_decision_obs=False, delay_mode="normal"):
    assert batch_size % 4 == 0
    per_class = batch_size // 4

    forced_cueAs = []
    forced_cueBs = []
    for cueA in [0, 1]:
        for cueB in [0, 1]:
            forced_cueAs.extend([cueA] * per_class)
            forced_cueBs.extend([cueB] * per_class)

    idx = np.random.permutation(batch_size)
    forced_cueAs = np.array(forced_cueAs)[idx]
    forced_cueBs = np.array(forced_cueBs)[idx]

    delay1_overrides, delay2_overrides = sample_safe_delay_overrides(env, batch_size=batch_size, mode=delay_mode)

    return env.sample_batch(
        batch_size=batch_size,
        forced_cueAs=forced_cueAs,
        forced_cueBs=forced_cueBs,
        delay1_overrides=delay1_overrides,
        delay2_overrides=delay2_overrides,
        mask_decision_obs=mask_decision_obs
    )


def build_curriculum_batch(env, batch_size=32, epoch=0, total_epochs=500):
    progress = epoch / max(1, total_epochs - 1)
    r = np.random.rand()

    if progress < 0.33:
        mode = "short" if r < 0.45 else ("normal" if r < 0.90 else "long")
    elif progress < 0.66:
        mode = "short" if r < 0.25 else ("normal" if r < 0.70 else "long")
    else:
        mode = "short" if r < 0.15 else ("normal" if r < 0.45 else "long")

    batch = build_balanced_batch(env, batch_size=batch_size, delay_mode=mode)
    return batch, mode


# =========================================================
# 9. Fixed validation/test helpers
# =========================================================
def clone_env_for_split(env, seed):
    """Clone task semantics while giving the split an independent RNG stream."""
    split_env = SharedPathFourContextEnv(
        track_len=env.track_len,
        obs_dim=env.obs_dim,
        cue_width=env.cue_width,
        cueA_start_range=env.cueA_start_range,
        delay1_range=env.delay1_range,
        delay2_range=env.delay2_range,
        n_cues=env.n_cues,
        alias_prob=env.alias_prob,
        noise_std=env.noise_std,
        seed=seed,
    )
    # Validation/test must use the same learned environment semantics (position
    # and cue dictionaries), not an unseen randomly remapped task.
    split_env.pos_map = env.pos_map.copy()
    split_env.cueA_map = env.cueA_map.copy()
    split_env.cueB_map = env.cueB_map.copy()
    split_env.rng = np.random.RandomState(seed)
    return split_env


def build_fixed_batch_bank(
    env,
    n_batches,
    batch_size,
    delay_mode,
    seed,
    mask_decision_obs=False,
):
    """Generate a reusable held-out bank without changing training RNG state."""
    split_env = clone_env_for_split(env, seed=seed)
    global_np_state = np.random.get_state()
    try:
        # Delay sampling and class-order shuffling in the legacy batch builder
        # use NumPy's global RNG. Preserve and restore it so evaluation cannot
        # alter the subsequent training trajectory.
        np.random.seed(seed + 7919)
        bank = [
            build_balanced_batch(
                split_env,
                batch_size=batch_size,
                mask_decision_obs=mask_decision_obs,
                delay_mode=delay_mode,
            )
            for _ in range(n_batches)
        ]
    finally:
        np.random.set_state(global_np_state)
    return bank


def evaluate_model_on_batch(model, batch):
    x, a, pos, phase, labels, cueA, cueB, cueA_zones, cueB_zones, decision_positions, delay1s, delay2s = batch
    x_device = x.to(device)
    a_device = a.to(device)
    labels_device = labels.to(device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(
            x_device,
            a_device,
            cueA_zones=cueA_zones,
            cueB_zones=cueB_zones,
            decision_positions=decision_positions,
            return_details=True,
        )
    model.train(was_training)

    return {
        "acc": compute_accuracy(out["logits"], labels_device),
        "per_class_acc": compute_per_class_accuracy(out["logits"], labels_device, n_classes=4),
        "cm": compute_confusion_matrix(out["logits"], labels_device, n_classes=4),
        "labels": labels_device,
        "out": out,
        "x": x_device,
        "a": a_device,
        "pos": pos,
        "phase": phase,
        "cueA": cueA,
        "cueB": cueB,
        "cueA_zones": cueA_zones,
        "cueB_zones": cueB_zones,
        "decision_positions": decision_positions,
        "delay1s": delay1s,
        "delay2s": delay2s,
    }


def evaluate_model_on_bank(model, bank):
    accs = []
    cms = np.zeros((4, 4), dtype=np.int64)
    class_correct = np.zeros(4, dtype=np.int64)
    class_total = np.zeros(4, dtype=np.int64)

    for batch in bank:
        ret = evaluate_model_on_batch(model, batch)
        accs.append(ret["acc"])
        cms += ret["cm"]
        class_correct += np.diag(ret["cm"])
        class_total += ret["cm"].sum(axis=1)

    class_acc = {
        c: None if class_total[c] == 0 else float(class_correct[c] / class_total[c])
        for c in range(4)
    }
    return {
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "class_acc": class_acc,
        "cm": cms,
    }


def evaluate_model_on_validation(model, bank):
    return evaluate_model_on_bank(model, bank)["mean_acc"]


# =========================================================
# 10. Training
# =========================================================
def train_models(seed=42, cfg=None):
    cfg = cfg if cfg is not None else ExperimentConfig()

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = SharedPathFourContextEnv(
        track_len=cfg.track_len,
        obs_dim=cfg.obs_dim,
        cue_width=cfg.cue_width,
        cueA_start_range=cfg.cueA_start_range,
        delay1_range=cfg.delay1_range,
        delay2_range=cfg.delay2_range,
        n_cues=cfg.n_cues,
        alias_prob=cfg.alias_prob,
        noise_std=cfg.noise_std,
        seed=seed
    )

    tem = TEMLite(
        obs_dim=cfg.obs_dim,
        hidden_dim=96,
        track_len=env.track_len,
        action_dim=env.action_dim,
        n_classes=4,
        cfg=cfg
    ).to(device)

    gate = HDGM(
        obs_dim=cfg.obs_dim,
        hidden_dim=96,
        track_len=env.track_len,
        action_dim=env.action_dim,
        n_classes=4,
        cfg=cfg
    ).to(device)

    opt_tem = optim.Adam(tem.parameters(), lr=cfg.lr)
    opt_gate = optim.Adam(gate.parameters(), lr=cfg.lr)

    split_base = int(cfg.split_seed_version * 100_000)
    split_seeds = {
        "val_normal": int(seed + split_base + 10_001),
        "val_long": int(seed + split_base + 10_002),
        "test_normal": int(seed + split_base + 20_001),
        "test_long": int(seed + split_base + 20_002),
        "analysis": int(seed + split_base + 30_001),
        "probe_train": int(seed + split_base + 40_001),
        "probe_val": int(seed + split_base + 40_002),
        "probe_test": int(seed + split_base + 40_003),
    }
    val_banks = {
        "normal": build_fixed_batch_bank(
            env, cfg.val_batches, cfg.val_batch_size, "normal", split_seeds["val_normal"]
        ),
        "long": build_fixed_batch_bank(
            env, cfg.val_batches, cfg.val_batch_size, "long", split_seeds["val_long"]
        ),
    }
    test_banks = {
        "normal": build_fixed_batch_bank(
            env, cfg.test_batches, cfg.test_batch_size, "normal", split_seeds["test_normal"]
        ),
        "long": build_fixed_batch_bank(
            env, cfg.test_batches, cfg.test_batch_size, "long", split_seeds["test_long"]
        ),
    }
    analysis_bank = build_fixed_batch_bank(
        env,
        cfg.analysis_batches,
        cfg.analysis_batch_size,
        "normal",
        split_seeds["analysis"],
    )
    probe_banks = {
        "train": build_fixed_batch_bank(
            env,
            cfg.probe_train_batches,
            cfg.probe_batch_size,
            "normal",
            split_seeds["probe_train"],
        ),
        "val": build_fixed_batch_bank(
            env,
            cfg.probe_val_batches,
            cfg.probe_batch_size,
            "normal",
            split_seeds["probe_val"],
        ),
        "test": build_fixed_batch_bank(
            env,
            cfg.probe_test_batches,
            cfg.probe_batch_size,
            "normal",
            split_seeds["probe_test"],
        ),
    }

    epochs = cfg.epochs
    batch_size = cfg.batch_size
    sampled_delay_modes = []

    history = {
        "epoch": [],
        "val_epoch": [],
        "delay_mode": [],
        "tem_train_acc": [],
        "gate_train_acc": [],
        "tem_train_loss": [],
        "gate_train_loss": [],
        "tem_val_normal": [],
        "tem_val_long": [],
        "tem_val_score": [],
        "gate_val_normal": [],
        "gate_val_long": [],
        "gate_val_score": [],
    }

    best_tem_state = None
    best_gate_state = None
    best_tem_score = -1.0
    best_gate_score = -1.0
    best_tem_epoch = -1
    best_gate_epoch = -1

    ce_loss_plain = nn.CrossEntropyLoss()

    print(f"===== TRAINING START [{cfg.name}] =====")
    for epoch in range(epochs):
        batch, delay_mode = build_curriculum_batch(env, batch_size=batch_size, epoch=epoch, total_epochs=epochs)
        sampled_delay_modes.append(delay_mode)

        x, a, pos, phase, labels, cueA, cueB, cueA_zones, cueB_zones, decision_positions, delay1s, delay2s = batch
        x = x.to(device)
        a = a.to(device)
        phase = phase.to(device)
        labels = labels.to(device)

        # ----- TEM -----
        opt_tem.zero_grad()
        out_t = tem(
            x, a,
            cueA_zones=cueA_zones,
            cueB_zones=cueB_zones,
            decision_positions=decision_positions,
            return_details=True
        )
        cls_loss_t = ce_loss_plain(out_t["logits"], labels)
        total_loss_t = out_t["pred_loss"] + 0.7 * cls_loss_t
        total_loss_t.backward()
        torch.nn.utils.clip_grad_norm_(tem.parameters(), 1.0)
        opt_tem.step()

        # ----- Gate -----
        opt_gate.zero_grad()
        out_g = gate(
            x, a,
            cueA_zones=cueA_zones,
            cueB_zones=cueB_zones,
            decision_positions=decision_positions,
            return_details=True
        )
        cls_loss_g = ce_loss_plain(out_g["logits"], labels)

        cueB_margin_loss_g = compute_cueB_margin_loss(out_g["write_hist"], phase, margin=0.05) \
            if cfg.use_cueB_margin_loss else torch.tensor(0.0, device=device)

        ec_phase_rank_loss_g = compute_ec_phase_ranking_loss(out_g["write_hist"], phase, margin=0.02) \
            if cfg.use_ec_phase_rank_loss else torch.tensor(0.0, device=device)

        total_loss_g = (
            out_g["pred_loss"]
            + 0.7 * cls_loss_g
            + cfg.gate_cueB_margin_weight * cueB_margin_loss_g
            + cfg.gate_ec_phase_rank_weight * ec_phase_rank_loss_g
        )
        total_loss_g.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
        opt_gate.step()

        history["epoch"].append(epoch + 1)
        history["delay_mode"].append(delay_mode)
        history["tem_train_acc"].append(compute_accuracy(out_t["logits"], labels))
        history["gate_train_acc"].append(compute_accuracy(out_g["logits"], labels))
        history["tem_train_loss"].append(float(total_loss_t.item()))
        history["gate_train_loss"].append(float(total_loss_g.item()))

        should_validate = (
            epoch == 0
            or (epoch + 1) % cfg.val_interval == 0
            or (epoch + 1) == epochs
        )
        if should_validate:
            val_tem_normal = evaluate_model_on_validation(tem, val_banks["normal"])
            val_tem_long = evaluate_model_on_validation(tem, val_banks["long"])
            val_tem_score = cfg.val_score_alpha * val_tem_normal + (1.0 - cfg.val_score_alpha) * val_tem_long

            val_gate_normal = evaluate_model_on_validation(gate, val_banks["normal"])
            val_gate_long = evaluate_model_on_validation(gate, val_banks["long"])
            val_gate_score = cfg.val_score_alpha * val_gate_normal + (1.0 - cfg.val_score_alpha) * val_gate_long

            history["val_epoch"].append(epoch + 1)
            history["tem_val_normal"].append(float(val_tem_normal))
            history["tem_val_long"].append(float(val_tem_long))
            history["tem_val_score"].append(float(val_tem_score))
            history["gate_val_normal"].append(float(val_gate_normal))
            history["gate_val_long"].append(float(val_gate_long))
            history["gate_val_score"].append(float(val_gate_score))

            if val_tem_score > best_tem_score:
                best_tem_score = val_tem_score
                best_tem_epoch = epoch + 1
                best_tem_state = copy.deepcopy(tem.state_dict())

            if val_gate_score > best_gate_score:
                best_gate_score = val_gate_score
                best_gate_epoch = epoch + 1
                best_gate_state = copy.deepcopy(gate.state_dict())

        if (epoch + 1) % cfg.print_interval == 0:
            print(
                f"Epoch {epoch+1:03d} | mode={delay_mode:<6} | "
                f"TEM train acc={compute_accuracy(out_t['logits'], labels):.3f} | "
                f"HDGM train acc={compute_accuracy(out_g['logits'], labels):.3f} | "
                f"TEM best val={best_tem_score:.3f} | HDGM best val={best_gate_score:.3f}"
            )

    tem.load_state_dict(best_tem_state)
    gate.load_state_dict(best_gate_state)

    print(f"\n===== TRAINING FINISHED [{cfg.name}] =====")
    print(f"Best TEM val score  : {best_tem_score:.4f}")
    print(f"Best HDGM val score : {best_gate_score:.4f}")
    print(f"Best epochs         : TEM={best_tem_epoch} | HDGM={best_gate_epoch}")

    short_count = sampled_delay_modes.count("short")
    normal_count = sampled_delay_modes.count("normal")
    long_count = sampled_delay_modes.count("long")
    print("\n===== Curriculum Summary =====")
    print(f"short  batches: {short_count}")
    print(f"normal batches: {normal_count}")
    print(f"long   batches: {long_count}")

    return {
        "env": env,
        "tem": tem,
        "gate": gate,
        "cfg": cfg,
        "sampled_delay_modes": sampled_delay_modes,
        "history": history,
        "val_banks": val_banks,
        "test_banks": test_banks,
        "analysis_bank": analysis_bank,
        "probe_banks": probe_banks,
        "split_seeds": split_seeds,
        "best_tem_score": float(best_tem_score),
        "best_gate_score": float(best_gate_score),
        "best_tem_epoch": int(best_tem_epoch),
        "best_gate_epoch": int(best_gate_epoch),
    }


# =========================================================
# 11. Evaluation
# =========================================================
def evaluate_model_once(model, env, batch_size=64, mask_decision_obs=False, delay_mode="normal"):
    x, a, pos, phase, labels, cueA, cueB, cueA_zones, cueB_zones, decision_positions, delay1s, delay2s = build_balanced_batch(
        env, batch_size=batch_size, mask_decision_obs=mask_decision_obs, delay_mode=delay_mode
    )

    x = x.to(device)
    a = a.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        out = model(
            x, a,
            cueA_zones=cueA_zones,
            cueB_zones=cueB_zones,
            decision_positions=decision_positions,
            return_details=True
        )

    acc = compute_accuracy(out["logits"], labels)
    per_class_acc = compute_per_class_accuracy(out["logits"], labels, n_classes=4)
    cm = compute_confusion_matrix(out["logits"], labels, n_classes=4)

    return {
        "acc": acc,
        "per_class_acc": per_class_acc,
        "cm": cm,
        "labels": labels,
        "out": out,
        "x": x,
        "a": a,
        "cueA_zones": cueA_zones,
        "cueB_zones": cueB_zones,
        "decision_positions": decision_positions,
        "phase": phase,
    }


def evaluate_model_multi_batches(model, env, n_batches=50, batch_size=64, mask_decision_obs=False, delay_mode="normal"):
    accs = []
    cms = np.zeros((4, 4), dtype=np.int64)
    class_acc_sum = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    class_acc_cnt = {0: 0, 1: 0, 2: 0, 3: 0}

    for _ in range(n_batches):
        ret = evaluate_model_once(
            model,
            env,
            batch_size=batch_size,
            mask_decision_obs=mask_decision_obs,
            delay_mode=delay_mode
        )
        accs.append(ret["acc"])
        cms += ret["cm"]

        for c, v in ret["per_class_acc"].items():
            if v is not None:
                class_acc_sum[c] += v
                class_acc_cnt[c] += 1

    class_acc = {}
    for c in range(4):
        class_acc[c] = None if class_acc_cnt[c] == 0 else class_acc_sum[c] / class_acc_cnt[c]

    return {
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "class_acc": class_acc,
        "cm": cms
    }


# =========================================================
# 11b. Frozen decision-state linear probes
# =========================================================
class FrozenDecisionProbe(nn.Module):
    """Strictly linear probes for cue A, cue B, and their four-class pair."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.cueA_head = nn.Linear(hidden_dim, 2)
        self.cueB_head = nn.Linear(hidden_dim, 2)
        self.class_head = nn.Linear(hidden_dim, 4)

    def forward(self, states):
        return {
            "cueA": self.cueA_head(states),
            "cueB": self.cueB_head(states),
            "class": self.class_head(states),
        }


def extract_frozen_decision_features(model, bank):
    """Extract the state immediately before decision without cue-window input."""
    features, cue_as, cue_bs, labels = [], [], [], []
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for batch in bank:
            x, a, pos, phase, y, cueA, cueB, cueA_zones, cueB_zones, decision_positions, delay1s, delay2s = batch
            x_device = x.to(device)
            a_device = a.to(device)

            # Crucially, no cueA_zones/cueB_zones/decision_positions are passed
            # into the model. The terminal event-aligned decoder is skipped.
            out = model(
                x_device,
                a_device,
                return_details=True,
                skip_readout=True,
            )
            mem = out["mem"]
            indices = torch.tensor(
                [max(0, min(int(d) - 1, mem.shape[1] - 1)) for d in decision_positions],
                dtype=torch.long,
                device=device,
            )
            batch_indices = torch.arange(mem.shape[0], device=device)
            decision_state = mem[batch_indices, indices]

            features.append(decision_state.detach().cpu())
            cue_as.append(cueA.detach().cpu())
            cue_bs.append(cueB.detach().cpu())
            labels.append(y.detach().cpu())

    model.train(was_training)
    return {
        "x": torch.cat(features, dim=0),
        "cueA": torch.cat(cue_as, dim=0),
        "cueB": torch.cat(cue_bs, dim=0),
        "class": torch.cat(labels, dim=0),
    }


def probe_accuracy(logits, targets):
    return float((logits.argmax(dim=1) == targets).float().mean().item())


def train_and_test_frozen_probe(model, probe_banks, cfg, probe_seed):
    train_data = extract_frozen_decision_features(model, probe_banks["train"])
    val_data = extract_frozen_decision_features(model, probe_banks["val"])
    test_data = extract_frozen_decision_features(model, probe_banks["test"])

    train_mean = train_data["x"].mean(dim=0, keepdim=True)
    train_std = train_data["x"].std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)

    def standardized(data):
        return {
            "x": ((data["x"] - train_mean) / train_std).to(device),
            "cueA": data["cueA"].to(device),
            "cueB": data["cueB"].to(device),
            "class": data["class"].to(device),
        }

    train_data = standardized(train_data)
    val_data = standardized(val_data)
    test_data = standardized(test_data)

    fork_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(probe_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(probe_seed)

        probe = FrozenDecisionProbe(train_data["x"].shape[1]).to(device)
        optimizer = optim.Adam(
            probe.parameters(),
            lr=cfg.probe_lr,
            weight_decay=cfg.probe_weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        best_state = None
        best_val_score = -1.0
        best_epoch = -1

        for probe_epoch in range(cfg.probe_epochs):
            probe.train()
            optimizer.zero_grad()
            logits = probe(train_data["x"])
            loss = (
                criterion(logits["cueA"], train_data["cueA"])
                + criterion(logits["cueB"], train_data["cueB"])
                + criterion(logits["class"], train_data["class"])
            )
            loss.backward()
            optimizer.step()

            probe.eval()
            with torch.no_grad():
                val_logits = probe(val_data["x"])
                val_score = np.mean([
                    probe_accuracy(val_logits["cueA"], val_data["cueA"]),
                    probe_accuracy(val_logits["cueB"], val_data["cueB"]),
                    probe_accuracy(val_logits["class"], val_data["class"]),
                ])
            if val_score > best_val_score:
                best_val_score = float(val_score)
                best_epoch = probe_epoch + 1
                best_state = copy.deepcopy(probe.state_dict())

        probe.load_state_dict(best_state)
        probe.eval()
        with torch.no_grad():
            test_logits = probe(test_data["x"])
            cueA_acc = probe_accuracy(test_logits["cueA"], test_data["cueA"])
            cueB_acc = probe_accuracy(test_logits["cueB"], test_data["cueB"])
            class_acc = probe_accuracy(test_logits["class"], test_data["class"])
            class_cm = compute_confusion_matrix(
                test_logits["class"],
                test_data["class"],
                n_classes=4,
            )

    return {
        "cueA_acc": float(cueA_acc),
        "cueB_acc": float(cueB_acc),
        "class_acc": float(class_acc),
        "best_val_score": float(best_val_score),
        "best_epoch": int(best_epoch),
        "class_confusion_matrix": class_cm.tolist(),
    }


def compute_frozen_decision_probe_metrics(results):
    cfg = results["cfg"]
    probe_seed = int(results["split_seeds"]["probe_train"] + 97)
    tem_probe = train_and_test_frozen_probe(
        results["tem"], results["probe_banks"], cfg, probe_seed
    )
    hdgm_probe = train_and_test_frozen_probe(
        results["gate"], results["probe_banks"], cfg, probe_seed
    )
    print(
        "Frozen decision probe | "
        f"TEM A/B/class={tem_probe['cueA_acc']:.3f}/{tem_probe['cueB_acc']:.3f}/{tem_probe['class_acc']:.3f} | "
        f"HDGM A/B/class={hdgm_probe['cueA_acc']:.3f}/{hdgm_probe['cueB_acc']:.3f}/{hdgm_probe['class_acc']:.3f}"
    )
    return {
        "tem_decision_probe_cueA": tem_probe["cueA_acc"],
        "tem_decision_probe_cueB": tem_probe["cueB_acc"],
        "tem_decision_probe_class": tem_probe["class_acc"],
        "tem_decision_probe_best_val": tem_probe["best_val_score"],
        "tem_decision_probe_best_epoch": tem_probe["best_epoch"],
        "tem_decision_probe_confusion_matrix": tem_probe["class_confusion_matrix"],
        "hdgm_decision_probe_cueA": hdgm_probe["cueA_acc"],
        "hdgm_decision_probe_cueB": hdgm_probe["cueB_acc"],
        "hdgm_decision_probe_class": hdgm_probe["class_acc"],
        "hdgm_decision_probe_best_val": hdgm_probe["best_val_score"],
        "hdgm_decision_probe_best_epoch": hdgm_probe["best_epoch"],
        "hdgm_decision_probe_confusion_matrix": hdgm_probe["class_confusion_matrix"],
    }


# =========================================================
# 12. Representation analysis
# =========================================================
def collect_representations(model, env, n_trials=300, gated=False):
    mem_all, pos_all, label_all, phase_all = [], [], [], []
    write_all = []

    for _ in range(n_trials):
        x, a, pos, phase, labels, cueA, cueB, cueA_zones, cueB_zones, decision_positions, delay1s, delay2s = env.sample_batch(batch_size=1)
        x = x.to(device)
        a = a.to(device)

        with torch.no_grad():
            out = model(
                x, a,
                cueA_zones=cueA_zones,
                cueB_zones=cueB_zones,
                decision_positions=decision_positions,
                return_details=True
            )

        mem = out["mem"][0].detach().cpu().numpy()
        mem_all.append(mem)
        pos_all.append(pos[0, :-1].numpy())
        phase_all.append(phase[0, :-1].numpy())
        label_all.append(np.full(mem.shape[0], labels.item(), dtype=np.int64))

        if gated:
            write_all.append(out["write_hist"][0].detach().cpu().numpy())

    results = {
        "mem": np.concatenate(mem_all, axis=0),
        "pos": np.concatenate(pos_all, axis=0),
        "phase": np.concatenate(phase_all, axis=0),
        "label": np.concatenate(label_all, axis=0),
    }

    if gated:
        results["write"] = np.concatenate(write_all, axis=0)

    return results


def collect_representations_on_bank(model, bank, gated=False):
    """Collect representations from exactly the same held-out trials."""
    mem_all, pos_all, label_all, phase_all = [], [], [], []
    write_all = []

    for batch in bank:
        ret = evaluate_model_on_batch(model, batch)
        mem = ret["out"]["mem"].detach().cpu().numpy()
        batch_size, time_steps, hidden_dim = mem.shape
        mem_all.append(mem.reshape(batch_size * time_steps, hidden_dim))
        pos_all.append(ret["pos"][:, :time_steps].numpy().reshape(-1))
        phase_all.append(ret["phase"][:, :time_steps].numpy().reshape(-1))
        labels_cpu = ret["labels"].detach().cpu().numpy()
        label_all.append(np.repeat(labels_cpu, time_steps))
        if gated:
            write_all.append(ret["out"]["write_hist"].detach().cpu().numpy().reshape(-1))

    collected = {
        "mem": np.concatenate(mem_all, axis=0),
        "pos": np.concatenate(pos_all, axis=0),
        "phase": np.concatenate(phase_all, axis=0),
        "label": np.concatenate(label_all, axis=0),
    }
    if gated:
        collected["write"] = np.concatenate(write_all, axis=0)
    return collected


def compute_place_fields(mem, pos, track_len, top_k=6):
    H = mem.shape[1]
    fields = np.zeros((H, track_len), dtype=np.float32)
    counts = np.zeros(track_len, dtype=np.float32)

    for i in range(len(pos)):
        p = pos[i]
        fields[:, p] += mem[i]
        counts[p] += 1

    counts[counts == 0] = 1.0
    fields = fields / counts[None, :]
    selectivity = fields.std(axis=1)
    top_idx = np.argsort(selectivity)[-top_k:]
    return fields, selectivity, top_idx


# =========================================================
# 13. Statistics helpers
# =========================================================
def summarize_scalar_dict(list_of_dicts):
    keys = list(list_of_dicts[0].keys())
    out = {}
    for k in keys:
        vals = [float(d[k]) for d in list_of_dicts]
        arr = np.array(vals, dtype=np.float64)
        out[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "values": vals
        }
    return out


def format_summary_block(title, summary_dict, ordered_keys=None):
    print(f"\n===== {title} =====")
    keys = ordered_keys if ordered_keys is not None else list(summary_dict.keys())
    for k in keys:
        print(f"{k}: {summary_dict[k]['mean']:.4f} ± {summary_dict[k]['std']:.4f}")


def paired_permutation_test(x, y, n_perm=20000, seed=0):
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    d = x - y
    obs = np.abs(d.mean())

    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(d))
        stat = np.abs((d * signs).mean())
        if stat >= obs:
            count += 1

    p = (count + 1) / (n_perm + 1)
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    cohend_z = float(d.mean() / (sd + 1e-12))

    return {
        "mean_diff": float(d.mean()),
        "p_value": float(p),
        "cohend_z": cohend_z
    }


# =========================================================
# 14. Metrics collection
# =========================================================
def compute_biological_metrics_only(results):
    env = results["env"]
    tem = results["tem"]
    gate = results["gate"]
    analysis_bank = results["analysis_bank"]

    tem_repr = collect_representations_on_bank(tem, analysis_bank, gated=False)
    gate_repr = collect_representations_on_bank(gate, analysis_bank, gated=True)

    _, tem_selectivity, _ = compute_place_fields(tem_repr["mem"], tem_repr["pos"], env.track_len, top_k=6)
    _, gate_selectivity, _ = compute_place_fields(gate_repr["mem"], gate_repr["pos"], env.track_len, top_k=6)

    phase_stats = phase_gate_statistics(gate_repr["write"], gate_repr["phase"])

    return {
        "tem_place_selectivity": float(tem_selectivity.mean()),
        "gate_place_selectivity": float(gate_selectivity.mean()),
        "gate_baseline_mean": float(phase_stats["baseline_mean"]),
        "gate_cueA_mean": float(phase_stats["cueA_mean"]),
        "gate_delay1_mean": float(phase_stats["delay1_mean"]),
        "gate_cueB_mean": float(phase_stats["cueB_mean"]),
        "gate_delay2_mean": float(phase_stats["delay2_mean"]),
        "gate_decision_mean": float(phase_stats["decision_mean"]),
    }


def compute_tem_metrics_only(results):
    tem = results["tem"]

    tem_once = evaluate_model_on_batch(tem, results["analysis_bank"][0])
    out_t = tem_once["out"]
    tem_mem = out_t["mem"].detach().cpu().numpy()

    return {
        "tem_similarity": compute_similarity(tem_mem),
        "tem_sparsity": compute_sparsity(tem_mem),
        "tem_state_change": compute_temporal_state_change(tem_mem),
    }


def compute_gate_metrics_only(results):
    gate = results["gate"]

    gate_once = evaluate_model_on_batch(gate, results["analysis_bank"][0])
    out_g = gate_once["out"]
    gate_mem = out_g["mem"].detach().cpu().numpy()
    phase_eval = gate_once["phase"].to(device)
    cueB_margin_eval = compute_cueB_margin_loss(out_g["write_hist"], phase_eval, margin=0.05)
    ec_phase_rank_eval = compute_ec_phase_ranking_loss(out_g["write_hist"], phase_eval, margin=0.02)

    return {
        "gate_similarity": compute_similarity(gate_mem),
        "gate_sparsity": compute_sparsity(gate_mem),
        "gate_state_change": compute_temporal_state_change(gate_mem),
        "cueB_margin_loss": float(cueB_margin_eval.item()),
        "ec_phase_rank_loss": float(ec_phase_rank_eval.item()),
    }


def compute_main_behavior_metrics(results, n_batches=50, batch_size=64):
    tem = results["tem"]
    gate = results["gate"]

    # Both models see byte-for-byte identical held-out batches. These banks are
    # created before training and never participate in checkpoint selection.
    tem_eval = evaluate_model_on_bank(tem, results["test_banks"]["normal"])
    gate_eval = evaluate_model_on_bank(gate, results["test_banks"]["normal"])

    tem_long = evaluate_model_on_bank(tem, results["test_banks"]["long"])
    gate_long = evaluate_model_on_bank(gate, results["test_banks"]["long"])

    return {
        "tem_event_aligned_decoding_acc": float(tem_eval["mean_acc"]),
        "hdgm_event_aligned_decoding_acc": float(gate_eval["mean_acc"]),
        # Legacy aliases retained so existing plotting notebooks do not break.
        "tem_mean_acc": float(tem_eval["mean_acc"]),
        "gate_mean_acc": float(gate_eval["mean_acc"]),

        "tem_class0": float(tem_eval["class_acc"][0]),
        "tem_class1": float(tem_eval["class_acc"][1]),
        "tem_class2": float(tem_eval["class_acc"][2]),
        "tem_class3": float(tem_eval["class_acc"][3]),

        "gate_class0": float(gate_eval["class_acc"][0]),
        "gate_class1": float(gate_eval["class_acc"][1]),
        "gate_class2": float(gate_eval["class_acc"][2]),
        "gate_class3": float(gate_eval["class_acc"][3]),

        # Long delays occur in the training curriculum, so this is performance
        # under a long-delay condition, not out-of-distribution generalization.
        "tem_long_delay_performance": float(tem_long["mean_acc"]),
        "gate_long_delay_performance": float(gate_long["mean_acc"]),

        "tem_confusion_matrix": tem_eval["cm"].tolist(),
        "gate_confusion_matrix": gate_eval["cm"].tolist(),
    }


# =========================================================
# 15. One-seed runner
# =========================================================
def run_one_seed(seed: int, cfg: ExperimentConfig, verbose: bool = True, return_results: bool = False):
    set_global_seed(seed)

    if verbose:
        print("\n" + "=" * 80)
        print(f"RUN SEED = {seed} | CONFIG = {cfg.name}")
        print("=" * 80)

    results = train_models(seed=seed, cfg=cfg)

    record = {
        "seed": int(seed),
        "config_name": cfg.name,
        "config": asdict(cfg),
        "evaluation_protocol": {
            "primary_accuracy_name": "event_aligned_decoding_accuracy",
            "cue_window_metadata_enters_dynamics": False,
            "cue_window_metadata_enters_terminal_decoder": bool(cfg.use_branch_readout),
            "test_used_for_checkpoint_selection": False,
            "models_share_identical_held_out_batches": True,
            "long_delay_is_ood_generalization": False,
            "decision_probe_freezes_tem_and_hdgm": True,
            "decision_probe_receives_cue_windows": False,
            "decision_probe_train_val_test_are_disjoint": True,
        },
        "split_seeds": results["split_seeds"],
        "best_tem_val_score": results["best_tem_score"],
        "best_gate_val_score": results["best_gate_score"],
        "best_tem_epoch": results["best_tem_epoch"],
        "best_gate_epoch": results["best_gate_epoch"],
        "training_history": results["history"],
        "curriculum_counts": {
            mode: int(results["sampled_delay_modes"].count(mode))
            for mode in ("short", "normal", "long")
        },
    }

    record.update(compute_main_behavior_metrics(results))
    record.update(compute_frozen_decision_probe_metrics(results))
    record.update(compute_biological_metrics_only(results))
    record.update(compute_tem_metrics_only(results))
    record.update(compute_gate_metrics_only(results))

    checkpoint_dir = ensure_dir(cfg.checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f"{cfg.name}_seed_{seed}.pt")
    torch.save(
        {
            "seed": int(seed),
            "config": asdict(cfg),
            "split_seeds": results["split_seeds"],
            "best_tem_epoch": results["best_tem_epoch"],
            "best_gate_epoch": results["best_gate_epoch"],
            "best_tem_val_score": results["best_tem_score"],
            "best_gate_val_score": results["best_gate_score"],
            "tem_state_dict": results["tem"].state_dict(),
            "hdgm_state_dict": results["gate"].state_dict(),
            "environment_maps": {
                "pos_map": results["env"].pos_map.copy(),
                "cueA_map": results["env"].cueA_map.copy(),
                "cueB_map": results["env"].cueB_map.copy(),
            },
            "training_history": results["history"],
        },
        checkpoint_path,
    )
    record["checkpoint_path"] = checkpoint_path

    if return_results:
        return record, results
    return record


# =========================================================
# 16. Reporting
# =========================================================
def print_significance(records):
    if len(records) < 2:
        print("\n===== Significance =====")
        print("Skipped: paired inference requires multiple independent seeds.")
        return

    tem_acc = [r["tem_event_aligned_decoding_acc"] for r in records]
    gate_acc = [r["hdgm_event_aligned_decoding_acc"] for r in records]

    tem_long = [r["tem_long_delay_performance"] for r in records]
    gate_long = [r["gate_long_delay_performance"] for r in records]

    tem_change = [r["tem_state_change"] for r in records]
    gate_change = [r["gate_state_change"] for r in records]

    sig_main = paired_permutation_test(gate_acc, tem_acc, n_perm=20000, seed=1)
    sig_long = paired_permutation_test(gate_long, tem_long, n_perm=20000, seed=2)
    sig_change = paired_permutation_test(gate_change, tem_change, n_perm=20000, seed=3)
    sig_probe = paired_permutation_test(
        [r["hdgm_decision_probe_class"] for r in records],
        [r["tem_decision_probe_class"] for r in records],
        n_perm=20000,
        seed=4,
    )

    print("\n===== Significance (Paired permutation) =====")
    print(f"Event-aligned decode | diff={sig_main['mean_diff']:.4f}, p={sig_main['p_value']:.5f}, dz={sig_main['cohend_z']:.4f}")
    print(f"Long-delay Accuracy | diff={sig_long['mean_diff']:.4f}, p={sig_long['p_value']:.5f}, dz={sig_long['cohend_z']:.4f}")
    print(f"State change (HDGM-TEM) | diff={sig_change['mean_diff']:.4f}, p={sig_change['p_value']:.5f}, dz={sig_change['cohend_z']:.4f}")
    print(f"Frozen decision probe   | diff={sig_probe['mean_diff']:.4f}, p={sig_probe['p_value']:.5f}, dz={sig_probe['cohend_z']:.4f}")


def print_multirun_summary(records, title="FOUR-CLASS PAPER MAIN SUMMARY"):
    print("\n" + "#" * 100)
    print(title)
    print("#" * 100)

    main_metrics = summarize_scalar_dict([
        {
            "tem_event_aligned_decoding_acc": r["tem_event_aligned_decoding_acc"],
            "hdgm_event_aligned_decoding_acc": r["hdgm_event_aligned_decoding_acc"],
            "tem_long_delay_performance": r["tem_long_delay_performance"],
            "gate_long_delay_performance": r["gate_long_delay_performance"],
            "tem_state_change": r["tem_state_change"],
            "gate_state_change": r["gate_state_change"],
            "tem_place_selectivity": r["tem_place_selectivity"],
            "gate_place_selectivity": r["gate_place_selectivity"],
        }
        for r in records
    ])

    format_summary_block(
        "Main Results",
        main_metrics,
        ordered_keys=[
            "tem_event_aligned_decoding_acc", "hdgm_event_aligned_decoding_acc",
            "tem_long_delay_performance", "gate_long_delay_performance",
            "tem_state_change", "gate_state_change",
            "tem_place_selectivity", "gate_place_selectivity",
        ]
    )

    probe_summary = summarize_scalar_dict([
        {
            "tem_decision_probe_cueA": r["tem_decision_probe_cueA"],
            "tem_decision_probe_cueB": r["tem_decision_probe_cueB"],
            "tem_decision_probe_class": r["tem_decision_probe_class"],
            "hdgm_decision_probe_cueA": r["hdgm_decision_probe_cueA"],
            "hdgm_decision_probe_cueB": r["hdgm_decision_probe_cueB"],
            "hdgm_decision_probe_class": r["hdgm_decision_probe_class"],
        }
        for r in records
    ])
    format_summary_block(
        "Frozen Decision-State Linear Probes",
        probe_summary,
        ordered_keys=[
            "tem_decision_probe_cueA",
            "tem_decision_probe_cueB",
            "tem_decision_probe_class",
            "hdgm_decision_probe_cueA",
            "hdgm_decision_probe_cueB",
            "hdgm_decision_probe_class",
        ],
    )

    gate_class_summary = summarize_scalar_dict([
        {
            "gate_class0": r["gate_class0"],
            "gate_class1": r["gate_class1"],
            "gate_class2": r["gate_class2"],
            "gate_class3": r["gate_class3"],
        }
        for r in records
    ])
    format_summary_block(
        "HDGM Per-class Accuracy",
        gate_class_summary,
        ordered_keys=["gate_class0", "gate_class1", "gate_class2", "gate_class3"]
    )

    phase_summary = summarize_scalar_dict([
        {
            "gate_baseline_mean": r["gate_baseline_mean"],
            "gate_cueA_mean": r["gate_cueA_mean"],
            "gate_delay1_mean": r["gate_delay1_mean"],
            "gate_cueB_mean": r["gate_cueB_mean"],
            "gate_delay2_mean": r["gate_delay2_mean"],
            "gate_decision_mean": r["gate_decision_mean"],
        }
        for r in records
    ])
    format_summary_block(
        "HDGM Write by Phase",
        phase_summary,
        ordered_keys=[
            "gate_baseline_mean",
            "gate_cueA_mean",
            "gate_delay1_mean",
            "gate_cueB_mean",
            "gate_delay2_mean",
            "gate_decision_mean",
        ]
    )

    gate_mech_summary = summarize_scalar_dict([
        {
            "cueB_margin_loss": r["cueB_margin_loss"],
            "ec_phase_rank_loss": r["ec_phase_rank_loss"],
            "gate_similarity": r["gate_similarity"],
            "gate_sparsity": r["gate_sparsity"],
        }
        for r in records
    ])
    format_summary_block(
        "HDGM Mechanism",
        gate_mech_summary,
        ordered_keys=[
            "cueB_margin_loss",
            "ec_phase_rank_loss",
            "gate_similarity",
            "gate_sparsity",
        ]
    )

    print_significance(records)

    print("\n===== Example Cumulative Confusion Matrix (last seed) =====")
    print("TEM:")
    print(np.array(records[-1]["tem_confusion_matrix"]))
    print("HDGM:")
    print(np.array(records[-1]["gate_confusion_matrix"]))


def save_records(records, save_path):
    payload = {"num_runs": len(records), "records": records}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results to: {save_path}")




# =========================================================
# 17. Visualization helpers
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def pca_project(X, n_components=2):
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:n_components].T


def plot_training_curves(history, save_path):
    epochs = np.array(history["epoch"])
    val_epochs = np.array(history["val_epoch"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(epochs, history["tem_train_loss"], label="TEM train loss")
    axes[0].plot(epochs, history["gate_train_loss"], label="Gate train loss")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["tem_train_acc"], label="TEM train acc")
    axes[1].plot(epochs, history["gate_train_acc"], label="Gate train acc")
    axes[1].set_title("Training accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    axes[2].plot(val_epochs, history["tem_val_score"], label="TEM val score")
    axes[2].plot(val_epochs, history["gate_val_score"], label="HDGM val score")
    axes[2].plot(val_epochs, history["tem_val_long"], "--", label="TEM val long")
    axes[2].plot(val_epochs, history["gate_val_long"], "--", label="HDGM val long")
    axes[2].set_title("Validation curves")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score / Accuracy")
    axes[2].legend()

    fig.suptitle("Representative training curves")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(tem_cm, gate_cm, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, cm, title in zip(axes, [tem_cm, gate_cm], ["TEM", "HDGM"]):
        im = ax.imshow(cm, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Representative confusion matrices")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gate_phase_means(record, save_path):
    phases = ["baseline", "cueA", "delay1", "cueB", "delay2", "decision"]
    vals = [
        record["gate_baseline_mean"],
        record["gate_cueA_mean"],
        record["gate_delay1_mean"],
        record["gate_cueB_mean"],
        record["gate_delay2_mean"],
        record["gate_decision_mean"],
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(phases, vals)
    ax.set_title("HDGM write strength by phase")
    ax.set_ylabel("Mean write strength")
    ax.set_xlabel("Phase")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_multiseed_main_bars(records, save_path):
    labels = ["TEM", "HDGM"]
    mean_acc = [np.mean([r["tem_event_aligned_decoding_acc"] for r in records]), np.mean([r["hdgm_event_aligned_decoding_acc"] for r in records])]
    std_acc = [np.std([r["tem_event_aligned_decoding_acc"] for r in records]), np.std([r["hdgm_event_aligned_decoding_acc"] for r in records])]
    mean_long = [np.mean([r["tem_long_delay_performance"] for r in records]), np.mean([r["gate_long_delay_performance"] for r in records])]
    std_long = [np.std([r["tem_long_delay_performance"] for r in records]), np.std([r["gate_long_delay_performance"] for r in records])]
    mean_change = [np.mean([r["tem_state_change"] for r in records]), np.mean([r["gate_state_change"] for r in records])]
    std_change = [np.std([r["tem_state_change"] for r in records]), np.std([r["gate_state_change"] for r in records])]

    x = np.arange(len(labels))
    width = 0.23

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width, mean_acc, width, yerr=std_acc, capsize=4, label="Event-aligned decode")
    ax.bar(x, mean_long, width, yerr=std_long, capsize=4, label="Long-delay acc")
    ax.bar(x + width, mean_change, width, yerr=std_change, capsize=4, label="State change (lower=stable)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Multi-seed summary")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_multiseed_class_accuracy(records, save_path):
    classes = np.arange(4)
    tem_means = [np.mean([r[f"tem_class{i}"] for r in records]) for i in classes]
    tem_stds = [np.std([r[f"tem_class{i}"] for r in records]) for i in classes]
    gate_means = [np.mean([r[f"gate_class{i}"] for r in records]) for i in classes]
    gate_stds = [np.std([r[f"gate_class{i}"] for r in records]) for i in classes]

    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(classes - width/2, tem_means, width, yerr=tem_stds, capsize=4, label="TEM")
    ax.bar(classes + width/2, gate_means, width, yerr=gate_stds, capsize=4, label="HDGM")
    ax.set_xticks(classes)
    ax.set_xticklabels([f"Class {i}" for i in classes])
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-class accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_place_pca_and_distance(results, save_path):
    env = results["env"]
    tem_repr = collect_representations_on_bank(results["tem"], results["analysis_bank"], gated=False)
    gate_repr = collect_representations_on_bank(results["gate"], results["analysis_bank"], gated=True)

    # Subsample for clarity
    n_show = min(2500, len(tem_repr["mem"]), len(gate_repr["mem"]))
    plot_rng = np.random.RandomState(20260824)
    idx_t = plot_rng.choice(len(tem_repr["mem"]), size=n_show, replace=False)
    idx_g = plot_rng.choice(len(gate_repr["mem"]), size=n_show, replace=False)

    tem_proj = pca_project(tem_repr["mem"][idx_t], 2)
    gate_proj = pca_project(gate_repr["mem"][idx_g], 2)
    tem_pos = tem_repr["pos"][idx_t]
    gate_pos = gate_repr["pos"][idx_g]

    def adjacent_distance(mem, pos, track_len):
        means = []
        for p in range(track_len):
            m = mem[pos == p]
            if len(m) == 0:
                means.append(np.zeros(mem.shape[1], dtype=np.float64))
            else:
                means.append(m.mean(axis=0))
        means = np.stack(means, axis=0)
        adj = np.linalg.norm(means[1:] - means[:-1], axis=1)
        far = np.linalg.norm(means[track_len//2:] - means[:track_len-track_len//2], axis=1)
        return adj.mean(), far.mean()

    tem_adj, tem_far = adjacent_distance(tem_repr["mem"], tem_repr["pos"], env.track_len)
    gate_adj, gate_far = adjacent_distance(gate_repr["mem"], gate_repr["pos"], env.track_len)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sc0 = axes[0].scatter(tem_proj[:, 0], tem_proj[:, 1], c=tem_pos, s=8)
    axes[0].set_title("TEM position PCA")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04)

    sc1 = axes[1].scatter(gate_proj[:, 0], gate_proj[:, 1], c=gate_pos, s=8)
    axes[1].set_title("HDGM position PCA")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].bar(["TEM-adj", "TEM-far", "HDGM-adj", "HDGM-far"], [tem_adj, tem_far, gate_adj, gate_far])
    axes[2].set_title("Representation distance")
    axes[2].set_ylabel("Mean distance")

    fig.suptitle("Place representation analysis")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_top_place_fields(results, save_path, top_k=6):
    env = results["env"]
    tem_repr = collect_representations_on_bank(results["tem"], results["analysis_bank"], gated=False)
    gate_repr = collect_representations_on_bank(results["gate"], results["analysis_bank"], gated=True)

    tem_fields, tem_sel, tem_top = compute_place_fields(tem_repr["mem"], tem_repr["pos"], env.track_len, top_k=top_k)
    gate_fields, gate_sel, gate_top = compute_place_fields(gate_repr["mem"], gate_repr["pos"], env.track_len, top_k=top_k)

    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5))
    im0 = axes[0].imshow(tem_fields[tem_top], aspect="auto")
    axes[0].set_title("TEM top place-like units")
    axes[0].set_ylabel("Selected units")
    axes[0].set_xlabel("Position")
    fig.colorbar(im0, ax=axes[0], fraction=0.02, pad=0.02)

    im1 = axes[1].imshow(gate_fields[gate_top], aspect="auto")
    axes[1].set_title("HDGM top place-like units")
    axes[1].set_ylabel("Selected units")
    axes[1].set_xlabel("Position")
    fig.colorbar(im1, ax=axes[1], fraction=0.02, pad=0.02)

    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_paper_figures(cfg, records, representative_results, representative_record, save_dir, representative_seed=11):
    ensure_dir(save_dir)
    rep_dir = ensure_dir(os.path.join(save_dir, f"representative_seed_{representative_seed}"))

    # Reuse the already evaluated representative run. Plotting must never
    # retrain a seed or create a second, inconsistent set of reported results.
    results = representative_results

    plot_training_curves(results["history"], os.path.join(rep_dir, "fig_train_curves.png"))

    tem_eval = evaluate_model_on_bank(results["tem"], results["test_banks"]["normal"])
    gate_eval = evaluate_model_on_bank(results["gate"], results["test_banks"]["normal"])
    plot_confusion_matrices(tem_eval["cm"], gate_eval["cm"], os.path.join(rep_dir, "fig_confusion_matrices.png"))

    plot_gate_phase_means(representative_record, os.path.join(rep_dir, "fig_gate_phase_write.png"))

    plot_place_pca_and_distance(results, os.path.join(rep_dir, "fig_place_pca_distance.png"))
    plot_top_place_fields(results, os.path.join(rep_dir, "fig_place_fields.png"))

    plot_multiseed_main_bars(records, os.path.join(save_dir, "fig_multiseed_main_summary.png"))
    plot_multiseed_class_accuracy(records, os.path.join(save_dir, "fig_multiseed_class_accuracy.png"))

    with open(os.path.join(save_dir, "multiseed_records.json"), "w", encoding="utf-8") as f:
        json.dump({"records": records}, f, ensure_ascii=False, indent=2)

    print(f"\nSaved visualization figures to: {save_dir}")



# =========================================================
# 17. Main experiment
# =========================================================
def multi_seed_experiment(
    seeds=None,
    cfg=None,
    save_path="fourclass_paper_main_results.json",
    verbose=True,
    title="FOUR-CLASS PAPER MAIN SUMMARY",
    generate_figures=True
):
    cfg = cfg if cfg is not None else ExperimentConfig()
    if seeds is None:
        seeds=[11, 22, 33, 44, 55, 66, 77, 88, 99, 111]

    all_records = []
    representative_results = None
    representative_record = None

    print("\n" + "=" * 100)
    print(f"START MULTI-SEED EXPERIMENT | CONFIG = {cfg.name}")
    print(f"SEEDS = {seeds}")
    print("=" * 100)

    for seed in seeds:
        keep_results = generate_figures and seed == cfg.representative_seed
        run_output = run_one_seed(
            seed=seed,
            cfg=cfg,
            verbose=verbose,
            return_results=keep_results,
        )
        if keep_results:
            rec, representative_results = run_output
            representative_record = rec
        else:
            rec = run_output
        all_records.append(rec)

    print_multirun_summary(all_records, title=title)
    save_records(all_records, save_path)
    if generate_figures:
        if representative_results is None:
            raise ValueError(
                f"representative_seed={cfg.representative_seed} must be included in seeds when generate_figures=True"
            )
        generate_paper_figures(
            cfg=cfg,
            records=all_records,
            representative_results=representative_results,
            representative_record=representative_record,
            save_dir=cfg.save_dir,
            representative_seed=cfg.representative_seed
        )
    return all_records


# =========================================================
# 18. V4 rapid-acquisition protocol
# =========================================================
def state_dict_to_cpu(model):
    """Immutable checkpoint snapshot that does not keep a GPU computation graph."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def rapid_curriculum_batch(env, batch_size, update, primary_horizon=150):
    """
    Lock the first 150 updates to the same curriculum even when the optional
    500-update extended-exposure control is enabled.

    This prevents the early learning curve from changing merely because a
    longer terminal budget was requested.
    """
    capped_update = min(int(update), int(primary_horizon))
    progress = (capped_update - 1) / max(1, int(primary_horizon) - 1)
    r = np.random.rand()
    if progress < 0.33:
        mode = "short" if r < 0.45 else ("normal" if r < 0.90 else "long")
    elif progress < 0.66:
        mode = "short" if r < 0.25 else ("normal" if r < 0.70 else "long")
    else:
        mode = "short" if r < 0.15 else ("normal" if r < 0.45 else "long")
    return build_balanced_batch(env, batch_size=batch_size, delay_mode=mode), mode


def build_v4_banks(env, cfg, seed):
    """Fresh V4 validation, test, analysis, and probe splits."""
    split_base = int(cfg.split_seed_version * 100_000)
    split_seeds = {
        "val_normal": int(seed + split_base + 10_001),
        "val_long": int(seed + split_base + 10_002),
        "test_normal": int(seed + split_base + 20_001),
        "test_long": int(seed + split_base + 20_002),
        "analysis": int(seed + split_base + 30_001),
        "probe_train": int(seed + split_base + 40_001),
        "probe_val": int(seed + split_base + 40_002),
        "probe_test": int(seed + split_base + 40_003),
    }
    val_banks = {
        "normal": build_fixed_batch_bank(
            env, cfg.val_batches, cfg.val_batch_size, "normal", split_seeds["val_normal"]
        ),
        "long": build_fixed_batch_bank(
            env, cfg.val_batches, cfg.val_batch_size, "long", split_seeds["val_long"]
        ),
    }
    test_banks = {
        "normal": build_fixed_batch_bank(
            env, cfg.test_batches, cfg.test_batch_size, "normal", split_seeds["test_normal"]
        ),
        "long": build_fixed_batch_bank(
            env, cfg.test_batches, cfg.test_batch_size, "long", split_seeds["test_long"]
        ),
    }
    analysis_bank = build_fixed_batch_bank(
        env, cfg.analysis_batches, cfg.analysis_batch_size, "normal", split_seeds["analysis"]
    )
    probe_banks = {
        "train": build_fixed_batch_bank(
            env, cfg.probe_train_batches, cfg.probe_batch_size,
            "normal", split_seeds["probe_train"]
        ),
        "val": build_fixed_batch_bank(
            env, cfg.probe_val_batches, cfg.probe_batch_size,
            "normal", split_seeds["probe_val"]
        ),
        "test": build_fixed_batch_bank(
            env, cfg.probe_test_batches, cfg.probe_batch_size,
            "normal", split_seeds["probe_test"]
        ),
    }
    return val_banks, test_banks, analysis_bank, probe_banks, split_seeds


def instantiate_v4_models(env, cfg):
    tem = TEMLite(
        obs_dim=cfg.obs_dim,
        hidden_dim=96,
        track_len=env.track_len,
        action_dim=env.action_dim,
        n_classes=4,
        cfg=cfg,
    ).to(device)
    hdgm = HDGM(
        obs_dim=cfg.obs_dim,
        hidden_dim=96,
        track_len=env.track_len,
        action_dim=env.action_dim,
        n_classes=4,
        cfg=cfg,
    ).to(device)
    return tem, hdgm


def selected_representation_history(out, representation):
    mapping = {
        "output_state": "mem",
        "feedback_state": "v_hist",
        "ca1_gate_pattern": "ca1_hist",
    }
    if representation not in mapping:
        raise ValueError(f"Unsupported V4 representation: {representation}")
    key = mapping[representation]
    if key not in out:
        raise KeyError(
            f"Model output has no '{key}' required for '{representation}'. "
            f"Available keys: {sorted(out.keys())}"
        )
    return out[key]


def extract_v4_representation_features(model, bank, representation):
    """Decision-pre features; event windows never enter the model forward pass."""
    features, cue_as, cue_bs, labels = [], [], [], []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in bank:
            x, a, pos, phase, y, cueA, cueB, cueA_zones, cueB_zones, decision_positions, delay1s, delay2s = batch
            out = model(
                x.to(device),
                a.to(device),
                return_details=True,
                skip_readout=True,
            )
            history = selected_representation_history(out, representation)
            indices = torch.tensor(
                [max(0, min(int(d) - 1, history.shape[1] - 1)) for d in decision_positions],
                dtype=torch.long,
                device=device,
            )
            batch_indices = torch.arange(history.shape[0], device=device)
            features.append(history[batch_indices, indices].detach().cpu())
            cue_as.append(cueA.detach().cpu())
            cue_bs.append(cueB.detach().cpu())
            labels.append(y.detach().cpu())
    model.train(was_training)
    return {
        "x": torch.cat(features, dim=0),
        "cueA": torch.cat(cue_as, dim=0),
        "cueB": torch.cat(cue_bs, dim=0),
        "class": torch.cat(labels, dim=0),
    }


def fit_v4_linear_probe(train_data, val_data, test_data, cfg, probe_seed):
    """Fit only a linear readout; model parameters remain frozen."""
    train_mean = train_data["x"].mean(dim=0, keepdim=True)
    train_std = train_data["x"].std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)

    def standardize(data):
        return {
            "x": ((data["x"] - train_mean) / train_std).to(device),
            "cueA": data["cueA"].to(device),
            "cueB": data["cueB"].to(device),
            "class": data["class"].to(device),
        }

    train_data = standardize(train_data)
    val_data = standardize(val_data)
    test_data = standardize(test_data)

    fork_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(int(probe_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(probe_seed))
        probe = FrozenDecisionProbe(train_data["x"].shape[1]).to(device)
        optimizer = optim.Adam(
            probe.parameters(), lr=cfg.probe_lr, weight_decay=cfg.probe_weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        best_state, best_val, best_epoch = None, -1.0, -1
        for epoch in range(cfg.probe_epochs):
            probe.train()
            optimizer.zero_grad()
            logits = probe(train_data["x"])
            loss = sum(
                criterion(logits[name], train_data[name])
                for name in ("cueA", "cueB", "class")
            )
            loss.backward()
            optimizer.step()
            probe.eval()
            with torch.no_grad():
                val_logits = probe(val_data["x"])
                score = float(np.mean([
                    probe_accuracy(val_logits[name], val_data[name])
                    for name in ("cueA", "cueB", "class")
                ]))
            if score > best_val:
                best_val = score
                best_epoch = epoch + 1
                best_state = copy.deepcopy(probe.state_dict())
        probe.load_state_dict(best_state)
        probe.eval()
        with torch.no_grad():
            logits = probe(test_data["x"])
            result = {
                name: probe_accuracy(logits[name], test_data[name])
                for name in ("cueA", "cueB", "class")
            }
            result["class_confusion_matrix"] = compute_confusion_matrix(
                logits["class"], test_data["class"], n_classes=4
            ).tolist()
    result["best_val_score"] = float(best_val)
    result["best_epoch"] = int(best_epoch)
    return result


def run_v4_probe(model, probe_banks, representation, cfg, probe_seed):
    datasets = {
        split: extract_v4_representation_features(model, probe_banks[split], representation)
        for split in ("train", "val", "test")
    }
    return fit_v4_linear_probe(
        datasets["train"], datasets["val"], datasets["test"], cfg, probe_seed
    )


def normalized_trapezoid_auc(checkpoints, values, horizon=150):
    pairs = sorted(
        (int(x), float(y))
        for x, y in zip(checkpoints, values)
        if int(x) <= int(horizon)
    )
    if len(pairs) < 2:
        raise ValueError("AULC requires at least two checkpoints.")
    area = sum(
        (x1 - x0) * (y0 + y1) * 0.5
        for (x0, y0), (x1, y1) in zip(pairs[:-1], pairs[1:])
    )
    return float(area / (pairs[-1][0] - pairs[0][0]))


def first_evaluated_threshold(checkpoints, values, threshold):
    for checkpoint, value in zip(checkpoints, values):
        if float(value) >= float(threshold):
            return int(checkpoint)
    return None


def train_v4_exact_checkpoints(seed, cfg, max_updates):
    """Train once and snapshot exact, predeclared updates without model selection."""
    set_global_seed(seed)
    env = SharedPathFourContextEnv(
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
    tem, hdgm = instantiate_v4_models(env, cfg)
    opt_tem = optim.Adam(tem.parameters(), lr=cfg.lr)
    opt_hdgm = optim.Adam(hdgm.parameters(), lr=cfg.lr)
    val_banks, test_banks, analysis_bank, probe_banks, split_seeds = build_v4_banks(
        env, cfg, seed
    )

    requested = cfg.extended_checkpoints if max_updates > cfg.primary_updates else cfg.rapid_checkpoints
    checkpoints = tuple(sorted({int(x) for x in requested if int(x) <= int(max_updates)}))
    if not checkpoints or checkpoints[-1] != int(max_updates):
        checkpoints = tuple(sorted(set(checkpoints + (int(max_updates),))))

    history = {
        "update": [], "delay_mode": [],
        "tem_train_acc": [], "hdgm_train_acc": [],
        "tem_train_loss": [], "hdgm_train_loss": [],
        "monitor_update": [],
        "tem_val_normal": [], "tem_val_long": [],
        "hdgm_val_normal": [], "hdgm_val_long": [],
    }
    snapshots = {}
    sampled_modes = []
    criterion = nn.CrossEntropyLoss()

    print(f"===== V4 TRAINING START [{cfg.name}] | MAX_UPDATES={max_updates} =====")
    for update in range(1, int(max_updates) + 1):
        batch, delay_mode = rapid_curriculum_batch(
            env, cfg.batch_size, update, primary_horizon=cfg.primary_updates
        )
        sampled_modes.append(delay_mode)
        x, a, pos, phase, labels, cueA, cueB, cueA_zones, cueB_zones, decision_positions, delay1s, delay2s = batch
        x, a, phase, labels = x.to(device), a.to(device), phase.to(device), labels.to(device)

        opt_tem.zero_grad()
        out_t = tem(
            x, a, cueA_zones=cueA_zones, cueB_zones=cueB_zones,
            decision_positions=decision_positions, return_details=True
        )
        tem_loss = out_t["pred_loss"] + 0.7 * criterion(out_t["logits"], labels)
        tem_loss.backward()
        torch.nn.utils.clip_grad_norm_(tem.parameters(), 1.0)
        opt_tem.step()

        opt_hdgm.zero_grad()
        out_h = hdgm(
            x, a, cueA_zones=cueA_zones, cueB_zones=cueB_zones,
            decision_positions=decision_positions, return_details=True
        )
        margin_loss = compute_cueB_margin_loss(out_h["write_hist"], phase, margin=0.05) \
            if cfg.use_cueB_margin_loss else torch.tensor(0.0, device=device)
        rank_loss = compute_ec_phase_ranking_loss(out_h["write_hist"], phase, margin=0.02) \
            if cfg.use_ec_phase_rank_loss else torch.tensor(0.0, device=device)
        hdgm_loss = (
            out_h["pred_loss"]
            + 0.7 * criterion(out_h["logits"], labels)
            + cfg.gate_cueB_margin_weight * margin_loss
            + cfg.gate_ec_phase_rank_weight * rank_loss
        )
        hdgm_loss.backward()
        torch.nn.utils.clip_grad_norm_(hdgm.parameters(), 1.0)
        opt_hdgm.step()

        tem_acc = compute_accuracy(out_t["logits"], labels)
        hdgm_acc = compute_accuracy(out_h["logits"], labels)
        history["update"].append(update)
        history["delay_mode"].append(delay_mode)
        history["tem_train_acc"].append(float(tem_acc))
        history["hdgm_train_acc"].append(float(hdgm_acc))
        history["tem_train_loss"].append(float(tem_loss.item()))
        history["hdgm_train_loss"].append(float(hdgm_loss.item()))

        if update in checkpoints:
            snapshots[update] = {
                "tem": state_dict_to_cpu(tem),
                "hdgm": state_dict_to_cpu(hdgm),
            }
            tem_vn = evaluate_model_on_bank(tem, val_banks["normal"])["mean_acc"]
            tem_vl = evaluate_model_on_bank(tem, val_banks["long"])["mean_acc"]
            hdgm_vn = evaluate_model_on_bank(hdgm, val_banks["normal"])["mean_acc"]
            hdgm_vl = evaluate_model_on_bank(hdgm, val_banks["long"])["mean_acc"]
            history["monitor_update"].append(update)
            history["tem_val_normal"].append(float(tem_vn))
            history["tem_val_long"].append(float(tem_vl))
            history["hdgm_val_normal"].append(float(hdgm_vn))
            history["hdgm_val_long"].append(float(hdgm_vl))
            print(
                f"Update {update:04d}/{max_updates} | mode={delay_mode:<6} | "
                f"train TEM/HDGM={tem_acc:.3f}/{hdgm_acc:.3f} | "
                f"monitor normal={tem_vn:.3f}/{hdgm_vn:.3f} | "
                f"long={tem_vl:.3f}/{hdgm_vl:.3f}"
            )

    return {
        "env": env, "tem": tem, "hdgm": hdgm,
        "val_banks": val_banks, "test_banks": test_banks,
        "analysis_bank": analysis_bank, "probe_banks": probe_banks,
        "split_seeds": split_seeds, "history": history,
        "snapshots": snapshots, "checkpoints": checkpoints,
        "sampled_modes": sampled_modes,
    }


def evaluate_v4_checkpoints(results, cfg, seed, run_probes=True):
    tem, hdgm = results["tem"], results["hdgm"]
    curve = {}
    probe_results = {}
    probe_checkpoints = {
        int(x) for x in cfg.rapid_probe_checkpoints
        if int(x) in results["checkpoints"]
    }

    for checkpoint in results["checkpoints"]:
        tem.load_state_dict(results["snapshots"][checkpoint]["tem"], strict=True)
        hdgm.load_state_dict(results["snapshots"][checkpoint]["hdgm"], strict=True)
        tem_normal = evaluate_model_on_bank(tem, results["test_banks"]["normal"])
        tem_long = evaluate_model_on_bank(tem, results["test_banks"]["long"])
        hdgm_normal = evaluate_model_on_bank(hdgm, results["test_banks"]["normal"])
        hdgm_long = evaluate_model_on_bank(hdgm, results["test_banks"]["long"])
        curve[str(checkpoint)] = {
            "tem_normal": float(tem_normal["mean_acc"]),
            "hdgm_normal": float(hdgm_normal["mean_acc"]),
            "tem_long": float(tem_long["mean_acc"]),
            "hdgm_long": float(hdgm_long["mean_acc"]),
            "tem_confusion_matrix": tem_normal["cm"].tolist(),
            "hdgm_confusion_matrix": hdgm_normal["cm"].tolist(),
        }
        if run_probes and checkpoint in probe_checkpoints:
            print(f"Running frozen V4 probes at exact update {checkpoint} ...")
            probe_base = int(results["split_seeds"]["probe_train"] + checkpoint * 1000)
            probe_results[str(checkpoint)] = {
                "tem_output_state": run_v4_probe(
                    tem, results["probe_banks"], "output_state", cfg, probe_base + 1
                ),
                "hdgm_output_state": run_v4_probe(
                    hdgm, results["probe_banks"], "output_state", cfg, probe_base + 2
                ),
                "hdgm_feedback_state": run_v4_probe(
                    hdgm, results["probe_banks"], "feedback_state", cfg, probe_base + 3
                ),
                "hdgm_ca1_gate_pattern": run_v4_probe(
                    hdgm, results["probe_banks"], "ca1_gate_pattern", cfg, probe_base + 4
                ),
            }

    primary_points = [x for x in results["checkpoints"] if x <= cfg.primary_updates]
    tem_values = [curve[str(x)]["tem_normal"] for x in primary_points]
    hdgm_values = [curve[str(x)]["hdgm_normal"] for x in primary_points]
    tem_long_values = [curve[str(x)]["tem_long"] for x in primary_points]
    hdgm_long_values = [curve[str(x)]["hdgm_long"] for x in primary_points]
    summary = {
        "tem_early_aulc": normalized_trapezoid_auc(
            primary_points, tem_values, cfg.primary_updates
        ),
        "hdgm_early_aulc": normalized_trapezoid_auc(
            primary_points, hdgm_values, cfg.primary_updates
        ),
        "tem_early_long_aulc": normalized_trapezoid_auc(
            primary_points, tem_long_values, cfg.primary_updates
        ),
        "hdgm_early_long_aulc": normalized_trapezoid_auc(
            primary_points, hdgm_long_values, cfg.primary_updates
        ),
        "tem_acc_at_100": curve.get("100", {}).get("tem_normal"),
        "hdgm_acc_at_100": curve.get("100", {}).get("hdgm_normal"),
        "tem_long_at_100": curve.get("100", {}).get("tem_long"),
        "hdgm_long_at_100": curve.get("100", {}).get("hdgm_long"),
        "tem_acc_at_150": curve.get("150", {}).get("tem_normal"),
        "hdgm_acc_at_150": curve.get("150", {}).get("hdgm_normal"),
        "tem_terminal_acc": curve[str(results["checkpoints"][-1])]["tem_normal"],
        "hdgm_terminal_acc": curve[str(results["checkpoints"][-1])]["hdgm_normal"],
        "tem_terminal_long": curve[str(results["checkpoints"][-1])]["tem_long"],
        "hdgm_terminal_long": curve[str(results["checkpoints"][-1])]["hdgm_long"],
    }
    for threshold in cfg.acquisition_thresholds:
        suffix = str(threshold).replace(".", "_")
        summary[f"tem_first_checkpoint_ge_{suffix}"] = first_evaluated_threshold(
            primary_points, tem_values, threshold
        )
        summary[f"hdgm_first_checkpoint_ge_{suffix}"] = first_evaluated_threshold(
            primary_points, hdgm_values, threshold
        )
    return curve, probe_results, summary


def run_v4_seed(seed, cfg, max_updates=150, run_probes=True, label="primary"):
    print("\n" + "=" * 110)
    print(f"V4 RAPID ACQUISITION | SEED={seed} | MODE={label} | MAX_UPDATES={max_updates}")
    print("Exact checkpoints are evaluated without best-validation checkpoint selection.")
    print("=" * 110)
    results = train_v4_exact_checkpoints(seed, cfg, max_updates=max_updates)
    curve, probes, rapid_summary = evaluate_v4_checkpoints(
        results, cfg, seed, run_probes=run_probes
    )

    terminal = int(results["checkpoints"][-1])
    results["tem"].load_state_dict(results["snapshots"][terminal]["tem"], strict=True)
    results["hdgm"].load_state_dict(results["snapshots"][terminal]["hdgm"], strict=True)
    legacy_results = {
        "env": results["env"], "tem": results["tem"], "gate": results["hdgm"],
        "analysis_bank": results["analysis_bank"], "test_banks": results["test_banks"],
        "cfg": cfg,
    }

    record = {
        "seed": int(seed),
        "mode": label,
        "max_updates": int(max_updates),
        "config": asdict(cfg),
        "split_seeds": results["split_seeds"],
        "protocol": {
            "confirmatory_primary_horizon": int(cfg.primary_updates),
            "exact_checkpoints_no_model_selection": True,
            "fresh_v4_test_split": True,
            "models_receive_identical_training_batches": True,
            "test_used_for_training_or_selection": False,
            "cue_windows_enter_recurrent_dynamics": False,
            "cue_windows_enter_event_aligned_decoder": bool(cfg.use_branch_readout),
            "probe_models_frozen": True,
            "probe_receives_cue_windows": False,
            "extended_control_is_secondary": bool(max_updates > cfg.primary_updates),
        },
        "checkpoint_curve": curve,
        "frozen_probes": probes,
        "training_history": results["history"],
        "curriculum_counts": {
            mode: int(results["sampled_modes"].count(mode))
            for mode in ("short", "normal", "long")
        },
    }
    record.update(rapid_summary)
    # Mechanism/place summaries belong to the 150-update biological endpoint.
    # The optional 500-update run is deliberately a lightweight behavioral
    # overexposure control and therefore skips these extra analyses.
    if max_updates <= cfg.primary_updates:
        record.update(compute_biological_metrics_only(legacy_results))
        record.update(compute_tem_metrics_only(legacy_results))
        record.update(compute_gate_metrics_only(legacy_results))

    checkpoint_dir = ensure_dir(cfg.checkpoint_dir)
    checkpoint_path = os.path.join(
        checkpoint_dir, f"{cfg.name}_{label}_seed_{seed}.pt"
    )
    torch.save({
        "seed": int(seed),
        "mode": label,
        "max_updates": int(max_updates),
        "config": asdict(cfg),
        "split_seeds": results["split_seeds"],
        "snapshots": results["snapshots"],
        "environment_maps": {
            "pos_map": results["env"].pos_map.copy(),
            "cueA_map": results["env"].cueA_map.copy(),
            "cueB_map": results["env"].cueB_map.copy(),
        },
        "training_history": results["history"],
    }, checkpoint_path)
    record["checkpoint_path"] = checkpoint_path

    print("\n===== V4 SEED SUMMARY =====")
    print(
        f"Early AULC TEM/HDGM = {record['tem_early_aulc']:.4f}/"
        f"{record['hdgm_early_aulc']:.4f}"
    )
    print(
        f"Accuracy@100 TEM/HDGM = {record['tem_acc_at_100']:.4f}/"
        f"{record['hdgm_acc_at_100']:.4f}"
    )
    print(
        f"Accuracy@150 TEM/HDGM = {record['tem_acc_at_150']:.4f}/"
        f"{record['hdgm_acc_at_150']:.4f}"
    )
    return record


def optional_number(value):
    return np.nan if value is None else float(value)


def print_v4_multiseed_summary(records, title):
    print("\n" + "#" * 110)
    print(title)
    print("#" * 110)
    metrics = summarize_scalar_dict([{
        "tem_early_aulc": r["tem_early_aulc"],
        "hdgm_early_aulc": r["hdgm_early_aulc"],
        "tem_early_long_aulc": r["tem_early_long_aulc"],
        "hdgm_early_long_aulc": r["hdgm_early_long_aulc"],
        "tem_acc_at_100": r["tem_acc_at_100"],
        "hdgm_acc_at_100": r["hdgm_acc_at_100"],
        "tem_long_at_100": r["tem_long_at_100"],
        "hdgm_long_at_100": r["hdgm_long_at_100"],
        "tem_acc_at_150": r["tem_acc_at_150"],
        "hdgm_acc_at_150": r["hdgm_acc_at_150"],
        "tem_terminal_acc": r["tem_terminal_acc"],
        "hdgm_terminal_acc": r["hdgm_terminal_acc"],
    } for r in records])
    format_summary_block("Rapid-acquisition endpoints", metrics)

    if len(records) >= 2:
        aulc_test = paired_permutation_test(
            [r["hdgm_early_aulc"] for r in records],
            [r["tem_early_aulc"] for r in records],
            n_perm=20000, seed=401,
        )
        acc100_test = paired_permutation_test(
            [r["hdgm_acc_at_100"] for r in records],
            [r["tem_acc_at_100"] for r in records],
            n_perm=20000, seed=402,
        )
        print("\n===== Confirmatory paired permutation tests =====")
        print(
            f"Early AULC (HDGM-TEM) | diff={aulc_test['mean_diff']:.4f}, "
            f"p={aulc_test['p_value']:.5f}, dz={aulc_test['cohend_z']:.4f}"
        )
        print(
            f"Accuracy@100 (HDGM-TEM) | diff={acc100_test['mean_diff']:.4f}, "
            f"p={acc100_test['p_value']:.5f}, dz={acc100_test['cohend_z']:.4f}"
        )

    print("\n===== First evaluated checkpoint reaching threshold =====")
    for threshold in records[0]["config"]["acquisition_thresholds"]:
        suffix = str(threshold).replace(".", "_")
        tem_values = [optional_number(r[f"tem_first_checkpoint_ge_{suffix}"]) for r in records]
        hdgm_values = [optional_number(r[f"hdgm_first_checkpoint_ge_{suffix}"]) for r in records]
        print(
            f">={threshold:.2f} | TEM median={np.nanmedian(tem_values):.1f} "
            f"({np.sum(np.isfinite(tem_values))}/{len(records)} reached) | "
            f"HDGM median={np.nanmedian(hdgm_values):.1f} "
            f"({np.sum(np.isfinite(hdgm_values))}/{len(records)} reached)"
        )


def plot_v4_learning_curves(records, save_path):
    show_extended = int(records[0]["max_updates"]) > int(
        records[0]["config"]["primary_updates"]
    )
    checkpoints = [
        int(x) for x in records[0]["checkpoint_curve"].keys()
        if show_extended
        or int(x) <= int(records[0]["config"]["primary_updates"])
    ]
    checkpoints = sorted(checkpoints)
    tem = np.array([
        [r["checkpoint_curve"][str(x)]["tem_normal"] for x in checkpoints]
        for r in records
    ], dtype=np.float64)
    hdgm = np.array([
        [r["checkpoint_curve"][str(x)]["hdgm_normal"] for x in checkpoints]
        for r in records
    ], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for values, color, label in (
        (tem, "#475569", "TEM"),
        (hdgm, "#2E5BFF", "HDGM"),
    ):
        mean, std = values.mean(axis=0), values.std(axis=0)
        ax.plot(checkpoints, mean, marker="o", lw=2.2, color=color, label=label)
        ax.fill_between(checkpoints, mean - std, mean + std, color=color, alpha=0.18)
    ax.axhline(0.25, color="#94A3B8", ls="--", lw=1.2, label="Chance")
    ax.set_xlabel("Training updates")
    ax.set_ylabel("Held-out event-aligned decoding accuracy")
    ax.set_title(
        "Extended repeated-exposure control" if show_extended
        else "Rapid acquisition under a locked 150-update budget"
    )
    ax.set_xticks(checkpoints)
    ax.set_ylim(0.20, 1.02)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_v4_probe_acquisition(records, save_path):
    available = sorted({
        int(checkpoint)
        for record in records
        for checkpoint in record["frozen_probes"].keys()
    })
    if not available:
        return
    series = {
        "TEM output": ("tem_output_state", "#475569"),
        "HDGM output": ("hdgm_output_state", "#14B8A6"),
        "HDGM feedback": ("hdgm_feedback_state", "#2E5BFF"),
        "HDGM CA1 gate": ("hdgm_ca1_gate_pattern", "#F59E0B"),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for label, (key, color) in series.items():
        values = np.array([
            [r["frozen_probes"][str(x)][key]["class"] for x in available]
            for r in records
        ], dtype=np.float64)
        mean, std = values.mean(axis=0), values.std(axis=0)
        ax.plot(available, mean, marker="o", lw=2.0, color=color, label=label)
        ax.fill_between(available, mean - std, mean + std, color=color, alpha=0.14)
    ax.axhline(0.25, color="#94A3B8", ls="--", lw=1.2)
    ax.set_xlabel("Training updates")
    ax.set_ylabel("Frozen linear-probe class accuracy")
    ax.set_title("Acquisition of decision-pre representations")
    ax.set_xticks(available)
    ax.set_ylim(0.20, 1.02)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def run_v4_multiseed(
    seeds, cfg, max_updates=150, run_probes=True,
    label="primary", save_path=None, generate_figures=True,
):
    print("\n" + "=" * 110)
    print(f"START V4 MULTI-SEED | MODE={label} | MAX_UPDATES={max_updates}")
    print(f"SEEDS={seeds}")
    print("=" * 110)
    records = [
        run_v4_seed(seed, cfg, max_updates=max_updates, run_probes=run_probes, label=label)
        for seed in seeds
    ]
    title = (
        "V4 RAPID-ACQUISITION CONFIRMATORY SUMMARY"
        if max_updates <= cfg.primary_updates
        else "V4 EXTENDED-EXPOSURE SECONDARY CONTROL"
    )
    print_v4_multiseed_summary(records, title)
    if save_path is None:
        save_path = f"{cfg.name}_{label}_results.json"
    save_records(records, save_path)
    if generate_figures:
        ensure_dir(cfg.save_dir)
        plot_v4_learning_curves(
            records, os.path.join(cfg.save_dir, f"fig_v4_learning_curve_{label}.png")
        )
        if run_probes:
            plot_v4_probe_acquisition(
                records, os.path.join(cfg.save_dir, f"fig_v4_probe_acquisition_{label}.png")
            )
    return records


# =========================================================
# 19. Shared frozen-checkpoint helpers used by V6.2
# =========================================================
def load_torch_checkpoint(path):
    """Compatibility loader for PyTorch versions with/without weights_only."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def state_dict_sha256(state_dict):
    """Stable fingerprint proving which frozen HDGM tensor set was analysed."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def experiment_config_from_payload(payload):
    """Reconstruct only known V4 fields; tolerate future metadata fields."""
    cfg = ExperimentConfig()
    valid = set(ExperimentConfig.__dataclass_fields__.keys())
    for key, value in payload.get("config", {}).items():
        if key in valid:
            current = getattr(cfg, key)
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(cfg, key, value)
    return cfg


def environment_from_checkpoint(payload, cfg, seed):
    env = SharedPathFourContextEnv(
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
    maps = payload.get("environment_maps")
    if maps is None:
        raise KeyError("V4 checkpoint has no environment_maps.")
    env.pos_map = np.asarray(maps["pos_map"], dtype=np.float32).copy()
    env.cueA_map = np.asarray(maps["cueA_map"], dtype=np.float32).copy()
    env.cueB_map = np.asarray(maps["cueB_map"], dtype=np.float32).copy()
    return env


def build_v62_banks(env, analysis_cfg, seed):
    base = int(analysis_cfg.split_seed_version * 100_000)
    seeds = {
        "behavior_normal": seed + base + 10_001,
        "behavior_long": seed + base + 10_002,
        "calibration": seed + base + 15_001,
        "probe_train": seed + base + 20_001,
        "probe_val": seed + base + 20_002,
        "probe_test": seed + base + 20_003,
    }
    if len(set(seeds.values())) != len(seeds):
        raise AssertionError("Calibration/probe/behavior RNG streams are not disjoint.")
    calibration = build_fixed_batch_bank(
        env, analysis_cfg.calibration_batches,
        analysis_cfg.calibration_batch_size, "normal", seeds["calibration"]
    )
    behavior = {
        "normal": build_fixed_batch_bank(
            env, analysis_cfg.behavior_batches,
            analysis_cfg.behavior_batch_size, "normal", seeds["behavior_normal"]
        ),
        "long": build_fixed_batch_bank(
            env, analysis_cfg.behavior_batches,
            analysis_cfg.behavior_batch_size, "long", seeds["behavior_long"]
        ),
    }
    probe = {
        "train": build_fixed_batch_bank(
            env, analysis_cfg.probe_train_batches,
            analysis_cfg.probe_batch_size, "normal", seeds["probe_train"]
        ),
        "val": build_fixed_batch_bank(
            env, analysis_cfg.probe_val_batches,
            analysis_cfg.probe_batch_size, "normal", seeds["probe_val"]
        ),
        "test": build_fixed_batch_bank(
            env, analysis_cfg.probe_test_batches,
            analysis_cfg.probe_batch_size, "normal", seeds["probe_test"]
        ),
    }
    return behavior, calibration, probe, {key: int(value) for key, value in seeds.items()}


def v6_representation_history(out, representation):
    mapping = {
        "feedback_state": "v_hist",
        "ca1_gate_pattern": "learned_ca1_hist",
        "pre_ca1_state": "phat_hist",
        "output_state": "mem",
    }
    if representation not in mapping:
        raise ValueError(f"Unsupported V6 representation: {representation}")
    return out[mapping[representation]]


def fit_v6_probe(train_data, val_data, test_data, analysis_cfg, probe_seed):
    """Independent train/validation/test probe; no recurrent parameter updates."""
    train_mean = train_data["x"].mean(dim=0, keepdim=True)
    train_std = train_data["x"].std(
        dim=0, unbiased=False, keepdim=True
    ).clamp_min(1e-4)

    def standardized(data):
        return {
            "x": ((data["x"] - train_mean) / train_std).to(device),
            "cueA": data["cueA"].to(device),
            "cueB": data["cueB"].to(device),
            "class": data["class"].to(device),
        }

    train = standardized(train_data)
    val = standardized(val_data)
    test = standardized(test_data)
    fork_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(int(probe_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(probe_seed))
        probe = FrozenDecisionProbe(train["x"].shape[1]).to(device)
        optimizer = optim.Adam(
            probe.parameters(), lr=analysis_cfg.probe_lr,
            weight_decay=analysis_cfg.probe_weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        best_state, best_score, best_epoch = None, -np.inf, -1
        for epoch in range(analysis_cfg.probe_epochs):
            probe.train()
            optimizer.zero_grad()
            logits = probe(train["x"])
            loss = sum(
                criterion(logits[target], train[target])
                for target in ("cueA", "cueB", "class")
            )
            loss.backward()
            optimizer.step()
            probe.eval()
            with torch.no_grad():
                val_logits = probe(val["x"])
                score = float(np.mean([
                    probe_accuracy(val_logits[target], val[target])
                    for target in ("cueA", "cueB", "class")
                ]))
            if score > best_score:
                best_score = score
                best_epoch = epoch + 1
                best_state = copy.deepcopy(probe.state_dict())
        probe.load_state_dict(best_state)
        probe.eval()
        with torch.no_grad():
            logits = probe(test["x"])
            result = {
                target: probe_accuracy(logits[target], test[target])
                for target in ("cueA", "cueB", "class")
            }
            result["class_confusion_matrix"] = compute_confusion_matrix(
                logits["class"], test["class"], n_classes=4
            ).tolist()
    result["best_val_score"] = float(best_score)
    result["best_epoch"] = int(best_epoch)
    return result


def class_geometry_ratio(features, labels):
    x = features.detach().float()
    y = labels.detach().long()
    centroids, within = [], []
    for klass in range(4):
        members = x[y == klass]
        if members.shape[0] == 0:
            return float("nan")
        centroid = members.mean(dim=0)
        centroids.append(centroid)
        within.append(torch.linalg.vector_norm(members - centroid, dim=1).mean())
    centroid_matrix = torch.stack(centroids, dim=0)
    between = []
    for left in range(4):
        for right in range(left + 1, 4):
            between.append(torch.linalg.vector_norm(
                centroid_matrix[left] - centroid_matrix[right]
            ))
    return float(torch.stack(between).mean() / (torch.stack(within).mean() + 1e-8))


def paired_feature_metrics(output_data, pre_data, feedback_data=None):
    output_x = output_data["x"].float()
    pre_x = pre_data["x"].float()
    if output_x.shape != pre_x.shape:
        raise ValueError("Output and pre-CA1 datasets are not trial-aligned.")
    metrics = {
        "output_geometry_ratio": class_geometry_ratio(
            output_x, output_data["class"]
        ),
        "pre_ca1_geometry_ratio": class_geometry_ratio(
            pre_x, pre_data["class"]
        ),
        "geometry_change_output_minus_pre": (
            class_geometry_ratio(output_x, output_data["class"])
            - class_geometry_ratio(pre_x, pre_data["class"])
        ),
        "ca1_transformation_norm": float(
            torch.linalg.vector_norm(output_x - pre_x, dim=1).mean()
        ),
        "pre_output_cosine": float(torch.nn.functional.cosine_similarity(
            pre_x, output_x, dim=1
        ).mean()),
    }
    if feedback_data is not None:
        feedback_x = feedback_data["x"].float()
        metrics["feedback_output_cosine"] = float(
            torch.nn.functional.cosine_similarity(
                feedback_x, output_x, dim=1
            ).mean()
        )
    return metrics


def exact_paired_sign_flip(x, y):
    differences = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    n = len(differences)
    if n == 0:
        return {"mean_diff": np.nan, "ci95_low": np.nan, "ci95_high": np.nan,
                "p_value": np.nan, "cohend_z": np.nan}
    observed = abs(float(differences.mean()))
    if n <= 16:
        null = []
        for code in range(1 << n):
            signs = np.array([
                1.0 if ((code >> bit) & 1) else -1.0 for bit in range(n)
            ])
            null.append(abs(float(np.mean(signs * differences))))
        p_value = float(np.mean(np.asarray(null) >= observed - 1e-15))
    else:
        rng = np.random.RandomState(86)
        null = np.abs(np.mean(
            rng.choice([-1.0, 1.0], size=(100_000, n)) * differences,
            axis=1,
        ))
        p_value = float((np.sum(null >= observed) + 1) / (len(null) + 1))
    sd = float(differences.std(ddof=1)) if n > 1 else np.nan
    dz = float(differences.mean() / sd) if np.isfinite(sd) and sd > 0 else np.nan
    bootstrap_rng = np.random.RandomState(62_000 + n)
    bootstrap_index = bootstrap_rng.randint(0, n, size=(50_000, n))
    ci_low, ci_high = np.percentile(
        differences[bootstrap_index].mean(axis=1), [2.5, 97.5]
    )
    return {
        "mean_diff": float(differences.mean()),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_value": p_value,
        "cohend_z": dz,
    }


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    sd = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    return float(np.mean(finite)), sd


# =========================================================
# 20. V6.2 causal/cross-trial-calibrated CA1 validation
# =========================================================
@dataclass
class CA1CausalAnalysisConfig:
    """Locked corrective analysis after auditing the V6 per-trial mean control."""

    primary_checkpoint: int = 100
    conditions: tuple = (
        "full",
        "fixed_open",
        "calibrated_mean",
        "causal_running_mean",
        "one_step_lagged",
    )
    representations: tuple = (
        "feedback_state",
        "ca1_gate_pattern",
        "pre_ca1_state",
        "output_state",
    )

    # Six mutually independent fixed streams. Calibration never reuses probe
    # train/validation/test or behavior trials and never reads their labels.
    split_seed_version: int = 62
    behavior_batches: int = 30
    behavior_batch_size: int = 64
    calibration_batches: int = 10
    calibration_batch_size: int = 64
    probe_train_batches: int = 20
    probe_val_batches: int = 5
    probe_test_batches: int = 20
    probe_batch_size: int = 64

    probe_epochs: int = 200
    probe_lr: float = 1e-2
    probe_weight_decay: float = 1e-4

    checkpoint_name_template: str = (
        "fourclass_rapid_acquisition_v4_primary_150_seed_{seed}.pt"
    )
    checkpoint_dir: str = "paper_rapid_v4_outputs/checkpoints"
    output_dir: str = "paper_ca1_v6_2_outputs"
    output_json: str = "ca1_causal_validation_v6_2_results.json"


def calibrate_ca1_unit_mean(model, calibration_bank):
    """
    Estimate one global gain per hidden unit from an independent calibration
    bank. The vector is shared by all times and all later trials. No label,
    phase, cue window, validation trial, or test trial is used.
    """
    total = None
    count = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in calibration_bank:
            x, a = batch[0].to(device), batch[1].to(device)
            out = model(x, a, return_details=True, skip_readout=True)
            gates = out["learned_ca1_hist"]
            batch_sum = gates.sum(dim=(0, 1))
            total = batch_sum if total is None else total + batch_sum
            count += int(gates.shape[0] * gates.shape[1])
    model.train(was_training)
    if total is None or count == 0:
        raise ValueError("Calibration bank is empty.")
    return (total / float(count)).detach().cpu()


def make_v62_schedule(reference_gates, condition, calibrated_unit_mean):
    """Every V6.2 schedule is test-trial causal or trial-independent."""
    if condition == "full":
        return None
    if condition == "fixed_open":
        return torch.ones_like(reference_gates)

    calibration = calibrated_unit_mean.to(
        device=reference_gates.device, dtype=reference_gates.dtype
    ).view(1, 1, -1)
    if calibration.shape[-1] != reference_gates.shape[-1]:
        raise ValueError(
            "Calibrated CA1 vector and checkpoint hidden dimension disagree: "
            f"{calibration.shape[-1]} vs {reference_gates.shape[-1]}"
        )
    if condition == "calibrated_mean":
        return calibration.expand_as(reference_gates).clone()
    if condition == "causal_running_mean":
        denominators = torch.arange(
            1, reference_gates.shape[1] + 1,
            device=reference_gates.device,
            dtype=reference_gates.dtype,
        ).view(1, -1, 1)
        return reference_gates.cumsum(dim=1) / denominators
    if condition == "one_step_lagged":
        schedule = torch.empty_like(reference_gates)
        schedule[:, :1] = calibration.expand(
            reference_gates.shape[0], 1, reference_gates.shape[2]
        )
        schedule[:, 1:] = reference_gates[:, :-1]
        return schedule
    raise ValueError(f"Unknown V6.2 CA1 condition: {condition}")


def validate_v62_schedule_definitions():
    """Fail fast if a future edit breaks the locked causal controls."""
    reference = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3)
    calibration = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float32)

    calibrated = make_v62_schedule(
        reference, "calibrated_mean", calibration
    )
    expected_calibrated = calibration.view(1, 1, -1).expand_as(reference)
    if not torch.equal(calibrated, expected_calibrated):
        raise AssertionError("calibrated_mean is not a trial-independent constant.")

    running = make_v62_schedule(
        reference, "causal_running_mean", calibration
    )
    for t in range(reference.shape[1]):
        expected = reference[:, :t + 1].mean(dim=1)
        if not torch.allclose(running[:, t], expected):
            raise AssertionError("causal_running_mean used a non-prefix gate.")

    lagged = make_v62_schedule(reference, "one_step_lagged", calibration)
    if not torch.equal(lagged[:, 0], calibration.view(1, -1).expand(2, -1)):
        raise AssertionError("one_step_lagged has an invalid initial value.")
    if not torch.equal(lagged[:, 1:], reference[:, :-1]):
        raise AssertionError("one_step_lagged is not an exact one-step delay.")


def forward_v62_condition(
    model, x, a, condition, calibrated_unit_mean,
    cueA_zones=None, cueB_zones=None, decision_positions=None,
    with_readout=False,
):
    """
    Interventions are open-loop replays of a causal endogenous reference.
    Running-mean at time t uses gates 0..t only; lagged at t uses t-1 only.
    The calibrated condition is constant across all test trials and times.
    """
    if condition == "full":
        return model(
            x, a,
            cueA_zones=cueA_zones if with_readout else None,
            cueB_zones=cueB_zones if with_readout else None,
            decision_positions=decision_positions if with_readout else None,
            return_details=True,
            skip_readout=not with_readout,
        )

    # This reference is causal: each learned gate at t depends only on inputs
    # through t. No permutation or full-trial test mean is ever constructed.
    reference = model(x, a, return_details=True, skip_readout=True)
    schedule = make_v62_schedule(
        reference["learned_ca1_hist"], condition, calibrated_unit_mean
    )
    return model(
        x, a,
        cueA_zones=cueA_zones if with_readout else None,
        cueB_zones=cueB_zones if with_readout else None,
        decision_positions=decision_positions if with_readout else None,
        return_details=True,
        skip_readout=not with_readout,
        ca1_schedule=schedule,
    )


def extract_v62_features(
    model, bank, representation, condition, calibrated_unit_mean,
):
    features, cue_as, cue_bs, labels = [], [], [], []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in bank:
            x, a, pos, phase, y, cueA, cueB, cueA_zones, cueB_zones, decision_positions, delay1s, delay2s = batch
            out = forward_v62_condition(
                model, x.to(device), a.to(device), condition,
                calibrated_unit_mean=calibrated_unit_mean,
                with_readout=False,
            )
            history = v6_representation_history(out, representation)
            indices = torch.tensor(
                [max(0, min(int(d) - 1, history.shape[1] - 1))
                 for d in decision_positions],
                dtype=torch.long, device=device,
            )
            rows = torch.arange(history.shape[0], device=device)
            features.append(history[rows, indices].detach().cpu())
            cue_as.append(cueA.detach().cpu())
            cue_bs.append(cueB.detach().cpu())
            labels.append(y.detach().cpu())
    model.train(was_training)
    return {
        "x": torch.cat(features, dim=0),
        "cueA": torch.cat(cue_as, dim=0),
        "cueB": torch.cat(cue_bs, dim=0),
        "class": torch.cat(labels, dim=0),
    }


def run_v62_probe(
    model, probe_banks, representation, condition,
    calibrated_unit_mean, analysis_cfg, probe_seed,
):
    datasets = {
        split: extract_v62_features(
            model, probe_banks[split], representation, condition,
            calibrated_unit_mean,
        )
        for split in ("train", "val", "test")
    }
    result = fit_v6_probe(
        datasets["train"], datasets["val"], datasets["test"],
        analysis_cfg, probe_seed,
    )
    return result, datasets["test"]


def evaluate_v62_behavior(
    model, bank, condition, calibrated_unit_mean,
):
    accuracies = []
    confusion = np.zeros((4, 4), dtype=np.int64)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in bank:
            x, a, pos, phase, y, cueA, cueB, cueA_zones, cueB_zones, decision_positions, delay1s, delay2s = batch
            out = forward_v62_condition(
                model, x.to(device), a.to(device), condition,
                calibrated_unit_mean=calibrated_unit_mean,
                cueA_zones=cueA_zones,
                cueB_zones=cueB_zones,
                decision_positions=decision_positions,
                with_readout=True,
            )
            logits = out["logits"]
            targets = y.to(device)
            accuracies.append(compute_accuracy(logits, targets))
            confusion += compute_confusion_matrix(logits, targets, 4)
    model.train(was_training)
    return {
        "accuracy": float(np.mean(accuracies)),
        "batch_std": float(np.std(accuracies)),
        "confusion_matrix": confusion.tolist(),
    }


def resolve_v62_checkpoint(analysis_cfg, seed):
    directory = analysis_cfg.checkpoint_dir
    if not os.path.isabs(directory):
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)
    path = os.path.join(
        directory, analysis_cfg.checkpoint_name_template.format(seed=seed)
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing V4 checkpoint for seed {seed}: {path}\n"
            "Place V6.2 beside paper_rapid_v4_outputs or edit checkpoint_dir. "
            "V6.2 must not silently retrain the source model."
        )
    return path


def run_v62_seed(seed, analysis_cfg):
    checkpoint_path = resolve_v62_checkpoint(analysis_cfg, seed)
    payload = load_torch_checkpoint(checkpoint_path)
    source_cfg = experiment_config_from_payload(payload)
    env = environment_from_checkpoint(payload, source_cfg, seed)
    behavior_banks, calibration_bank, probe_banks, split_seeds = build_v62_banks(
        env, analysis_cfg, seed
    )
    _, model = instantiate_v4_models(env, source_cfg)
    snapshots = payload.get("snapshots", {})
    checkpoint = int(analysis_cfg.primary_checkpoint)
    if checkpoint not in snapshots and str(checkpoint) not in snapshots:
        raise KeyError(
            f"Checkpoint {checkpoint} absent. Available: {sorted(snapshots.keys())}"
        )
    snapshot = snapshots.get(checkpoint, snapshots.get(str(checkpoint)))
    state = snapshot.get("hdgm", snapshot.get("gate"))
    if state is None:
        raise KeyError(f"Checkpoint {checkpoint} has no hdgm state_dict.")
    model.load_state_dict(state, strict=True)
    model.eval()
    checkpoint_fingerprint = state_dict_sha256(state)

    print("\n" + "=" * 120)
    print(f"V6.2 CAUSAL CA1 VALIDATION | SEED={seed} | UPDATE={checkpoint}")
    print(f"CHECKPOINT={checkpoint_path}")
    print("The source HDGM is frozen; calibration uses a separate unlabeled fixed bank.")
    print("=" * 120)

    calibrated_unit_mean = calibrate_ca1_unit_mean(
        model, calibration_bank
    )
    record = {
        "seed": int(seed),
        "source_checkpoint": checkpoint_path,
        "source_checkpoint_audit": {
            "checkpoint_update": checkpoint,
            "source_run_max_updates": int(payload.get("max_updates", -1)),
            "source_config_name": str(payload.get("config", {}).get("name", "unknown")),
            "hdgm_state_dict_sha256": checkpoint_fingerprint,
            "strict_state_dict_load": True,
        },
        "fresh_split_seeds": split_seeds,
        "calibration": {
            "source": "independent_calibration_observations_only",
            "uses_labels": False,
            "uses_phase_or_event_masks": False,
            "uses_validation_or_test_trials": False,
            "reuses_probe_train_trials": False,
            "unit_mean_mean": float(calibrated_unit_mean.mean()),
            "unit_mean_std": float(calibrated_unit_mean.std(unbiased=False)),
        },
        "protocol": {
            "analysis_status": "corrective_locked_before_v6_2_rerun",
            "source_model": "V4 exact update-100 checkpoint",
            "recurrent_parameters_updated": False,
            "full_trial_test_mean_forbidden": True,
            "future_gate_values_used_at_earlier_times": False,
            "probe_train_val_test_disjoint": True,
            "calibration_independent_of_probe_train_val_test_and_behavior": True,
            "cue_windows_enter_recurrent_dynamics": False,
            "interventions_share_identical_frozen_weights": True,
            "source_training_uses_classification_loss": True,
            "source_training_uses_cueB_margin_regularizer": bool(source_cfg.use_cueB_margin_loss),
            "source_training_uses_ec_phase_rank_regularizer": bool(source_cfg.use_ec_phase_rank_loss),
            "phase_or_labels_enter_CA1_intervention_schedule": False,
            "corrective_primary_family": (
                "output_probe_full_vs_each_CA1_control_Holm"
            ),
        },
        "full_representations": {},
        "conditions": {},
    }

    full_data = {}
    shared_probe_seed = int(seed * 1_000_000 + checkpoint * 100 + 62)
    print("Running full-model decision-pre representation probes ...")
    for representation in analysis_cfg.representations:
        probe, test_data = run_v62_probe(
            model, probe_banks, representation, "full",
            calibrated_unit_mean, analysis_cfg, shared_probe_seed,
        )
        record["full_representations"][representation] = probe
        full_data[representation] = test_data
        print(
            f"  {representation:19s} A/B/class="
            f"{probe['cueA']:.3f}/{probe['cueB']:.3f}/{probe['class']:.3f}"
        )

    print("Running same-weight causal/calibrated CA1 controls ...")
    for condition in analysis_cfg.conditions:
        if condition == "full":
            output_probe = record["full_representations"]["output_state"]
            output_data = full_data["output_state"]
            pre_data = full_data["pre_ca1_state"]
            feedback_data = full_data["feedback_state"]
        else:
            output_probe, output_data = run_v62_probe(
                model, probe_banks, "output_state", condition,
                calibrated_unit_mean, analysis_cfg, shared_probe_seed,
            )
            pre_data = extract_v62_features(
                model, probe_banks["test"], "pre_ca1_state", condition,
                calibrated_unit_mean,
            )
            feedback_data = extract_v62_features(
                model, probe_banks["test"], "feedback_state", condition,
                calibrated_unit_mean,
            )
        geometry = paired_feature_metrics(
            output_data, pre_data, feedback_data
        )
        normal = evaluate_v62_behavior(
            model, behavior_banks["normal"], condition, calibrated_unit_mean
        )
        long_delay = evaluate_v62_behavior(
            model, behavior_banks["long"], condition, calibrated_unit_mean
        )
        record["conditions"][condition] = {
            "output_probe": output_probe,
            "geometry": geometry,
            "behavior_normal": normal,
            "behavior_long": long_delay,
        }
        print(
            f"  {condition:20s} probe={output_probe['class']:.3f} "
            f"geom={geometry['output_geometry_ratio']:.3f} "
            f"normal/long={normal['accuracy']:.3f}/{long_delay['accuracy']:.3f}"
        )
    return record


def compute_v62_statistics(records, analysis_cfg):
    controls = [c for c in analysis_cfg.conditions if c != "full"]

    def values(condition, section, key):
        return [r["conditions"][condition][section][key] for r in records]

    families = {}
    for family_name, section, key in (
        ("corrective_output_probe", "output_probe", "class"),
        ("secondary_output_geometry", "geometry", "output_geometry_ratio"),
        ("secondary_behavior_normal", "behavior_normal", "accuracy"),
        ("secondary_behavior_long", "behavior_long", "accuracy"),
    ):
        tests = [
            exact_paired_sign_flip(
                values("full", section, key), values(control, section, key)
            )
            for control in controls
        ]
        adjusted = holm_adjust([test["p_value"] for test in tests])
        families[family_name] = {
            f"full_minus_{control}": {**test, "holm_p": adjusted_p}
            for control, test, adjusted_p in zip(controls, tests, adjusted)
        }

    # Descriptive integration family: quantify whether the post-CA1 output
    # changes linearly accessible joint-class information relative to each
    # incoming source. Results are reported regardless of direction.
    output_values = [
        r["full_representations"]["output_state"]["class"] for r in records
    ]
    integration_sources = ("pre_ca1_state", "feedback_state")
    integration_tests = [
        exact_paired_sign_flip(
            output_values,
            [r["full_representations"][source]["class"] for r in records],
        )
        for source in integration_sources
    ]
    integration_adjusted = holm_adjust([
        test["p_value"] for test in integration_tests
    ])
    families["secondary_CA1_output_integration"] = {
        f"output_minus_{source}": {**test, "holm_p": adjusted_p}
        for source, test, adjusted_p in zip(
            integration_sources, integration_tests, integration_adjusted
        )
    }
    return families


def print_v62_summary(records, statistics, analysis_cfg):
    print("\n" + "#" * 120)
    print("HDGM V6.2 CAUSAL/CROSS-TRIAL-CALIBRATED CA1 VALIDATION SUMMARY")
    print("#" * 120)
    print("\n===== Full-model decision-pre representations at update 100 =====")
    for representation in analysis_cfg.representations:
        vals = [
            r["full_representations"][representation]["class"]
            for r in records
        ]
        mean, std = mean_std(vals)
        print(f"{representation:20s}: {mean:.4f} ± {std:.4f}")

    print("\n===== Same-weight CA1 conditions =====")
    for condition in analysis_cfg.conditions:
        probe = mean_std([
            r["conditions"][condition]["output_probe"]["class"]
            for r in records
        ])
        geometry = mean_std([
            r["conditions"][condition]["geometry"]["output_geometry_ratio"]
            for r in records
        ])
        normal = mean_std([
            r["conditions"][condition]["behavior_normal"]["accuracy"]
            for r in records
        ])
        long_delay = mean_std([
            r["conditions"][condition]["behavior_long"]["accuracy"]
            for r in records
        ])
        print(
            f"{condition:20s} | probe={probe[0]:.4f}±{probe[1]:.4f} "
            f"geometry={geometry[0]:.4f}±{geometry[1]:.4f} "
            f"normal={normal[0]:.4f}±{normal[1]:.4f} "
            f"long={long_delay[0]:.4f}±{long_delay[1]:.4f}"
        )

    for family_name, family in statistics.items():
        print(f"\n===== {family_name} =====")
        for contrast, result in family.items():
            print(
                f"{contrast:42s} diff={result['mean_diff']:.4f} "
                f"CI95=[{result['ci95_low']:.4f},{result['ci95_high']:.4f}] "
                f"exact-p={result['p_value']:.5f} "
                f"Holm-p={result['holm_p']:.5f} dz={result['cohend_z']:.4f}"
            )
    print("\nInterpretation rules:")
    print("  fixed_open tests adaptive gain versus saturated tonic gain.")
    print("  calibrated_mean tests trial/time-dependent gain versus a train-calibrated tonic profile.")
    print("  causal_running_mean tests fast fluctuations without using future gates.")
    print("  one_step_lagged tests temporal alignment using past gates only.")
    print("  output-minus-input probe contrasts quantify transformation, not biological information creation.")
    print("  Probe effects are representational; behavior effects must be reported separately.")


def plot_v62_results(records, analysis_cfg):
    output_dir = ensure_dir(analysis_cfg.output_dir)
    conditions = list(analysis_cfg.conditions)
    labels = [condition.replace("_", "\n") for condition in conditions]
    x = np.arange(len(conditions))

    endpoints = {
        "Output probe": ("output_probe", "class"),
        "Normal behavior": ("behavior_normal", "accuracy"),
        "Long-delay behavior": ("behavior_long", "accuracy"),
    }
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    for index, (name, (section, key)) in enumerate(endpoints.items()):
        values = np.asarray([
            [r["conditions"][condition][section][key] for condition in conditions]
            for r in records
        ], dtype=np.float64)
        ax.bar(
            x + (index - 1) * width,
            values.mean(axis=0), width,
            yerr=values.std(axis=0), capsize=3, label=name,
        )
    ax.axhline(0.25, color="#94A3B8", ls="--", lw=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Accuracy")
    ax.set_title("Causal and cross-trial-calibrated CA1 controls at update 100")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_v62_ca1_controls.png"), dpi=240)
    plt.close(fig)

    representations = list(analysis_cfg.representations)
    values = np.asarray([
        [r["full_representations"][representation]["class"]
         for representation in representations]
        for r in records
    ], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(
        np.arange(len(representations)), values.mean(axis=0),
        yerr=values.std(axis=0), capsize=4,
        color=["#2E5BFF", "#F59E0B", "#64748B", "#14B8A6"],
    )
    ax.axhline(0.25, color="#94A3B8", ls="--", lw=1.1)
    ax.set_xticks(np.arange(len(representations)))
    ax.set_xticklabels([r.replace("_", "\n") for r in representations])
    ax.set_ylim(0.20, 0.70)
    ax.set_ylabel("Frozen class-probe accuracy")
    ax.set_title("Decision-pre information across the feedback–CA1–output pathway")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_v62_representation_pathway.png"), dpi=240)
    plt.close(fig)


def run_v62_multiseed(seeds, analysis_cfg, generate_figures=True):
    print("\n" + "=" * 120)
    print("V6.2 CAUSAL/CROSS-TRIAL-CALIBRATED CA1 VALIDATION")
    print(f"SEEDS={seeds}")
    print("Only frozen V4 update-100 checkpoints are analyzed; no source model is trained.")
    print("=" * 120)
    records = [run_v62_seed(seed, analysis_cfg) for seed in seeds]
    statistics = compute_v62_statistics(records, analysis_cfg)
    print_v62_summary(records, statistics, analysis_cfg)
    payload = {
        "analysis_config": asdict(analysis_cfg),
        "records": records,
        "statistics": statistics,
    }
    output_dir = ensure_dir(analysis_cfg.output_dir)
    save_path = os.path.join(output_dir, analysis_cfg.output_json)
    with open(save_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"\nSaved V6.2 results to: {save_path}")
    if generate_figures:
        plot_v62_results(records, analysis_cfg)
        print(f"Saved V6.2 figures to: {output_dir}")
    return records, statistics


# =========================================================
# 21. Main
# =========================================================
if __name__ == "__main__":
    validate_v62_schedule_definitions()
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    analysis_cfg = CA1CausalAnalysisConfig(
        primary_checkpoint=100,
        split_seed_version=62,
        behavior_batches=30,
        behavior_batch_size=64,
        calibration_batches=10,
        calibration_batch_size=64,
        probe_train_batches=20,
        probe_val_batches=5,
        probe_test_batches=20,
        probe_batch_size=64,
        probe_epochs=200,
        probe_lr=1e-2,
        probe_weight_decay=1e-4,
        checkpoint_dir=os.path.join(
            SCRIPT_DIR, "paper_rapid_v4_outputs", "checkpoints"
        ),
        output_dir=os.path.join(SCRIPT_DIR, "paper_ca1_v6_2_outputs"),
        output_json="ca1_causal_validation_v6_2_results.json",
    )

    PILOT_SEEDS = [11, 22, 33]
    FINAL_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]

    # Use True only for path/runtime verification. Paper output uses 10 seeds.
    RUN_PILOT_ONLY = False
    selected_seeds = PILOT_SEEDS if RUN_PILOT_ONLY else FINAL_SEEDS
    run_v62_multiseed(selected_seeds, analysis_cfg, generate_figures=True)
