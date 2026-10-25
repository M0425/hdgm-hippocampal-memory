# -*- coding: utf-8 -*-
"""
HDGM Difficulty V2.1: frozen population-robustness scan built on the complete
V7.3 splitter model with locked dual endpoints.

The first section retains the full V7.3 task, TEM, HDGM, training, auditing,
probe, checkpoint, statistics, and plotting implementation. The final section
loads update-150 and update-300 checkpoints and performs the difficulty scan
without changing any TEM/HDGM parameter.

Scientific safeguards
---------------------
1. No branch label or cue-window mask enters TEM/HDGM forward dynamics.
2. Branch supervision is used only at the pre-action junction state. The
   left/right action token occurs one step later, so it cannot reveal the label.
3. Shared-path activity is analysed only after training with frozen linear
   probes; no loss is applied inside the analysed mid-corridor window.
4. Every trial starts with a balanced random branch and then alternates. Thus
   absolute position, trip number, or elapsed time alone cannot predict branch.
5. Probe train/validation/test sets are independently generated full trials.
6. A true history-reset control clears recurrent state and associative memory
   at every trip boundary. Observation-only, timing-only, and group-wise
   label-permutation controls are also reported.
7. The full canonical TEM/HDGM modules are retained, including every state
   history and dormant diagnostic head used by the four-class paper model.
8. Core and task-readout learning rates are separated without modifying the
   canonical HDGM equations. Branch labels still occur only at the pre-action
   junction state.
9. Training budgets and checkpoints are fixed in advance. Update 150 is the
   representation endpoint and update 300 is the behavioral endpoint. The
   complete 25--300 acquisition curve is retained as a secondary analysis.
10. The old seeds 11/22/33 are development only. A separate list of new seeds
    is provided for the locked confirmatory run.

The biological claim supported by this experiment is deliberately limited to
"trajectory-dependent / splitter-like population separation". It does not
claim literal single-neuron identity or superiority on a generic ML benchmark.
"""

import copy
import glob
import hashlib
import itertools
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


@dataclass
class SplitterV7Config:
    # Task
    corridor_len: int = 18
    arm_len: int = 7
    trips_per_trial: int = 4
    endpoint_hold_len: int = 2
    alias_start: int = 6
    alias_end: int = 12                 # Python-exclusive
    obs_dim: int = 24
    noise_std: float = 0.05
    alias_strength: float = 1.0

    # Model: kept consistent with the original splitter implementation
    hidden_dim: int = 96
    # Same structural dimensions as the locked four-class paper model.
    action_emb_dim: int = 32
    ca3_steps: int = 5
    lambda_m: float = 0.92
    alpha: float = 0.32
    rho: float = 0.84
    c1: float = 0.20
    c2: float = 1.00
    beta: float = 0.10

    # Locked rapid-acquisition protocol
    checkpoints: Tuple[int, ...] = (25, 50, 100, 150, 200, 300)
    representation_update: int = 150
    behavior_update: int = 300
    batch_size: int = 16
    core_lr: float = 6e-4
    choice_head_lr: float = 2e-3
    stateless_lr: float = 2e-3
    weight_decay: float = 0.0
    prediction_weight: float = 0.20
    choice_weight: float = 1.50
    grad_clip: float = 1.0

    # Fixed banks
    val_batches: int = 4
    test_batches: int = 8
    bank_batch_size: int = 32
    probe_train_batches: int = 8
    probe_val_batches: int = 4
    probe_test_batches: int = 8

    # Frozen probes
    probe_epochs: int = 160
    probe_lr: float = 1e-2
    probe_weight_decay: float = 1e-4
    probe_patience: int = 25
    permutation_repeats: int = 10
    permutation_probe_epochs: int = 60

    # Output
    name: str = "hdgm_v7_3_splitter_dual_endpoint"
    output_dir: str = "hdgm_v7_3_splitter_outputs"


class CounterbalancedBifurcationEnv:
    """Fixed-length shared corridor -> branch arm -> return trajectories."""

    def __init__(self, cfg: SplitterV7Config, seed: int = 0,
                 maps: Optional[Dict[str, np.ndarray]] = None):
        self.cfg = cfg
        self.rng = np.random.RandomState(seed)
        if maps is None:
            corridor = self.rng.randn(cfg.corridor_len, cfg.obs_dim).astype(np.float32)
            if cfg.alias_end > cfg.alias_start:
                proto = self.rng.randn(cfg.obs_dim).astype(np.float32)
                for p in range(cfg.alias_start, cfg.alias_end):
                    corridor[p] = (
                        cfg.alias_strength * proto
                        + (1.0 - cfg.alias_strength) * corridor[p]
                    )
            self.maps = {
                "corridor": corridor,
                "left": self.rng.randn(cfg.arm_len, cfg.obs_dim).astype(np.float32),
                "right": self.rng.randn(cfg.arm_len, cfg.obs_dim).astype(np.float32),
            }
        else:
            self.maps = {k: np.asarray(v, dtype=np.float32).copy() for k, v in maps.items()}
        self.action_dim = 7

    def clone(self, seed: int) -> "CounterbalancedBifurcationEnv":
        return CounterbalancedBifurcationEnv(self.cfg, seed=seed, maps=self.maps)

    def _obs(self, kind: str, pos: int) -> np.ndarray:
        x = self.maps[kind][pos].copy()
        x += self.rng.randn(self.cfg.obs_dim).astype(np.float32) * self.cfg.noise_std
        return x

    def _append(self, rows: List[dict], kind: str, pos: int, action: int,
                branch: int, trip: int, direction: int, choice: bool = False,
                alias: bool = False, commit: bool = False) -> None:
        rows.append({
            "kind": kind, "pos": pos, "action": action, "branch": branch,
            "trip": trip, "direction": direction, "choice": int(choice),
            "alias": int(alias), "commit": int(commit),
        })

    def build_trial(self, initial_branch: int) -> Tuple[np.ndarray, ...]:
        c = self.cfg
        rows: List[dict] = []
        for trip in range(c.trips_per_trial):
            branch = int(initial_branch ^ (trip & 1))

            # The state at corridor_len-1 is supervised only for trip > 0.
            # Its action is still neutral (0); the branch-coded action is a
            # separate next token and therefore cannot leak into this state.
            for p in range(c.corridor_len):
                self._append(
                    rows, "corridor", p, 0, branch, trip, 0,
                    choice=(trip > 0 and p == c.corridor_len - 1),
                    alias=(trip > 0 and c.alias_start <= p < c.alias_end),
                )

            self._append(rows, "corridor", c.corridor_len - 1,
                         2 if branch == 0 else 3, branch, trip, 0, commit=True)

            arm_kind = "left" if branch == 0 else "right"
            for p in range(c.arm_len):
                self._append(rows, arm_kind, p, 4, branch, trip, -1)
            for _ in range(c.endpoint_hold_len):
                self._append(rows, arm_kind, c.arm_len - 1, 6, branch, trip, -1)
            for p in range(c.arm_len - 1, -1, -1):
                self._append(rows, arm_kind, p, 5, branch, trip, -1)
            for p in range(c.corridor_len - 1, -1, -1):
                self._append(rows, "corridor", p, 1, branch, trip, 1)

        x = np.stack([self._obs(r["kind"], r["pos"]) for r in rows])
        action = np.asarray([r["action"] for r in rows], dtype=np.int64)
        branch = np.asarray([r["branch"] for r in rows], dtype=np.int64)
        trip = np.asarray([r["trip"] for r in rows], dtype=np.int64)
        pos = np.asarray([r["pos"] if r["kind"] == "corridor" else -1 for r in rows], dtype=np.int64)
        direction = np.asarray([r["direction"] for r in rows], dtype=np.int64)
        choice = np.asarray([r["choice"] for r in rows], dtype=np.bool_)
        alias = np.asarray([r["alias"] for r in rows], dtype=np.bool_)
        commit = np.asarray([r["commit"] for r in rows], dtype=np.bool_)
        reset = np.zeros(len(rows), dtype=np.bool_)
        reset[0] = True
        for t in range(1, len(rows)):
            reset[t] = trip[t] != trip[t - 1]
        return x, action, branch, trip, pos, direction, choice, alias, commit, reset

    def sample_batch(self, batch_size: int, balanced: bool = True) -> Dict[str, torch.Tensor]:
        if balanced:
            initial = np.arange(batch_size, dtype=np.int64) & 1
            self.rng.shuffle(initial)
        else:
            initial = self.rng.randint(0, 2, size=batch_size)
        fields = [[] for _ in range(10)]
        for b in initial:
            trial = self.build_trial(int(b))
            for bucket, value in zip(fields, trial):
                bucket.append(value)
        names = ["x", "action", "branch", "trip", "pos", "direction",
                 "choice_mask", "alias_mask", "commit_mask", "reset_mask"]
        dtypes = [torch.float32, torch.long, torch.long, torch.long, torch.long,
                  torch.long, torch.bool, torch.bool, torch.bool, torch.bool]
        return {name: torch.as_tensor(np.stack(values), dtype=dtype)
                for name, values, dtype in zip(names, fields, dtypes)}


def build_bank(env: CounterbalancedBifurcationEnv, n_batches: int,
               batch_size: int) -> List[Dict[str, torch.Tensor]]:
    return [env.sample_batch(batch_size, balanced=True) for _ in range(n_batches)]


def to_device(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.to(DEVICE) for k, v in batch.items()}


def attractor(M: torch.Tensor, h: torch.Tensor, steps: int = 5) -> torch.Tensor:
    for _ in range(steps):
        h = torch.tanh(0.5 * h + torch.bmm(M, h.unsqueeze(2)).squeeze(2))
    return h


class TokenChoiceHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, t, d = h.shape
        return self.net(h.reshape(b * t, d)).reshape(b, t, 2)


def apply_row_reset(state: torch.Tensor, reset: Optional[torch.Tensor]) -> torch.Tensor:
    if reset is None:
        return state
    shape = [reset.shape[0]] + [1] * (state.ndim - 1)
    return state * (~reset).to(state.dtype).reshape(shape)


class TEMCore(nn.Module):
    def __init__(self, cfg: SplitterV7Config):
        super().__init__()
        h = cfg.hidden_dim
        self.cfg = cfg
        self.fc_x = nn.Linear(cfg.obs_dim, h)
        self.action_emb = nn.Embedding(7, cfg.action_emb_dim)
        self.fc_a = nn.Linear(cfg.action_emb_dim, h)
        self.fc_g = nn.Linear(h, h)
        self.decoder = nn.Linear(h, cfg.obs_dim)
        # Canonical diagnostic heads. They are deliberately not optimized by
        # an auxiliary target in V7.1, but are retained for state-dict and
        # architecture completeness relative to the paper model.
        self.pos_head = nn.Linear(h, cfg.corridor_len)
        self.phase_head = nn.Linear(h, 6)

    def forward(self, x: torch.Tensor, action: torch.Tensor,
                reset_mask: Optional[torch.Tensor] = None,
                return_details: bool = False) -> Dict[str, torch.Tensor]:
        b, tmax, _ = x.shape
        hdim = self.cfg.hidden_dim
        g = x.new_zeros(b, hdim)
        M = x.new_zeros(b, hdim, hdim)
        pred_loss = x.new_zeros(())
        mem_hist, pred_hist, attr_hist, read_hist = [], [], [], []
        xhat_hist, g_hist, pos_logits_hist, phase_logits_hist = [], [], [], []
        for t in range(tmax - 1):
            r = None if reset_mask is None else reset_mask[:, t]
            g, M = apply_row_reset(g, r), apply_row_reset(M, r)
            x_emb = torch.tanh(self.fc_x(x[:, t]))
            a_emb = self.fc_a(self.action_emb(action[:, t]))
            g_pred = torch.tanh(self.fc_g(g) + a_emb)
            p_t = attractor(M, x_emb, self.cfg.ca3_steps)
            g = g_pred + self.cfg.alpha * p_t
            p_hat = torch.tanh(0.5 * g + torch.bmm(M, g.unsqueeze(2)).squeeze(2))
            x_hat = self.decoder(p_hat)
            pos_logits = self.pos_head(p_hat)
            phase_logits = self.phase_head(p_hat)
            pred_loss = pred_loss + (x_hat - x[:, t + 1]).square().mean()
            M = self.cfg.lambda_m * M + torch.bmm(p_t.unsqueeze(2), p_t.unsqueeze(1))
            mem_hist.append(p_hat)
            pred_hist.append(g_pred)
            attr_hist.append(p_t)
            read_hist.append(p_hat)
            xhat_hist.append(x_hat)
            g_hist.append(g)
            pos_logits_hist.append(pos_logits)
            phase_logits_hist.append(phase_logits)
        out = {
            "pred_loss": pred_loss / max(1, tmax - 1),
            "mem": torch.stack(mem_hist, 1),
            "predictive_state": torch.stack(pred_hist, 1),
            "attractor_state": torch.stack(attr_hist, 1),
            "associative_read": torch.stack(read_hist, 1),
            "xhat": torch.stack(xhat_hist, 1),
            "g_hist": torch.stack(g_hist, 1),
            "pos_logits_hist": torch.stack(pos_logits_hist, 1),
            "phase_logits_hist": torch.stack(phase_logits_hist, 1),
        }
        return out


class HDGMCore(nn.Module):
    def __init__(self, cfg: SplitterV7Config):
        super().__init__()
        h = cfg.hidden_dim
        self.cfg = cfg
        self.fc_x = nn.Linear(cfg.obs_dim, h)
        self.action_emb = nn.Embedding(7, cfg.action_emb_dim)
        self.fc_a = nn.Linear(cfg.action_emb_dim, h)
        self.fc_g = nn.Linear(h, h)
        self.decoder = nn.Linear(h, cfg.obs_dim)
        self.gate_ec = nn.Linear(cfg.obs_dim + h + h, cfg.obs_dim)
        self.gate_ca3 = nn.Linear(h * 2, h)
        self.gate_ca1 = nn.Linear(h * 3, h)
        self.fb_proj = nn.Linear(h, h)
        self.pos_head = nn.Linear(h, cfg.corridor_len)
        self.phase_head = nn.Linear(h, 6)

    def forward(self, x: torch.Tensor, action: torch.Tensor,
                reset_mask: Optional[torch.Tensor] = None,
                return_details: bool = False) -> Dict[str, torch.Tensor]:
        b, tmax, _ = x.shape
        hdim = self.cfg.hidden_dim
        g = x.new_zeros(b, hdim)
        M = x.new_zeros(b, hdim, hdim)
        v = x.new_zeros(b, hdim)
        pred_loss = x.new_zeros(())
        mem_hist, pred_hist, attr_hist, read_hist, fb_hist = [], [], [], [], []
        ec_hist, ca3_hist, ca1_hist, write_hist = [], [], [], []
        xhat_hist, g_hist, pos_logits_hist, phase_logits_hist = [], [], [], []
        for t in range(tmax - 1):
            r = None if reset_mask is None else reset_mask[:, t]
            g, M, v = apply_row_reset(g, r), apply_row_reset(M, r), apply_row_reset(v, r)
            ec = torch.sigmoid(self.gate_ec(torch.cat([x[:, t], g, v], 1)))
            x_emb = torch.tanh(self.fc_x(ec * x[:, t]))
            a_emb = self.fc_a(self.action_emb(action[:, t]))
            g_pred = torch.tanh(self.fc_g(g) + a_emb)
            hstate = x_emb
            ca3_steps = []
            for _ in range(self.cfg.ca3_steps):
                ca3 = torch.sigmoid(self.gate_ca3(torch.cat([hstate, g], 1)))
                recurrent = torch.tanh(
                    0.5 * hstate + torch.bmm(M, hstate.unsqueeze(2)).squeeze(2)
                )
                hstate = (1.0 - ca3) * hstate + ca3 * recurrent
                ca3_steps.append(ca3)
            p_t = hstate
            g = g_pred + self.cfg.alpha * p_t
            p_hat = torch.tanh(0.5 * g + torch.bmm(M, g.unsqueeze(2)).squeeze(2))
            ca1 = torch.sigmoid(self.gate_ca1(torch.cat([g, p_hat, v], 1)))
            p_out = torch.relu(p_hat * (self.cfg.c1 + self.cfg.c2 * ca1) - self.cfg.beta)
            x_hat = self.decoder(p_out)
            pos_logits = self.pos_head(p_out)
            phase_logits = self.phase_head(p_out)
            pred_loss = pred_loss + (x_hat - x[:, t + 1]).square().mean()
            v = self.cfg.rho * v + self.fb_proj(p_out)
            write = ec.mean(1, keepdim=True).unsqueeze(-1)
            M = self.cfg.lambda_m * M + write * torch.bmm(p_t.unsqueeze(2), p_t.unsqueeze(1))
            mem_hist.append(p_out)
            pred_hist.append(g_pred)
            attr_hist.append(p_t)
            read_hist.append(p_hat)
            fb_hist.append(v)
            ec_hist.append(ec)
            ca3_hist.append(torch.stack(ca3_steps, 1).mean(1))
            ca1_hist.append(ca1)
            write_hist.append(write.squeeze(-1).squeeze(-1))
            xhat_hist.append(x_hat)
            g_hist.append(g)
            pos_logits_hist.append(pos_logits)
            phase_logits_hist.append(phase_logits)
        out = {
            "pred_loss": pred_loss / max(1, tmax - 1),
            "mem": torch.stack(mem_hist, 1),
            "predictive_state": torch.stack(pred_hist, 1),
            "attractor_state": torch.stack(attr_hist, 1),
            "associative_read": torch.stack(read_hist, 1),
            "feedback_state": torch.stack(fb_hist, 1),
            "ec_gate_pattern": torch.stack(ec_hist, 1),
            "ca3_gate_pattern": torch.stack(ca3_hist, 1),
            "ca1_gate_pattern": torch.stack(ca1_hist, 1),
            # Canonical paper-model aliases retained verbatim in spirit.
            "xhat": torch.stack(xhat_hist, 1),
            "g_hist": torch.stack(g_hist, 1),
            "ec_hist": torch.stack(ec_hist, 1),
            "ca3_hist": torch.stack(ca3_hist, 1),
            "ca1_hist": torch.stack(ca1_hist, 1),
            "write_hist": torch.stack(write_hist, 1),
            "v_hist": torch.stack(fb_hist, 1),
            "pos_logits_hist": torch.stack(pos_logits_hist, 1),
            "phase_logits_hist": torch.stack(phase_logits_hist, 1),
        }
        return out


class SequenceChoiceModel(nn.Module):
    def __init__(self, core: nn.Module, hidden_dim: int):
        super().__init__()
        self.core = core
        self.choice_head = TokenChoiceHead(hidden_dim)

    def forward(self, x: torch.Tensor, action: torch.Tensor,
                reset_mask: Optional[torch.Tensor] = None,
                return_details: bool = False) -> Dict[str, torch.Tensor]:
        out = self.core(x, action, reset_mask=reset_mask, return_details=True)
        out["choice_logits"] = self.choice_head(out["mem"])
        return out


class ObservationOnlyChoice(nn.Module):
    """Negative control: current observation/action only, no recurrent history."""
    def __init__(self, cfg: SplitterV7Config):
        super().__init__()
        self.action_emb = nn.Embedding(7, cfg.action_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(cfg.obs_dim + cfg.action_emb_dim, cfg.hidden_dim), nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU(), nn.Linear(cfg.hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x[:, :-1], self.action_emb(action[:, :-1])], -1)
        return self.net(z)


def masked_ce(logits: torch.Tensor, labels: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    if int(mask.sum()) == 0:
        return logits.sum() * 0.0
    return nn.functional.cross_entropy(logits[mask], labels[mask])


def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor,
                    mask: torch.Tensor) -> float:
    mask = mask.bool()
    if int(mask.sum()) == 0:
        return float("nan")
    return float((logits.argmax(-1)[mask] == labels[mask]).float().mean().item())


def batch_loss(model: SequenceChoiceModel, batch: Dict[str, torch.Tensor],
               cfg: SplitterV7Config) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    out = model(batch["x"], batch["action"], return_details=True)
    labels = batch["branch"][:, :-1]
    mask = batch["choice_mask"][:, :-1]
    choice = masked_ce(out["choice_logits"], labels, mask)
    return cfg.prediction_weight * out["pred_loss"] + cfg.choice_weight * choice, out


@torch.no_grad()
def evaluate_choice(model, bank: Sequence[Dict[str, torch.Tensor]], kind: str) -> float:
    model.eval()
    correct = total = 0
    for cpu_batch in bank:
        b = to_device(cpu_batch)
        mask = b["choice_mask"][:, :-1]
        labels = b["branch"][:, :-1]
        if kind == "stateless":
            logits = model(b["x"], b["action"])
        else:
            logits = model(b["x"], b["action"], return_details=True)["choice_logits"]
        correct += int(((logits.argmax(-1) == labels) & mask).sum().item())
        total += int(mask.sum().item())
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_choice_by_trip(model, bank: Sequence[Dict[str, torch.Tensor]],
                            kind: str, cfg: SplitterV7Config) -> Dict[str, float]:
    model.eval()
    correct = {trip: 0 for trip in range(1, cfg.trips_per_trial)}
    total = {trip: 0 for trip in range(1, cfg.trips_per_trial)}
    for cpu_batch in bank:
        b = to_device(cpu_batch)
        labels = b["branch"][:, :-1]
        if kind == "stateless":
            logits = model(b["x"], b["action"])
        else:
            logits = model(b["x"], b["action"], return_details=True)["choice_logits"]
        pred = logits.argmax(-1)
        for trip in range(1, cfg.trips_per_trial):
            mask = b["choice_mask"][:, :-1] & (b["trip"][:, :-1] == trip)
            correct[trip] += int(((pred == labels) & mask).sum().item())
            total[trip] += int(mask.sum().item())
    return {str(trip): correct[trip] / max(total[trip], 1) for trip in correct}


def state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def effective_parameter_count(model: nn.Module) -> int:
    """Exclude retained-but-unsupervised diagnostic heads from task capacity."""
    return sum(
        p.numel() for name, p in model.named_parameters()
        if p.requires_grad and ".pos_head." not in name and ".phase_head." not in name
    )


def instantiate_models(cfg: SplitterV7Config) -> Dict[str, nn.Module]:
    return {
        "tem": SequenceChoiceModel(TEMCore(cfg), cfg.hidden_dim).to(DEVICE),
        "hdgm": SequenceChoiceModel(HDGMCore(cfg), cfg.hidden_dim).to(DEVICE),
        "stateless": ObservationOnlyChoice(cfg).to(DEVICE),
    }


def build_optimizer(name: str, model: nn.Module, cfg: SplitterV7Config):
    """Use a faster readout without changing or selectively favoring HDGM core dynamics."""
    if name in ("tem", "hdgm"):
        return optim.Adam(
            [
                {"params": model.core.parameters(), "lr": cfg.core_lr},
                {"params": model.choice_head.parameters(), "lr": cfg.choice_head_lr},
            ],
            weight_decay=cfg.weight_decay,
        )
    return optim.Adam(
        model.parameters(), lr=cfg.stateless_lr, weight_decay=cfg.weight_decay
    )


def architecture_contract(cfg: SplitterV7Config, models: Dict[str, nn.Module]) -> Dict[str, object]:
    """Fail loudly if the canonical paper HDGM dynamics were accidentally simplified."""
    hdgm = models["hdgm"].core
    tem = models["tem"].core
    h, d, ae = cfg.hidden_dim, cfg.obs_dim, cfg.action_emb_dim
    expected_hdgm = {
        "fc_x.weight": (h, d),
        "action_emb.weight": (7, ae),
        "fc_a.weight": (h, ae),
        "fc_g.weight": (h, h),
        "decoder.weight": (d, h),
        "gate_ec.weight": (d, d + h + h),
        "gate_ca3.weight": (h, h * 2),
        "gate_ca1.weight": (h, h * 3),
        "fb_proj.weight": (h, h),
        "pos_head.weight": (cfg.corridor_len, h),
        "phase_head.weight": (6, h),
    }
    expected_tem = {
        "fc_x.weight": (h, d),
        "action_emb.weight": (7, ae),
        "fc_a.weight": (h, ae),
        "fc_g.weight": (h, h),
        "decoder.weight": (d, h),
        "pos_head.weight": (cfg.corridor_len, h),
        "phase_head.weight": (6, h),
    }

    def check(module: nn.Module, expected: Dict[str, tuple], label: str) -> Dict[str, list]:
        actual = dict(module.named_parameters())
        errors = []
        shapes = {}
        for name, shape in expected.items():
            if name not in actual:
                errors.append(f"{label}: missing {name}")
                continue
            got = tuple(actual[name].shape)
            shapes[name] = list(got)
            if got != tuple(shape):
                errors.append(f"{label}: {name} expected {shape}, got {got}")
        if errors:
            raise RuntimeError("Architecture contract failed:\n" + "\n".join(errors))
        return shapes

    if cfg.hidden_dim != 96 or cfg.action_emb_dim != 32 or cfg.ca3_steps != 5:
        raise RuntimeError(
            "Paper contract requires hidden_dim=96, action_emb_dim=32, ca3_steps=5"
        )
    constants = {
        "lambda_m": cfg.lambda_m, "alpha": cfg.alpha, "rho": cfg.rho,
        "c1": cfg.c1, "c2": cfg.c2, "beta": cfg.beta,
    }
    expected_constants = {
        "lambda_m": .92, "alpha": .32, "rho": .84,
        "c1": .20, "c2": 1.0, "beta": .10,
    }
    for key, target in expected_constants.items():
        if not math.isclose(float(constants[key]), target, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Paper contract requires {key}={target}, got {constants[key]}")
    return {
        "status": "passed",
        "hdgm_parameter_shapes": check(hdgm, expected_hdgm, "HDGM"),
        "tem_parameter_shapes": check(tem, expected_tem, "TEM"),
        "dynamical_constants": constants,
        "hdgm_total_parameters": parameter_count(models["hdgm"]),
        "hdgm_effective_task_parameters": effective_parameter_count(models["hdgm"]),
        "tem_total_parameters": parameter_count(models["tem"]),
        "tem_effective_task_parameters": effective_parameter_count(models["tem"]),
        "allowed_task_specific_difference": "7-action input and binary pre-action choice head",
    }


def task_integrity_audit(batch: Dict[str, torch.Tensor], cfg: SplitterV7Config) -> Dict[str, object]:
    """Auditable assertions that rule out the known shortcuts in the old experiment."""
    x = batch["x"]
    action, branch = batch["action"], batch["branch"]
    trip, pos = batch["trip"], batch["pos"]
    choice, alias, reset = batch["choice_mask"], batch["alias_mask"], batch["reset_mask"]
    if bool((choice & alias).any()):
        raise RuntimeError("choice supervision overlaps analysed alias corridor")
    if not bool((action[choice] == 0).all()):
        raise RuntimeError("branch-coded action leaked into the supervised pre-action state")
    b_idx, t_idx = torch.where(choice)
    if int(t_idx.max()) + 1 >= action.shape[1]:
        raise RuntimeError("choice event has no subsequent commit token")
    next_action = action[b_idx, t_idx + 1]
    expected_action = torch.where(branch[choice] == 0, 2, 3)
    if not bool((next_action == expected_action).all()):
        raise RuntimeError("commit action is not located exactly after the supervised state")
    if bool((trip[alias] == 0).any()):
        raise RuntimeError("uninferable first trip entered splitter analysis")
    expected_alias = (
        (trip > 0) & (pos >= cfg.alias_start) & (pos < cfg.alias_end)
        & (action == 0)
    )
    if not bool((alias == expected_alias).all()):
        raise RuntimeError("alias mask does not identify only the outbound shared mid-corridor")
    for tr in range(1, cfg.trips_per_trial):
        labels = branch[:, :][trip == tr]
        # Each trial contributes the same number of tokens, so exact balance is expected.
        if abs(float(labels.float().mean()) - .5) > 1e-6:
            raise RuntimeError(f"branch labels are not balanced at trip {tr}")
    transitions = torch.zeros_like(reset)
    transitions[:, 0] = True
    transitions[:, 1:] = trip[:, 1:] != trip[:, :-1]
    if not bool((reset == transitions).all()):
        raise RuntimeError("history-reset mask is not aligned with true trip boundaries")
    return {
        "status": "passed",
        "sequence_length": int(x.shape[1]),
        "choice_tokens_per_trial_batch": int(choice.sum()),
        "alias_tokens_per_trial_batch": int(alias.sum()),
        "choice_alias_overlap": int((choice & alias).sum()),
        "choice_action_values": sorted(set(action[choice].tolist())),
        "first_trip_excluded_from_alias": True,
        "balanced_random_initial_branch": True,
        "true_trip_boundary_reset": True,
    }


@torch.no_grad()
def output_contract_audit(models: Dict[str, nn.Module], batch: Dict[str, torch.Tensor]) -> Dict[str, object]:
    b = to_device({k: v[:2] for k, v in batch.items()})
    required_tem = {
        "pred_loss", "mem", "predictive_state", "attractor_state",
        "associative_read", "xhat", "g_hist", "pos_logits_hist", "phase_logits_hist",
        "choice_logits",
    }
    required_hdgm = required_tem | {
        "feedback_state", "ec_gate_pattern", "ca3_gate_pattern", "ca1_gate_pattern",
        "ec_hist", "ca3_hist", "ca1_hist", "write_hist", "v_hist",
    }
    result = {}
    for name, required in (("tem", required_tem), ("hdgm", required_hdgm)):
        out = models[name](b["x"], b["action"], return_details=True)
        missing = sorted(required - set(out))
        if missing:
            raise RuntimeError(f"{name} output contract missing: {missing}")
        result[name] = {k: list(out[k].shape) if out[k].ndim else [] for k in sorted(required) if k != "pred_loss"}
    return {"status": "passed", "output_shapes": result}


def make_banks(base_env: CounterbalancedBifurcationEnv, cfg: SplitterV7Config,
               seed: int) -> Dict[str, List[Dict[str, torch.Tensor]]]:
    return {
        "val": build_bank(base_env.clone(seed + 10101), cfg.val_batches, cfg.bank_batch_size),
        "test": build_bank(base_env.clone(seed + 20202), cfg.test_batches, cfg.bank_batch_size),
        "probe_train": build_bank(base_env.clone(seed + 30303), cfg.probe_train_batches, cfg.bank_batch_size),
        "probe_val": build_bank(base_env.clone(seed + 40404), cfg.probe_val_batches, cfg.bank_batch_size),
        "probe_test": build_bank(base_env.clone(seed + 50505), cfg.probe_test_batches, cfg.bank_batch_size),
    }


def train_one_seed(seed: int, cfg: SplitterV7Config) -> Tuple[dict, dict, dict]:
    set_global_seed(seed)
    env = CounterbalancedBifurcationEnv(cfg, seed=seed + 17)
    banks = make_banks(env, cfg, seed)
    models = instantiate_models(cfg)
    audit_batch = env.clone(seed + 60606).sample_batch(cfg.bank_batch_size, balanced=True)
    audits = {
        "architecture": architecture_contract(cfg, models),
        "task_integrity": task_integrity_audit(audit_batch, cfg),
        "output_contract": output_contract_audit(models, audit_batch),
    }
    optimizers = {k: build_optimizer(k, m, cfg) for k, m in models.items()}
    history = {k: [] for k in models}
    snapshots: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}
    max_updates = max(cfg.checkpoints)

    print("=" * 112)
    print(
        f"V7.3 DUAL-ENDPOINT SPLITTER | SEED={seed} | "
        f"REPRESENTATION={cfg.representation_update} | BEHAVIOR={cfg.behavior_update}"
    )
    print("=" * 112)
    print("Preflight audits: architecture=PASS | task-integrity=PASS | output-contract=PASS")
    print("No branch labels are applied inside the analysed shared corridor.")
    print("Total parameters | " + " | ".join(f"{k}={parameter_count(v):,}" for k, v in models.items()))
    print("Effective task parameters | " + " | ".join(f"{k}={effective_parameter_count(v):,}" for k, v in models.items()))
    print(
        f"Locked optimization | core_lr={cfg.core_lr:.1e} | "
        f"choice_head_lr={cfg.choice_head_lr:.1e} | choice_weight={cfg.choice_weight:.2f}"
    )

    for update in range(1, max_updates + 1):
        batch = to_device(env.sample_batch(cfg.batch_size, balanced=True))
        train_acc = {}
        train_loss = {}
        for name in ("tem", "hdgm"):
            model = models[name]
            model.train()
            optimizers[name].zero_grad(set_to_none=True)
            loss, out = batch_loss(model, batch, cfg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizers[name].step()
            train_loss[name] = float(loss.detach().item())
            train_acc[name] = masked_accuracy(
                out["choice_logits"], batch["branch"][:, :-1], batch["choice_mask"][:, :-1]
            )

        stateless = models["stateless"]
        stateless.train()
        optimizers["stateless"].zero_grad(set_to_none=True)
        sl = stateless(batch["x"], batch["action"])
        s_loss = masked_ce(sl, batch["branch"][:, :-1], batch["choice_mask"][:, :-1])
        s_loss.backward()
        nn.utils.clip_grad_norm_(stateless.parameters(), cfg.grad_clip)
        optimizers["stateless"].step()
        train_loss["stateless"] = float(s_loss.detach().item())
        train_acc["stateless"] = masked_accuracy(
            sl, batch["branch"][:, :-1], batch["choice_mask"][:, :-1]
        )

        if update in cfg.checkpoints:
            val = {k: evaluate_choice(m, banks["val"], k) for k, m in models.items()}
            for k in models:
                history[k].append({
                    "update": update, "train": train_acc[k],
                    "train_loss": train_loss[k], "val": val[k],
                })
            snapshots[update] = {k: state_dict_cpu(m) for k, m in models.items()}
            print(
                f"Update {update:04d}/{max_updates} | "
                f"train TEM/HDGM/NoHist={train_acc['tem']:.3f}/{train_acc['hdgm']:.3f}/{train_acc['stateless']:.3f} | "
                f"fixed-val={val['tem']:.3f}/{val['hdgm']:.3f}/{val['stateless']:.3f} | "
                f"loss={train_loss['tem']:.3f}/{train_loss['hdgm']:.3f}/{train_loss['stateless']:.3f}"
            )

    behavior = snapshots[cfg.behavior_update]
    for k, model in models.items():
        model.load_state_dict(behavior[k])
    test_choice = {k: evaluate_choice(m, banks["test"], k) for k, m in models.items()}
    return {"models": models, "banks": banks, "env": env}, {
        "seed": seed, "history": history, "choice_test": test_choice,
        "representation_update": cfg.representation_update,
        "behavior_update": cfg.behavior_update,
        "preflight_audits": audits,
    }, snapshots


@torch.no_grad()
def collect_alias_dataset(model: SequenceChoiceModel,
                          bank: Sequence[Dict[str, torch.Tensor]],
                          representation: str = "mem",
                          reset_history: bool = False) -> Dict[str, np.ndarray]:
    model.eval()
    xs, ys, positions, trips, times, groups, raw_obs = [], [], [], [], [], [], []
    group_offset = 0
    for cpu_batch in bank:
        b = to_device(cpu_batch)
        reset = b["reset_mask"] if reset_history else None
        out = model(b["x"], b["action"], reset_mask=reset, return_details=True)
        rep = out[representation]
        mask = b["alias_mask"][:, :-1]
        batch_n, time_n = mask.shape
        group = (
            torch.arange(batch_n, device=DEVICE)[:, None] * model.core.cfg.trips_per_trial
            + b["trip"][:, :-1] + group_offset
        )
        time_grid = torch.arange(time_n, device=DEVICE)[None, :].expand(batch_n, -1)
        xs.append(rep[mask].cpu().numpy())
        ys.append(b["branch"][:, :-1][mask].cpu().numpy())
        positions.append(b["pos"][:, :-1][mask].cpu().numpy())
        trips.append(b["trip"][:, :-1][mask].cpu().numpy())
        times.append(time_grid[mask].cpu().numpy())
        groups.append(group[mask].cpu().numpy())
        raw_obs.append(b["x"][:, :-1][mask].cpu().numpy())
        group_offset += batch_n * model.core.cfg.trips_per_trial
    return {
        "X": np.concatenate(xs), "y": np.concatenate(ys),
        "pos": np.concatenate(positions), "trip": np.concatenate(trips),
        "time": np.concatenate(times), "group": np.concatenate(groups),
        "obs": np.concatenate(raw_obs),
    }


def timing_features(data: Dict[str, np.ndarray], cfg: SplitterV7Config) -> np.ndarray:
    pos = np.eye(cfg.corridor_len, dtype=np.float32)[data["pos"]]
    trip = np.eye(cfg.trips_per_trial, dtype=np.float32)[data["trip"]]
    time = (data["time"].astype(np.float32) / max(1.0, float(data["time"].max())))[:, None]
    return np.concatenate([pos, trip, time], 1)


def standardize(train_x: np.ndarray, *others: np.ndarray) -> Tuple[np.ndarray, ...]:
    mean = train_x.mean(0, keepdims=True)
    std = train_x.std(0, keepdims=True) + 1e-6
    return tuple((x - mean) / std for x in (train_x,) + others)


def balanced_indices(y: np.ndarray, rng: np.random.RandomState,
                     max_per_class: int = 12000) -> np.ndarray:
    indices = []
    n = min(*(int((y == c).sum()) for c in (0, 1)), max_per_class)
    for c in (0, 1):
        take = rng.choice(np.flatnonzero(y == c), n, replace=False)
        indices.append(take)
    out = np.concatenate(indices)
    rng.shuffle(out)
    return out


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def fit_probe(train_x: np.ndarray, train_y: np.ndarray,
              val_x: np.ndarray, val_y: np.ndarray,
              test_x: np.ndarray, test_y: np.ndarray,
              cfg: SplitterV7Config, seed: int) -> float:
    rng = np.random.RandomState(seed)
    ti = balanced_indices(train_y, rng)
    vi = balanced_indices(val_y, rng)
    qi = balanced_indices(test_y, rng)
    train_x, train_y = train_x[ti], train_y[ti]
    val_x, val_y = val_x[vi], val_y[vi]
    test_x, test_y = test_x[qi], test_y[qi]
    train_x, val_x, test_x = standardize(train_x, val_x, test_x)
    tx = torch.as_tensor(train_x, dtype=torch.float32, device=DEVICE)
    ty = torch.as_tensor(train_y, dtype=torch.long, device=DEVICE)
    vx = torch.as_tensor(val_x, dtype=torch.float32, device=DEVICE)
    vy = torch.as_tensor(val_y, dtype=torch.long, device=DEVICE)
    qx = torch.as_tensor(test_x, dtype=torch.float32, device=DEVICE)
    qy = torch.as_tensor(test_y, dtype=torch.long, device=DEVICE)
    torch.manual_seed(seed)
    probe = LinearProbe(tx.shape[1]).to(DEVICE)
    opt = optim.Adam(probe.parameters(), lr=cfg.probe_lr,
                     weight_decay=cfg.probe_weight_decay)
    best, best_state, stale = -1.0, None, 0
    for _ in range(cfg.probe_epochs):
        probe.train()
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(probe(tx), ty)
        loss.backward()
        opt.step()
        probe.eval()
        with torch.no_grad():
            acc = float((probe(vx).argmax(1) == vy).float().mean().item())
        if acc > best + 1e-8:
            best, stale = acc, 0
            best_state = state_dict_cpu(probe)
        else:
            stale += 1
        if stale >= cfg.probe_patience:
            break
    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        return float((probe(qx).argmax(1) == qy).float().mean().item())


def groupwise_permuted_labels(data: Dict[str, np.ndarray], seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    groups = np.unique(data["group"])
    group_labels = np.asarray([data["y"][data["group"] == g][0] for g in groups])
    permuted = group_labels.copy()
    rng.shuffle(permuted)
    mapping = dict(zip(groups.tolist(), permuted.tolist()))
    return np.asarray([mapping[int(g)] for g in data["group"]], dtype=np.int64)


def geometry_metrics(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    distances, dispersions, ratios = [], [], []
    for p in sorted(np.unique(data["pos"]).tolist()):
        xp = data["X"][data["pos"] == p]
        yp = data["y"][data["pos"] == p]
        if min((yp == 0).sum(), (yp == 1).sum()) < 2:
            continue
        c0, c1 = xp[yp == 0].mean(0), xp[yp == 1].mean(0)
        d = float(np.linalg.norm(c0 - c1) / math.sqrt(xp.shape[1]))
        w0 = np.linalg.norm(xp[yp == 0] - c0, axis=1).mean()
        w1 = np.linalg.norm(xp[yp == 1] - c1, axis=1).mean()
        w = float((w0 + w1) / (2.0 * math.sqrt(xp.shape[1])))
        distances.append(d)
        dispersions.append(w)
        ratios.append(d / (w + 1e-8))
    return {
        "centroid_distance": float(np.mean(distances)),
        "within_branch_dispersion": float(np.mean(dispersions)),
        "separation_ratio": float(np.mean(ratios)),
    }


def observation_match_audit(data: Dict[str, np.ndarray]) -> float:
    """Mean same-position L2 distance between opposite-branch raw observations."""
    vals = []
    for p in np.unique(data["pos"]):
        x0 = data["obs"][(data["pos"] == p) & (data["y"] == 0)]
        x1 = data["obs"][(data["pos"] == p) & (data["y"] == 1)]
        n = min(len(x0), len(x1))
        if n:
            vals.append(np.linalg.norm(x0[:n] - x1[:n], axis=1).mean())
    return float(np.mean(vals))


@torch.no_grad()
def collect_layerwise_alias(model: SequenceChoiceModel,
                            bank: Sequence[Dict[str, torch.Tensor]],
                            representations: Sequence[str],
                            reset_history: bool = False) -> Dict[str, Dict[str, np.ndarray]]:
    """Collect all requested layers in one forward pass per held-out batch."""
    model.eval()
    storage = {rep: {"X": [], "y": [], "pos": [], "trip": [], "time": [],
                     "group": [], "obs": []} for rep in representations}
    group_offset = 0
    for cpu_batch in bank:
        b = to_device(cpu_batch)
        out = model(
            b["x"], b["action"],
            reset_mask=b["reset_mask"] if reset_history else None,
            return_details=True,
        )
        mask = b["alias_mask"][:, :-1]
        batch_n, time_n = mask.shape
        group = (
            torch.arange(batch_n, device=DEVICE)[:, None] * model.core.cfg.trips_per_trial
            + b["trip"][:, :-1] + group_offset
        )
        time_grid = torch.arange(time_n, device=DEVICE)[None, :].expand(batch_n, -1)
        common = {
            "y": b["branch"][:, :-1][mask].cpu().numpy(),
            "pos": b["pos"][:, :-1][mask].cpu().numpy(),
            "trip": b["trip"][:, :-1][mask].cpu().numpy(),
            "time": time_grid[mask].cpu().numpy(),
            "group": group[mask].cpu().numpy(),
            "obs": b["x"][:, :-1][mask].cpu().numpy(),
        }
        for rep in representations:
            if rep not in out:
                raise KeyError(f"Requested representation '{rep}' not in outputs: {sorted(out)}")
            storage[rep]["X"].append(out[rep][mask].cpu().numpy())
            for key, value in common.items():
                storage[rep][key].append(value)
        group_offset += batch_n * model.core.cfg.trips_per_trial
    return {
        rep: {key: np.concatenate(parts) for key, parts in fields.items()}
        for rep, fields in storage.items()
    }


@torch.no_grad()
def collect_layerwise_region(model: SequenceChoiceModel,
                             bank: Sequence[Dict[str, torch.Tensor]],
                             representations: Sequence[str],
                             region: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Collect frozen states from all outbound positions or the exact choice state."""
    if region not in ("outbound", "choice"):
        raise ValueError("region must be 'outbound' or 'choice'")
    model.eval()
    storage = {rep: {"X": [], "y": [], "pos": [], "trip": [], "time": [],
                     "group": [], "obs": []} for rep in representations}
    group_offset = 0
    for cpu_batch in bank:
        b = to_device(cpu_batch)
        out = model(b["x"], b["action"], return_details=True)
        if region == "choice":
            mask = b["choice_mask"][:, :-1]
        else:
            mask = (
                (b["trip"][:, :-1] > 0)
                & (b["direction"][:, :-1] == 0)
                & (b["action"][:, :-1] == 0)
                & (b["pos"][:, :-1] >= 0)
            )
        batch_n, time_n = mask.shape
        group = (
            torch.arange(batch_n, device=DEVICE)[:, None] * model.core.cfg.trips_per_trial
            + b["trip"][:, :-1] + group_offset
        )
        time_grid = torch.arange(time_n, device=DEVICE)[None, :].expand(batch_n, -1)
        common = {
            "y": b["branch"][:, :-1][mask].cpu().numpy(),
            "pos": b["pos"][:, :-1][mask].cpu().numpy(),
            "trip": b["trip"][:, :-1][mask].cpu().numpy(),
            "time": time_grid[mask].cpu().numpy(),
            "group": group[mask].cpu().numpy(),
            "obs": b["x"][:, :-1][mask].cpu().numpy(),
        }
        for rep in representations:
            storage[rep]["X"].append(out[rep][mask].cpu().numpy())
            for key, value in common.items():
                storage[rep][key].append(value)
        group_offset += batch_n * model.core.cfg.trips_per_trial
    return {
        rep: {key: np.concatenate(parts) for key, parts in fields.items()}
        for rep, fields in storage.items()
    }


def model_representation_names(model: SequenceChoiceModel) -> Tuple[str, ...]:
    if isinstance(model.core, HDGMCore):
        return (
            "mem", "g_hist", "predictive_state", "attractor_state",
            "associative_read", "feedback_state", "ec_gate_pattern",
            "ca3_gate_pattern", "ca1_gate_pattern",
        )
    return ("mem", "g_hist", "predictive_state", "attractor_state", "associative_read")


def full_position_probe_curve(model: SequenceChoiceModel, banks: Dict[str, list],
                              cfg: SplitterV7Config, seed: int) -> Dict[str, float]:
    datasets = {
        split: collect_layerwise_region(
            model, banks[f"probe_{split}"], ("mem",), "outbound"
        )["mem"]
        for split in ("train", "val", "test")
    }
    curve = {}
    for pos in range(cfg.corridor_len):
        sliced = {}
        for split, data in datasets.items():
            mask = data["pos"] == pos
            sliced[split] = {key: value[mask] for key, value in data.items()}
        curve[str(pos)] = fit_probe(
            sliced["train"]["X"], sliced["train"]["y"],
            sliced["val"]["X"], sliced["val"]["y"],
            sliced["test"]["X"], sliced["test"]["y"],
            cfg, seed + 3000 + pos,
        )
    return curve


def choice_state_layerwise_probes(model: SequenceChoiceModel, banks: Dict[str, list],
                                  cfg: SplitterV7Config, seed: int) -> Dict[str, float]:
    reps = model_representation_names(model)
    datasets = {
        split: collect_layerwise_region(
            model, banks[f"probe_{split}"], reps, "choice"
        ) for split in ("train", "val", "test")
    }
    results = {}
    for index, rep in enumerate(reps):
        tr, va, te = datasets["train"][rep], datasets["val"][rep], datasets["test"][rep]
        results[rep] = fit_probe(
            tr["X"], tr["y"], va["X"], va["y"], te["X"], te["y"],
            cfg, seed + 4000 + index,
        )
    return results


def layerwise_probe_table(model: SequenceChoiceModel, banks: Dict[str, list],
                          cfg: SplitterV7Config, seed: int) -> Dict[str, Dict[str, float]]:
    reps = model_representation_names(model)
    datasets = {
        split: collect_layerwise_alias(model, banks[f"probe_{split}"], reps, False)
        for split in ("train", "val", "test")
    }
    table = {}
    for index, rep in enumerate(reps):
        tr, va, te = datasets["train"][rep], datasets["val"][rep], datasets["test"][rep]
        accuracy = fit_probe(
            tr["X"], tr["y"], va["X"], va["y"], te["X"], te["y"],
            cfg, seed + 2000 + index,
        )
        geom = geometry_metrics(te)
        table[rep] = {"probe_accuracy": accuracy, **geom}
    return table


@torch.no_grad()
def hdgm_gate_event_metrics(model: SequenceChoiceModel,
                            bank: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    """Descriptive only: no event mask is passed into the model or loss."""
    if not isinstance(model.core, HDGMCore):
        raise TypeError("gate-event metrics require HDGMCore")
    model.eval()
    buckets = {
        "ec_alias": [], "ec_choice": [], "ec_other": [],
        "ca3_alias": [], "ca3_choice": [], "ca3_other": [],
        "ca1_alias": [], "ca1_choice": [], "ca1_other": [],
        "state_change_alias": [], "state_change_choice": [], "state_change_other": [],
    }
    for cpu_batch in bank:
        b = to_device(cpu_batch)
        out = model(b["x"], b["action"], return_details=True)
        alias = b["alias_mask"][:, :-1]
        choice = b["choice_mask"][:, :-1]
        outbound = (b["direction"][:, :-1] == 0) & (b["trip"][:, :-1] > 0)
        other = outbound & (~alias) & (~choice) & (b["action"][:, :-1] == 0)
        for prefix, key in (("ec", "ec_gate_pattern"), ("ca3", "ca3_gate_pattern"),
                            ("ca1", "ca1_gate_pattern")):
            values = out[key].mean(-1)
            for event, mask in (("alias", alias), ("choice", choice), ("other", other)):
                if int(mask.sum()):
                    buckets[f"{prefix}_{event}"].append(values[mask].cpu().numpy())
        mem = out["mem"]
        delta = torch.zeros(mem.shape[:2], device=mem.device)
        delta[:, 1:] = torch.linalg.vector_norm(mem[:, 1:] - mem[:, :-1], dim=-1)
        for event, mask in (("alias", alias), ("choice", choice), ("other", other)):
            if int(mask.sum()):
                buckets[f"state_change_{event}"].append(delta[mask].cpu().numpy())
    means = {key: float(np.concatenate(values).mean()) for key, values in buckets.items()}
    means.update({
        "ec_choice_minus_alias": means["ec_choice"] - means["ec_alias"],
        "ca3_choice_minus_alias": means["ca3_choice"] - means["ca3_alias"],
        "ca1_choice_minus_alias": means["ca1_choice"] - means["ca1_alias"],
        "state_change_choice_minus_alias": (
            means["state_change_choice"] - means["state_change_alias"]
        ),
    })
    return means


@torch.no_grad()
def choice_confusion_matrix(model, bank: Sequence[Dict[str, torch.Tensor]], kind: str) -> List[List[int]]:
    cm = np.zeros((2, 2), dtype=np.int64)
    model.eval()
    for cpu_batch in bank:
        b = to_device(cpu_batch)
        mask = b["choice_mask"][:, :-1]
        labels = b["branch"][:, :-1][mask]
        logits = (model(b["x"], b["action"])[mask] if kind == "stateless"
                  else model(b["x"], b["action"], return_details=True)["choice_logits"][mask])
        pred = logits.argmax(-1)
        for y, q in zip(labels.cpu().tolist(), pred.cpu().tolist()):
            cm[int(y), int(q)] += 1
    return cm.tolist()


def probe_suite(model: SequenceChoiceModel, banks: Dict[str, list],
                cfg: SplitterV7Config, seed: int) -> Dict[str, object]:
    full = {s: collect_alias_dataset(model, banks[f"probe_{s}"], "mem", False)
            for s in ("train", "val", "test")}
    reset = {s: collect_alias_dataset(model, banks[f"probe_{s}"], "mem", True)
             for s in ("train", "val", "test")}

    def run(feature: str, datasets: Dict[str, Dict[str, np.ndarray]], offset: int,
            train_labels: Optional[np.ndarray] = None,
            val_labels: Optional[np.ndarray] = None,
            local_cfg: Optional[SplitterV7Config] = None) -> float:
        get = (lambda d: timing_features(d, cfg)) if feature == "timing" else (lambda d: d[feature])
        fit_cfg = cfg if local_cfg is None else local_cfg
        return fit_probe(
            get(datasets["train"]), datasets["train"]["y"] if train_labels is None else train_labels,
            get(datasets["val"]), datasets["val"]["y"] if val_labels is None else val_labels,
            get(datasets["test"]), datasets["test"]["y"], fit_cfg, seed + offset,
        )

    # Proper negative control: both training and validation labels are shuffled.
    # The true validation labels must never select the epoch of a null probe.
    null_cfg = copy.deepcopy(cfg)
    null_cfg.probe_epochs = cfg.permutation_probe_epochs
    null_cfg.probe_patience = min(15, cfg.permutation_probe_epochs)
    permutation_scores = []
    for repeat in range(cfg.permutation_repeats):
        print(f"    permutation null {repeat + 1:02d}/{cfg.permutation_repeats}")
        shuffled_train = groupwise_permuted_labels(full["train"], seed + 7000 + repeat)
        shuffled_val = groupwise_permuted_labels(full["val"], seed + 8000 + repeat)
        permutation_scores.append(run(
            "X", full, 500 + repeat,
            train_labels=shuffled_train,
            val_labels=shuffled_val,
            local_cfg=null_cfg,
        ))
    result = {
        "full_history_probe": run("X", full, 1),
        "history_reset_probe": run("X", reset, 2),
        "observation_only_probe": run("obs", full, 3),
        "timing_position_probe": run("timing", full, 4),
        "group_label_permutation_probe": float(np.mean(permutation_scores)),
        "group_label_permutation_repeats": permutation_scores,
        "full_geometry": geometry_metrics(full["test"]),
        "reset_geometry": geometry_metrics(reset["test"]),
        "raw_observation_opposite_branch_l2": observation_match_audit(full["test"]),
    }
    print("    fitting full-history layer-wise probes")
    result["layerwise_full_history"] = layerwise_probe_table(model, banks, cfg, seed)
    print("    fitting outbound position 0-17 probes")
    result["outbound_position_curve"] = full_position_probe_curve(model, banks, cfg, seed)
    print("    fitting exact choice-state layer-wise probes")
    result["choice_state_layerwise"] = choice_state_layerwise_probes(model, banks, cfg, seed)
    return result


def exact_sign_flip_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if d.size == 0 or not np.isfinite(d).all():
        raise ValueError("sign-flip test received empty or non-finite paired differences")
    observed = abs(float(d.mean()))
    if len(d) <= 20:
        means = []
        for signs in itertools.product((-1.0, 1.0), repeat=len(d)):
            means.append(abs(float(np.mean(d * np.asarray(signs)))))
        p = float(np.mean(np.asarray(means) >= observed - 1e-15))
    else:
        rng = np.random.RandomState(12345)
        signs = rng.choice((-1.0, 1.0), size=(200000, len(d)))
        p = float((1 + np.sum(np.abs((signs * d).mean(1)) >= observed)) / 200001)
    bootstrap_rng = np.random.RandomState(24680)
    bootstrap_indices = bootstrap_rng.randint(0, len(d), size=(20000, len(d)))
    bootstrap_means = d[bootstrap_indices].mean(axis=1)
    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
    dz = float(d.mean() / (d.std(ddof=1) + 1e-12)) if len(d) > 1 else float("nan")
    return {
        "difference": float(d.mean()),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_raw": p,
        "dz": dz,
    }


def holm_adjust(tests: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    names = sorted(tests, key=lambda n: tests[n]["p_raw"])
    running = 0.0
    m = len(names)
    for rank, name in enumerate(names):
        adjusted = min(1.0, (m - rank) * tests[name]["p_raw"])
        running = max(running, adjusted)
        tests[name]["p_holm"] = running
    return tests


def summarize(values: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(values, dtype=float)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("summary received empty or non-finite values")
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0}


def normalized_auc(checkpoints: Sequence[int], values: Sequence[float]) -> float:
    x = np.asarray(checkpoints, dtype=float)
    y = np.asarray(values, dtype=float)
    if len(x) < 2:
        return float(y[0])
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def first_threshold(checkpoints: Sequence[int], values: Sequence[float],
                    threshold: float) -> Optional[int]:
    for update, value in zip(checkpoints, values):
        if value >= threshold:
            return int(update)
    return None


def save_checkpoint(path: str, cfg: SplitterV7Config, seed: int, update: int,
                    states: Dict[str, Dict[str, torch.Tensor]]) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save({"config": asdict(cfg), "seed": seed, "update": update,
                "models": states}, path)


def run_seed(seed: int, cfg: SplitterV7Config) -> dict:
    bundle, record, snapshots = train_one_seed(seed, cfg)
    models, banks = bundle["models"], bundle["banks"]
    checkpoint_test = {k: [] for k in models}
    for update in cfg.checkpoints:
        for name, model in models.items():
            model.load_state_dict(snapshots[update][name])
            checkpoint_test[name].append(evaluate_choice(model, banks["test"], name))
    # The behavioral family is evaluated only at the locked late endpoint.
    for name, model in models.items():
        model.load_state_dict(snapshots[cfg.behavior_update][name])
    record["choice_test"] = {
        name: evaluate_choice(model, banks["test"], name)
        for name, model in models.items()
    }
    record["checkpoint_test_choice"] = {
        name: {str(u): float(v) for u, v in zip(cfg.checkpoints, values)}
        for name, values in checkpoint_test.items()
    }
    record["early_acquisition"] = {
        name: {
            "normalized_aulc": normalized_auc(cfg.checkpoints, values),
            "first_update_at_0_60": first_threshold(cfg.checkpoints, values, .60),
            "first_update_at_0_70": first_threshold(cfg.checkpoints, values, .70),
        }
        for name, values in checkpoint_test.items()
    }
    record["choice_confusion_matrix"] = {
        name: choice_confusion_matrix(model, banks["test"], name)
        for name, model in models.items()
    }
    record["choice_accuracy_by_trip"] = {
        name: evaluate_choice_by_trip(model, banks["test"], name, cfg)
        for name, model in models.items()
    }

    # The representation family is evaluated on a separately locked earlier
    # checkpoint. No behavioral result is used to select this snapshot.
    for name, model in models.items():
        model.load_state_dict(snapshots[cfg.representation_update][name])
    record["probes"] = {}
    for model_name in ("tem", "hdgm"):
        print(f"Running frozen {model_name.upper()} splitter probes ...")
        record["probes"][model_name] = probe_suite(
            models[model_name], banks, cfg, seed + (0 if model_name == "tem" else 1000)
        )
    record["hdgm_gate_event_metrics"] = hdgm_gate_event_metrics(
        models["hdgm"], banks["probe_test"]
    )
    ckpt_dir = ensure_dir(os.path.join(cfg.output_dir, "checkpoints"))
    for update, states in snapshots.items():
        save_checkpoint(
            os.path.join(ckpt_dir, f"{cfg.name}_seed_{seed}_update_{update}.pt"),
            cfg, seed, update, states,
        )
    return record


def aggregate_records(records: Sequence[dict], cfg: SplitterV7Config) -> dict:
    summary = {
        "config": asdict(cfg),
        "seeds": [r["seed"] for r in records],
        "locked_endpoints": {
            "representation_update": cfg.representation_update,
            "behavior_update": cfg.behavior_update,
        },
        "analysis_contract": {
            "representation_primary": [
                "HDGM full-history probe > HDGM history-reset probe",
                "HDGM full-history probe > HDGM observation-only probe",
            ],
            "behavior_primary": [
                "HDGM choice accuracy > TEM choice accuracy at behavior endpoint",
                "HDGM choice accuracy > 0.50 at behavior endpoint",
            ],
            "secondary": [
                "fixed-checkpoint acquisition curve and normalized AULC",
                "HDGM-vs-TEM full-history probe",
                "full-vs-reset population geometry",
                "layer-wise and gate-event localization",
            ],
        },
    }
    summary["choice_test"] = {
        model: summarize([r["choice_test"][model] for r in records])
        for model in ("tem", "hdgm", "stateless")
    }
    summary["choice_accuracy_by_trip"] = {
        model: {
            str(trip): summarize([
                r["choice_accuracy_by_trip"][model][str(trip)] for r in records
            ])
            for trip in range(1, cfg.trips_per_trial)
        }
        for model in ("tem", "hdgm", "stateless")
    }
    summary["early_acquisition"] = {
        model: {
            "normalized_aulc": summarize([
                r["early_acquisition"][model]["normalized_aulc"] for r in records
            ]),
            "checkpoint_accuracy": {
                str(update): summarize([
                    r["checkpoint_test_choice"][model][str(update)] for r in records
                ]) for update in cfg.checkpoints
            },
        } for model in ("tem", "hdgm", "stateless")
    }
    probe_keys = ["full_history_probe", "history_reset_probe", "observation_only_probe",
                  "timing_position_probe", "group_label_permutation_probe"]
    summary["probes"] = {
        model: {key: summarize([r["probes"][model][key] for r in records]) for key in probe_keys}
        for model in ("tem", "hdgm")
    }
    summary["geometry"] = {}
    for model in ("tem", "hdgm"):
        summary["geometry"][model] = {}
        for condition in ("full_geometry", "reset_geometry"):
            summary["geometry"][model][condition] = {
                key: summarize([r["probes"][model][condition][key] for r in records])
                for key in ("centroid_distance", "within_branch_dispersion", "separation_ratio")
            }
    summary["layerwise_full_history"] = {}
    for model in ("tem", "hdgm"):
        representations = records[0]["probes"][model]["layerwise_full_history"].keys()
        summary["layerwise_full_history"][model] = {
            rep: {
                metric: summarize([
                    r["probes"][model]["layerwise_full_history"][rep][metric]
                    for r in records
                ])
                for metric in ("probe_accuracy", "centroid_distance",
                               "within_branch_dispersion", "separation_ratio")
            }
            for rep in representations
        }
    summary["outbound_position_curve"] = {
        model: {
            str(pos): summarize([
                r["probes"][model]["outbound_position_curve"][str(pos)]
                for r in records
            ])
            for pos in range(cfg.corridor_len)
        }
        for model in ("tem", "hdgm")
    }
    summary["choice_state_layerwise"] = {}
    for model in ("tem", "hdgm"):
        reps = records[0]["probes"][model]["choice_state_layerwise"].keys()
        summary["choice_state_layerwise"][model] = {
            rep: summarize([
                r["probes"][model]["choice_state_layerwise"][rep]
                for r in records
            ])
            for rep in reps
        }
    gate_metric_keys = records[0]["hdgm_gate_event_metrics"].keys()
    summary["hdgm_gate_event_metrics"] = {
        key: summarize([r["hdgm_gate_event_metrics"][key] for r in records])
        for key in gate_metric_keys
    }
    representation_tests = {
        "HDGM full probe vs history reset": exact_sign_flip_test(
            [r["probes"]["hdgm"]["full_history_probe"] for r in records],
            [r["probes"]["hdgm"]["history_reset_probe"] for r in records]),
        "HDGM full probe vs observation only": exact_sign_flip_test(
            [r["probes"]["hdgm"]["full_history_probe"] for r in records],
            [r["probes"]["hdgm"]["observation_only_probe"] for r in records]),
    }
    summary["representation_primary_statistics_holm"] = holm_adjust(representation_tests)
    behavior_tests = {
        "HDGM vs TEM choice at behavior endpoint": exact_sign_flip_test(
            [r["choice_test"]["hdgm"] for r in records],
            [r["choice_test"]["tem"] for r in records]),
        "HDGM choice vs chance at behavior endpoint": exact_sign_flip_test(
            [r["choice_test"]["hdgm"] for r in records],
            [0.5 for _ in records]),
    }
    summary["behavior_primary_statistics_holm"] = holm_adjust(behavior_tests)
    summary["secondary_statistics_unadjusted"] = {
        "HDGM vs TEM full probe": exact_sign_flip_test(
            [r["probes"]["hdgm"]["full_history_probe"] for r in records],
            [r["probes"]["tem"]["full_history_probe"] for r in records]),
        "HDGM full geometry vs reset geometry": exact_sign_flip_test(
            [r["probes"]["hdgm"]["full_geometry"]["separation_ratio"] for r in records],
            [r["probes"]["hdgm"]["reset_geometry"]["separation_ratio"] for r in records]),
    }
    return summary


def print_summary(summary: dict) -> None:
    rep_update = summary["locked_endpoints"]["representation_update"]
    beh_update = summary["locked_endpoints"]["behavior_update"]
    print("\n" + "#" * 112)
    print("HDGM V7.3 LOCKED DUAL-ENDPOINT SPLITTER SUMMARY")
    print("#" * 112)
    print(f"\n===== Primary behavioral endpoint: pre-action choice @ update {beh_update} =====")
    for k, v in summary["choice_test"].items():
        print(f"{k:12s}: {v['mean']:.4f} ± {v['sd']:.4f}")
    print(f"\n===== Choice accuracy by trip @ behavior update {beh_update} =====")
    for model, trips in summary["choice_accuracy_by_trip"].items():
        print(f"{model:12s} | " + " | ".join(
            f"trip{trip}={value['mean']:.4f}±{value['sd']:.4f}"
            for trip, value in trips.items()
        ))
    print("\n===== Fixed-checkpoint acquisition =====")
    for model, section in summary["early_acquisition"].items():
        a = section["normalized_aulc"]
        points = " | ".join(
            f"@{u}={v['mean']:.3f}±{v['sd']:.3f}"
            for u, v in section["checkpoint_accuracy"].items()
        )
        print(f"{model:12s} AULC={a['mean']:.4f}±{a['sd']:.4f} | {points}")
    print(f"\n===== Primary representation endpoint: frozen probes @ update {rep_update} =====")
    for model, metrics in summary["probes"].items():
        print(model.upper())
        for k, v in metrics.items():
            print(f"  {k:34s}: {v['mean']:.4f} ± {v['sd']:.4f}")
    print("\n===== Same-position population geometry =====")
    for model, conditions in summary["geometry"].items():
        for condition, metrics in conditions.items():
            print(f"{model:5s} {condition:15s} | " + " | ".join(
                f"{k}={v['mean']:.4f}±{v['sd']:.4f}" for k, v in metrics.items()))
    print("\n===== Frozen layer-wise alias decoding =====")
    for model, table in summary["layerwise_full_history"].items():
        print(model.upper())
        for rep, metrics in table.items():
            p = metrics["probe_accuracy"]
            s = metrics["separation_ratio"]
            print(f"  {rep:24s} decode={p['mean']:.4f}±{p['sd']:.4f} | separation={s['mean']:.4f}±{s['sd']:.4f}")
    print("\n===== Exact pre-action choice-state layer-wise decoding =====")
    for model, table in summary["choice_state_layerwise"].items():
        print(model.upper())
        for rep, value in table.items():
            print(f"  {rep:24s}: {value['mean']:.4f} ± {value['sd']:.4f}")
    print("\n===== Outbound position decoding curve =====")
    for model, curve in summary["outbound_position_curve"].items():
        print(f"{model:5s} | " + " | ".join(
            f"p{pos}={value['mean']:.3f}" for pos, value in curve.items()
        ))
    print("\n===== HDGM descriptive gate-event metrics =====")
    for key, value in summary["hdgm_gate_event_metrics"].items():
        print(f"{key:34s}: {value['mean']:.4f} ± {value['sd']:.4f}")
    print(f"\n===== Representation primary family @ update {rep_update}: exact paired sign-flip + Holm =====")
    for name, v in summary["representation_primary_statistics_holm"].items():
        print(
            f"{name:48s} | diff={v['difference']:.4f} "
            f"CI95=[{v['ci95_low']:.4f},{v['ci95_high']:.4f}] "
            f"p={v['p_raw']:.5f} Holm={v['p_holm']:.5f} dz={v['dz']:.3f}"
        )
    print(f"\n===== Behavioral primary family @ update {beh_update}: exact paired sign-flip + Holm =====")
    for name, v in summary["behavior_primary_statistics_holm"].items():
        print(
            f"{name:48s} | diff={v['difference']:.4f} "
            f"CI95=[{v['ci95_low']:.4f},{v['ci95_high']:.4f}] "
            f"p={v['p_raw']:.5f} Holm={v['p_holm']:.5f} dz={v['dz']:.3f}"
        )
    print("\n===== Secondary statistics (unadjusted; descriptive) =====")
    for name, v in summary["secondary_statistics_unadjusted"].items():
        print(
            f"{name:48s} | diff={v['difference']:.4f} "
            f"CI95=[{v['ci95_low']:.4f},{v['ci95_high']:.4f}] "
            f"p={v['p_raw']:.5f} dz={v['dz']:.3f}"
        )
    print("\nInterpretation guardrails:")
    print("  * Observation/timing/permuted probes should remain near 0.50.")
    print("  * Full-history > reset supports history-dependent splitter-like coding.")
    print(f"  * Representation claims are locked to update {rep_update}.")
    print(f"  * Behavioral claims are locked to update {beh_update}.")


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def plot_learning(records: Sequence[dict], cfg: SplitterV7Config) -> None:
    ensure_dir(cfg.output_dir)
    plt.figure(figsize=(7.2, 4.6))
    for model, color in (("tem", "#4C78A8"), ("hdgm", "#E45756"),
                         ("stateless", "#777777")):
        curves = np.asarray([[z["val"] for z in r["history"][model]] for r in records])
        x = np.asarray(cfg.checkpoints)
        mean, sd = curves.mean(0), curves.std(0, ddof=1) if len(records) > 1 else np.zeros_like(mean)
        plt.plot(x, mean, marker="o", label=model.upper(), color=color)
        plt.fill_between(x, mean - sd, mean + sd, alpha=.18, color=color)
    plt.axhline(.5, color="black", linestyle="--", linewidth=1, label="chance")
    plt.axvline(
        cfg.representation_update, color="#14B8A6", linestyle=":", linewidth=1.5,
        label=f"representation endpoint ({cfg.representation_update})",
    )
    plt.axvline(
        cfg.behavior_update, color="#F59E0B", linestyle="-.", linewidth=1.5,
        label=f"behavior endpoint ({cfg.behavior_update})",
    )
    plt.xlabel("Training updates (fixed checkpoints)")
    plt.ylabel("Pre-action junction choice accuracy")
    plt.ylim(0.35, 1.02)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "v7_3_learning_curve.png"), dpi=220)
    plt.close()


def plot_probe_controls(records: Sequence[dict], cfg: SplitterV7Config) -> None:
    labels = ["Full history", "History reset", "Observation", "Timing", "Permuted"]
    keys = ["full_history_probe", "history_reset_probe", "observation_only_probe",
            "timing_position_probe", "group_label_permutation_probe"]
    x = np.arange(len(keys))
    width = .36
    plt.figure(figsize=(8.2, 4.8))
    for shift, model, color in ((-width/2, "tem", "#4C78A8"),
                                (width/2, "hdgm", "#E45756")):
        vals = np.asarray([[r["probes"][model][k] for k in keys] for r in records])
        mean = vals.mean(0)
        sd = vals.std(0, ddof=1) if len(records) > 1 else np.zeros_like(mean)
        plt.bar(x + shift, mean, width, yerr=sd, capsize=3, label=model.upper(), color=color)
    plt.axhline(.5, color="black", linestyle="--", linewidth=1)
    plt.xticks(x, labels, rotation=18, ha="right")
    plt.ylabel("Held-out branch decoding accuracy")
    plt.ylim(0.35, 1.02)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "v7_3_probe_controls.png"), dpi=220)
    plt.close()


def plot_outbound_position_curve(records: Sequence[dict], cfg: SplitterV7Config) -> None:
    plt.figure(figsize=(7.6, 4.6))
    positions = np.arange(cfg.corridor_len)
    for model, color in (("tem", "#4C78A8"), ("hdgm", "#E45756")):
        curves = np.asarray([
            [r["probes"][model]["outbound_position_curve"][str(p)] for p in positions]
            for r in records
        ])
        mean = curves.mean(0)
        sd = curves.std(0, ddof=1) if len(records) > 1 else np.zeros_like(mean)
        plt.plot(positions, mean, marker="o", markersize=3.5, color=color, label=model.upper())
        plt.fill_between(positions, mean - sd, mean + sd, color=color, alpha=.16)
    plt.axhline(.5, color="black", linestyle="--", linewidth=1)
    plt.axvspan(cfg.alias_start, cfg.alias_end - 1, color="#F59E0B", alpha=.12,
                label="analysed alias region")
    plt.axvline(cfg.corridor_len - 1, color="#475569", linestyle=":", linewidth=1,
                label="pre-action choice state")
    plt.xticks(positions)
    plt.xlabel("Outbound shared-corridor position")
    plt.ylabel("Held-out future-branch decoding")
    plt.ylim(.35, 1.02)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "v7_3_outbound_position_curve.png"), dpi=220)
    plt.close()


def plot_choice_state_layers(records: Sequence[dict], cfg: SplitterV7Config) -> None:
    reps = list(records[0]["probes"]["hdgm"]["choice_state_layerwise"].keys())
    values = np.asarray([
        [r["probes"]["hdgm"]["choice_state_layerwise"][rep] for rep in reps]
        for r in records
    ])
    mean = values.mean(0)
    sd = values.std(0, ddof=1) if len(records) > 1 else np.zeros_like(mean)
    plt.figure(figsize=(9.2, 4.8))
    plt.bar(np.arange(len(reps)), mean, yerr=sd, capsize=3, color="#E45756")
    plt.axhline(.5, color="black", linestyle="--", linewidth=1)
    plt.xticks(np.arange(len(reps)), reps, rotation=28, ha="right")
    plt.ylabel("Pre-action branch decoding accuracy")
    plt.ylim(.35, 1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "v7_3_choice_state_layers.png"), dpi=220)
    plt.close()


def run_multiseed(seeds: Sequence[int], cfg: Optional[SplitterV7Config] = None) -> dict:
    cfg = SplitterV7Config() if cfg is None else cfg
    if cfg.representation_update not in cfg.checkpoints:
        raise ValueError("representation_update must be one of the fixed checkpoints")
    if cfg.behavior_update not in cfg.checkpoints:
        raise ValueError("behavior_update must be one of the fixed checkpoints")
    if cfg.representation_update >= cfg.behavior_update:
        raise ValueError("representation_update must precede behavior_update")
    ensure_dir(cfg.output_dir)
    print("=" * 112)
    print(f"START V7.3 LOCKED DUAL-ENDPOINT MULTI-SEED | CONFIG={cfg.name} | DEVICE={DEVICE}")
    print(
        f"SEEDS={list(seeds)} | CHECKPOINTS={list(cfg.checkpoints)} | "
        f"REPRESENTATION={cfg.representation_update} | BEHAVIOR={cfg.behavior_update}"
    )
    print("=" * 112)
    records = []
    for seed in seeds:
        record = run_seed(int(seed), cfg)
        records.append(record)
        with open(os.path.join(cfg.output_dir, f"v7_3_seed_{seed}.json"), "w", encoding="utf-8") as f:
            json.dump(json_safe(record), f, ensure_ascii=False, indent=2)
    summary = aggregate_records(records, cfg)
    print_summary(summary)
    with open(os.path.join(cfg.output_dir, "v7_3_multiseed_results.json"), "w", encoding="utf-8") as f:
        json.dump(json_safe({"records": records, "summary": summary}), f, ensure_ascii=False, indent=2)
    plot_learning(records, cfg)
    plot_probe_controls(records, cfg)
    plot_outbound_position_curve(records, cfg)
    plot_choice_state_layers(records, cfg)
    print(f"\nSaved outputs to: {os.path.abspath(cfg.output_dir)}")
    return summary


# ============================================================================
# Difficulty V2.1: frozen-checkpoint population-robustness analysis
# ============================================================================

@dataclass
class DifficultyV2Config:
    """Locked analysis settings; none of these fields changes TEM/HDGM weights."""
    # Zero is the exact observation map used during V7.3 training. Increasing
    # values additionally blend the previously distinct corridor positions
    # toward the already-existing shared alias prototype. This makes the axis
    # a genuine monotonic test-time corruption rather than distance from the
    # reference distribution.
    alias_corruptions: Tuple[float, ...] = (0.00, 0.25, 0.50, 0.75)
    noise_stds: Tuple[float, ...] = (0.02, 0.05, 0.08, 0.12)
    reference_alias_corruption: float = 0.00
    reference_noise: float = 0.05
    representation_update: int = 150
    behavior_update: int = 300
    test_batches: int = 8
    probe_train_batches: int = 8
    probe_val_batches: int = 4
    probe_test_batches: int = 8
    bank_batch_size: int = 32
    transfer_probe_epochs: int = 140
    transfer_probe_patience: int = 22
    split_half_repeats: int = 24
    checkpoint_dir: str = "hdgm_v7_3_splitter_outputs_confirmatory/checkpoints"
    output_dir: str = "hdgm_difficulty_v2_1_frozen_outputs"
    checkpoint_name_hint: str = "hdgm_v7_3_splitter_dual_endpoint_confirmatory"
    train_if_missing: bool = False
    name: str = "hdgm_difficulty_v2_1_frozen_population_robustness"


def tensor_state_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        arr = value.detach().cpu().contiguous().numpy()
        digest.update(str(arr.dtype).encode("ascii"))
        digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def reconstruct_unaliased_maps(cfg: SplitterV7Config, seed: int) -> Dict[str, np.ndarray]:
    """Reproduce the exact semantic RNG stream used by the V7.3 training env."""
    rng = np.random.RandomState(seed + 17)
    corridor = rng.randn(cfg.corridor_len, cfg.obs_dim).astype(np.float32)
    alias_prototype = rng.randn(cfg.obs_dim).astype(np.float32)
    left = rng.randn(cfg.arm_len, cfg.obs_dim).astype(np.float32)
    right = rng.randn(cfg.arm_len, cfg.obs_dim).astype(np.float32)
    return {
        "corridor_raw": corridor,
        "alias_prototype": alias_prototype,
        "left": left,
        "right": right,
    }


def maps_at_alias_strength(raw: Dict[str, np.ndarray], cfg: SplitterV7Config,
                           alias_strength: float) -> Dict[str, np.ndarray]:
    if not 0.0 <= alias_strength <= 1.0:
        raise ValueError("alias_strength must lie in [0, 1]")
    corridor = raw["corridor_raw"].copy()
    proto = raw["alias_prototype"]
    for p in range(cfg.alias_start, cfg.alias_end):
        corridor[p] = alias_strength * proto + (1.0 - alias_strength) * corridor[p]
    return {"corridor": corridor, "left": raw["left"], "right": raw["right"]}


def maps_with_additional_alias_corruption(raw: Dict[str, np.ndarray],
                                          cfg: SplitterV7Config,
                                          corruption: float) -> Dict[str, np.ndarray]:
    """Start from the exact training map and add monotonic global aliasing.

    The analysed mid-corridor window is already fully aliased in V7.3. The
    corruption therefore acts only on the corridor positions that remained
    distinct during training. At corruption=0 the map is bit-identical to the
    training map; at corruption=1 the entire shared corridor approaches the
    same prototype. Arm observations are never altered.
    """
    if not 0.0 <= corruption <= 1.0:
        raise ValueError("additional alias corruption must lie in [0, 1]")
    base = maps_at_alias_strength(raw, cfg, cfg.alias_strength)
    corridor = base["corridor"].copy()
    proto = raw["alias_prototype"]
    for p in range(cfg.corridor_len):
        if cfg.alias_start <= p < cfg.alias_end:
            continue
        corridor[p] = corruption * proto + (1.0 - corruption) * corridor[p]
    return {"corridor": corridor, "left": base["left"], "right": base["right"]}


def make_condition_banks(seed: int, base_cfg: SplitterV7Config,
                         dcfg: DifficultyV2Config, alias_corruption: float,
                         noise_std: float) -> Dict[str, List[Dict[str, torch.Tensor]]]:
    """Independent full-trial banks with common random numbers across conditions."""
    local_cfg = copy.deepcopy(base_cfg)
    local_cfg.noise_std = float(noise_std)
    raw = reconstruct_unaliased_maps(base_cfg, seed)
    maps = maps_with_additional_alias_corruption(
        raw, local_cfg, float(alias_corruption)
    )
    env = CounterbalancedBifurcationEnv(local_cfg, seed=seed + 17, maps=maps)
    # The same RNG offsets are used at every condition. Thus trial identities and
    # Gaussian draws are paired; only the locked corruption magnitude changes.
    return {
        "test": build_bank(env.clone(seed + 62001), dcfg.test_batches, dcfg.bank_batch_size),
        "probe_train": build_bank(env.clone(seed + 63001), dcfg.probe_train_batches, dcfg.bank_batch_size),
        "probe_val": build_bank(env.clone(seed + 64001), dcfg.probe_val_batches, dcfg.bank_batch_size),
        "probe_test": build_bank(env.clone(seed + 65001), dcfg.probe_test_batches, dcfg.bank_batch_size),
    }


def base_map_reconstruction_audit(seed: int, cfg: SplitterV7Config) -> None:
    """Assert that alias=1 reconstructs the semantic maps seen during training."""
    original = CounterbalancedBifurcationEnv(cfg, seed=seed + 17)
    raw = reconstruct_unaliased_maps(cfg, seed)
    rebuilt = maps_at_alias_strength(raw, cfg, cfg.alias_strength)
    for key in ("corridor", "left", "right"):
        if not np.array_equal(original.maps[key], rebuilt[key]):
            raise RuntimeError(f"semantic-map reconstruction failed for {key}")


def locate_checkpoint(root: str, seed: int, update: int,
                      name_hint: str = "") -> str:
    patterns = []
    if name_hint:
        patterns.append(os.path.join(root, f"{name_hint}_seed_{seed}_update_{update}.pt"))
    patterns.extend([
        os.path.join(root, f"*seed_{seed}_update_{update}.pt"),
        os.path.join(root, "**", f"*seed_{seed}_update_{update}.pt"),
    ])
    matches: List[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern, recursive=True))
    matches = sorted(set(os.path.abspath(p) for p in matches if os.path.isfile(p)))
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint for seed={seed}, update={update} under {os.path.abspath(root)}"
        )
    exact = [p for p in matches if name_hint and name_hint in os.path.basename(p)]
    if len(exact) == 1:
        return exact[0]
    if len(matches) != 1:
        raise RuntimeError(
            f"Ambiguous checkpoints for seed={seed}, update={update}: {matches}"
        )
    return matches[0]


def load_checkpoint_models(path: str, expected_seed: int,
                           expected_update: int) -> Tuple[SplitterV7Config, Dict[str, nn.Module]]:
    payload = torch.load(path, map_location=DEVICE)
    if int(payload.get("seed", -1)) != int(expected_seed):
        raise RuntimeError(f"checkpoint seed mismatch in {path}")
    if int(payload.get("update", -1)) != int(expected_update):
        raise RuntimeError(f"checkpoint update mismatch in {path}")
    cfg_dict = dict(payload["config"])
    if isinstance(cfg_dict.get("checkpoints"), list):
        cfg_dict["checkpoints"] = tuple(cfg_dict["checkpoints"])
    cfg = SplitterV7Config(**cfg_dict)
    models = instantiate_models(cfg)
    for name, model in models.items():
        if name not in payload["models"]:
            raise KeyError(f"model '{name}' absent from {path}")
        model.load_state_dict(payload["models"][name], strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return cfg, models


def train_and_save_missing_seed(seed: int, cfg: SplitterV7Config,
                                checkpoint_dir: str) -> None:
    """Optional exact V7.3 training fallback; disabled by default."""
    _, _, snapshots = train_one_seed(seed, cfg)
    ensure_dir(checkpoint_dir)
    for update, states in snapshots.items():
        save_checkpoint(
            os.path.join(checkpoint_dir, f"{cfg.name}_seed_{seed}_update_{update}.pt"),
            cfg, seed, update, states,
        )


class FrozenProbeArtifact:
    def __init__(self, probe: LinearProbe, mean: np.ndarray, std: np.ndarray):
        self.probe = probe
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)


def fit_reference_probe(train: Dict[str, np.ndarray], val: Dict[str, np.ndarray],
                        cfg: SplitterV7Config, dcfg: DifficultyV2Config,
                        seed: int) -> FrozenProbeArtifact:
    rng = np.random.RandomState(seed)
    ti = balanced_indices(train["y"], rng)
    vi = balanced_indices(val["y"], rng)
    train_x, train_y = train["X"][ti], train["y"][ti]
    val_x, val_y = val["X"][vi], val["y"][vi]
    mean = train_x.mean(0, keepdims=True)
    std = train_x.std(0, keepdims=True) + 1e-6
    tx = torch.as_tensor((train_x - mean) / std, dtype=torch.float32, device=DEVICE)
    ty = torch.as_tensor(train_y, dtype=torch.long, device=DEVICE)
    vx = torch.as_tensor((val_x - mean) / std, dtype=torch.float32, device=DEVICE)
    vy = torch.as_tensor(val_y, dtype=torch.long, device=DEVICE)
    torch.manual_seed(seed)
    probe = LinearProbe(tx.shape[1]).to(DEVICE)
    optimizer = optim.Adam(
        probe.parameters(), lr=cfg.probe_lr, weight_decay=cfg.probe_weight_decay
    )
    best, best_state, stale = -1.0, None, 0
    for _ in range(dcfg.transfer_probe_epochs):
        probe.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(probe(tx), ty)
        loss.backward()
        optimizer.step()
        probe.eval()
        with torch.no_grad():
            score = float((probe(vx).argmax(1) == vy).float().mean().item())
        if score > best + 1e-8:
            best, stale = score, 0
            best_state = state_dict_cpu(probe)
        else:
            stale += 1
        if stale >= dcfg.transfer_probe_patience:
            break
    if best_state is None:
        raise RuntimeError("reference probe did not produce a checkpoint")
    probe.load_state_dict(best_state)
    probe.eval()
    for parameter in probe.parameters():
        parameter.requires_grad_(False)
    return FrozenProbeArtifact(probe, mean, std)


@torch.no_grad()
def evaluate_frozen_probe(artifact: FrozenProbeArtifact,
                          data: Dict[str, np.ndarray], seed: int) -> float:
    rng = np.random.RandomState(seed)
    qi = balanced_indices(data["y"], rng)
    x = (data["X"][qi] - artifact.mean) / artifact.std
    y = data["y"][qi]
    tx = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
    ty = torch.as_tensor(y, dtype=torch.long, device=DEVICE)
    return float((artifact.probe(tx).argmax(1) == ty).float().mean().item())


def branch_difference_vector(data: Dict[str, np.ndarray], selected_groups: np.ndarray) -> Optional[np.ndarray]:
    keep = np.isin(data["group"], selected_groups)
    vectors = []
    for pos in sorted(np.unique(data["pos"]).tolist()):
        local = keep & (data["pos"] == pos)
        x0 = data["X"][local & (data["y"] == 0)]
        x1 = data["X"][local & (data["y"] == 1)]
        if len(x0) < 2 or len(x1) < 2:
            return None
        vectors.append(x1.mean(0) - x0.mean(0))
    return np.concatenate(vectors) if vectors else None


def population_split_half_stability(data: Dict[str, np.ndarray], cfg: SplitterV7Config,
                                    repeats: int, seed: int) -> float:
    """Cosine agreement of position-wise population branch-difference vectors.

    Pearson correlation is undefined when one half has a nearly constant
    vector. Cosine agreement remains defined; a zero-norm half contains no
    reproducible branch vector and is conservatively assigned stability zero.
    """
    rng = np.random.RandomState(seed)
    # group ids identify trips. Split at whole-trial level to prevent leakage.
    trial_ids = np.unique(data["group"] // cfg.trips_per_trial)
    scores = []
    for _ in range(repeats):
        shuffled = trial_ids.copy()
        rng.shuffle(shuffled)
        cut = len(shuffled) // 2
        first_trials, second_trials = shuffled[:cut], shuffled[cut:]
        first_groups = np.unique(data["group"][np.isin(data["group"] // cfg.trips_per_trial, first_trials)])
        second_groups = np.unique(data["group"][np.isin(data["group"] // cfg.trips_per_trial, second_trials)])
        a = branch_difference_vector(data, first_groups)
        b = branch_difference_vector(data, second_groups)
        if a is None or b is None:
            continue
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denominator < 1e-12:
            scores.append(0.0)
        else:
            scores.append(float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0)))
    if not scores:
        return 0.0
    value = float(np.mean(scores))
    if not np.isfinite(value):
        raise RuntimeError("non-finite split-half stability after cosine computation")
    return value


def evaluate_representation_condition(model: SequenceChoiceModel,
                                      banks: Dict[str, list],
                                      artifact: FrozenProbeArtifact,
                                      cfg: SplitterV7Config,
                                      dcfg: DifficultyV2Config,
                                      seed: int) -> Dict[str, float]:
    full = collect_alias_dataset(model, banks["probe_test"], "mem", False)
    reset = collect_alias_dataset(model, banks["probe_test"], "mem", True)
    full_geometry = geometry_metrics(full)
    reset_geometry = geometry_metrics(reset)
    return {
        "reference_transfer_probe": evaluate_frozen_probe(artifact, full, seed + 1),
        "history_reset_transfer_probe": evaluate_frozen_probe(artifact, reset, seed + 2),
        "centroid_distance": full_geometry["centroid_distance"],
        "within_branch_dispersion": full_geometry["within_branch_dispersion"],
        "separation_ratio": full_geometry["separation_ratio"],
        "reset_separation_ratio": reset_geometry["separation_ratio"],
        "history_dependent_separation": (
            full_geometry["separation_ratio"] - reset_geometry["separation_ratio"]
        ),
        "population_split_half_stability": population_split_half_stability(
            full, cfg, dcfg.split_half_repeats, seed + 3
        ),
        "raw_observation_opposite_branch_l2": observation_match_audit(full),
    }


def evaluate_behavior_condition(models: Dict[str, nn.Module],
                                bank: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    return {name: evaluate_choice(model, bank, name) for name, model in models.items()}


def condition_key(axis: str, value: float) -> str:
    return f"{axis}_{value:.3f}"


def linear_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < 2 or x.size != y.size or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("linear_slope requires paired finite arrays of length >= 2")
    return float(np.polyfit(x, y, 1)[0])


def curve_auc(xs: Sequence[float], ys: Sequence[float]) -> float:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < 2 or x.size != y.size or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("curve_auc requires paired finite arrays of length >= 2")
    order = np.argsort(x)
    x, y = x[order], y[order]
    return float(np.trapezoid(y, x) / max(1e-12, x[-1] - x[0]))


def difficulty_curves(record: dict, dcfg: DifficultyV2Config) -> dict:
    out = {"representation": {}, "behavior": {}}
    for axis, values in (("alias", dcfg.alias_corruptions), ("noise", dcfg.noise_stds)):
        out["representation"][axis] = {}
        for model in ("tem", "hdgm"):
            ys = [record["conditions"][condition_key(axis, float(v))]["representation"][model]
                  ["reference_transfer_probe"] for v in values]
            ss = [record["conditions"][condition_key(axis, float(v))]["representation"][model]
                  ["population_split_half_stability"] for v in values]
            hs = [record["conditions"][condition_key(axis, float(v))]["representation"][model]
                  ["history_dependent_separation"] for v in values]
            out["representation"][axis][model] = {
                "transfer_probe_auc": curve_auc(values, ys),
                "transfer_probe_slope": linear_slope(values, ys),
                "stability_auc": curve_auc(values, ss),
                "history_dependent_separation_auc": curve_auc(values, hs),
            }
        out["behavior"][axis] = {}
        for model in ("tem", "hdgm", "stateless"):
            ys = [record["conditions"][condition_key(axis, float(v))]["behavior"][model]
                  for v in values]
            out["behavior"][axis][model] = {
                "choice_auc": curve_auc(values, ys),
                "choice_slope": linear_slope(values, ys),
            }
    return out


def run_difficulty_seed(seed: int, base_cfg: SplitterV7Config,
                        dcfg: DifficultyV2Config) -> dict:
    print("=" * 118)
    print(
        f"DIFFICULTY V2.1 FROZEN ROBUSTNESS | SEED={seed} | "
        f"REP={dcfg.representation_update} | BEHAV={dcfg.behavior_update}"
    )
    print("=" * 118)
    base_map_reconstruction_audit(seed, base_cfg)
    try:
        rep_path = locate_checkpoint(
            dcfg.checkpoint_dir, seed, dcfg.representation_update, dcfg.checkpoint_name_hint
        )
        beh_path = locate_checkpoint(
            dcfg.checkpoint_dir, seed, dcfg.behavior_update, dcfg.checkpoint_name_hint
        )
    except FileNotFoundError:
        if not dcfg.train_if_missing:
            raise
        print("Checkpoint missing: running the unchanged V7.3 training protocol once.")
        train_cfg = copy.deepcopy(base_cfg)
        train_cfg.name = dcfg.checkpoint_name_hint
        train_and_save_missing_seed(seed, train_cfg, dcfg.checkpoint_dir)
        rep_path = locate_checkpoint(
            dcfg.checkpoint_dir, seed, dcfg.representation_update, dcfg.checkpoint_name_hint
        )
        beh_path = locate_checkpoint(
            dcfg.checkpoint_dir, seed, dcfg.behavior_update, dcfg.checkpoint_name_hint
        )
    rep_cfg, rep_models = load_checkpoint_models(rep_path, seed, dcfg.representation_update)
    beh_cfg, beh_models = load_checkpoint_models(beh_path, seed, dcfg.behavior_update)
    if asdict(rep_cfg) != asdict(beh_cfg):
        raise RuntimeError("representation and behavior checkpoints use different configurations")
    if rep_cfg.hidden_dim != base_cfg.hidden_dim or rep_cfg.action_emb_dim != base_cfg.action_emb_dim:
        raise RuntimeError("checkpoint architecture differs from the locked V7.3 architecture")
    fingerprints_before = {
        f"rep_{name}": tensor_state_fingerprint(rep_models[name]) for name in ("tem", "hdgm")
    }
    fingerprints_before.update({
        f"behavior_{name}": tensor_state_fingerprint(beh_models[name])
        for name in ("tem", "hdgm", "stateless")
    })

    reference_banks = make_condition_banks(
        seed, rep_cfg, dcfg, dcfg.reference_alias_corruption, dcfg.reference_noise
    )
    artifacts = {}
    for offset, model_name in enumerate(("tem", "hdgm")):
        train_data = collect_alias_dataset(
            rep_models[model_name], reference_banks["probe_train"], "mem", False
        )
        val_data = collect_alias_dataset(
            rep_models[model_name], reference_banks["probe_val"], "mem", False
        )
        artifacts[model_name] = fit_reference_probe(
            train_data, val_data, rep_cfg, dcfg, seed + 70000 + 1000 * offset
        )

    conditions: Dict[str, dict] = {}
    scan = []
    for value in dcfg.alias_corruptions:
        scan.append(("alias", float(value), float(value), dcfg.reference_noise))
    for value in dcfg.noise_stds:
        scan.append(("noise", float(value), dcfg.reference_alias_corruption, float(value)))
    for axis, value, alias_corruption, noise_std in scan:
        key = condition_key(axis, value)
        banks = make_condition_banks(seed, rep_cfg, dcfg, alias_corruption, noise_std)
        representation = {
            name: evaluate_representation_condition(
                rep_models[name], banks, artifacts[name], rep_cfg, dcfg,
                seed + 80000 + int(round(value * 1000)) + (0 if name == "tem" else 5000)
            ) for name in ("tem", "hdgm")
        }
        behavior = evaluate_behavior_condition(beh_models, banks["test"])
        conditions[key] = {
            "axis": axis, "value": value,
            "additional_alias_corruption": alias_corruption, "noise_std": noise_std,
            "representation": representation, "behavior": behavior,
        }
        print(
            f"{key:14s} | transfer TEM/HDGM="
            f"{representation['tem']['reference_transfer_probe']:.3f}/"
            f"{representation['hdgm']['reference_transfer_probe']:.3f} | "
            f"split-half={representation['tem']['population_split_half_stability']:.3f}/"
            f"{representation['hdgm']['population_split_half_stability']:.3f} | "
            f"choice={behavior['tem']:.3f}/{behavior['hdgm']:.3f}/{behavior['stateless']:.3f}"
        )

    fingerprints_after = {
        f"rep_{name}": tensor_state_fingerprint(rep_models[name]) for name in ("tem", "hdgm")
    }
    fingerprints_after.update({
        f"behavior_{name}": tensor_state_fingerprint(beh_models[name])
        for name in ("tem", "hdgm", "stateless")
    })
    if fingerprints_before != fingerprints_after:
        raise RuntimeError("frozen-model audit failed: at least one parameter changed")
    record = {
        "seed": int(seed),
        "representation_checkpoint": os.path.abspath(rep_path),
        "behavior_checkpoint": os.path.abspath(beh_path),
        "frozen_parameter_audit": "PASS",
        "conditions": conditions,
    }
    record["curves"] = difficulty_curves(record, dcfg)
    return record


def nested_value(record: dict, path: Sequence[str]) -> float:
    value = record
    for key in path:
        value = value[key]
    return float(value)


def summarize_difficulty(records: Sequence[dict], base_cfg: SplitterV7Config,
                         dcfg: DifficultyV2Config) -> dict:
    summary = {
        "base_config": asdict(base_cfg), "difficulty_config": asdict(dcfg),
        "seeds": [r["seed"] for r in records], "conditions": {}, "curves": {},
    }
    keys = list(records[0]["conditions"].keys())
    for key in keys:
        summary["conditions"][key] = {"representation": {}, "behavior": {}}
        for model in ("tem", "hdgm"):
            summary["conditions"][key]["representation"][model] = {}
            for metric in records[0]["conditions"][key]["representation"][model]:
                summary["conditions"][key]["representation"][model][metric] = summarize([
                    r["conditions"][key]["representation"][model][metric] for r in records
                ])
        for model in ("tem", "hdgm", "stateless"):
            summary["conditions"][key]["behavior"][model] = summarize([
                r["conditions"][key]["behavior"][model] for r in records
            ])
    for axis in ("alias", "noise"):
        summary["curves"][axis] = {"representation": {}, "behavior": {}}
        for model in ("tem", "hdgm"):
            summary["curves"][axis]["representation"][model] = {}
            for metric in records[0]["curves"]["representation"][axis][model]:
                summary["curves"][axis]["representation"][model][metric] = summarize([
                    r["curves"]["representation"][axis][model][metric] for r in records
                ])
        for model in ("tem", "hdgm", "stateless"):
            summary["curves"][axis]["behavior"][model] = {}
            for metric in records[0]["curves"]["behavior"][axis][model]:
                summary["curves"][axis]["behavior"][model][metric] = summarize([
                    r["curves"]["behavior"][axis][model][metric] for r in records
                ])

    # Four locked population-level primary contrasts; behavior remains secondary.
    primary_specs = {
        "alias_transfer_auc_hdgm_minus_tem":
            ("alias", "transfer_probe_auc"),
        "noise_transfer_auc_hdgm_minus_tem":
            ("noise", "transfer_probe_auc"),
        "alias_stability_auc_hdgm_minus_tem":
            ("alias", "stability_auc"),
        "noise_stability_auc_hdgm_minus_tem":
            ("noise", "stability_auc"),
    }
    tests = {}
    for label, (axis, metric) in primary_specs.items():
        hdgm = [r["curves"]["representation"][axis]["hdgm"][metric] for r in records]
        tem = [r["curves"]["representation"][axis]["tem"][metric] for r in records]
        tests[label] = exact_sign_flip_test(hdgm, tem)
    summary["primary_tests_holm"] = holm_adjust(tests)
    return summary


def print_difficulty_summary(summary: dict) -> None:
    print("\n" + "#" * 118)
    print("HDGM DIFFICULTY V2.1 FROZEN POPULATION-ROBUSTNESS SUMMARY")
    print("#" * 118)
    for axis in ("alias", "noise"):
        print(f"\n===== {axis.upper()} CURVE SUMMARY =====")
        for model in ("tem", "hdgm"):
            block = summary["curves"][axis]["representation"][model]
            print(
                f"{model.upper():4s} transfer AUC={block['transfer_probe_auc']['mean']:.4f} ± "
                f"{block['transfer_probe_auc']['sd']:.4f} | stability AUC="
                f"{block['stability_auc']['mean']:.4f} ± {block['stability_auc']['sd']:.4f} | "
                f"transfer slope={block['transfer_probe_slope']['mean']:.4f}"
            )
    print("\n===== LOCKED PRIMARY TESTS (HDGM - TEM; HOLM) =====")
    for label, result in summary["primary_tests_holm"].items():
        print(
            f"{label:43s} diff={result['difference']:.4f} | "
            f"p={result['p_raw']:.5f} | Holm={result['p_holm']:.5f} | dz={result['dz']:.3f}"
        )


def plot_difficulty_summary(summary: dict, dcfg: DifficultyV2Config) -> None:
    ensure_dir(dcfg.output_dir)
    colors = {"tem": "#475569", "hdgm": "#2E5BFF", "stateless": "#14B8A6"}
    for axis, values in (("alias", dcfg.alias_corruptions), ("noise", dcfg.noise_stds)):
        fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))
        for model in ("tem", "hdgm"):
            means = [summary["conditions"][condition_key(axis, float(v))]
                     ["representation"][model]["reference_transfer_probe"]["mean"] for v in values]
            stds = [summary["conditions"][condition_key(axis, float(v))]
                    ["representation"][model]["reference_transfer_probe"]["sd"] for v in values]
            axes[0].errorbar(values, means, yerr=stds, marker="o", linewidth=2,
                             capsize=3, color=colors[model], label=model.upper())
            means = [summary["conditions"][condition_key(axis, float(v))]
                     ["representation"][model]["population_split_half_stability"]["mean"] for v in values]
            stds = [summary["conditions"][condition_key(axis, float(v))]
                    ["representation"][model]["population_split_half_stability"]["sd"] for v in values]
            axes[1].errorbar(values, means, yerr=stds, marker="o", linewidth=2,
                             capsize=3, color=colors[model], label=model.upper())
        for model in ("tem", "hdgm", "stateless"):
            means = [summary["conditions"][condition_key(axis, float(v))]
                     ["behavior"][model]["mean"] for v in values]
            stds = [summary["conditions"][condition_key(axis, float(v))]
                    ["behavior"][model]["sd"] for v in values]
            axes[2].errorbar(values, means, yerr=stds, marker="o", linewidth=2,
                             capsize=3, color=colors[model], label=model.upper())
        axes[0].set_ylabel("Frozen reference-probe accuracy")
        axes[1].set_ylabel("Population split-half stability")
        axes[2].set_ylabel("Junction choice accuracy (secondary)")
        xlabel = "Additional alias corruption" if axis == "alias" else "Observation noise SD"
        for ax in axes:
            ax.set_xlabel(xlabel)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False)
        fig.suptitle(f"Frozen HDGM population robustness: {axis} manipulation")
        fig.tight_layout()
        fig.savefig(os.path.join(dcfg.output_dir, f"difficulty_v2_1_{axis}_scan.png"),
                    dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(dcfg.output_dir, f"difficulty_v2_1_{axis}_scan.pdf"),
                    bbox_inches="tight")
        plt.close(fig)


def run_difficulty_multiseed(seeds: Sequence[int],
                             base_cfg: Optional[SplitterV7Config] = None,
                             dcfg: Optional[DifficultyV2Config] = None) -> dict:
    base_cfg = SplitterV7Config() if base_cfg is None else base_cfg
    dcfg = DifficultyV2Config() if dcfg is None else dcfg
    if dcfg.representation_update != base_cfg.representation_update:
        raise ValueError("Difficulty representation endpoint must equal locked V7.3 endpoint")
    if dcfg.behavior_update != base_cfg.behavior_update:
        raise ValueError("Difficulty behavior endpoint must equal locked V7.3 endpoint")
    if dcfg.reference_alias_corruption not in dcfg.alias_corruptions:
        raise ValueError("reference_alias_corruption must be included in alias_corruptions")
    if dcfg.reference_noise not in dcfg.noise_stds:
        raise ValueError("reference_noise must be included in noise_stds")
    ensure_dir(dcfg.output_dir)
    print("=" * 118)
    print(f"START DIFFICULTY V2.1 | CONFIG={dcfg.name} | DEVICE={DEVICE}")
    print(
        f"SEEDS={list(seeds)} | ADDITIONAL_ALIAS={list(dcfg.alias_corruptions)} | "
        f"NOISE={list(dcfg.noise_stds)}"
    )
    print("Core training labels: junction only | shared-corridor labels in model loss: NO")
    print("TEM/HDGM updates during scan: NONE | fixed independent full-trial banks: YES")
    print("=" * 118)
    records = []
    for seed in seeds:
        record = run_difficulty_seed(int(seed), base_cfg, dcfg)
        records.append(record)
        with open(os.path.join(dcfg.output_dir, f"difficulty_v2_1_seed_{seed}.json"),
                  "w", encoding="utf-8") as handle:
            json.dump(json_safe(record), handle, ensure_ascii=False, indent=2)
    summary = summarize_difficulty(records, base_cfg, dcfg)
    print_difficulty_summary(summary)
    with open(os.path.join(dcfg.output_dir, "difficulty_v2_1_multiseed_results.json"),
              "w", encoding="utf-8") as handle:
        json.dump(json_safe({"records": records, "summary": summary}),
                  handle, ensure_ascii=False, indent=2)
    plot_difficulty_summary(summary, dcfg)
    print(f"\nSaved Difficulty V2.1 outputs to: {os.path.abspath(dcfg.output_dir)}")
    return summary


if __name__ == "__main__":
    # Locked local confirmatory analysis. The checkpoints already exist, so no
    # TEM/HDGM parameter is trained or updated in this script.
    RUN_MODE = "confirmatory"
    TRAIN_IF_MISSING = False

    DEVELOPMENT_SEEDS = [11, 22, 33]
    CONFIRMATORY_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
    if RUN_MODE == "development":
        SEEDS = DEVELOPMENT_SEEDS
        CHECKPOINT_DIR = (
            r"C:\Users\马明哲\Desktop\建模\pythonProject32"
            r"\hdgm_v7_3_splitter_outputs_development\checkpoints"
        )
        CHECKPOINT_HINT = "hdgm_v7_3_splitter_dual_endpoint_development"
        OUTPUT_DIR = (
            r"C:\Users\马明哲\Desktop\建模\pythonProject32"
            r"\hdgm_difficulty_v2_1_frozen_outputs_development"
        )
    elif RUN_MODE == "confirmatory":
        SEEDS = CONFIRMATORY_SEEDS
        CHECKPOINT_DIR = (
            r"C:\Users\马明哲\Desktop\建模\pythonProject32"
            r"\hdgm_v7_3_splitter_outputs_confirmatory\checkpoints"
        )
        CHECKPOINT_HINT = "hdgm_v7_3_splitter_dual_endpoint_confirmatory"
        OUTPUT_DIR = (
            r"C:\Users\马明哲\Desktop\建模\pythonProject32"
            r"\hdgm_difficulty_v2_1_frozen_outputs_confirmatory"
        )
    else:
        raise ValueError("RUN_MODE must be 'development' or 'confirmatory'")

    BASE_CONFIG = SplitterV7Config()
    DIFFICULTY_CONFIG = DifficultyV2Config(
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_name_hint=CHECKPOINT_HINT,
        output_dir=OUTPUT_DIR,
        train_if_missing=TRAIN_IF_MISSING,
    )
    run_difficulty_multiseed(SEEDS, BASE_CONFIG, DIFFICULTY_CONFIG)
