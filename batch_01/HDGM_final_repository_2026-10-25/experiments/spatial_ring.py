# -*- coding: utf-8 -*-
"""
HDGM Place V5 -- paper-complete unsupervised place-representation experiment.

The TEM/HDGM cores are trained only by next-observation prediction. Position
labels never enter either core or its training loss. After the selected fixed
checkpoint is loaded, all position analyses are fitted on frozen states using
independent train/validation/test episode banks.

Primary claim (limited deliberately): HDGM develops stable population-level
spatial organization on a partially observable ring track. The claim concerns
distributed firing-like geometry, not one-to-one place labels or proof that
individual hidden units are biological place cells.

Controls:
  * recurrent/associative history reset at every step (secondary control);
  * current observation + current action only;
  * elapsed time + current action only;
  * position-label permutation null;
  * TEM with the same predictive objective.

The old seeds 11/22/33 are development seeds. The default entry point uses ten
new confirmatory seeds and must not be tuned after their results are inspected.
"""

import copy
import itertools
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
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
class PlaceV5Config:
    # Partially observable ring
    track_len: int = 20
    obs_dim: int = 24
    sequence_len: int = 100
    cue_period: int = 5
    landmark_positions: Tuple[int, ...] = (0, 7, 13)
    landmark_strength: float = 0.85
    noise_std: float = 0.05
    p_forward: float = 0.45
    p_backward: float = 0.45
    p_stay: float = 0.10

    # Canonical paper-model dimensions/equations
    hidden_dim: int = 96
    action_emb_dim: int = 32
    ca3_steps: int = 5
    lambda_m: float = 0.92
    alpha: float = 0.32
    rho: float = 0.84
    c1: float = 0.20
    c2: float = 1.00
    beta: float = 0.10

    # Locked unsupervised training
    checkpoints: Tuple[int, ...] = (25, 50, 100, 150, 200, 300)
    representation_update: int = 150
    consolidation_update: int = 300
    batch_size: int = 24
    lr: float = 7e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # Fixed independent banks
    val_batches: int = 4
    test_batches: int = 8
    probe_train_batches: int = 8
    probe_val_batches: int = 4
    probe_test_batches: int = 8
    bank_batch_size: int = 32

    # Frozen probes
    probe_epochs: int = 180
    probe_lr: float = 1e-2
    probe_weight_decay: float = 1e-4
    probe_patience: int = 25
    probe_train_per_position: int = 240
    probe_val_per_position: int = 120
    probe_test_per_position: int = 160
    permutation_repeats: int = 10
    permutation_probe_epochs: int = 70
    field_shuffle_repeats: int = 20
    sensory_residual_ridge: float = 1e-2

    name: str = "hdgm_place_v5_unsupervised_frozen_probe"
    output_dir: str = "hdgm_place_v5_outputs"


class PartialObservableRing:
    """Ring with repeated local cues and sparse unique landmarks."""

    def __init__(self, cfg: PlaceV5Config, seed: int,
                 maps: Optional[Dict[str, np.ndarray]] = None):
        self.cfg = cfg
        self.rng = np.random.RandomState(seed)
        if maps is None:
            periodic = self.rng.randn(cfg.cue_period, cfg.obs_dim).astype(np.float32)
            landmark = self.rng.randn(cfg.track_len, cfg.obs_dim).astype(np.float32)
            pos_map = np.stack([periodic[p % cfg.cue_period] for p in range(cfg.track_len)])
            for p in cfg.landmark_positions:
                pos_map[p] = (
                    (1.0 - cfg.landmark_strength) * pos_map[p]
                    + cfg.landmark_strength * landmark[p]
                )
            self.maps = {"periodic": periodic, "landmark": landmark, "pos_map": pos_map}
        else:
            self.maps = {k: np.asarray(v, dtype=np.float32).copy() for k, v in maps.items()}
        self.action_dim = 3       # stay / forward / backward

    def clone(self, seed: int) -> "PartialObservableRing":
        return PartialObservableRing(self.cfg, seed, self.maps)

    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        c = self.cfg
        x = np.zeros((batch_size, c.sequence_len, c.obs_dim), np.float32)
        action = np.zeros((batch_size, c.sequence_len), np.int64)
        pos = np.zeros((batch_size, c.sequence_len), np.int64)
        probs = np.asarray([c.p_stay, c.p_forward, c.p_backward], dtype=np.float64)
        probs /= probs.sum()
        for b in range(batch_size):
            p = int(self.rng.randint(0, c.track_len))
            for t in range(c.sequence_len):
                pos[b, t] = p
                x[b, t] = self.maps["pos_map"][p] + self.rng.randn(c.obs_dim) * c.noise_std
                a = int(self.rng.choice(3, p=probs))
                action[b, t] = a
                if a == 1:
                    p = (p + 1) % c.track_len
                elif a == 2:
                    p = (p - 1) % c.track_len
        return {
            "x": torch.from_numpy(x),
            "action": torch.from_numpy(action),
            "pos": torch.from_numpy(pos),
        }


def to_device(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.to(DEVICE) for k, v in batch.items()}


def build_bank(env: PartialObservableRing, n: int, batch_size: int) -> List[dict]:
    return [env.sample_batch(batch_size) for _ in range(n)]


def make_banks(env: PartialObservableRing, cfg: PlaceV5Config, seed: int) -> Dict[str, List[dict]]:
    return {
        "val": build_bank(env.clone(seed + 10101), cfg.val_batches, cfg.bank_batch_size),
        "test": build_bank(env.clone(seed + 20202), cfg.test_batches, cfg.bank_batch_size),
        "probe_train": build_bank(env.clone(seed + 30303), cfg.probe_train_batches, cfg.bank_batch_size),
        "probe_val": build_bank(env.clone(seed + 40404), cfg.probe_val_batches, cfg.bank_batch_size),
        "probe_test": build_bank(env.clone(seed + 50505), cfg.probe_test_batches, cfg.bank_batch_size),
    }


def attractor(M: torch.Tensor, h: torch.Tensor, steps: int) -> torch.Tensor:
    for _ in range(steps):
        h = torch.tanh(0.5 * h + torch.bmm(M, h.unsqueeze(2)).squeeze(2))
    return h


def apply_row_reset(state: torch.Tensor, reset: Optional[torch.Tensor]) -> torch.Tensor:
    if reset is None:
        return state
    shape = [reset.shape[0]] + [1] * (state.ndim - 1)
    return state * (~reset).to(state.dtype).reshape(shape)


class TEMCore(nn.Module):
    def __init__(self, cfg: PlaceV5Config):
        super().__init__()
        h = cfg.hidden_dim
        self.cfg = cfg
        self.fc_x = nn.Linear(cfg.obs_dim, h)
        self.action_emb = nn.Embedding(3, cfg.action_emb_dim)
        self.fc_a = nn.Linear(cfg.action_emb_dim, h)
        self.fc_g = nn.Linear(h, h)
        self.decoder = nn.Linear(h, cfg.obs_dim)
        # Retained for architecture/state-dict parity; never receives labels.
        self.pos_head = nn.Linear(h, cfg.track_len)
        self.phase_head = nn.Linear(h, 6)

    def forward(self, x: torch.Tensor, action: torch.Tensor,
                reset_mask: Optional[torch.Tensor] = None,
                return_details: bool = False) -> Dict[str, torch.Tensor]:
        b, tmax, _ = x.shape
        hdim = self.cfg.hidden_dim
        g = x.new_zeros(b, hdim)
        M = x.new_zeros(b, hdim, hdim)
        pred_loss = x.new_zeros(())
        mem, pred, attr, read, xh, gh, pl, ph = [], [], [], [], [], [], [], []
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
            pred_loss = pred_loss + (x_hat - x[:, t + 1]).square().mean()
            M = self.cfg.lambda_m * M + torch.bmm(p_t.unsqueeze(2), p_t.unsqueeze(1))
            mem.append(p_hat); pred.append(g_pred); attr.append(p_t); read.append(p_hat)
            xh.append(x_hat); gh.append(g)
            pl.append(self.pos_head(p_hat)); ph.append(self.phase_head(p_hat))
        return {
            "pred_loss": pred_loss / max(1, tmax - 1),
            "mem": torch.stack(mem, 1),
            "predictive_state": torch.stack(pred, 1),
            "attractor_state": torch.stack(attr, 1),
            "associative_read": torch.stack(read, 1),
            "xhat": torch.stack(xh, 1),
            "g_hist": torch.stack(gh, 1),
            "pos_logits_hist": torch.stack(pl, 1),
            "phase_logits_hist": torch.stack(ph, 1),
        }


class HDGMCore(nn.Module):
    def __init__(self, cfg: PlaceV5Config):
        super().__init__()
        h = cfg.hidden_dim
        self.cfg = cfg
        self.fc_x = nn.Linear(cfg.obs_dim, h)
        self.action_emb = nn.Embedding(3, cfg.action_emb_dim)
        self.fc_a = nn.Linear(cfg.action_emb_dim, h)
        self.fc_g = nn.Linear(h, h)
        self.decoder = nn.Linear(h, cfg.obs_dim)
        self.gate_ec = nn.Linear(cfg.obs_dim + h + h, cfg.obs_dim)
        self.gate_ca3 = nn.Linear(h * 2, h)
        self.gate_ca1 = nn.Linear(h * 3, h)
        self.fb_proj = nn.Linear(h, h)
        self.pos_head = nn.Linear(h, cfg.track_len)
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
        mem, pred, attr, read, fb = [], [], [], [], []
        ec_h, ca3_h, ca1_h, write_h, xh, gh, pl, ph = [], [], [], [], [], [], [], []
        for t in range(tmax - 1):
            r = None if reset_mask is None else reset_mask[:, t]
            g, M, v = apply_row_reset(g, r), apply_row_reset(M, r), apply_row_reset(v, r)
            ec = torch.sigmoid(self.gate_ec(torch.cat([x[:, t], g, v], 1)))
            x_emb = torch.tanh(self.fc_x(ec * x[:, t]))
            a_emb = self.fc_a(self.action_emb(action[:, t]))
            g_pred = torch.tanh(self.fc_g(g) + a_emb)
            hstate, ca3_steps = x_emb, []
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
            pred_loss = pred_loss + (x_hat - x[:, t + 1]).square().mean()
            v = self.cfg.rho * v + self.fb_proj(p_out)
            write = ec.mean(1, keepdim=True).unsqueeze(-1)
            M = self.cfg.lambda_m * M + write * torch.bmm(p_t.unsqueeze(2), p_t.unsqueeze(1))
            mem.append(p_out); pred.append(g_pred); attr.append(p_t); read.append(p_hat); fb.append(v)
            ec_h.append(ec); ca3_h.append(torch.stack(ca3_steps, 1).mean(1)); ca1_h.append(ca1)
            write_h.append(write.squeeze(-1).squeeze(-1)); xh.append(x_hat); gh.append(g)
            pl.append(self.pos_head(p_out)); ph.append(self.phase_head(p_out))
        return {
            "pred_loss": pred_loss / max(1, tmax - 1),
            "mem": torch.stack(mem, 1),
            "predictive_state": torch.stack(pred, 1),
            "attractor_state": torch.stack(attr, 1),
            "associative_read": torch.stack(read, 1),
            "feedback_state": torch.stack(fb, 1),
            "ec_gate_pattern": torch.stack(ec_h, 1),
            "ca3_gate_pattern": torch.stack(ca3_h, 1),
            "ca1_gate_pattern": torch.stack(ca1_h, 1),
            "xhat": torch.stack(xh, 1), "g_hist": torch.stack(gh, 1),
            "ec_hist": torch.stack(ec_h, 1), "ca3_hist": torch.stack(ca3_h, 1),
            "ca1_hist": torch.stack(ca1_h, 1), "write_hist": torch.stack(write_h, 1),
            "v_hist": torch.stack(fb, 1),
            "pos_logits_hist": torch.stack(pl, 1),
            "phase_logits_hist": torch.stack(ph, 1),
        }


class LinearPositionProbe(nn.Module):
    def __init__(self, in_dim: int, n_pos: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_pos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def prediction_loss(model: nn.Module, bank: Sequence[dict]) -> float:
    model.eval()
    return float(np.mean([model(**{k: v.to(DEVICE) for k, v in b.items() if k in ("x", "action")})["pred_loss"].item() for b in bank]))


def instantiate_models(cfg: PlaceV5Config) -> Dict[str, nn.Module]:
    # Identical seeds make all shape-compatible shared tensors identical.
    set_seed(91001); tem = TEMCore(cfg).to(DEVICE)
    set_seed(91001); hdgm = HDGMCore(cfg).to(DEVICE)
    return {"tem": tem, "hdgm": hdgm}


def architecture_audit(cfg: PlaceV5Config, models: Dict[str, nn.Module], batch: dict) -> dict:
    required_tem = {"pred_loss", "mem", "predictive_state", "attractor_state",
                    "associative_read", "xhat", "g_hist", "pos_logits_hist", "phase_logits_hist"}
    required_hdgm = required_tem | {"feedback_state", "ec_gate_pattern", "ca3_gate_pattern",
                                   "ca1_gate_pattern", "ec_hist", "ca3_hist", "ca1_hist",
                                   "write_hist", "v_hist"}
    result = {}
    b = to_device(batch)
    for name, required in (("tem", required_tem), ("hdgm", required_hdgm)):
        out = models[name](b["x"], b["action"], return_details=True)
        missing = sorted(required - set(out))
        if missing:
            raise RuntimeError(f"{name} output contract missing: {missing}")
        result[name] = {"parameters": parameter_count(models[name]), "outputs": sorted(required)}
    # Explicitly document that no position tensor is accepted by forward.
    result["position_supervision_in_core"] = False
    return result


def train_seed(seed: int, cfg: PlaceV5Config) -> Tuple[Dict[str, nn.Module], dict, dict, dict]:
    set_seed(seed)
    env = PartialObservableRing(cfg, seed + 17)
    banks = make_banks(env, cfg, seed)
    models = instantiate_models(cfg)
    audit = architecture_audit(cfg, models, env.clone(seed + 60606).sample_batch(8))
    opts = {k: optim.Adam(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            for k, m in models.items()}
    history = {k: [] for k in models}
    snapshots: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}
    max_update = max(cfg.checkpoints)
    print("=" * 112)
    print(f"PLACE V5 | SEED={seed} | REPRESENTATION={cfg.representation_update} | CONSOLIDATION={cfg.consolidation_update}")
    print("Preflight: architecture=PASS | no-position-supervision=PASS | fixed-banks=PASS")
    print("Parameters | " + " | ".join(f"{k}={parameter_count(v):,}" for k, v in models.items()))
    for update in range(1, max_update + 1):
        b = to_device(env.sample_batch(cfg.batch_size))
        losses = {}
        for name, model in models.items():
            model.train(); opts[name].zero_grad(set_to_none=True)
            out = model(b["x"], b["action"], return_details=True)
            loss = out["pred_loss"]
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip); opts[name].step()
            losses[name] = float(loss.detach().item())
        if update in cfg.checkpoints:
            val = {k: prediction_loss(m, banks["val"]) for k, m in models.items()}
            for k in models:
                history[k].append({"update": update, "train_prediction_loss": losses[k],
                                   "fixed_val_prediction_loss": val[k]})
            snapshots[update] = {k: state_dict_cpu(m) for k, m in models.items()}
            print(f"Update {update:04d}/{max_update} | train pred TEM/HDGM={losses['tem']:.4f}/{losses['hdgm']:.4f} | fixed-val={val['tem']:.4f}/{val['hdgm']:.4f}")
    return models, banks, snapshots, {"seed": seed, "history": history, "preflight": audit}


def flatten_output(out: dict, batch: dict, rep: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = out[rep].detach().cpu().numpy()
    # Canonical state at transition t is analysed against the reached position t+1.
    y = batch["pos"][:, 1:].numpy()
    b, t, d = z.shape
    trial = np.repeat(np.arange(b)[:, None], t, axis=1)
    time = np.repeat(np.arange(t)[None, :], b, axis=0)
    obs_act = np.concatenate([
        batch["x"][:, :-1].numpy(),
        np.eye(3, dtype=np.float32)[batch["action"][:, :-1].numpy()],
    ], axis=-1)
    return z.reshape(-1, d), y.reshape(-1), trial.reshape(-1), time.reshape(-1), obs_act.reshape(-1, obs_act.shape[-1])


@torch.no_grad()
def collect_dataset(model: nn.Module, bank: Sequence[dict], rep: str,
                    reset_history: bool = False) -> Dict[str, np.ndarray]:
    model.eval()
    xs, ys, groups, times, obs = [], [], [], [], []
    offset = 0
    for cpu in bank:
        b = to_device(cpu)
        reset = torch.ones_like(b["action"][:, :-1], dtype=torch.bool) if reset_history else None
        out = model(b["x"], b["action"], reset_mask=reset, return_details=True)
        X, y, group, time, oa = flatten_output(out, cpu, rep)
        xs.append(X); ys.append(y); groups.append(group + offset); times.append(time); obs.append(oa)
        offset += cpu["x"].shape[0]
    return {"X": np.concatenate(xs), "y": np.concatenate(ys),
            "group": np.concatenate(groups), "time": np.concatenate(times),
            "obs_action": np.concatenate(obs)}


def timing_action_features(data: Dict[str, np.ndarray], cfg: PlaceV5Config) -> np.ndarray:
    t = data["time"].astype(np.float32)
    phase = 2.0 * np.pi * t / max(1, cfg.track_len)
    action_one_hot = data["obs_action"][:, -3:].astype(np.float32)
    return np.concatenate([
        np.stack([t / max(1, cfg.sequence_len - 1), np.sin(phase), np.cos(phase)], 1),
        action_one_hot,
    ], axis=1)


def circular_distance(pred: np.ndarray, target: np.ndarray, n: int) -> np.ndarray:
    d = np.abs(pred - target)
    return np.minimum(d, n - d)


def fit_probe(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray,
              test_x: np.ndarray, test_y: np.ndarray, cfg: PlaceV5Config,
              seed: int, epochs: Optional[int] = None) -> Dict[str, float]:
    set_seed(seed)
    rng = np.random.RandomState(seed)
    def balanced_index(labels: np.ndarray, per_position: int) -> np.ndarray:
        pieces = []
        for p in range(cfg.track_len):
            candidates = np.flatnonzero(labels == p)
            if len(candidates) == 0:
                raise RuntimeError(f"probe split contains no samples for position {p}")
            take = min(per_position, len(candidates))
            pieces.append(rng.choice(candidates, size=take, replace=False))
        idx = np.concatenate(pieces); rng.shuffle(idx); return idx
    ti = balanced_index(train_y, cfg.probe_train_per_position)
    vi = balanced_index(val_y, cfg.probe_val_per_position)
    qi = balanced_index(test_y, cfg.probe_test_per_position)
    train_x, train_y = train_x[ti], train_y[ti]
    val_x, val_y = val_x[vi], val_y[vi]
    test_x, test_y = test_x[qi], test_y[qi]
    mean, sd = train_x.mean(0, keepdims=True), train_x.std(0, keepdims=True) + 1e-6
    tx = torch.tensor((train_x - mean) / sd, dtype=torch.float32, device=DEVICE)
    vx = torch.tensor((val_x - mean) / sd, dtype=torch.float32, device=DEVICE)
    qx = torch.tensor((test_x - mean) / sd, dtype=torch.float32, device=DEVICE)
    ty = torch.tensor(train_y, dtype=torch.long, device=DEVICE)
    vy = torch.tensor(val_y, dtype=torch.long, device=DEVICE)
    qy = torch.tensor(test_y, dtype=torch.long, device=DEVICE)
    probe = LinearPositionProbe(tx.shape[1], cfg.track_len).to(DEVICE)
    opt = optim.Adam(probe.parameters(), lr=cfg.probe_lr, weight_decay=cfg.probe_weight_decay)
    best, best_state, stale = -1.0, None, 0
    for _ in range(cfg.probe_epochs if epochs is None else epochs):
        probe.train(); opt.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(probe(tx), ty); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            score = float((probe(vx).argmax(1) == vy).float().mean().item())
        if score > best + 1e-8:
            best, best_state, stale = score, state_dict_cpu(probe), 0
        else:
            stale += 1
        if stale >= cfg.probe_patience:
            break
    probe.load_state_dict(best_state); probe.eval()
    with torch.no_grad():
        pred = probe(qx).argmax(1).cpu().numpy(); true = qy.cpu().numpy()
    dist = circular_distance(pred, true, cfg.track_len)
    cm = np.zeros((cfg.track_len, cfg.track_len), dtype=np.int64)
    for target, estimate in zip(true, pred):
        cm[int(target), int(estimate)] += 1
    return {"exact_accuracy": float(np.mean(pred == true)),
            "tolerance1_accuracy": float(np.mean(dist <= 1)),
            "mean_circular_error": float(np.mean(dist)),
            "confusion_matrix": cm.tolist()}


def rate_maps(X: np.ndarray, y: np.ndarray, n_pos: int) -> Tuple[np.ndarray, np.ndarray]:
    # Shift every unit to nonnegative activity without changing spatial ordering.
    activity = X - X.min(axis=0, keepdims=True) + 1e-8
    maps = np.zeros((activity.shape[1], n_pos), np.float64)
    occupancy = np.zeros(n_pos, np.float64)
    for p in range(n_pos):
        mask = y == p; occupancy[p] = mask.sum()
        if mask.any(): maps[:, p] = activity[mask].mean(0)
    occupancy /= occupancy.sum() + 1e-12
    return maps, occupancy


def field_metrics(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  cfg: PlaceV5Config) -> Dict[str, float]:
    maps, occ = rate_maps(X, y, cfg.track_len)
    mean_rate = (maps * occ[None]).sum(1) + 1e-12
    ratio = maps / mean_rate[:, None]
    info = (occ[None] * ratio * np.log2(ratio + 1e-12)).sum(1)
    sparsity = np.square(mean_rate) / ((np.square(maps) * occ[None]).sum(1) + 1e-12)
    selectivity = maps.max(1) / mean_rate
    # Independent odd/even-trial field stability.
    unique = np.unique(groups); half_a = np.isin(groups, unique[::2]); half_b = ~half_a
    map_a, _ = rate_maps(X[half_a], y[half_a], cfg.track_len)
    map_b, _ = rate_maps(X[half_b], y[half_b], cfg.track_len)
    corr = []
    for u in range(maps.shape[0]):
        if map_a[u].std() > 1e-8 and map_b[u].std() > 1e-8:
            corr.append(np.corrcoef(map_a[u], map_b[u])[0, 1])
    # Width above half maximum on a circular track.
    widths = (maps >= 0.5 * maps.max(1, keepdims=True)).sum(1)
    return {"spatial_information": float(np.nanmean(info)),
            "spatial_sparsity": float(np.nanmean(sparsity)),
            "peak_mean_selectivity": float(np.nanmean(selectivity)),
            "split_half_stability": float(np.nanmean(corr)) if corr else float("nan"),
            "halfmax_field_width": float(np.nanmean(widths))}


def exemplar_place_fields(X: np.ndarray, y: np.ndarray, cfg: PlaceV5Config,
                          n_units: int = 12) -> Dict[str, object]:
    maps, occ = rate_maps(X, y, cfg.track_len)
    mean_rate = (maps * occ[None]).sum(1) + 1e-12
    ratio = maps / mean_rate[:, None]
    info = (occ[None] * ratio * np.log2(ratio + 1e-12)).sum(1)
    selected = np.argsort(info)[::-1][:min(n_units, len(info))]
    chosen = maps[selected]
    chosen = chosen / (chosen.max(1, keepdims=True) + 1e-12)
    return {"unit_indices": selected.tolist(), "spatial_information": info[selected].tolist(),
            "normalized_rate_maps": chosen.tolist()}


def shuffled_information(X: np.ndarray, y: np.ndarray, cfg: PlaceV5Config, seed: int) -> float:
    rng, vals = np.random.RandomState(seed), []
    for _ in range(cfg.field_shuffle_repeats):
        yp = rng.permutation(y)
        vals.append(field_metrics(X, yp, np.arange(len(yp)), cfg)["spatial_information"])
    return float(np.mean(vals))


def circular_geometry(X: np.ndarray, y: np.ndarray, cfg: PlaceV5Config) -> Dict[str, object]:
    means = np.stack([X[y == p].mean(0) for p in range(cfg.track_len)])
    by_delta = {}
    for delta in range(cfg.track_len // 2 + 1):
        by_delta[str(delta)] = float(np.mean([
            np.linalg.norm(means[p] - means[(p + delta) % cfg.track_len])
            for p in range(cfg.track_len)
        ]))
    local = np.mean([by_delta["1"], by_delta["2"]])
    far = np.mean([by_delta[str(d)] for d in range(max(3, cfg.track_len // 4), cfg.track_len // 2 + 1)])
    centered = means - means.mean(0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    xy = centered @ vt[:2].T
    return {"local_distance": float(local), "far_distance": float(far),
            "local_far_ratio": float(local / (far + 1e-12)),
            "distance_by_delta": by_delta, "pca_xy": xy.tolist(),
            "pca_variance_2d": float((s[:2] ** 2).sum() / ((s ** 2).sum() + 1e-12))}


def average_ranks(x: np.ndarray) -> np.ndarray:
    """Dependency-free average ranks, used for the population geometry test."""
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    if a.size < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def population_geometry_metrics(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                                cfg: PlaceV5Config) -> Dict[str, float]:
    """Population-level spatial code; no single-neuron place-cell claim is required."""
    X = np.asarray(X, np.float64)
    y, groups = np.asarray(y), np.asarray(groups)
    Xz = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
    centroids = np.stack([Xz[y == p].mean(0) for p in range(cfg.track_len)])
    unit_centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    similarity = unit_centroids @ unit_centroids.T

    neural_distance, spatial_distance = [], []
    adjacent_similarity, distant_similarity = [], []
    for p in range(cfg.track_len):
        for q in range(p + 1, cfg.track_len):
            delta = min(abs(p - q), cfg.track_len - abs(p - q))
            spatial_distance.append(delta)
            neural_distance.append(np.linalg.norm(centroids[p] - centroids[q]))
            if delta == 1:
                adjacent_similarity.append(similarity[p, q])
            if delta >= cfg.track_len // 4:
                distant_similarity.append(similarity[p, q])

    spatial_distance = np.asarray(spatial_distance, np.float64)
    neural_distance = np.asarray(neural_distance, np.float64)
    geometry_spearman = safe_corr(average_ranks(spatial_distance), average_ranks(neural_distance))

    unique = np.unique(groups)
    half_a, half_b = np.isin(groups, unique[::2]), np.isin(groups, unique[1::2])
    map_a = np.stack([Xz[half_a & (y == p)].mean(0) for p in range(cfg.track_len)])
    map_b = np.stack([Xz[half_b & (y == p)].mean(0) for p in range(cfg.track_len)])
    position_reliability = [safe_corr(map_a[p], map_b[p]) for p in range(cfg.track_len)]
    position_reliability = np.asarray(position_reliability, np.float64)

    within = np.linalg.norm(Xz - centroids[y], axis=1).mean()
    far = np.mean([neural_distance[i] for i, d in enumerate(spatial_distance)
                   if d >= cfg.track_len // 4])
    covariance = np.cov(centroids, rowvar=False)
    eig = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    participation = (eig.sum() ** 2) / (np.square(eig).sum() + 1e-12)

    return {
        "population_map_stability": safe_corr(map_a, map_b),
        "position_vector_reliability": float(np.nanmean(position_reliability)),
        "adjacent_similarity": float(np.mean(adjacent_similarity)),
        "distant_similarity": float(np.mean(distant_similarity)),
        "topology_contrast": float(np.mean(adjacent_similarity) - np.mean(distant_similarity)),
        "distance_geometry_spearman": geometry_spearman,
        "far_centroid_to_within_ratio": float(far / (within + 1e-12)),
        "centroid_participation_ratio": float(participation),
    }


def shuffled_population_geometry(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                                 cfg: PlaceV5Config, seed: int) -> Dict[str, float]:
    """Structure-preserving nulls for population reliability and ring order.

    Split-half maps retain their activity statistics, but the position rows in
    the second half are permuted. Ring-geometry nulls permute centroid order.
    This avoids the anti-correlation induced by independently relabelling all
    samples after global centering.
    """
    rng = np.random.RandomState(seed)
    X = np.asarray(X, np.float64)
    y, groups = np.asarray(y), np.asarray(groups)
    Xz = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
    unique = np.unique(groups)
    half_a, half_b = np.isin(groups, unique[::2]), np.isin(groups, unique[1::2])
    map_a = np.stack([Xz[half_a & (y == p)].mean(0) for p in range(cfg.track_len)])
    map_b = np.stack([Xz[half_b & (y == p)].mean(0) for p in range(cfg.track_len)])
    centroids = np.stack([Xz[y == p].mean(0) for p in range(cfg.track_len)])
    values = {k: [] for k in ("population_map_stability", "position_vector_reliability",
                              "topology_contrast", "distance_geometry_spearman")}
    for _ in range(cfg.field_shuffle_repeats):
        perm = rng.permutation(cfg.track_len)
        values["population_map_stability"].append(safe_corr(map_a, map_b[perm]))
        values["position_vector_reliability"].append(float(np.nanmean([
            safe_corr(map_a[p], map_b[perm[p]]) for p in range(cfg.track_len)])))
        ordered = centroids[rng.permutation(cfg.track_len)]
        norms = ordered / (np.linalg.norm(ordered, axis=1, keepdims=True) + 1e-12)
        similarity = norms @ norms.T
        spatial, neural, adjacent, distant = [], [], [], []
        for p in range(cfg.track_len):
            for q in range(p + 1, cfg.track_len):
                delta = min(abs(p - q), cfg.track_len - abs(p - q))
                spatial.append(delta); neural.append(np.linalg.norm(ordered[p] - ordered[q]))
                if delta == 1: adjacent.append(similarity[p, q])
                if delta >= cfg.track_len // 4: distant.append(similarity[p, q])
        values["topology_contrast"].append(float(np.mean(adjacent) - np.mean(distant)))
        values["distance_geometry_spearman"].append(
            safe_corr(average_ranks(np.asarray(spatial)), average_ranks(np.asarray(neural))))
    return {k: float(np.nanmean(v)) for k, v in values.items()}


def fit_sensory_action_residualizer(train: Dict[str, np.ndarray],
                                    cfg: PlaceV5Config) -> Dict[str, np.ndarray]:
    """Fit X_state = f(current observation, current action) on probe-train only."""
    F = np.asarray(train["obs_action"], np.float64)
    X = np.asarray(train["X"], np.float64)
    f_mean, f_sd = F.mean(0, keepdims=True), F.std(0, keepdims=True) + 1e-8
    x_mean = X.mean(0, keepdims=True)
    Fs, Xc = (F - f_mean) / f_sd, X - x_mean
    gram = Fs.T @ Fs
    ridge = cfg.sensory_residual_ridge * np.eye(gram.shape[0], dtype=np.float64)
    weights = np.linalg.solve(gram + ridge, Fs.T @ Xc)
    return {"feature_mean": f_mean, "feature_sd": f_sd,
            "state_mean": x_mean, "weights": weights}


def sensory_action_residual(data: Dict[str, np.ndarray], fit: Dict[str, np.ndarray]) -> np.ndarray:
    F = (np.asarray(data["obs_action"], np.float64) - fit["feature_mean"]) / fit["feature_sd"]
    predicted = fit["state_mean"] + F @ fit["weights"]
    return np.asarray(data["X"], np.float64) - predicted


def alias_transition_labels(data: Dict[str, np.ndarray], cfg: PlaceV5Config
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return valid mask, sensory-action condition, and absolute reached position.

    State t is labelled by reached position t+1. Current position is recovered
    from reached position and action, solely for constructing an analysis mask;
    it never enters a model forward pass or core loss.
    """
    action = np.asarray(data["obs_action"][:, -3:]).argmax(1)
    delta = np.asarray([0, 1, -1], dtype=np.int64)[action]
    target = np.asarray(data["y"], dtype=np.int64)
    current = (target - delta) % cfg.track_len
    valid = ~np.isin(current, np.asarray(cfg.landmark_positions, dtype=np.int64))
    condition = (current % cfg.cue_period) * 3 + action
    return valid, condition.astype(np.int64), target


def alias_population_metrics(X: np.ndarray, data: Dict[str, np.ndarray],
                             cfg: PlaceV5Config) -> Dict[str, float]:
    """Separate absolute positions while local cue class and action are matched."""
    valid, condition, target = alias_transition_labels(data, cfg)
    X = np.asarray(X, np.float64)[valid]
    condition, target = condition[valid], target[valid]
    groups = np.asarray(data["group"])[valid]
    Xz = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
    unique_trials = np.unique(groups)
    half_a = np.isin(groups, unique_trials[::2]); half_b = np.isin(groups, unique_trials[1::2])
    same_scores, wrong_scores, separation_ratios = [], [], []
    valid_conditions = 0
    for c in np.unique(condition):
        mask_c = condition == c
        positions = np.unique(target[mask_c])
        if len(positions) < 2:
            continue
        if any(np.sum(mask_c & half_a & (target == p)) < 2 or
               np.sum(mask_c & half_b & (target == p)) < 2 for p in positions):
            continue
        valid_conditions += 1
        ma = np.stack([Xz[mask_c & half_a & (target == p)].mean(0) for p in positions])
        mb = np.stack([Xz[mask_c & half_b & (target == p)].mean(0) for p in positions])
        ma = ma / (np.linalg.norm(ma, axis=1, keepdims=True) + 1e-12)
        mb = mb / (np.linalg.norm(mb, axis=1, keepdims=True) + 1e-12)
        cross = ma @ mb.T
        same_scores.extend(np.diag(cross).tolist())
        wrong_scores.extend(cross[~np.eye(len(positions), dtype=bool)].tolist())
        centroids = np.stack([Xz[mask_c & (target == p)].mean(0) for p in positions])
        between = np.mean([np.linalg.norm(centroids[i] - centroids[j])
                           for i in range(len(positions)) for j in range(i + 1, len(positions))])
        lookup = {int(p): centroids[i] for i, p in enumerate(positions)}
        within = np.mean([np.linalg.norm(z - lookup[int(p)])
                          for z, p in zip(Xz[mask_c], target[mask_c])])
        separation_ratios.append(between / (within + 1e-12))
    return {
        "alias_identity_same_similarity": float(np.mean(same_scores)),
        "alias_identity_wrong_similarity": float(np.mean(wrong_scores)),
        "alias_identity_contrast": float(np.mean(same_scores) - np.mean(wrong_scores)),
        "alias_centroid_to_within_ratio": float(np.mean(separation_ratios)),
        "alias_conditions_evaluated": int(valid_conditions),
        "alias_samples_evaluated": int(valid.sum()),
    }


def shuffled_alias_null(X: np.ndarray, data: Dict[str, np.ndarray], cfg: PlaceV5Config,
                        seed: int) -> Dict[str, float]:
    """Shuffle absolute positions only within matched sensory-action conditions."""
    rng = np.random.RandomState(seed)
    valid, condition, _ = alias_transition_labels(data, cfg)
    X = np.asarray(X)
    vals = {"alias_identity_contrast": [], "alias_centroid_to_within_ratio": []}
    for _ in range(cfg.field_shuffle_repeats):
        shuffled_x = X.copy()
        for c in np.unique(condition[valid]):
            idx = np.flatnonzero(valid & (condition == c))
            shuffled_x[idx] = shuffled_x[rng.permutation(idx)]
        m = alias_population_metrics(shuffled_x, data, cfg)
        for k in vals: vals[k].append(m[k])
    return {k: float(np.nanmean(v)) for k, v in vals.items()}


def permuted_probe(full: Dict[str, Dict[str, np.ndarray]], cfg: PlaceV5Config, seed: int) -> float:
    rng, vals = np.random.RandomState(seed), []
    null_cfg = copy.copy(cfg); null_cfg.probe_epochs = cfg.permutation_probe_epochs
    for r in range(cfg.permutation_repeats):
        train_y = rng.permutation(full["train"]["y"])
        val_y = rng.permutation(full["val"]["y"])
        score = fit_probe(full["train"]["X"], train_y,
                          full["val"]["X"], val_y,
                          full["test"]["X"], full["test"]["y"],
                          null_cfg, seed + r + 1, epochs=cfg.permutation_probe_epochs)
        vals.append(score["exact_accuracy"])
    return float(np.mean(vals))


def probe_suite(model: nn.Module, banks: Dict[str, List[dict]], rep: str,
                cfg: PlaceV5Config, seed: int) -> Dict[str, object]:
    full = {s: collect_dataset(model, banks[f"probe_{s}"], rep, False)
            for s in ("train", "val", "test")}
    reset = {s: collect_dataset(model, banks[f"probe_{s}"], rep, True)
             for s in ("train", "val", "test")}
    def run(source: Dict[str, Dict[str, np.ndarray]], key: str, offset: int) -> dict:
        return fit_probe(source["train"][key], source["train"]["y"],
                         source["val"][key], source["val"]["y"],
                         source["test"][key], source["test"]["y"], cfg, seed + offset)
    time_data = {}
    for s in ("train", "val", "test"):
        time_data[s] = dict(full[s]); time_data[s]["time_action"] = timing_action_features(full[s], cfg)
    sensory_fit = fit_sensory_action_residualizer(full["train"], cfg)
    reset_sensory_fit = fit_sensory_action_residualizer(reset["train"], cfg)
    residual = {s: dict(full[s]) for s in ("train", "val", "test")}
    reset_residual = {s: dict(reset[s]) for s in ("train", "val", "test")}
    for s in ("train", "val", "test"):
        residual[s]["X"] = sensory_action_residual(full[s], sensory_fit)
        reset_residual[s]["X"] = sensory_action_residual(reset[s], reset_sensory_fit)
    result = {
        "full_history": run(full, "X", 1),
        "history_reset": run(reset, "X", 2),
        "observation_action_only": run(full, "obs_action", 3),
        "timing_action_only": run(time_data, "time_action", 4),
        "label_permutation_accuracy": permuted_probe(full, cfg, seed + 100),
        "field_metrics": field_metrics(full["test"]["X"], full["test"]["y"], full["test"]["group"], cfg),
        "shuffled_spatial_information": shuffled_information(full["test"]["X"], full["test"]["y"], cfg, seed + 200),
        "population_geometry": population_geometry_metrics(
            full["test"]["X"], full["test"]["y"], full["test"]["group"], cfg),
        "reset_population_geometry": population_geometry_metrics(
            reset["test"]["X"], reset["test"]["y"], reset["test"]["group"], cfg),
        "shuffled_population_geometry": shuffled_population_geometry(
            full["test"]["X"], full["test"]["y"], full["test"]["group"], cfg, seed + 300),
        "sensory_residual_probe": run(residual, "X", 5),
        "sensory_residual_population_geometry": population_geometry_metrics(
            residual["test"]["X"], residual["test"]["y"], residual["test"]["group"], cfg),
        "sensory_residual_shuffled_population_geometry": shuffled_population_geometry(
            residual["test"]["X"], residual["test"]["y"], residual["test"]["group"], cfg, seed + 400),
        "alias_population": alias_population_metrics(
            full["test"]["X"], full["test"], cfg),
        "alias_population_shuffled": shuffled_alias_null(
            full["test"]["X"], full["test"], cfg, seed + 500),
        "sensory_residual_alias_population": alias_population_metrics(
            residual["test"]["X"], residual["test"], cfg),
        "sensory_residual_alias_shuffled": shuffled_alias_null(
            residual["test"]["X"], residual["test"], cfg, seed + 600),
        "reset_sensory_residual_alias_population": alias_population_metrics(
            reset_residual["test"]["X"], reset_residual["test"], cfg),
        "geometry": circular_geometry(full["test"]["X"], full["test"]["y"], cfg),
        "example_place_fields": exemplar_place_fields(
            full["test"]["X"], full["test"]["y"], cfg
        ),
    }
    return result


def layerwise_suite(model: nn.Module, banks: Dict[str, List[dict]], reps: Sequence[str],
                    cfg: PlaceV5Config, seed: int) -> Dict[str, dict]:
    result = {}
    for i, rep in enumerate(reps):
        print(f"    frozen layer probe: {rep}")
        data = {s: collect_dataset(model, banks[f"probe_{s}"], rep, False)
                for s in ("train", "val", "test")}
        result[rep] = fit_probe(data["train"]["X"], data["train"]["y"],
                                data["val"]["X"], data["val"]["y"],
                                data["test"]["X"], data["test"]["y"], cfg, seed + i)
    return result


def exact_sign_flip(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    d = np.asarray(a, float) - np.asarray(b, float); observed = abs(float(d.mean()))
    if len(d) <= 20:
        null = [abs(float(np.mean(d * np.asarray(s))))
                for s in itertools.product((-1.0, 1.0), repeat=len(d))]
        p = float(np.mean(np.asarray(null) >= observed - 1e-15))
    else:
        rng = np.random.RandomState(12345); signs = rng.choice((-1.0, 1.0), (200000, len(d)))
        p = float((1 + np.sum(np.abs((signs * d).mean(1)) >= observed)) / 200001)
    rng = np.random.RandomState(24680); idx = rng.randint(0, len(d), (20000, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    dz = float(d.mean() / (d.std(ddof=1) + 1e-12)) if len(d) > 1 else float("nan")
    return {"difference": float(d.mean()), "ci95_low": float(lo), "ci95_high": float(hi),
            "p_raw": p, "dz": dz}


def holm(tests: Dict[str, dict]) -> Dict[str, dict]:
    names = sorted(tests, key=lambda k: tests[k]["p_raw"]); running, m = 0.0, len(names)
    for rank, name in enumerate(names):
        running = max(running, min(1.0, (m - rank) * tests[name]["p_raw"]))
        tests[name]["p_holm"] = running
    return tests


def summarize(values: Sequence[float]) -> Dict[str, object]:
    x = np.asarray(values, float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "values": x.tolist()}


def save_checkpoint(path: str, cfg: PlaceV5Config, seed: int, update: int, states: dict) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save({"config": asdict(cfg), "seed": seed, "update": update, "models": states}, path)


def load_frozen_seed(seed: int, cfg: PlaceV5Config) -> Tuple[Dict[str, nn.Module], dict, dict, dict]:
    """Rebuild deterministic banks and load already-trained checkpoints."""
    set_seed(seed)
    env = PartialObservableRing(cfg, seed + 17)
    banks = make_banks(env, cfg, seed)
    models = instantiate_models(cfg)
    snapshots = {}
    ckpt_dir = os.path.join(cfg.output_dir, "checkpoints")
    for update in cfg.checkpoints:
        path = os.path.join(ckpt_dir, f"{cfg.name}_seed_{seed}_update_{update}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing frozen checkpoint: {path}")
        payload = torch.load(path, map_location="cpu")
        snapshots[update] = payload["models"]
    audit = architecture_audit(cfg, models, env.clone(seed + 60606).sample_batch(8))
    print("Loaded frozen checkpoints; TEM/HDGM parameters will not be updated.")
    return models, banks, snapshots, {"seed": seed, "history": {}, "preflight": audit,
                                      "evaluation_only": True}


def evaluate_seed(models: Dict[str, nn.Module], banks: dict, snapshots: dict,
                  record: dict, seed: int, cfg: PlaceV5Config) -> dict:
    # Persist every frozen representation checkpoint before running any probe.
    # Probe/statistics failures must not discard an otherwise completed training run.
    ckpt = ensure_dir(os.path.join(cfg.output_dir, "checkpoints"))
    for update, states in snapshots.items():
        save_checkpoint(os.path.join(ckpt, f"{cfg.name}_seed_{seed}_update_{update}.pt"),
                        cfg, seed, update, states)
    print(f"Saved frozen checkpoints before probe analyses: {ckpt}")

    record["prediction_test_by_checkpoint"] = {k: {} for k in models}
    for update in cfg.checkpoints:
        for name, model in models.items():
            model.load_state_dict(snapshots[update][name])
            record["prediction_test_by_checkpoint"][name][str(update)] = prediction_loss(model, banks["test"])
    # Locked primary representation snapshot.
    for name, model in models.items():
        model.load_state_dict(snapshots[cfg.representation_update][name])
    record["primary_probes"] = {}
    for name, model in models.items():
        print(f"Running frozen {name.upper()} place controls ...")
        record["primary_probes"][name] = probe_suite(model, banks, "mem", cfg,
                                                       seed + (0 if name == "tem" else 1000))
    record["layerwise"] = {
        "tem": layerwise_suite(models["tem"], banks,
                               ("mem", "g_hist", "predictive_state", "attractor_state", "associative_read"),
                               cfg, seed + 2000),
        "hdgm": layerwise_suite(models["hdgm"], banks,
                                ("mem", "g_hist", "predictive_state", "attractor_state", "associative_read",
                                 "feedback_state", "ec_gate_pattern", "ca3_gate_pattern", "ca1_gate_pattern"),
                                cfg, seed + 3000),
    }
    # Secondary consolidation endpoint: full-history memory probe only.
    for name, model in models.items():
        model.load_state_dict(snapshots[cfg.consolidation_update][name])
    record["consolidation_probe"] = {}
    for i, (name, model) in enumerate(models.items()):
        data = {s: collect_dataset(model, banks[f"probe_{s}"], "mem", False)
                for s in ("train", "val", "test")}
        record["consolidation_probe"][name] = fit_probe(
            data["train"]["X"], data["train"]["y"], data["val"]["X"], data["val"]["y"],
            data["test"]["X"], data["test"]["y"], cfg, seed + 4000 + i)
    return record


def run_seed(seed: int, cfg: PlaceV5Config, evaluation_only: bool = False) -> dict:
    payload = load_frozen_seed(seed, cfg) if evaluation_only else train_seed(seed, cfg)
    return evaluate_seed(*payload, seed, cfg)


def aggregate(records: Sequence[dict], cfg: PlaceV5Config) -> dict:
    out = {"config": asdict(cfg), "seeds": [r["seed"] for r in records],
           "analysis_contract": {
               "primary_update": cfg.representation_update,
               "primary": ["HDGM sensory-residual population-map stability > structured null",
                           "HDGM sensory-residual alias centroid/within ratio > within-condition null"],
               "secondary": ["HDGM spatial information > shuffled-position null",
                             "HDGM vs TEM population geometry", "full-history vs reset controls",
                             "linear position decoding", "ring topology", "layer localization",
                             "update-300 consolidation"]}}
    out["probe_controls"] = {}
    for model in ("tem", "hdgm"):
        out["probe_controls"][model] = {}
        for condition in ("full_history", "history_reset", "observation_action_only", "timing_action_only"):
            for metric in ("exact_accuracy", "tolerance1_accuracy", "mean_circular_error"):
                out["probe_controls"][model][f"{condition}_{metric}"] = summarize(
                    [r["primary_probes"][model][condition][metric] for r in records])
        out["probe_controls"][model]["label_permutation_accuracy"] = summarize(
            [r["primary_probes"][model]["label_permutation_accuracy"] for r in records])
    field_keys = ("spatial_information", "spatial_sparsity", "peak_mean_selectivity",
                  "split_half_stability", "halfmax_field_width")
    out["field_metrics"] = {m: {k: summarize([r["primary_probes"][m]["field_metrics"][k] for r in records])
                                      for k in field_keys} for m in ("tem", "hdgm")}
    out["shuffled_spatial_information"] = {m: summarize([
        r["primary_probes"][m]["shuffled_spatial_information"] for r in records]) for m in ("tem", "hdgm")}
    population_keys = ("population_map_stability", "position_vector_reliability",
                       "adjacent_similarity", "distant_similarity", "topology_contrast",
                       "distance_geometry_spearman", "far_centroid_to_within_ratio",
                       "centroid_participation_ratio")
    null_population_keys = ("population_map_stability", "position_vector_reliability",
                            "topology_contrast", "distance_geometry_spearman")
    out["population_geometry"] = {m: {k: summarize([
        r["primary_probes"][m]["population_geometry"][k] for r in records])
        for k in population_keys} for m in ("tem", "hdgm")}
    out["reset_population_geometry"] = {m: {k: summarize([
        r["primary_probes"][m]["reset_population_geometry"][k] for r in records])
        for k in population_keys} for m in ("tem", "hdgm")}
    out["shuffled_population_geometry"] = {m: {k: summarize([
        r["primary_probes"][m]["shuffled_population_geometry"][k] for r in records])
        for k in null_population_keys} for m in ("tem", "hdgm")}
    out["sensory_residual_probe"] = {m: {k: summarize([
        r["primary_probes"][m]["sensory_residual_probe"][k] for r in records])
        for k in ("exact_accuracy", "tolerance1_accuracy", "mean_circular_error")}
        for m in ("tem", "hdgm")}
    out["sensory_residual_population_geometry"] = {m: {k: summarize([
        r["primary_probes"][m]["sensory_residual_population_geometry"][k] for r in records])
        for k in population_keys} for m in ("tem", "hdgm")}
    out["sensory_residual_shuffled_population_geometry"] = {m: {k: summarize([
        r["primary_probes"][m]["sensory_residual_shuffled_population_geometry"][k] for r in records])
        for k in null_population_keys} for m in ("tem", "hdgm")}
    alias_keys = ("alias_identity_same_similarity", "alias_identity_wrong_similarity",
                  "alias_identity_contrast", "alias_centroid_to_within_ratio",
                  "alias_conditions_evaluated", "alias_samples_evaluated")
    alias_null_keys = ("alias_identity_contrast", "alias_centroid_to_within_ratio")
    for source in ("alias_population", "sensory_residual_alias_population",
                   "reset_sensory_residual_alias_population"):
        out[source] = {m: {k: summarize([
            r["primary_probes"][m][source][k] for r in records]) for k in alias_keys}
            for m in ("tem", "hdgm")}
    for source in ("alias_population_shuffled", "sensory_residual_alias_shuffled"):
        out[source] = {m: {k: summarize([
            r["primary_probes"][m][source][k] for r in records]) for k in alias_null_keys}
            for m in ("tem", "hdgm")}
    out["layerwise"] = {}
    for model in ("tem", "hdgm"):
        out["layerwise"][model] = {rep: {metric: summarize([
            r["layerwise"][model][rep][metric] for r in records])
            for metric in ("exact_accuracy", "tolerance1_accuracy", "mean_circular_error")}
            for rep in records[0]["layerwise"][model]}
    out["consolidation_probe"] = {m: {metric: summarize([
        r["consolidation_probe"][m][metric] for r in records])
        for metric in ("exact_accuracy", "tolerance1_accuracy", "mean_circular_error")}
        for m in ("tem", "hdgm")}
    out["prediction_test_by_checkpoint"] = {m: {str(u): summarize([
        r["prediction_test_by_checkpoint"][m][str(u)] for r in records]) for u in cfg.checkpoints}
        for m in ("tem", "hdgm")}
    primary = {
        "HDGM residual population stability vs structured null": exact_sign_flip(
            [r["primary_probes"]["hdgm"]["sensory_residual_population_geometry"]["population_map_stability"] for r in records],
            [r["primary_probes"]["hdgm"]["sensory_residual_shuffled_population_geometry"]["population_map_stability"] for r in records]),
        "HDGM residual alias centroid/within ratio vs null": exact_sign_flip(
            [r["primary_probes"]["hdgm"]["sensory_residual_alias_population"]["alias_centroid_to_within_ratio"] for r in records],
            [r["primary_probes"]["hdgm"]["sensory_residual_alias_shuffled"]["alias_centroid_to_within_ratio"] for r in records]),
    }
    out["primary_statistics_holm"] = holm(primary)
    out["secondary_statistics"] = {
        "HDGM spatial information vs shuffled null": exact_sign_flip(
            [r["primary_probes"]["hdgm"]["field_metrics"]["spatial_information"] for r in records],
            [r["primary_probes"]["hdgm"]["shuffled_spatial_information"] for r in records]),
        "HDGM vs TEM full-history exact decoding": exact_sign_flip(
            [r["primary_probes"]["hdgm"]["full_history"]["exact_accuracy"] for r in records],
            [r["primary_probes"]["tem"]["full_history"]["exact_accuracy"] for r in records]),
        "HDGM vs TEM split-half stability": exact_sign_flip(
            [r["primary_probes"]["hdgm"]["field_metrics"]["split_half_stability"] for r in records],
            [r["primary_probes"]["tem"]["field_metrics"]["split_half_stability"] for r in records]),
        "HDGM vs TEM population-map stability": exact_sign_flip(
            [r["primary_probes"]["hdgm"]["population_geometry"]["population_map_stability"] for r in records],
            [r["primary_probes"]["tem"]["population_geometry"]["population_map_stability"] for r in records]),
        "HDGM full vs reset population-map stability": exact_sign_flip(
            [r["primary_probes"]["hdgm"]["population_geometry"]["population_map_stability"] for r in records],
            [r["primary_probes"]["hdgm"]["reset_population_geometry"]["population_map_stability"] for r in records]),
        "HDGM topology contrast vs structured null": exact_sign_flip(
            [r["primary_probes"]["hdgm"]["population_geometry"]["topology_contrast"] for r in records],
            [r["primary_probes"]["hdgm"]["shuffled_population_geometry"]["topology_contrast"] for r in records]),
        "HDGM residual alias contrast vs TEM": exact_sign_flip(
            [r["primary_probes"]["hdgm"]["sensory_residual_alias_population"]["alias_identity_contrast"] for r in records],
            [r["primary_probes"]["tem"]["sensory_residual_alias_population"]["alias_identity_contrast"] for r in records]),
        "HDGM residual alias contrast full vs reset": exact_sign_flip(
            [r["primary_probes"]["hdgm"]["sensory_residual_alias_population"]["alias_identity_contrast"] for r in records],
            [r["primary_probes"]["hdgm"]["reset_sensory_residual_alias_population"]["alias_identity_contrast"] for r in records]),
        "HDGM residual alias identity contrast vs null": exact_sign_flip(
            [r["primary_probes"]["hdgm"]["sensory_residual_alias_population"]["alias_identity_contrast"] for r in records],
            [r["primary_probes"]["hdgm"]["sensory_residual_alias_shuffled"]["alias_identity_contrast"] for r in records]),
        "HDGM vs TEM residual population-map stability": exact_sign_flip(
            [r["primary_probes"]["hdgm"]["sensory_residual_population_geometry"]["population_map_stability"] for r in records],
            [r["primary_probes"]["tem"]["sensory_residual_population_geometry"]["population_map_stability"] for r in records]),
    }
    return out


def fmt(v: dict) -> str:
    return f"{v['mean']:.4f} ± {v['sd']:.4f}"


def print_summary(s: dict) -> None:
    print("\n" + "#" * 112)
    print("HDGM PLACE V5 UNSUPERVISED FROZEN-PROBE SUMMARY")
    print("#" * 112)
    print(f"\n===== Frozen position controls @ update {s['analysis_contract']['primary_update']} =====")
    for model, metrics in s["probe_controls"].items():
        print(model.upper())
        for k, v in metrics.items(): print(f"  {k:48s}: {fmt(v)}")
    print("\n===== Standard place-field metrics =====")
    for model, metrics in s["field_metrics"].items():
        print(model.upper())
        for k, v in metrics.items(): print(f"  {k:32s}: {fmt(v)}")
        print(f"  {'shuffled_spatial_information':32s}: {fmt(s['shuffled_spatial_information'][model])}")
    print("\n===== Primary population-level spatial organization =====")
    for model, metrics in s["population_geometry"].items():
        print(model.upper())
        for k, v in metrics.items():
            print(f"  {k:38s}: {fmt(v)}")
        print("  -- history-reset population control --")
        for k in ("population_map_stability", "position_vector_reliability",
                  "topology_contrast", "distance_geometry_spearman"):
            print(f"  reset_{k:32s}: {fmt(s['reset_population_geometry'][model][k])}")
        print("  -- shuffled-position null --")
        for k, v in s["shuffled_population_geometry"][model].items():
            print(f"  shuffled_{k:29s}: {fmt(v)}")
    print("\n===== Sensory-action residual population code =====")
    for model in ("tem", "hdgm"):
        print(model.upper())
        probe = s["sensory_residual_probe"][model]
        print(f"  residual exact decoding                 : {fmt(probe['exact_accuracy'])}")
        print(f"  residual tolerance-1 decoding           : {fmt(probe['tolerance1_accuracy'])}")
        for k, v in s["sensory_residual_population_geometry"][model].items():
            print(f"  residual_{k:29s}: {fmt(v)}")
        print("  -- structure-preserving residual null --")
        for k, v in s["sensory_residual_shuffled_population_geometry"][model].items():
            print(f"  residual_null_{k:24s}: {fmt(v)}")
    print("\n===== Alias-matched population separation =====")
    for model in ("tem", "hdgm"):
        print(model.upper())
        for source, label in (("alias_population", "raw"),
                              ("sensory_residual_alias_population", "residual"),
                              ("reset_sensory_residual_alias_population", "reset-residual")):
            m = s[source][model]
            print(f"  {label:14s} contrast={fmt(m['alias_identity_contrast'])} | "
                  f"centroid/within={fmt(m['alias_centroid_to_within_ratio'])}")
        null = s["sensory_residual_alias_shuffled"][model]
        print(f"  {'residual-null':14s} contrast={fmt(null['alias_identity_contrast'])} | "
              f"centroid/within={fmt(null['alias_centroid_to_within_ratio'])}")
    print("\n===== Frozen layer-wise position decoding =====")
    for model, reps in s["layerwise"].items():
        print(model.upper())
        for rep, metrics in reps.items():
            print(f"  {rep:24s} exact={fmt(metrics['exact_accuracy'])} | tol1={fmt(metrics['tolerance1_accuracy'])}")
    print(f"\n===== Secondary consolidation probe @ update {s['config']['consolidation_update']} =====")
    for model, metrics in s["consolidation_probe"].items():
        print(f"{model:8s} exact={fmt(metrics['exact_accuracy'])} | tol1={fmt(metrics['tolerance1_accuracy'])}")
    print("\n===== Primary exact paired sign-flip tests + Holm =====")
    for name, v in s["primary_statistics_holm"].items():
        print(f"{name:50s} | diff={v['difference']:.4f} CI95=[{v['ci95_low']:.4f},{v['ci95_high']:.4f}] p={v['p_raw']:.5f} Holm={v['p_holm']:.5f} dz={v['dz']:.3f}")
    print("\n===== Secondary paired tests (unadjusted) =====")
    for name, v in s["secondary_statistics"].items():
        print(f"{name:50s} | diff={v['difference']:.4f} CI95=[{v['ci95_low']:.4f},{v['ci95_high']:.4f}] p={v['p_raw']:.5f} dz={v['dz']:.3f}")
    print("\nGuardrails: population geometry is primary; exact decoding is a control. "
          "Chance exact position accuracy is 0.05; position labels never train TEM/HDGM.")


def plot_results(records: Sequence[dict], summary: dict, cfg: PlaceV5Config) -> None:
    ensure_dir(cfg.output_dir)
    # Confirmatory population endpoints after removing current sensory-action effects.
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5))
    models, colors = ("tem", "hdgm"), ("#2E5BFF", "#F59E0B")
    for ax, metric, real_source, null_source, title in (
        (axes[0], "population_map_stability", "sensory_residual_population_geometry",
         "sensory_residual_shuffled_population_geometry", "Residual population-map stability"),
        (axes[1], "alias_centroid_to_within_ratio", "sensory_residual_alias_population",
         "sensory_residual_alias_shuffled", "Alias centroid / within-position spread")):
        xx = np.arange(2)
        real = [summary[real_source][m][metric]["mean"] for m in models]
        real_sd = [summary[real_source][m][metric]["sd"] for m in models]
        null = [summary[null_source][m][metric]["mean"] for m in models]
        ax.bar(xx - .18, real, .36, yerr=real_sd, color=colors, alpha=.9, label="Observed")
        ax.bar(xx + .18, null, .36, color="#94A3B8", alpha=.85, label="Structured null")
        ax.axhline(0, color="#475569", linewidth=.8)
        ax.set_xticks(xx, [m.upper() for m in models]); ax.set_title(title)
    axes[0].set_ylabel("Population metric")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.output_dir, "place_v5_primary_population_endpoints.png"), dpi=240)
    plt.close(fig)

    # Descriptive full-state population-level spatial organization.
    pop_labels = ["Map stability", "Position-vector\nreliability",
                  "Topology contrast", "Distance-geometry\nSpearman"]
    pop_keys = ["population_map_stability", "position_vector_reliability",
                "topology_contrast", "distance_geometry_spearman"]
    xpop = np.arange(len(pop_labels)); width_pop = 0.34
    plt.figure(figsize=(8.5, 4.8))
    for j, (m, color) in enumerate((("tem", "#2E5BFF"), ("hdgm", "#F59E0B"))):
        means = [summary["population_geometry"][m][k]["mean"] for k in pop_keys]
        sds = [summary["population_geometry"][m][k]["sd"] for k in pop_keys]
        plt.bar(xpop + (j - .5) * width_pop, means, width_pop, yerr=sds,
                color=color, alpha=.88, label=m.upper())
    nulls = [summary["shuffled_population_geometry"]["hdgm"][k]["mean"] for k in pop_keys]
    plt.scatter(xpop, nulls, marker="x", s=55, color="#475569", label="HDGM shuffled null", zorder=5)
    plt.axhline(0.0, color="#475569", linewidth=.8)
    plt.xticks(xpop, pop_labels); plt.ylabel("Population spatial organization")
    plt.legend(frameon=False); plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "place_v5_population_geometry.png"), dpi=240); plt.close()
    # Frozen control comparison.
    labels = ["Full", "Reset", "Obs+act", "Time+act", "Permuted"]
    keys = ["full_history_exact_accuracy", "history_reset_exact_accuracy",
            "observation_action_only_exact_accuracy", "timing_action_only_exact_accuracy",
            "label_permutation_accuracy"]
    x = np.arange(len(labels)); width = 0.34
    plt.figure(figsize=(8.0, 4.8))
    for j, (m, color) in enumerate((("tem", "#2E5BFF"), ("hdgm", "#F59E0B"))):
        means = [summary["probe_controls"][m][k]["mean"] for k in keys]
        sds = [summary["probe_controls"][m][k]["sd"] for k in keys]
        plt.bar(x + (j - .5) * width, means, width, yerr=sds, color=color, alpha=.88, label=m.upper())
    plt.axhline(1 / cfg.track_len, color="#475569", linestyle="--", label="chance")
    plt.xticks(x, labels); plt.ylabel("Frozen exact position decoding"); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(os.path.join(cfg.output_dir, "place_v5_probe_controls.png"), dpi=240); plt.close()
    # Predictive acquisition curve.
    plt.figure(figsize=(7.2, 4.6))
    for m, color in (("tem", "#2E5BFF"), ("hdgm", "#F59E0B")):
        means = [summary["prediction_test_by_checkpoint"][m][str(u)]["mean"] for u in cfg.checkpoints]
        sds = [summary["prediction_test_by_checkpoint"][m][str(u)]["sd"] for u in cfg.checkpoints]
        plt.plot(cfg.checkpoints, means, marker="o", color=color, label=m.upper())
        plt.fill_between(cfg.checkpoints, np.asarray(means)-sds, np.asarray(means)+sds, color=color, alpha=.18)
    plt.axvline(cfg.representation_update, color="#14B8A6", linestyle=":", label="representation endpoint")
    plt.axvline(cfg.consolidation_update, color="#475569", linestyle="-.", label="consolidation endpoint")
    plt.xlabel("Unsupervised training updates"); plt.ylabel("Held-out next-observation MSE")
    plt.legend(frameon=False); plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "place_v5_predictive_learning.png"), dpi=240); plt.close()
    # HDGM layer localization.
    reps = list(summary["layerwise"]["hdgm"])
    means = [summary["layerwise"]["hdgm"][r]["exact_accuracy"]["mean"] for r in reps]
    sds = [summary["layerwise"]["hdgm"][r]["exact_accuracy"]["sd"] for r in reps]
    plt.figure(figsize=(9.5, 4.8)); plt.bar(np.arange(len(reps)), means, yerr=sds, color="#14B8A6")
    plt.axhline(1/cfg.track_len, color="#475569", linestyle="--")
    plt.xticks(np.arange(len(reps)), reps, rotation=28, ha="right"); plt.ylabel("Exact position decoding")
    plt.tight_layout(); plt.savefig(os.path.join(cfg.output_dir, "place_v5_layerwise.png"), dpi=240); plt.close()
    # Descriptive single-unit fields from the last seed; every row is normalized
    # independently and selected by spatial information on the held-out bank.
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.2), sharey=True)
    for ax, model, title in zip(axes, ("tem", "hdgm"), ("TEM", "HDGM")):
        fields = np.asarray(records[-1]["primary_probes"][model]["example_place_fields"]["normalized_rate_maps"])
        im = ax.imshow(fields, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
        ax.set_title(title); ax.set_xlabel("Ring position")
        ax.set_xticks(np.arange(0, cfg.track_len, 2))
    axes[0].set_ylabel("Held-out unit (ranked by spatial information)")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Normalized activity", shrink=.82)
    fig.subplots_adjust(left=.08, right=.90, bottom=.12, top=.90, wspace=.12)
    fig.savefig(os.path.join(cfg.output_dir, "place_v5_example_fields.png"), dpi=240)
    plt.close(fig)


def json_safe(obj):
    if isinstance(obj, dict): return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)): return obj.item()
    return obj


def run_multiseed(seeds: Sequence[int], cfg: Optional[PlaceV5Config] = None,
                  evaluation_only: bool = False) -> dict:
    cfg = PlaceV5Config() if cfg is None else cfg
    if cfg.representation_update not in cfg.checkpoints or cfg.consolidation_update not in cfg.checkpoints:
        raise ValueError("Both locked endpoints must be fixed checkpoints")
    if cfg.representation_update >= cfg.consolidation_update:
        raise ValueError("representation_update must precede consolidation_update")
    ensure_dir(cfg.output_dir)
    print("=" * 112)
    print(f"START PLACE V5 MULTI-SEED | CONFIG={cfg.name} | DEVICE={DEVICE}")
    print(f"SEEDS={list(seeds)} | CHECKPOINTS={list(cfg.checkpoints)} | POSITION LABELS IN CORE LOSS: NO")
    print("=" * 112)
    records = []
    for seed in seeds:
        rec = run_seed(int(seed), cfg, evaluation_only=evaluation_only); records.append(rec)
        with open(os.path.join(cfg.output_dir, f"place_v5_seed_{seed}.json"), "w", encoding="utf-8") as f:
            json.dump(json_safe(rec), f, ensure_ascii=False, indent=2)
    summary = aggregate(records, cfg); print_summary(summary)
    with open(os.path.join(cfg.output_dir, "place_v5_multiseed_results.json"), "w", encoding="utf-8") as f:
        json.dump(json_safe({"records": records, "summary": summary}), f, ensure_ascii=False, indent=2)
    plot_results(records, summary, cfg)
    print(f"\nSaved outputs to: {os.path.abspath(cfg.output_dir)}")
    return summary


if __name__ == "__main__":
    CONFIG = PlaceV5Config()
    # Locked paper run. Do not tune configuration after inspecting these seeds.
    # Use development_evaluate_only only to reproduce the three development seeds.
    RUN_MODE = "confirmatory"
    DEVELOPMENT_SEEDS = [11, 22, 33]
    CONFIRMATORY_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
    EVALUATION_ONLY = False
    if RUN_MODE in ("development", "development_evaluate_only"):
        SEEDS = DEVELOPMENT_SEEDS; CONFIG.name += "_development"; CONFIG.output_dir += "_development"
        EVALUATION_ONLY = RUN_MODE.endswith("evaluate_only")
    elif RUN_MODE == "confirmatory":
        SEEDS = CONFIRMATORY_SEEDS; CONFIG.name += "_confirmatory"; CONFIG.output_dir += "_confirmatory"
    else:
        raise ValueError("RUN_MODE must be 'development', 'development_evaluate_only', or 'confirmatory'")
    run_multiseed(SEEDS, CONFIG, evaluation_only=EVALUATION_ONLY)
