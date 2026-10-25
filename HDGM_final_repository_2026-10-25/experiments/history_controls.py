"""
Standalone frozen-evaluation suite for HDGM Figure 2.

This file contains the exact task, TEMLite, HDGM and branch-readout definitions
used by the V4 primary-150 experiment. It does NOT retrain TEM or HDGM. It
recursively discovers the original checkpoint files even when their uploaded
filenames are random, deduplicates them by embedded seed, restores the frozen
150-update snapshots, and runs the missing controlled evaluations.

Main outputs
------------
1. figure2_history_controls.csv
   - full_history
   - complete_history_reset
   - decision_observation_only
   - timing_only
2. figure2_shortcut_diagnostics.csv
   - independently fitted linear observation-only and timing-only null models
3. figure2_delay_transfer.csv
   - frozen-model transfer after inserting matched-observation hold steps into
     both delay periods
4. figure2_endpoint_predictions.npz
5. figure2_missing_summary.json
6. figure2_history_controls.png / figure2_delay_transfer.png

The statistical unit is the independently trained seed. No model weights are
updated in the frozen TEM/HDGM analyses.

Minimal command
---------------
    python figure2_missing_experiments_standalone.py --search-root .

Typical server command
----------------------
    CUDA_VISIBLE_DEVICES=1 python figure2_missing_experiments_standalone.py \
        --search-root /path/containing/checkpoints \
        --output-dir figure2_missing_outputs \
        --device cuda

The checkpoint files do not need to be renamed or moved into one directory.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class ExperimentConfig:
    use_branch_readout: bool = True
    use_cueA_aux_head: bool = False
    use_cueB_aux_head: bool = False
    use_phase_aux_head: bool = False
    use_ec_phase_rank_loss: bool = True
    use_position_aux_loss: bool = False
    use_cueB_margin_loss: bool = True
    use_cueB_peak_write_loss: bool = False
    gate_phase_weight: float = 0.0
    gate_cueB_margin_weight: float = 0.08
    gate_ec_phase_rank_weight: float = 0.12
    gate_pos_aux_weight: float = 0.0
    gate_cueA_aux_weight: float = 0.0
    gate_cueB_aux_weight: float = 0.0
    gate_phase_aux_weight: float = 0.0
    gate_cueB_peak_weight: float = 0.0
    tem_pos_aux_weight: float = 0.0
    gate_class_weights: tuple = (1.0, 1.0, 1.0, 1.0)
    epochs: int = 150
    batch_size: int = 32
    lr: float = 6e-4
    primary_updates: int = 150
    rapid_checkpoints: tuple = (10, 25, 50, 75, 100, 150)
    rapid_probe_checkpoints: tuple = (50, 100, 150)
    acquisition_thresholds: tuple = (0.50, 0.60)
    extended_updates: int = 500
    extended_checkpoints: tuple = (10, 25, 50, 75, 100, 150, 300, 500)
    val_batches: int = 10
    val_batch_size: int = 64
    val_score_alpha: float = 0.75
    val_interval: int = 10
    print_interval: int = 50
    test_batches: int = 50
    test_batch_size: int = 64
    analysis_batches: int = 5
    analysis_batch_size: int = 64
    split_seed_version: int = 4
    checkpoint_dir: str = "paper_rapid_v4_outputs/checkpoints"
    probe_train_batches: int = 20
    probe_val_batches: int = 5
    probe_test_batches: int = 20
    probe_batch_size: int = 64
    probe_epochs: int = 200
    probe_lr: float = 1e-2
    probe_weight_decay: float = 1e-4
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
    save_dir: str = "paper_rapid_v4_outputs"
    representative_seed: int = 11


def config_from_dict(raw: dict) -> ExperimentConfig:
    allowed = {f.name for f in fields(ExperimentConfig)}
    clean = {k: v for k, v in raw.items() if k in allowed}
    tuple_fields = {
        "gate_class_weights", "rapid_checkpoints", "rapid_probe_checkpoints",
        "acquisition_thresholds", "extended_checkpoints", "cueA_start_range",
        "delay1_range", "delay2_range",
    }
    for key in tuple_fields:
        if key in clean:
            clean[key] = tuple(clean[key])
    return ExperimentConfig(**clean)


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
        self.track_len = int(track_len)
        self.obs_dim = int(obs_dim)
        self.cue_width = int(cue_width)
        self.cueA_start_range = tuple(cueA_start_range)
        self.delay1_range = tuple(delay1_range)
        self.delay2_range = tuple(delay2_range)
        self.n_cues = int(n_cues)
        self.alias_prob = float(alias_prob)
        self.noise_std = float(noise_std)
        self.rng = np.random.RandomState(seed) if seed is not None else None
        self.pos_map = self._randn(track_len, obs_dim).astype(np.float32)
        for i in range(track_len):
            if self._rand() < alias_prob:
                self.pos_map[i] = self.pos_map[self._randint(0, track_len)]
        self.cueA_map = self._randn(n_cues, obs_dim).astype(np.float32)
        self.cueB_map = self._randn(n_cues, obs_dim).astype(np.float32)
        self.action_dim = 1
        self.forward_action = 0

    def _randint(self, low, high=None):
        return np.random.randint(low, high) if self.rng is None else self.rng.randint(low, high)

    def _randn(self, *shape):
        return np.random.randn(*shape) if self.rng is None else self.rng.randn(*shape)

    def _rand(self):
        return np.random.rand() if self.rng is None else self.rng.rand()

    @staticmethod
    def context_rule(cueA, cueB):
        return int(cueA * 2 + cueB)

    def _sample_valid_structure(self):
        for _ in range(500):
            a0 = self._randint(self.cueA_start_range[0], self.cueA_start_range[1] + 1)
            d1 = self._randint(self.delay1_range[0], self.delay1_range[1] + 1)
            d2 = self._randint(self.delay2_range[0], self.delay2_range[1] + 1)
            za = (a0, a0 + self.cue_width)
            b0 = za[1] + d1
            zb = (b0, b0 + self.cue_width)
            decision = zb[1] + d2
            if decision < self.track_len:
                return za, zb, decision, d1, d2
        raise RuntimeError("Cannot sample a legal trial; check track_len and delay ranges.")

    def sample_trial(
        self,
        forced_cueA=None,
        forced_cueB=None,
        delay1_override=None,
        delay2_override=None,
        mask_decision_obs=False,
    ):
        cueA = self._randint(0, self.n_cues) if forced_cueA is None else int(forced_cueA)
        cueB = self._randint(0, self.n_cues) if forced_cueB is None else int(forced_cueB)
        if delay1_override is None and delay2_override is None:
            za, zb, decision, d1, d2 = self._sample_valid_structure()
        else:
            a0 = self._randint(self.cueA_start_range[0], self.cueA_start_range[1] + 1)
            d1 = int(delay1_override)
            d2 = int(delay2_override)
            za = (a0, a0 + self.cue_width)
            zb = (za[1] + d1, za[1] + d1 + self.cue_width)
            decision = zb[1] + d2
            if decision >= self.track_len:
                raise ValueError("Requested delay does not fit the base track.")
        label = self.context_rule(cueA, cueB)
        xs, acts, pos, phase = [], [], [], []
        for t in range(self.track_len):
            x = self.pos_map[t].copy()
            in_a = za[0] <= t < za[1]
            in_b = zb[0] <= t < zb[1]
            in_d = t == decision
            if in_a:
                x += self.cueA_map[cueA]
                ph = 1
            elif in_b:
                x += self.cueB_map[cueB]
                ph = 3
            elif in_d:
                ph = 5
            elif t < za[0]:
                ph = 0
            elif za[1] <= t < zb[0]:
                ph = 2
            elif zb[1] <= t < decision:
                ph = 4
            else:
                ph = 0
            if mask_decision_obs and in_d:
                x = np.zeros_like(x, dtype=np.float32)
            x += self._randn(self.obs_dim).astype(np.float32) * self.noise_std
            xs.append(x)
            acts.append(self.forward_action)
            pos.append(t)
            phase.append(ph)
        return (
            np.asarray(xs, np.float32), np.asarray(acts, np.int64),
            np.asarray(pos, np.int64), np.asarray(phase, np.int64), int(label),
            int(cueA), int(cueB), tuple(za), tuple(zb), int(decision),
            int(d1), int(d2),
        )

    def sample_batch(self, batch_size, forced_cueAs, forced_cueBs, delay1_overrides, delay2_overrides):
        rows = [self.sample_trial(forced_cueAs[i], forced_cueBs[i], delay1_overrides[i], delay2_overrides[i]) for i in range(batch_size)]
        return (
            torch.tensor(np.stack([r[0] for r in rows]), dtype=torch.float32),
            torch.tensor(np.stack([r[1] for r in rows]), dtype=torch.long),
            torch.tensor(np.stack([r[2] for r in rows]), dtype=torch.long),
            torch.tensor(np.stack([r[3] for r in rows]), dtype=torch.long),
            torch.tensor([r[4] for r in rows], dtype=torch.long),
            torch.tensor([r[5] for r in rows], dtype=torch.long),
            torch.tensor([r[6] for r in rows], dtype=torch.long),
            [r[7] for r in rows], [r[8] for r in rows], [r[9] for r in rows],
            torch.tensor([r[10] for r in rows], dtype=torch.long),
            torch.tensor([r[11] for r in rows], dtype=torch.long),
        )


def sample_safe_delay_overrides(env, batch_size, mode="normal"):
    max_a0 = env.cueA_start_range[1]
    max_total = env.track_len - 1 - max_a0 - 2 * env.cue_width
    d1s, d2s = [], []
    for _ in range(batch_size):
        if mode == "short":
            lo1 = max(2, env.delay1_range[0] - 4)
            hi1 = max(lo1 + 1, min(env.delay1_range[0] + 2, max_total - 2) + 1)
            d1 = np.random.randint(lo1, hi1)
            rem = max_total - d1
            lo2 = max(2, env.delay2_range[0] - 5)
            hi2 = min(env.delay2_range[0] + 2, rem) + 1
            d2 = max(2, rem) if hi2 <= lo2 else np.random.randint(lo2, hi2)
        elif mode == "long":
            lo1 = max(2, min(env.delay1_range[1], max_total - 2))
            hi1 = min(env.delay1_range[1] + 8, max_total - 2) + 1
            d1 = lo1 if hi1 <= lo1 else np.random.randint(lo1, hi1)
            rem = max_total - d1
            lo2 = max(2, min(env.delay2_range[1], rem))
            hi2 = min(env.delay2_range[1] + 10, rem) + 1
            d2 = lo2 if hi2 <= lo2 else np.random.randint(lo2, hi2)
        else:
            lo1 = env.delay1_range[0]
            hi1 = min(env.delay1_range[1], max_total - 2) + 1
            d1 = np.random.randint(lo1, hi1)
            rem = max_total - d1
            lo2 = env.delay2_range[0]
            hi2 = min(env.delay2_range[1], rem) + 1
            d2 = max(2, rem) if hi2 <= lo2 else np.random.randint(lo2, hi2)
        d1s.append(int(d1)); d2s.append(int(d2))
    return np.asarray(d1s), np.asarray(d2s)


def build_balanced_batch(env, batch_size=64, delay_mode="normal"):
    if batch_size % 4:
        raise ValueError("batch_size must be divisible by four")
    per = batch_size // 4
    ca, cb = [], []
    for a in (0, 1):
        for b in (0, 1):
            ca += [a] * per; cb += [b] * per
    order = np.random.permutation(batch_size)
    ca, cb = np.asarray(ca)[order], np.asarray(cb)[order]
    d1, d2 = sample_safe_delay_overrides(env, batch_size, delay_mode)
    return env.sample_batch(batch_size, ca, cb, d1, d2)


def clone_env_for_split(env, seed):
    out = SharedPathFourContextEnv(
        env.track_len, env.obs_dim, env.cue_width, env.cueA_start_range,
        env.delay1_range, env.delay2_range, env.n_cues, env.alias_prob,
        env.noise_std, seed,
    )
    out.pos_map = env.pos_map.copy(); out.cueA_map = env.cueA_map.copy(); out.cueB_map = env.cueB_map.copy()
    out.rng = np.random.RandomState(seed)
    return out


def build_fixed_bank(env, n_batches, batch_size, seed, mode="normal"):
    split = clone_env_for_split(env, seed)
    state = np.random.get_state()
    try:
        np.random.seed(seed + 7919)
        return [build_balanced_batch(split, batch_size, mode) for _ in range(n_batches)]
    finally:
        np.random.set_state(state)


def attractor(M, h, steps=5):
    for _ in range(steps):
        h = torch.tanh(0.5 * h + torch.bmm(M, h.unsqueeze(2)).squeeze(2))
    return h


class WindowAttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__(); self.score = nn.Linear(hidden_dim, 1)
    def forward(self, feats):
        w = torch.softmax(self.score(feats).squeeze(-1), dim=0)
        return (w.unsqueeze(-1) * feats).sum(dim=0), w


class BranchReadout(nn.Module):
    def __init__(self, hidden_dim, n_classes=4):
        super().__init__()
        self.cueA_pool = WindowAttentionPool(hidden_dim)
        self.cueB_pool = WindowAttentionPool(hidden_dim)
        def branch():
            return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU())
        self.cueA_branch = branch(); self.cueB_branch = branch(); self.decision_branch = branch()
        self.fusion = nn.Sequential(nn.Linear((hidden_dim // 2) * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_classes))
    def forward(self, mem, cueA_zones, cueB_zones, decision_positions, return_details=False):
        ra, rb, rd, wa, wb = [], [], [], [], []
        B, tm1, _ = mem.shape
        for i in range(B):
            a0, a1 = cueA_zones[i]; b0, b1 = cueB_zones[i]
            d = max(0, min(int(decision_positions[i]) - 1, tm1 - 1))
            xa, xa_w = self.cueA_pool(mem[i, a0:a1]); xb, xb_w = self.cueB_pool(mem[i, b0:b1])
            ra.append(xa); rb.append(xb); rd.append(mem[i, d]); wa.append(xa_w); wb.append(xb_w)
        ra, rb, rd = torch.stack(ra), torch.stack(rb), torch.stack(rd)
        logits = self.fusion(torch.cat([self.cueA_branch(ra), self.cueB_branch(rb), self.decision_branch(rd)], dim=1))
        if return_details:
            return {"logits": logits, "cueA_attn": wa, "cueB_attn": wb, "cueA_repr": ra, "cueB_repr": rb, "decision_repr": rd}
        return logits


class SimpleReadout(nn.Module):
    def __init__(self, hidden_dim, n_classes=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim * 2), nn.ReLU(), nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_classes))
    def forward(self, mem, cueA_zones, cueB_zones, decision_positions):
        reps = []
        for i in range(mem.shape[0]):
            a0, a1 = cueA_zones[i]; b0, b1 = cueB_zones[i]
            d = max(0, min(int(decision_positions[i]) - 1, mem.shape[1] - 1))
            reps.append(torch.cat([mem[i, a0:a1].mean(0), mem[i, b0:b1].mean(0), mem[i, d]], 0))
        return self.net(torch.stack(reps))


class TEMLite(nn.Module):
    def __init__(self, obs_dim, hidden_dim, track_len, action_dim=1, action_emb_dim=32, n_classes=4, cfg=None):
        super().__init__(); self.obs_dim=obs_dim; self.hidden_dim=hidden_dim; self.track_len=track_len; self.cfg=cfg or ExperimentConfig()
        self.fc_x=nn.Linear(obs_dim,hidden_dim); self.action_emb=nn.Embedding(action_dim,action_emb_dim); self.fc_a=nn.Linear(action_emb_dim,hidden_dim); self.fc_g=nn.Linear(hidden_dim,hidden_dim)
        self.decoder=nn.Linear(hidden_dim,obs_dim); self.readout=BranchReadout(hidden_dim,n_classes) if self.cfg.use_branch_readout else SimpleReadout(hidden_dim,n_classes)
        self.pos_head=nn.Linear(hidden_dim,track_len); self.phase_head=nn.Linear(hidden_dim,6); self.lambda_m=0.92; self.alpha=0.32
    def forward(self,x,a,cueA_zones=None,cueB_zones=None,decision_positions=None,return_details=False,skip_readout=False):
        B,T,_=x.shape; g=torch.zeros(B,self.hidden_dim,device=x.device); M=torch.zeros(B,self.hidden_dim,self.hidden_dim,device=x.device)
        mem=[]; gh=[]
        for t in range(T-1):
            xe=torch.tanh(self.fc_x(x[:,t])); ap=self.fc_a(self.action_emb(a[:,t])); gp=torch.tanh(self.fc_g(g)+ap)
            pt=attractor(M,xe,5); g=gp+self.alpha*pt; po=torch.tanh(0.5*g+torch.bmm(M,g.unsqueeze(2)).squeeze(2)); M=self.lambda_m*M+torch.bmm(pt.unsqueeze(2),pt.unsqueeze(1)); mem.append(po); gh.append(g)
        mem=torch.stack(mem,1); logits=None if skip_readout else self.readout(mem,cueA_zones,cueB_zones,decision_positions)
        return {"logits":logits,"mem":mem,"g_hist":torch.stack(gh,1)} if return_details else (torch.tensor(0.,device=x.device),logits)


class HDGM(nn.Module):
    def __init__(self,obs_dim,hidden_dim,track_len,action_dim=1,action_emb_dim=32,n_classes=4,fb_dim=None,c1=0.2,c2=1.0,beta=0.1,cfg=None):
        super().__init__(); self.obs_dim=obs_dim; self.hidden_dim=hidden_dim; self.track_len=track_len; self.fb_dim=hidden_dim if fb_dim is None else fb_dim; self.cfg=cfg or ExperimentConfig()
        self.fc_x=nn.Linear(obs_dim,hidden_dim); self.action_emb=nn.Embedding(action_dim,action_emb_dim); self.fc_a=nn.Linear(action_emb_dim,hidden_dim); self.fc_g=nn.Linear(hidden_dim,hidden_dim); self.decoder=nn.Linear(hidden_dim,obs_dim)
        self.gate_ec=nn.Linear(obs_dim+hidden_dim+self.fb_dim,obs_dim); self.gate_ca3=nn.Linear(hidden_dim*2,hidden_dim); self.gate_ca1=nn.Linear(hidden_dim*2+self.fb_dim,hidden_dim); self.fb_proj=nn.Linear(hidden_dim,self.fb_dim)
        self.readout=BranchReadout(hidden_dim,n_classes) if self.cfg.use_branch_readout else SimpleReadout(hidden_dim,n_classes)
        self.pos_head=nn.Linear(hidden_dim,track_len); self.phase_head=nn.Linear(hidden_dim,6)
        self.cueA_aux_head=None; self.cueB_aux_head=None; self.lambda_m=.92; self.alpha=.32; self.rho=.84; self.c1=c1; self.c2=c2; self.beta=beta
    def forward(self,x,a,cueA_zones=None,cueB_zones=None,decision_positions=None,return_details=False,skip_readout=False):
        B,T,_=x.shape; g=torch.zeros(B,self.hidden_dim,device=x.device); M=torch.zeros(B,self.hidden_dim,self.hidden_dim,device=x.device); v=torch.zeros(B,self.fb_dim,device=x.device)
        mem=[]; gh=[]; eh=[]; c3h=[]; c1h=[]; wh=[]; vh=[]
        for t in range(T-1):
            ge=torch.sigmoid(self.gate_ec(torch.cat([x[:,t],g,v],1))); xm=ge*x[:,t]; xe=torch.tanh(self.fc_x(xm)); ap=self.fc_a(self.action_emb(a[:,t])); gp=torch.tanh(self.fc_g(g)+ap)
            h=xe; steps=[]
            for _ in range(5):
                gc=torch.sigmoid(self.gate_ca3(torch.cat([h,g],1))); recur=torch.tanh(.5*h+torch.bmm(M,h.unsqueeze(2)).squeeze(2)); h=(1-gc)*h+gc*recur; steps.append(gc)
            pt=h; g=gp+self.alpha*pt; ph=torch.tanh(.5*g+torch.bmm(M,g.unsqueeze(2)).squeeze(2)); ga=torch.sigmoid(self.gate_ca1(torch.cat([g,ph,v],1))); po=torch.relu(ph*(self.c1+self.c2*ga)-self.beta)
            v=self.rho*v+self.fb_proj(po); ws=ge.mean(1,keepdim=True); M=self.lambda_m*M+ws.unsqueeze(-1)*torch.bmm(pt.unsqueeze(2),pt.unsqueeze(1))
            mem.append(po); gh.append(g); eh.append(ge); c3h.append(torch.stack(steps,1).mean(1)); c1h.append(ga); wh.append(ws.squeeze(1)); vh.append(v)
        mem=torch.stack(mem,1); logits=None if skip_readout else self.readout(mem,cueA_zones,cueB_zones,decision_positions)
        if return_details:
            return {"logits":logits,"mem":mem,"g_hist":torch.stack(gh,1),"ec_hist":torch.stack(eh,1),"ca3_hist":torch.stack(c3h,1),"ca1_hist":torch.stack(c1h,1),"write_hist":torch.stack(wh,1),"v_hist":torch.stack(vh,1)}
        return torch.tensor(0.,device=x.device),logits


def instantiate_models(cfg):
    tem=TEMLite(cfg.obs_dim,96,cfg.track_len,1,n_classes=4,cfg=cfg).to(DEVICE)
    hdgm=HDGM(cfg.obs_dim,96,cfg.track_len,1,n_classes=4,cfg=cfg).to(DEVICE)
    return tem,hdgm


def reset_mask_at(t, cueA_zones, device):
    return torch.tensor([int(t)==int(z[1]) for z in cueA_zones],dtype=torch.bool,device=device)


def forward_complete_history_reset(model, x, a, cueA_zones, cueB_zones, decision_positions):
    """Reset dynamic state at Cue-A offset and remove pre-reset readout access.

    The second operation is required because the trained branch readout directly
    pools stored Cue-A states. Resetting recurrent state alone would leave that
    explicit history path intact.
    """
    B,T,_=x.shape
    g=torch.zeros(B,model.hidden_dim,device=x.device); M=torch.zeros(B,model.hidden_dim,model.hidden_dim,device=x.device)
    is_hdgm=isinstance(model,HDGM)
    v=torch.zeros(B,model.fb_dim,device=x.device) if is_hdgm else None
    mem=[]
    for t in range(T-1):
        rm=reset_mask_at(t,cueA_zones,x.device)
        if rm.any():
            keep=(~rm).to(x.dtype); g=g*keep[:,None]; M=M*keep[:,None,None]
            if is_hdgm: v=v*keep[:,None]
        if not is_hdgm:
            xe=torch.tanh(model.fc_x(x[:,t])); ap=model.fc_a(model.action_emb(a[:,t])); gp=torch.tanh(model.fc_g(g)+ap); pt=attractor(M,xe,5); g=gp+model.alpha*pt; po=torch.tanh(.5*g+torch.bmm(M,g.unsqueeze(2)).squeeze(2)); M=model.lambda_m*M+torch.bmm(pt.unsqueeze(2),pt.unsqueeze(1))
        else:
            ge=torch.sigmoid(model.gate_ec(torch.cat([x[:,t],g,v],1))); xe=torch.tanh(model.fc_x(ge*x[:,t])); ap=model.fc_a(model.action_emb(a[:,t])); gp=torch.tanh(model.fc_g(g)+ap); h=xe
            for _ in range(5):
                gc=torch.sigmoid(model.gate_ca3(torch.cat([h,g],1))); recur=torch.tanh(.5*h+torch.bmm(M,h.unsqueeze(2)).squeeze(2)); h=(1-gc)*h+gc*recur
            pt=h; g=gp+model.alpha*pt; ph=torch.tanh(.5*g+torch.bmm(M,g.unsqueeze(2)).squeeze(2)); ga=torch.sigmoid(model.gate_ca1(torch.cat([g,ph,v],1))); po=torch.relu(ph*(model.c1+model.c2*ga)-model.beta); v=model.rho*v+model.fb_proj(po); ws=ge.mean(1,keepdim=True); M=model.lambda_m*M+ws.unsqueeze(-1)*torch.bmm(pt.unsqueeze(2),pt.unsqueeze(1))
        mem.append(po)
    mem=torch.stack(mem,1).clone()
    for i,z in enumerate(cueA_zones):
        mem[i,:min(int(z[1]),mem.shape[1])]=0.0
    return model.readout(mem,cueA_zones,cueB_zones,decision_positions)


def transform_batch(batch, condition):
    x,a,pos,phase,y,ca,cb,za,zb,dp,d1,d2=batch
    if condition in ("full_history","complete_history_reset"):
        return batch
    x2=torch.zeros_like(x)
    if condition=="decision_observation_only":
        for i,d in enumerate(dp):
            idx=max(0,min(int(d)-1,x.shape[1]-1)); x2[i,idx]=x[i,idx]
    elif condition!="timing_only":
        raise ValueError(condition)
    return x2,a.clone(),pos.clone(),phase.clone(),y.clone(),ca.clone(),cb.clone(),list(za),list(zb),list(dp),d1.clone(),d2.clone()


def insert_matched_holds(batch, extra):
    if extra==0: return batch
    x,a,pos,phase,y,ca,cb,za,zb,dp,d1,d2=batch
    xs=[]; acts=[]; poss=[]; phases=[]; za2=[]; zb2=[]; dp2=[]
    for i in range(x.shape[0]):
        ae=int(za[i][1]); bs,be=map(int,zb[i]); dec=int(dp[i])
        h1=x[i,ae:ae+1].repeat(extra,1); h2=x[i,be:be+1].repeat(extra,1)
        xs.append(torch.cat([x[i,:ae],h1,x[i,ae:be],h2,x[i,be:]],0))
        acts.append(torch.cat([a[i,:ae],a[i,ae:ae+1].repeat(extra),a[i,ae:be],a[i,be:be+1].repeat(extra),a[i,be:]],0))
        poss.append(torch.cat([pos[i,:ae],pos[i,ae:ae+1].repeat(extra),pos[i,ae:be],pos[i,be:be+1].repeat(extra),pos[i,be:]],0))
        phases.append(torch.cat([phase[i,:ae],torch.full((extra,),2,dtype=phase.dtype),phase[i,ae:be],torch.full((extra,),4,dtype=phase.dtype),phase[i,be:]],0))
        za2.append(tuple(za[i])); zb2.append((bs+extra,be+extra)); dp2.append(dec+2*extra)
    return torch.stack(xs),torch.stack(acts),torch.stack(poss),torch.stack(phases),y.clone(),ca.clone(),cb.clone(),za2,zb2,dp2,d1+extra,d2+extra


def confusion(y,p,n=4):
    cm=np.zeros((n,n),dtype=np.int64)
    for a,b in zip(y,p): cm[int(a),int(b)]+=1
    return cm


def macro_f1(cm):
    vals=[]
    for i in range(cm.shape[0]):
        tp=cm[i,i]; fp=cm[:,i].sum()-tp; fn=cm[i,:].sum()-tp
        precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else 0.
        vals.append(2*precision*recall/(precision+recall) if precision+recall else 0.)
    return float(np.mean(vals))


def evaluate_frozen(model,bank,condition="full_history",extra_delay=0,collect=False):
    ys=[]; ps=[]; probs=[]
    model.eval()
    with torch.no_grad():
        for raw in bank:
            batch=insert_matched_holds(raw,extra_delay)
            batch=transform_batch(batch,condition)
            x,a,_,_,y,_,_,za,zb,dp,_,_=batch
            x=x.to(DEVICE); a=a.to(DEVICE)
            if condition=="complete_history_reset": logits=forward_complete_history_reset(model,x,a,za,zb,dp)
            else: logits=model(x,a,cueA_zones=za,cueB_zones=zb,decision_positions=dp,return_details=True)["logits"]
            ys.append(y.numpy()); ps.append(logits.argmax(1).cpu().numpy()); probs.append(torch.softmax(logits,1).cpu().numpy())
    y=np.concatenate(ys); p=np.concatenate(ps); pr=np.concatenate(probs); cm=confusion(y,p)
    out={"accuracy":float((y==p).mean()),"macro_f1":macro_f1(cm),"confusion_matrix":cm.tolist(),"n_trials":int(len(y))}
    if collect: out.update({"y_true":y,"y_pred":p,"probabilities":pr})
    return out


class LinearNull(nn.Module):
    def __init__(self,d): super().__init__(); self.fc=nn.Linear(d,4)
    def forward(self,x): return self.fc(x)


def shortcut_features(bank,kind):
    feats=[]; labels=[]
    for batch in bank:
        x,_,_,_,y,_,_,za,zb,dp,d1,d2=batch
        if kind=="observation_only":
            f=torch.stack([x[i,int(dp[i])] for i in range(x.shape[0])])
        elif kind=="timing_only":
            f=torch.tensor([[za[i][0],za[i][1],zb[i][0],zb[i][1],dp[i],int(d1[i]),int(d2[i])] for i in range(x.shape[0])],dtype=torch.float32)
        else: raise ValueError(kind)
        feats.append(f); labels.append(y)
    return torch.cat(feats),torch.cat(labels)


def fit_shortcut_null(train_bank,val_bank,test_bank,kind,seed,epochs=300):
    tr_x,tr_y=shortcut_features(train_bank,kind); va_x,va_y=shortcut_features(val_bank,kind); te_x,te_y=shortcut_features(test_bank,kind)
    mean=tr_x.mean(0,keepdim=True); std=tr_x.std(0,unbiased=False,keepdim=True).clamp_min(1e-5)
    tr_x=((tr_x-mean)/std).to(DEVICE); va_x=((va_x-mean)/std).to(DEVICE); te_x=((te_x-mean)/std).to(DEVICE); tr_y=tr_y.to(DEVICE); va_y=va_y.to(DEVICE)
    set_global_seed(seed); net=LinearNull(tr_x.shape[1]).to(DEVICE); opt=torch.optim.Adam(net.parameters(),lr=.01,weight_decay=1e-4); loss_fn=nn.CrossEntropyLoss(); best=None; best_acc=-1.
    for _ in range(epochs):
        net.train(); opt.zero_grad(); loss=loss_fn(net(tr_x),tr_y); loss.backward(); opt.step(); net.eval()
        with torch.no_grad(): acc=float((net(va_x).argmax(1)==va_y).float().mean())
        if acc>best_acc: best_acc=acc; best=copy.deepcopy(net.state_dict())
    net.load_state_dict(best); net.eval()
    with torch.no_grad(): logits=net(te_x); pred=logits.argmax(1).cpu().numpy()
    y=te_y.numpy(); cm=confusion(y,pred)
    return {"accuracy":float((y==pred).mean()),"macro_f1":macro_f1(cm),"confusion_matrix":cm.tolist(),"n_trials":int(len(y)),"best_val_accuracy":best_acc}


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def discover_checkpoints(root,seeds=None):
    found={}; conflicts=[]
    for path in sorted(Path(root).rglob("*.pt")):
        try:
            obj=torch.load(path,map_location="cpu",weights_only=False)
        except Exception as e:
            print(f"[skip] {path}: {type(e).__name__}: {e}"); continue
        if not isinstance(obj,dict) or "snapshots" not in obj or "seed" not in obj: continue
        if str(obj.get("mode"))!="primary_150" or int(obj.get("max_updates",-1))!=150: continue
        seed=int(obj["seed"])
        if seeds is not None and seed not in seeds: continue
        digest=sha256(path)
        if seed in found:
            if found[seed][2]!=digest: conflicts.append((seed,str(found[seed][0]),str(path)))
            continue
        found[seed]=(path,obj,digest)
    if conflicts:
        raise RuntimeError("Conflicting non-identical checkpoints for the same seed: "+repr(conflicts))
    return found


def restore_env(obj,cfg,seed):
    env=SharedPathFourContextEnv(cfg.track_len,cfg.obs_dim,cfg.cue_width,cfg.cueA_start_range,cfg.delay1_range,cfg.delay2_range,cfg.n_cues,cfg.alias_prob,cfg.noise_std,seed)
    maps=obj["environment_maps"]; env.pos_map=np.asarray(maps["pos_map"],np.float32); env.cueA_map=np.asarray(maps["cueA_map"],np.float32); env.cueB_map=np.asarray(maps["cueB_map"],np.float32)
    return env


def write_csv(path,rows):
    if not rows: return
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def mean_ci(values):
    x=np.asarray(values,float); return float(x.mean()),float(1.96*x.std(ddof=1)/math.sqrt(len(x))) if len(x)>1 else 0.


def plot_outputs(outdir,control_rows,delay_rows):
    colors={"TEM":"#475569","HDGM":"#2E5BFF"}
    fig,ax=plt.subplots(figsize=(8.2,4.5))
    conditions=["full_history","complete_history_reset","decision_observation_only","timing_only"]
    labels=["Full history","History reset","Observation only","Timing only"]
    for j,model in enumerate(("TEM","HDGM")):
        for i,c in enumerate(conditions):
            vals=[r["accuracy"] for r in control_rows if r["model"]==model and r["condition"]==c]
            xs=np.full(len(vals),i+(j-.5)*.16); ax.scatter(xs,vals,s=24,color=colors[model],alpha=.58,zorder=2)
            m,h=mean_ci(vals); ax.errorbar(i+(j-.5)*.16,m,yerr=h,fmt="o",color=colors[model],capsize=3,lw=2,label=model if i==0 else None,zorder=3)
    ax.axhline(.25,color="#94A3B8",ls="--",lw=1.2); ax.set_xticks(range(4),labels); ax.set_ylabel("Held-out accuracy"); ax.set_ylim(.15,1.0); ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False); fig.tight_layout(); fig.savefig(outdir/"figure2_history_controls.png",dpi=300); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.2,4.5))
    for model in ("TEM","HDGM"):
        seeds=sorted({r["seed"] for r in delay_rows if r["model"]==model})
        extras=sorted({r["inserted_steps_per_delay"] for r in delay_rows if r["model"]==model})
        for s in seeds:
            ys=[next(r["accuracy"] for r in delay_rows if r["model"]==model and r["seed"]==s and r["inserted_steps_per_delay"]==e) for e in extras]
            ax.plot(extras,ys,color=colors[model],alpha=.14,lw=.8)
        means=[]; cis=[]
        for e in extras:
            vals=[r["accuracy"] for r in delay_rows if r["model"]==model and r["inserted_steps_per_delay"]==e]; m,h=mean_ci(vals); means.append(m); cis.append(h)
        means=np.asarray(means); cis=np.asarray(cis); ax.plot(extras,means,"o-",color=colors[model],lw=2.4,label=model); ax.fill_between(extras,means-cis,means+cis,color=colors[model],alpha=.16)
    ax.axhline(.25,color="#94A3B8",ls="--",lw=1.2); ax.set_xlabel("Inserted matched-observation hold steps per delay"); ax.set_ylabel("Held-out accuracy"); ax.set_ylim(.15,1.0); ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False); fig.tight_layout(); fig.savefig(outdir/"figure2_delay_transfer.png",dpi=300); plt.close(fig)


def parse_args():
    p=argparse.ArgumentParser(description="Run missing frozen Figure-2 evaluations from V4 primary-150 checkpoints.")
    p.add_argument("--search-root",default=".",help="Root searched recursively for .pt checkpoints; no folder merging required.")
    p.add_argument("--output-dir",default="figure2_missing_outputs")
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    p.add_argument("--seeds",default="11,22,33,44,55,66,77,88,99,111")
    p.add_argument("--checkpoint-step",type=int,default=150)
    p.add_argument("--test-batches",type=int,default=50)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--delay-batches",type=int,default=20)
    p.add_argument("--delay-steps",default="0,4,8,16,32,64")
    p.add_argument("--null-train-batches",type=int,default=20)
    p.add_argument("--null-val-batches",type=int,default=5)
    p.add_argument("--null-test-batches",type=int,default=20)
    p.add_argument("--null-epochs",type=int,default=300)
    p.add_argument("--quick",action="store_true",help="Smoke test: 2 seeds, 2 test batches and delay steps 0,8.")
    return p.parse_args()


def main():
    global DEVICE
    args=parse_args()
    if args.device=="cuda" and not torch.cuda.is_available(): raise RuntimeError("--device cuda requested but CUDA is unavailable")
    DEVICE=torch.device("cuda" if (args.device=="cuda" or args.device=="auto" and torch.cuda.is_available()) else "cpu")
    seeds=[int(x) for x in args.seeds.split(",") if x.strip()]; delay_steps=[int(x) for x in args.delay_steps.split(",") if x.strip()]
    if args.quick:
        seeds=seeds[:2]; args.test_batches=2; args.delay_batches=2; args.null_train_batches=2; args.null_val_batches=1; args.null_test_batches=2; args.null_epochs=20; delay_steps=[0,8]
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    checkpoints=discover_checkpoints(args.search_root,set(seeds))
    missing=sorted(set(seeds)-set(checkpoints))
    if missing: raise FileNotFoundError(f"Missing primary_150 checkpoints for seeds {missing}. Found {sorted(checkpoints)} under {args.search_root}")
    print(f"Using {len(checkpoints)} checkpoints on {DEVICE}: {sorted(checkpoints)}")
    controls=[]; delays=[]; shortcuts=[]; pred_seed=[]; pred_model=[]; pred_condition=[]; pred_y=[]; pred_p=[]; pred_prob=[]; manifests=[]
    conditions=("full_history","complete_history_reset","decision_observation_only","timing_only")
    for seed in seeds:
        path,obj,digest=checkpoints[seed]; cfg=config_from_dict(obj["config"]); env=restore_env(obj,cfg,seed); tem,hdgm=instantiate_models(cfg)
        snap=obj["snapshots"]; key=args.checkpoint_step if args.checkpoint_step in snap else str(args.checkpoint_step)
        if key not in snap: raise KeyError(f"Checkpoint {args.checkpoint_step} absent for seed {seed}; available={list(snap)}")
        tem.load_state_dict(snap[key]["tem"],strict=True); hdgm.load_state_dict(snap[key]["hdgm"],strict=True)
        split=obj.get("split_seeds",{}); test_seed=int(split.get("test_normal",seed+420001)); bank=build_fixed_bank(env,args.test_batches,args.batch_size,test_seed,"normal")
        manifests.append({"seed":seed,"path":str(path.resolve()),"sha256":digest,"test_seed":test_seed,"checkpoint_step":args.checkpoint_step})
        print(f"\nSeed {seed}: frozen controls")
        full_by_model={}
        for model_name,model in (("TEM",tem),("HDGM",hdgm)):
            for cond in conditions:
                result=evaluate_frozen(model,bank,cond,0,collect=True); full_by_model.setdefault(model_name,result["accuracy"] if cond=="full_history" else None)
                loss=100*(full_by_model[model_name]-result["accuracy"])
                controls.append({"seed":seed,"model":model_name,"condition":cond,"accuracy":result["accuracy"],"macro_f1":result["macro_f1"],"loss_vs_full_pp":loss,"n_trials":result["n_trials"]})
                pred_seed.extend([seed]*result["n_trials"]); pred_model.extend([model_name]*result["n_trials"]); pred_condition.extend([cond]*result["n_trials"]); pred_y.extend(result["y_true"]); pred_p.extend(result["y_pred"]); pred_prob.extend(result["probabilities"])
                print(f"  {model_name:4s} {cond:27s} acc={result['accuracy']:.4f} macroF1={result['macro_f1']:.4f}")
        print(f"Seed {seed}: frozen delay transfer")
        delay_bank=bank[:args.delay_batches]
        for model_name,model in (("TEM",tem),("HDGM",hdgm)):
            for extra in delay_steps:
                result=evaluate_frozen(model,delay_bank,"full_history",extra,False)
                delays.append({"seed":seed,"model":model_name,"inserted_steps_per_delay":extra,"total_inserted_steps":2*extra,"accuracy":result["accuracy"],"macro_f1":result["macro_f1"],"n_trials":result["n_trials"],"frozen":True,"adaptation":False,"delay_type":"matched_observation_hold"})
                print(f"  {model_name:4s} +{extra:3d}/delay acc={result['accuracy']:.4f}")
        print(f"Seed {seed}: shortcut-only linear diagnostics")
        base=int(seed+cfg.split_seed_version*100000+70000)
        tr=build_fixed_bank(env,args.null_train_batches,args.batch_size,base+1,"normal"); va=build_fixed_bank(env,args.null_val_batches,args.batch_size,base+2,"normal"); te=build_fixed_bank(env,args.null_test_batches,args.batch_size,base+3,"normal")
        for i,kind in enumerate(("observation_only","timing_only")):
            result=fit_shortcut_null(tr,va,te,kind,base+100+i,args.null_epochs)
            shortcuts.append({"seed":seed,"diagnostic":kind,"accuracy":result["accuracy"],"macro_f1":result["macro_f1"],"best_val_accuracy":result["best_val_accuracy"],"n_trials":result["n_trials"]})
            print(f"  linear {kind:18s} acc={result['accuracy']:.4f}")
    write_csv(outdir/"figure2_history_controls.csv",controls); write_csv(outdir/"figure2_delay_transfer.csv",delays); write_csv(outdir/"figure2_shortcut_diagnostics.csv",shortcuts)
    np.savez_compressed(outdir/"figure2_endpoint_predictions.npz",seed=np.asarray(pred_seed,np.int32),model=np.asarray(pred_model),condition=np.asarray(pred_condition),y_true=np.asarray(pred_y,np.int8),y_pred=np.asarray(pred_p,np.int8),probabilities=np.asarray(pred_prob,np.float32))
    summary={"protocol":{"checkpoint_step":args.checkpoint_step,"device":str(DEVICE),"seeds":seeds,"chance":0.25,"frozen_model_weights":True,"no_retraining_or_adaptation":True,"history_reset":"At Cue-A offset: recurrent state and associative memory are reset; all pre-reset states are removed from event-aligned readout access.","observation_only_frozen":"All inputs are zero except the decision-pre observation used by the trained model.","timing_only_frozen":"All sensory inputs are zero; cue-window and decision indices remain available to the trained event-aligned readout.","shortcut_nulls":"Independent linear classifiers trained on decision observation alone or timing metadata alone.","delay_transfer":"The first matched delay observation is repeated equally in Delay 1 and Delay 2; weights remain frozen.","warning":"The original branch readout is event-aligned and directly pools Cue-A/Cue-B windows. Results must be called event-aligned decision accuracy, not autonomous online behavior."},"checkpoints":manifests,"files":{"history_controls":"figure2_history_controls.csv","delay_transfer":"figure2_delay_transfer.csv","shortcut_diagnostics":"figure2_shortcut_diagnostics.csv","predictions":"figure2_endpoint_predictions.npz"}}
    with open(outdir/"figure2_missing_summary.json","w",encoding="utf-8") as f: json.dump(summary,f,ensure_ascii=False,indent=2)
    plot_outputs(outdir,controls,delays)
    print(f"\nDone. Outputs saved to: {outdir.resolve()}")


if __name__=="__main__":
    main()
