#!/usr/bin/env python3
# bxd.py — 1D TinyNODE with ESRK-15, RK4, Euler; optional Fractional (SOE) memory
# Synthetic sequence benchmarks (k-th last). Weakref fix to avoid recursion.
# Adds: causal convolutions, sinusoidal positional encoding, CLI flags.

import argparse, time, math, random, weakref
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =====================================================================================
# Utils
# =====================================================================================

def set_all_seeds(seed: int = 244):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(1) == y).float().mean().item() * 100.0

def add_pos_enc(x: torch.Tensor) -> torch.Tensor:
    """
    Add fixed sinusoidal positional encoding into the input channels.
    x: (B, C, T)  -- assumes C >= 2
    """
    B, C, T = x.shape
    half = (C // 2) * 2
    if half == 0:
        return x
    pos = torch.arange(T, device=x.device, dtype=torch.float32)  # (T,)
    div = torch.exp(torch.arange(0, half, 2, device=x.device, dtype=torch.float32)
                    * (-math.log(10000.0) / max(1, half)))
    sin = torch.sin(pos[None, :] * div[:, None])  # (half/2, T)
    cos = torch.cos(pos[None, :] * div[:, None])  # (half/2, T)
    pe = torch.zeros(C, T, device=x.device, dtype=x.dtype)
    pe[0:half:2, :] = sin
    pe[1:half:2, :] = cos
    return x + pe.unsqueeze(0)

# =====================================================================================
# Synthetic datasets
# =====================================================================================

class KthLastDataset(Dataset):
    """
    Tokens in {0..(vocab-1)}. Length L. Label is token at position (L-k) (0-indexed).
    If k==1 => last symbol; k==L => first symbol.
    """
    def __init__(self, n_samples: int, L: int, k: int, vocab: int = 10):
        assert 1 <= k <= L, "k must be within [1, L]"
        self.N, self.L, self.k, self.V = int(n_samples), int(L), int(k), int(vocab)

    def __len__(self): return self.N

    def __getitem__(self, idx):
        seq = torch.randint(0, self.V, (self.L,), dtype=torch.long)
        target = seq[-self.k].item()
        # one-hot (C=V, T=L)
        x = F.one_hot(seq, num_classes=self.V).float().transpose(0,1)  # (L,V) -> (V,L)
        return x, torch.tensor(target, dtype=torch.long)

# =====================================================================================
# ESRK-15 tableau (2S-8 structure) — 1D use is identical
# =====================================================================================

_a = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0243586417803786, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0258303808904268, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0667956303329210, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0140960387721938, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0412105997557866, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0149469583607297, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.414086419082813, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00395908281378477, 0, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.480561088337756, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.319660987317690, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0.00668808071535874, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0, 0.0374638233561973, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0, 0.422645975498266, 0.439499983548480, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0, 0.422645975498266, 0.0327614907498598, 0.367805790222090, 0]
], dtype=np.float32)
_b = np.array([
    0.0358989325, 0.0358989325, 0.0358989325,
    0.0358989325, 0.0358989325, 0.0358989325,
    0.0358989325, 0.0358989325, 0.0066124579,
    0.2167468695, 0.0, 0.4226459755, 0.0327614907,
    0.0330623264, 0.0009799086
], dtype=np.float32)
A = torch.tensor(_a)
Bv = torch.tensor(_b)
Cv = A.tril(-1).sum(1)

# =====================================================================================
# Vector field (1D feature-time map)
# =====================================================================================

class ApproxSiLU(nn.Module):
    def forward(self, x):
        x = torch.clamp(x, -4, 4)
        return x * (0.5 + 0.25*x - (1/12)*x**2 + (1/48)*x**3)

def make_f_1d(ch: int, approx_act: bool = False, use_groupnorm: bool = False, causal: bool = True) -> nn.Module:
    """
    Causal 1D residual dynamics: two Conv1d layers with left padding.
    """
    act = ApproxSiLU() if approx_act else nn.SiLU()
    gn  = nn.GroupNorm(8, ch) if use_groupnorm else nn.Identity()
    k = 3
    left_pad = (k - 1) if causal else 0

    class VF(nn.Module):
        def __init__(self):
            super().__init__()
            self.causal = causal
            self.conv1 = nn.Conv1d(ch, ch, kernel_size=k, padding=0 if causal else 1, bias=False)
            self.conv2 = nn.Conv1d(ch, ch, kernel_size=k, padding=0 if causal else 1, bias=False)
            self.gn, self.act = gn, act

        def forward(self, t: float, x: torch.Tensor) -> torch.Tensor:  # x: (B,C,T)
            if self.causal:
                x = F.pad(x, (left_pad, 0))
                x = self.conv1(x)
                x = self.gn(x)
                x = self.act(x)
                x = F.pad(x, (left_pad, 0))
                x = self.conv2(x)
                return x
            else:
                x = self.conv1(x); x = self.gn(x); x = self.act(x); x = self.conv2(x)
                return x
    return VF()

class ScaleVF1D(nn.Module):
    """Scale RHS by T (and pass scaled time)."""
    def __init__(self, f_core: nn.Module, T: float):
        super().__init__()
        self.f = f_core; self.T = float(T)
    def forward(self, t, x): return self.T * self.f(t * self.T, x)

# =====================================================================================
# Solvers (1D)
# =====================================================================================

class EulerBlock1D(nn.Module):
    def __init__(self, f, h=1.0, steps=1):
        super().__init__(); self.f, self.h, self.steps = f, float(h), int(steps)
    def forward(self, x, t0: float = 0.0):
        t = t0
        for _ in range(self.steps):
            x = x + self.h * self.f(t, x); t += self.h
        return x

class RK4Block1D(nn.Module):
    def __init__(self, f, h=1.0, steps=1):
        super().__init__(); self.f, self.h, self.steps = f, float(h), int(steps)
    def forward(self, x, t0: float = 0.0):
        t = t0
        for _ in range(self.steps): x = self._step(x, t); t += self.h
        return x
    def _step(self, x, t: float):
        h, f = self.h, self.f
        k1 = f(t, x)
        k2 = f(t + 0.5*h, x + 0.5*h*k1)
        k3 = f(t + 0.5*h, x + 0.5*h*k2)
        k4 = f(t + h,     x + h*k3)
        return x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

class ESRKBlock1D(nn.Module):
    """ESRK-15 in-place 1D, two-register-friendly."""
    def __init__(self, f, A, B, C, h=1.0, steps=1):
        super().__init__()
        self.f, self.h, self.steps = f, float(h), int(steps)
        A_l = torch.as_tensor(A, dtype=torch.float32).tril(-1)
        self.register_buffer("A_l", A_l)
        self.register_buffer("Bv",  torch.as_tensor(B, dtype=torch.float32))
        self.register_buffer("Cv",  torch.as_tensor(C, dtype=torch.float32))
        self.s = A_l.shape[0]
        self._last = None

    def _compute_last(self):
        last = torch.zeros(self.s, dtype=torch.long)
        for j in range(self.s):
            for i in range(self.s-1, -1, -1):
                if self.A_l[i, j] != 0: last[j] = i; break
        self._last = last

    def forward(self, x_in, t0: float = 0.0):
        t = t0
        for _ in range(self.steps):
            x_in = self._step(x_in, t); t += self.h
        return x_in

    def _step(self, x_in, t):
        h, A_l, Bv, Cv, s = self.h, self.A_l, self.Bv, self.Cv, self.s
        if self._last is None: self._compute_last()
        last = self._last
        R1 = R2 = None; idx1 = idx2 = None; tmp = {}; sum_bk = None
        for i in range(s):
            Y = x_in
            for j in range(i):
                k = R1 if j==idx1 else (R2 if j==idx2 else tmp[j])
                if A_l[i, j] != 0:
                    Y = Y + h * A_l[i, j] * k
            k_new = self.f(t + Cv[i]*h, Y)
            if Bv[i] != 0:
                sum_bk = k_new * (h * Bv[i]) if sum_bk is None else sum_bk + h * Bv[i] * k_new
            placed = False
            if idx1 is None or i >= last[idx1]:
                R1, idx1 = k_new, i; placed = True
            elif idx2 is None or i >= last[idx2]:
                R2, idx2 = k_new, i; placed = True
            if not placed: tmp[i] = k_new
        return x_in + sum_bk

# =====================================================================================
# Fractional (SOE) memory wrappers — with WEAKREF FIX
# =====================================================================================

class Frac1D(nn.Module):
    """
    dZ_j = -λ_j Z_j + ω_j f(t,x),  dx = sum_j Z_j
    Solved by ESRKBlock1D on augmented state [x, Z_1..Z_m].
    """
    def __init__(self, f_core, A, Bv, Cv, h=1.0, steps=1, m=16, T=None):
        super().__init__()
        self.f_core = f_core
        self.m = int(m)
        H_eff = float(T) if (T is not None) else float(h)
        H_eff = max(H_eff, 1.0)
        lam_init = torch.exp(torch.linspace(math.log(1.0/H_eff), math.log(3.0), self.m))
        om_init  = torch.ones(self.m) / self.m
        self.log_lam = nn.Parameter(torch.log(lam_init))
        self.log_om  = nn.Parameter(torch.log(om_init))

        class AugVF(nn.Module):
            def __init__(self, outer):
                super().__init__()
                object.__setattr__(self, "o", weakref.proxy(outer))
            def forward(self, t, aug):
                o = self.o
                B, C, T = aug.shape
                m = o.m; d = C // (1 + m)
                x = aug[:, :d, :]
                Z = aug[:, d:, :].view(B, m, d, T)
                fx = o.f_core(t, x)                          # B,d,T
                lam = F.softplus(o.log_lam).view(1, m, 1, 1) # B,m,1,1
                omr = F.softplus(o.log_om)
                om  = (omr / (omr.sum() + 1e-12)).view(1, m, 1, 1)
                dZ = -lam * Z + om * fx.unsqueeze(1)         # B,m,d,T
                dx = Z.sum(1)                                # B,d,T
                return torch.cat([dx, dZ.view(B, m*d, T)], 1)

        self._aug_vf = AugVF(self)
        self._ode = ESRKBlock1D(self._aug_vf, A, Bv, Cv, h=h, steps=steps)

    def forward(self, x):
        B, d, T = x.shape
        Z0 = torch.zeros(B, self.m*d, T, device=x.device, dtype=x.dtype)
        aug0 = torch.cat([x, Z0], dim=1)
        augT = self._ode(aug0)
        return augT[:, :d, :]

class FracRK41D(nn.Module):
    """Same fractional state, integrated by RK4."""
    def __init__(self, f_core, h=1.0, steps=1, m=16, T=None):
        super().__init__()
        self.f_core = f_core
        self.m = int(m)
        H_eff = float(T) if (T is not None) else float(h)
        H_eff = max(H_eff, 1.0)
        lam_init = torch.exp(torch.linspace(math.log(1.0/H_eff), math.log(3.0), self.m))
        om_init  = torch.ones(self.m) / self.m
        self.log_lam = nn.Parameter(torch.log(lam_init))
        self.log_om  = nn.Parameter(torch.log(om_init))

        class AugVF(nn.Module):
            def __init__(self, outer):
                super().__init__()
                object.__setattr__(self, "o", weakref.proxy(outer))
            def forward(self, t, aug):
                o = self.o
                B, C, T = aug.shape
                m = o.m; d = C // (1 + m)
                x = aug[:, :d, :]
                Z = aug[:, d:, :].view(B, m, d, T)
                fx = o.f_core(t, x)
                lam = F.softplus(o.log_lam).view(1, m, 1, 1)
                omr = F.softplus(o.log_om)
                om  = (omr / (omr.sum() + 1e-12)).view(1, m, 1, 1)
                dZ = -lam * Z + om * fx.unsqueeze(1)
                dx = Z.sum(1)
                return torch.cat([dx, dZ.view(B, m*d, T)], 1)

        self._aug_vf = AugVF(self)
        self._ode = RK4Block1D(self._aug_vf, h=h, steps=steps)

    def forward(self, x):
        B, d, T = x.shape
        Z0 = torch.zeros(B, self.m*d, T, device=x.device, dtype=x.dtype)
        aug0 = torch.cat([x, Z0], dim=1)
        augT = self._ode(aug0)
        return augT[:, :d, :]

# =====================================================================================
# Model
# =====================================================================================

class TinyNODE1D(nn.Module):
    """
    Encoder: 1x1 conv to width -> ODE core over (C,T) -> head: take last time step, linear to classes.
    """
    def __init__(self, solver='esrk', width=64, h=1.0, steps=1, T=None,
                 approx_act=False, groupnorm=False, fractional=False, m=16,
                 in_ch=10, n_classes=10, pos_enc=True, causal=True):
        super().__init__()
        self.pos_enc = pos_enc
        self.enc = nn.Conv1d(in_ch, width, kernel_size=1, bias=False)
        f = make_f_1d(width, approx_act=approx_act, use_groupnorm=groupnorm, causal=causal)
        if T is not None:
            f = ScaleVF1D(f, T)

        if solver == 'euler':
            self.ode = EulerBlock1D(f, h=h, steps=steps)
        elif solver == 'rk4':
            self.ode = FracRK41D(f, h=h, steps=steps, m=m, T=T) if fractional else RK4Block1D(f, h=h, steps=steps)
        elif solver == 'esrk':
            self.ode = Frac1D(f, A, Bv, Cv, h=h, steps=steps, m=m, T=T) if fractional else ESRKBlock1D(f, A, Bv, Cv, h=h, steps=steps)
        else:
            raise ValueError("solver must be one of: euler, rk4, esrk")

        self.head = nn.Linear(width, n_classes)

    def forward(self, x):         # x: (B, in_ch, T)
        if self.pos_enc:
            x = add_pos_enc(x)
        z = self.enc(x)           # (B, width, T)
        z = self.ode(z)           # (B, width, T)
        z_last = z[:, :, -1]      # (B, width)
        return self.head(z_last)  # (B, n_classes)

# =====================================================================================
# Training
# =====================================================================================

@dataclass
class TrainCfg:
    epochs: int = 30
    lr: float = 1e-3
    wd: float = 0.0

def train_eval(model, tr_ld, va_ld, epochs=30, lr=1e-3, wd=0.0, device=None, print_every=1):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossF = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(1, epochs+1):
        model.train()
        t0 = time.time()
        for xb, yb in tr_ld:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = lossF(model(xb), yb)
            loss.backward()
            opt.step()
        # eval
        model.eval(); acc_tr = 0.0; n_tr = 0
        with torch.no_grad():
            for xb, yb in tr_ld:
                xb, yb = xb.to(device), yb.to(device)
                acc_tr += (model(xb).argmax(1) == yb).sum().item(); n_tr += yb.size(0)
        acc_tr = 100.0 * acc_tr / max(1, n_tr)

        acc_va = 0.0; n_va = 0
        with torch.no_grad():
            for xb, yb in va_ld:
                xb, yb = xb.to(device), yb.to(device)
                acc_va += (model(xb).argmax(1) == yb).sum().item(); n_va += yb.size(0)
        acc_va = 100.0 * acc_va / max(1, n_va)

        best = max(best, acc_va); sched.step()
        if ep % print_every == 0:
            print(f"Ep{ep:03d}  train={acc_tr:5.2f}%  val={acc_va:5.2f}%  best={best:5.2f}%  time={time.time()-t0:4.1f}s")
    return best

# =====================================================================================
# CLI
# =====================================================================================

def make_loaders(task:str, L:int, k:int, batch:int, ntrain:int, nvalid:int, vocab:int=10):
    if task != 'kthlast':
        raise ValueError("Supported tasks: kthlast")
    tr = KthLastDataset(ntrain, L=L, k=k, vocab=vocab)
    va = KthLastDataset(nvalid, L=L, k=k, vocab=vocab)
    tr_ld = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)
    va_ld = DataLoader(va, batch_size=512,   shuffle=False, num_workers=2, pin_memory=True)
    return tr_ld, va_ld, vocab

def main():
    p = argparse.ArgumentParser()
    # task
    p.add_argument('--task', type=str, default='kthlast', choices=['kthlast'])
    p.add_argument('--L', type=int, default=200, help='sequence length')
    p.add_argument('--k', type=int, default=50, help='kth last to predict (1..L)')
    p.add_argument('--ntrain', type=int, default=50000)
    p.add_argument('--nvalid', type=int, default=10000)
    p.add_argument('--batch', type=int, default=256)
    # model
    p.add_argument('--solver', choices=['euler','rk4','esrk'], default='esrk')
    p.add_argument('--fractional', action='store_true')
    p.add_argument('--m', type=int, default=16, help='SOE terms (memory channels per feature)')
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--h', type=float, default=1.0)
    p.add_argument('--steps', type=int, default=1)
    p.add_argument('--T', type=float, default=None, help='total time rescale (keep h=1)')
    p.add_argument('--approx_act', action='store_true')
    p.add_argument('--groupnorm', action='store_true')
    p.add_argument('--pos_enc', action='store_true')
    p.add_argument('--no_pos_enc', dest='pos_enc', action='store_false')
    p.set_defaults(pos_enc=True)
    p.add_argument('--causal', action='store_true')
    p.add_argument('--no_causal', dest='causal', action='store_false')
    p.set_defaults(causal=True)
    # train
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=244)
    args = p.parse_args()

    set_all_seeds(args.seed)

    tr_ld, va_ld, vocab = make_loaders(args.task, args.L, args.k, args.batch, args.ntrain, args.nvalid, vocab=10)

    model = TinyNODE1D(
        solver=args.solver,
        width=args.width,
        h=args.h,
        steps=args.steps,
        T=args.T,
        approx_act=args.approx_act,
        groupnorm=args.groupnorm,
        fractional=args.fractional,
        m=args.m,
        in_ch=vocab,
        n_classes=vocab,
        pos_enc=args.pos_enc,
        causal=args.causal
    )

    print(f"Params: {count_params(model):,d} | solver={args.solver} | frac={args.fractional} m={args.m} | width={args.width} | L={args.L} k={args.k}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best = train_eval(model, tr_ld, va_ld, epochs=args.epochs, lr=args.lr, wd=args.wd, device=device)
    print(f"Best val: {best:.2f}%")

if __name__ == "__main__":
    main()
