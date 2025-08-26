#!/usr/bin/env python
# tiny_esrk_frac_T_fixed.py
"""
TinyNODE with ESRK-15 (two-register), optional Fractional (SOE) augmentation,
and high-T support (rescaled or direct big-h). Includes:
  • Δ-norm per stage
  • grad-norm per mini-batch
  • h·σ̂ stability product each epoch
  • optional CIFAR-10-C and adversarial eval

Examples:
  # Recommended (rescaled): big total time without huge h·σ̂
  python tiny_esrk_frac_T_fixed.py --solver esrk --width 32 --epochs 200 \
    --T 400 --h 1 --steps 1 --fractional --m 24 --groupnorm --approx_act --diag

  # Proof mode (direct big-h):
  python tiny_esrk_frac_T_fixed.py --solver esrk --width 32 --epochs 200 \
    --h 400 --steps 1 --fractional --m 24 --groupnorm --approx_act --diag

  # Fractional + RK4 (rescaled):
  python tiny_esrk_frac_T_fixed.py --solver rk4 --width 32 --epochs 200 \
    --T 400 --h 1 --steps 1 --fractional --m 24 --groupnorm --approx_act --diag
"""

import argparse, time, collections, numpy as np, weakref
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd.functional import jvp
try:
    import torchattacks
except Exception:
    torchattacks = None
import torch.cuda as tcuda

# ─────────────────────────────────────────────────────────────────────────────
#  ESRK-15 tableau (2S-8 structure)
# ─────────────────────────────────────────────────────────────────────────────
a_np = np.array([
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
], dtype=np.float64)
b_np = np.array([
    0.035898932499408134, 0.035898932499408134, 0.035898932499408134,
    0.035898932499408134, 0.035898932499408134, 0.035898932499408134,
    0.035898932499408134, 0.035898932499408134, 0.006612457947210495,
    0.21674686949693006, 0.0, 0.42264597549826616, 0.03276149074985981,
    0.0330623263939421, 0.0009799086295048407
], dtype=np.float64)

a = torch.tensor(a_np, dtype=torch.float32)
b = torch.tensor(b_np, dtype=torch.float32)
c = a.tril(-1).sum(1)

# ─────────────────────────────────────────────────────────────────────────────
#  Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
diag = collections.defaultdict(list)
def reset_diag(): diag.clear()
def log_stage_delta(Y, x):
    denom = x.norm().clamp_min(1e-12)
    diag['delta_norm'].append((Y - x).norm().div(denom).item())
def log_grad_norm(model):
    g = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
    diag['grad_norm'].append(g.item())

# ─────────────────────────────────────────────────────────────────────────────
#  Vector field & blocks
# ─────────────────────────────────────────────────────────────────────────────
class ApproxSiLU(nn.Module):
    def forward(self, x):
        x = torch.clamp(x, -4, 4)
        return x * (0.5 + 0.25*x - (1/12)*x**2 + (1/48)*x**3)

def make_f(ch, use_groupnorm=False, approx_act=False):
    act = ApproxSiLU() if approx_act else nn.SiLU()
    layers = [nn.Conv2d(ch, ch, 3, padding=1, bias=False)]
    if use_groupnorm:
        layers.append(nn.GroupNorm(8, ch))
    layers += [act, nn.Conv2d(ch, ch, 3, padding=1, bias=False)]
    net = nn.Sequential(*layers)
    class VF(nn.Module):
        def __init__(self): super().__init__(); self.net = net
        def forward(self, t, x): return self.net(x)
    return VF()

class ScaleVF(nn.Module):
    """Scale RHS by T and (optionally) time argument."""
    def __init__(self, f_core, T): super().__init__(); self.f=f_core; self.T=float(T)
    def forward(self, t, x): return self.T * self.f(t*self.T, x)

# ─────────────────────────────────────────────────────────────────────────────
#  ESRK + Fractional blocks
# ─────────────────────────────────────────────────────────────────────────────
class ESRKBlock(nn.Module):
    def __init__(self, f, a, b, c, h=1.0, steps=1, diag=False):
        super().__init__()
        self.f, self.h, self.steps, self.diag = f, float(h), int(steps), diag
        A_l = torch.as_tensor(a, dtype=torch.float32).tril(-1)
        self.register_buffer("A_l", A_l)
        self.register_buffer("b",  torch.as_tensor(b, dtype=torch.float32))
        self.register_buffer("c",  torch.as_tensor(c, dtype=torch.float32))
        self.s = A_l.shape[0]
        self._last = None
    def vf(self, t, x): return self.f(t, x)
    def _compute_last(self):
        last = torch.zeros(self.s, dtype=torch.long)
        for j in range(self.s):
            for i in range(self.s-1, -1, -1):
                if self.A_l[i, j] != 0: last[j] = i; break
        self._last = last
    def forward(self, x, t0=0.0):
        t = t0
        for _ in range(self.steps):
            x = self._step(x, t); t += self.h
        return x
    def _step(self, x_in, t):
        h, A_l, b, c, s = self.h, self.A_l, self.b, self.c, self.s
        if self._last is None: self._compute_last()
        last = self._last
        R1 = R2 = None; idx1 = idx2 = None; tmp = {}; sum_bk = None
        for i in range(s):
            Y = x_in
            for j in range(i):
                k = R1 if j==idx1 else (R2 if j==idx2 else tmp[j])
                Y = Y + h * A_l[i, j] * k
            if self.diag: log_stage_delta(Y, x_in)
            k_new = self.f(t + c[i]*h, Y)
            if b[i] != 0:
                sum_bk = k_new * (h*b[i]) if sum_bk is None else sum_bk + h*b[i]*k_new
            placed = False
            if idx1 is None or i >= last[idx1]:
                R1, idx1 = k_new, i; placed = True
            elif idx2 is None or i >= last[idx2]:
                R2, idx2 = k_new, i; placed = True
            if not placed: tmp[i] = k_new
        return x_in + sum_bk

class FracODEBlock(nn.Module):
    """
    Fractional ODE via SOE memory:
      dZ_j = -λ_j Z_j + ω_j f(t,x),  dx = sum_j Z_j
    Uses ESRKBlock on the augmented state. ω normalized each forward.
    λ initialized log-uniform on [1/Heff, 3], where Heff = T (rescaled, h=1) or h (direct).
    """
    def __init__(self, f_core, a, b, c, h=1.0, steps=1, m=16, T=None, diag=False):
        super().__init__()
        self.f_core = f_core
        self.m, self.diag = int(m), bool(diag)
        H_eff = float(T) if (T is not None) else float(h)
        H_eff = max(H_eff, 1.0)
        lam_init = torch.exp(torch.linspace(np.log(1.0/H_eff), np.log(3.0), self.m))
        om_init  = torch.ones(self.m) / self.m
        self.log_lam = nn.Parameter(torch.log(lam_init))
        self.log_om  = nn.Parameter(torch.log(om_init))
        class AugVF(nn.Module):
            def __init__(self, outer):
                super().__init__()
                object.__setattr__(self, "o_ref", weakref.proxy(outer))
            def forward(self, t, aug):
                o = self.o_ref
                B, D, H, W = aug.shape
                m = o.m; C = D // (1+m)
                x = aug[:, :C]
                Z = aug[:, C:].view(B, m, C, H, W)
                fx = o.f_core(t, x)                               # B,C,H,W
                lam = F.softplus(o.log_lam).view(1, m, 1, 1, 1)
                omr = F.softplus(o.log_om)
                om  = (omr / (omr.sum() + 1e-12)).view(1, m, 1, 1, 1)
                dZ = -lam * Z + om * fx.unsqueeze(1)             # B,m,C,H,W
                dx = dZ.sum(dim=1)                               # B,C,H,W
                return torch.cat([dx, dZ.view(B, m*C, H, W)], dim=1)
        self._aug_vf = AugVF(self)
        self._ode = ESRKBlock(self._aug_vf, a, b, c, h=h, steps=steps, diag=diag)
    def vf(self, t, x): return self.f_core(t, x)
    def forward(self, x):
        B, C, H, W = x.shape
        Z0 = torch.zeros(B, self.m*C, H, W, device=x.device, dtype=x.dtype)
        aug0 = torch.cat([x, Z0], dim=1)
        augT = self._ode(aug0)
        return augT[:, :C]

# ─────────────────────────────────────────────────────────────────────────────
#  Euler / RK4 blocks
# ─────────────────────────────────────────────────────────────────────────────
class EulerBlock(nn.Module):
    def __init__(self, f, h=1.0, steps=1):
        super().__init__(); self.f, self.h, self.steps = f, h, steps
    def forward(self, x, t0: float = 0.0):
        t = t0
        for _ in range(self.steps): x = x + self.h*self.f(t,x); t += self.h
        return x

class RK4Block(nn.Module):
    def __init__(self, f, h=1.0, steps=1):
        super().__init__(); self.f, self.h, self.steps = f, h, steps
    def forward(self, x, t0: float = 0.0):
        t = t0
        for _ in range(self.steps): x = self._step(x,t); t += self.h
        return x
    def _step(self, x, t: float):
        h, f = self.h, self.f
        k1 = f(t, x)
        k2 = f(t + 0.5*h, x + 0.5*h*k1)
        k3 = f(t + 0.5*h, x + 0.5*h*k2)
        k4 = f(t + h,     x + h*k3)
        return x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

class FracODEBlockRK4(nn.Module):
    """
    Fractional ODE via SOE memory, stepped with RK4:
      dZ_j = -λ_j Z_j + ω_j f(t,x),  dx = sum_j Z_j
    Uses RK4Block on the augmented state. ω normalized each forward.
    λ initialized log-uniform on [1/Heff, 3], where Heff = T (rescaled, h=1) or h (direct).
    """
    def __init__(self, f_core, h=1.0, steps=1, m=16, T=None, diag=False):
        super().__init__()
        self.f_core = f_core
        self.m, self.diag = int(m), bool(diag)
        H_eff = float(T) if (T is not None) else float(h)
        H_eff = max(H_eff, 1.0)
        lam_init = torch.exp(torch.linspace(np.log(1.0/H_eff), np.log(3.0), self.m))
        om_init  = torch.ones(self.m) / self.m
        self.log_lam = nn.Parameter(torch.log(lam_init))
        self.log_om  = nn.Parameter(torch.log(om_init))

        class AugVF(nn.Module):
            def __init__(self, outer):
                super().__init__()
                object.__setattr__(self, "o_ref", weakref.proxy(outer))
            def forward(self, t, aug):
                o = self.o_ref
                B, D, H, W = aug.shape
                m = o.m; C = D // (1+m)
                x = aug[:, :C]
                Z = aug[:, C:].view(B, m, C, H, W)
                fx = o.f_core(t, x)
                lam = F.softplus(o.log_lam).view(1, m, 1, 1, 1)
                omr = F.softplus(o.log_om)
                om  = (omr / (omr.sum() + 1e-12)).view(1, m, 1, 1, 1)
                dZ = -lam * Z + om * fx.unsqueeze(1)
                dx = dZ.sum(dim=1)
                return torch.cat([dx, dZ.view(B, m*C, H, W)], dim=1)

        self._aug_vf = AugVF(self)
        self._ode = RK4Block(self._aug_vf, h=h, steps=steps)

    def vf(self, t, x):  # for diagnostics compatibility
        return self.f_core(t, x)

    def forward(self, x):
        B, C, H, W = x.shape
        Z0 = torch.zeros(B, self.m*C, H, W, device=x.device, dtype=x.dtype)
        aug0 = torch.cat([x, Z0], dim=1)
        augT = self._ode(aug0)
        return augT[:, :C]

# ─────────────────────────────────────────────────────────────────────────────
#  Tiny model
# ─────────────────────────────────────────────────────────────────────────────
class TinyNODE(nn.Module):
    def __init__(self, solver='esrk', width=32, h=1.0, steps=1,
                 use_groupnorm=False, approx_act=False, diag=False,
                 fractional=False, m=16, T=None):
        super().__init__()
        act = ApproxSiLU() if approx_act else nn.SiLU()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False), act, nn.MaxPool2d(2)
        )
        f = make_f(width, use_groupnorm, approx_act)
        if T is not None:
            f = ScaleVF(f, T)  # rescale RHS by T (keep h=1 recommended)

        if solver == 'euler':
            self.ode = EulerBlock(f, h=0.6667, steps=15)
        elif solver == 'rk4':
            if fractional:
                self.ode = FracODEBlockRK4(f, h=h, steps=steps, m=m, T=T, diag=diag)
            else:
                self.ode = RK4Block(f, h=h, steps=steps)
        elif solver == 'esrk':
            if fractional:
                self.ode = FracODEBlock(f, a, b, c, h=h, steps=steps, m=m, T=T, diag=diag)
            else:
                self.ode = ESRKBlock(f, a, b, c, h=h, steps=steps, diag=diag)
        else:
            raise ValueError("solver must be euler, rk4 or esrk")

        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width, 10, bias=False))

    def forward(self, x):
        return self.head(self.ode(self.encoder(x)))

# ─────────────────────────────────────────────────────────────────────────────
#  Spectral-norm proxy
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def estimate_spectral_norm(model, x, iters=5):
    model.eval()
    z = model.encoder(x).detach().requires_grad_(True)
    v = torch.randn_like(z); sigma = 0.0
    vf = (model.ode.vf if hasattr(model.ode, "vf") else (lambda t, z: model.ode.f(t, z)))
    for _ in range(iters):
        _, jv = jvp(lambda z_: vf(0.0, z_), (z,), (v,), create_graph=False)
        sigma = jv.norm().item(); v = jv / (sigma + 1e-12)
    return sigma

@torch.no_grad()
def validate(model, loader, device):
    model.eval(); tot=ok=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        ok += (model(x).argmax(1)==y).sum().item(); tot += y.size(0)
    return 100.0*ok/tot

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ─────────────────────────────────────────────────────────────────────────────
def train_loop(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tfm = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
    ])
    tr_ds = datasets.CIFAR10(args.data, True,  tfm, download=False)
    va_ds = datasets.CIFAR10(args.data, False, tfm, download=False)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=256,      shuffle=False, num_workers=4, pin_memory=True)

    model = TinyNODE(args.solver, width=args.width, h=args.h, steps=args.steps,
                     use_groupnorm=args.groupnorm, approx_act=args.approx_act,
                     diag=args.diag, fractional=args.fractional, m=args.m, T=args.T).to(device)
    print(f"Trainable parameters: {count_params(model):,d}")

    if args.fractional:
        # Inspect initial λ spectrum
        lam = F.softplus(model.ode.log_lam).detach().cpu().numpy()
        print("SOE λ range: [%.5f, %.5f]  m=%d" % (lam.min(), lam.max(), args.m))

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossF = nn.CrossEntropyLoss()

    start = time.time()
    for ep in range(args.epochs):
        ep0 = time.time()
        model.train()
        if device=='cuda': tcuda.reset_peak_memory_stats(device)
        for x,y in tr_ld:
            x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            lossF(model(x),y).backward()
            if args.diag: log_grad_norm(model)
            opt.step()

        train_acc = validate(model, tr_ld, device)
        val_acc   = validate(model, va_ld, device)

        xb,_ = next(iter(va_ld))
        sigma   = estimate_spectral_norm(model, xb[:args.spec_batch].to(device), args.spec_iters)
        h_sigma = (args.h if args.h is not None else 1.0) * sigma

        mem_alloc = (tcuda.memory_allocated(device) / 1e6) if device=='cuda' else 0.0
        mem_peak  = (tcuda.max_memory_allocated(device) / 1e6) if device=='cuda' else 0.0

        if args.diag:
            g_mean = np.mean(diag['grad_norm'])  if diag['grad_norm']  else 0
            d_mean = np.mean(diag['delta_norm']) if diag['delta_norm'] else 0
            extras = (f"hσ={h_sigma:5.1f}  Δ̄={d_mean:6.2e}  ∥g∥̄={g_mean:6.2f}")
        else:
            extras = f"hσ={h_sigma:5.1f}"

        print(f"Ep{ep:03d} train={train_acc:5.2f}% val={val_acc:5.2f}% "
              f"σ̂={sigma:4.3f} {extras}  mem={mem_alloc:6.0f}/{mem_peak:6.0f} MB  "
              f"time={time.time()-ep0:5.1f}s")

        reset_diag(); sched.step()

    print(f"Total time: {(time.time()-start)/60:.1f} min")
    tag = f"T{int(args.T)}" if args.T is not None else f"h{args.h}"
    torch.save(model.state_dict(), f"tiny_{args.solver}{'_frac' if args.fractional else ''}_{tag}.pth")

    if torchattacks is not None and args.eval_adv:
        atk_fgsm = torchattacks.FGSM(model, eps=8/255.)
        atk_pgd  = torchattacks.PGD(model, eps=8/255., alpha=2/255., steps=10)
        def adv_acc(atk):
            ok=tot=0; model.eval()
            for x,y in va_ld:
                x,y = x.to(device), y.to(device)
                x_adv = atk(x, y)
                ok  += (model(x_adv).argmax(1)==y).sum().item()
                tot += y.size(0)
            return 100.0 * ok / tot
        print("Adversarial FGSM-8 :", f"{adv_acc(atk_fgsm):5.2f}%")
        print("Adversarial PGD-10 :", f"{adv_acc(atk_pgd) :5.2f}%")

    if args.eval_cifar_c:
        from robustness.cifar import CIFAR10C
        for corr in ['gaussian_noise','brightness','defocus_blur']:
            ds = CIFAR10C('./cifar_c', corruption=corr, severity=3)
            acc = validate(model, DataLoader(ds,batch_size=256), device)
            print(f"CIFAR-C {corr:15s}: {acc:5.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
def parse():
    p=argparse.ArgumentParser()
    p.add_argument('--solver',choices=['euler','rk4','esrk'],default='esrk')
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--width',type=int,default=32)
    p.add_argument('--h',type=float,default=1.0, help='step size; set 1.0 when using --T')
    p.add_argument('--steps',type=int,default=1)
    p.add_argument('--T',type=float,default=None, help='total integration time; use with h=1 for rescaled mode')
    p.add_argument('--groupnorm',action='store_true')
    p.add_argument('--approx_act',action='store_true')
    p.add_argument('--diag',action='store_true',help='enable extra diagnostics')
    p.add_argument('--eval_cifar_c',action='store_true')
    p.add_argument('--eval_adv',action='store_true')
    p.add_argument('--data',type=str,default='./cifar_data')
    p.add_argument('--spec_iters',type=int,default=5)
    p.add_argument('--spec_batch',type=int,default=1)
    p.add_argument('--lr',type=float,default=1e-3)
    # Fractional options
    p.add_argument('--fractional', action='store_true', help='use fractional (SOE) augmented ODE')
    p.add_argument('--m', type=int, default=16, help='SOE terms (memory channels per feature)')
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
if __name__=='__main__':
    torch.manual_seed(244)
    args=parse()
    train_loop(args)
