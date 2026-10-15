"""Batched SIARW / PG-NODE models for national weekly cholera counts (DRC).

Every model in a batch is an independent fit (its own parameters, its own
training window, its own fixed constants). Batching only shares the Python
loop of the ODE solver, so many forecast origins, seeds or profile values
train in a single pass.

State per batch element: [S, I, A, R, w, C]
  S, I, A, R  persons; w = normalised environmental concentration
  C           cumulative symptomatic exits gamma*I (reported with prob. rho)
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "vintage_2026-W30.csv"   # series analysed in Paper A (frozen)
(ROOT / "results").mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def load_long() -> pd.DataFrame:
    """Wide CSV -> long table with ISO-week Monday dates."""
    wide = pd.read_csv(DATA)
    rows = []
    for col in wide.columns[1:]:
        year = int(col)
        for _, r in wide.iterrows():
            v = r[col]
            if pd.isna(v):
                continue
            week = int(str(r["Week"])[1:])
            try:
                d = dt.date.fromisocalendar(year, week, 1)
                ok = True
            except ValueError:
                d, ok = None, False
            rows.append(dict(year=year, week=week, cases=int(v), date=d, iso_valid=ok))
    return pd.DataFrame(rows)


def model_series(first_year: int = 2022) -> pd.DataFrame:
    df = load_long()
    df = df[df.iso_valid & (df.year >= first_year)].sort_values("date").reset_index(drop=True)
    gaps = pd.to_datetime(df.date).diff().dt.days.dropna()
    assert (gaps == 7).all(), "series is not consecutive weekly"
    return df


# ----------------------------------------------------------------------------
# Fixed constants (per day)
# ----------------------------------------------------------------------------

@dataclass
class Constants:
    N: float = 109.3e6           # population, DRC 2024 (UN WPP 2024)
    gamma: float = 1 / 5         # recovery, symptomatic and asymptomatic
    p: float = 0.2               # symptomatic fraction
    eta: float = 0.1             # relative infectiousness of asymptomatic
    mu: float = 1 / (61 * 365)   # natural mortality = birth rate
    sigma: float = 1 / (3 * 365) # waning of infection-derived immunity
    days_per_step: float = 7 / 3

    @property
    def D(self) -> float:
        """Infectious-person-days per new infection, weighted by infectiousness."""
        return self.p / self.gamma + (1 - self.p) * self.eta / self.gamma


KINDS = ("const", "seasonal", "pg_r", "pg_f", "node")


@dataclass
class Task:
    """One independent fit."""
    origin: int                  # number of training weeks
    seed: int = 0
    rho: float = 0.5             # reporting probability of a symptomatic case
    tau: float = 0.05            # bound on |delta| as fraction of I scale per day
    delta_w: float = 0.33        # net loss rate of V. cholerae in water, per day (Codeco 2001)
    R_I: float | None = None     # fixed direct-transmission R (None = estimated)
    kappa: float | None = None   # log ascertainment step at declaration (None = absent)
    use_delta: bool = True
    label: str = ""


DECLARATION = dt.date(2025, 5, 5)


# ----------------------------------------------------------------------------
# Batched MLP
# ----------------------------------------------------------------------------

class BMLP(nn.Module):
    def __init__(self, B: int, nin: int, H: int, nout: int, gen: torch.Generator):
        super().__init__()
        def w(i, o, s=1.0):
            return nn.Parameter(torch.randn(B, i, o, generator=gen) * s / math.sqrt(i))
        self.W1, self.b1 = w(nin, H), nn.Parameter(torch.zeros(B, 1, H))
        self.W2, self.b2 = w(H, H), nn.Parameter(torch.zeros(B, 1, H))
        self.W3, self.b3 = w(H, nout, 0.1), nn.Parameter(torch.zeros(B, 1, nout))

    def forward(self, x):  # x: (B, n, nin)
        h = torch.tanh(torch.baddbmm(self.b1, x, self.W1))
        h = torch.tanh(torch.baddbmm(self.b2, h, self.W2))
        return torch.baddbmm(self.b3, h, self.W3)

    def sq_norm(self):  # per batch element
        return sum((p ** 2).flatten(1).sum(1) for p in self.parameters())


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

class Batch(nn.Module):
    def __init__(self, kind: str, tasks: list[Task], y: np.ndarray, dates,
                 n_weeks_sim: int, const: Constants = Constants(), hidden: int = 32):
        super().__init__()
        assert kind in KINDS
        self.kind, self.tasks, self.c = kind, tasks, const
        B = self.B = len(tasks)
        self.n_weeks = n_weeks_sim
        self.spw = int(round(7 / const.days_per_step))       # solver steps per week
        self.h = 7 / self.spw
        self.T_days = 7 * n_weeks_sim

        f = lambda a: torch.tensor(a, dtype=torch.float64)
        self.rho = f([t.rho for t in tasks])
        self.tau = f([t.tau for t in tasks])
        self.dw = f([t.delta_w for t in tasks])
        self.use_delta = f([float(t.use_delta) for t in tasks])
        self.origin = torch.tensor([t.origin for t in tasks])

        # I scale from the data (typical infectious stock implied by reports)
        ybar = np.array([y[: t.origin].mean() for t in tasks])
        self.Iscale = f(ybar / (7 * self.rho.numpy() * const.gamma))
        y0 = np.array([max(y[:4].mean(), 1.0)] * B)
        I0 = y0 / (7 * self.rho.numpy() * const.gamma)

        gen = torch.Generator().manual_seed(1000 + 7919 * tasks[0].seed)
        self.force_zero_delta = False

        self.logI0 = nn.Parameter(f(np.log(I0)))
        self.log_phi = nn.Parameter(torch.full((B,), math.log(10.0)))

        # direct transmission R_I (estimated or fixed)
        self.RI_fixed = f([np.nan if t.R_I is None else t.R_I for t in tasks])
        self.RI_raw = nn.Parameter(torch.full((B,), -1.0))  # softplus(-1)=0.31

        # ascertainment step at outbreak declaration
        self.kappa = f([0.0 if t.kappa is None else t.kappa for t in tasks])

        # time grid (half-steps) and features
        G = 2 * self.spw * n_weeks_sim + 1
        self.tgrid = torch.arange(G) * (self.h / 2)
        d0 = dates[0]
        doy0 = (d0 - dt.date(d0.year, 1, 1)).days
        self.decl_day = (DECLARATION - d0).days
        ang = 2 * math.pi * (doy0 + self.tgrid) / 365.25
        seas = torch.stack([torch.sin(ang), torch.cos(ang),
                            torch.sin(2 * ang), torch.cos(2 * ang)], -1)   # (G,4)
        t_or = (self.origin * 7).double()                                  # (B,)
        tc = torch.minimum(self.tgrid[None, :], t_or[:, None]) / self.T_days  # clamp at origin
        self.feat = torch.cat([tc[..., None], seas[None].expand(B, -1, -1)], -1)  # (B,G,5)
        self.seas = seas

        if kind == "const":
            self.a0 = nn.Parameter(torch.zeros(B))
        elif kind == "seasonal":
            self.a0 = nn.Parameter(torch.zeros(B))
            self.af = nn.Parameter(torch.zeros(B, 4))
        elif kind == "pg_r":
            self.net = BMLP(B, 5, hidden, 2, gen)
        elif kind == "pg_f":
            self.net = BMLP(B, 5 + 4, hidden, 2, gen)
        if kind in ("const", "seasonal", "pg_r", "pg_f"):
            # start every element at total R close to 1, whatever R_I is fixed to
            ri0 = torch.where(torch.isnan(self.RI_fixed), nn.functional.softplus(self.RI_raw.detach()),
                              torch.nan_to_num(self.RI_fixed))
            g0 = torch.log(torch.expm1((1.0 - ri0).clamp_min(0.05)))
            with torch.no_grad():
                if kind in ("const", "seasonal"):
                    self.a0.copy_(g0)
                else:
                    self.net.b3[:, 0, 0] = g0
        elif kind == "node":
            self.net = BMLP(B, 5 + 3, hidden, 3, gen)
            with torch.no_grad():
                self.net.W3.mul_(0.01)
            self.z0 = nn.Parameter(torch.stack([f(np.log(y0)), torch.zeros(B), torch.zeros(B)], 1))

    # -- transmission pieces -------------------------------------------------
    def R_I(self):
        free = nn.functional.softplus(self.RI_raw)
        return torch.where(torch.isnan(self.RI_fixed), free, torch.nan_to_num(self.RI_fixed))

    def _time_outputs(self):
        """Precomputed (B,G) g and h for models whose inputs exclude the state."""
        if self.kind == "const":
            g = self.a0[:, None].expand(-1, self.tgrid.numel())
            return g, torch.zeros_like(g)
        if self.kind == "seasonal":
            g = self.a0[:, None] + self.af @ self.seas.T
            return g, torch.zeros_like(g)
        if self.kind == "pg_r":
            out = self.net(self.feat)
            return out[..., 0], out[..., 1]
        return None, None

    def _state_feats(self, x):
        c = self.c
        I, A, w = x[:, 1], x[:, 2], x[:, 4]
        s = self.Iscale
        return torch.stack([
            torch.log(I.clamp_min(1e-6) / s),
            torch.log(A.clamp_min(1e-6) / (s * (1 - c.p) / c.p)),
            torch.log(w.clamp_min(1e-12) * c.N / (s * (1 + c.eta * (1 - c.p) / c.p))),
            100 * (c.N - x[:, 0]) / c.N,
        ], 1)

    def rhs(self, x, k, g=None, hd=None):
        """x: (B,6); k: half-step grid index."""
        c = self.c
        if self.kind == "pg_f":
            inp = torch.cat([self.feat[:, k], self._state_feats(x)], 1)[:, None]
            out = self.net(inp)[:, 0]
            g, hd = out[:, 0], out[:, 1]
            if self.force_zero_delta:
                hd = torch.zeros_like(hd)
        S, I, A, R, w = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
        RW = nn.functional.softplus(g)
        beta_I, beta_W = self.R_I() / c.D, RW / c.D
        lam = beta_I * (I + c.eta * A) / c.N + beta_W * w
        inf = lam * S
        delta = self.use_delta * self.tau * self.Iscale * torch.tanh(hd) \
            if self.kind in ("pg_r", "pg_f") else torch.zeros_like(S)
        dS = c.mu * c.N - inf + c.sigma * R - c.mu * S
        dI = c.p * inf - (c.gamma + c.mu) * I + delta
        dA = (1 - c.p) * inf - (c.gamma + c.mu) * A
        dR = c.gamma * (I + A) - (c.sigma + c.mu) * R - delta
        dw = self.dw * ((I + c.eta * A) / c.N - w)
        dC = c.gamma * I
        return torch.stack([dS, dI, dA, dR, dw, dC], 1), (c.p * inf, delta, RW)

    def rhs_node(self, z, k):
        inp = torch.cat([self.feat[:, k], z[:, :3]], 1)[:, None]
        dz = 0.1 * torch.tanh(self.net(inp)[:, 0])
        dC = torch.exp(z[:, 0]) / 7
        return torch.cat([dz, dC[:, None]], 1)

    def init_state(self):
        c = self.c
        I0 = torch.exp(self.logI0)
        A0 = I0 * (1 - c.p) / c.p
        w0 = (I0 + c.eta * A0) / c.N
        R0 = torch.zeros_like(I0)
        S0 = c.N - I0 - A0 - R0
        return torch.stack([S0, I0, A0, R0, w0, torch.zeros_like(I0)], 1)

    # -- solve -----------------------------------------------------------------
    def forward(self, diagnostics: bool = False, perturb: tuple | None = None,
                zero_delta: bool = False):
        """Return weekly expected reported cases (B, n_weeks) [+ diagnostics].

        perturb = (week, factor): multiply I, A, w by factor at that week start.
        """
        h, spw = self.h, self.spw
        if self.kind == "node":
            x = torch.cat([self.z0, torch.zeros(self.B, 1)], 1)
            Cw = [x[:, -1]]
            for n in range(self.n_weeks):
                for j in range(spw):
                    k = 2 * (n * spw + j)
                    k1 = self.rhs_node(x, k)
                    k2 = self.rhs_node(x + h / 2 * k1, k + 1)
                    k3 = self.rhs_node(x + h / 2 * k2, k + 1)
                    k4 = self.rhs_node(x + h * k3, k + 2)
                    x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                Cw.append(x[:, -1])
            C = torch.stack(Cw, 1)
            return (C[:, 1:] - C[:, :-1]).clamp_min(1e-6), {}

        g_all, h_all = self._time_outputs()
        if zero_delta and h_all is not None:
            h_all = torch.zeros_like(h_all)
        self.force_zero_delta = zero_delta
        x = self.init_state()
        Cw = [x[:, 5]]
        diag = {"inflow": [], "delta": [], "RW": [], "S": []}
        for n in range(self.n_weeks):
            if perturb is not None and n == perturb[0]:
                fac = torch.ones(6); fac[[1, 2, 4]] = perturb[1]
                x = x * fac
            for j in range(spw):
                k = 2 * (n * spw + j)
                if g_all is not None:
                    gg = (g_all[:, k], g_all[:, k + 1], g_all[:, k + 2])
                    hh = (h_all[:, k], h_all[:, k + 1], h_all[:, k + 2])
                else:
                    gg = hh = (None, None, None)
                k1, aux = self.rhs(x, k, gg[0], hh[0])
                k2, _ = self.rhs(x + h / 2 * k1, k + 1, gg[1], hh[1])
                k3, _ = self.rhs(x + h / 2 * k2, k + 1, gg[1], hh[1])
                k4, _ = self.rhs(x + h * k3, k + 2, gg[2], hh[2])
                x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                if diagnostics:
                    diag["inflow"].append(aux[0]); diag["delta"].append(aux[1])
                    diag["RW"].append(aux[2]); diag["S"].append(x[:, 0])
            Cw.append(x[:, 5])
        C = torch.stack(Cw, 1)
        mu = self.rho[:, None] * (C[:, 1:] - C[:, :-1])
        mu = mu * self._ascertainment()
        diag = {k: torch.stack(v, 1) for k, v in diag.items()} if diagnostics else {}
        self.force_zero_delta = False
        return mu.clamp_min(1e-6), diag

    def _ascertainment(self):
        weeks_days = torch.arange(self.n_weeks) * 7.0
        step = (weeks_days >= self.decl_day).double()
        return torch.exp(self.kappa[:, None] * step[None, :])

    def phi(self):
        return torch.exp(self.log_phi)

    def reg(self):
        return self.net.sq_norm() if hasattr(self, "net") else torch.zeros(self.B)


# ----------------------------------------------------------------------------
# Likelihood and training
# ----------------------------------------------------------------------------

def nb_logpmf(y, mu, phi):
    return (torch.lgamma(y + phi) - torch.lgamma(phi) - torch.lgamma(y + 1)
            + phi * torch.log(phi / (phi + mu)) + y * torch.log(mu / (phi + mu)))


def clip_per_element(model: nn.Module, max_norm: float):
    """Gradient-norm clipping applied separately to each batch element.

    Every parameter has the batch as its leading dimension, so elements stay
    independent: one element's gradient never rescales another's. Elements
    with a non-finite gradient skip the step."""
    ps = [p for p in model.parameters() if p.grad is not None]
    B = ps[0].shape[0]
    sq = torch.zeros(B)
    for p in ps:
        assert p.shape[0] == B
        sq = sq + (p.grad.reshape(B, -1) ** 2).sum(1)
    norm = sq.sqrt()
    ok = torch.isfinite(norm)
    scale = torch.where(ok, (max_norm / (norm + 1e-12)).clamp(max=1.0), torch.zeros(B))
    for p in ps:
        g = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        p.grad.copy_(g * scale.reshape(B, *([1] * (p.dim() - 1))))


def train(model: Batch, y: np.ndarray, iters: int = 1500, lr: float = 0.02,
          lam_reg: float = 1e-4, lam_delta: float = 0.1, log_every: int = 0,
          warmup: int = 100):
    """Adam with linear warm-up and cosine decay.

    Elements whose loss becomes non-finite are rolled back to their last
    finite parameters, with their Adam moments reset; other elements are
    unaffected."""
    yt = torch.as_tensor(np.asarray(y)[..., : model.n_weeks], dtype=torch.float64)
    Y = yt if yt.dim() == 2 else yt[None, :]   # (1 or B, weeks): one series per element allowed
    nw = Y.shape[1]
    mask = (torch.arange(nw)[None, :] < model.origin[:, None]).double()
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    lr_at = lambda it: lr * min(1.0, (it + 1) / warmup) * (
        0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * it / iters)))
    good = [p.detach().clone() for p in params]
    hist, n_rollbacks = [], 0
    for it in range(iters):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(it)
        opt.zero_grad()
        mu, diag = model(diagnostics=model.kind in ("pg_r", "pg_f"))
        ll = nb_logpmf(Y, mu[:, :nw], model.phi()[:, None]) * mask
        nll = -ll.sum(1) / mask.sum(1)
        loss = nll + lam_reg * model.reg()
        if diag:
            steps_mask = mask.repeat_interleave(model.spw, 1)
            dpen = ((diag["delta"] / (model.tau * model.Iscale)[:, None]) ** 2 * steps_mask).sum(1) / steps_mask.sum(1)
            loss = loss + lam_delta * dpen
        bad = ~torch.isfinite(loss.detach())
        with torch.no_grad():  # remember parameters that gave a finite loss
            for p, g in zip(params, good):
                g[~bad] = p[~bad]
        torch.where(bad, torch.zeros_like(loss), loss).sum().backward()
        clip_per_element(model, 50.0)
        opt.step()
        if bad.any():  # after the step, so autograd never sees the in-place edit
            n_rollbacks += int(bad.sum())
            with torch.no_grad():
                for p, g in zip(params, good):
                    p[bad] = g[bad]
                    st = opt.state.get(p)
                    if st:
                        st["exp_avg"][bad] = 0
                        st["exp_avg_sq"][bad] = 0
        if log_every and (it % log_every == 0 or it == iters - 1):
            f = nll[~bad]
            print(f"[{model.kind}] it {it:5d}  nll mean {f.mean().item():.4f}  "
                  f"min {f.min().item():.4f} max {f.max().item():.4f}  "
                  f"non-finite {int(bad.sum())}  rollbacks {n_rollbacks}", flush=True)
        hist.append(nll.detach().numpy().copy())
    model.n_rollbacks = n_rollbacks
    return np.array(hist)


def final_nll(model: Batch, y: np.ndarray):
    """Per-element training NLL (sum over training weeks), no penalties."""
    with torch.no_grad():
        mu, _ = model()
        yt = torch.tensor(y[: model.n_weeks], dtype=torch.float64)
        nw = yt.numel()
        mask = (torch.arange(nw)[None, :] < model.origin[:, None]).double()
        ll = nb_logpmf(yt[None, :nw], mu[:, :nw], model.phi()[:, None]) * mask
        return (-ll.sum(1)).numpy(), mu.numpy()
