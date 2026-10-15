"""Zone-level PG-NODE with DHIS2 completeness as an observation offset.

Each batch element is one health zone. The expected count is

    E[y_jt] = rho0 * c_jt * (symptomatic exits in zone j, week t)

where c_jt in (0, 1] is the DHIS2 reporting rate (ACTUAL_REPORTS /
EXPECTED_REPORTS) of the dataset for zone j and week t, entered as data.

    python zone_model.py selftest

runs a simulation-recovery test: synthetic zones whose reporting
completeness drops mid-series, fitted with and without the offset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from pgnode import Batch, Constants, Task, nb_logpmf, train  # noqa: E402


class ZoneBatch(Batch):
    """Batch whose elements are zones; `completeness` is (zones, weeks)."""

    def __init__(self, kind, tasks, y_zones, dates, n_weeks, completeness,
                 populations, **kw):
        # Batch computes data-driven scales from a 1-D series; use the zone mean
        super().__init__(kind, tasks, y_zones.mean(0), dates, n_weeks,
                         const=Constants(N=float(np.mean(populations))), **kw)
        self.comp = torch.as_tensor(completeness, dtype=torch.float64).clamp(1e-3, 1.0)
        ybar = y_zones[:, : tasks[0].origin].mean(1) / np.clip(completeness[:, : tasks[0].origin].mean(1), 1e-3, None)
        self.Iscale = torch.as_tensor(ybar / (7 * self.rho.numpy() * self.c.gamma)).clamp_min(1.0)
        y0 = np.maximum(y_zones[:, :4].mean(1), 1.0)
        with torch.no_grad():
            self.logI0.copy_(torch.as_tensor(np.log(y0 / (7 * self.rho.numpy() * self.c.gamma))))

    def _ascertainment(self):
        return self.comp


def train_zones(model, Y, iters=800, lr=0.02):
    """pgnode.train with one observed series per zone (Y: zones x weeks)."""
    return train(model, Y, iters=iters, lr=lr, log_every=100)


def selftest(n_zones=6, n_weeks=156, seed=1):
    import datetime as dt
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    dates = [dt.date(2022, 1, 3) + dt.timedelta(weeks=k) for k in range(n_weeks)]
    tasks = [Task(origin=n_weeks, seed=0, label=f"z{j}") for j in range(n_zones)]

    # completeness: stable ~0.9, then in half the zones drops to ~0.45 from week 80
    comp = np.full((n_zones, n_weeks), 0.9)
    comp[: n_zones // 2, 80:] = 0.45
    comp = np.clip(comp + rng.normal(0, 0.03, comp.shape), 0.05, 1)

    # truth: seasonal model with zone-specific level
    truth = ZoneBatch("seasonal", tasks, np.ones((n_zones, n_weeks)) * 50, dates, n_weeks,
                      comp, [2e6] * n_zones)
    with torch.no_grad():
        truth.a0.copy_(torch.as_tensor(rng.normal(0.35, 0.05, n_zones)))
        truth.af[:, 0] = 0.25
        truth.RI_raw.fill_(-3.0)
        truth.logI0.fill_(np.log(300.0))
        mu_true, _ = truth()
    Y = rng.negative_binomial(20, 20 / (20 + mu_true.numpy()))
    true_trans = (truth.a0[:, None] + truth.af @ truth.seas.T).detach()

    res = {}
    for name, c in (("with_offset", comp), ("no_offset", np.ones_like(comp))):
        m = ZoneBatch("pg_r", tasks, Y.astype(float), dates, n_weeks, c, [2e6] * n_zones)
        train_zones(m, Y.astype(float))
        with torch.no_grad():
            _, d = m(diagnostics=True)
        # compare total R(t) (direct + environmental) after the drop, dropped zones
        RW_true = torch.nn.functional.softplus(true_trans[:, :-1:2])   # solver-step starts
        R_true = truth.R_I()[:, None] + RW_true
        R_hat = m.R_I()[:, None] + d["RW"]
        k0 = 80 * m.spw
        err = (R_hat[: n_zones // 2, k0:] - R_true[: n_zones // 2, k0:]).abs().mean().item()
        res[name] = err
        print(name, "mean |R_hat - R_true| after the drop, affected zones:", round(err, 4))
    return res


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        print(__doc__)
