"""Rolling-origin forecast fits.

    python forecast.py <kind> [iters]

Fits every origin of the evaluation design in one batch and stores the
model mean trajectories (training fit + 8-week projection) and the
negative-binomial dispersion per origin. Scoring is done in score.py.
"""
import sys
import time

import numpy as np
import torch

from pgnode import ROOT, Batch, Task, final_nll, model_series, train

FIRST_ORIGIN, STEP, H = 104, 2, 8   # 2 years of training before the first forecast
RESTARTS = {"node": 2, "pg_f": 2, "pg_r": 2}  # random initialisations per origin


def origins(T):
    return list(range(FIRST_ORIGIN, T, STEP))


if __name__ == "__main__":
    torch.set_num_threads(1)
    kind = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    df = model_series()
    y = df.cases.values.astype(float)
    T = len(y)
    orgs = origins(T)
    R = RESTARTS.get(kind, 1)
    tasks = [Task(origin=o, seed=r) for r in range(R) for o in orgs]
    m = Batch(kind, tasks, y, list(df.date), T)
    t0 = time.time()
    hist = train(m, y, iters=iters, log_every=100)
    nll, mu = final_nll(m, y)
    # keep, for each origin, the initialisation with the lowest training NLL
    n = len(orgs)
    pick = np.argmin(nll.reshape(R, n), 0) * n + np.arange(n)
    phi = m.phi().detach().numpy()
    np.savez(ROOT / "results" / f"fc_{kind}.npz", mu=mu[pick], phi=phi[pick], pick=pick,
             nll_all=nll, origins=np.array(orgs), hist=hist, seconds=time.time() - t0)
    print("done", kind, round(time.time() - t0), "s")
