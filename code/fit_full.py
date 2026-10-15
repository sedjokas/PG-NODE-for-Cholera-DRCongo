"""Full-window fits: seeds, sensitivity analyses and profile likelihoods.

    python fit_full.py train <group>     group in {pg_r, pg_f, node, const, seasonal}
    python fit_full.py analyse
"""
import json
import math
import pickle
import sys
import time

import numpy as np
import torch

from pgnode import ROOT, Batch, Task, final_nll, model_series, train

RES = ROOT / "results"
SEEDS = (0, 1, 2)
ITERS = 2000
RI_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
KAPPA_GRID = tuple(math.log(v) for v in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0))
DW_GRID = (0.05, 0.1, 0.33, 1.0, 3.0)


def tasks_for(group, T):
    ts = [Task(origin=T, seed=s, label=f"seed{s}") for s in SEEDS]
    if group != "pg_r":
        return ts
    ts += [Task(origin=T, rho=0.1, label="rho0.1"), Task(origin=T, rho=1.0, label="rho1.0"),
           Task(origin=T, tau=0.02, label="tau0.02"), Task(origin=T, tau=0.2, label="tau0.2"),
           Task(origin=T, use_delta=False, label="nodelta")]
    for s in (0, 1):
        ts += [Task(origin=T, seed=s, R_I=r, label=f"RI{r}") for r in RI_GRID]
        ts += [Task(origin=T, seed=s, kappa=k, label=f"kappa{math.exp(k):.2f}") for k in KAPPA_GRID]
    ts += [Task(origin=T, delta_w=d, label=f"dw{d}") for d in DW_GRID]
    return ts


def do_train(group):
    torch.set_num_threads(1)
    df = model_series()
    y = df.cases.values.astype(float)
    T = len(y)
    tasks = tasks_for(group, T)
    m = Batch(group, tasks, y, list(df.date), T)
    t0 = time.time()
    hist = train(m, y, iters=ITERS, log_every=100)
    torch.save({"state": m.state_dict(), "tasks": tasks, "hist": hist,
                "seconds": time.time() - t0}, RES / f"full_{group}.pt")
    print("done", group, round(time.time() - t0), "s")


def load(group):
    df = model_series()
    y = df.cases.values.astype(float)
    T = len(y)
    ck = torch.load(RES / f"full_{group}.pt", weights_only=False)
    m = Batch(group, ck["tasks"], y, list(df.date), T)
    m.load_state_dict(ck["state"])
    return m, ck, y, df


def analyse():
    out = {}
    for group in ("const", "seasonal", "node", "pg_f", "pg_r"):
        f = RES / f"full_{group}.pt"
        if not f.exists():
            continue
        m, ck, y, df = load(group)
        labels = [t.label for t in ck["tasks"]]
        nll, mu = final_nll(m, y)
        g = {"labels": labels, "nll": nll.tolist(), "mu": mu.tolist(),
             "phi": m.phi().detach().numpy().tolist(), "seconds": ck["seconds"]}
        if group in ("pg_r", "pg_f"):
            with torch.no_grad():
                mu0, d = m(diagnostics=True)
                muz, _ = m(zero_delta=True)
                wk = 150
                mup, _ = m(perturb=(wk, 0.5))
                x_mass = None
            S = d["S"].numpy()
            g["R_I"] = m.R_I().detach().numpy().tolist()
            g["R_t"] = ((m.R_I().detach().numpy()[:, None] + d["RW"].numpy()) * S / m.c.N).tolist()
            inflow, delta = d["inflow"].numpy(), d["delta"].numpy()
            g["neural_share"] = (np.abs(delta) / (inflow + np.abs(delta))).tolist()
            g["mu_zero_delta"] = muz.numpy().tolist()
            g["perturb_week"] = wk
            g["perturb_ratio"] = (mup.numpy() / mu0.numpy()).tolist()
            g["min_S_over_N"] = (S.min(1) / m.c.N).tolist()
        if group == "pg_r":
            g["phi_labels"] = labels
        out[group] = g
        print(group, "nll", np.round(nll, 2))
    # mass conservation check on one restricted fit
    m, ck, y, df = load("pg_r")
    ref_idx = [i for i, t in enumerate(ck["tasks"]) if t.label.startswith("seed")]
    with torch.no_grad():
        x = m.init_state()
        h, spw = m.h, m.spw
        g_all, h_all = m._time_outputs()
        err = 0.0
        for n in range(m.n_weeks):
            for j in range(spw):
                k = 2 * (n * spw + j)
                k1, _ = m.rhs(x, k, g_all[:, k], h_all[:, k])
                k2, _ = m.rhs(x + h / 2 * k1, k + 1, g_all[:, k + 1], h_all[:, k + 1])
                k3, _ = m.rhs(x + h / 2 * k2, k + 1, g_all[:, k + 1], h_all[:, k + 1])
                k4, _ = m.rhs(x + h * k3, k + 2, g_all[:, k + 2], h_all[:, k + 2])
                x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                rel = ((x[:, :4].sum(1) - m.c.N).abs() / m.c.N)[ref_idx]
                err = max(err, float(rel.max()))
    out["mass_rel_err_max"] = err
    pickle.dump(out, open(RES / "full_analysis.pkl", "wb"))
    print("mass conservation max rel err", err)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        do_train(sys.argv[2])
    else:
        analyse()
