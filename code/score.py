"""Probabilistic scoring of all models on the rolling-origin design.

Statistical baselines (persistence, ARIMA + Fourier) are fitted here; the
ODE-based models are read from results/fc_<kind>.npz.

Forecast distributions for the ODE-based models: the model's mean
trajectory, multiplied by a Gaussian random walk on the log scale with
innovation s.d. sigma_1 (the s.d. of weekly log differences of the
observed series over the 52 weeks before the origin), then negative-binomial
observation noise with the model's fitted dispersion. The same rule is
applied to every ODE-based model, so differences between them come from
their mean trajectories and dispersion only.
"""
import json
import warnings

import numpy as np
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX

from forecast import H, origins
from pgnode import ROOT, model_series

warnings.filterwarnings("ignore")
RES = ROOT / "results"
QL = np.array([0.01, 0.025, *np.arange(0.05, 0.951, 0.05), 0.975, 0.99])  # 23 levels
ALPHAS = np.array([0.02, 0.05, *np.arange(0.1, 0.91, 0.1)])                 # 11 intervals
NS = 4000
ODE_KINDS = ["const", "seasonal", "node", "pg_f", "pg_r"]


def sigma1(y, o):
    ly = np.log(y[max(0, o - 53):o])
    return float(np.std(np.diff(ly), ddof=1))


def wis(q, yobs):
    """Weighted interval score (Bracher et al. 2021) from the 23 quantiles.

    Returns total, dispersion, overprediction, underprediction."""
    qd = dict(zip(np.round(QL, 3), q))
    med = qd[0.5]
    K = len(ALPHAS)
    disp = over = under = 0.0
    for a in ALPHAS:
        lo, hi = qd[round(a / 2, 3)], qd[round(1 - a / 2, 3)]
        w = a / 2
        disp += w * (hi - lo)
        over += w * (2 / a) * max(lo - yobs, 0)
        under += w * (2 / a) * max(yobs - hi, 0)
    ae = 0.5 * abs(yobs - med)
    # the absolute-error term is split into over/under by sign, as in scoringutils
    over += ae if med > yobs else 0
    under += ae if med < yobs else 0
    n = K + 0.5
    return (disp + over + under) / n, disp / n, over / n, under / n


def q_ode(mu_path, phi, s1, rng):
    """Quantiles for horizons 1..len(mu_path)."""
    h = len(mu_path)
    rw = np.cumsum(rng.normal(0, s1, size=(NS, h)), 1)
    m = mu_path[None, :] * np.exp(rw)
    p = phi / (phi + m)
    ys = rng.negative_binomial(phi, p)
    return np.quantile(ys, QL, axis=0).T  # (h, 23)


def q_naive(y, o, s1):
    hs = np.arange(1, H + 1)
    z = stats.norm.ppf(QL)
    return np.exp(np.log(y[o - 1]) + s1 * np.sqrt(hs)[:, None] * z[None, :])


def fourier(t, K=2, period=365.25 / 7):
    ang = 2 * np.pi * np.asarray(t)[:, None] * np.arange(1, K + 1)[None, :] / period
    return np.hstack([np.sin(ang), np.cos(ang)])


def q_arima(y, o, t_offset):
    ly = np.log1p(y[:o])
    X = fourier(np.arange(o) + t_offset)
    Xf = fourier(np.arange(o, o + H) + t_offset)
    best = None
    for p in (0, 1, 2):
        for q in (0, 1, 2):
            try:
                r = SARIMAX(ly, exog=X, order=(p, 1, q), trend="n").fit(disp=False)
            except Exception:
                continue
            if best is None or r.aic < best.aic:
                best = r
    fc = best.get_forecast(H, exog=Xf)
    mean, se = fc.predicted_mean, np.sqrt(fc.var_pred_mean)
    z = stats.norm.ppf(QL)
    return np.expm1(mean[:, None] + se[:, None] * z[None, :]).clip(0), best.model_orders


def main():
    df = model_series()
    y = df.cases.values.astype(float)
    T = len(y)
    orgs = origins(T)
    doy0 = df.date.iloc[0].timetuple().tm_yday / 7
    rng = np.random.default_rng(20261014)

    fc = {}
    for k in ODE_KINDS:
        f = RES / f"fc_{k}.npz"
        if f.exists():
            fc[k] = np.load(f)

    rows, failures = [], []
    for i, o in enumerate(orgs):
        s1 = sigma1(y, o)
        hmax = min(H, T - o)
        preds = {"naive": q_naive(y, o, s1)}
        preds["arima"], _ = q_arima(y, o, doy0)
        for k, d in fc.items():
            assert d["origins"][i] == o
            path = d["mu"][i, o:o + H]
            if not (np.isfinite(path).all() and np.isfinite(d["phi"][i])):
                failures.append(dict(model=k, origin=int(o)))   # fit or projection failed
                continue
            preds[k] = q_ode(path, d["phi"][i], s1, rng)
        for k, Q in preds.items():
            for h in range(1, hmax + 1):
                q = Q[h - 1]
                yobs = y[o + h - 1]
                w, dsp, ov, un = wis(q, yobs)
                lw = wis(np.log1p(q), np.log1p(yobs))[0]
                qd = dict(zip(np.round(QL, 3), q))
                rows.append(dict(model=k, origin=int(o), h=h, y=yobs, median=float(qd[0.5]),
                                 wis=w, disp=dsp, over=ov, under=un, logwis=lw,
                                 in50=bool(qd[0.25] <= yobs <= qd[0.75]),
                                 in95=bool(qd[0.025] <= yobs <= qd[0.975]),
                                 pit=float(np.interp(yobs, q, QL, left=0, right=1)),
                                 growth=bool(y[o - 1] > y[o - 5])))
        print("origin", o, flush=True)
    json.dump(rows, open(RES / "scores.json", "w"))
    json.dump(failures, open(RES / "failures.json", "w"))
    print("failures", failures)


if __name__ == "__main__":
    main()
