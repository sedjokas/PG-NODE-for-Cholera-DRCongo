"""Generate every number, table and figure used in the paper from results/.

Writes paper/generated/*.tex and paper/figures/*.pdf. Nothing in the paper's
results section is typed by hand.
"""
import datetime as dt
import json
import math
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pgnode import ROOT, model_series
from style import COL, LABEL, apply

apply()
RES, GEN, FIG = ROOT / "results", ROOT / "paper" / "generated", ROOT / "paper" / "figures"
GEN.mkdir(exist_ok=True, parents=True)
FIG.mkdir(exist_ok=True, parents=True)
ORDER = ["naive", "arima", "const", "seasonal", "node", "pg_f", "pg_r"]
TEXLABEL = {k: v.replace("$\\mathcal{R}$", "$\\R$") for k, v in LABEL.items()}

macros = {}


def datefmt(a, n=4):
    loc = mdates.AutoDateLocator(maxticks=n)
    a.xaxis.set_major_locator(loc)
    a.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def mac(name, value):
    macros[name] = value


def fmt(x, d=2):
    return f"{x:.{d}f}"


# ---------------------------------------------------------------------------
# Data description
# ---------------------------------------------------------------------------
desc = json.load(open(RES / "describe.json"))
ann = {r["year"]: r for r in desc["annual"]}
mac("nAllWeeks", sum(r["weeks"] for r in desc["annual"]))
mac("totTwentyTwenty", ann[2020]["total"])
mac("zerosTwentyTwenty", ann[2020]["zeros"])
mac("runTwentyTwenty", ann[2020]["longest_run"])
mac("nWeeks", desc["modelled_weeks"])
mac("nCases", desc["modelled_total"])
mac("nExclWfiftythree", desc["excluded_2022W53"])
mac("autocorr", fmt(desc["lag1_autocorr_log"]))
mac("sdlogdiff", fmt(desc["sd_weekly_logdiff"]))
ratios = [b["ratio"] for b in desc["benchmarks"]]
mac("ratioMin", fmt(min(ratios)))
mac("ratioMax", fmt(max(ratios)))
rv = desc["revisions"]
mac("revWeeksTwentySix", rv["2026"]["weeks"])
mac("revOldTwentySix", rv["2026"]["old"])
mac("revNewTwentySix", rv["2026"]["new"])
mac("revPctTwentySix", fmt(rv["2026"]["pct"], 1))
mac("revMaxPctTwentySix", fmt(rv["2026"]["max_abs_pct"], 0))
mac("revWeeksTwentyFive", rv["2025"]["weeks"])
mac("revPctTwentyFive", fmt(rv["2025"]["pct"], 1))
mac("revMaxPctTwentyFive", fmt(rv["2025"]["max_abs_pct"], 1))
mac("revLastOld", rv["last_week_old"])
mac("revLastNew", rv["last_week_new"])
mac("revLastPct", fmt(100 * (rv["last_week_new"] / rv["last_week_old"] - 1), 0))

with open(GEN / "tab_annual.tex", "w") as f:
    f.write("\\begin{table}[t]\\centering\\small\n\\caption{Annual totals and quality checks "
            "of the supplied national series. Longest run: longest sequence of identical "
            "consecutive weekly values.}\\label{tab:annual}\n"
            "\\begin{tabular}{lrrrrrl}\\toprule\n"
            "Year & Total & Weeks & Zero weeks & Max & Longest run & Use\\\\\\midrule\n")
    for y, r in sorted(ann.items()):
        use = "excluded" if y < 2022 else ("modelled" + ("$^a$" if r["invalid_iso"] else ""))
        f.write(f"{y} & \\num{{{r['total']}}} & {r['weeks']} & {r['zeros']} & \\num{{{r['max']}}} "
                f"& {r['longest_run']} & {use}\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n\\par\\smallskip\\footnotesize $^a$ Includes a value "
            "labelled W53, which is not an ISO week of 2022; it is excluded.\n\\end{table}\n")

with open(GEN / "tab_bench.tex", "w") as f:
    f.write("\\begin{table}[t]\\centering\\small\n\\caption{Cumulative counts in the extract "
            "versus published national totals for the same period.}\\label{tab:bench}\n"
            "\\begin{tabular}{lrrrl}\\toprule\n"
            "Period & Extract & Published & Ratio & Source\\\\\\midrule\n")
    for b in desc["benchmarks"]:
        f.write(f"{b['label']} (to W{b['week']}) & \\num{{{b['file']}}} & \\num{{{b['published']}}} & "
                f"{b['ratio']:.2f} & \\citet{{{b['source']}}}\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n\\end{table}\n")

df = model_series()
y = df.cases.values.astype(float)
dates = pd.to_datetime(df.date)
T = len(y)

# ---------------------------------------------------------------------------
# Full-window fits
# ---------------------------------------------------------------------------
full = pickle.load(open(RES / "full_analysis.pkl", "rb")) if (RES / "full_analysis.pkl").exists() else {}
if full:
    mac("massErr", f"{full['mass_rel_err_max']:.1e}".replace("e-", "\\times 10^{-") + "}")
    seeds = lambda g: [i for i, l in enumerate(full[g]["labels"]) if l.startswith("seed")]
    for g in ("const", "seasonal", "node", "pg_f", "pg_r"):
        if g in full:
            nll = np.array(full[g]["nll"])[seeds(g)]
            mac(f"nll{g.replace('_', '')}", fmt(nll.min() / T, 3))

    pr = full["pg_r"]
    lab = pr["labels"]
    nll = np.array(pr["nll"])
    best_seed = min(seeds("pg_r"), key=lambda i: nll[i])

    # replicate fits: configurations identical to the reference model
    rep = [i for i, l in enumerate(lab) if l.startswith("seed") or l in ("kappa1.00", "dw0.33")]
    rep_nll = nll[rep]
    rep_best, rep_range = np.nanmin(rep_nll), np.nanmax(rep_nll) - np.nanmin(rep_nll)
    mac("repN", len(rep))
    mac("repLo", fmt(np.nanmin(rep_nll), 0))
    mac("repHi", fmt(np.nanmax(rep_nll), 0))
    mac("repRange", fmt(rep_range, 0))

    # profiles: all initialisations per grid value; failures are NaN
    def profile(prefix):
        vals = {}
        for i, l in enumerate(lab):
            if l.startswith(prefix):
                vals.setdefault(float(l[len(prefix):]), []).append(nll[i])
        xs = np.array(sorted(vals))
        return xs, [np.array(vals[x]) for x in xs]

    profout = {}
    for k in ("RI", "kappa", "dw"):
        xs, lists = profile(k)
        best = np.array([np.nanmin(v) if np.isfinite(v).any() else np.nan for v in lists])
        d = best - rep_best
        profout[k] = (xs, d, [v - rep_best for v in lists])
        failed = xs[~np.isfinite(best)]
        outside = xs[np.isfinite(d) & (d > rep_range)]
        mac(f"prof{k}Failed", ", ".join(f"{x:g}" for x in failed) if len(failed) else "none")
        mac(f"prof{k}Outside", ", ".join(f"{x:g}" for x in outside) if len(outside) else "none")
        mac(f"prof{k}MaxDev", fmt(np.nanmax(d), 0))
    json.dump({k: dict(x=v[0].tolist(), best=v[1].tolist(), all=[a.tolist() for a in v[2]])
               for k, v in profout.items()}, open(RES / "profiles.json", "w"), indent=1)

    # sensitivity: R(t) differences
    Rt = np.array(pr["R_t"])
    spw = Rt.shape[1] // T
    Rw = Rt.reshape(len(lab), T, spw).mean(2)  # weekly mean R(t)
    base = Rw[best_seed]
    sens = {}
    for name in ("rho0.1", "rho1.0", "tau0.02", "tau0.2", "nodelta"):
        i = lab.index(name)
        sens[name] = dict(nll=nll[i] - nll[best_seed], maxdR=float(np.abs(Rw[i] - base).max()),
                          meandR=float(np.abs(Rw[i] - base).mean()))
    import itertools
    pairs = [(np.abs(Rw[a] - Rw[b]).mean(), np.abs(Rw[a] - Rw[b]).max())
             for a, b in itertools.combinations(rep, 2)]
    mac("repdRmeanLo", fmt(min(p[0] for p in pairs), 2))
    mac("repdRmeanHi", fmt(max(p[0] for p in pairs), 2))
    mac("repdRmaxLo", fmt(min(p[1] for p in pairs), 2))
    mac("repdRmaxHi", fmt(max(p[1] for p in pairs), 2))
    mac("repMeanRLo", fmt(Rw[rep].mean(1).min(), 2))
    mac("repMeanRHi", fmt(Rw[rep].mean(1).max(), 2))
    mac("meanRTauHigh", fmt(Rw[lab.index("tau0.2")].mean(), 2))
    mac("RtMin", fmt(base.min()))
    mac("RtMax", fmt(base.max()))
    mac("RIhat", fmt(pr["R_I"][best_seed]))
    mac("minSN", fmt(min(pr["min_S_over_N"][i] for i in seeds("pg_r")), 3))
    keyname = {"rho0.1": "RhoLow", "rho1.0": "RhoHigh", "tau0.02": "TauLow",
               "tau0.2": "TauHigh", "nodelta": "NoDelta"}
    for k, v in sens.items():
        key = keyname[k]
        mac(f"sens{key}dR", fmt(v["maxdR"], 3))
        mac(f"sens{key}dNLL", fmt(v["nll"], 1))

    with open(GEN / "tab_sens.tex", "w") as f:
        f.write("\\begin{table}[t]\\centering\\small\n\\caption{Sensitivity of the restricted "
                "PG-NODE full-window fit. $\\Delta$NLL: change in training negative "
                "log-likelihood (summed over weeks) relative to the reference fit; "
                "$|\\Delta\\R|$: mean and maximum absolute difference in weekly $\\R(t)$.}"
                "\\label{tab:sens}\n\\begin{tabular}{lrrr}\\toprule\n"
                "Variant & $\\Delta$NLL & mean $|\\Delta\\R|$ & max $|\\Delta\\R|$\\\\\\midrule\n")
        names = {"rho0.1": "$\\rho=0.1$", "rho1.0": "$\\rho=1$", "tau0.02": "$\\tau=0.02$",
                 "tau0.2": "$\\tau=0.2$", "nodelta": "no correction ($\\delta\\equiv0$)"}
        for k, v in sens.items():
            f.write(f"{names[k]} & {v['nll']:+.1f} & {v['meandR']:.3f} & {v['maxdR']:.3f}\\\\\n")
        f.write("\\midrule\n")
        f.write(f"replicate pairs$^a$ & $\\le${rep_range:.0f} & "
                f"{min(p[0] for p in pairs):.3f}--{max(p[0] for p in pairs):.3f} & "
                f"{min(p[1] for p in pairs):.3f}--{max(p[1] for p in pairs):.3f}\\\\\n")
        f.write("\\bottomrule\\end{tabular}\n\\par\\smallskip\\footnotesize $^a$ Range over "
                f"the {len(rep)} fits of the reference configuration, which differ only in "
                "random initialisation: optimisation noise.\n\\end{table}\n")

    # ablation numbers
    for g in ("pg_r", "pg_f"):
        G = full[g]
        si = [i for i, l in enumerate(G["labels"]) if l.startswith("seed")]
        b = min(si, key=lambda i: G["nll"][i])
        ns = np.array(G["neural_share"][b])
        mac(f"share{g.replace('_', '')}", fmt(100 * ns.mean(), 1))
        mu = np.array(G["mu"][b]); muz = np.array(G["mu_zero_delta"][b])
        mac(f"zeroDev{g.replace('_', '')}", fmt(100 * np.median(np.abs(muz / mu - 1)), 1))
        pratio = np.array(G["perturb_ratio"][b])
        wk = G["perturb_week"]
        mac(f"pert{g.replace('_', '')}Four", fmt(pratio[wk + 4], 2))
        mac(f"pert{g.replace('_', '')}Twenty", fmt(pratio[wk + 20], 2))
        # NB likelihood with delta removed
        from scipy.special import gammaln
        phi = G["phi"][b]
        def nb(mu_):
            return -(gammaln(y + phi) - gammaln(phi) - gammaln(y + 1)
                     + phi * np.log(phi / (phi + mu_)) + y * np.log(mu_ / (phi + mu_))).sum()
        mac(f"zeroNLL{g.replace('_', '')}", fmt((nb(muz) - nb(mu)) / T, 3))

    # ---- figure 2: fits and R(t)
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True,
                           gridspec_kw={"height_ratios": [1.3, 1]})
    ax[0].plot(dates, y, ".", ms=2.5, color=COL["obs"], label="Observed")
    for g in ("seasonal", "node", "pg_f", "pg_r"):
        if g in full:
            G = full[g]
            si = [i for i, l in enumerate(G["labels"]) if l.startswith("seed")]
            b = min(si, key=lambda i: G["nll"][i])
            ax[0].plot(dates, G["mu"][b], color=COL[g], lw=1.1, label=LABEL[g])
    ax[0].set_ylabel("Weekly cases")
    ax[0].legend(frameon=False, ncol=3, loc="upper left")
    for i in seeds("pg_r"):
        ax[1].plot(dates, Rw[i], color=COL["pg_r"], lw=1.0 if i == best_seed else 0.6,
                   alpha=1 if i == best_seed else 0.5)
    ax[1].axhline(1, color=COL["muted"], lw=0.6, ls=":")
    ax[1].axvline(dt.date(2025, 5, 5), color=COL["accent"], lw=0.7, ls="--")
    ax[1].set_ylabel("$\\mathcal{R}(t)$, restricted PG-NODE")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_fits.pdf")
    plt.close(fig)

    # ---- figure 4: ablation
    fig = plt.figure(figsize=(7.2, 3.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], hspace=0.35, wspace=0.3)
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])]
    ax[1].sharex(ax[0])
    for g in ("pg_r", "pg_f"):
        G = full[g]
        si = [i for i, l in enumerate(G["labels"]) if l.startswith("seed")]
        b = min(si, key=lambda i: G["nll"][i])
        ns = np.array(G["neural_share"][b])
        nsw = pd.Series(ns.reshape(T, -1).mean(1)).rolling(4, min_periods=1).mean()
        ax[0].plot(dates, 100 * nsw, color=COL[g], lw=1, label=LABEL[g])
        mu = np.array(G["mu"][b]); muz = np.array(G["mu_zero_delta"][b])
        ax[1].plot(dates, muz / mu, color=COL[g], lw=1)
        pratio = np.array(G["perturb_ratio"][b])
        wk = G["perturb_week"]
        ax[2].plot(np.arange(-4, 41), pratio[wk - 4:wk + 41], color=COL[g], lw=1.2)
    ax[0].set_ylabel("Neural share of\n$I$ inflow (%)")
    ax[0].legend(frameon=False, loc="upper right")
    ax[0].tick_params(labelbottom=False)
    ax[1].axhline(1, color=COL["muted"], lw=0.6, ls=":")
    ax[1].set_ylabel("Cases with $\\delta\\equiv0$\n/ fitted cases")
    ax[1].xaxis.set_major_locator(mdates.YearLocator())
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax[2].axhline(0.5, color=COL["muted"], lw=0.6, ls=":")
    ax[2].axvline(0, color=COL["muted"], lw=0.6, ls=":")
    ax[2].set_xlabel("Weeks after halving $I, A, w$")
    ax[2].set_ylabel("Perturbed / unperturbed cases")
    ax[2].set_ylim(0.45, 1.05)
    for a, lab_ in zip(ax, "abc"):
        a.text(-0.02, 1.02, f"({lab_})", transform=a.transAxes, ha="right", va="bottom",
               fontsize=8, fontweight="bold")
    fig.savefig(FIG / "fig4_ablation.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- figure 5: profiles
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    xl = {"RI": "$\\mathcal{R}_I$ (fixed)", "kappa": "Ascertainment step $e^{\\kappa}$",
          "dw": "$\\delta_W$ (d$^{-1}$)"}
    top = 400
    for a, k in zip(ax, ("RI", "kappa", "dw")):
        xs, d, alls = profout[k]
        a.axhspan(0, rep_range, color=COL["muted"], alpha=0.25, lw=0)
        for x, v in zip(xs, alls):
            fin = np.isfinite(v)
            a.plot([x] * fin.sum(), np.minimum(v[fin], top), "o", ms=2.2, color=COL["pg_r"], alpha=0.5)
            if not fin.any():
                a.plot([x], [top], "x", color=COL["accent"], ms=5)
        ok = np.isfinite(d)
        a.plot(xs[ok], np.minimum(d[ok], top), "-", color=COL["pg_r"], lw=1)
        a.set_xlabel(xl[k])
        if k in ("kappa", "dw"):
            a.set_xscale("log")
            a.set_xticks(xs)
            a.set_xticklabels([f"{x:g}" for x in xs], fontsize=6)
            a.minorticks_off()
    ax[0].set_ylim(-60, top + 20)
    ax[0].set_ylabel("$\\Delta$ training NLL vs. best\nreference replicate")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_profiles.pdf")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Forecast scores
# ---------------------------------------------------------------------------
if (RES / "scores.json").exists():
    S = pd.DataFrame(json.load(open(RES / "scores.json")))
    models = [m for m in ORDER if m in S.model.unique()]
    # restrict to (origin, h) pairs scored for every model (paired comparison)
    S = S[S.model.isin(models)]
    full_n = S.groupby(["origin", "h"]).model.transform("nunique")
    S = S[full_n == len(models)].copy()
    fails = json.load(open(RES / "failures.json")) if (RES / "failures.json").exists() else []
    for mname in models:
        mac(f"fail{mname.replace('_', '')}", sum(f["model"] == mname for f in fails))
    mac("nOriginsScored", S.origin.nunique())
    S["ae"] = (S["median"] - S["y"]).abs()
    agg = S.groupby("model").agg(wis=("wis", "mean"), logwis=("logwis", "mean"), ae=("ae", "mean"),
                                 c50=("in50", "mean"), c95=("in95", "mean"),
                                 disp=("disp", "mean"), over=("over", "mean"), under=("under", "mean"),
                                 n=("wis", "size")).loc[models]
    agg["rel"] = agg.wis / agg.loc["naive", "wis"]
    agg["rellog"] = agg.logwis / agg.loc["naive", "logwis"]
    agg.to_csv(RES / "scores_summary.csv")
    byh = S.groupby(["model", "h"]).wis.mean().unstack()
    byh_rel = byh / byh.loc["naive"]
    byph = S.groupby(["model", "growth"]).wis.mean().unstack()
    byph_rel = byph / byph.loc["naive"]

    best = agg.wis.idxmin()
    mac("nScored", int(agg.n.iloc[0]))
    mac("bestModel", TEXLABEL[best])
    for m in models:
        k = m.replace("_", "")
        mac(f"wis{k}", fmt(agg.loc[m, "wis"], 0))
        mac(f"rel{k}", fmt(agg.loc[m, "rel"]))
        mac(f"rellog{k}", fmt(agg.loc[m, "rellog"]))
        mac(f"cfifty{k}", fmt(100 * agg.loc[m, "c50"], 0))
        mac(f"overshare{k}", fmt(100 * agg.loc[m, "over"] / agg.loc[m, "wis"], 0))
        mac(f"cninetyfive{k}", fmt(100 * agg.loc[m, "c95"], 0))
        for h in (1, 4, 8):
            mac(f"rel{k}h{['', 'one', '', '', 'four', '', '', '', 'eight'][h]}", fmt(byh_rel.loc[m, h]))
        mac(f"rel{k}grow", fmt(byph_rel.loc[m, True]))
        mac(f"rel{k}decl", fmt(byph_rel.loc[m, False]))

    with open(GEN / "tab_scores.tex", "w") as f:
        f.write("\\begin{table}[t]\\centering\\small\n\\caption{Forecast performance over "
                f"{len(S.origin.unique())} origins and horizons 1--8 weeks "
                f"({int(agg.n.iloc[0])} forecasts per model). WIS: mean weighted interval "
                "score (cases); rel.: relative to persistence; log-WIS: WIS on the "
                "$\\log(1+y)$ scale, relative to persistence; MAE: mean absolute error of "
                "the median; C50, C95: empirical coverage of 50\\% and 95\\% intervals. "
                "Lowest WIS in bold.}\\label{tab:scores}\n"
                "\\begin{tabular}{lrrrrrr}\\toprule\n"
                "Model & WIS & rel. & rel.\\ log-WIS & MAE & C50 & C95\\\\\\midrule\n")
        for m in models:
            r = agg.loc[m]
            w = f"\\textbf{{{r.wis:.0f}}}" if m == best else f"{r.wis:.0f}"
            f.write(f"{TEXLABEL[m]} & {w} & {r.rel:.2f} & {r.rellog:.2f} & {r.ae:.0f} & "
                    f"{100 * r.c50:.0f}\\% & {100 * r.c95:.0f}\\%\\\\\n")
        f.write("\\bottomrule\\end{tabular}\n\\end{table}\n")

    with open(GEN / "tab_horizon.tex", "w") as f:
        hs = list(byh.columns)
        f.write("\\begin{table}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{3.5pt}\n\\caption{WIS relative to persistence, "
                "by forecast horizon (weeks) and by epidemic phase at the origin "
                "(growth: $y_{t_o-1}>y_{t_o-5}$).}\\label{tab:horizon}\n"
                "\\begin{tabular}{l" + "r" * (len(hs) + 2) + "}\\toprule\n"
                "Model & " + " & ".join(f"$h={h}$" for h in hs) + " & growth & decline\\\\\\midrule\n")
        for m in models:
            f.write(TEXLABEL[m] + " & " + " & ".join(f"{byh_rel.loc[m, h]:.2f}" for h in hs)
                    + f" & {byph_rel.loc[m, True]:.2f} & {byph_rel.loc[m, False]:.2f}\\\\\n")
        f.write("\\bottomrule\\end{tabular}\n\\end{table}\n")
    n_growth = S[S.model == "naive"].growth.mean()
    mac("pctGrowth", fmt(100 * n_growth, 0))

    # ---- figure 3: WIS by horizon + PIT
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for m in models:
        ax[0].plot(byh.columns, byh_rel.loc[m], "o-", ms=2.5, lw=1, color=COL[m], label=LABEL[m])
    ax[0].axhline(1, color=COL["muted"], lw=0.6, ls=":")
    ax[0].set_xlabel("Horizon (weeks)")
    ax[0].set_ylabel("WIS relative to persistence")
    ax[0].legend(frameon=False, fontsize=5.5, loc="best")
    for j, cov in enumerate(("in50", "in95")):
        c = S.groupby(["model", "h"])[cov].mean().unstack()
        for m in models:
            ax[j + 1].plot(c.columns, 100 * c.loc[m], "o-", ms=2.5, lw=1, color=COL[m])
        ax[j + 1].axhline(50 if j == 0 else 95, color=COL["muted"], lw=0.6, ls=":")
        ax[j + 1].set_xlabel("Horizon (weeks)")
        ax[j + 1].set_ylabel(f"{'50' if j == 0 else '95'}% interval coverage (%)")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_scores.pdf")
    plt.close(fig)

    # ---- figure 6: example forecasts (medians) at four origins
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.2), sharey=False)
    for a, o in zip(ax, (124, 164, 196, 224)):
        lo = o - 26
        a.plot(dates[lo:min(o + 8, T)], y[lo:min(o + 8, T)], ".", ms=2.5, color=COL["obs"])
        for m in ("naive", "arima", "node", "pg_f", "pg_r"):
            sub = S[(S.model == m) & (S.origin == o)].sort_values("h")
            if len(sub):
                a.plot(dates[o - 1:o + len(sub)], np.r_[y[o - 1], sub["median"].values],
                       color=COL[m], lw=1, label=LABEL[m])
        a.axvline(dates[o - 1], color=COL["muted"], lw=0.6, ls=":")
        a.set_title("origin " + dates[o].strftime("%G-W%V"), fontsize=7)
        datefmt(a, 3)
    ax[0].set_ylabel("Weekly cases")
    ax[0].legend(frameon=False, fontsize=5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_examples.pdf")
    plt.close(fig)

with open(GEN / "numbers.tex", "w") as f:
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"{len(macros)} macros written")
