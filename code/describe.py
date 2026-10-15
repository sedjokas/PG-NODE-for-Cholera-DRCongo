"""Descriptive analysis of the supplied series: data-quality flags, annual
totals, and comparison with published national totals (verified sources)."""
import datetime as dt
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pgnode import ROOT, load_long
from style import apply, COL

apply()
OUT = ROOT / "results"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

df = load_long()
summary = {}

# --- annual totals and quality flags ------------------------------------------------
rows = []
for y, g in df.groupby("year"):
    v = g.cases.values
    runs = np.diff(np.flatnonzero(np.r_[True, v[1:] != v[:-1], True]))  # run lengths
    rows.append(dict(year=int(y), total=int(v.sum()), weeks=int(len(v)),
                     zeros=int((v == 0).sum()), max=int(v.max()),
                     longest_run=int(runs.max()),
                     invalid_iso=int((~g.iso_valid).sum())))
summary["annual"] = rows

# --- benchmark comparison (each published figure checked against its source) ---------
bench = [
    # label, year, last ISO week included, published, source key
    ("2023, Jan--Jul", 2023, 30, 31342, "unicef2023"),
    ("2025, to 8 Jun", 2025, 23, 29392, "who_feature2025"),
    ("2025, to 10 Aug", 2025, 32, 46800, "who_don579"),
    ("2025, to 7 Dec", 2025, 49, 64427, "unicef2025"),
    ("2026, to 31 May", 2026, 22, 28567, "who_upd38"),
]
comp = []
for label, y, w, pub, src in bench:
    g = df[(df.year == y) & (df.week <= w) & df.iso_valid]
    f = int(g.cases.sum())
    comp.append(dict(label=label, year=y, week=w, file=f, published=pub,
                     ratio=round(f / pub, 3), source=src))
summary["benchmarks"] = comp

# --- seasonality / autocorrelation on the modelled window --------------------------
m = df[df.iso_valid & (df.year >= 2022)].sort_values("date")
yv = m.cases.values.astype(float)
ly = np.log(yv)
summary["lag1_autocorr_log"] = float(np.corrcoef(ly[:-1], ly[1:])[0, 1])
summary["sd_weekly_logdiff"] = float(np.std(np.diff(ly), ddof=1))
summary["peak_week_by_year"] = {int(y): int(g.loc[g.cases.idxmax(), "week"])
                                for y, g in df[df.iso_valid].groupby("year")}
summary["modelled_weeks"] = int(len(m))
summary["modelled_total"] = int(yv.sum())
summary["excluded_2022W53"] = int(df[~df.iso_valid].cases.sum())

json.dump(summary, open(OUT / "describe.json", "w"), indent=2)

# --- figure 1: the series -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 2.8))
d = df[df.iso_valid].sort_values("date")
early = d.year < 2022
ax.plot(d.date[early], d.cases[early], color=COL["muted"], lw=1.0, label="2020–2021 (excluded)")
ax.plot(d.date[~early], d.cases[~early], color=COL["obs"], lw=1.2, label="2022–2026 (modelled)")
ax.axvline(dt.date(2025, 5, 5), color=COL["accent"], lw=0.8, ls="--")
ax.text(dt.date(2025, 5, 20), ax.get_ylim()[1] * 0.92, "outbreak\ndeclared", fontsize=7,
        color=COL["accent"], va="top")
ax.set_ylabel("Weekly suspected cases")
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(FIG / "fig1_series.pdf")
print(json.dumps(summary, indent=1))

# --- revisions: frozen extract (to 2026-W30) vs. later extract (to 2026-W41) -------------
import pandas as pd
old = pd.read_csv(ROOT / "data" / "vintage_2026-W30.csv").set_index("Week")
new = pd.read_csv(ROOT / "cholera_weekly_cases.csv").set_index("Week")
rev = {}
for col in ("2025", "2026"):
    both = old[col].notna() & new[col].notna()
    o, n = old.loc[both, col], new.loc[both, col]
    rev[col] = dict(weeks=int(both.sum()), old=int(o.sum()), new=int(n.sum()),
                    pct=round(100 * (n.sum() / o.sum() - 1), 1),
                    max_abs_pct=round(float((100 * (n / o - 1)).abs().max()), 1))
w = "W30"
rev["last_week_old"] = int(old.loc[w, "2026"])
rev["last_week_new"] = int(new.loc[w, "2026"])
rev["missing_in_new"] = {c: [i for i in new.index if pd.isna(new.loc[i, c]) and pd.notna(old.loc[i, c])]
                         for c in ("2025", "2026")}
summary["revisions"] = rev
json.dump(summary, open(OUT / "describe.json", "w"), indent=2)
print(json.dumps(rev, indent=1))
