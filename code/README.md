# Code — Paper A (national series)

Python 3.13. Install: `pip install -r requirements.txt`.
Run from this folder; set `OMP_NUM_THREADS=1` when running several fits in parallel.

| Step | Command | Output |
|---|---|---|
| 1. Data checks, Fig. 1 | `python describe.py` | `results/describe.json` |
| 2. Rolling-origin fits | `python forecast.py <kind>` for kind in const, seasonal, node, pg_f, pg_r | `results/fc_<kind>.npz` |
| 3. Full-window fits | `python fit_full.py train <kind>` (same kinds) | `results/full_<kind>.pt` |
| 4. Ablations, profiles | `python fit_full.py analyse` | `results/full_analysis.pkl` |
| 5. Baselines + scoring | `python score.py` | `results/scores.json` |
| 6. Tables, figures, numbers | `python make_outputs.py` | `paper/generated/`, `paper/figures/` |

Steps 2 and 3 are independent and can run in parallel (one process per kind).
On one CPU core per process, the slowest (pg_f, rolling origin) took about the time
reported in `results/fc_pg_f.npz` (`seconds`).

`pgnode.py` holds the data loader, the batched SIARW / PG-NODE models, the
negative-binomial likelihood and the training loop.
