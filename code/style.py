"""Shared figure style (print-friendly, colour-blind safe)."""
import matplotlib as mpl

COL = {
    "obs": "#222222",
    "muted": "#9a9a9a",
    "accent": "#b2182b",
    "const": "#999933",
    "seasonal": "#88ccee",
    "pg_r": "#0072b2",
    "pg_f": "#d55e00",
    "node": "#cc79a7",
    "arima": "#009e73",
    "naive": "#6b6b6b",
}

LABEL = {
    "naive": "Persistence",
    "arima": "ARIMA + Fourier",
    "const": "SIARW, constant $\\mathcal{R}$",
    "seasonal": "SIARW, seasonal $\\mathcal{R}$",
    "node": "Neural ODE",
    "pg_f": "PG-NODE, full-state",
    "pg_r": "PG-NODE, restricted",
}


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    })
