"""
Plot SER_Melo and SER_Root as a function of staves-per-page (n),
baseline vs full-page (masking r=1.0), on the real test set.

Reads analysis_out/per_sample_metrics.csv. n is parsed from the page
stem pattern `..._n<N>`. Pages are bucketed; the rightmost bucket
aggregates the long-tail (default n>=9).

Usage:
    python plot_per_n.py
    python plot_per_n.py --csv analysis_out/per_sample_metrics.csv \
                         --out analysis_out/per_n_results.pdf
"""

from pathlib import Path
import csv as csv_mod
import re
from collections import defaultdict

import fire
import matplotlib as mpl
import matplotlib.pyplot as plt

# Match ISMIR template font (Times Roman, ptm in ismir.sty)
mpl.rcParams["font.family"]      = "serif"
mpl.rcParams["font.serif"]       = ["Times New Roman", "Times", "STIX", "DejaVu Serif"]
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["axes.unicode_minus"] = False


METRIC_LABELS = {
    "kern_ser": r"$\mathrm{SER}_{\mathrm{Melo}}$",
    "root_ser": r"$\mathrm{SER}_{\mathrm{Root}}$",
}

METRIC_STYLE = {
    "kern_ser": {"linestyle": "-",  "marker": "o"},
    "root_ser": {"linestyle": "--", "marker": "s"},
}

# (csv_prefix, display_name, color)
MODELS = [
    ("base",   "Baseline",            "#6FA8C9"),
    ("replay", "Masking $r{=}1.00$",  "#1F3A5F"),
]

N_RE = re.compile(r"_n(\d+)(?:_|$)")


def _parse_n(page: str):
    m = N_RE.search(page)
    return int(m.group(1)) if m else None


def main(
    csv:  str = "analysis_out/per_sample_metrics.csv",
    cap:  int = 9,
    out:  str = "analysis_out/per_n_results.pdf",
):
    csv_path = Path(csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    # buckets[metric][prefix][n_bucket] -> list of values
    buckets = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    page_counts = defaultdict(int)

    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            n = _parse_n(row["page"])
            if n is None:
                continue
            b = min(n, cap)
            page_counts[b] += 1
            for metric in METRIC_LABELS:
                for prefix, _, _ in MODELS:
                    key = f"{prefix}_{metric}"
                    if key in row and row[key] != "":
                        buckets[metric][prefix][b].append(float(row[key]))

    ns = sorted(page_counts.keys())
    xlabels = [f"{n}" if n < cap else f"$\\geq${n}" for n in ns]

    fig, ax = plt.subplots(figsize=(6.0, 3.4))

    for prefix, display, color in MODELS:
        for metric, metric_label in METRIC_LABELS.items():
            xs, ys = [], []
            for n in ns:
                vals = buckets[metric][prefix].get(n, [])
                if not vals:
                    continue
                xs.append(n)
                ys.append(sum(vals) / len(vals))
            ax.plot(xs, ys, linewidth=1.8, markersize=5.5,
                    markeredgecolor="white", markeredgewidth=0.6,
                    color=color, label=f"{display}, {metric_label}",
                    **METRIC_STYLE[metric])

    ax.set_xlabel("$N$ staves per page", labelpad=10)
    ax.set_ylabel("Error rate (\\%)")
    ax.set_xticks(ns)
    ax.set_xticklabels(xlabels)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#AAAAAA")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="best", fontsize=8, frameon=False, ncol=2)

    # page-count annotations sit between the tick labels and the x-axis title
    for n in ns:
        ax.annotate(f"{page_counts[n]}p", xy=(n, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -18), textcoords="offset points",
                    ha="center", va="top", fontsize=7, color="#666666")

    fig.tight_layout()
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
