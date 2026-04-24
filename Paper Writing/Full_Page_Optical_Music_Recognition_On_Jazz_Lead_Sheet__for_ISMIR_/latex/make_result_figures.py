"""Generate bar chart figures for the Results section.

Outputs:
- overall_results.png   : grouped bar chart of CER / SER / LER for baseline vs masking r=1.0
- spine_results.png     : grouped bar chart of Kern CER/SER/LER, Root SER, Chord SER
                          across baseline and masking r in {0, 0.25, 0.50, 1.00}
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parent

# ── Table 1: Overall ───────────────────────────────────────────────────────
overall_methods = [
    "Baseline",
    "Masking $r{=}0$",
    "Masking $r{=}0.25$",
    "Masking $r{=}0.50$",
    "Masking $r{=}1.00$",
    "Stacking",
]
overall_metrics = ["CER", "SER", "LER"]
overall_values = np.array([
    [13.55, 12.78, 32.35],   # Baseline
    [17.61, 16.80, 31.41],   # Masking r=0
    [17.48, 16.73, 29.13],   # Masking r=0.25
    [14.11, 13.59, 26.25],   # Masking r=0.50
    [13.32, 12.50, 23.15],   # Masking r=1.00
    [24.30, 22.90, 44.55],   # Stacking
])

# ── Table 2: Spine-specific ────────────────────────────────────────────────
spine_methods = [
    "Baseline",
    "Masking $r{=}0$",
    "Masking $r{=}0.25$",
    "Masking $r{=}0.50$",
    "Masking $r{=}1.00$",
    "Stacking $r{=}0.25$",
]
spine_metrics = [
    r"$\mathrm{CER}_{\mathrm{Kern}}$",
    r"$\mathrm{SER}_{\mathrm{Kern}}$",
    r"$\mathrm{LER}_{\mathrm{Kern}}$",
    r"$\mathrm{SER}_{\mathrm{Root}}$",
    r"$\mathrm{SER}_{\mathrm{Chord}}$",
]
spine_values = np.array([
    [11.96, 14.40, 26.66, 29.17, 47.82],   # Baseline
    [17.20, 18.39, 26.62, 26.39, 39.16],   # Masking r=0
    [15.08, 16.12, 22.31, 26.31, 37.59],   # Masking r=0.25
    [14.10, 15.68, 22.18, 23.36, 35.70],   # Masking r=0.50
    [11.63, 13.02, 18.67, 21.25, 30.67],   # Masking r=1.00
    [21.56, 21.48, 35.78, 36.06, 48.39],   # Stacking r=0.25
])


def grouped_bars(values, row_labels, col_labels, title, out_path,
                 figsize=(7, 3.5), bar_width=None, ylabel="Error rate (\\%)"):
    """Draw a grouped bar chart where each x-axis group is a training
    configuration (row_labels) and the bars within a group are metrics
    (col_labels).
    """
    n_methods = values.shape[0]   # training configs → x-axis groups
    n_metrics = values.shape[1]   # metrics → bars within a group

    if bar_width is None:
        bar_width = 0.8 / n_metrics

    x = np.arange(n_methods)
    fig, ax = plt.subplots(figsize=figsize)

    # Greyscale-safe palette: monotonic luminance ramp (dark → light blue).
    palette = [
        "#1F3A5F",  # very dark blue
        "#3E6FA0",  # dark blue
        "#6FA8C9",  # mid blue
        "#A8C8DD",  # light blue
        "#D4E2EC",  # very light blue
        "#EEF2F6",  # near white
    ]
    colors = [palette[j % len(palette)] for j in range(n_metrics)]

    for j in range(n_metrics):
        offset = (j - (n_metrics - 1) / 2) * bar_width
        col_vals = values[:, j]
        bars = ax.bar(x + offset, col_vals, bar_width,
                      label=col_labels[j], color=colors[j],
                      edgecolor="#222222", linewidth=0.5)
        for b, v in zip(bars, col_vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7, color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels(row_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, values.max() * 1.20)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#AAAAAA")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8, frameon=False,
              ncol=min(n_metrics, 3))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ── Masking stage trajectories: best val SER per stage, by replay ratio ────
stage_xs = np.arange(1, 10)
stage_series = {
    r"$r{=}0$":    [12.90, 15.70, 15.46, 21.68, 26.77, 26.44, 25.79, 25.83, 18.50],
    r"$r{=}0.25$": [11.58, 15.36, 16.87, 16.67, 14.33, 18.36, 13.96, 14.15, 16.05],
    r"$r{=}0.50$": [11.09, 12.45, 14.18, 15.55, 15.11, 16.23, 15.93, 13.24, 12.94],
    r"$r{=}1.00$": [11.08, 13.17, 14.94, 15.50, 15.52, 16.30, 16.62, 13.62, 12.79],
}


def stage_trajectories(out_path, figsize=(6.5, 3.4)):
    # Blue luminance ramp (dark → mid). Each line also gets a unique
    # linestyle + marker so the plot stays readable in B&W.
    styles = [
        {"color": "#0F2A4A", "linestyle": "-",  "marker": "o"},
        {"color": "#2F5A85", "linestyle": "--", "marker": "s"},
        {"color": "#4F7FA8", "linestyle": "-.", "marker": "^"},
        {"color": "#7FA3C2", "linestyle": ":",  "marker": "D"},
    ]

    fig, ax = plt.subplots(figsize=figsize)
    for (label, ys), style in zip(stage_series.items(), styles):
        ax.plot(stage_xs, ys, linewidth=1.8, markersize=5.5,
                markeredgecolor="white", markeredgewidth=0.6,
                label=label, **style)

    ax.set_xticks(stage_xs)
    ax.set_xlabel("Curriculum stage")
    ax.set_ylabel("Best validation SER (\\%)")
    ax.set_ylim(8, max(max(v) for v in stage_series.values()) * 1.12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#AAAAAA")
    ax.set_axisbelow(True)
    ax.legend(title="Replay ratio", loc="upper left",
              fontsize=9, title_fontsize=9, frameon=False, ncol=2,
              handlelength=2.4, columnspacing=1.2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ── Per-page-size breakdown: baseline vs full-page as a function of staves ──
# Fill in None entries once the per-n breakdown is available.
per_n_data = {
    # n : (page_count, baseline_kern_cer, fullpage_kern_cer,
    #                   baseline_chord_ser, fullpage_chord_ser)
    6: (9,  6.42, 15.91, 45.98, 39.76),
    7: (6,  None, None,  None,  None),
    8: (11, None, None,  42.85, 18.04),
    9: (6,  None, None,  53.41, 21.90),   # n>=9 bucket
}


def per_n_plot(out_path, figsize=(7.0, 3.2)):
    ns = sorted(per_n_data.keys())
    counts    = [per_n_data[n][0] for n in ns]
    base_kern = [per_n_data[n][1] for n in ns]
    full_kern = [per_n_data[n][2] for n in ns]
    base_ch   = [per_n_data[n][3] for n in ns]
    full_ch   = [per_n_data[n][4] for n in ns]

    fig, (ax_k, ax_c) = plt.subplots(1, 2, figsize=figsize, sharex=True)
    xlabels = [f"{n}" if n < max(ns) else f"$\\geq${n}" for n in ns]

    def _plot(ax, base, full, title, ylabel):
        base_p = [(n, v) for n, v in zip(ns, base) if v is not None]
        full_p = [(n, v) for n, v in zip(ns, full) if v is not None]
        if base_p:
            ax.plot([p[0] for p in base_p], [p[1] for p in base_p],
                    color="#6FA8C9", linestyle="--", marker="o",
                    markeredgecolor="white", markeredgewidth=0.6,
                    linewidth=1.8, markersize=5.5, label="Baseline")
        if full_p:
            ax.plot([p[0] for p in full_p], [p[1] for p in full_p],
                    color="#1F3A5F", linestyle="-",  marker="s",
                    markeredgecolor="white", markeredgewidth=0.6,
                    linewidth=1.8, markersize=5.5, label="Full-page")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Staves per page $n$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ns)
        ax.set_xticklabels(xlabels)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#AAAAAA")
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(loc="best", fontsize=8, frameon=False)

    _plot(ax_k, base_kern, full_kern,
          r"Melodic error ($\mathrm{CER}_{\mathrm{Kern}}$)",
          "Error rate (\\%)")
    _plot(ax_c, base_ch, full_ch,
          r"Chord error ($\mathrm{SER}_{\mathrm{Chord}}$)",
          "Error rate (\\%)")

    # Page-count annotations under the x-axis.
    for ax in (ax_k, ax_c):
        ymin, ymax = ax.get_ylim()
        for n, c in zip(ns, counts):
            ax.text(n, ymin - (ymax - ymin) * 0.12,
                    f"{c}p", ha="center", va="top",
                    fontsize=7, color="#666666")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    # Overall (Table 1): methods as groups, metrics as bars within a group
    grouped_bars(
        overall_values, overall_methods, overall_metrics,
        title="Overall full-page recognition error rates",
        out_path=OUT_DIR / "overall_results.pdf",
        figsize=(9.0, 4.0),
    )

    # Spine-specific (Table 2): each x-axis group = one training config;
    # bars within a group = the 5 metrics, so all metrics for one
    # configuration can be compared side-by-side.
    grouped_bars(
        spine_values, spine_methods, spine_metrics,
        title="Spine-specific error rates by configuration",
        out_path=OUT_DIR / "spine_results_bar.pdf",
        figsize=(9.0, 4.0),
    )

    stage_trajectories(out_path=OUT_DIR / "masking_stage_validation.pdf")

    per_n_plot(out_path=OUT_DIR / "per_n_results.pdf")
