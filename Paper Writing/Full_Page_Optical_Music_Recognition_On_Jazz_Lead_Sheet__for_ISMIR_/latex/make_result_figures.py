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
overall_methods = ["Baseline", "Masking, $r{=}1.00$"]
overall_metrics = ["CER", "SER", "LER"]
overall_values = np.array([
    [13.55, 12.78, 32.35],   # Baseline
    [13.32, 12.50, 23.15],   # Masking r=1.00
])

# ── Table 2: Spine-specific ────────────────────────────────────────────────
spine_methods = [
    "Baseline",
    "Masking $r{=}0$",
    "Masking $r{=}0.25$",
    "Masking $r{=}0.50$",
    "Masking $r{=}1.00$",
]
spine_metrics = ["Kern CER", "Kern SER", "Kern LER", "Root SER", "Chord SER"]
spine_values = np.array([
    [11.96, 14.40, 26.66, 29.17, 47.82],   # Baseline
    [17.20, 18.39, 26.62, 26.39, 39.16],   # r=0
    [15.08, 16.12, 22.31, 26.31, 37.59],   # r=0.25
    [14.10, 15.68, 22.18, 23.36, 35.70],   # r=0.50
    [11.63, 13.02, 18.67, 21.25, 30.67],   # r=1.00
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

    # Blue-dominant muted palette — all cool tones, no warm accents
    palette = [
        "#2F4A6B",  # deep navy         (Kern CER)
        "#5B7CA0",  # steel blue        (Kern SER)
        "#8FA8BF",  # soft periwinkle   (Kern LER)
        "#6B8E8A",  # dusty teal        (Root SER)
        "#7F6B94",  # dusty slate purple (Chord SER) — distinct from navy
        "#A8B8C4",  # cool pale blue-grey
    ]
    colors = [palette[j % len(palette)] for j in range(n_metrics)]

    for j in range(n_metrics):
        offset = (j - (n_metrics - 1) / 2) * bar_width
        col_vals = values[:, j]
        bars = ax.bar(x + offset, col_vals, bar_width,
                      label=col_labels[j], color=colors[j],
                      edgecolor="#555555", linewidth=0.4)
        for b, v in zip(bars, col_vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7, color="#333333")

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
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    # Overall (Table 1): methods as groups, metrics as bars within a group
    grouped_bars(
        overall_values, overall_methods, overall_metrics,
        title="Overall full-page recognition error rates",
        out_path=OUT_DIR / "overall_results.png",
        figsize=(5.5, 3.2),
    )

    # Spine-specific (Table 2): each x-axis group = one training config;
    # bars within a group = the 5 metrics, so all metrics for one
    # configuration can be compared side-by-side.
    grouped_bars(
        spine_values, spine_methods, spine_metrics,
        title="Spine-specific error rates by configuration",
        out_path=OUT_DIR / "spine_results_bar.png",
        figsize=(9.0, 4.0),
    )
