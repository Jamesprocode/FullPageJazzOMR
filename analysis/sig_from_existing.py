"""
Run paired Wilcoxon (one-sided, H1: replay better than baseline) + Holm
correction on the per-page metrics already saved in
`analysis_out/per_sample_metrics.csv`.

This is a preliminary check using the previously generated `replay100`
predictions (the dropped `pagecrop_fold0_best.ckpt`) — the full sweep on
the cluster will replace this once it finishes.
"""

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).parent))
from jazzmus.dataset.eval_functions import compute_poliphony_metrics

CSV       = Path("analysis_out/per_sample_metrics.csv")
PREDS_DIR = Path("analysis_out/preds")
OUT_MD    = Path("analysis_out/significance_existing.md")

# (display_name, baseline_col, model_col)
PAIRS = [
    ("cer",       "base_cer",       "replay_cer"),
    ("ser",       "base_ser",       "replay_ser"),
    ("ler",       "base_ler",       "replay_ler"),
    ("kern_cer",  "base_kern_cer",  "replay_kern_cer"),
    ("kern_ser",  "base_kern_ser",  "replay_kern_ser"),
    ("kern_ler",  "base_kern_ler",  "replay_kern_ler"),
    ("root_ser",  "base_root_ser",  "replay_root_ser"),
    ("chord_ser", "base_chord_ser", "replay_chord_ser"),
]


def add_overall_metrics(df):
    """Compute per-page overall CER/SER/LER from the raw .kern prediction files
    and merge into df (keyed on `page`)."""
    overall = []
    for page in df["page"]:
        gt_path   = PREDS_DIR / f"{page}__gt.kern"
        base_path = PREDS_DIR / f"{page}__baseline.kern"
        rep_path  = PREDS_DIR / f"{page}__replay100.kern"
        if not (gt_path.exists() and base_path.exists() and rep_path.exists()):
            print(f"  WARN: missing prediction file for {page}")
            overall.append({"page": page})
            continue
        gt   = gt_path.read_text()
        base = base_path.read_text()
        rep  = rep_path.read_text()
        b_cer, b_ser, b_ler = compute_poliphony_metrics([base], [gt])
        r_cer, r_ser, r_ler = compute_poliphony_metrics([rep],  [gt])
        overall.append({
            "page":       page,
            "base_cer":   b_cer, "base_ser":   b_ser, "base_ler":   b_ler,
            "replay_cer": r_cer, "replay_ser": r_ser, "replay_ler": r_ler,
        })
    return df.merge(pd.DataFrame(overall), on="page", how="left")


def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def main():
    if not CSV.exists():
        sys.exit(f"missing {CSV}")
    df = pd.read_csv(CSV)
    print(f"computing overall CER/SER/LER from {PREDS_DIR}/ ...")
    df = add_overall_metrics(df)
    n = len(df)

    rows = []
    for name, b, m in PAIRS:
        if b not in df.columns or m not in df.columns:
            print(f"skip {name}: missing columns")
            continue
        diffs = df[b] - df[m]                # >0 = model better
        wins   = int((diffs > 0).sum())
        ties   = int((diffs == 0).sum())
        losses = int((diffs < 0).sum())
        try:
            _, p = wilcoxon(diffs, alternative="greater", zero_method="wilcox")
        except ValueError:
            p = 1.0
        rows.append({
            "metric":    name,
            "n":         n,
            "base_med":  float(df[b].median()),
            "mod_med":   float(df[m].median()),
            "med_diff":  float(diffs.median()),
            "mean_diff": float(diffs.mean()),
            "wins":      wins,
            "ties":      ties,
            "losses":    losses,
            "p_raw":     float(p),
        })

    p_raw = [r["p_raw"] for r in rows]
    _, p_adj, _, _ = multipletests(p_raw, method="holm")
    for r, pa in zip(rows, p_adj):
        r["p_adj"] = float(pa)
        r["sig"]   = stars(pa)

    print(f"Wilcoxon paired (one-sided, H1: replay100 better than baseline)")
    print(f"  N = {n} pages, {len(rows)} tests, holm-corrected\n")
    print(f"{'metric':<10} {'base':>7} {'mod':>7} {'Δmed':>8} "
          f"{'W/T/L':>10} {'p_raw':>9} {'p_adj':>9}  sig")
    print("-" * 78)
    for r in rows:
        print(f"{r['metric']:<10} "
              f"{r['base_med']:>6.2f}% {r['mod_med']:>6.2f}% {r['med_diff']:>+7.2f}% "
              f"{r['wins']:>3}/{r['ties']:>2}/{r['losses']:>3} "
              f"{r['p_raw']:>9.4f} {r['p_adj']:>9.4f}  {r['sig']}")

    md = [
        "# Paired significance — baseline vs replay100 (existing predictions)", "",
        f"- N pages: **{n}**",
        f"- Test: Wilcoxon signed-rank, **one-sided** (H1: replay100 better than baseline)",
        f"- Correction: **holm** across {len(rows)} tests",
        f"- Significance: `*` p<0.05, `**` p<0.01, `***` p<0.001 (corrected)",
        "- Source: `analysis_out/per_sample_metrics.csv` (previous replay100 ckpt — preliminary)",
        "",
        "| metric | baseline median | model median | median Δ | mean Δ | W/T/L | p_raw | p_adj | sig |",
        "|--------|-----------------|--------------|----------|--------|-------|-------|-------|-----|",
    ]
    for r in rows:
        md.append(
            f"| {r['metric']} | {r['base_med']:.2f}% | {r['mod_med']:.2f}% | "
            f"{r['med_diff']:+.2f}% | {r['mean_diff']:+.2f}% | "
            f"{r['wins']}/{r['ties']}/{r['losses']} | "
            f"{r['p_raw']:.4f} | {r['p_adj']:.4f} | {r['sig']} |"
        )
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
