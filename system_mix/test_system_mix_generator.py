"""
Sanity-check for the curriculum stacking generator.

Outputs:
  debug_stacking/{n}sys/sample.png      — stacked image
  debug_stacking/{n}sys/gt.txt          — kern ground truth
  debug_stacking/{n}sys/sample_gt.png   — image + GT rendered together
  debug_stacking/outliers/              — systems with pool < 10 at ±15 %
  debug_stacking/pool_analysis.png      — width histogram + pool size plot

Run from the project root:
    python test_curriculum_generator.py \
        --system_data_path data/jazzmus_systems/splits \
        --fold 0
"""

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from jazzmus.curriculum.dataset import stack_systems
from jazzmus.dataset.smt_dataset import load_set as load_system_set
from jazzmus.dataset.tokenizer import process_text, untokenize

OUT_DIR        = Path("debug_stacking")
TOKENIZER_TYPE = "medium"
SYSTEM_HEIGHT  = 128
OUTLIER_POOL_THRESHOLD = 10   # systems with fewer compatible partners than this are "outliers"


# ── helpers ────────────────────────────────────────────────────────────────────

def check_gt(combined_gt, n):
    assert combined_gt[0] == "<bos>",        f"GT does not start with <bos>: {combined_gt[:3]}"
    assert combined_gt[-1] == "<eos>",       f"GT does not end with <eos>:   {combined_gt[-3:]}"
    assert "<bos>" not in combined_gt[1:-1], f"<bos> found inside GT for n={n}"
    assert "<eos>" not in combined_gt[1:-1], f"<eos> found inside GT for n={n}"
    print(f"    GT length: {len(combined_gt)} tokens  ✓")


def tokens_to_kern(tokens):
    """Convert token list (without <bos>/<eos>) back to readable kern string."""
    return untokenize(tokens)


def save_image_with_gt(img, kern_text, out_path, title=""):
    """Save a figure with the stacked image on top and kern GT as text below."""
    # Wrap long kern text for display
    lines = kern_text.split("\n")
    display_lines = lines[:40]          # cap at 40 lines to keep figure readable
    if len(lines) > 40:
        display_lines.append(f"... ({len(lines) - 40} more lines)")
    display_text = "\n".join(display_lines)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(14, 6 + len(display_lines) * 0.18),
        gridspec_kw={"height_ratios": [3, max(1, len(display_lines) * 0.5)]},
    )

    axes[0].imshow(img, cmap="gray", aspect="auto")
    axes[0].set_title(title, fontsize=10, loc="left")
    axes[0].axis("off")

    axes[1].text(
        0.01, 0.99, display_text,
        transform=axes[1].transAxes,
        fontsize=7, verticalalignment="top", fontfamily="monospace",
        wrap=True,
    )
    axes[1].axis("off")
    axes[1].set_title("Ground truth (kern)", fontsize=9, loc="left")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=130)
    plt.close()


# ── pool analysis ──────────────────────────────────────────────────────────────

def pool_analysis(widths, raw_x, raw_paths, tolerances=(0.10, 0.15, 0.20, 0.25)):
    widths_arr = np.array(widths)
    print("\n── Width pool analysis ──────────────────────────────────────")
    print(f"  Total systems : {len(widths_arr)}")
    print(f"  Width  min / mean / median / max : "
          f"{widths_arr.min()} / {widths_arr.mean():.0f} / "
          f"{np.median(widths_arr):.0f} / {widths_arr.max()} px\n")

    print(f"  {'Tolerance':>10}  {'Min pool':>9}  {'Mean pool':>10}  "
          f"{'Median':>8}  {'Max':>6}  {'% pool<5':>9}")
    print("  " + "-" * 62)

    results = {}
    for tol in tolerances:
        pool_sizes = np.array([
            int(np.sum(np.abs(widths_arr - w) / w <= tol))
            for w in widths_arr
        ])
        pct_small = 100 * np.mean(pool_sizes < 5)
        print(f"  {tol*100:>9.0f}%  {pool_sizes.min():>9}  {pool_sizes.mean():>10.1f}  "
              f"{np.median(pool_sizes):>8.0f}  {pool_sizes.max():>6}  {pct_small:>8.1f}%")
        results[tol] = pool_sizes

    # ── save outlier images ────────────────────────────────────────────────────
    outlier_tol   = 0.10
    outlier_pools = results[outlier_tol]
    outlier_idx   = np.where(outlier_pools < OUTLIER_POOL_THRESHOLD)[0]

    outlier_dir = OUT_DIR / "outliers"
    outlier_dir.mkdir(exist_ok=True)
    print(f"\n  Outliers (pool < {OUTLIER_POOL_THRESHOLD} at ±{int(outlier_tol*100)}%): "
          f"{len(outlier_idx)} systems")

    for idx in outlier_idx:
        img   = raw_x[idx]
        path  = Path(raw_paths[idx])
        w     = widths_arr[idx]
        pool  = outlier_pools[idx]
        out   = outlier_dir / f"{path.stem}_w{w}_pool{pool}.png"
        cv2.imwrite(str(out), img)
        print(f"    {path.name}  width={w}  pool={pool}  → {out.name}")

    # ── plots ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].hist(widths_arr, bins=40, color="steelblue", edgecolor="white")
    axes[0].axvline(widths_arr.mean(), color="red", linestyle="--",
                    linewidth=1.2, label=f"mean={widths_arr.mean():.0f}")
    axes[0].set_title("Width distribution of system images (px)", fontsize=11)
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=8)

    for tol, pool_sizes in results.items():
        axes[1].hist(pool_sizes, bins=30, alpha=0.6, label=f"±{int(tol*100)}%")
    axes[1].axvline(5, color="red", linestyle="--", linewidth=1.2, label="min needed (n=5)")
    axes[1].set_title("Compatible pool size per system", fontsize=11)
    axes[1].set_xlabel("Pool size")
    axes[1].set_ylabel("Count")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plot_path = OUT_DIR / "pool_analysis.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n  Plot saved → {plot_path}")
    plt.show()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system_data_path", default="data/jazzmus_systems/splits")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    # ── load system images ─────────────────────────────────────────────────────
    print("\n── Loading system images ────────────────────────────────────")
    raw_x, raw_y, raw_paths = load_system_set(
        args.system_data_path,
        fold=args.fold,
        split="train",
        fixed_img_height=SYSTEM_HEIGHT,
        max_fix_img_width=None,
        include_synthetic=False,
    )
    print(f"  {len(raw_x)} systems loaded")
    print(f"  Height range : {min(i.shape[0] for i in raw_x)} – {max(i.shape[0] for i in raw_x)} px")
    print(f"  Width range  : {min(i.shape[1] for i in raw_x)} – {max(i.shape[1] for i in raw_x)} px")

    system_y = [
        ["<bos>"] + process_text(lines=lines, tokenizer_type=TOKENIZER_TYPE) + ["<eos>"]
        for lines in raw_y
    ]
    print(f"  GT length range: {min(len(g) for g in system_y)} – {max(len(g) for g in system_y)} tokens")

    # ── pool analysis + outliers ───────────────────────────────────────────────
    widths = [img.shape[1] for img in raw_x]
    pool_analysis(widths, raw_x, raw_paths)

    # ── stacking samples with GT ───────────────────────────────────────────────
    print("\n── Stacking samples ─────────────────────────────────────────")
    path_to_idx = {p: i for i, p in enumerate(raw_paths)}

    for n in [2, 3, 4, 5, 6]:
        folder = OUT_DIR / f"{n}sys"
        folder.mkdir(parents=True, exist_ok=True)

        img, combined_gt, sampled = stack_systems(
            raw_x, system_y, n, SYSTEM_HEIGHT,
            paths=raw_paths, width_tolerance=0.15,
        )

        # ── save raw image ─────────────────────────────────────────────────────
        cv2.imwrite(str(folder / "sample.png"), img)

        # ── save kern GT as text ───────────────────────────────────────────────
        kern_text = tokens_to_kern(combined_gt[1:-1])   # strip bos/eos
        (folder / "gt.txt").write_text(kern_text)

        # ── save image + GT as a single figure ────────────────────────────────
        sampled_widths = [raw_x[path_to_idx[p]].shape[1] for p in sampled]
        slot_info = "  |  ".join(
            f"slot{i+1}: {Path(p).stem} (w={w})"
            for i, (p, w) in enumerate(zip(sampled, sampled_widths))
        )
        save_image_with_gt(
            img, kern_text,
            out_path=folder / "sample_gt.png",
            title=f"{n} systems stacked  —  {slot_info}",
        )

        print(f"\n  {n} systems → shape {img.shape}  widths: {sampled_widths}")
        for slot, p in enumerate(sampled):
            print(f"    slot {slot+1} : {p}")
        check_gt(combined_gt, n)
        print(f"    → {folder}/sample.png  |  gt.txt  |  sample_gt.png")

    # ── stage gating sanity check ──────────────────────────────────────────────
    print("\n── Stage gating ─────────────────────────────────────────────")
    increase_steps, curriculum_start, num_cl_stages = 6000, 2, 4
    max_cl_steps = increase_steps * num_cl_stages
    for step in [0, 3000, 6000, 12000, 18000, 24000]:
        if step < max_cl_steps:
            stage = (step // increase_steps) + curriculum_start
            n     = random.randint(1, stage)
            phase = f"CL stage {stage}"
        else:
            n     = random.randint(curriculum_start, curriculum_start + num_cl_stages - 1)
            phase = "fine-tune"
        print(f"  step={step:6d} → {phase:12s}  n={n}")

    print(f"\n✓ Done — see {OUT_DIR}/\n")


if __name__ == "__main__":
    main()
