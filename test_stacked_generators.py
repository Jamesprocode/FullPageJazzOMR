"""
Sanity check for StackedPageDataset.

For each curriculum stage N=1..max_n:
  - Generates a random stacked sample
  - Checks <linebreak> count == N-1 in GT
  - Checks <linebreak> is in vocab
  - Plots the stacked image
  - Saves all stages to a single PDF

Run from FullPageJazzOMR/ project root:
    python test_stacked_generators.py
    python test_stacked_generators.py \\
        --system_data_path /path/to/jazzmus_systems \\
        --real_data_path   /path/to/jazzmus_pagecrop \\
        --fold 0 --max_n 9
"""

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from datasets.stacked_page_dataset import StackedPageDataset
from jazzmus.dataset.smt_dataset_utils import check_and_retrieveVocabulary

SYSTEM_DATA_PATH = (
    "/home/hice1/jwang3180/scratch/jazzmus/ISMIR-Jazzmus/data/jazzmus_systems"
)
REAL_DATA_PATH = (
    "/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmuss_Data/jazzmus_pagecrop"
)


def run_checks(system_data_path: str, real_data_path: str, fold: int, max_n: int):
    print(f"\n── StackedPageDataset sanity check ───────────────────────────────")
    print(f"   system_data_path : {system_data_path}")
    print(f"   real_data_path   : {real_data_path}")
    print(f"   fold             : {fold}   max_n : {max_n}")

    train_set = StackedPageDataset(
        system_data_path=system_data_path,
        split="train",
        fold=fold,
        system_height=256,
        num_cl_stages=max_n,
        final_stage=max_n,
        real_mix_prob=0.5,
        real_data_path=real_data_path,
        width_tolerance=0.15,
        augment=False,
        dataset_length=100,
    )
    val_set = StackedPageDataset(
        system_data_path=system_data_path,
        split="val",
        fold=fold,
        system_height=256,
        num_cl_stages=max_n,
        final_stage=max_n,
        real_mix_prob=0.0,
        real_data_path=None,
        width_tolerance=0.15,
        augment=False,
        dataset_length=20,
    )

    # Build vocab
    w2i, i2w = check_and_retrieveVocabulary(
        [train_set.get_gt(), val_set.get_gt()],
        "vocab",
        "vocab_stacked_check",
    )
    train_set.set_dictionaries(w2i, i2w)
    val_set.set_dictionaries(w2i, i2w)

    # ── check 1: <linebreak> in vocab ────────────────────────────────────────
    if "<linebreak>" in w2i:
        print(f"\n✓ <linebreak> in vocab  (id={w2i['<linebreak>']})")
    else:
        print("\n✗ <linebreak> NOT in vocab — stacking GT is broken!")
    print(f"  Vocab size: {train_set.vocab_size()} tokens")
    print(f"  Train systems: {len(train_set.system_x)}  "
          f"Val systems: {len(val_set.system_x)}")
    if train_set._real_paths:
        print(f"  Real mix pool: {len(train_set._real_paths)} pages")

    # ── check 2: per-stage stacked image + GT ─────────────────────────────────
    fig, axes = plt.subplots(max_n, 1, figsize=(14, 3 * max_n))
    if max_n == 1:
        axes = [axes]

    all_passed = True
    for n in range(1, max_n + 1):
        train_set.set_stage_direct(n)
        val_set.set_stage_direct(n)

        # Draw one sample (val set: always stacks exactly n systems)
        x, _, y, label = val_set[0]
        tokens = [i2w.get(int(tid), "?") for tid in y.tolist()]

        linebreaks = tokens.count("<linebreak>")
        expected   = n - 1
        passed     = (linebreaks == expected)
        all_passed = all_passed and passed
        status     = "✓" if passed else "✗"

        h, w = x.shape[1], x.shape[2]
        print(f"\n  N={n}  [{status}]  {len(tokens)} tokens  "
              f"{linebreaks} <linebreak> (expected {expected})  "
              f"image {h}×{w}px  label={label}")
        if not passed:
            print(f"    ✗ FAIL — expected {expected} <linebreak>, got {linebreaks}")

        ax = axes[n - 1]
        ax.imshow(x.squeeze().numpy(), cmap="gray", aspect="auto")
        ax.set_title(
            f"N={n}  |  {len(tokens)} tokens  |  {linebreaks} <linebreak>  |  {h}×{w}px",
            fontsize=8,
        )
        ax.axis("off")

    plt.tight_layout()
    out = "test_stacked_generators_output.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved visualisation → {out}")

    # ── check 3: final-stage real-mix sample ─────────────────────────────────
    if train_set._real_paths:
        train_set.set_stage_direct(max_n)
        # Force a real sample by temporarily setting real_mix_prob=1.0
        orig_prob = train_set.real_mix_prob
        train_set.real_mix_prob = 1.0
        rx, _, ry, rpath = train_set[0]
        train_set.real_mix_prob = orig_prob
        rtokens = [i2w.get(int(tid), "?") for tid in ry.tolist()]
        rlb = rtokens.count("<linebreak>")
        print(f"\n  Final-stage real sample: {Path(rpath).name}  "
              f"{len(rtokens)} tokens  {rlb} <linebreak>  "
              f"image {rx.shape[1]}×{rx.shape[2]}px")

    print(f"\n{'✓ All checks passed' if all_passed else '✗ Some checks FAILED'}\n")
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system_data_path", default=SYSTEM_DATA_PATH)
    parser.add_argument("--real_data_path",   default=REAL_DATA_PATH)
    parser.add_argument("--fold",   type=int, default=0)
    parser.add_argument("--max_n",  type=int, default=9)
    args = parser.parse_args()
    run_checks(args.system_data_path, args.real_data_path, args.fold, args.max_n)
