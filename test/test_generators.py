"""
Visual sanity check for the PageCropDataset.

For each curriculum stage N=1..max_n:
  - Picks a random eligible sample
  - Displays the stacked image
  - Prints GT token count and verifies <linebreak> count == N-1
  - Verifies vocab contains <linebreak>

Run from the FullPageJazzOMR/ project root after running prepare_pagecrop.py:
    python test_generators.py
    python test_generators.py --data_path data/jazzmus_pagecrop --fold 0 --max_n 5
"""

import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display needed; saves to PDF
import matplotlib.pyplot as plt

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

from datasets.page_crop_dataset import PageCropDataset
from jazzmus.dataset.smt_dataset_utils import check_and_retrieveVocabulary


def run_checks(data_path: str = "data/jazzmus_pagecrop", fold: int = 0, max_n: int = 5):
    print(f"\n── PageCropDataset sanity check ──────────────────────────────────")
    print(f"   data_path : {data_path}")
    print(f"   fold      : {fold}")

    # Build dataset with max stages so all N values are eligible at some point
    train_set = PageCropDataset(
        data_path=data_path,
        split="train",
        fold=fold,
        fixed_img_height=None,
        num_cl_stages=max_n,
        increase_epochs=1,     # stage advances every epoch → stage = epoch + 1
        curriculum_start=1,
        dataset_length=100,
    )
    val_set = PageCropDataset(
        data_path=data_path,
        split="val",
        fold=fold,
        fixed_img_height=None,
        num_cl_stages=max_n,
        increase_epochs=1,
        curriculum_start=1,
        dataset_length=20,
    )

    # Build vocab
    w2i, i2w = check_and_retrieveVocabulary(
        [train_set.get_gt(), val_set.get_gt()],
        "vocab",
        "vocab_check",
    )
    train_set.set_dictionaries(w2i, i2w)
    val_set.set_dictionaries(w2i, i2w)

    # ── check 1: <linebreak> in vocab ────────────────────────────────────────
    if "<linebreak>" in w2i:
        print(f"\n✓ <linebreak> in vocab  (id={w2i['<linebreak>']})")
    else:
        print("\n✗ <linebreak> NOT in vocab — check GT generation!")

    print(f"  Vocab size: {train_set.vocab_size()} tokens")

    # ── check 2: per-stage image + GT ────────────────────────────────────────
    fig, axes = plt.subplots(max_n, 1, figsize=(14, 3 * max_n))
    if max_n == 1:
        axes = [axes]

    # Build a lookup: N → list of sample indices with exactly N systems
    exact_n = {}
    for i, (_, _, sn) in enumerate(train_set.samples):
        exact_n.setdefault(sn, []).append(i)

    all_passed = True
    for n in range(1, max_n + 1):
        train_set.set_epoch(n - 1)

        # Pick a sample with exactly N systems to verify GT correctness
        if n not in exact_n:
            print(f"\n  Stage N={n}  [SKIP] no samples with exactly N={n}")
            continue
        idx = random.choice(exact_n[n])
        img_path_exact = train_set.samples[idx][0]
        tokens_exact   = train_set.gt_tokens[idx]
        x, decoder_input, y, img_path = train_set[0]  # random eligible item for display
        linebreaks = tokens_exact.count("<linebreak>")
        expected   = n - 1
        passed     = (linebreaks == expected)
        all_passed = all_passed and passed

        h, w = x.shape[1], x.shape[2]
        status = "✓" if passed else "✗"
        print(f"\n  N={n}  [{status}]")
        print(f"    Check sample : {Path(img_path_exact).name}  |  {len(tokens_exact)} tokens  |  {linebreaks} <linebreak> (expected {expected})")
        print(f"    Display img  : {Path(img_path).name}  ({h}×{w} px)")
        if not passed:
            print(f"    ✗ FAIL: expected {expected} <linebreak>, got {linebreaks}")

        # Plot image
        ax = axes[n - 1]
        img_np = x.squeeze().numpy()
        ax.imshow(img_np, cmap="gray", aspect="auto")
        ax.set_title(
            f"N={n}  |  {len(tokens_exact)} tokens  |  {linebreaks} <linebreak>  "
            f"|  {Path(img_path).name}",
            fontsize=8,
        )
        ax.axis("off")

    plt.tight_layout()
    out_path = "test_generators_output.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved visualisation → {out_path}")

    # ── check 3: split counts ─────────────────────────────────────────────────
    print(f"\n── Split statistics ───────────────────────────────────────────────")
    for split in ["train", "val", "test"]:
        try:
            ds = PageCropDataset(
                data_path=data_path, split=split, fold=fold,
                num_cl_stages=max_n, increase_epochs=1, curriculum_start=1,
                dataset_length=10,
            )
            by_n = {}
            for _, _, n in ds.samples:
                by_n[n] = by_n.get(n, 0) + 1
            print(f"  {split:5s}: {len(ds.samples):4d} samples  "
                  f"by N: {dict(sorted(by_n.items()))}")
        except FileNotFoundError as e:
            print(f"  {split:5s}: skipped ({e})")

    print(f"\n{'✓ All checks passed' if all_passed else '✗ Some checks FAILED'}")
    return all_passed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/jazzmus_pagecrop")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max_n", type=int, default=5)
    args = parser.parse_args()
    run_checks(args.data_path, args.fold, args.max_n)
