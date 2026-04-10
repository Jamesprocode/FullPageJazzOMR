"""
Compare HuggingFace-generated data against local-generated data.

Checks that GT files and image dimensions match between two output directories
(one from local scripts, one from HF scripts).

Run from the FullPageJazzOMR/ project root:
    python data_prep/compare_hf_vs_local.py \\
        --local   data/jazzmus_pagecrop \\
        --hf      data/jazzmus_pagecrop_hf \\
        --split   train --fold 0

    python data_prep/compare_hf_vs_local.py \\
        --local   data/jazzmus_synthetic \\
        --hf      data/jazzmus_synthetic_hf \\
        --split   train --fold 0
"""

import argparse
from pathlib import Path

import cv2


def load_split(data_dir: Path, split: str, fold: int):
    split_file = data_dir / "splits" / f"{split}_{fold}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    entries = []
    base = data_dir.parent
    with open(split_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            img_p, gt_p, n = parts[0], parts[1], int(parts[2])
            img_path = Path(img_p) if Path(img_p).exists() else base / img_p
            gt_path  = Path(gt_p)  if Path(gt_p).exists()  else base / gt_p
            entries.append((img_path, gt_path, n))
    return entries


def compare(local_dir, hf_dir, split, fold):
    print(f"\nComparing {split}_{fold}:")
    print(f"  local : {local_dir}")
    print(f"  hf    : {hf_dir}")

    local = load_split(Path(local_dir), split, fold)
    hf    = load_split(Path(hf_dir),    split, fold)

    print(f"  local entries : {len(local)}")
    print(f"  hf    entries : {len(hf)}")

    if len(local) != len(hf):
        print(f"  WARNING: entry count mismatch ({len(local)} vs {len(hf)})")

    n_checked = 0
    gt_mismatches = []
    img_mismatches = []

    for i, ((img_l, gt_l, n_l), (img_h, gt_h, n_h)) in enumerate(zip(local, hf)):
        if n_l != n_h:
            gt_mismatches.append((i, f"n mismatch: local={n_l} hf={n_h}"))
            continue

        # Compare GT text
        gt_local = gt_l.read_text() if gt_l.exists() else None
        gt_hf    = gt_h.read_text() if gt_h.exists() else None
        if gt_local != gt_hf:
            gt_mismatches.append((i, f"GT text differs  ({gt_l.name} vs {gt_h.name})"))

        # Compare image dimensions
        img_local = cv2.imread(str(img_l), cv2.IMREAD_GRAYSCALE)
        img_hf    = cv2.imread(str(img_h), cv2.IMREAD_GRAYSCALE)
        if img_local is None or img_hf is None:
            img_mismatches.append((i, "could not read image"))
        elif img_local.shape != img_hf.shape:
            img_mismatches.append((i, f"shape mismatch: {img_local.shape} vs {img_hf.shape}"))

        n_checked += 1

    print(f"\n  Checked {n_checked} entries")
    if gt_mismatches:
        print(f"  GT mismatches: {len(gt_mismatches)}")
        for idx, msg in gt_mismatches[:10]:
            print(f"    [{idx}] {msg}")
        if len(gt_mismatches) > 10:
            print(f"    ... and {len(gt_mismatches)-10} more")
    else:
        print(f"  GT: all match ✓")

    if img_mismatches:
        print(f"  Image mismatches: {len(img_mismatches)}")
        for idx, msg in img_mismatches[:10]:
            print(f"    [{idx}] {msg}")
        if len(img_mismatches) > 10:
            print(f"    ... and {len(img_mismatches)-10} more")
    else:
        print(f"  Images: all dimensions match ✓")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--hf",    type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--fold",  type=int, default=0)
    args = parser.parse_args()

    compare(args.local, args.hf, args.split, args.fold)


if __name__ == "__main__":
    main()
