"""
Offline data preparation: generate pre-computed N-system page crops.

For each full page, crops the page image vertically to include only the first
N systems (N = 1 … min(max_n, systems_on_page)), using bounding-box coordinates
from the JAZZMUS parquet dataset.

Output:
  - data/jazzmus_pagecrop/jpg/img_<X>_n<N>.jpg  — real page crop (first N systems)
  - data/jazzmus_pagecrop/gt/img_<X>_n<N>.txt   — GT tokens (one per line)

Split files are written to data/jazzmus_pagecrop/splits/{split}_{fold}.txt
with format:  <img_path> <gt_path> <N>

Bounding boxes (fromY, toY) are in the parquet's original image coordinates,
which match the jazzmus_fullpage/jpg/ image dimensions exactly.

The crop is: img[0 : toY_of_system_N + bottom_pad, :]

Run from the FullPageJazzOMR/ project root:
    python data_prep/prepare_pagecrop.py \\
        --jazzmus_fullpage ../ISMIR-Jazzmus/data/jazzmus_fullpage \\
        --jazzmus_parquet  ../JAZZMUS/data/train-00000-of-00001.parquet \\
        --out_dir          data/jazzmus_pagecrop \\
        --folds 0 \\
        --max_n 5 \\
        --bottom_pad 20
"""

import argparse
import ast
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

SPLITS = ["train", "val", "test"]


# ── GT helpers ─────────────────────────────────────────────────────────────────

SPINE_TERM = "*-"

def build_gt_from_fullpage(fullpage_gt_path: Path, n: int) -> list:
    """
    Extract the first n systems from the full-page kern GT as raw kern lines.

    The full-page kern file has !!linebreak:original separating systems and
    *-<TAB>*- at the very end.  We keep everything up to (not including) the
    N-th !!linebreak:original, then close with *-<TAB>*-.

    Returns raw kern lines (strings with newlines) — NOT tokenized.
    Tokenization happens in PageCropDataset at load time.
    """
    with open(fullpage_gt_path) as f:
        all_lines = f.readlines()

    # Split into per-system blocks on the !!linebreak:original marker
    blocks = []
    current = []
    for line in all_lines:
        if line.strip() == "!!linebreak:original":
            blocks.append(current)
            current = []
        else:
            current.append(line)
    if current:
        blocks.append(current)

    selected = blocks[:n]
    kern_lines = []
    for i, block in enumerate(selected):
        if i < n - 1:
            # Inner block: strip *- terminators, add linebreak marker
            block = [l for l in block
                     if not all(c.strip() in (SPINE_TERM, "") for c in l.split("\t"))]
            kern_lines.extend(block)
            kern_lines.append("!!linebreak:original\n")
        else:
            # Last block: ensure *- terminators are present
            has_term = any(
                all(c.strip() in (SPINE_TERM, "") for c in l.split("\t"))
                for l in block if l.strip()
            )
            kern_lines.extend(block)
            if not has_term:
                kern_lines.append(f"{SPINE_TERM}\t{SPINE_TERM}\n")

    return kern_lines


# ── cropping ───────────────────────────────────────────────────────────────────

def crop_page(img: np.ndarray, systems: list, n: int, bottom_pad: int) -> np.ndarray:
    """
    Crop a full-page image to include only the first n systems.

    Bottom edge = toY of system n (0-indexed: systems[n-1]) + bottom_pad px,
    clamped to the image height.  Full width is preserved.
    """
    crop_y = systems[n - 1]["bounding_box"]["toY"] + bottom_pad
    crop_y = min(crop_y, img.shape[0])
    return img[:crop_y, :]


# ── main generation ────────────────────────────────────────────────────────────

def generate_pagecrop(split, fold, jazzmus_fullpage, parquet_df, out_dir, max_n, bottom_pad):
    """Generate N-system crops for one split/fold combination.

    Train + val splits: N=1..min(max_n, num_systems) real page crops, so
    curriculum stage filtering works at both train and validation time.

    Test split: full page only (no cropping), for final evaluation on real pages.
    """
    split_file = jazzmus_fullpage / "splits" / f"{split}_{fold}.txt"
    if not split_file.exists():
        print(f"  Skipping {split}_{fold}: split file not found at {split_file}")
        return []

    # Paths in the fullpage split file are relative to the ISMIR-Jazzmus root
    base_dir = jazzmus_fullpage.parent.parent   # .../ISMIR-Jazzmus

    do_crop = (split != "test")   # train + val get crops; test = full page only
    print(f"\n── {split} fold {fold}  ({'crop N=1..{max_n}' if do_crop else 'full page only'}) ───")

    jpg_dir = out_dir / "jpg"
    gt_dir  = out_dir / "gt"
    jpg_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    split_entries = []
    n_generated = 0
    n_skipped   = 0

    with open(split_file) as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue

        raw_img, raw_gt = parts[0], parts[1]
        img_path = Path(raw_img)
        gt_path  = Path(raw_gt)
        if not img_path.exists():
            img_path = base_dir / raw_img
        if not gt_path.exists():
            gt_path = base_dir / raw_gt
        if not img_path.exists():
            print(f"  Warning: image not found: {raw_img}", flush=True)
            n_skipped += 1
            continue

        # img_X.jpg → page index X
        m = re.search(r"img_(\d+)\.jpg$", str(img_path))
        if not m:
            print(f"  Warning: cannot parse index from {img_path}", flush=True)
            n_skipped += 1
            continue
        page_idx = int(m.group(1))

        if page_idx >= len(parquet_df):
            print(f"  Warning: page_idx={page_idx} out of parquet range", flush=True)
            n_skipped += 1
            continue

        ann = ast.literal_eval(parquet_df["annotation"].iloc[page_idx])
        systems = ann["systems"]
        if not systems:
            n_skipped += 1
            continue

        num_sys = len(systems)

        if do_crop:
            # Train + val: N=1..min(max_n, num_systems) real page crops
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  Warning: could not read {img_path}", flush=True)
                n_skipped += 1
                continue

            for n in range(1, min(max_n, num_sys) + 1):
                kern_lines = build_gt_from_fullpage(gt_path, n)

                stem   = f"img_{page_idx}_n{n}"
                out_gt = gt_dir / f"{stem}.txt"
                out_gt.write_text("".join(kern_lines))
                rel_gt = out_gt.relative_to(out_dir.parent)

                out_jpg = jpg_dir / f"{stem}.jpg"
                if n == num_sys:
                    # Last system = full page — copy original image into output dir
                    if not out_jpg.exists():
                        shutil.copy(str(img_path), str(out_jpg))
                else:
                    cropped = crop_page(img, systems, n, bottom_pad)
                    cv2.imwrite(str(out_jpg), cropped)
                rel_jpg = out_jpg.relative_to(out_dir.parent)
                split_entries.append(f"{rel_jpg} {rel_gt} {n}\n")

                n_generated += 1
        else:
            # Test only: full page, no cropping — GT from full-page file directly
            kern_lines = build_gt_from_fullpage(gt_path, num_sys)
            stem   = f"img_{page_idx}_n{num_sys}"
            out_gt = gt_dir / f"{stem}.txt"
            out_gt.write_text("".join(kern_lines))

            rel_gt = out_gt.relative_to(out_dir.parent)
            out_jpg = jpg_dir / f"{stem}.jpg"
            if not out_jpg.exists():
                shutil.copy(str(img_path), str(out_jpg))
            rel_jpg = out_jpg.relative_to(out_dir.parent)
            split_entries.append(f"{rel_jpg} {rel_gt} {num_sys}\n")
            n_generated += 1

    print(f"  Recorded {n_generated} entries  ({n_skipped} pages skipped)")
    return split_entries


def write_split_file(entries, out_dir, split, fold):
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_path = splits_dir / f"{split}_{fold}.txt"
    split_path.write_text("".join(entries))
    print(f"  Split file → {split_path}  ({len(entries)} entries)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate N-system page crops from real full-page images."
    )
    parser.add_argument("--jazzmus_fullpage", type=Path,
                        default=Path("../ISMIR-Jazzmus/data/jazzmus_fullpage"),
                        help="Path to jazzmus_fullpage data folder")
    parser.add_argument("--jazzmus_parquet", type=Path,
                        default=Path("../JAZZMUS/data/train-00000-of-00001.parquet"),
                        help="Path to JAZZMUS parquet with system bounding boxes")
    parser.add_argument("--out_dir", type=Path,
                        default=Path("data/jazzmus_pagecrop"),
                        help="Output directory")
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--max_n", type=int, default=11,
                        help="Max systems to include per crop (default: 11 = full curriculum)")
    parser.add_argument("--bottom_pad", type=int, default=20,
                        help="Pixels of padding below last system toY (default: 20)")
    args = parser.parse_args()

    print(f"Output directory  : {args.out_dir.resolve()}")
    print(f"Full-page source  : {args.jazzmus_fullpage.resolve()}")
    print(f"Parquet source    : {args.jazzmus_parquet.resolve()}")
    print(f"Folds             : {args.folds}")
    print(f"Max N             : {args.max_n}")
    print(f"Bottom pad        : {args.bottom_pad} px")

    print("\nLoading parquet…")
    parquet_df = pd.read_parquet(args.jazzmus_parquet)
    print(f"  {len(parquet_df)} pages")

    for fold in args.folds:
        for split in SPLITS:
            entries = generate_pagecrop(
                split, fold,
                jazzmus_fullpage=args.jazzmus_fullpage,
                parquet_df=parquet_df,
                out_dir=args.out_dir,
                max_n=args.max_n,
                bottom_pad=args.bottom_pad,
            )
            if entries:
                write_split_file(entries, args.out_dir, split, fold)

    print(f"\n✓ Done — data written to {args.out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
