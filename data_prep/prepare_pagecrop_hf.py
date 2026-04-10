"""
HuggingFace version of prepare_pagecrop.py.

Reads real handwritten data from PRAIG/JAZZMUS instead of local files.
All core logic (cropping, GT building, split files) is identical to
prepare_pagecrop.py — only the data source changes.

Run from the FullPageJazzOMR/ project root:
    python data_prep/prepare_pagecrop_hf.py \\
        --out_dir data/jazzmus_pagecrop_hf \\
        --folds 0 \\
        --max_n 11 \\
        --bottom_pad 20

Requires:
    pip install datasets
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

# Reuse GT building and cropping helpers from prepare_pagecrop.py
from data_prep.prepare_pagecrop import (
    build_gt_from_fullpage,
    crop_page,
    write_split_file,
)

HF_DATASET = "PRAIG/JAZZMUS"


def generate_pagecrop_hf(split, fold, hf_split, out_dir, max_n, bottom_pad):
    """Generate N-system crops for one split using HuggingFace dataset rows."""
    jpg_dir = out_dir / "jpg"
    gt_dir  = out_dir / "gt"
    jpg_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    do_crop = (split != "test")
    print(f"\n── {split} fold {fold}  ({'crop N=1..'+str(max_n) if do_crop else 'full page only'}) ───")

    split_entries = []
    n_generated = 0
    n_skipped   = 0

    for page_idx, row in enumerate(hf_split):
        # ── decode image from HF bytes ──────────────────────────────────────
        img_bytes = row["image"]["bytes"]
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  Warning: could not decode image for page {page_idx}")
            n_skipped += 1
            continue

        # ── bounding boxes from annotation ──────────────────────────────────
        ann = row["annotation"]
        if isinstance(ann, str):
            import ast
            ann = ast.literal_eval(ann)
        systems = ann.get("systems", [])
        if not systems:
            n_skipped += 1
            continue
        num_sys = len(systems)

        # ── write full-page GT to a temp file so build_gt_from_fullpage works,
        #    OR build GT directly from annotation kern fields if available ────
        # JAZZMUS annotation systems have "**kern" fields (same as synthetic).
        # Re-use build_gt_from_systems logic directly.
        from data_prep.prepare_synthetic import build_gt_from_systems

        if do_crop:
            for n in range(1, min(max_n, num_sys) + 1):
                stem    = f"img_{page_idx}_n{n}"
                out_gt  = gt_dir  / f"{stem}.txt"
                out_jpg = jpg_dir / f"{stem}.jpg"

                try:
                    kern_lines = build_gt_from_systems(systems, n)
                    out_gt.write_text("".join(kern_lines))

                    if n == num_sys:
                        cv2.imwrite(str(out_jpg), img)
                    else:
                        cropped = crop_page(img, systems, n, bottom_pad)
                        cv2.imwrite(str(out_jpg), cropped)

                    rel_jpg = out_jpg.relative_to(out_dir.parent)
                    rel_gt  = out_gt.relative_to(out_dir.parent)
                    split_entries.append(f"{rel_jpg} {rel_gt} {n}\n")
                    n_generated += 1
                except Exception as e:
                    print(f"  Warning: page {page_idx} n={n}: {e}")
                    n_skipped += 1
        else:
            # Test: full page only
            stem    = f"img_{page_idx}_n{num_sys}"
            out_gt  = gt_dir  / f"{stem}.txt"
            out_jpg = jpg_dir / f"{stem}.jpg"
            try:
                kern_lines = build_gt_from_systems(systems, num_sys)
                out_gt.write_text("".join(kern_lines))
                cv2.imwrite(str(out_jpg), img)
                rel_jpg = out_jpg.relative_to(out_dir.parent)
                rel_gt  = out_gt.relative_to(out_dir.parent)
                split_entries.append(f"{rel_jpg} {rel_gt} {num_sys}\n")
                n_generated += 1
            except Exception as e:
                print(f"  Warning: page {page_idx} full-page: {e}")
                n_skipped += 1

    print(f"  Recorded {n_generated} entries  ({n_skipped} pages skipped)")
    return split_entries


def main():
    parser = argparse.ArgumentParser(
        description="Generate N-system page crops from PRAIG/JAZZMUS on HuggingFace."
    )
    parser.add_argument("--out_dir",    type=Path, default=Path("data/jazzmus_pagecrop_hf"))
    parser.add_argument("--folds",      type=int, nargs="+", default=[0])
    parser.add_argument("--max_n",      type=int, default=11)
    parser.add_argument("--bottom_pad", type=int, default=20)
    args = parser.parse_args()

    print(f"HuggingFace dataset : {HF_DATASET}")
    print(f"Output directory    : {args.out_dir.resolve()}")
    print(f"Folds               : {args.folds}")
    print(f"Max N               : {args.max_n}")
    print(f"Bottom pad          : {args.bottom_pad} px")

    print(f"\nLoading {HF_DATASET} from HuggingFace…")
    ds = load_dataset(HF_DATASET)
    print(f"  Splits available: {list(ds.keys())}")

    hf_split_map = {
        "train": ds.get("train"),
        "val":   ds.get("validation") or ds.get("val"),
        "test":  ds.get("test"),
    }

    for fold in args.folds:
        for split, hf_split in hf_split_map.items():
            if hf_split is None:
                print(f"  Skipping {split}: not found in dataset")
                continue
            entries = generate_pagecrop_hf(
                split, fold, hf_split, args.out_dir, args.max_n, args.bottom_pad
            )
            if entries:
                write_split_file(entries, args.out_dir, split, fold)

    print(f"\n✓ Done — data written to {args.out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
