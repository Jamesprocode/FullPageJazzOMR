"""
Offline data preparation: generate pre-computed synthetic N-system crops.

Mirrors prepare_pagecrop.py exactly, but uses the JAZZMUS_Synthetic parquet
instead of the real handwritten parquet.

For each synthetic page, crops the full rendered image vertically to include
only the first N systems (N = 1 … min(max_n, systems_on_page)), using
bounding-box coordinates from the synthetic parquet.  GT is cut from the
per-system kern fields in the parquet annotation (identical logic to
build_gt_from_fullpage in prepare_pagecrop.py).

Only a train split is generated — val and test always use real handwritten data.

Output:
  - data/jazzmus_synthetic/jpg/img_<X>_n<N>.jpg  — cropped synthetic image
  - data/jazzmus_synthetic/gt/img_<X>_n<N>.txt   — GT kern (raw, not tokenized)

Split file written to data/jazzmus_synthetic/splits/train_{fold}.txt
with format:  <img_path> <gt_path> <N>

Run from the FullPageJazzOMR/ project root:
    python data_prep/prepare_synthetic.py \\
        --parquet  ../JAZZMUS/JAZZMUS_Synthetic/data/train-00000-of-00001.parquet \\
        --out_dir  data/jazzmus_synthetic \\
        --max_n    11 \\
        --bottom_pad 20
"""

import argparse
import ast
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

SPINE_TERM = "*-"


# ── GT helpers ──────────────────────────────────────────────────────────────────

def _extract_block(kern_text: str, is_first: bool) -> list:
    """
    Extract the music block from one system's kern text field.

    System kern text has one of two shapes:

      first system  : header lines  →  music lines  →  !!linebreak:original  →  *-
      later systems : header lines  →  !!linebreak:original  →  music lines
                      →  !!linebreak:original  →  *-

    Returns only the music portion as a list of lines (strings with \\n).
    """
    lines = kern_text.splitlines(keepends=True)
    lines = [l if l.endswith("\n") else l + "\n" for l in lines if l.strip()]

    if is_first:
        # Everything before the first !!linebreak:original = header + music
        block = []
        for l in lines:
            if l.strip() == "!!linebreak:original":
                break
            block.append(l)
        return block
    else:
        # Skip header lines (everything up to and including the first
        # !!linebreak:original), then collect music up to the second one.
        found_lb = False
        block = []
        for l in lines:
            if l.strip() == "!!linebreak:original":
                if not found_lb:
                    found_lb = True   # consume the header-end marker
                else:
                    break             # stop at the trailing linebreak
            elif found_lb:
                block.append(l)
        return block


def build_gt_from_systems(systems: list, n: int) -> list:
    """
    Build GT kern for the first n systems by cutting from the parquet kern fields.

    Identical output format to build_gt_from_fullpage() in prepare_pagecrop.py:
      - inner blocks have *- stripped and end with !!linebreak:original
      - last block ends with *-  *-
    """
    blocks = [_extract_block(systems[i]["**kern"], i == 0) for i in range(n)]

    kern_lines = []
    for i, block in enumerate(blocks):
        if i < n - 1:
            # Inner block: strip any stray *- terminators, add linebreak
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


# ── image cropping ──────────────────────────────────────────────────────────────

def crop_page(img: np.ndarray, systems: list, n: int, bottom_pad: int) -> np.ndarray:
    """Crop full-page image to include only the first n systems (same as prepare_pagecrop)."""
    crop_y = systems[n - 1]["bounding_box"]["toY"] + bottom_pad
    crop_y = min(crop_y, img.shape[0])
    return img[:crop_y, :]


# ── main generation ─────────────────────────────────────────────────────────────

def generate_synthetic(parquet_df, out_dir, max_n, bottom_pad):
    """Generate N-system crops for all pages in the synthetic parquet."""
    jpg_dir = out_dir / "jpg"
    gt_dir  = out_dir / "gt"
    jpg_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n── synthetic train  (N=1..{max_n}) ───")

    split_entries = []
    n_generated = 0
    n_skipped   = 0

    for page_idx, row in parquet_df.iterrows():
        ann = row["annotation"]
        if isinstance(ann, str):
            ann = ast.literal_eval(ann)

        systems = ann.get("systems", [])
        if not systems:
            n_skipped += 1
            continue

        # Decode full-page image from parquet bytes
        img_bytes = row["image"]["bytes"]
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  Warning: could not decode image for page {page_idx}")
            n_skipped += 1
            continue

        num_sys = len(systems)

        for n in range(1, min(max_n, num_sys) + 1):
            stem    = f"img_{page_idx}_n{n}"
            out_gt  = gt_dir  / f"{stem}.txt"
            out_jpg = jpg_dir / f"{stem}.jpg"

            try:
                kern_lines = build_gt_from_systems(systems, n)
                out_gt.write_text("".join(kern_lines))

                if n == num_sys:
                    # Last system = full page — no crop
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

    print(f"  Generated {n_generated} entries  ({n_skipped} skipped)")
    return split_entries


def write_split_file(entries, out_dir, fold):
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_path = splits_dir / f"train_{fold}.txt"
    split_path.write_text("".join(entries))
    print(f"  Split file → {split_path}  ({len(entries)} entries)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic N-system crops from the JAZZMUS_Synthetic parquet."
    )
    parser.add_argument("--parquet", type=Path,
                        default=Path("../JAZZMUS/JAZZMUS_Synthetic/data/train-00000-of-00001.parquet"),
                        help="Path to JAZZMUS_Synthetic parquet with system bounding boxes")
    parser.add_argument("--out_dir", type=Path,
                        default=Path("data/jazzmus_synthetic"))
    parser.add_argument("--folds", type=int, nargs="+", default=[0],
                        help="Fold indices to write split files for (all pages used for training)")
    parser.add_argument("--max_n", type=int, default=11)
    parser.add_argument("--bottom_pad", type=int, default=20,
                        help="Pixels of padding below last system toY (default: 20)")
    args = parser.parse_args()

    print(f"Output directory  : {args.out_dir.resolve()}")
    print(f"Synthetic parquet : {args.parquet.resolve()}")
    print(f"Folds             : {args.folds}")
    print(f"Max N             : {args.max_n}")
    print(f"Bottom pad        : {args.bottom_pad} px")

    print("\nLoading synthetic parquet…")
    parquet_df = pd.read_parquet(args.parquet)
    print(f"  {len(parquet_df)} pages")

    entries = generate_synthetic(
        parquet_df=parquet_df,
        out_dir=args.out_dir,
        max_n=args.max_n,
        bottom_pad=args.bottom_pad,
    )

    # All synthetic pages are used for training in every fold
    for fold in args.folds:
        write_split_file(entries, args.out_dir, fold)

    print(f"\n✓ Done — data written to {args.out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
