"""
Offline data preparation: generate pre-computed synthetic N-system stacks.

Reads individual system crops from jazzmus_systems/ and stacks N consecutive
systems per page to produce synthetic multi-system training images.

GT is extracted from the corresponding full-page kern file (same as
prepare_pagecrop.py) — NOT by concatenating individual system kern files.
This keeps the GT format identical to real page crops, including the
full-page header (title, composer, etc.) present in the first system block.

Only a train split is generated — val and test always use real handwritten data.

Output:
  - data/jazzmus_synthetic/jpg/img_<X>_n<N>.jpg  — stacked system image
  - data/jazzmus_synthetic/gt/img_<X>_n<N>.txt   — GT kern (raw, not tokenized)

Split file written to data/jazzmus_synthetic/splits/train_{fold}.txt
with format:  <img_path> <gt_path> <N>

Run from the FullPageJazzOMR/ project root:
    python data_prep/prepare_synthetic.py \\
        --jazzmus_systems  ../ISMIR-Jazzmus/data/jazzmus_systems \\
        --jazzmus_fullpage ../ISMIR-Jazzmus/data/jazzmus_fullpage \\
        --out_dir          data/jazzmus_synthetic \\
        --folds 0 \\
        --max_n 11 \\
        --system_height 256
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

SPINE_TERM = "*-"


# ── GT helpers ──────────────────────────────────────────────────────────────────

def build_gt_from_fullpage(fullpage_gt_path: Path, n: int) -> list:
    """
    Extract the first n systems from the full-page kern GT as raw kern lines.

    Identical logic to prepare_pagecrop.py:
    - Splits on !!linebreak:original to get per-system blocks
    - Keeps the first n blocks (preserving title/header from block 0)
    - Strips *- terminators from inner blocks, adds !!linebreak:original between
    - Ensures *- terminators on the last block
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


# ── image stacking ──────────────────────────────────────────────────────────────

def stack_images(img_paths: list, n: int, system_height: int) -> np.ndarray:
    """Load and vertically stack the first n system images."""
    resized = []
    for p in img_paths[:n]:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {p}")
        h, w = img.shape[:2]
        if h != system_height:
            new_w = max(1, int(round(w * system_height / h)))
            img = cv2.resize(img, (new_w, system_height), interpolation=cv2.INTER_LINEAR)
        resized.append(img)

    max_w = max(img.shape[1] for img in resized)
    padded = [
        np.pad(img, ((0, 0), (0, max_w - img.shape[1])), constant_values=255)
        for img in resized
    ]
    return np.vstack(padded)


# ── main generation ─────────────────────────────────────────────────────────────

def generate_synthetic(fold, jazzmus_systems, jazzmus_fullpage, out_dir, max_n, system_height):
    """Generate N-system synthetic stacks for one fold (train only)."""
    split_file = jazzmus_systems / "splits" / f"train_{fold}.txt"
    if not split_file.exists():
        print(f"  Skipping fold {fold}: split file not found at {split_file}")
        return []

    base_dir = jazzmus_systems.parent.parent   # .../ISMIR-Jazzmus

    jpg_dir = out_dir / "jpg"
    gt_dir  = out_dir / "gt"
    jpg_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    fullpage_gt_dir = jazzmus_fullpage / "gt"

    # Parse split file and group by page
    pages = defaultdict(list)   # page_idx → list of (sys_idx, img_path)
    with open(split_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            raw_img = parts[0]

            img_path = Path(raw_img)
            if not img_path.exists():
                img_path = base_dir / raw_img

            m = re.search(r"img_(\d+)_(\d+)", str(img_path))
            if not m:
                continue
            page_idx = int(m.group(1))
            sys_idx  = int(m.group(2))
            pages[page_idx].append((sys_idx, img_path))

    # Sort systems within each page
    for page_idx in pages:
        pages[page_idx].sort(key=lambda t: t[0])

    print(f"\n── synthetic train fold {fold}  (N=1..{max_n}) ───")

    split_entries = []
    n_generated = 0
    n_skipped   = 0

    for page_idx, systems in sorted(pages.items()):
        num_sys = len(systems)
        img_paths = [s[1] for s in systems]

        # Full-page kern GT for this page
        fullpage_gt_path = fullpage_gt_dir / f"img_{page_idx}.txt"
        if not fullpage_gt_path.exists():
            print(f"  Warning: fullpage GT not found for page {page_idx}: {fullpage_gt_path}")
            n_skipped += num_sys
            continue

        for n in range(1, min(max_n, num_sys) + 1):
            stem    = f"img_{page_idx}_n{n}"
            out_gt  = gt_dir  / f"{stem}.txt"
            out_jpg = jpg_dir / f"{stem}.jpg"

            try:
                kern_lines = build_gt_from_fullpage(fullpage_gt_path, n)
                out_gt.write_text("".join(kern_lines))

                stacked = stack_images(img_paths, n, system_height)
                cv2.imwrite(str(out_jpg), stacked)

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
        description="Generate synthetic N-system stacks from system crops."
    )
    parser.add_argument("--jazzmus_systems", type=Path,
                        default=Path("../ISMIR-Jazzmus/data/jazzmus_systems_syn"))
    parser.add_argument("--jazzmus_fullpage", type=Path,
                        default=Path("../ISMIR-Jazzmus/data/jazzmus_fullpage"),
                        help="Path to jazzmus_fullpage (for full-page kern GT files)")
    parser.add_argument("--out_dir", type=Path,
                        default=Path("data/jazzmus_synthetic"))
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--max_n", type=int, default=11)
    parser.add_argument("--system_height", type=int, default=256,
                        help="Height in pixels for each system row (default: 256)")
    args = parser.parse_args()

    print(f"Output directory  : {args.out_dir.resolve()}")
    print(f"Systems source    : {args.jazzmus_systems.resolve()}")
    print(f"Full-page GT      : {args.jazzmus_fullpage.resolve()}")
    print(f"Folds             : {args.folds}")
    print(f"Max N             : {args.max_n}")
    print(f"System height     : {args.system_height} px")

    for fold in args.folds:
        entries = generate_synthetic(
            fold,
            jazzmus_systems=args.jazzmus_systems,
            jazzmus_fullpage=args.jazzmus_fullpage,
            out_dir=args.out_dir,
            max_n=args.max_n,
            system_height=args.system_height,
        )
        if entries:
            write_split_file(entries, args.out_dir, fold)

    print(f"\n✓ Done — data written to {args.out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
