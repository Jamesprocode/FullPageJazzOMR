"""
Prepare stacked full-page data for curriculum training.

Reads jazzmus_systems / jazzmus_systems_syn splits and generates stacked page
images (N=2..max_n systems per page) saved to disk.  At N=1 the system-level
images themselves are used directly.  Real and synthetic full pages
(jazzmus_fullpage / jazzmus_fullpage_syn) are appended at N=max_n so they are
eligible only at the final curriculum stage.

Output layout:
    <out_dir>/
        jpg/   stacked_train_0_n2_000.jpg  ...
        gt/    stacked_train_0_n2_000.txt  ...
        splits/train_0.txt  val_0.txt  test_0.txt

GT files use !!linebreak:original between systems, exactly like jazzmus_pagecrop,
so the output can be used directly with PageCropDataset and train.py.

Usage:
    python prepare_stacked_data.py \\
        --system_data_path     ".../jazzmus_systems" \\
        --synthetic_system_path ".../jazzmus_systems_syn" \\
        --fullpage_data_path   ".../jazzmus_fullpage" \\
        --fullpage_syn_path    ".../jazzmus_fullpage_syn" \\
        --out_dir              ".../jazzmus_stacked" \\
        --fold 0 --max_n 9 --samples_per_n 1000 --val_samples 100
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ── GT helpers ────────────────────────────────────────────────────────────────

def _strip_linebreak_markers(lines):
    """Remove !!linebreak:original lines from a raw kern line list."""
    return [l for l in lines if l.strip() != "!!linebreak:original"]


def _strip_header(lines):
    """Return lines from the first barline (=) onward — drops kern header."""
    for i, l in enumerate(lines):
        if l.strip().startswith("="):
            return lines[i:]
    return lines   # no barline found; return as-is


def _strip_terminator(lines):
    """Remove trailing *- spine-terminator lines."""
    lines = list(lines)
    while lines:
        stripped = lines[-1].strip()
        # match "*-\t*-", "*-", and similar
        if all(part == "*-" for part in stripped.split("\t")):
            lines.pop()
        else:
            break
    return lines


def build_stacked_gt(system_kern_lines_list):
    """
    Combine per-system raw kern lines into a single stacked kern file.

    Output format (same as jazzmus_pagecrop):
        <system 0 header + notes>
        !!linebreak:original
        <system 1 notes — no header>
        !!linebreak:original
        ...
        <system N notes>
        *-\\t*-
    """
    result = []
    n = len(system_kern_lines_list)

    for i, raw_lines in enumerate(system_kern_lines_list):
        is_first = (i == 0)
        is_last  = (i == n - 1)

        lines = _strip_linebreak_markers(list(raw_lines))

        if not is_first:
            lines = _strip_header(lines)      # keep only barline + notes
            result.append("!!linebreak:original\n")

        if not is_last:
            lines = _strip_terminator(lines)  # drop *- for non-last systems

        result.extend(lines)

    return result


# ── image helpers ─────────────────────────────────────────────────────────────

def stack_images(imgs, system_height):
    """Resize each image to system_height rows, pad widths to match, vstack."""
    resized = []
    for img in imgs:
        h, w = img.shape[:2]
        if h != system_height:
            new_w = max(1, int(round(w * system_height / h)))
            img = cv2.resize(img, (new_w, system_height),
                             interpolation=cv2.INTER_LINEAR)
        resized.append(img)
    max_w = max(im.shape[1] for im in resized)
    padded = [
        np.pad(im, ((0, 0), (0, max_w - im.shape[1])), constant_values=255)
        for im in resized
    ]
    return np.vstack(padded)


# ── per-split generation ──────────────────────────────────────────────────────

def resolve_path(raw: str, base_dir: Path) -> Path:
    """Resolve a potentially relative path, trying multiple base locations."""
    p = Path(raw)
    if p.exists():
        return p
    # Try as-is relative to base_dir
    candidate = base_dir / p
    if candidate.exists():
        return candidate
    # Strip the first path component (e.g. "data/" prefix from old layout)
    parts = p.parts
    if len(parts) > 1:
        stripped = base_dir / Path(*parts[1:])
        if stripped.exists():
            return stripped
    return candidate   # return best guess even if missing (caller checks)


def load_split(split_file, base_dir):
    """Load (img_path, gt_path) pairs from a 2-column split file."""
    entries = []
    with open(split_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            raw_img, raw_gt = parts[0], parts[1]
            img_path = resolve_path(raw_img, base_dir)
            gt_path  = resolve_path(raw_gt,  base_dir)
            if img_path.exists() and gt_path.exists():
                entries.append((img_path, gt_path))
    return entries


def load_fullpage_split(split_file, base_dir):
    """Load (img_path, gt_path) pairs from a fullpage split file (no N column).

    Returns list of (img_path_str, gt_path_str) using the original raw strings
    so they can be written directly into the output split file.
    """
    lines_out = []
    with open(split_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            raw_img, raw_gt = parts[0], parts[1]
            img_path = resolve_path(raw_img, base_dir)
            gt_path  = resolve_path(raw_gt,  base_dir)
            if img_path.exists() and gt_path.exists():
                lines_out.append((raw_img, raw_gt))
    return lines_out


def precompute_compatible(entries, width_tolerance=0.15, system_height=256):
    """
    Load all images (resize to system_height) and build per-image width-compatible pools.
    Returns (imgs, gt_lines, compatible_pools).
    """
    imgs, gt_lines = [], []
    print(f"  Loading {len(entries)} system images…", end="", flush=True)
    for img_path, gt_path in entries:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            imgs.append(None)
            gt_lines.append([])
            continue
        h, w = img.shape[:2]
        if h != system_height:
            new_w = max(1, int(round(w * system_height / h)))
            img = cv2.resize(img, (new_w, system_height),
                             interpolation=cv2.INTER_LINEAR)
        imgs.append(img)
        with open(gt_path) as f:
            gt_lines.append(f.readlines())
    print(" done")

    widths = np.array([
        img.shape[1] if img is not None else 0 for img in imgs
    ])
    compatible = [
        [j for j in range(len(widths))
         if imgs[j] is not None
         and abs(int(widths[j]) - int(widths[i])) / max(int(widths[i]), 1) <= width_tolerance]
        if imgs[i] is not None else []
        for i in range(len(imgs))
    ]
    pool_sizes = [len(c) for c in compatible]
    print(f"  Width pools — min: {min(pool_sizes)}  "
          f"median: {int(np.median(pool_sizes))}  max: {max(pool_sizes)}")
    return imgs, gt_lines, compatible


def sample_indices(n, imgs, compatible):
    """Sample n compatible system indices."""
    valid = [i for i in range(len(imgs)) if imgs[i] is not None]
    anchor = random.choice(valid)
    pool = compatible[anchor]
    if not pool or len(pool) < 1:
        pool = valid
    if len(pool) < n:
        pool = valid
    rest = random.choices(pool, k=n - 1)
    return [anchor] + rest


def generate_split(
    split_name, fold, entries,
    out_dir, max_n,
    n_real_per_n, n_syn_per_n,
    system_height, width_tolerance,
    seed=42,
    syn_entries=None,
):
    """Generate stacked pages from entries (and optionally syn_entries).

    n_real_per_n: number of real-stacked pages per N
    n_syn_per_n:  number of syn-stacked pages per N (requires syn_entries)
    """
    random.seed(seed)
    np.random.seed(seed)

    imgs, gt_lines, compatible = precompute_compatible(
        entries, width_tolerance=width_tolerance, system_height=system_height
    )

    syn_imgs, syn_gt_lines, syn_compatible = None, None, None
    if syn_entries:
        print(f"  Loading synthetic systems…", end="", flush=True)
        syn_imgs, syn_gt_lines, syn_compatible = precompute_compatible(
            syn_entries, width_tolerance=width_tolerance, system_height=system_height
        )

    jpg_dir    = out_dir / "jpg"
    gt_dir     = out_dir / "gt"
    splits_dir = out_dir / "splits"
    for d in (jpg_dir, gt_dir, splits_dir):
        d.mkdir(parents=True, exist_ok=True)

    split_lines = []
    total = 0

    for n in range(2, max_n + 1):   # N=1 is handled by system-level entries, skip stacking
        print(f"  [{split_name}_fold{fold}]  N={n}  "
              f"real={n_real_per_n}  syn={n_syn_per_n}…", flush=True)

        # ── real stacked samples ──────────────────────────────────────────────
        for s in range(n_real_per_n):
            idxs     = sample_indices(n, imgs, compatible)
            stacked  = stack_images([imgs[i] for i in idxs], system_height)
            gt_content = build_stacked_gt([gt_lines[i] for i in idxs])
            stem     = f"stacked_{split_name}_{fold}_n{n}_{s:05d}"
            img_rel  = f"jazzmus_stacked/jpg/{stem}.jpg"
            gt_rel   = f"jazzmus_stacked/gt/{stem}.txt"
            cv2.imwrite(str(jpg_dir / f"{stem}.jpg"), stacked)
            with open(gt_dir / f"{stem}.txt", "w") as f:
                f.writelines(gt_content)
            split_lines.append(f"{img_rel} {gt_rel} {n}\n")
            total += 1

        # ── synthetic stacked samples ─────────────────────────────────────────
        if n_syn_per_n > 0 and syn_imgs:
            for s in range(n_syn_per_n):
                idxs     = sample_indices(n, syn_imgs, syn_compatible)
                stacked  = stack_images([syn_imgs[i] for i in idxs], system_height)
                gt_content = build_stacked_gt([syn_gt_lines[i] for i in idxs])
                stem     = f"syn_stacked_{split_name}_{fold}_n{n}_{s:05d}"
                img_rel  = f"jazzmus_stacked/jpg/{stem}.jpg"
                gt_rel   = f"jazzmus_stacked/gt/{stem}.txt"
                cv2.imwrite(str(jpg_dir / f"{stem}.jpg"), stacked)
                with open(gt_dir / f"{stem}.txt", "w") as f:
                    f.writelines(gt_content)
                split_lines.append(f"{img_rel} {gt_rel} {n}\n")
                total += 1

    # Write split file
    split_file = splits_dir / f"{split_name}_{fold}.txt"
    with open(split_file, "w") as f:
        f.writelines(split_lines)
    print(f"  → {total} samples written, split: {split_file}")
    return split_lines


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system_data_path",     required=True,
                    help="Path to jazzmus_systems/ directory")
    ap.add_argument("--synthetic_system_path", default=None,
                    help="Path to jazzmus_systems_syn/ directory")
    ap.add_argument("--fullpage_data_path",   default=None,
                    help="Path to jazzmus_fullpage/ directory (real full pages, no N column)")
    ap.add_argument("--fullpage_syn_path",    default=None,
                    help="Path to jazzmus_fullpage_syn/ directory (synthetic full pages, no N column)")
    ap.add_argument("--out_dir",              required=True,
                    help="Output directory (will be created)")
    ap.add_argument("--fold",                 type=int,   default=0)
    ap.add_argument("--max_n",                type=int,   default=9,
                    help="Max systems per page; full pages added at this stage only")
    ap.add_argument("--real_samples_per_n",    type=int,   default=1688,
                    help="Real-stacked samples to generate per N for train split")
    ap.add_argument("--syn_samples_per_n",    type=int,   default=1872,
                    help="Syn-stacked samples to generate per N for train split")
    # val stacking count is derived automatically from len(val_entries)
    ap.add_argument("--system_height",        type=int,   default=256)
    ap.add_argument("--width_tolerance",      type=float, default=0.15)
    args = ap.parse_args()

    system_data_path  = Path(args.system_data_path)
    out_dir           = Path(args.out_dir)
    base_dir          = system_data_path.parent   # Jazzmuss_Data root

    print(f"System data      : {system_data_path}")
    print(f"Output dir       : {out_dir}")
    print(f"Fold             : {args.fold}  |  Max N: {args.max_n}")
    print(f"Train samples/N  : {args.real_samples_per_n} real + {args.syn_samples_per_n} syn  |  Val: auto (matches system-level count)")
    print()

    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # ── load synthetic system path (shared across splits) ─────────────────────
    syn_sys_path = Path(args.synthetic_system_path) if args.synthetic_system_path else None
    syn_base = syn_sys_path.parent if syn_sys_path else None

    # ── load fullpage paths ────────────────────────────────────────────────────
    fp_path     = Path(args.fullpage_data_path)     if args.fullpage_data_path     else None
    fp_syn_path = Path(args.fullpage_syn_path)      if args.fullpage_syn_path      else None
    fp_base     = fp_path.parent                    if fp_path                     else None
    fp_syn_base = fp_syn_path.parent                if fp_syn_path                 else None

    # ── TRAIN ─────────────────────────────────────────────────────────────────
    print("=== TRAIN ===")
    train_entries = load_split(
        system_data_path / "splits" / f"train_{args.fold}.txt", base_dir)
    print(f"  {len(train_entries)} real systems")

    syn_train_entries = None
    if syn_sys_path:
        syn_split = syn_sys_path / "splits" / f"train_{args.fold}.txt"
        if syn_split.exists():
            syn_train_entries = load_split(syn_split, syn_base)
            print(f"  {len(syn_train_entries)} synthetic systems (50/50 stacking)")

    # Generate stacked pages N=2..max_n (N=1 handled by system-level entries below)
    generate_split(
        split_name="train", fold=args.fold,
        entries=train_entries, out_dir=out_dir,
        max_n=args.max_n,
        n_real_per_n=args.real_samples_per_n,
        n_syn_per_n=args.syn_samples_per_n if syn_train_entries else 0,
        system_height=args.system_height,
        width_tolerance=args.width_tolerance, seed=42,
        syn_entries=syn_train_entries,
    )

    # Stage 1 = system-level images directly (no stacking needed for N=1)
    sys_split_file = system_data_path / "splits" / f"train_{args.fold}.txt"
    with open(sys_split_file) as f:
        sys_lines = [l for l in f if l.strip()]
    sys_n1_lines = [f"{l.split()[0]} {l.split()[1]} 1\n" for l in sys_lines]
    with open(splits_dir / f"train_{args.fold}.txt", "a") as f:
        f.writelines(sys_n1_lines)
    print(f"  + {len(sys_n1_lines)} real system-level N=1 entries")

    if syn_sys_path and syn_train_entries:
        syn_split_file = syn_sys_path / "splits" / f"train_{args.fold}.txt"
        with open(syn_split_file) as f:
            syn_sys_raw = [f"{l.split()[0]} {l.split()[1]} 1\n" for l in f if l.strip()]
        with open(splits_dir / f"train_{args.fold}.txt", "a") as f:
            f.writelines(syn_sys_raw)
        print(f"  + {len(syn_sys_raw)} synthetic system-level N=1 entries")

    # Real full pages at N=max_n (eligible only at final stage)
    if fp_path:
        fp_train_file = fp_path / "splits" / f"train_{args.fold}.txt"
        if fp_train_file.exists():
            fp_train = load_fullpage_split(fp_train_file, fp_base)
            fp_train_lines = [f"{img} {gt} {args.max_n}\n" for img, gt in fp_train]
            with open(splits_dir / f"train_{args.fold}.txt", "a") as f:
                f.writelines(fp_train_lines)
            print(f"  + {len(fp_train_lines)} real full-page entries (N={args.max_n})")
        else:
            print(f"  Warning: fullpage train split not found: {fp_train_file}")

    # Synthetic full pages at N=max_n (eligible only at final stage)
    if fp_syn_path:
        fp_syn_train_file = fp_syn_path / "splits" / f"train_{args.fold}.txt"
        if fp_syn_train_file.exists():
            fp_syn_train = load_fullpage_split(fp_syn_train_file, fp_syn_base)
            fp_syn_train_lines = [f"{img} {gt} {args.max_n}\n" for img, gt in fp_syn_train]
            with open(splits_dir / f"train_{args.fold}.txt", "a") as f:
                f.writelines(fp_syn_train_lines)
            print(f"  + {len(fp_syn_train_lines)} synthetic full-page entries (N={args.max_n})")
        else:
            print(f"  Warning: fullpage_syn train split not found: {fp_syn_train_file}")
    print()

    # ── VAL ───────────────────────────────────────────────────────────────────
    print("=== VAL ===")
    val_entries = load_split(
        system_data_path / "splits" / f"val_{args.fold}.txt", base_dir)
    n_val_systems = len(val_entries)
    print(f"  {n_val_systems} real systems → {n_val_systems} stacked per stage")

    # Stacking for N=2..max_n-1 only; stage max_n uses fullpage val exclusively.
    # n_real_per_n matches the system-level val count for a consistent epoch size.
    generate_split(
        split_name="val", fold=args.fold,
        entries=val_entries, out_dir=out_dir,
        max_n=args.max_n - 1,          # stacking stops at stage max_n-1
        n_real_per_n=n_val_systems,
        n_syn_per_n=0,
        system_height=args.system_height,
        width_tolerance=args.width_tolerance, seed=123,
    )

    # Stage 1 val = system-level images (direct copy, no stacking)
    val_sys_file = system_data_path / "splits" / f"val_{args.fold}.txt"
    with open(val_sys_file) as f:
        val_sys_lines = [l for l in f if l.strip()]
    val_n1_lines = [f"{l.split()[0]} {l.split()[1]} 1\n" for l in val_sys_lines]
    with open(splits_dir / f"val_{args.fold}.txt", "a") as f:
        f.writelines(val_n1_lines)
    print(f"  + {len(val_n1_lines)} system-level N=1 val entries")

    # Stage max_n val = real fullpage only (replaces stacking at final stage)
    if fp_path:
        fp_val_file = fp_path / "splits" / f"val_{args.fold}.txt"
        if fp_val_file.exists():
            fp_val = load_fullpage_split(fp_val_file, fp_base)
            fp_val_lines = [f"{img} {gt} {args.max_n}\n" for img, gt in fp_val]
            with open(splits_dir / f"val_{args.fold}.txt", "a") as f:
                f.writelines(fp_val_lines)
            print(f"  + {len(fp_val_lines)} real fullpage val entries at N={args.max_n} (final stage only)")
        else:
            print(f"  Warning: fullpage val split not found: {fp_val_file}")
    print()

    # ── TEST: real fullpage only — honest evaluation on real scans ─────────────
    print("=== TEST ===")
    test_out_file = splits_dir / f"test_{args.fold}.txt"
    if fp_path:
        fp_test_file = fp_path / "splits" / f"test_{args.fold}.txt"
        if fp_test_file.exists():
            fp_test = load_fullpage_split(fp_test_file, fp_base)
            fp_test_lines = [f"{img} {gt} {args.max_n}\n" for img, gt in fp_test]
            with open(test_out_file, "w") as f:
                f.writelines(fp_test_lines)
            print(f"  {len(fp_test_lines)} real fullpage test entries → {test_out_file}")
        else:
            print(f"  Warning: fullpage test split not found: {fp_test_file}")
    else:
        print("  No --fullpage_data_path provided; test split not written.")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    def _count(split_name):
        p = splits_dir / f"{split_name}_{args.fold}.txt"
        return sum(1 for _ in open(p)) if p.exists() else 0

    print("Summary:")
    print(f"  Train : {_count('train')} total entries")
    print(f"  Val   : {_count('val')} total entries")
    print(f"  Test  : {_count('test')} entries (real fullpage only)")
    print()
    print("Train with:")
    print("  python train.py --config config/stacked_precomputed_9stage.gin")


if __name__ == "__main__":
    main()
