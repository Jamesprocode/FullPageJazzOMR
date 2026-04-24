"""
Generate stacked train pages from system-level data into a standalone
upload folder.  The output's train_<fold>.txt only lists the new samples
(not merged with any existing split), and its paths point at
<target_folder>/jpg/... and <target_folder>/gt/... on the target server.

Outputs:
    upload_dir/
        jpg/      stacked_train_<fold>_n<N>_<src>_<i>.jpg   ...
        gt/       stacked_train_<fold>_n<N>_<src>_<i>.txt   ...
        splits/   train_<fold>.txt   (only the new stacked entries)

After rsync-ing to the server, append upload_dir/splits/train_<fold>.txt
to whichever dataset split you want to extend, or point a second syn
dataset at this new folder directly.

Sources:
    real:      jazzmus_systems/splits/train_<fold>.txt         (handwritten)
    synthetic: jazzmus_systems_syn/splits/train_<fold>.txt     (engraved)

Usage (run locally where real system jpgs exist):
    python add_stacked_train.py \\
        --system_data_path  /path/to/jazzmus_systems \\
        --syn_system_path   /path/to/jazzmus_systems_syn \\
        --upload_dir        upload_stacked_train \\
        --target_folder     jazzmus_fullpage_stacked \\
        --fold 0
"""

import argparse
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from prepare_stacked_data import (
    build_stacked_gt,
    stack_images,
    precompute_compatible,
    sample_indices,
    load_split,
)


# skewed toward n=9, 50/50 real/syn split applied at generation time
DEFAULT_DIST = {
    9: 300,
    8: 100,
    7: 60,
    6: 40,
    5: 40,
    4: 30,
    3: 20,
    2: 10,
    1: 10,
}


def _generate(
    out_jpg, out_gt,
    entries, prefix,
    counts,            # {n: num_samples_from_this_source}
    system_height,
    width_tolerance,
    fold,
    seed,
    target_folder,     # split-file path prefix, e.g. "jazzmus_fullpage_stacked"
):
    """Generate stacked pages from `entries`. Returns list of (img_rel, gt_rel)."""
    random.seed(seed)
    np.random.seed(seed)

    imgs, gt_lines, compatible = precompute_compatible(
        entries,
        width_tolerance=width_tolerance,
        system_height=system_height,
    )

    written = []
    for n, k in counts.items():
        print(f"  [{prefix}] generating {k} stacked n={n} samples")
        for i in range(k):
            idxs = sample_indices(n, imgs, compatible)
            stacked = stack_images([imgs[j] for j in idxs], system_height)
            gt_content = build_stacked_gt([gt_lines[j] for j in idxs])
            stem = f"stacked_train_{fold}_n{n}_{prefix}_{i:05d}"
            cv2.imwrite(str(out_jpg / f"{stem}.jpg"), stacked)
            with open(out_gt / f"{stem}.txt", "w") as f:
                f.writelines(gt_content)
            written.append((
                f"{target_folder}/jpg/{stem}.jpg",
                f"{target_folder}/gt/{stem}.txt",
            ))
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system_data_path", required=True)
    ap.add_argument("--syn_system_path",  required=True)
    ap.add_argument("--upload_dir",       required=True)
    ap.add_argument("--target_folder",    default="jazzmus_fullpage_stacked",
                    help="folder name used as path prefix inside the split "
                         "file (paths look like <target_folder>/jpg/<stem>.jpg)")
    ap.add_argument("--fold",             type=int, default=0)
    ap.add_argument("--system_height",     type=int, default=256)
    ap.add_argument("--width_tolerance",   type=float, default=0.15)
    ap.add_argument("--seed",              type=int, default=42)
    ap.add_argument("--total",             type=int, default=500,
                    help="total samples to generate; scales DEFAULT_DIST "
                         "proportionally if != 500")
    args = ap.parse_args()

    # Scale distribution to total
    scale = args.total / sum(DEFAULT_DIST.values())
    dist = {n: int(round(k * scale)) for n, k in DEFAULT_DIST.items()}
    # 50/50 split per n
    real_counts = {n: k // 2 for n, k in dist.items()}
    syn_counts  = {n: k - (k // 2) for n, k in dist.items()}

    print("Sample plan:")
    for n in sorted(dist):
        print(f"  n={n}:  real={real_counts[n]}  syn={syn_counts[n]}")
    print(f"  total: {sum(real_counts.values()) + sum(syn_counts.values())}")

    # ── paths ────────────────────────────────────────────────────────────────
    real_root = Path(args.system_data_path)
    syn_root  = Path(args.syn_system_path)
    upload    = Path(args.upload_dir)

    out_jpg    = upload / "jpg"
    out_gt     = upload / "gt"
    out_splits = upload / "splits"
    for d in (out_jpg, out_gt, out_splits):
        d.mkdir(parents=True, exist_ok=True)

    # ── load source splits ──────────────────────────────────────────────────
    real_split = real_root / "splits" / f"train_{args.fold}.txt"
    syn_split  = syn_root  / "splits" / f"train_{args.fold}.txt"
    for p in (real_split, syn_split):
        if not p.exists():
            raise FileNotFoundError(p)

    real_entries = load_split(real_split, real_root.parent)
    syn_entries  = load_split(syn_split,  syn_root.parent)
    print(f"\nLoaded system-level splits:")
    print(f"  real: {len(real_entries)} entries from {real_split}")
    print(f"  syn:  {len(syn_entries)} entries from {syn_split}")

    # ── generate ────────────────────────────────────────────────────────────
    new_lines = []
    new_lines += _generate(
        out_jpg, out_gt, real_entries, "real",
        real_counts, args.system_height, args.width_tolerance,
        args.fold, args.seed, args.target_folder,
    )
    new_lines += _generate(
        out_jpg, out_gt, syn_entries, "syn",
        syn_counts, args.system_height, args.width_tolerance,
        args.fold, args.seed + 1, args.target_folder,
    )
    print(f"\nGenerated {len(new_lines)} new stacked train samples")

    # ── write standalone split (only the new lines) ─────────────────────────
    out_split_file = out_splits / f"train_{args.fold}.txt"
    with open(out_split_file, "w") as f:
        for img_rel, gt_rel in new_lines:
            f.write(f"{img_rel} {gt_rel}\n")
    print(f"\nWrote split with {len(new_lines)} new stacked lines → {out_split_file}")

    # ── deploy instructions ─────────────────────────────────────────────────
    print(f"\nTo deploy:")
    print(f"  rsync -av {upload}/jpg/  <server>:/path/to/{args.target_folder}/jpg/")
    print(f"  rsync -av {upload}/gt/   <server>:/path/to/{args.target_folder}/gt/")
    print(f"  scp {out_split_file}     <server>:/path/to/{args.target_folder}/splits/train_{args.fold}.txt")
    print(f"\nThen append its lines to whichever train split your dataset reads,")
    print(f"or point a second syn dataset at {args.target_folder}.")


if __name__ == "__main__":
    main()
