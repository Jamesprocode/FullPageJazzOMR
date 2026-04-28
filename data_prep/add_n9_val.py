"""
Surgical patch: generate n=9 stacked val samples and rewrite splits/val_0.txt.

Outputs ONLY the new files into <upload_dir> so you can rsync that single
folder to the cluster (mirroring the jazzmus_stacked layout).  The existing
jazzmus_stacked/ directory is not modified.

Layout of <upload_dir> after running:
    upload_dir/
      jpg/       stacked_val_0_n9_00000.jpg  ...   (105 new images)
      gt/        stacked_val_0_n9_00000.txt  ...   (105 new GTs)
      splits/    val_0.txt                         (full rewritten split)

Then on the cluster:
    rsync -av upload_dir/jpg/    .../jazzmus_stacked/jpg/
    rsync -av upload_dir/gt/     .../jazzmus_stacked/gt/
    cp upload_dir/splits/val_0.txt .../jazzmus_stacked/splits/val_0.txt

Usage:
    python add_n9_val.py \
        --system_data_path  Jazzmuss_Data/jazzmus_systems \
        --stacked_dir       Jazzmuss_Data/jazzmus_stacked \
        --upload_dir        upload_n9_val \
        --fold 0 --n 9 --n_samples 105
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system_data_path", required=True,
                    help="Path to jazzmus_systems/ directory")
    ap.add_argument("--stacked_dir", required=True,
                    help="Path to existing jazzmus_stacked/ (read-only; only "
                         "splits/val_<fold>.txt is used as the base to rewrite)")
    ap.add_argument("--upload_dir", required=True,
                    help="Output dir for new files (rsync this to cluster)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=9,
                    help="N for stacked val samples")
    ap.add_argument("--n_samples", type=int, default=105,
                    help="Number of n=N val samples to generate")
    ap.add_argument("--system_height", type=int, default=256)
    ap.add_argument("--width_tolerance", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    system_data_path = Path(args.system_data_path)
    stacked_dir      = Path(args.stacked_dir)
    upload_dir       = Path(args.upload_dir)
    base_dir         = system_data_path.parent

    src_split = stacked_dir / "splits" / f"val_{args.fold}.txt"
    if not src_split.exists():
        raise FileNotFoundError(f"{src_split} does not exist.")

    out_jpg    = upload_dir / "jpg"
    out_gt     = upload_dir / "gt"
    out_splits = upload_dir / "splits"
    for d in (out_jpg, out_gt, out_splits):
        d.mkdir(parents=True, exist_ok=True)

    # ── load system-level val entries ─────────────────────────────────────────
    val_entries = load_split(
        system_data_path / "splits" / f"val_{args.fold}.txt", base_dir)
    print(f"Loaded {len(val_entries)} system-level val entries")
    if len(val_entries) < args.n:
        raise ValueError(f"Need at least {args.n} val systems to stack, "
                         f"got {len(val_entries)}")

    # ── generate stacked n=N samples ──────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)

    imgs, gt_lines, compatible = precompute_compatible(
        val_entries,
        width_tolerance=args.width_tolerance,
        system_height=args.system_height,
    )

    new_lines = []
    print(f"Generating {args.n_samples} stacked n={args.n} val samples "
          f"→ {upload_dir}/")
    for s in range(args.n_samples):
        idxs    = sample_indices(args.n, imgs, compatible)
        stacked = stack_images([imgs[i] for i in idxs], args.system_height)
        gt_content = build_stacked_gt([gt_lines[i] for i in idxs])
        stem    = f"stacked_val_{args.fold}_n{args.n}_{s:05d}"
        img_rel = f"jazzmus_stacked/jpg/{stem}.jpg"
        gt_rel  = f"jazzmus_stacked/gt/{stem}.txt"
        cv2.imwrite(str(out_jpg / f"{stem}.jpg"), stacked)
        with open(out_gt / f"{stem}.txt", "w") as f:
            f.writelines(gt_content)
        new_lines.append(f"{img_rel} {gt_rel} {args.n}\n")
    print(f"  wrote {len(new_lines)} images → {out_jpg}/")
    print(f"  wrote {len(new_lines)} GTs    → {out_gt}/")

    # ── write new val_<fold>.txt into upload_dir/splits/ ──────────────────────
    existing = src_split.read_text().splitlines(keepends=True)
    kept, dropped = [], 0
    for line in existing:
        parts = line.strip().split()
        if len(parts) >= 3 and int(parts[2]) == args.n:
            dropped += 1
            continue
        if not line.endswith("\n"):
            line += "\n"
        kept.append(line)
    print(f"\nRewriting splits/val_{args.fold}.txt:")
    print(f"  dropped {dropped} existing n={args.n} entries (real fullpage)")

    out_split = out_splits / f"val_{args.fold}.txt"
    with open(out_split, "w") as f:
        f.writelines(kept)
        f.writelines(new_lines)
    print(f"  appended {len(new_lines)} new stacked n={args.n} entries")
    print(f"  → {out_split}")

    # ── report final distribution ─────────────────────────────────────────────
    dist = Counter()
    with open(out_split) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                dist[int(parts[2])] += 1
    print(f"\nFinal val split distribution:")
    for n in sorted(dist):
        marker = "  ← updated" if n == args.n else ""
        print(f"  n={n}: {dist[n]}{marker}")
    print(f"  total: {sum(dist.values())}")

    # ── upload instructions ───────────────────────────────────────────────────
    print(f"\nTo deploy on cluster:")
    print(f"  rsync -av {upload_dir}/jpg/    <cluster>:.../jazzmus_stacked/jpg/")
    print(f"  rsync -av {upload_dir}/gt/     <cluster>:.../jazzmus_stacked/gt/")
    print(f"  scp   {upload_dir}/splits/val_{args.fold}.txt "
          f"<cluster>:.../jazzmus_stacked/splits/val_{args.fold}.txt")


if __name__ == "__main__":
    main()
