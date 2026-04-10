"""
HuggingFace version of prepare_synthetic.py.

Reads synthetic data from PRAIG/JAZZMUS_Synthetic instead of a local parquet.
All core logic (cropping, GT building, split files) is identical to
prepare_synthetic.py — only the data source changes.

Run from the FullPageJazzOMR/ project root:
    python data_prep/prepare_synthetic_hf.py \\
        --out_dir data/jazzmus_synthetic_hf \\
        --folds 0 \\
        --max_n 11 \\
        --bottom_pad 20

Requires:
    pip install datasets
"""

import argparse
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

# Reuse all core logic from prepare_synthetic.py
from data_prep.prepare_synthetic import (
    generate_synthetic,
    write_split_file,
)

HF_DATASET = "PRAIG/JAZZMUS_Synthetic"


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic N-system crops from PRAIG/JAZZMUS_Synthetic on HuggingFace."
    )
    parser.add_argument("--out_dir",    type=Path, default=Path("data/jazzmus_synthetic_hf"))
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
    ds = load_dataset(HF_DATASET, split="train")
    print(f"  {len(ds)} pages")

    # Convert HF dataset to a pandas-compatible iterable that generate_synthetic expects
    # (generate_synthetic iterates with .iterrows() and accesses row["annotation"] / row["image"])
    import pandas as pd
    df = ds.to_pandas()
    print(f"  Converted to DataFrame: {len(df)} rows")

    entries = generate_synthetic(
        parquet_df=df,
        out_dir=args.out_dir,
        max_n=args.max_n,
        bottom_pad=args.bottom_pad,
    )

    for fold in args.folds:
        write_split_file(entries, args.out_dir, fold)

    print(f"\n✓ Done — data written to {args.out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
