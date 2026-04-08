"""
Evaluate a fixed list of best checkpoints on kern-spine-only CER / SER / LER.

Run from FullPageJazzOMR/:
    python eval_checkpoints.py
    python eval_checkpoints.py --data_path /path/to/jazzmus_pagecrop --fold 0
"""

import sys
from pathlib import Path

import torch
import fire

torch.set_float32_matmul_precision("high")
sys.path.insert(0, str(Path(__file__).parent))

from datasets.page_crop_dataset import PageCropDataset
from jazzmus.curriculum.trainer import CurriculumSMTTrainer
from jazzmus.dataset.smt_dataset import batch_preparation_img2seq
from jazzmus.dataset.tokenizer import untokenize
from jazzmus.dataset.eval_functions import compute_poliphony_metrics
from torch.utils.data import DataLoader

WEIGHTS_ROOT = Path("/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmus_weights")

# All 7 checkpoints to evaluate, in order
CHECKPOINTS = [
    ("no_replay",           WEIGHTS_ROOT / "pagecrop"          / "pagecrop_fold0_best.ckpt"),
    ("replay_25pct",        WEIGHTS_ROOT / "replay"            / "pagecrop_fold0_best.ckpt"),
    ("replay_50pct",        WEIGHTS_ROOT / "replay_50percent"  / "pagecrop_fold0_best.ckpt"),
    ("replay_100pct",       WEIGHTS_ROOT / "replay_100percent" / "pagecrop_fold0_best.ckpt"),
    ("replay_100pct_v1",    WEIGHTS_ROOT / "replay_100percent" / "pagecrop_fold0_best-v1.ckpt"),
    ("replay_100pct_v2",    WEIGHTS_ROOT / "replay_100percent" / "pagecrop_fold0_best-v2.ckpt"),
    ("replay_100pct_v2_bs1",WEIGHTS_ROOT / "replay_100percent" / "pagecrop_fold0_best-v2-bachsize1.ckpt"),
]


# ── kern-spine extraction ─────────────────────────────────────────────────────

def kern_spine_only(text: str) -> str:
    """Return only the **kern spine column from an untokenized humdrum string."""
    lines = text.split("\n")
    out = []
    kern_col = None
    for line in lines:
        if not line.strip():
            continue
        parts = line.split("\t")
        if kern_col is None:
            for i, p in enumerate(parts):
                if p.strip().startswith("**kern"):
                    kern_col = i
                    break
        if kern_col is not None and kern_col < len(parts):
            out.append(parts[kern_col])
        else:
            out.append(parts[0])
    return "\n".join(out)


# ── inference ─────────────────────────────────────────────────────────────────

def run_inference(checkpoint_path, data_path, fold, system_height, final_stage, batch_size, num_workers):
    model = CurriculumSMTTrainer.load_from_checkpoint(
        checkpoint_path, load_pretrained=False, strict=False,
    )
    w2i = model.model.w2i
    i2w = model.model.i2w

    test_set = PageCropDataset(
        data_path=data_path, split="test", fold=fold,
        augment=False, system_height=system_height,
    )
    test_set.set_dictionaries(w2i, i2w)
    test_set.set_stage_direct(final_stage)

    seq_maxlen = int(max(len(t) for t in test_set.gt_tokens) * 1.1)
    if seq_maxlen > model.model.maxlen:
        model.model.maxlen = seq_maxlen

    model.freeze()
    model.eval()

    loader = DataLoader(
        test_set, batch_size=batch_size, num_workers=num_workers,
        collate_fn=batch_preparation_img2seq,
    )

    preds_kern, gts_kern = [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        for batch in loader:
            x, di, y, _ = batch
            x = x.to(device)
            for x_s, y_s in zip(x, y):
                pred_seq, _ = model.model.predict(input=x_s)
                pred_text = untokenize(pred_seq)

                gt_toks = [i2w[t.item()] for t in y_s]
                clean = []
                for t in gt_toks:
                    if t in ("<eos>", "<pad>"):
                        break
                    clean.append(t)
                gt_text = untokenize(clean)

                preds_kern.append(kern_spine_only(pred_text))
                gts_kern.append(kern_spine_only(gt_text))

    cer, ser, ler = compute_poliphony_metrics(preds_kern, gts_kern)
    return {"cer": cer, "ser": ser, "ler": ler}


# ── main ──────────────────────────────────────────────────────────────────────

def main(
    data_path:    str = "/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmuss_Data/jazzmus_pagecrop",
    fold:          int = 0,
    final_stage:   int = 9,
    system_height: int = 256,
    batch_size:    int = 1,
    num_workers:   int = 4,
):
    # ── list all paths upfront ────────────────────────────────────────────────
    print("Checkpoints to evaluate:")
    missing = []
    for name, path in CHECKPOINTS:
        status = "OK" if path.exists() else "MISSING"
        print(f"  [{status}]  {name:<25}  {path}")
        if not path.exists():
            missing.append(name)
    if missing:
        print(f"\nWARNING: {len(missing)} checkpoint(s) missing — they will be skipped.")
    print()

    # ── evaluate ──────────────────────────────────────────────────────────────
    results = []
    for name, path in CHECKPOINTS:
        if not path.exists():
            continue
        print(f"=== {name} ===")
        m = run_inference(str(path), data_path, fold, system_height, final_stage, batch_size, num_workers)
        results.append((name, m))
        print(f"  Kern  CER={m['cer']:.2f}%  SER={m['ser']:.2f}%  LER={m['ler']:.2f}%\n")

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'Experiment':<25} {'Kern CER':>10} {'Kern SER':>10} {'Kern LER':>10}")
    print("-" * 65)
    for name, m in results:
        print(f"{name:<25} {m['cer']:>10.2f} {m['ser']:>10.2f} {m['ler']:>10.2f}")
    print("=" * 65)


if __name__ == "__main__":
    fire.Fire(main)
