"""
Evaluate a fixed list of checkpoints on the real full-page test split.

Reports 8 metrics per checkpoint:
  Overall : CER, SER, LER  (full untokenized humdrum text)
  Chord   : chord SER (no-dots), root SER
  Kern    : CER, SER, LER  (**kern spine only)

Run from FullPageJazzOMR/:
    python eval_checkpoints.py
    python eval_checkpoints.py --data_path /path/to/jazzmus_fullpage --fold 0
"""

import sys
from pathlib import Path

import torch
import fire

torch.set_float32_matmul_precision("high")
sys.path.insert(0, str(Path(__file__).parent))

from datasets.full_page_dataset import FullPageDataset
from jazzmus.curriculum.trainer import CurriculumSMTTrainer
from jazzmus.dataset.smt_dataset import batch_preparation_img2seq
from jazzmus.dataset.tokenizer import untokenize
from jazzmus.dataset.eval_functions import compute_poliphony_metrics
from jazzmus.dataset.chord_metrics import (
    extract_spines,
    extract_tokens_from_mxhm,
    compute_page_chord_metrics,
    aggregate_page_chord_metrics,
)
from torch.utils.data import DataLoader

# 4 checkpoints to evaluate, in order
CHECKPOINTS = [
    ("1", Path("/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmus_weights/replay_100percent/pagecrop_fold0_best-v1.ckpt")),
    ("2", Path("/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmus_weights/replay_100percent/pagecrop_fold0_best-v2-bachsize1.ckpt")),
    ("3", Path("/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmus_weights/replay_100percent/pagecrop_fold0_best-v2.ckpt")),
    ("4", Path("/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmus_weights/replay_100percent/pagecrop_fold0_best.ckpt")),
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


def mxhm_tokens(text: str):
    """Extract chord tokens (with dots) from the **mxhm spine of a humdrum page."""
    spines = extract_spines(text)
    mxhm = spines.get("**mxhm", "")
    if not mxhm:
        return []
    return extract_tokens_from_mxhm(mxhm)


# ── inference ─────────────────────────────────────────────────────────────────

def run_inference(checkpoint_path, data_path, fold, final_stage, num_workers):
    model = CurriculumSMTTrainer.load_from_checkpoint(
        checkpoint_path, load_pretrained=False, strict=False,
    )
    w2i = model.model.w2i
    i2w = model.model.i2w

    test_set = FullPageDataset(data_path=data_path, split="test", fold=fold)
    test_set.set_dictionaries(w2i, i2w)

    seq_maxlen = int(max(len(t) for t in test_set.gt_tokens) * 1.1)
    if seq_maxlen > model.model.maxlen:
        model.model.maxlen = seq_maxlen

    model.set_stage(final_stage)
    model.freeze()
    model.eval()

    loader = DataLoader(
        test_set, batch_size=1, num_workers=num_workers,
        collate_fn=batch_preparation_img2seq,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    preds_full, gts_full = [], []
    preds_kern, gts_kern = [], []
    page_chord_metrics = []

    with torch.no_grad():
        for batch in loader:
            x, _, y, _ = batch
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

                preds_full.append(pred_text)
                gts_full.append(gt_text)
                preds_kern.append(kern_spine_only(pred_text))
                gts_kern.append(kern_spine_only(gt_text))

                pred_chord_toks = mxhm_tokens(pred_text)
                gt_chord_toks   = mxhm_tokens(gt_text)
                page_chord_metrics.append(
                    compute_page_chord_metrics(pred_chord_toks, gt_chord_toks)
                )

    cer, ser, ler             = compute_poliphony_metrics(preds_full, gts_full)
    kern_cer, kern_ser, kern_ler = compute_poliphony_metrics(preds_kern, gts_kern)
    agg = aggregate_page_chord_metrics(page_chord_metrics)

    return {
        "cer":       cer,
        "ser":       ser,
        "ler":       ler,
        "chord_ser": agg.get("agg_ser_no_dots", float("nan")),
        "root_ser":  agg.get("agg_root_ser",  float("nan")),
        "kern_cer":  kern_cer,
        "kern_ser":  kern_ser,
        "kern_ler":  kern_ler,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main(
    data_path:    str = "/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmuss_Data/jazzmus_fullpage",
    fold:         int = 0,
    final_stage:  int = 9,
    num_workers:  int = 4,
):
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

    results = []
    for name, path in CHECKPOINTS:
        if not path.exists():
            continue
        print(f"=== {name} ===")
        m = run_inference(str(path), data_path, fold, final_stage, num_workers)
        results.append((name, m))
        print(
            f"  Overall  CER={m['cer']:.2f}%  SER={m['ser']:.2f}%  LER={m['ler']:.2f}%\n"
            f"  Chord    SER={m['chord_ser']:.2f}%  Root SER={m['root_ser']:.2f}%\n"
            f"  Kern     CER={m['kern_cer']:.2f}%  SER={m['kern_ser']:.2f}%  LER={m['kern_ler']:.2f}%\n"
        )

    # ── summary table ─────────────────────────────────────────────────────────
    cols = ["CER", "SER", "LER", "ChSER", "RtSER", "KCER", "KSER", "KLER"]
    keys = ["cer", "ser", "ler", "chord_ser", "root_ser", "kern_cer", "kern_ser", "kern_ler"]
    header_width = 15 + 9 * len(cols)
    print("\n" + "=" * header_width)
    print(f"{'Experiment':<15}" + "".join(f"{c:>9}" for c in cols))
    print("-" * header_width)
    for name, m in results:
        row = f"{name:<15}" + "".join(f"{m[k]:>8.2f}%" for k in keys)
        print(row)
    print("=" * header_width)


if __name__ == "__main__":
    fire.Fire(main)
