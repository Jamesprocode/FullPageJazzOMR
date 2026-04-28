"""
Test a full-page checkpoint on jazzmus_fullpage/splits/test_<fold>.txt.

Reports both overall CER/SER/LER and kern-spine-only CER/SER/LER.

Usage:
    python test_fullpage.py \\
        --checkpoint /path/to/fullpage_ft_fold0_best.ckpt \\
        --data_path  /home/ubuntu/FullPageJazzOMR/data/jazzmus_fullpage \\
        --fold 0 \\
        --stage 9
"""

import sys
from pathlib import Path

import fire
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from datasets.full_page_dataset import FullPageDataset
from jazzmus.curriculum.trainer import CurriculumSMTTrainer
from jazzmus.dataset.smt_dataset import batch_preparation_img2seq
from jazzmus.dataset.tokenizer import untokenize
from jazzmus.dataset.eval_functions import compute_poliphony_metrics


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


def test(
    checkpoint:   str,
    data_path:    str = "/home/ubuntu/FullPageJazzOMR/data/jazzmus_fullpage",
    fold:         int = 0,
    stage:        int = 9,
    num_workers:  int = 4,
):
    torch.set_float32_matmul_precision("high")

    print(f"Loading checkpoint: {checkpoint}")
    model = CurriculumSMTTrainer.load_from_checkpoint(
        checkpoint, load_pretrained=False, strict=False,
    )
    model.set_stage(stage)
    model.freeze()
    model.eval()

    w2i = model.model.w2i
    i2w = model.model.i2w
    print(f"Vocab size: {len(w2i)}")

    test_set = FullPageDataset(data_path=data_path, split="test", fold=fold)
    test_set.set_dictionaries(w2i, i2w)
    print(f"Test samples: {len(test_set)}")

    seq_maxlen = int(max(len(t) for t in test_set.gt_tokens) * 1.1)
    if seq_maxlen > model.model.maxlen:
        model.model.maxlen = seq_maxlen

    test_loader = DataLoader(
        test_set, batch_size=1, num_workers=num_workers,
        collate_fn=batch_preparation_img2seq,
        persistent_workers=(num_workers > 0),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    preds_full, gts_full = [], []
    preds_kern, gts_kern = [], []

    with torch.no_grad():
        for batch in test_loader:
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

    cer, ser, ler = compute_poliphony_metrics(preds_full, gts_full)
    cer_k, ser_k, ler_k = compute_poliphony_metrics(preds_kern, gts_kern)

    print("\n" + "=" * 55)
    print(f"{'Metric':<15} {'Overall':>12} {'Kern-only':>12}")
    print("-" * 55)
    print(f"{'CER':<15} {cer:>11.2f}% {cer_k:>11.2f}%")
    print(f"{'SER':<15} {ser:>11.2f}% {ser_k:>11.2f}%")
    print(f"{'LER':<15} {ler:>11.2f}% {ler_k:>11.2f}%")
    print("=" * 55)


if __name__ == "__main__":
    fire.Fire(test)
