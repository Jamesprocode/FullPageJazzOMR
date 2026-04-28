"""
Per-sample comparative analysis: detect-and-concatenate baseline vs.
masking-curriculum (replay r=1.00) on the full-page test set.

For every page in jazzmus_pagecrop test split:
  1. Run the YOLO + system-level SMT baseline -> full-page kern prediction.
  2. Run the curriculum SMT (r=1.00 checkpoint) -> full-page kern prediction.
  3. Compute the five paper metrics on each prediction:
       Kern CER, Kern SER, Kern LER, Root SER, Chord SER

Outputs (under --out_dir):
  per_sample_metrics.csv   – per-page metrics for both methods + deltas
  preds/<page>__baseline.kern  – baseline prediction text
  preds/<page>__replay100.kern – curriculum prediction text
  preds/<page>__gt.kern        – ground-truth (for convenience)
  summary.txt              – aggregate metrics, top-K interesting pages

Run:
  python per_sample_analysis.py \
      --data_path /path/to/jazzmus_pagecrop \
      --replay_ckpt /path/to/replay_100percent/pagecrop_fold0_best.ckpt \
      --baseline_ckpt /path/to/smt_pre_syn_medium.ckpt \
      --yolo_path /path/to/yolov11s_20241108.pt \
      --out_dir analysis_out
"""

import csv
import os
import sys
from pathlib import Path

import fire
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

torch.set_float32_matmul_precision("high")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "baseline"))

from datasets.page_crop_dataset import PageCropDataset
from jazzmus.curriculum.trainer import CurriculumSMTTrainer
from jazzmus.dataset.tokenizer import untokenize
from jazzmus.dataset.eval_functions import compute_poliphony_metrics
from jazzmus.dataset.chord_metrics import (
    extract_spines,
    extract_tokens_from_mxhm,
    compute_page_chord_metrics,
)

from baseline.full_page_baseline import segment_staves, concatenate_systems
from baseline.inference import FullPageInference


# ── helpers ───────────────────────────────────────────────────────────────────

def kern_spine_only(text: str) -> str:
    """Return only the **kern column from a humdrum string."""
    out, kern_col = [], None
    for line in text.split("\n"):
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


def five_metrics(pred_full: str, gt_full: str) -> dict:
    """Compute the 5 paper metrics on a single page."""
    pred_k = kern_spine_only(pred_full)
    gt_k   = kern_spine_only(gt_full)
    cer, ser, ler = compute_poliphony_metrics([pred_k], [gt_k])

    root_ser, chord_ser = float("nan"), float("nan")
    try:
        ps = extract_spines(pred_full)
        gs = extract_spines(gt_full)
        if "**mxhm" in ps and "**mxhm" in gs:
            pt = extract_tokens_from_mxhm(ps["**mxhm"])
            gt = extract_tokens_from_mxhm(gs["**mxhm"])
            if pt and gt:
                m = compute_page_chord_metrics(pt, gt)
                root_ser  = m["root_ser"]
                chord_ser = m["chord_ser_no_dots"]
    except Exception:
        pass

    return {
        "kern_cer":  cer,
        "kern_ser":  ser,
        "kern_ler":  ler,
        "root_ser":  root_ser,
        "chord_ser": chord_ser,
    }


# ── method runners ────────────────────────────────────────────────────────────

def predict_baseline(img_path: str, inference_model, yolo_path: str) -> str:
    crops = segment_staves(
        image_path=img_path,
        yolo_model_path=yolo_path,
        confidence_threshold=0.5,
        deskew=True,
        max_skew_angle=10.0,
    )
    sys_kerns = []
    for sys_img in crops:
        arr = np.array(sys_img.convert("L"))
        result = inference_model.predict(arr)
        sys_kerns.append(result["prediction"])
    return concatenate_systems(sys_kerns)


def predict_curriculum(test_set, model, device, w2i, i2w):
    """Yield (img_path, pred_full, gt_full) for every page in test_set."""
    model.eval()
    for idx in range(len(test_set)):
        x, di, y, img_path = test_set[idx]
        x_s = x.to(device)
        with torch.no_grad():
            pred_seq, _ = model.model.predict(input=x_s)
        pred_full = untokenize(pred_seq)

        gt_toks = [i2w[t.item()] for t in y]
        clean = []
        for t in gt_toks:
            if t in ("<eos>", "<pad>"):
                break
            clean.append(t)
        gt_full = untokenize(clean)
        yield img_path, pred_full, gt_full


# ── main ──────────────────────────────────────────────────────────────────────

def main(
    data_path:      str = "/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmuss_Data/jazzmus_pagecrop",
    replay_ckpt:    str = "/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmus_weights/replay_100percent/pagecrop_fold0_best.ckpt",
    baseline_ckpt:  str = "/home/hice1/jwang3180/scratch/jazzmus/ISMIR-Jazzmus/weights/smt_sys_best/smt_pre_syn_medium.ckpt",
    yolo_path:      str = "/home/hice1/jwang3180/scratch/jazzmus/ISMIR-Jazzmus/yolo_weigths/yolov11s_20241108.pt",
    fold:           int = 0,
    final_stage:    int = 9,
    system_height:  int = 256,
    out_dir:        str = "analysis_out",
):
    out = Path(out_dir)
    (out / "preds").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load curriculum model & test set ──────────────────────────────────────
    print(f"Loading curriculum checkpoint: {replay_ckpt}")
    cl_model = CurriculumSMTTrainer.load_from_checkpoint(
        replay_ckpt, load_pretrained=False, strict=False,
    ).to(device)
    cl_model.freeze()
    w2i, i2w = cl_model.model.w2i, cl_model.model.i2w

    test_set = PageCropDataset(
        data_path=data_path, split="test", fold=fold,
        augment=False, system_height=system_height,
    )
    test_set.set_dictionaries(w2i, i2w)
    test_set.set_stage_direct(final_stage)

    seq_maxlen = int(max(len(t) for t in test_set.gt_tokens) * 1.1)
    if seq_maxlen > cl_model.model.maxlen:
        cl_model.model.maxlen = seq_maxlen

    # ── load baseline model ───────────────────────────────────────────────────
    print(f"Loading baseline (system-level) checkpoint: {baseline_ckpt}")
    inference_model = FullPageInference(baseline_ckpt, device=str(device))

    # ── iterate test pages ────────────────────────────────────────────────────
    rows = []
    cl_iter = list(predict_curriculum(test_set, cl_model, device, w2i, i2w))
    print(f"Test pages: {len(cl_iter)}")

    for img_path, cl_pred, gt_full in tqdm(cl_iter, desc="pages"):
        page_id = Path(img_path).stem

        # baseline
        try:
            base_pred = predict_baseline(img_path, inference_model, yolo_path)
        except Exception as e:
            print(f"  baseline failed on {page_id}: {e}")
            base_pred = ""

        # metrics
        m_base = five_metrics(base_pred, gt_full) if base_pred else {
            k: float("nan") for k in ("kern_cer","kern_ser","kern_ler","root_ser","chord_ser")
        }
        m_cl   = five_metrics(cl_pred, gt_full)

        # save predictions
        (out / "preds" / f"{page_id}__baseline.kern").write_text(base_pred)
        (out / "preds" / f"{page_id}__replay100.kern").write_text(cl_pred)
        (out / "preds" / f"{page_id}__gt.kern").write_text(gt_full)

        row = {"page": page_id, "img": img_path}
        for k, v in m_base.items(): row[f"base_{k}"]    = v
        for k, v in m_cl.items():   row[f"replay_{k}"]  = v
        for k in m_base:
            row[f"delta_{k}"] = m_base[k] - m_cl[k]   # positive = replay better
        rows.append(row)

    # ── write CSV ─────────────────────────────────────────────────────────────
    if not rows:
        print("No rows produced — aborting.")
        return
    csv_path = out / "per_sample_metrics.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    # ── summary: aggregate + top-K interesting ────────────────────────────────
    def agg(prefix):
        return {
            k.replace(f"{prefix}_", ""): float(np.nanmean([r[k] for r in rows]))
            for k in fieldnames if k.startswith(prefix + "_") and k != f"{prefix}_img"
        }

    base_agg, cl_agg = agg("base"), agg("replay")
    lines = ["=== AGGREGATE (mean over pages) ==="]
    for k in ("kern_cer","kern_ser","kern_ler","root_ser","chord_ser"):
        lines.append(f"  {k:10s}  baseline={base_agg[k]:6.2f}   replay100={cl_agg[k]:6.2f}   "
                     f"Δ={base_agg[k]-cl_agg[k]:+6.2f}")

    for metric in ("chord_ser", "root_ser", "kern_ler"):
        ranked = sorted(rows, key=lambda r: r[f"delta_{metric}"], reverse=True)
        lines.append(f"\n--- top 5 pages where replay100 BEATS baseline by Δ{metric} ---")
        for r in ranked[:5]:
            lines.append(f"  {r['page']:40s}  base={r[f'base_{metric}']:6.2f}  "
                         f"replay={r[f'replay_{metric}']:6.2f}  Δ={r[f'delta_{metric}']:+6.2f}")
        lines.append(f"--- top 5 pages where baseline BEATS replay100 by Δ{metric} ---")
        for r in ranked[-5:][::-1]:
            lines.append(f"  {r['page']:40s}  base={r[f'base_{metric}']:6.2f}  "
                         f"replay={r[f'replay_{metric}']:6.2f}  Δ={r[f'delta_{metric}']:+6.2f}")

    summary = "\n".join(lines)
    (out / "summary.txt").write_text(summary)
    print("\n" + summary)
    print(f"\nDone. All artifacts in {out.resolve()}")


if __name__ == "__main__":
    fire.Fire(main)
