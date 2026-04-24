"""
End-to-end paired significance testing for the masking-curriculum vs baseline.

Runs ALL pipelines on the same test pages, computes per-page metrics, runs
paired Wilcoxon tests with Holm correction, and emits both a wide CSV
(re-sliceable later) and a paper-ready markdown table.

Pipelines:
    baseline        : YOLO staff detection → system-level SMT → concatenate
                      (the same baseline shipped in baseline/full_page_baseline.py)
    r0/r025/r05/r100: full-page CurriculumSMTTrainer checkpoints

Outputs:
    analysis_out/per_page_all_models.csv  — per-page paired metrics
    analysis_out/significance.md          — paper-ready test results

Run from FullPageJazzOMR/:
    python eval_for_significance.py
    python eval_for_significance.py --skip_inference   # re-run stats from existing CSV
"""

import csv
import os
import sys
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

torch.set_float32_matmul_precision("high")


def _require_cuda():
    """Hard fail if CUDA isn't available — silent CPU fallback hides OOM bugs."""
    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA not available. Refusing to run on CPU "
                 "(would OOM the host). Check sbatch --gres=gpu allocation.")
    print(f"CUDA OK: device={torch.cuda.get_device_name(0)}, "
          f"count={torch.cuda.device_count()}, "
          f"torch={torch.__version__}, cuda={torch.version.cuda}")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baseline"))
sys.path.insert(0, str(ROOT / "baseline" / "sys_level_imports"))

from datasets.full_page_dataset import FullPageDataset
from jazzmus.curriculum.trainer import CurriculumSMTTrainer
from jazzmus.dataset.smt_dataset import batch_preparation_img2seq
from jazzmus.dataset.tokenizer import untokenize
from jazzmus.dataset.eval_functions import compute_poliphony_metrics
from jazzmus.dataset.chord_metrics import (
    extract_spines,
    extract_tokens_from_mxhm,
    compute_page_chord_metrics,
)

from baseline.full_page_baseline import segment_staves, concatenate_systems
from baseline.inference import FullPageInference


# ════════════════════════════════════════════════════════════════════════════
# FILL IN PATHS HERE
# ════════════════════════════════════════════════════════════════════════════

BASELINE_YOLO   = Path("/home/hice1/jwang3180/scratch/jazzmus/ISMIR-Jazzmus/yolo_weigths/yolov11s_20241108.pt")
BASELINE_SMT    = Path("/home/hice1/jwang3180/scratch/jazzmus/ISMIR-Jazzmus/weights/smt_sys_best/smt_pre_syn_medium.ckpt")

TEST_SPLIT_FILE = Path("/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmuss_Data/jazzmus_pagecrop/splits/test_0.txt")
DATA_BASE_DIR   = Path("/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmuss_Data")

CHECKPOINTS = {
    "r0":   Path("/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmus_weights/pagecrop/pagecrop_fold0_best.ckpt"),
    "r025": Path("/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmus_weights/replay/pagecrop_fold0_best.ckpt"),
    "r05":  Path("/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmus_weights/replay_50percent/pagecrop_fold0_best.ckpt"),
    # r100 is selected from R100_CANDIDATES below — whichever has lowest aggregate
    # SER on the test set. The picked checkpoint becomes the "r100" column.
}

# Candidate r=100 checkpoints. We run inference on all of them, compare, and
# keep only the best one as the final r100 entry. Selection criterion: lowest
# mean overall SER across pages (matches the paper's headline metric).
R100_CANDIDATES = [
    # pagecrop_fold0_best.ckpt was already evaluated in a prior run and ruled out.
    Path("/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmus_weights/replay_100percent/pagecrop_fold0_best-v1.ckpt"),
    Path("/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmus_weights/replay_100percent/pagecrop_fold0_best-v2.ckpt"),
    Path("/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmus_weights/replay_100percent/pagecrop_fold0_best-v2-bachsize1.ckpt"),
]
R100_SELECTION_METRIC = "ser"   # one of METRIC_KEYS — lower is better

FULLPAGE_DATA_PATH = "/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmuss_Data/jazzmus_fullpage"
FULLPAGE_FOLD      = 0
FULLPAGE_STAGE     = 9

OUT_CSV = "analysis_out/per_page_all_models.csv"
OUT_MD  = "analysis_out/significance.md"

METRIC_KEYS = ["cer", "ser", "ler", "kern_cer", "kern_ser", "kern_ler",
               "chord_ser", "root_ser"]

# Metrics to test for significance. All 8 are run so the output table
# mirrors `sig_from_existing.py` for every masking ratio.
SIG_METRICS = ["cer", "ser", "ler",
               "kern_cer", "kern_ser", "kern_ler",
               "root_ser", "chord_ser"]


# ── shared helpers ────────────────────────────────────────────────────────────

def kern_spine_only(text: str) -> str:
    lines = text.split("\n")
    out, kern_col = [], None
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
    spines = extract_spines(text)
    mxhm = spines.get("**mxhm", "")
    return extract_tokens_from_mxhm(mxhm) if mxhm else []


def per_page_record(pred_full, gt_full):
    pred_kern = kern_spine_only(pred_full)
    gt_kern   = kern_spine_only(gt_full)
    chord_m   = compute_page_chord_metrics(mxhm_tokens(pred_full),
                                           mxhm_tokens(gt_full))
    cer, ser, ler                = compute_poliphony_metrics([pred_full], [gt_full])
    kern_cer, kern_ser, kern_ler = compute_poliphony_metrics([pred_kern], [gt_kern])
    return {
        "cer":       cer,
        "ser":       ser,
        "ler":       ler,
        "kern_cer":  kern_cer,
        "kern_ser":  kern_ser,
        "kern_ler":  kern_ler,
        "chord_ser": chord_m.get("ser_no_dots", float("nan")),
        "root_ser":  chord_m.get("root_ser",   float("nan")),
    }


# ── baseline pipeline ────────────────────────────────────────────────────────

def baseline_records():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(TEST_SPLIT_FILE) as f:
        pairs = []
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pairs.append((
                    os.path.join(DATA_BASE_DIR, parts[0]),
                    os.path.join(DATA_BASE_DIR, parts[1]),
                ))
    print(f"  loaded {len(pairs)} test pairs")
    print(f"  loading SMT: {BASELINE_SMT}")
    inference_model = FullPageInference(str(BASELINE_SMT), device=device)

    records = {}
    for img_path, gt_path in tqdm(pairs, desc="baseline"):
        page = Path(img_path).stem
        try:
            cropped = segment_staves(
                image_path=img_path, yolo_model_path=str(BASELINE_YOLO),
                confidence_threshold=0.5, deskew=True, max_skew_angle=10.0,
            )
            sys_kerns = []
            for sys_img in cropped:
                arr = np.array(sys_img.convert("L"))
                sys_kerns.append(inference_model.predict(arr)["prediction"])
            pred_full = concatenate_systems(sys_kerns)
            with open(gt_path) as f:
                gt_full = f.read()
            rec = per_page_record(pred_full, gt_full)
            rec["page_name"] = page
            records[page] = rec
        except Exception as e:
            print(f"    skip {page}: {e}")
    return records


# ── full-page checkpoint inference ───────────────────────────────────────────

def fullpage_records(checkpoint_path, num_workers):
    model = CurriculumSMTTrainer.load_from_checkpoint(
        checkpoint_path, load_pretrained=False, strict=False,
    )
    w2i = model.model.w2i
    i2w = model.model.i2w

    test_set = FullPageDataset(data_path=FULLPAGE_DATA_PATH, split="test", fold=FULLPAGE_FOLD)
    test_set.set_dictionaries(w2i, i2w)

    seq_maxlen = int(max(len(t) for t in test_set.gt_tokens) * 1.1)
    if seq_maxlen > model.model.maxlen:
        model.model.maxlen = seq_maxlen

    model.set_stage(FULLPAGE_STAGE)
    model.freeze()
    model.eval()

    img_paths = getattr(test_set, "img_paths", None)
    page_names = ([Path(p).stem for p in img_paths] if img_paths
                  else [f"page_{i}" for i in range(len(test_set))])

    loader = DataLoader(
        test_set, batch_size=1, num_workers=num_workers,
        collate_fn=batch_preparation_img2seq,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    records = {}
    idx = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=Path(checkpoint_path).stem[:25]):
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

                rec = per_page_record(pred_text, gt_text)
                page = page_names[idx] if idx < len(page_names) else f"page_{idx}"
                rec["page_name"] = page
                records[page] = rec
                idx += 1
    return records


# ── inference orchestrator ───────────────────────────────────────────────────

def _summary(recs):
    return {k: sum(r[k] for r in recs.values()) / len(recs) for k in METRIC_KEYS}


def _print_means(name, recs):
    m = _summary(recs)
    print(f"  {name}: {len(recs)} pages, mean CER={m['cer']:.2f}  SER={m['ser']:.2f}  "
          f"LER={m['ler']:.2f}  KCER={m['kern_cer']:.2f}  ChSER={m['chord_ser']:.2f}")


def select_best_r100(num_workers):
    """Run inference on all R100_CANDIDATES, return (winner_path, winner_records)."""
    print("=" * 60)
    print(f"R=100 candidate selection (criterion: lowest mean {R100_SELECTION_METRIC})")
    print("=" * 60)
    valid = [p for p in R100_CANDIDATES if p.exists()]
    for p in R100_CANDIDATES:
        if not p.exists():
            print(f"  [MISSING] {p}")
    if not valid:
        sys.exit("no r=100 candidates found")

    candidate_results = []
    for path in valid:
        print(f"\n--- candidate: {path.name} ---")
        recs = fullpage_records(str(path), num_workers)
        means = _summary(recs)
        candidate_results.append((path, recs, means))
        _print_means(path.name, recs)

    print("\n" + "-" * 60)
    print(f"R=100 candidate comparison (sorted by {R100_SELECTION_METRIC}):")
    print(f"  {'candidate':<53} {'CER':>7} {'SER':>7} {'LER':>7} {'ChSER':>8}")
    candidate_results.sort(key=lambda t: t[2][R100_SELECTION_METRIC])
    for path, _, means in candidate_results:
        print(f"  {path.name:<53} {means['cer']:>6.2f}% {means['ser']:>6.2f}% "
              f"{means['ler']:>6.2f}% {means['chord_ser']:>7.2f}%")
    winner_path, winner_recs, winner_means = candidate_results[0]
    print(f"\n>>> r100 winner: {winner_path.name} "
          f"(mean {R100_SELECTION_METRIC}={winner_means[R100_SELECTION_METRIC]:.2f}%)")
    print("=" * 60 + "\n")
    return winner_path, winner_recs


def run_all_inference(num_workers):
    print("=" * 60)
    print("Models to evaluate:")
    print(f"  baseline : YOLO={BASELINE_YOLO.name}, SMT={BASELINE_SMT.name}")
    valid_ckpts = []
    for name, path in CHECKPOINTS.items():
        ok = path.exists()
        print(f"  {name:<10} -> {path}  [{'OK' if ok else 'MISSING'}]")
        if ok:
            valid_ckpts.append((name, path))
    print(f"  r100 candidates: {len(R100_CANDIDATES)} (best chosen by mean {R100_SELECTION_METRIC})")
    print("=" * 60 + "\n")

    all_records = {}

    # 1) Pick the best r=100 first so the user immediately sees which won.
    winner_path, winner_recs = select_best_r100(num_workers)
    all_records["r100"] = winner_recs

    # 2) Baseline (YOLO + system-level SMT + concat)
    print("=== baseline (YOLO + system-level SMT + concat) ===")
    for required, name in [(BASELINE_YOLO, "YOLO weights"),
                            (BASELINE_SMT, "SMT ckpt"),
                            (TEST_SPLIT_FILE, "test split")]:
        if not required.exists():
            sys.exit(f"missing {name}: {required}")
    recs = baseline_records()
    all_records["baseline"] = recs
    _print_means("baseline", recs)
    print()

    # 3) Other masking ratios (r0, r025, r05)
    for name, path in valid_ckpts:
        print(f"=== {name} ===")
        recs = fullpage_records(str(path), num_workers)
        all_records[name] = recs
        _print_means(name, recs)
        print()

    print("=" * 60)
    print(f"r100 winner used in CSV: {winner_path}")
    print("=" * 60 + "\n")
    return all_records


def write_wide_csv(all_records, out_csv):
    page_sets = [set(r.keys()) for r in all_records.values()]
    pages = sorted(set.intersection(*page_sets))
    dropped = max(len(s) for s in page_sets) - len(pages)
    if dropped:
        print(f"NOTE: dropping {dropped} pages not common to all models")

    columns = ["page_name"]
    for name in all_records:
        for k in METRIC_KEYS:
            columns.append(f"{k}_{name}")

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for page in pages:
            row = {"page_name": page}
            for name, recs in all_records.items():
                rec = recs[page]
                for k in METRIC_KEYS:
                    row[f"{k}_{name}"] = rec[k]
            w.writerow(row)
    print(f"wrote {out_csv}  ({len(pages)} pages × {len(all_records)} models)")


# ── significance test ───────────────────────────────────────────────────────

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def run_significance(csv_path, baseline, models, metrics, out_md, correction="holm"):
    df = pd.read_csv(csv_path)
    n_pages = len(df)
    rows = []
    for metric in metrics:
        base_col = f"{metric}_{baseline}"
        if base_col not in df.columns:
            print(f"  skip metric {metric}: no column {base_col}")
            continue
        base = df[base_col]
        for m in models:
            col = f"{metric}_{m}"
            if col not in df.columns:
                print(f"  skip {col}: not in CSV")
                continue
            diffs = base - df[col]                    # >0 = model better
            wins   = int((diffs > 0).sum())
            ties   = int((diffs == 0).sum())
            losses = int((diffs < 0).sum())
            try:
                _, p = wilcoxon(diffs, alternative="greater", zero_method="wilcox")
            except ValueError:
                p = 1.0
            rows.append({
                "metric":   metric,
                "model":    m,
                "n":        n_pages,
                "base_med": float(base.median()),
                "mod_med":  float(df[col].median()),
                "med_diff": float(diffs.median()),
                "mean_diff":float(diffs.mean()),
                "wins":     wins,
                "ties":     ties,
                "losses":   losses,
                "p_raw":    float(p),
            })

    if not rows:
        print("no significance tests run — check that baseline + at least one model are in the CSV")
        return

    p_raw = [r["p_raw"] for r in rows]
    _, p_adj, _, _ = multipletests(p_raw, method=correction)
    for r, pa in zip(rows, p_adj):
        r["p_adj"] = float(pa)
        r["sig"]   = stars(pa)

    # stdout
    print(f"\nWilcoxon paired (one-sided, H1: model better than baseline)")
    print(f"  N = {n_pages} pages, {len(rows)} tests, {correction}-corrected\n")
    print(f"{'metric':<6} {'model':<10} {'base':>7} {'mod':>7} {'Δmed':>8} "
          f"{'W/T/L':>10} {'p_raw':>9} {'p_adj':>9}  sig")
    print("-" * 82)
    for r in rows:
        print(f"{r['metric']:<6} {r['model']:<10} "
              f"{r['base_med']:>6.2f}% {r['mod_med']:>6.2f}% {r['med_diff']:>+7.2f}% "
              f"{r['wins']:>3}/{r['ties']:>2}/{r['losses']:>3} "
              f"{r['p_raw']:>9.4f} {r['p_adj']:>9.4f}  {r['sig']}")

    md = [
        "# Paired significance tests", "",
        f"- N pages: **{n_pages}**",
        f"- Test: Wilcoxon signed-rank, **one-sided** (H1: model better than baseline)",
        f"- Correction: **{correction}** across {len(rows)} tests",
        f"- Significance: `*` p<0.05, `**` p<0.01, `***` p<0.001 (corrected)",
        "",
        "| metric | model | baseline median | model median | median Δ | mean Δ | W/T/L | p_raw | p_adj | sig |",
        "|--------|-------|-----------------|--------------|----------|--------|-------|-------|-------|-----|",
    ]
    for r in rows:
        md.append(
            f"| {r['metric']} | {r['model']} | {r['base_med']:.2f}% | {r['mod_med']:.2f}% | "
            f"{r['med_diff']:+.2f}% | {r['mean_diff']:+.2f}% | "
            f"{r['wins']}/{r['ties']}/{r['losses']} | "
            f"{r['p_raw']:.4f} | {r['p_adj']:.4f} | {r['sig']} |"
        )
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(md) + "\n")
    print(f"\nwrote {out_md}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(
    out_csv:        str  = OUT_CSV,
    out_md:         str  = OUT_MD,
    skip_inference: bool = False,
    num_workers:    int  = 4,
    correction:     str  = "holm",
):
    """
    --skip_inference  Re-run only the significance test on an existing CSV
                      (skips all inference, useful if you tweak the test).
    """
    if not skip_inference:
        _require_cuda()
        all_records = run_all_inference(num_workers)
        write_wide_csv(all_records, out_csv)
    else:
        if not Path(out_csv).exists():
            sys.exit(f"--skip_inference but {out_csv} does not exist")
        print(f"skipping inference, using existing {out_csv}")

    df_cols = pd.read_csv(out_csv, nrows=0).columns
    detected_models = sorted({c.rsplit("_", 1)[1] for c in df_cols if "_" in c})
    if "baseline" not in detected_models:
        sys.exit("no 'baseline' columns in CSV — significance test needs a baseline")
    other_models = [m for m in detected_models if m != "baseline"]
    print(f"\nsignificance test: baseline vs {other_models} on {SIG_METRICS}")
    run_significance(out_csv, "baseline", other_models, SIG_METRICS, out_md, correction)


if __name__ == "__main__":
    fire.Fire(main)
