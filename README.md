# FullPageJazzOMR

End-to-end Optical Music Recognition for **handwritten jazz lead sheets**, transcribing complete page images directly into Humdrum `**kern` + `**mxhm` notation. The system extends the Sheet Music Transformer (SMT) from staff-level to full-page input via curriculum learning, and is benchmarked against a YOLO-based detect-and-concatenate baseline.

This repository accompanies the ISMIR submission *"Full-Page Optical Music Recognition of Jazz Lead Sheets"* (Wang & Lerch, 2026).

## What's in this repo

- **Two curriculum strategies** for adapting a staff-level SMT checkpoint to full-page input:
  - **Masking curriculum** — progressively reveal more staves on real full-page scans, sweeping replay ratio `r ∈ {0, 0.25, 0.5, 1.0}`.
  - **Stacking curriculum** — synthesize multi-staff training samples by vertically concatenating staff crops.
- **Detect-and-concatenate baseline** — YOLOv11s staff segmentation + staff-level SMT recognition + page assembly. The comparison system for full-page training.
- **Chord-oriented evaluation metrics** — `SER_Root` and `SER_Chord` on the `**mxhm` spine, plus melody-only versions of CER/SER/LER on the `**kern` spine.
- **Significance + error-analysis pipeline** — paired Wilcoxon tests with Holm correction across all (model × metric) pairs, plus per-page and per-token error breakdowns.

## Architecture

The proposed model is the [Sheet Music Transformer](https://github.com/multiscore/smt) with one vocabulary change: a dedicated `<linebreak>` token to mark staff boundaries (vocab 154 → 155). The input embedding and output projection are extended by one position; the new entries are randomly initialized while the original 154 are loaded from the staff-level pretrained checkpoint.

```
Full-page handwritten lead sheet
            │
            ▼
   ConvNeXt encoder  ──►  2D positional encoding  ──►  Transformer decoder (cross-attn)
                                                                    │
                                                                    ▼
                                              Humdrum (**kern + **mxhm), <linebreak>-separated
```

Both curricula proceed through 9 stages, where stage `k` exposes the model to inputs containing exactly `k` staves. Stage advancement is dynamic: advance once validation SER < 20% for 3 consecutive validation epochs.

## Project structure

```
FullPageJazzOMR/
├── config/                          # gin configs
│   ├── pagecrop_9stage.gin          # masking curriculum, no replay (r=0)
│   ├── pagecrop_9stage_replay.gin   # masking curriculum, with replay
│   ├── stacked_9stage.gin           # stacking curriculum (online)
│   ├── stacked_precomputed_9stage.gin       # stacking, precomputed samples
│   └── fullpage_finetune.gin        # final-stage fine-tune on real pages only
├── data_prep/
│   └── prepare_pagecrop.py          # pre-compute top-N staff crops
├── datasets/
│   ├── page_crop_dataset.py         # masking-curriculum dataset
│   └── full_page_dataset.py         # full-page test-set wrapper
├── jazzmus/                         # core library
│   ├── curriculum/trainer.py        # CurriculumSMTTrainer + DynamicCurriculumAdvancer
│   ├── dataset/
│   │   ├── eval_functions.py        # CER / SER / LER
│   │   └── chord_metrics.py         # SER_Root, SER_Chord, alignment helpers
│   └── model/smt/                   # SMT model + config
├── baseline/
│   ├── full_page_baseline.py        # YOLO segmentation + concatenation
│   └── inference.py                 # FullPageInference (staff-level SMT runner)
├── train.py                         # masking-curriculum training entry point
├── train_stacked.py                 # stacking-curriculum training (online)
├── train_precomputed_stacking.py    # stacking-curriculum training (precomputed)
├── finetune_fullpage.py             # fine-tune final stage on real pages
├── prepare_stacked_data.py          # build stacked dataset for precomputed pipeline
├── test.py                          # legacy single-checkpoint eval
├── test_fullpage.py                 # full-page checkpoint eval (overall + kern-only)
├── eval_checkpoints.py              # evaluate a fixed list of checkpoints
├── eval_for_significance.py         # paired Wilcoxon pipeline (see below)
├── analyze_errors.py                # per-token error analysis (replay vs. baseline)
├── per_sample_analysis.py           # per-page metric dump
├── plot_per_n.py                    # error rate vs. staves-per-page (Fig. in paper)
└── sbatch/                          # PACE Slurm submission scripts
```

## Getting started

### Installation

```bash
git clone git@github.com:Jamesprocode/FullPageJazzOMR.git
cd FullPageJazzOMR
conda env create -f environment.yml
conda activate jazzmus
```

Key dependencies: PyTorch 2.6, PyTorch Lightning 2.5, Transformers 4.57, gin-config, music21, OpenCV, Ultralytics YOLO, scipy, statsmodels.

### Data

The dataset is [JazzMus](https://huggingface.co/datasets/PRAIG/JAZZMUS) (163 jazz standards, 326 synthetic full pages + 293 handwritten full pages, with Humdrum and MusicXML annotations). We contributed annotation fixes to the upstream repository — see Section 3.1.1 of the paper.

Pre-compute top-N crops for the masking curriculum:

```bash
python data_prep/prepare_pagecrop.py \
    --jazzmus_fullpage data/jazzmus_fullpage \
    --jazzmus_parquet  data/train-00000-of-00001.parquet \
    --out_dir          data/jazzmus_pagecrop \
    --folds 0 --max_n 9 --bottom_pad 20
```

Build the precomputed stacked dataset for the stacking curriculum:

```bash
python prepare_stacked_data.py
```

## Training

### Masking curriculum (proposed model)

```bash
# r = 0 (no replay)
python train.py --config config/pagecrop_9stage.gin \
    --checkpoint ../ISMIR-Jazzmus/weights/smt/smt_0.ckpt --fold 0

# r ∈ {0.25, 0.5, 1.0} — set replay_ratio in the gin file
python train.py --config config/pagecrop_9stage_replay.gin \
    --checkpoint ../ISMIR-Jazzmus/weights/smt/smt_0.ckpt --fold 0
```

Stages 1–8 train with batch size 8 at lr `5e-5`. Stage 9 drops to batch size 1 and lr `5e-6` due to memory pressure from full-page inputs.

### Stacking curriculum

```bash
python train_precomputed_stacking.py \
    --config config/stacked_precomputed_9stage.gin \
    --checkpoint ../ISMIR-Jazzmus/weights/smt/smt_0.ckpt --fold 0
```

A fixed replay ratio `r = 0.25` is used. Final stage (k = 9) appends real full-page samples to the pool.

## Evaluation

### Single checkpoint, full-page test set

```bash
python test_fullpage.py \
    --checkpoint weights/pagecrop/pagecrop_fold0_best.ckpt \
    --data_path data/jazzmus_fullpage --fold 0 --stage 9
```

Reports overall CER / SER / LER and `**kern`-only versions.

### Paired significance pipeline (paper Table 1)

[`eval_for_significance.py`](eval_for_significance.py) is a single-file pipeline that:

1. Runs inference for the **4 candidate r=100 checkpoints**, picks the one with lowest mean SER as the final `r100` entry.
2. Runs the **detect-and-concatenate baseline** (YOLO segmentation → staff-level SMT → page assembly).
3. Runs inference for the other masking checkpoints (`r0`, `r025`, `r05`).
4. Writes per-page metrics to a wide CSV (re-sliceable later).
5. Runs **paired Wilcoxon signed-rank tests** (one-sided, H1: model better than baseline) with **Holm correction** across all (model × metric) combinations and emits a paper-ready markdown table.

Submit via Slurm:

```bash
sbatch sbatch/submit_eval_for_significance.sh
```

Or re-run only the stats from an existing CSV:

```bash
python eval_for_significance.py --skip_inference
```

### Error analysis

```bash
python analyze_errors.py     # per-token replay vs. baseline error breakdown
python per_sample_analysis.py # per-page metric dump
python plot_per_n.py          # SER_Melo and SER_Root vs. staves-per-page
```

## Evaluation metrics

### Overall (full untokenized humdrum)

| Metric | Description |
|--------|-------------|
| **CER** | Character-level Levenshtein edit distance |
| **SER** | Symbol-level Levenshtein edit distance |
| **LER** | Line-level edit distance (any 1-char mismatch counts as a line substitution) |

### Spine-specific (paper Table 2)

Computed on the `**kern` spine only to isolate melodic accuracy:

| Metric | Description |
|--------|-------------|
| `CER_Melo`, `SER_Melo`, `LER_Melo` | CER / SER / LER on `**kern` only |

Computed on the `**mxhm` spine to isolate harmonic accuracy:

| Metric | Description |
|--------|-------------|
| `SER_Root`  | Levenshtein over chord-root tokens only — does the model recover the harmonic backbone? |
| `SER_Chord` | Levenshtein over full chord symbols (root + quality + extensions + inversion) |

## Headline results (32-page test set)

From paper Tables 1 and 2:

| Model | CER | SER | LER | SER_Root | SER_Chord |
|-------|-----|-----|-----|----------|-----------|
| Baseline (YOLO + staff-SMT)  | 13.55 | 12.78 | 32.35 | 29.17 | 47.82 |
| Masking, r=0                 | 17.61 | 16.80 | 31.41 |   —   |   —   |
| Masking, r=0.25              | 17.48 | 16.73 | 29.13 |   —   |   —   |
| Masking, r=0.50              | 14.11 | 13.59 | 26.25 |   —   |   —   |
| **Masking, r=1.00**          | **13.32** | **12.50** | **23.15** | **21.25** | **30.67** |
| Stacking, r=0.25             | 24.30 | 22.90 | 44.55 |   —   |   —   |

The largest gains come from **LER** (32.35 → 23.15) and **SER_Chord** (47.82 → 30.67), confirming that page-level context primarily helps structural alignment and chord-symbol transcription rather than note-level accuracy.

## Configuration

Training is configured via [gin-config](https://github.com/google/gin-config). Key knobs:

```python
# Curriculum
PageCropDataset.num_cl_stages   = 9
PageCropDataset.system_height   = 256
PageCropDataset.replay_ratio    = 1.0     # 0.0 / 0.25 / 0.5 / 1.0

# Stage advancement
DynamicCurriculumAdvancer.ser_threshold  = 20.0
DynamicCurriculumAdvancer.patience       = 3
DynamicCurriculumAdvancer.final_patience = 10

# Optimizer
train_hparams.lr                      = 5e-5     # 5e-6 at stage 9
train_hparams.batch_size              = 8        # 1 at stage 9
train_hparams.check_val_every_n_epoch = 10
```

## Experiment tracking

Runs are logged to [Weights & Biases](https://wandb.ai/) under the project `fullpage-jazz-omr`. All experiments use a single NVIDIA H200 GPU with BF16 mixed precision (PACE / Georgia Tech).
