# FullPageJazzOMR

End-to-end Optical Music Recognition for **handwritten jazz lead sheets**, transcribing complete page images directly into Humdrum `**kern` + `**mxhm` notation. The system extends the Sheet Music Transformer (SMT) from staff-level to full-page input via curriculum learning, and is benchmarked against a YOLO-based detect-and-concatenate baseline.

This repository accompanies the ISMIR 2026 submission *"Full-Page Optical Music Recognition of Jazz Lead Sheets"* (anonymized for double-blind review).

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

## Methodology

### Data

The dataset is [JazzMus](https://huggingface.co/datasets/PRAIG/JAZZMUS) (163 jazz standards, 326 synthetic full pages + 293 handwritten full pages, with Humdrum and MusicXML annotations). We contributed annotation fixes to the upstream repository — see Section 3.1.1 of the paper.

### Detect-and-concatenate baseline

The baseline serves as the staff-level comparison system for full-page training and follows three steps (paper §3.2.1):

1. **Pre-processing.** Page images are resized and de-skewed.
2. **Staff segmentation (YOLOv11s).** A music-specific YOLOv11s detector predicts per-staff bounding boxes. We add three post-processing fixes on top of the raw detections:
   - **Missed staves** are recovered by inserting a virtual box with median dimensions whenever the gap between consecutive boxes exceeds the page's median staff distance.
   - **Duplicate detections** of the same staff are detected via 1-D vertical IoU and the smaller box is dropped.
   - **Vertically narrow boxes** are expanded by allocating 25% / 75% of each inter-staff gap to the upper / lower crop, so chord symbols above the staff lines are included.
3. **Staff-level recognition + page assembly.** Each cropped staff is passed through a staff-level SMT checkpoint (the same one we initialize the curriculum models from). The first staff's Humdrum header and spine declarations are kept; subsequent staves have their headers stripped and are joined with `!!linebreak:original` separators, with spine terminators appended at the end.

### Curriculum training (proposed model)

To bridge the gap between the staff-level pretrained checkpoint and full-page recognition, we extend the SMT vocabulary with a `<linebreak>` token and progressively expose the model to longer, more structurally complex inputs through a 9-stage curriculum (paper §3.2.2). Two strategies are compared:

**Masking curriculum (best-performing).** For each full-page scan, we generate top-`N` crops by retaining only the top `N` staves and cropping away the rest. At stage `k`, the training/validation set contains only `k`-staff samples. Once a sample reaches its maximum number of staves, it remains as a full-page sample in all subsequent stages. Pages with 9+ staves are grouped into the final stage, yielding 1,688 training and 105 validation samples; the test set is 32 unmodified full pages.

A side-effect of staged advancement is that earlier-stage samples disappear once the model moves on, risking forgetting of shorter-input behavior. To mitigate this, we apply **experience replay** with ratio `r ∈ {0, 0.25, 0.5, 1.0}`, retaining a fraction of earlier-stage samples in the training pool at each stage. Higher `r` consistently improves performance, with `r = 1.00` (full replay) winning across all metrics.

**Stacking curriculum.** Instead of masking real full-page scans, multi-staff samples are synthesized at training time by sampling individual staff crops, resizing them to a uniform height, and vertically concatenating them into a multi-staff image resembling a full-page lead sheet. The corresponding ground truth is formed by concatenating the per-staff annotations with `<linebreak>` tokens. At the final stage, real full-page samples are appended to the training pool to expose the model to authentic page-level layout. Despite producing far more training data, the stacking curriculum underperforms the baseline — likely because stacked samples differ visually from real lead-sheet layouts (inconsistent staff lengths, discontinuous key/chord progressions across staves).

Both curricula advance to the next stage once validation SER falls below 20% for 3 consecutive validation epochs, and the final stage trains until early-stopping (patience 10).




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

**Overall metrics** (paper Table 1):

| Model | CER | SER | LER |
|-------|-----|-----|-----|
| Baseline (YOLO + staff-SMT)  | 13.55 | 12.78 | 32.35 |
| Masking, r=0                 | 17.61 | 16.80 | 31.41 |
| Masking, r=0.25              | 17.48 | 16.73 | 29.13 |
| Masking, r=0.50              | 14.11 | 13.59 | 26.25 |
| **Masking, r=1.00**          | **13.32** | **12.50** | **23.15**\* |
| Stacking, r=0.25             | 24.30 | 22.90 | 44.55 |

**Spine-specific metrics** for the best model vs. baseline (paper Table 2):

| Metric | Baseline | Masking, r=1.00 | Δ |
|--------|----------|-----------------|---|
| CER_Melo  | 11.96 | **11.63**       | -0.33 |
| SER_Melo  | 14.40 | **13.02**       | -1.38 |
| LER_Melo  | 26.66 | **18.67**\*     | -7.99 |
| SER_Root  | 29.17 | **21.25**\*     | -7.92 |
| SER_Chord | 47.82 | **30.67**\*\*   | -17.15 |

Stars: paired Wilcoxon (one-sided, Holm-corrected across all reported metrics) — `*` p<0.01, `**` p<0.001 vs. baseline.

The largest gains come from **SER_Chord** (47.82 → 30.67) and **LER** / **LER_Melo**, confirming that page-level context primarily helps chord-symbol transcription and structural / line-level alignment rather than fine-grained note-level accuracy.



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
└── plot_per_n.py                    # error rate vs. staves-per-page (Fig. in paper)
```

## Getting started

### Installation

```bash
git clone <repository-url>
cd FullPageJazzOMR
conda env create -f environment.yml
conda activate jazzmus
```


