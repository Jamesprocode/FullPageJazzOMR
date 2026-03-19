# FullPageJazzOMR

End-to-end Optical Music Recognition for full-page jazz sheet music using curriculum learning. The system progressively trains a neural model from single-system crops to full multi-system pages, transcribing jazz scores into symbolic notation (kern format).

## Architecture

FullPageJazzOMR uses a **ConvNext encoder + Transformer decoder** architecture with a 9-stage curriculum learning strategy:

```
Input: Full-page jazz music image (grayscale)
         │
         ▼
┌─────────────────────────┐
│   ConvNext Encoder      │   3 stages, hidden sizes [64, 128, 256]
│   (3 stages, depths     │   depths [3, 3, 9]
│    [3, 3, 9])           │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  2D Positional Encoding │   Sinusoidal spatial encoding
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Transformer Decoder    │   8 layers, 4 attention heads
│  (cross-attention to    │   d_model = 256
│   encoder features)     │
└────────┬────────────────┘
         │
         ▼
   Output: Symbolic music notation (kern)
```

### Curriculum Learning

Training proceeds through 9 stages, each adding one more system to the input crop:

| Stage | Input | Description |
|-------|-------|-------------|
| 1 | 1-system crop | Single staff system |
| 2 | 2-system crop | Two adjacent systems |
| ... | ... | ... |
| 9 | Full page | Up to 9 systems (full page) |

Stage advancement is automatic: the model advances when validation Symbol Error Rate (SER) drops below a threshold for a set number of consecutive epochs. An optional **experience replay** mechanism (50% replay ratio) prevents catastrophic forgetting of earlier stages.

## Project Structure

```
FullPageJazzOMR/
├── config/                      # Gin configuration files
│   ├── pagecrop_9stage.gin      # 9-stage curriculum (no replay)
│   └── pagecrop_9stage_replay.gin  # 9-stage with 50% replay
├── data_prep/                   # Offline data preparation
│   ├── prepare_pagecrop.py      # Pre-compute N-system page crops
│   └── prepare_synthetic.py     # Generate synthetic training data
├── datasets/
│   └── page_crop_dataset.py     # Curriculum-aware dataset
├── jazzmus/                     # Core library
│   ├── curriculum/
│   │   └── trainer.py           # CurriculumSMTTrainer
│   ├── dataset/
│   │   ├── eval_functions.py    # SER, CER, LER metrics
│   │   └── chord_metrics.py     # Chord-specific evaluation (MIREX)
│   ├── model/
│   │   └── smt/
│   │       ├── modeling_smt.py      # SMTModelForCausalLM
│   │       └── configuration_smt.py # SMTConfig
│   └── smt_trainer.py           # Base SMT training module
├── vocab/                       # Vocabulary mappings (w2i / i2w)
├── weights/                     # Model checkpoints
├── train.py                     # Main training entry point
└── test.py                      # Evaluation entry point
```

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)

### Installation

```bash
git clone git@github.com:Jamesprocode/FullPageJazzOMR.git
cd FullPageJazzOMR
pip install -r requirements.txt
```

Key dependencies:

| Package | Version |
|---------|---------|
| PyTorch | 2.6.0 |
| PyTorch Lightning | 2.5.5 |
| Transformers | 4.57.1 |
| gin-config | 0.5.0 |
| WandB | 0.22.2 |
| OpenCV | 4.12.0 |
| music21 | 9.9.1 |

### Data Preparation

Pre-compute N-system page crops from full-page data:

```bash
python data_prep/prepare_pagecrop.py \
    --jazzmus_fullpage ../ISMIR-Jazzmus/data/jazzmus_fullpage \
    --jazzmus_parquet ../JAZZMUS/data/train-00000-of-00001.parquet \
    --out_dir data/jazzmus_pagecrop \
    --folds 0 \
    --max_n 9 \
    --bottom_pad 20
```

Expected data layout:

```
jazzmus_pagecrop/
├── jpg/          # img_<X>_n<N>.jpg
├── gt/           # img_<X>_n<N>.txt (kern format)
└── splits/       # train_0.txt, val_0.txt, test_0.txt
```

### Training

Train with curriculum learning from a Phase-1 system-level checkpoint:

```bash
python train.py \
    --config config/pagecrop_9stage.gin \
    --checkpoint ../ISMIR-Jazzmus/weights/smt/smt_0.ckpt \
    --fold 0
```

With experience replay to reduce forgetting:

```bash
python train.py \
    --config config/pagecrop_9stage_replay.gin \
    --checkpoint ../ISMIR-Jazzmus/weights/smt/smt_0.ckpt \
    --fold 0
```

Override hyperparameters via CLI:

```bash
python train.py \
    --config config/pagecrop_9stage.gin \
    --checkpoint ../ISMIR-Jazzmus/weights/smt/smt_0.ckpt \
    --lr 1e-5 \
    --batch_size 4 \
    --accumulate_grad_batches 32 \
    --num_workers 8 \
    --fold 0
```

Resume from a checkpoint:

```bash
python train.py \
    --config config/pagecrop_9stage.gin \
    --resume weights/pagecrop/pagecrop_fold0_last.ckpt \
    --resume_stage 5 \
    --fold 0
```

### Evaluation

```bash
python test.py \
    --checkpoint weights/pagecrop/pagecrop_fold0_best.ckpt \
    --data_path data/jazzmus_pagecrop \
    --fold 0 \
    --final_stage 9
```

## Evaluation Metrics

### Transcription Metrics

| Metric | Level | Description |
|--------|-------|-------------|
| **SER** | Symbol | Symbol Error Rate — edit distance over ground truth tokens |
| **CER** | Character | Character Error Rate — character-level edit distance |
| **LER** | Line | Line Error Rate — system/line-level accuracy |

### Chord Metrics (MIREX-style)

| Metric | Description |
|--------|-------------|
| **CSR** | Chord Symbol Recall — duration-weighted chord accuracy |
| **Root F1** | Chord root detection accuracy |
| **Quality** | Major/minor/7th quality accuracy |
| **Full Match F1** | Exact full chord symbol match |

## Configuration

Training is configured via [gin-config](https://github.com/google/gin-config). Key parameters in the gin files:

```python
# Curriculum settings
PageCropDataset.num_cl_stages       = 9       # Number of curriculum stages
PageCropDataset.system_height       = 256     # Pixels per system row
PageCropDataset.replay_ratio        = 0.0     # Experience replay ratio (0.5 for replay config)

# Stage advancement
DynamicCurriculumAdvancer.ser_threshold  = 28.0   # Advance when val/SER < this
DynamicCurriculumAdvancer.patience       = 3      # Epochs below threshold to advance
DynamicCurriculumAdvancer.final_patience = 5      # Early stopping patience at final stage

# Training
train_hparams.lr                    = 1e-5
train_hparams.batch_size            = 4
train_hparams.accumulate_grad_batches = 1
train_hparams.num_workers           = 4
train_hparams.check_val_every_n_epoch = 10
```

## Experiment Tracking

Training logs are tracked with [Weights & Biases](https://wandb.ai/) under the project `fullpage-jazz-omr`, group `pagecrop_curriculum`.
