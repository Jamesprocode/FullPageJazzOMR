"""
Full-page curriculum training — page-crop strategy.

Trains from the Phase-1 system-level checkpoint using pre-computed N-system
page crops. Curriculum progresses from N=1 to N=max_n over CL stages.

Run from the FullPageJazzOMR/ project root:
    python train.py \\
        --config config/pagecrop_2stage.gin \\
        --checkpoint ../ISMIR-Jazzmus/weights/smt/smt_0.ckpt \\
        --fold 0

    # Override gin hparams on the CLI:
    python train.py --config ... --lr 1e-5 --accumulate_grad_batches 32
"""

import gc
import os
import sys
from pathlib import Path

import fire
import gin
import torch
from torch.nn import Conv1d

import lightning.pytorch as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from datasets.page_crop_dataset import PageCropDataset
from jazzmus.curriculum.trainer import CurriculumSMTTrainer
from jazzmus.dataset.smt_dataset import batch_preparation_img2seq
from jazzmus.dataset.smt_dataset_utils import check_and_retrieveVocabulary


# ── gin-configurable hyperparameter holder ─────────────────────────────────────

@gin.configurable
def train_hparams(lr: float = 5e-5, accumulate_grad_batches: int = 64):
    """Gin-configurable holder for training hyperparameters.

    Set in the gin config as:
        train_hparams.lr                      = 5e-5
        train_hparams.accumulate_grad_batches = 64
    CLI args (--lr, --accumulate_grad_batches) override gin values.
    """
    return lr, accumulate_grad_batches


# ── epoch callback to advance curriculum stage ─────────────────────────────────

class EpochSetter(Callback):
    """Propagates the current epoch to train and val datasets before each epoch.

    Both datasets need the epoch so curriculum stage filtering works correctly
    at validation time (val is filtered to the same N<=stage as train).
    """

    def __init__(self, train_set: PageCropDataset, val_set: PageCropDataset):
        self.train_set = train_set
        self.val_set   = val_set

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self.train_set.set_epoch(trainer.current_epoch)
        self.val_set.set_epoch(trainer.current_epoch)


# ── main training function ─────────────────────────────────────────────────────

def train(
    config: str,
    checkpoint: str,
    fold: int = 0,
    data_path: str = "data/jazzmus_pagecrop",
    epochs: int = 10000,
    batch_size: int = 1,
    num_workers: int = 4,
    accumulate_grad_batches: int = None,   # read from gin; CLI overrides
    lr: float = None,                      # read from gin; CLI overrides
    debug: bool = False,
):
    gc.collect()
    torch.cuda.empty_cache()
    seed_everything(seed=42, workers=True)

    gin.parse_config_file(config)

    for folder in ("weights/pagecrop", "logs", "vocab"):
        os.makedirs(folder, exist_ok=True)

    # Resolve hyperparams: gin config is the default, CLI arg overrides
    gin_lr, gin_accum = train_hparams()
    if lr is None:
        lr = gin_lr
    if accumulate_grad_batches is None:
        accumulate_grad_batches = gin_accum

    print("PAGE-CROP CURRICULUM TRAINING")
    print(f"  Config     : {config}")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Data path  : {data_path}")
    print(f"  Fold       : {fold}")
    print(f"  LR         : {lr}")
    print(f"  Accum grad : {accumulate_grad_batches}")

    # ── datasets ───────────────────────────────────────────────────────────────
    train_set = PageCropDataset(data_path=data_path, split="train", fold=fold, augment=False)
    val_set   = PageCropDataset(data_path=data_path, split="val",   fold=fold, augment=False)
    test_set  = PageCropDataset(data_path=data_path, split="test",  fold=fold, augment=False)

    # Build vocabulary from train + val + test GTs
    w2i, i2w = check_and_retrieveVocabulary(
        [train_set.get_gt(), val_set.get_gt(), test_set.get_gt()],
        "vocab",
        "vocab_cl",
    )
    train_set.set_dictionaries(w2i, i2w)
    val_set.set_dictionaries(w2i, i2w)
    test_set.set_dictionaries(w2i, i2w)

    print(f"\n  Train samples : {len(train_set.samples)}")
    print(f"  Val samples   : {len(val_set.samples)}")
    print(f"  Test samples  : {len(test_set.samples)}")
    print(f"  Vocab size    : {train_set.vocab_size()}")

    # ── model sizing ───────────────────────────────────────────────────────────
    # get_max_hw scans every image — only run on train+val to avoid loading test
    print("\nComputing max H×W across train split (scanning images)…")
    train_h, train_w = train_set.get_max_hw()
    val_h,   val_w   = val_set.get_max_hw()
    max_height = max(train_h, val_h)
    max_width  = max(train_w, val_w)

    max_len = int(max(
        train_set.get_max_seqlen(),
        val_set.get_max_seqlen(),
    ) * 1.1)   # 10 % headroom

    print(f"  Max H×W    : {max_height} × {max_width}")
    print(f"  Max seqlen : {max_len} (with 10 % buffer)")

    # ── load checkpoint ────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {checkpoint}")
    model = CurriculumSMTTrainer.load_from_checkpoint(
        checkpoint,
        maxh=int(max_height),
        maxw=int(max_width),
        maxlen=int(max_len),
        out_categories=train_set.vocab_size(),
        padding_token=w2i["<pad>"],
        in_channels=1,
        w2i=w2i,
        i2w=i2w,
        lr=lr,
        fold=fold,
        load_pretrained=False,   # weights come from ckpt, not HuggingFace
        strict=False,
    )

    # Replace output head if vocab size changed (e.g. <linebreak> added)
    out_layer = model.model.decoder.out_layer
    expected_vocab = train_set.vocab_size()
    if out_layer.out_channels != expected_vocab:
        print(f"  Replacing output layer: {out_layer.out_channels} → {expected_vocab}")
        model.model.decoder.out_layer = Conv1d(
            out_layer.in_channels, expected_vocab, kernel_size=1
        )
    else:
        print(f"  Output layer OK: {expected_vocab} tokens")

    # Wire up curriculum stage tracking
    model.set_stage(train_set.curriculum_stage_beginning)
    model.set_stage_calculator(train_set.get_stage_calculator())

    # ── dataloaders ────────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=batch_preparation_img2seq,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=batch_preparation_img2seq,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=batch_preparation_img2seq,
        persistent_workers=(num_workers > 0),
    )

    # ── callbacks ──────────────────────────────────────────────────────────────
    num_cl_stages   = gin.query_parameter("PageCropDataset.num_cl_stages")
    increase_epochs = gin.query_parameter("PageCropDataset.increase_epochs")
    total_cl_epochs = num_cl_stages * increase_epochs

    best_ckpt = ModelCheckpoint(
        dirpath="weights/pagecrop",
        filename=f"pagecrop_fold{fold}_best",
        monitor="val/ser",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True,
        save_on_train_epoch_end=False,
    )
    stage_ckpt = ModelCheckpoint(
        dirpath="weights/pagecrop",
        filename=f"pagecrop_fold{fold}_stage{{curriculum/stage:.0f}}",
        monitor="curriculum/stage",
        mode="max",
        save_top_k=num_cl_stages,
        verbose=True,
        enable_version_counter=False,
    )
    lr_monitor   = LearningRateMonitor(logging_interval="step")
    epoch_setter = EpochSetter(train_set, val_set)

    # ── logger ─────────────────────────────────────────────────────────────────
    wandb_logger = WandbLogger(
        project="fullpage-jazz-omr",
        name=f"pagecrop_fold{fold}_lr{lr}",
        group="pagecrop_curriculum",
        log_model=False,
        save_dir="logs",
    )

    # ── trainer ────────────────────────────────────────────────────────────────
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[best_ckpt, stage_ckpt, lr_monitor, epoch_setter],
        max_epochs=epochs,
        min_epochs=total_cl_epochs,
        precision="bf16-mixed",
        accelerator="auto",
        accumulate_grad_batches=accumulate_grad_batches,
        fast_dev_run=debug,
        deterministic=False,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # ── test with best checkpoint ──────────────────────────────────────────────
    best_model = CurriculumSMTTrainer.load_from_checkpoint(
        best_ckpt.best_model_path,
        strict=False,
    )
    best_model.freeze()
    best_model.eval()
    trainer.test(best_model, dataloaders=test_loader)


if __name__ == "__main__":
    fire.Fire(train)
