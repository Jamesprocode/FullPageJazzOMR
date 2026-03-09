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


# ── gin-configurable hyperparameter holders ────────────────────────────────────

@gin.configurable
def train_hparams(
    lr: float = 5e-5,
    accumulate_grad_batches: int = 64,
    batch_size: int = 1,
    num_workers: int = 4,
):
    """Gin-configurable training hyperparameters. CLI args override gin values."""
    return lr, accumulate_grad_batches, batch_size, num_workers


@gin.configurable
def train_paths(
    data_path:   str = "data/jazzmus_pagecrop",
    checkpoint:  str = "../ISMIR-Jazzmus/weights/smt/smt_0.ckpt",
    weights_dir: str = "weights/pagecrop",
):
    """Gin-configurable paths. CLI args override gin values."""
    return data_path, checkpoint, weights_dir


# ── dynamic curriculum callback ────────────────────────────────────────────────

@gin.configurable
class DynamicCurriculumAdvancer(Callback):
    """Advances curriculum stage when val/ser drops below a threshold.

    Args:
        train_set:      training dataset (passed at runtime, not from gin)
        val_set:        validation dataset (passed at runtime, not from gin)
        num_cl_stages:  total number of stages
        ser_threshold:  advance when val/ser < this value
        patience:       consecutive val epochs below threshold before advancing
    """

    def __init__(
        self,
        train_set:     PageCropDataset,
        val_set:       PageCropDataset,
        num_cl_stages: int   = 11,
        ser_threshold: float = 0.10,
        patience:      int   = 3,
    ):
        self.train_set     = train_set
        self.val_set       = val_set
        self.num_cl_stages = num_cl_stages
        self.ser_threshold = ser_threshold
        self.patience      = patience
        self._stage        = 1
        self._epochs_below = 0

        train_set.set_stage_direct(1)
        val_set.set_stage_direct(1)

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        pl_module.set_stage(self._stage)
        pl_module.set_stage_calculator(lambda epoch: self._stage)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if self._stage >= self.num_cl_stages:
            return
        val_ser = trainer.callback_metrics.get("val/ser")
        if val_ser is None:
            return
        if float(val_ser) < self.ser_threshold:
            self._epochs_below += 1
        else:
            self._epochs_below = 0
        if self._epochs_below >= self.patience:
            self._stage = min(self._stage + 1, self.num_cl_stages)
            self._epochs_below = 0
            self.train_set.set_stage_direct(self._stage)
            self.val_set.set_stage_direct(self._stage)
            pl_module.set_stage(self._stage)
            print(f"\n  ── Curriculum → stage {self._stage}  "
                  f"(val/ser={float(val_ser):.4f} < {self.ser_threshold}) ──")


# ── main training function ─────────────────────────────────────────────────────

def train(
    config: str,
    fold: int = 0,
    epochs: int = 10000,
    accumulate_grad_batches: int = None,   # read from gin; CLI overrides
    batch_size: int = None,                # read from gin; CLI overrides
    num_workers: int = None,               # read from gin; CLI overrides
    lr: float = None,                      # read from gin; CLI overrides
    data_path: str = None,                 # read from gin; CLI overrides
    checkpoint: str = None,                # read from gin; CLI overrides
    weights_dir: str = None,               # read from gin; CLI overrides
    debug: bool = False,
):
    gc.collect()
    torch.cuda.empty_cache()
    seed_everything(seed=42, workers=True)

    gin.parse_config_file(config)

    # Resolve all params: gin config is the default, CLI arg overrides
    gin_lr, gin_accum, gin_bs, gin_nw = train_hparams()
    gin_data, gin_ckpt, gin_wts        = train_paths()
    if lr is None:
        lr = gin_lr
    if accumulate_grad_batches is None:
        accumulate_grad_batches = gin_accum
    if batch_size is None:
        batch_size = gin_bs
    if num_workers is None:
        num_workers = gin_nw
    if data_path is None:
        data_path = gin_data
    if checkpoint is None:
        checkpoint = gin_ckpt
    if weights_dir is None:
        weights_dir = gin_wts

    for folder in (weights_dir, "logs", "vocab"):
        os.makedirs(folder, exist_ok=True)

    print("PAGE-CROP CURRICULUM TRAINING")
    print(f"  Config      : {config}")
    print(f"  Checkpoint  : {checkpoint}")
    print(f"  Data path   : {data_path}")
    print(f"  Weights dir : {weights_dir}")
    print(f"  Fold        : {fold}")
    print(f"  LR          : {lr}")
    print(f"  Batch size  : {batch_size}  (effective: {batch_size * accumulate_grad_batches})")
    print(f"  Accum grad  : {accumulate_grad_batches}")
    print(f"  Num workers : {num_workers}")

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

    # Replace output head if vocab size changed, preserving weights for known tokens
    out_layer      = model.model.decoder.out_layer
    expected_vocab = train_set.vocab_size()
    if out_layer.out_channels != expected_vocab:
        print(f"  Resizing output layer: {out_layer.out_channels} → {expected_vocab}")
        old_weights = out_layer.weight.data          # (old_vocab, d_model, 1)
        old_w2i     = model.hparams.get("w2i") or {}
        new_layer   = Conv1d(out_layer.in_channels, expected_vocab, kernel_size=1)
        for token, new_idx in w2i.items():
            if token in old_w2i:
                old_idx = old_w2i[token]
                if old_idx < old_weights.shape[0]:
                    new_layer.weight.data[new_idx] = old_weights[old_idx]
        model.model.decoder.out_layer = new_layer
        print(f"  Preserved weights for {len(old_w2i)} known tokens; "
              f"new tokens randomly initialized")
    else:
        print(f"  Output layer OK: {expected_vocab} tokens")

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
    num_cl_stages = gin.query_parameter("DynamicCurriculumAdvancer.num_cl_stages")

    curriculum_cb = DynamicCurriculumAdvancer(train_set=train_set, val_set=val_set)

    best_ckpt = ModelCheckpoint(
        dirpath=weights_dir,
        filename=f"pagecrop_fold{fold}_best",
        monitor="val/ser",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True,
        save_on_train_epoch_end=False,
    )
    stage_ckpt = ModelCheckpoint(
        dirpath=weights_dir,
        filename=f"pagecrop_fold{fold}_stage{{curriculum/stage:.0f}}",
        monitor="curriculum/stage",
        mode="max",
        save_top_k=num_cl_stages,
        verbose=True,
        enable_version_counter=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

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
        callbacks=[best_ckpt, stage_ckpt, lr_monitor, curriculum_cb],
        max_epochs=epochs,
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
