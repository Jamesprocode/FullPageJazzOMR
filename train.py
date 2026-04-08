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
    val_batch_size: int = 1,
    num_workers: int = 4,
    check_val_every_n_epoch: int = 5,
):
    """Gin-configurable training hyperparameters. CLI args override gin values."""
    return lr, accumulate_grad_batches, batch_size, val_batch_size, num_workers, check_val_every_n_epoch


@gin.configurable
def train_paths(
    data_path:            str = "data/jazzmus_pagecrop",
    checkpoint:           str = "../ISMIR-Jazzmus/weights/smt/smt_0.ckpt",
    weights_dir:          str = "weights/pagecrop",
    synthetic_data_path:  str = None,   # e.g. "data/jazzmus_synthetic"
    resume:               str = None,   # path to last.ckpt to resume training
    resume_stage:         int = 1,      # curriculum stage to restore when resuming
    wandb_name:           str = None,   # custom wandb run name (default: pagecrop_fold{fold}_lr{lr})
):
    """Gin-configurable paths. CLI args override gin values."""
    return data_path, checkpoint, weights_dir, synthetic_data_path, resume, resume_stage, wandb_name


# ── dynamic curriculum callback ────────────────────────────────────────────────

@gin.configurable
class DynamicCurriculumAdvancer(Callback):
    """Advances curriculum stage when val/ser drops below a threshold.

    Before the final stage:  only the *last* checkpoint is kept (no best tracking).
    At the final stage:      activates the *best* checkpoint (tracked by val/ser)
                             and applies early stopping with `final_patience`.

    Args:
        train_set:      training dataset (passed at runtime, not from gin)
        val_set:        validation dataset (passed at runtime, not from gin)
        best_ckpt:      ModelCheckpoint to activate at the final stage (passed at runtime)
        num_cl_stages:  total number of stages
        ser_threshold:  advance when val/ser < this value
        patience:       consecutive val epochs below threshold before advancing
        final_patience: val epochs without improvement before stopping at final stage
    """

    def __init__(
        self,
        train_set:      PageCropDataset,
        val_set:        PageCropDataset,
        best_ckpt:      ModelCheckpoint,
        weights_dir:    str,
        fold:           int,
        num_cl_stages:  int   = 11,
        ser_threshold:  float = 0.10,
        patience:       int   = 3,
        final_patience: int   = 15,
    ):
        self.train_set      = train_set
        self.val_set        = val_set
        self.best_ckpt      = best_ckpt
        self.weights_dir    = weights_dir
        self.fold           = fold
        self.num_cl_stages  = num_cl_stages
        self.ser_threshold  = ser_threshold
        self.patience       = patience
        self.final_patience = final_patience
        self._stage           = 1
        self._epochs_below    = 0
        self._stage_best_ser  = float("inf")   # best val/ser seen in current stage
        self._stage_best_ckpt = None           # path to best-within-stage checkpoint
        self._at_final        = False
        self._best_ser        = float("inf")
        self._no_improve      = 0
        self._clear_ser       = False          # deferred pop flag

        train_set.set_stage_direct(1)
        val_set.set_stage_direct(1)

    def _activate_final_stage(self, trainer, pl_module, val_ser):
        """Switch to best-checkpoint mode and notify."""
        self._at_final = True
        self._best_ser = val_ser   # anchor to activation epoch, not float("inf")
        self.best_ckpt.save_top_k = 1   # was 0 (disabled) → now tracks best
        pl_module.set_stage(self._stage)
        print(f"\n  ── Curriculum → FINAL stage {self._stage}  "
              f"(val/ser={val_ser:.4f} < {self.ser_threshold}) ──")
        print(f"  ── Best-val/ser checkpoint now active; early-stop patience={self.final_patience} ──")

    def _run_full_sweep(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Validate on ALL num_cl_stages val sets and log per-stage SER to WandB.

        Called once at each stage advance (patience met) before moving to the next
        stage.  Results are logged as val/sweep/ser_stageN.
        Does not affect training state, Lightning metrics, or the main val/ser metric.
        """
        from jazzmus.dataset.eval_functions import compute_poliphony_metrics

        device = pl_module.device
        current_stage = self._stage
        print(f"\n  ── Full-sweep validation (all {self.num_cl_stages} stages) ──")

        from tqdm import tqdm

        pl_module.eval()
        original_maxlen = pl_module.model.maxlen
        sweep_results = {}
        try:
            with torch.no_grad():
                stage_bar = tqdm(range(1, self.num_cl_stages + 1),
                                 desc="  Full sweep", unit="stage", leave=True)
                for k in stage_bar:
                    self.val_set.set_stage_direct(k)
                    capped_maxlen = min(original_maxlen, max(512, k * 550))
                    pl_module.model.maxlen = capped_maxlen
                    loader = DataLoader(
                        self.val_set, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=batch_preparation_img2seq,
                    )
                    pl_module.preds = []
                    pl_module.grtrs = []
                    sample_bar = tqdm(loader, desc=f"    stage {k}", unit="sample",
                                      leave=False)
                    for batch in sample_bar:
                        x, di, y, paths = batch
                        batch_on_device = (
                            x.to(device), di.to(device), y.to(device), paths
                        )
                        pl_module.predict_output(batch_on_device)
                    preds = pl_module.preds[:]
                    grtrs = pl_module.grtrs[:]
                    _, ser, _ = compute_poliphony_metrics(preds, grtrs)
                    ser = min(ser, 100.0)
                    sweep_results[k] = ser
                    stage_bar.write(f"    stage={k}  val/ser={ser:.2f}%  ({len(preds)} samples)")
                    print(f"    stage={k}  val/ser={ser:.2f}%  ({len(preds)} samples)")
        finally:
            # Always restore original stage, maxlen, preds/grtrs, and train mode
            self.val_set.set_stage_direct(current_stage)
            pl_module.model.maxlen = original_maxlen
            pl_module.preds = []
            pl_module.grtrs = []
            pl_module.train()

        # Log all stages to WandB in one call
        if trainer.logger is not None:
            trainer.logger.experiment.log(
                {f"val/sweep/ser_stage{k}": v for k, v in sweep_results.items()}
                | {"trainer/global_step": trainer.global_step}
            )
        print(f"  ── Full-sweep done ──\n")

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if self._clear_ser:
            trainer.callback_metrics.pop("val/ser", None)
            self._clear_ser = False

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        pl_module.set_stage(self._stage)
        pl_module.set_stage_calculator(lambda epoch: self._stage)

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self.val_set.set_stage_direct(self._stage)   # re-assert so persistent workers see current stage
        print(f"  [Val] epoch={trainer.current_epoch}  stage={self._stage}  "
              f"val_set_size={len(self.val_set)}")

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        val_ser = trainer.callback_metrics.get("val/ser")
        if val_ser is None:
            return
        val_ser = float(val_ser)

        # ── final-stage early stopping ────────────────────────────────────────
        if self._at_final:
            if val_ser < self._best_ser:
                self._best_ser   = val_ser
                self._no_improve = 0
            else:
                self._no_improve += 1
                print(f"  [EarlyStop] No improvement for {self._no_improve}/{self.final_patience} "
                      f"val epochs (best val/ser={self._best_ser:.4f})")
                if self._no_improve >= self.final_patience:
                    print(f"  ── Early stopping triggered at final stage ──")
                    self._run_full_sweep(trainer, pl_module)
                    trainer.should_stop = True
            return

        # ── pre-final: check whether to advance stage ─────────────────────────
        # Increment patience counter only when below threshold AND no longer improving.
        # If val/ser is still dropping, reset counter — keep training.
        if val_ser < self.ser_threshold:
            if val_ser < self._stage_best_ser:
                self._stage_best_ser = val_ser
                self._epochs_below = 0   # still improving — reset
                # Save best-within-stage checkpoint (overwrite on each new best)
                self._stage_best_ckpt = str(
                    Path(self.weights_dir) / f"pagecrop_fold{self.fold}_stage{self._stage}_best.ckpt"
                )
                trainer.save_checkpoint(self._stage_best_ckpt)
            else:
                self._epochs_below += 1  # below threshold but plateaued
        else:
            self._epochs_below = 0

        print(f"  [Curriculum] stage={self._stage}/{self.num_cl_stages}  "
              f"val/ser={val_ser:.2f}  threshold={self.ser_threshold:.2f}  "
              f"best={self._stage_best_ser:.2f}  below={self._epochs_below}/{self.patience}")

        if self._epochs_below >= self.patience:
            # Restore best-within-stage weights, save as stage checkpoint, delete temp best file
            stage_path = (Path(self.weights_dir)
                          / f"pagecrop_fold{self.fold}_stage{self._stage}.ckpt")
            if self._stage_best_ckpt and Path(self._stage_best_ckpt).exists():
                ckpt = torch.load(self._stage_best_ckpt, map_location=pl_module.device)
                pl_module.load_state_dict(ckpt["state_dict"])
                trainer.save_checkpoint(str(stage_path))
                Path(self._stage_best_ckpt).unlink()
            else:
                trainer.save_checkpoint(str(stage_path))
            print(f"  ── Stage {self._stage} checkpoint saved: {stage_path.name} "
                  f"(best val/ser={self._stage_best_ser:.2f}) ──")

            # Full validation sweep disabled to save time

            self._stage = min(self._stage + 1, self.num_cl_stages)
            self._epochs_below = 0
            self._stage_best_ser = float("inf")
            self._stage_best_ckpt = None
            self.train_set.set_stage_direct(self._stage)
            self.val_set.set_stage_direct(self._stage)

            # Defer clearing stale val/ser to next train epoch start,
            # so ModelCheckpoint can still read it on this epoch.
            self._clear_ser = True
            trainer.callback_metrics.pop("val/loss", None)

            # Force Lightning to update epoch size for the new stage
            # (includes replay samples, so epoch size grows with each stage)
            new_len = len(self.train_set)
            batch_size = trainer.train_dataloader.batch_size
            trainer.fit_loop.max_batches = (new_len + batch_size - 1) // batch_size
            print(f"  Updated epoch size: {new_len} samples, {trainer.fit_loop.max_batches} steps")

            if self._stage >= self.num_cl_stages:
                self._activate_final_stage(trainer, pl_module, val_ser)
            else:
                pl_module.set_stage(self._stage)
                print(f"\n  ── Curriculum → stage {self._stage}  "
                      f"(val/ser={val_ser:.4f} < {self.ser_threshold}) ──")


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
    resume: str = None,                    # path to last.ckpt to resume training
    resume_stage: int = 1,                 # curriculum stage to restore when resuming
    debug: bool = False,
    cpu_test: bool = False,   # skip image scan, use tiny model dims, fast_dev_run on CPU
):
    torch.set_float32_matmul_precision("high")
    gc.collect()
    torch.cuda.empty_cache()
    seed_everything(seed=42, workers=True)

    gin.parse_config_file(config)

    # Resolve all params: gin config is the default, CLI arg overrides
    gin_lr, gin_accum, gin_bs, gin_val_bs, gin_nw, gin_val_freq          = train_hparams()
    gin_data, gin_ckpt, gin_wts, gin_synthetic, gin_resume, gin_resume_s, gin_wandb_name = train_paths()
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
    if resume is None:
        resume = gin_resume            # None = fresh start
    if resume_stage == 1:
        resume_stage = gin_resume_s    # CLI default 1 → use gin value
    synthetic_data_path       = gin_synthetic   # None = no synthetic data
    val_batch_size            = gin_val_bs
    check_val_every_n_epoch   = gin_val_freq

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
    train_set = PageCropDataset(data_path=data_path, split="train", fold=fold, augment=False,
                                synthetic_data_path=synthetic_data_path)
    val_set   = PageCropDataset(data_path=data_path, split="val",   fold=fold, augment=False)
    test_set  = PageCropDataset(data_path=data_path, split="test",  fold=fold, augment=False)

    # Build or load vocabulary
    if resume is not None:
        # When resuming, use vocab from the resume checkpoint to guarantee consistency
        import torch as _torch
        _ckpt = _torch.load(resume, map_location="cpu", weights_only=False)
        w2i = _ckpt["hyper_parameters"]["w2i"]
        i2w = _ckpt["hyper_parameters"]["i2w"]
        del _ckpt
        print(f"  Vocab loaded from resume checkpoint ({len(w2i)} tokens)")
    else:
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
    if cpu_test:
        # Small fixed dims so torchinfo.summary fits in CPU RAM
        max_height, max_width, max_len = 256, 512, 512
        print(f"\n[cpu_test] Skipping image scan — using fixed dims {max_height}×{max_width}, seqlen {max_len}")
    else:
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
    # Train: persistent_workers=True so workers stay alive across epochs.
    # Stage updates propagate via a shared-memory tensor (_shared_stage).
    # Val: persistent_workers=False so workers are respawned fresh each val run,
    # guaranteeing they pick up the correct stage. Val runs infrequently
    # (check_val_every_n_epoch=10) so the respawn cost is negligible.
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
        batch_size=val_batch_size,
        num_workers=num_workers,
        collate_fn=batch_preparation_img2seq,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=val_batch_size,
        num_workers=num_workers,
        collate_fn=batch_preparation_img2seq,
        persistent_workers=(num_workers > 0),
    )

    # ── callbacks ──────────────────────────────────────────────────────────────
    final_patience = gin.query_parameter("DynamicCurriculumAdvancer.final_patience")
    num_cl_stages  = gin.query_parameter("DynamicCurriculumAdvancer.num_cl_stages")

    # last_ckpt: always saves the most recent checkpoint (for training continuity).
    # save_top_k=0 means no "best" tracking — only last.ckpt is written.
    last_ckpt = ModelCheckpoint(
        dirpath=weights_dir,
        filename=f"pagecrop_fold{fold}_last",
        monitor=None,
        save_top_k=0,
        save_last=True,
        verbose=False,
        save_on_train_epoch_end=True,   # save at epoch end so resume doesn't warn about incomplete epoch
    )

    # best_ckpt: starts disabled (save_top_k=0).
    # DynamicCurriculumAdvancer sets save_top_k=1 when the final stage is reached.
    best_ckpt = ModelCheckpoint(
        dirpath=weights_dir,
        filename=f"pagecrop_fold{fold}_best",
        monitor="val/ser",
        mode="min",
        save_top_k=0,           # disabled until final stage
        save_last=False,
        verbose=True,
        save_on_train_epoch_end=False,
    )

    curriculum_cb = DynamicCurriculumAdvancer(
        train_set=train_set,
        val_set=val_set,
        best_ckpt=best_ckpt,
        weights_dir=weights_dir,
        fold=fold,
        final_patience=final_patience,
    )

    print(f"\n  resume={resume}  resume_stage={resume_stage}")
    if resume is not None:
        # Restore curriculum stage (callback state is not saved in the checkpoint)
        curriculum_cb._stage = resume_stage
        train_set.set_stage_direct(resume_stage)
        val_set.set_stage_direct(resume_stage)
        print(f"  train _shared_stage={int(train_set._shared_stage[0])}  val _shared_stage={int(val_set._shared_stage[0])}")
        print(f"  Resuming from: {resume}  (curriculum stage={resume_stage})")
        if resume_stage >= curriculum_cb.num_cl_stages:
            # Already at final stage — skip pre-final advancement logic entirely
            curriculum_cb._at_final = True
            best_ckpt.save_top_k = 1
            print(f"  Resumed at final stage {resume_stage} — early-stop patience={curriculum_cb.final_patience}")

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ── logger ─────────────────────────────────────────────────────────────────
    wandb_logger = WandbLogger(
        project="fullpage-jazz-omr",
        name=gin_wandb_name or f"pagecrop_fold{fold}_lr{lr}",
        group="pagecrop_curriculum",
        log_model=False,
        save_dir="logs",
    )

    # ── trainer ────────────────────────────────────────────────────────────────
    trainer = Trainer(
        logger=wandb_logger,
        # curriculum_cb MUST come before best_ckpt so that when curriculum_cb
        # flips best_ckpt.save_top_k=1 at the final stage, best_ckpt still runs
        # and saves the model for that same val epoch.
        callbacks=[last_ckpt, curriculum_cb, best_ckpt, lr_monitor],
        max_epochs=epochs,
        precision="32" if cpu_test else "bf16-mixed",
        accelerator="cpu" if cpu_test else "auto",
        accumulate_grad_batches=1 if cpu_test else accumulate_grad_batches,
        fast_dev_run=True if cpu_test else debug,
        check_val_every_n_epoch=1 if cpu_test else check_val_every_n_epoch,
        log_every_n_steps=5,
        deterministic=False,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume,   # None = fresh start; path = resume from checkpoint
    )

    # ── test with best checkpoint (or last if final stage was never reached) ───
    test_ckpt_path = best_ckpt.best_model_path or last_ckpt.last_model_path
    print(f"\nTesting with checkpoint: {test_ckpt_path}")
    best_model = CurriculumSMTTrainer.load_from_checkpoint(
        test_ckpt_path,
        load_pretrained=False,
        strict=False,
    )
    # Use vocab from the test checkpoint (guaranteed match)
    test_w2i = best_model.model.w2i
    test_i2w = best_model.model.i2w
    test_set.set_dictionaries(test_w2i, test_i2w)
    test_set.set_stage_direct(num_cl_stages)
    best_model.freeze()
    best_model.eval()
    trainer.test(best_model, dataloaders=test_loader)


if __name__ == "__main__":
    fire.Fire(train)
