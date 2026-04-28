"""
Full-page curriculum training — pre-computed stacking strategy.

Differences from train.py:
  - Uses StackingCropDataset (strict n-matching, no maxed-out masking logic).
  - Starts at stage 2 (stage 1 = system-level, already covered by smt_0.ckpt).
  - Validation uses jazzmus_stacked val split (stacked N=1..8, real full pages at N=9).
  - Full-sweep validation on stage advance is disabled (too slow, not informative).

Run from FullPageJazzOMR/ project root:
    python train_precomputed_stacking.py \\
        --config config/stacked_precomputed_9stage.gin
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

sys.path.insert(0, str(Path(__file__).parent))

from datasets.stacking_crop_dataset import StackingCropDataset
from jazzmus.curriculum.trainer import CurriculumSMTTrainer
from jazzmus.dataset.smt_dataset import batch_preparation_img2seq
from jazzmus.dataset.smt_dataset_utils import check_and_retrieveVocabulary


# ── gin-configurable hyperparameter holders ────────────────────────────────────

@gin.configurable
def train_hparams(
    lr: float = 5e-5,
    accumulate_grad_batches: int = 1,
    batch_size: int = 8,
    val_batch_size: int = 1,
    num_workers: int = 8,
    check_val_every_n_epoch: int = 10,
):
    return lr, accumulate_grad_batches, batch_size, val_batch_size, num_workers, check_val_every_n_epoch


@gin.configurable
def train_paths(
    data_path:           str = "data/jazzmus_stacked",
    checkpoint:          str = "../ISMIR-Jazzmus/weights/smt/smt_0.ckpt",
    weights_dir:         str = "weights/stacked_precomputed",
    resume:              str = None,
    resume_stage:        int = 2,
    wandb_name:          str = None,
):
    return data_path, checkpoint, weights_dir, resume, resume_stage, wandb_name


# ── dynamic curriculum callback (no full sweep) ────────────────────────────────

@gin.configurable
class DynamicCurriculumAdvancer(Callback):
    """Advances curriculum stage when val/ser drops below a threshold.

    Identical to the version in train.py except _run_full_sweep is removed —
    it decodes all future stages at each advance which takes hours.
    """

    def __init__(
        self,
        train_set,
        val_set,
        best_ckpt:      ModelCheckpoint,
        weights_dir:    str,
        fold:           int,
        num_cl_stages:  int   = 9,
        ser_threshold:  float = 20.0,
        patience:       int   = 3,
        final_patience: int   = 10,
        start_stage:    int   = 2,
        early_accum_grad:    int = 1,   # accumulate_grad_batches for early stages
        early_stage_cutoff:  int = 0,   # use early_accum_grad for stages <= this
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
        self.early_accum_grad   = early_accum_grad
        self.early_stage_cutoff = early_stage_cutoff
        self._stage           = start_stage
        self._epochs_below    = 0
        self._stage_best_ser  = float("inf")
        self._stage_best_ckpt = None
        self._at_final        = False
        self._best_ser        = float("inf")
        self._no_improve      = 0
        self._clear_ser       = False

        train_set.set_stage_direct(start_stage)
        val_set.set_stage_direct(start_stage)

    def _activate_final_stage(self, trainer, pl_module, val_ser):
        self._at_final = True
        self._best_ser = val_ser
        self.best_ckpt.save_top_k = 1
        pl_module.set_stage(self._stage)
        print(f"\n  ── Curriculum → FINAL stage {self._stage}  "
              f"(val/ser={val_ser:.4f} < {self.ser_threshold}) ──")
        print(f"  ── Best-val/ser checkpoint now active; early-stop patience={self.final_patience} ──")

    def on_train_epoch_start(self, trainer, pl_module):
        if self._clear_ser:
            trainer.callback_metrics.pop("val/ser", None)
            self._clear_ser = False

    def on_train_start(self, trainer, pl_module):
        pl_module.set_stage(self._stage)
        pl_module.set_stage_calculator(lambda epoch: self._stage)
        # Apply early-stage accumulation + scaled lr if starting within the early range
        self._base_lr = pl_module.hparams.lr
        if self.early_accum_grad > 1 and self._stage <= self.early_stage_cutoff:
            trainer.accumulate_grad_batches = self.early_accum_grad
            scaled_lr = self._base_lr * self.early_accum_grad
            for pg in pl_module.optimizers().param_groups:
                pg["lr"] = scaled_lr
            print(f"  [Early batch] stage={self._stage} <= cutoff={self.early_stage_cutoff}  "
                  f"accumulate_grad_batches={self.early_accum_grad}  "
                  f"lr={self._base_lr} → {scaled_lr}  "
                  f"(effective batch={trainer.train_dataloader.batch_size * self.early_accum_grad})")

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_set.set_stage_direct(self._stage)
        print(f"  [Val] epoch={trainer.current_epoch}  stage={self._stage}  "
              f"val_set_size={len(self.val_set)}")

    def on_validation_epoch_end(self, trainer, pl_module):
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
                    trainer.should_stop = True
            return

        # ── pre-final: check whether to advance stage ─────────────────────────
        if val_ser < self.ser_threshold:
            if val_ser < self._stage_best_ser:
                self._stage_best_ser = val_ser
                self._epochs_below = 0
                self._stage_best_ckpt = str(
                    Path(self.weights_dir) / f"stacked_fold{self.fold}_stage{self._stage}_best.ckpt"
                )
                trainer.save_checkpoint(self._stage_best_ckpt)
            else:
                self._epochs_below += 1
        else:
            self._epochs_below = 0

        print(f"  [Curriculum] stage={self._stage}/{self.num_cl_stages}  "
              f"val/ser={val_ser:.2f}  threshold={self.ser_threshold:.2f}  "
              f"best={self._stage_best_ser:.2f}  below={self._epochs_below}/{self.patience}")

        if self._epochs_below >= self.patience:
            stage_path = Path(self.weights_dir) / f"stacked_fold{self.fold}_stage{self._stage}.ckpt"
            if self._stage_best_ckpt and Path(self._stage_best_ckpt).exists():
                ckpt = torch.load(self._stage_best_ckpt, map_location=pl_module.device)
                pl_module.load_state_dict(ckpt["state_dict"])
                trainer.save_checkpoint(str(stage_path))
                Path(self._stage_best_ckpt).unlink()
            else:
                trainer.save_checkpoint(str(stage_path))
            print(f"  ── Stage {self._stage} checkpoint saved: {stage_path.name} "
                  f"(best val/ser={self._stage_best_ser:.2f}) ──")

            self._stage = min(self._stage + 1, self.num_cl_stages)
            self._epochs_below = 0
            self._stage_best_ser = float("inf")
            self._stage_best_ckpt = None
            self.train_set.set_stage_direct(self._stage)
            self.val_set.set_stage_direct(self._stage)

            self._clear_ser = True
            trainer.callback_metrics.pop("val/loss", None)

            # Switch from early accumulation + lr to normal when crossing the cutoff
            if (self.early_accum_grad > 1
                    and self._stage > self.early_stage_cutoff
                    and trainer.accumulate_grad_batches != 1):
                trainer.accumulate_grad_batches = 1
                for pg in pl_module.optimizers().param_groups:
                    pg["lr"] = self._base_lr
                print(f"  [Early batch → normal] stage={self._stage} > cutoff={self.early_stage_cutoff}  "
                      f"accumulate_grad_batches=1  lr → {self._base_lr}  "
                      f"(effective batch={trainer.train_dataloader.batch_size})")

            new_len = len(self.train_set)
            batch_size = trainer.train_dataloader.batch_size
            accum = trainer.accumulate_grad_batches
            trainer.fit_loop.max_batches = (new_len + batch_size - 1) // batch_size
            print(f"  Updated epoch size: {new_len} samples, "
                  f"{trainer.fit_loop.max_batches} steps  "
                  f"(effective batch={batch_size * accum})")

            if self._stage >= self.num_cl_stages:
                self._activate_final_stage(trainer, pl_module, val_ser)
            else:
                pl_module.set_stage(self._stage)
                print(f"\n  ── Curriculum → stage {self._stage}  "
                      f"(val/ser={val_ser:.4f} < {self.ser_threshold}) ──")


# ── main training function ─────────────────────────────────────────────────────

def train(
    config:                  str,
    fold:                    int   = 0,
    epochs:                  int   = 10000,
    accumulate_grad_batches: int   = None,
    batch_size:              int   = None,
    num_workers:             int   = None,
    lr:                      float = None,
    data_path:               str   = None,
    checkpoint:              str   = None,
    weights_dir:             str   = None,
    resume:                  str   = None,
    resume_stage:            int   = None,
    debug:                   bool  = False,
):
    torch.set_float32_matmul_precision("high")
    gc.collect()
    torch.cuda.empty_cache()
    seed_everything(seed=42, workers=True)

    gin.parse_config_file(config)

    gin_lr, gin_accum, gin_bs, gin_val_bs, gin_nw, gin_val_freq = train_hparams()
    gin_data, gin_ckpt, gin_wts, gin_resume, gin_resume_s, gin_wandb = train_paths()

    if lr is None:                      lr                      = gin_lr
    if accumulate_grad_batches is None: accumulate_grad_batches = gin_accum
    if batch_size is None:              batch_size              = gin_bs
    if num_workers is None:             num_workers             = gin_nw
    if data_path is None:               data_path               = gin_data
    if checkpoint is None:              checkpoint              = gin_ckpt
    if weights_dir is None:             weights_dir             = gin_wts
    if resume is None:                  resume                  = gin_resume
    if resume_stage is None:            resume_stage            = gin_resume_s

    val_batch_size          = gin_val_bs
    check_val_every_n_epoch = gin_val_freq

    for folder in (weights_dir, "logs", "vocab"):
        os.makedirs(folder, exist_ok=True)

    print("PRE-COMPUTED STACKING CURRICULUM TRAINING")
    print(f"  Config        : {config}")
    print(f"  Checkpoint    : {checkpoint}")
    print(f"  Data          : {data_path}  (train/val/test from jazzmus_stacked)")
    print(f"  Weights dir   : {weights_dir}")
    print(f"  Fold          : {fold}")
    print(f"  LR            : {lr}  |  Batch: {batch_size}  |  Accum: {accumulate_grad_batches}")
    print(f"  Start stage   : {resume_stage}  (stage 1 skipped — covered by pretrained model)")

    # ── datasets ───────────────────────────────────────────────────────────────
    # All splits from jazzmus_stacked/ via StackingCropDataset:
    #   train: stacked N=1..9 + real/syn full pages at N=9
    #   val:   stacked N=1..8, real full pages at N=9 (~16 pages)
    #   test:  real full pages only
    train_set = StackingCropDataset(data_path=data_path, split="train", fold=fold, augment=False)
    val_set   = StackingCropDataset(data_path=data_path, split="val",   fold=fold, augment=False)
    test_set  = StackingCropDataset(data_path=data_path, split="test",  fold=fold, augment=False)

    # ── vocabulary ─────────────────────────────────────────────────────────────
    if resume is not None:
        _ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        w2i = _ckpt["hyper_parameters"]["w2i"]
        i2w = _ckpt["hyper_parameters"]["i2w"]
        del _ckpt
        print(f"  Vocab from resume checkpoint ({len(w2i)} tokens)")
    else:
        w2i, i2w = check_and_retrieveVocabulary(
            [train_set.get_gt(), val_set.get_gt(), test_set.get_gt()],
            "vocab", "vocab_stacked",
        )
    train_set.set_dictionaries(w2i, i2w)
    val_set.set_dictionaries(w2i, i2w)
    test_set.set_dictionaries(w2i, i2w)

    print(f"\n  Train samples : {len(train_set.samples)}")
    print(f"  Val samples   : {len(val_set.samples)}")
    print(f"  Test samples  : {len(test_set.samples)}")
    print(f"  Vocab size    : {train_set.vocab_size()}")

    # ── model sizing (pre-computed, stable across runs) ─────────────────────────
    max_height = 2304   # 9 stages × 256 system_height
    max_width  = 5860   # widest image after aspect-ratio resize
    max_len    = 1982   # longest GT sequence × 1.1
    print(f"\n  Model dims : {max_height} × {max_width}, seqlen {max_len}")

    # ── load pretrained checkpoint ─────────────────────────────────────────────
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
        load_pretrained=False,
        strict=False,
    )

    # Resize output head if vocab changed (adds <linebreak> row)
    out_layer      = model.model.decoder.out_layer
    expected_vocab = train_set.vocab_size()
    if out_layer.out_channels != expected_vocab:
        print(f"  Resizing output layer: {out_layer.out_channels} → {expected_vocab}")
        old_weights = out_layer.weight.data
        old_w2i     = model.hparams.get("w2i") or {}
        new_layer   = Conv1d(out_layer.in_channels, expected_vocab, kernel_size=1)
        for token, new_idx in w2i.items():
            if token in old_w2i:
                old_idx = old_w2i[token]
                if old_idx < old_weights.shape[0]:
                    new_layer.weight.data[new_idx] = old_weights[old_idx]
        model.model.decoder.out_layer = new_layer
        print(f"  Preserved weights for {len(old_w2i)} known tokens")
    else:
        print(f"  Output layer OK: {expected_vocab} tokens")

    # ── dataloaders ────────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_set, batch_size=batch_size, num_workers=num_workers,
        shuffle=True, collate_fn=batch_preparation_img2seq,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_set, batch_size=val_batch_size, num_workers=num_workers,
        collate_fn=batch_preparation_img2seq,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_set, batch_size=val_batch_size, num_workers=num_workers,
        collate_fn=batch_preparation_img2seq,
        persistent_workers=(num_workers > 0),
    )

    # ── callbacks ──────────────────────────────────────────────────────────────
    final_patience = gin.query_parameter("DynamicCurriculumAdvancer.final_patience")
    num_cl_stages  = gin.query_parameter("DynamicCurriculumAdvancer.num_cl_stages")

    last_ckpt = ModelCheckpoint(
        dirpath=weights_dir, filename=f"stacked_fold{fold}_last",
        monitor=None, save_top_k=0, save_last=True, verbose=False,
        save_on_train_epoch_end=True,
    )
    best_ckpt = ModelCheckpoint(
        dirpath=weights_dir, filename=f"stacked_fold{fold}_best",
        monitor="val/ser", mode="min", save_top_k=0,
        save_last=False, verbose=True, save_on_train_epoch_end=False,
    )
    curriculum_cb = DynamicCurriculumAdvancer(
        train_set=train_set, val_set=val_set,
        best_ckpt=best_ckpt, weights_dir=weights_dir,
        fold=fold, final_patience=final_patience,
        start_stage=resume_stage,
    )

    if resume is not None:
        curriculum_cb._stage = resume_stage
        train_set.set_stage_direct(resume_stage)
        val_set.set_stage_direct(resume_stage)
        if resume_stage >= curriculum_cb.num_cl_stages:
            curriculum_cb._at_final = True
            best_ckpt.save_top_k = 1

    # ── logger + trainer ───────────────────────────────────────────────────────
    wandb_logger = WandbLogger(
        project="fullpage-jazz-omr",
        name=gin_wandb or f"stacked_precomputed_fold{fold}_stage{resume_stage}start",
        group="stacked_precomputed",
        log_model=False,
        save_dir="logs",
    )

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[last_ckpt, curriculum_cb, best_ckpt, LearningRateMonitor("step")],
        max_epochs=epochs,
        precision="bf16-mixed",
        accelerator="auto",
        accumulate_grad_batches=accumulate_grad_batches,
        fast_dev_run=debug,
        check_val_every_n_epoch=check_val_every_n_epoch,
        log_every_n_steps=5,
        deterministic=False,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume,
    )

    # ── final test ─────────────────────────────────────────────────────────────
    test_ckpt_path = best_ckpt.best_model_path or last_ckpt.last_model_path
    print(f"\nTesting with: {test_ckpt_path}")
    best_model = CurriculumSMTTrainer.load_from_checkpoint(
        test_ckpt_path, load_pretrained=False, strict=False,
    )
    test_set.set_dictionaries(best_model.model.w2i, best_model.model.i2w)
    test_set.set_stage_direct(num_cl_stages)
    best_model.freeze()
    best_model.eval()
    trainer.test(best_model, dataloaders=test_loader)


if __name__ == "__main__":
    fire.Fire(train)
