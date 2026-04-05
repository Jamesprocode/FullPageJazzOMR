"""
Full-page curriculum training — on-the-fly stacking strategy.

Uses StackedPageDataset (train + val): pages synthesised by stacking real
system-level images at training time.  No pre-computed page crops needed.
Curriculum advances via DynamicCurriculumAdvancer (same as train.py).
At the final stage, real pagecrop images are mixed in with stacked samples.

Test set: PageCropDataset (real full pages) — same evaluation as train.py.

Run from FullPageJazzOMR/ project root:
    python train_stacked.py --config config/stacked_9stage.gin

    # resume from a checkpoint at a given curriculum stage:
    python train_stacked.py --config config/stacked_9stage.gin \\
        --resume weights/stacked/pagecrop_fold0_last.ckpt --resume_stage 4
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
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from datasets.page_crop_dataset import PageCropDataset
from datasets.stacked_page_dataset import StackedPageDataset
from jazzmus.curriculum.trainer import CurriculumSMTTrainer
from jazzmus.dataset.smt_dataset import batch_preparation_img2seq
from jazzmus.dataset.smt_dataset_utils import check_and_retrieveVocabulary

# Import shared callback + hparam holder from train.py
from train import DynamicCurriculumAdvancer, train_hparams


# ── gin-configurable paths for the stacking experiment ────────────────────────

@gin.configurable
def train_stacked_paths(
    system_data_path: str = "/home/hice1/jwang3180/scratch/jazzmus/ISMIR-Jazzmus/data/jazzmus_systems",
    real_data_path:   str = "/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmuss_Data/jazzmus_pagecrop",
    checkpoint:       str = "/home/hice1/jwang3180/scratch/jazzmus/ISMIR-Jazzmus/weights/smt/smt_0.ckpt",
    weights_dir:      str = "weights/stacked",
    resume:           str = None,
    resume_stage:     int = 1,
    wandb_name:       str = None,
):
    """Gin-configurable paths for the stacking training run."""
    return (system_data_path, real_data_path, checkpoint,
            weights_dir, resume, resume_stage, wandb_name)


# ── main ───────────────────────────────────────────────────────────────────────

def train(
    config:                  str,
    fold:                    int   = 0,
    epochs:                  int   = 10000,
    accumulate_grad_batches: int   = None,
    batch_size:              int   = None,
    num_workers:             int   = None,
    lr:                      float = None,
    resume:                  str   = None,
    resume_stage:            int   = 1,
    debug:                   bool  = False,
    cpu_test:                bool  = False,
):
    torch.set_float32_matmul_precision("high")
    gc.collect()
    torch.cuda.empty_cache()
    seed_everything(seed=42, workers=True)

    gin.parse_config_file(config)

    # Resolve params: gin default → CLI override
    gin_lr, gin_accum, gin_bs, gin_val_bs, gin_nw, gin_val_freq = train_hparams()
    (gin_sys, gin_real, gin_ckpt, gin_wts,
     gin_resume, gin_resume_s, gin_wandb) = train_stacked_paths()

    if lr is None:                      lr = gin_lr
    if accumulate_grad_batches is None: accumulate_grad_batches = gin_accum
    if batch_size is None:              batch_size = gin_bs
    if num_workers is None:             num_workers = gin_nw
    if resume is None:                  resume = gin_resume
    if resume_stage == 1:               resume_stage = gin_resume_s
    val_batch_size          = gin_val_bs
    check_val_every_n_epoch = gin_val_freq

    system_data_path = gin_sys
    real_data_path   = gin_real
    checkpoint       = gin_ckpt
    weights_dir      = gin_wts

    for folder in (weights_dir, "logs", "vocab"):
        os.makedirs(folder, exist_ok=True)

    print("PAGE-STACKING CURRICULUM TRAINING")
    print(f"  Config           : {config}")
    print(f"  Checkpoint       : {checkpoint}")
    print(f"  System data      : {system_data_path}")
    print(f"  Real data (mix)  : {real_data_path}")
    print(f"  Weights dir      : {weights_dir}")
    print(f"  Fold             : {fold}")
    print(f"  LR               : {lr}")
    print(f"  Batch size       : {batch_size}  "
          f"(effective: {batch_size * accumulate_grad_batches})")
    print(f"  Accum grad       : {accumulate_grad_batches}")
    print(f"  Num workers      : {num_workers}")

    # ── datasets ──────────────────────────────────────────────────────────────
    train_set = StackedPageDataset(
        system_data_path=system_data_path,
        split="train",
        fold=fold,
        augment=False,
        real_data_path=real_data_path,
    )
    # Val + test: PageCropDataset (real pre-computed crops) — identical to
    # the pagecrop/replay experiment so SER thresholds are directly comparable.
    val_set = PageCropDataset(
        data_path=real_data_path,
        split="val",
        fold=fold,
        augment=False,
    )
    test_set = PageCropDataset(
        data_path=real_data_path,
        split="test",
        fold=fold,
        augment=False,
    )

    # Build or restore vocabulary
    if resume is not None:
        _ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        w2i = _ckpt["hyper_parameters"]["w2i"]
        i2w = _ckpt["hyper_parameters"]["i2w"]
        del _ckpt
        print(f"  Vocab loaded from resume checkpoint ({len(w2i)} tokens)")
    else:
        w2i, i2w = check_and_retrieveVocabulary(
            [train_set.get_gt(), val_set.get_gt(), test_set.get_gt()],
            "vocab",
            "vocab_stacked",
        )
    train_set.set_dictionaries(w2i, i2w)
    val_set.set_dictionaries(w2i, i2w)
    test_set.set_dictionaries(w2i, i2w)

    print(f"\n  Train epoch size : {len(train_set)} samples  (stacked, on-the-fly)")
    print(f"  Val samples      : {len(val_set.samples)}  (real pagecrop)")
    print(f"  Test samples     : {len(test_set.samples)}  (real pagecrop)")
    print(f"  Vocab size       : {train_set.vocab_size()}")

    # ── model sizing ───────────────────────────────────────────────────────────
    if cpu_test:
        max_height, max_width, max_len = 256, 512, 512
        print(f"\n[cpu_test] Using fixed dims {max_height}×{max_width}, seqlen {max_len}")
    else:
        # train_set (StackedPageDataset) gives a conservative H×W estimate
        # from pre-loaded arrays — no disk scan needed.
        # val/test (PageCropDataset) scan is skipped; stacked estimate covers both.
        print("\nComputing max H×W from stacked train estimate…")
        max_height, max_width = train_set.get_max_hw()
        max_len = int(max(
            train_set.get_max_seqlen(),
            val_set.get_max_seqlen(),
        ) * 1.1)
        print(f"  Max H×W    : {max_height} × {max_width}")
    print(f"  Max seqlen : {max_len}")

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
        load_pretrained=False,
        strict=False,
    )

    # Resize output head if vocab changed
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

    last_ckpt = ModelCheckpoint(
        dirpath=weights_dir,
        filename=f"stacked_fold{fold}_last",
        monitor=None,
        save_top_k=0,
        save_last=True,
        verbose=False,
        save_on_train_epoch_end=True,
    )
    best_ckpt = ModelCheckpoint(
        dirpath=weights_dir,
        filename=f"stacked_fold{fold}_best",
        monitor="val/ser",
        mode="min",
        save_top_k=0,        # disabled until final stage
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
        curriculum_cb._stage = resume_stage
        train_set.set_stage_direct(resume_stage)
        val_set.set_stage_direct(resume_stage)
        if resume_stage >= curriculum_cb.num_cl_stages:
            curriculum_cb._at_final = True
            best_ckpt.save_top_k = 1
            print(f"  Resumed at final stage {resume_stage}")

    lr_monitor = LearningRateMonitor(logging_interval="step")

    wandb_logger = WandbLogger(
        project="fullpage-jazz-omr",
        name=gin_wandb or f"stacked_fold{fold}_lr{lr}",
        group="stacked_curriculum",
        log_model=False,
        save_dir="logs",
    )

    trainer = Trainer(
        logger=wandb_logger,
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
        ckpt_path=resume,
    )

    # ── test ───────────────────────────────────────────────────────────────────
    test_ckpt_path = best_ckpt.best_model_path or last_ckpt.last_model_path
    print(f"\nTesting with checkpoint: {test_ckpt_path}")
    best_model = CurriculumSMTTrainer.load_from_checkpoint(
        test_ckpt_path, load_pretrained=False, strict=False,
    )
    test_w2i = best_model.model.w2i
    test_i2w = best_model.model.i2w
    test_set.set_dictionaries(test_w2i, test_i2w)
    test_set.set_stage_direct(num_cl_stages)
    best_model.freeze()
    best_model.eval()
    trainer.test(best_model, dataloaders=test_loader)


if __name__ == "__main__":
    fire.Fire(train)
