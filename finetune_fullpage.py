"""
Full-page fine-tune — single-stage, no curriculum.

Starts from the final-stage stacking checkpoint and continues training
on real + synthetic full-page images.  Validates on real full pages and
keeps the best checkpoint by val/loss with early stopping.

Run from the FullPageJazzOMR/ project root:
    python finetune_fullpage.py --config config/fullpage_finetune.gin
"""

import gc
import os
import sys
from pathlib import Path

import fire
import gin
import torch
from torch.nn import Conv1d

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from datasets.full_page_dataset import FullPageDataset
from jazzmus.curriculum.trainer import CurriculumSMTTrainer
from jazzmus.dataset.smt_dataset import batch_preparation_img2seq
from jazzmus.dataset.smt_dataset_utils import check_and_retrieveVocabulary


# ── gin-configurable holders ───────────────────────────────────────────────────

@gin.configurable
def train_hparams(
    lr: float = 1e-6,
    accumulate_grad_batches: int = 1,
    batch_size: int = 1,
    val_batch_size: int = 1,
    num_workers: int = 8,
    check_val_every_n_epoch: int = 5,
):
    return lr, accumulate_grad_batches, batch_size, val_batch_size, num_workers, check_val_every_n_epoch


@gin.configurable
def train_paths(
    data_path:      str = "data/jazzmus_fullpage",
    syn_data_path:  str = None,
    checkpoint:     str = None,
    weights_dir:    str = "weights/fullpage_finetune",
    wandb_name:     str = None,
):
    return data_path, syn_data_path, checkpoint, weights_dir, wandb_name


@gin.configurable
def finetune_params(
    patience: int = 10,
    stage: int = 9,   # final stage of the loaded curriculum model
):
    return patience, stage


# ── main ────────────────────────────────────────────────────────────────────────

def train(
    config:                  str,
    fold:                    int   = 0,
    epochs:                  int   = 10000,
    accumulate_grad_batches: int   = None,
    batch_size:              int   = None,
    num_workers:             int   = None,
    lr:                      float = None,
    data_path:               str   = None,
    syn_data_path:           str   = None,
    checkpoint:              str   = None,
    weights_dir:             str   = None,
    debug:                   bool  = False,
):
    torch.set_float32_matmul_precision("high")
    gc.collect()
    torch.cuda.empty_cache()
    seed_everything(seed=42, workers=True)

    gin.parse_config_file(config)

    gin_lr, gin_accum, gin_bs, gin_val_bs, gin_nw, gin_val_freq = train_hparams()
    gin_data, gin_syn, gin_ckpt, gin_wts, gin_wandb             = train_paths()
    patience, stage                                             = finetune_params()

    if lr is None:                      lr                      = gin_lr
    if accumulate_grad_batches is None: accumulate_grad_batches = gin_accum
    if batch_size is None:              batch_size              = gin_bs
    if num_workers is None:             num_workers             = gin_nw
    if data_path is None:               data_path               = gin_data
    if syn_data_path is None:           syn_data_path           = gin_syn
    if checkpoint is None:              checkpoint              = gin_ckpt
    if weights_dir is None:             weights_dir             = gin_wts

    val_batch_size          = gin_val_bs
    check_val_every_n_epoch = gin_val_freq

    if checkpoint is None:
        raise ValueError("checkpoint must be set (via --checkpoint or gin config)")

    for folder in (weights_dir, "logs", "vocab"):
        os.makedirs(folder, exist_ok=True)

    print("FULL-PAGE FINE-TUNE")
    print(f"  Config        : {config}")
    print(f"  Checkpoint    : {checkpoint}")
    print(f"  Real data     : {data_path}")
    print(f"  Syn data      : {syn_data_path}")
    print(f"  Weights dir   : {weights_dir}")
    print(f"  Fold          : {fold}")
    print(f"  LR            : {lr}  |  Batch: {batch_size}  |  Accum: {accumulate_grad_batches}")
    print(f"  Early-stop patience: {patience} val epochs (val/loss)")

    # ── datasets ───────────────────────────────────────────────────────────────
    train_set = FullPageDataset(data_path=data_path, split="train", fold=fold,
                                syn_data_path=syn_data_path, augment=False)
    val_set   = FullPageDataset(data_path=data_path, split="val",   fold=fold)
    test_set  = FullPageDataset(data_path=data_path, split="test",  fold=fold)

    # Vocab: reuse ckpt's to guarantee index consistency
    _ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    w2i = _ckpt["hyper_parameters"]["w2i"]
    i2w = _ckpt["hyper_parameters"]["i2w"]
    del _ckpt
    print(f"  Vocab from checkpoint ({len(w2i)} tokens)")

    train_set.set_dictionaries(w2i, i2w)
    val_set.set_dictionaries(w2i, i2w)
    test_set.set_dictionaries(w2i, i2w)

    print(f"\n  Train samples : {len(train_set)}")
    print(f"  Val samples   : {len(val_set)}")
    print(f"  Test samples  : {len(test_set)}")

    # ── model sizing (shared with stacking run) ────────────────────────────────
    max_height = 2304
    max_width  = 5860
    max_len    = 1982
    print(f"\n  Model dims : {max_height} × {max_width}, seqlen {max_len}")

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

    # Vocab should match exactly (we loaded from the same ckpt), but keep the
    # resize guard in case the fine-tune ckpt was produced from an older run.
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

    # Pin the stage so any stage-dependent model behavior uses the final setting.
    model.set_stage(stage)
    model.set_stage_calculator(lambda epoch: stage)

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
    last_ckpt = ModelCheckpoint(
        dirpath=weights_dir, filename=f"fullpage_ft_fold{fold}_last",
        monitor=None, save_top_k=0, save_last=True, verbose=False,
        save_on_train_epoch_end=True,
    )
    best_ckpt = ModelCheckpoint(
        dirpath=weights_dir, filename=f"fullpage_ft_fold{fold}_best",
        monitor="val/loss", mode="min", save_top_k=1,
        save_last=False, verbose=True, save_on_train_epoch_end=False,
    )
    early_stop = EarlyStopping(
        monitor="val/loss", mode="min", patience=patience, verbose=True,
    )

    # ── logger + trainer ───────────────────────────────────────────────────────
    wandb_logger = WandbLogger(
        project="fullpage-jazz-omr",
        name=gin_wandb or f"fullpage_finetune_fold{fold}",
        group="fullpage_finetune",
        log_model=False,
        save_dir="logs",
    )

    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[last_ckpt, best_ckpt, early_stop, LearningRateMonitor("step")],
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
    )

    # ── final test with best checkpoint ────────────────────────────────────────
    test_ckpt_path = best_ckpt.best_model_path or last_ckpt.last_model_path
    print(f"\nTesting with: {test_ckpt_path}")
    best_model = CurriculumSMTTrainer.load_from_checkpoint(
        test_ckpt_path, load_pretrained=False, strict=False,
    )
    test_set.set_dictionaries(best_model.model.w2i, best_model.model.i2w)
    best_model.set_stage(stage)
    best_model.freeze()
    best_model.eval()
    trainer.test(best_model, dataloaders=test_loader)


if __name__ == "__main__":
    fire.Fire(train)
