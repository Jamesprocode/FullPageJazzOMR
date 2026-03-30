"""
Test a trained checkpoint on the page-crop test set.

Run from the FullPageJazzOMR/ project root:
    python test.py --checkpoint weights/pagecrop/pagecrop_fold0_best.ckpt
    python test.py --checkpoint weights/pagecrop/pagecrop_fold0_best.ckpt --data_path /path/to/data --fold 0
"""

import sys
from pathlib import Path

import torch
import fire
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision("high")

sys.path.insert(0, str(Path(__file__).parent))

from datasets.page_crop_dataset import PageCropDataset
from jazzmus.curriculum.trainer import CurriculumSMTTrainer
from jazzmus.dataset.smt_dataset import batch_preparation_img2seq


def test(
    checkpoint:   str = "/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmus_weights/replay_100percent/pagecrop_fold0_best-v2.ckpt",
    data_path:    str = "/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmuss_Data/jazzmus_pagecrop",
    fold:          int = 0,
    final_stage:   int = 9,
    system_height: int = 256,
    num_workers:   int = 4,
    batch_size:    int = 1,
):
    # ── load checkpoint using its own saved w2i/i2w ──────────────────────────
    print(f"\nLoading checkpoint: {checkpoint}")
    model = CurriculumSMTTrainer.load_from_checkpoint(
        checkpoint,
        load_pretrained=False,
        strict=False,
    )

    # Use the w2i/i2w from the checkpoint (exact mapping used during training)
    w2i = model.model.w2i
    i2w = model.model.i2w
    print(f"  Vocab size (from checkpoint): {len(w2i)}")

    # ── test dataset ─────────────────────────────────────────────────────────
    test_set = PageCropDataset(data_path=data_path, split="test", fold=fold, augment=False, system_height=system_height)
    test_set.set_dictionaries(w2i, i2w)
    test_set.set_stage_direct(final_stage)
    print(f"  Test samples : {len(test_set.samples)}")

    # Override maxlen if test sequences are longer than checkpoint's
    ckpt_maxlen = model.model.maxlen
    seq_maxlen = int(max(len(t) for t in test_set.gt_tokens) * 1.1)
    if seq_maxlen > ckpt_maxlen:
        print(f"  Overriding maxlen: {ckpt_maxlen} → {seq_maxlen}")
        model.model.maxlen = seq_maxlen

    model.freeze()
    model.eval()

    # ── test ─────────────────────────────────────────────────────────────────
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=batch_preparation_img2seq,
    )

    trainer = Trainer(accelerator="auto", devices=1, logger=False, enable_checkpointing=False)
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    fire.Fire(test)
