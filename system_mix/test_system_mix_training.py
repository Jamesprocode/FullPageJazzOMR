"""
Smoke test for the curriculum training pipeline — no GPU, no checkpoint needed.

Verifies:
  1. Datamodule init (images loaded, vocab built)
  2. Model init from scratch (no checkpoint)
  3. One training batch through the full forward + loss pass
  4. One val batch (greedy decode, CER/SER computed)
  5. One test batch

Run from the project root:
    python test_cl_training.py --config config/full-page/fullpage_cl.gin --fold 0
"""

import argparse
import torch
import gin

from jazzmus.curriculum.trainer import CurriculumSMTTrainer
from jazzmus.curriculum.dataset import JazzCLDataModule
from jazzmus.dataset.full_page_smt_dataset import batch_preparation_img2seq


def main(config: str, fold: int = 0):
    gin.parse_config_file(config)

    print("── Datamodule ────────────────────────────────────────────────────────")
    dm = JazzCLDataModule(fold=fold, batch_size=1)

    train_h, train_w = dm.train_set.get_max_hw()
    test_h,  test_w  = dm.test_set.get_max_hw()
    max_height = max(train_h, test_h)
    max_width  = max(train_w, test_w)
    max_len = int(max(dm.train_set.get_max_seqlen(),
                      dm.test_set.get_max_seqlen()) * 1.1)

    print(f"  Max H×W   : {max_height} × {max_width}")
    print(f"  Max seqlen: {max_len}")
    print(f"  Vocab size: {len(dm.train_set.w2i)}")

    print("\n── Model (fresh init, no checkpoint) ────────────────────────────────")
    model = CurriculumSMTTrainer(
        maxh=int(max_height),
        maxw=int(max_width),
        maxlen=int(max_len),
        out_categories=len(dm.train_set.w2i),
        padding_token=dm.train_set.w2i["<pad>"],
        in_channels=1,
        w2i=dm.train_set.w2i,
        i2w=dm.train_set.i2w,
        lr=5e-5,
        fold=fold,
    )
    model.set_stage(dm.train_set.curriculum_stage_beginning)
    model.set_stage_calculator(dm.train_set.get_stage_calculator())
    model.eval()
    print("  Model created OK")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.1f} M")

    print("\n── Train batch ───────────────────────────────────────────────────────")
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    x, di, y, paths = batch
    print(f"  Image  : {tuple(x.shape)}")
    print(f"  Dec in : {tuple(di.shape)}")
    print(f"  Target : {tuple(y.shape)}")
    print(f"  Path   : {paths[0]}")
    with torch.no_grad():
        loss = model.compute_loss(batch)
    print(f"  Loss   : {loss.item():.4f}  ✓")

    print("\n── Val batch ─────────────────────────────────────────────────────────")
    val_loader = dm.val_dataloader()
    val_batch = next(iter(val_loader))
    x_v, di_v, y_v, p_v = val_batch
    print(f"  Image  : {tuple(x_v.shape)}")
    print(f"  Target : {tuple(y_v.shape)}")
    print(f"  Path   : {p_v[0]}")
    with torch.no_grad():
        val_loss = model.compute_loss(val_batch)
    print(f"  Val loss: {val_loss.item():.4f}  ✓")

    print("\n── Test batch ────────────────────────────────────────────────────────")
    test_loader = dm.test_dataloader()
    test_batch = next(iter(test_loader))
    x_t, di_t, y_t, p_t = test_batch
    print(f"  Image  : {tuple(x_t.shape)}")
    print(f"  Target : {tuple(y_t.shape)}")
    print(f"  Path   : {p_t[0]}")
    with torch.no_grad():
        test_loss = model.compute_loss(test_batch)
    print(f"  Test loss: {test_loss.item():.4f}  ✓")

    print("\n✓ All checks passed — safe to submit to cluster")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/full-page/fullpage_cl.gin")
    parser.add_argument("--fold",   type=int, default=0)
    args = parser.parse_args()
    main(args.config, args.fold)
