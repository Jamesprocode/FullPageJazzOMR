"""
Curriculum-aware SMT trainer.

Subclasses the base SMT_Trainer without touching it, adding only:
  - set_stage() / set_stage_calculator()  for curriculum stage tracking
  - stage logging to wandb on every training step

All system-level and existing full-page training scripts continue to use
the original jazzmus.smt_trainer.SMT_Trainer unchanged.
"""

from typing import Callable

import torch
import transformers
import wandb
from jazzmus.smt_trainer import SMT_Trainer

try:
    from chord_metrics import (
        extract_tokens_from_mxhm,
        compute_page_chord_metrics,
        aggregate_page_chord_metrics,
    )
    from inference import extract_spines
    _HAS_CHORD_METRICS = True
except ImportError:
    _HAS_CHORD_METRICS = False


class CurriculumSMTTrainer(SMT_Trainer):
    """
    Extends SMT_Trainer with curriculum stage tracking.

    Usage in train_fullpage_cl.py:
        model = CurriculumSMTTrainer.load_from_checkpoint(
            ckpt_path, maxh=..., maxw=..., maxlen=..., strict=False, ...
        )
        model.set_stage(datamodule.train_set.curriculum_stage_beginning)
        model.set_stage_calculator(datamodule.train_set.get_stage_calculator())
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_stage: int = 2
        self._stage_calculator: Callable[[int], int] = lambda step: self.current_stage
        self._val_sample: dict | None = None   # stores first val batch for image logging
        self._last_logged_stage: int = -1      # track stage changes for training image log

    # ── lr schedule: warmup + constant ───────────────────────────────────────

    def configure_optimizers(self):
        params = [
            {
                "params": (
                    p for p in self.parameters() if p.requires_grad and p.ndim >= 2
                ),
                "weight_decay": self.weight_decay,
            },
            {
                "params": (
                    p for p in self.parameters() if p.requires_grad and p.ndim <= 1
                ),
                "weight_decay": 0,
            },
        ]
        optimizer = torch.optim.AdamW(params, lr=self.lr)
        scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps
        )
        return dict(
            optimizer=optimizer,
            lr_scheduler={"scheduler": scheduler, "interval": "step"},
        )

    # ── stage control ──────────────────────────────────────────────────────────

    def set_stage(self, stage: int):
        self.current_stage = stage

    def set_stage_calculator(self, calc: Callable[[int], int]):
        """
        Inject an epoch → stage calculator so training_step can log the
        current curriculum stage to wandb.
        """
        self._stage_calculator = calc

    # ── training / validation step overrides ──────────────────────────────────

    def on_load_checkpoint(self, checkpoint):
        """Drop weights whose shape doesn't match the current model.

        Handles vocab-size mismatches between the pretrained checkpoint
        (e.g. 20578 tokens) and the curriculum model (jazz vocab, 154 tokens).
        Mismatched layers (embedding, output Conv1d) are skipped so they keep
        their random initialisation; all other weights load normally.
        """
        state = checkpoint["state_dict"]
        own   = self.state_dict()
        filtered = {k: v for k, v in state.items()
                    if k in own and v.shape == own[k].shape}
        skipped = [k for k in state if k not in filtered]
        if skipped:
            print(f"  Skipping {len(skipped)} mismatched weights: {skipped}")
        checkpoint["state_dict"] = filtered

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)

        stage = self._stage_calculator(self.current_epoch)
        self.log("curriculum/stage", float(stage), on_step=True, prog_bar=True)

        # Log a training sample image whenever the curriculum stage changes.
        if batch_idx == 0 and stage != self._last_logged_stage:
            self._last_logged_stage = stage
            x = batch[0]
            img_np = x[0].squeeze().cpu().numpy()
            self.logger.experiment.log({
                "curriculum/train_input": wandb.Image(
                    img_np,
                    caption=f"stage={stage}  epoch={self.current_epoch}",
                ),
            })

        return loss

    def validation_step(self, batch, batch_idx):
        (x, di, y, path_to_images) = batch
        loss = self.compute_loss(batch)
        self.log("val/loss", loss, on_epoch=True, batch_size=x.shape[0], prog_bar=True)

        # Store first batch image for prediction table logging
        if batch_idx == 0:
            self._val_sample = {"x": x[0].squeeze().cpu().numpy(),
                                "path": path_to_images[0]}

        # Cap greedy-decode steps proportionally to the current curriculum stage.
        # model.maxlen is sized for the worst case (5 stacked systems + 10 % buffer).
        # At stage 2 the GT is only ~2/5 of that length, so decoding to the full
        # maxlen wastes ~3× the time.  We cap at (stage × 550) tokens which gives
        # generous headroom (~50 % above the expected per-system token count of ~346).
        stage = int(self._stage_calculator(self.current_epoch))
        capped_maxlen = min(self.model.maxlen, max(512, stage * 550))
        old_maxlen = self.model.maxlen
        self.model.maxlen = capped_maxlen
        try:
            self.predict_output(batch)
        finally:
            self.model.maxlen = old_maxlen

    # ── chord metrics ──────────────────────────────────────────────────────────

    def _log_chord_metrics(self, preds, grtrs, step):
        """Compute chord edit-distance metrics and log them to WandB.

        Requires chord_metrics.py and inference.py at the project root.
        Silently skips any sample whose prediction is malformed (e.g. early
        in training before the model produces valid kern output).

        Logged metrics (all 0-100 %):
          {step}/chord_ser  – chord-only Symbol Error Rate (no dots)
          {step}/root_ser   – root-note Symbol Error Rate
        """
        if not _HAS_CHORD_METRICS or not preds:
            return

        page_metrics_list = []
        for pred, gt in zip(preds, grtrs):
            try:
                pred_spines = extract_spines(pred)
                gt_spines   = extract_spines(gt)
                if "**mxhm" not in pred_spines or "**mxhm" not in gt_spines:
                    continue
                pred_tokens = extract_tokens_from_mxhm(pred_spines["**mxhm"])
                gt_tokens   = extract_tokens_from_mxhm(gt_spines["**mxhm"])
                if pred_tokens and gt_tokens:
                    page_metrics_list.append(
                        compute_page_chord_metrics(pred_tokens, gt_tokens)
                    )
            except Exception:
                pass

        if not page_metrics_list:
            return

        agg = aggregate_page_chord_metrics(page_metrics_list)
        self.log(f"{step}/chord_ser", agg["agg_ser_no_dots"],
                 on_epoch=True, prog_bar=True)
        self.log(f"{step}/root_ser",  agg["agg_root_ser"],
                 on_epoch=True, prog_bar=False)

    def on_validation_epoch_end(self):
        # Guard: Lightning 2.x can call this hook twice (before and after callbacks),
        # the second time with empty preds/grtrs after super() already cleared them.
        if not self.preds:
            return

        # Chord metrics must be computed before super() clears self.preds / self.grtrs
        self._log_chord_metrics(self.preds, self.grtrs, step="val")

        # Log val prediction sample for the first val batch
        if self._val_sample is not None and self.preds:
            stage = int(self._stage_calculator(self.current_epoch))
            self.logger.experiment.log({
                "val/prediction_sample": wandb.Html(
                    f"<b>Stage:</b> {stage}  <b>Epoch:</b> {self.current_epoch}<br>"
                    f"<b>Path:</b> {self._val_sample['path']}<br><br>"
                    f"<b>Prediction:</b><pre>{self.preds[0]}</pre>"
                    f"<b>Ground Truth:</b><pre>{self.grtrs[0]}</pre>"
                )
            })
            self._val_sample = None

        super().on_validation_epoch_end()

    def on_test_epoch_end(self):
        self._log_chord_metrics(self.preds, self.grtrs, step="test")
        super().on_test_epoch_end()
