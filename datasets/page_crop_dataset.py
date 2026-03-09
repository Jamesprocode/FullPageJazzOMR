"""
PageCropDataset — curriculum dataset for pre-computed N-system page crops.

Reads split files written by data_prep/prepare_pagecrop.py:
    <img_path> <gt_path> <N>   (one entry per line, paths relative to project root)

Curriculum stage is determined from the current epoch:
    stage = min((epoch // increase_epochs) + curriculum_start,
                curriculum_start + num_cl_stages - 1)

Only samples with N <= stage are eligible for sampling.  __len__ returns
dataset_length regardless of eligibility so Lightning sees a fixed-size
dataset per epoch; eligible samples are chosen randomly in __getitem__.

Usage in train.py:
    train_set = PageCropDataset(data_path, split="train", fold=0)
    val_set   = PageCropDataset(data_path, split="val",   fold=0)
    w2i, i2w  = check_and_retrieveVocabulary(
                    [train_set.get_gt(), val_set.get_gt()], "vocab", "vocab_cl")
    train_set.set_dictionaries(w2i, i2w)
    val_set.set_dictionaries(w2i, i2w)

    model.set_stage(train_set.curriculum_stage_beginning)
    model.set_stage_calculator(train_set.get_stage_calculator())

    # In an on_train_epoch_start callback:
    train_set.set_epoch(trainer.current_epoch)
"""

import random
from pathlib import Path

import cv2
import gin
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from jazzmus.dataset.data_preprocessing import augment, convert_img_to_tensor
from jazzmus.dataset.tokenizer import process_text

TOKENIZER_TYPE = "medium"


@gin.configurable
class PageCropDataset(Dataset):
    """
    Curriculum dataset backed by pre-computed N-system page crops.

    Args:
        data_path:        path to jazzmus_pagecrop/ directory (contains splits/)
        split:            "train", "val", or "test"
        fold:             fold index (int)
        fixed_img_height: if set, ALL images are scaled to exactly this height.
                          Prefer system_height instead (scales per-system uniformly).
        system_height:    if set (and fixed_img_height is None), each image is scaled
                          so that each system row is system_height pixels tall.
                          Target height for an N-system crop = N * system_height.
        augment:          apply random augmentation (train only)
        increase_epochs:  epochs per curriculum stage
        num_cl_stages:    total number of curriculum stages
        curriculum_start: N value at stage 1 (always 1)
        dataset_length:   virtual dataset size (number of __getitem__ calls per epoch)
    """

    def __init__(
        self,
        data_path,
        split,
        fold,
        fixed_img_height=None,
        system_height=None,
        augment=False,
        increase_epochs=50,
        num_cl_stages=2,
        curriculum_start=1,
        dataset_length=3500,
        synthetic_data_path=None,   # optional: add synthetic train samples
    ):
        super().__init__()
        self.split = split
        self.fold = fold
        self.fixed_img_height = fixed_img_height
        self.system_height = system_height
        self.do_augment = augment
        self.increase_epochs = increase_epochs
        self.num_cl_stages = num_cl_stages
        self.curriculum_start = curriculum_start
        self.dataset_length = dataset_length

        self.epoch = 0
        # Shared tensor so DataLoader worker processes see stage updates without
        # needing persistent_workers=False (which kills/respawns workers each epoch).
        # _shared_stage[0] == -1 means "use epoch-based calculation".
        self._shared_stage = torch.tensor([-1], dtype=torch.int32).share_memory_()
        # Per-worker cache: eligible list is recomputed only when stage changes.
        self._cached_stage: int = -1
        self._cached_eligible: list = []
        self.w2i = None
        self.i2w = None
        self.padding_token = 0
        self.teacher_forcing_error_rate = 0.2

        # ── load split file ──────────────────────────────────────────────────
        data_path = Path(data_path)
        split_file = data_path / "splits" / f"{split}_{fold}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        # Paths in the split file are relative to data_path.parent (e.g. "data/")
        path_base = data_path.parent

        self.samples = []      # list of (img_path_str, gt_path_str, N)
        with open(split_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                raw_img, raw_gt, n = parts[0], parts[1], int(parts[2])
                # Resolve paths: try as-is first, then relative to path_base
                img_path = Path(raw_img)
                gt_path  = Path(raw_gt)
                if not img_path.exists():
                    img_path = path_base / raw_img
                if not gt_path.exists():
                    gt_path = path_base / raw_gt
                self.samples.append((str(img_path), str(gt_path), n))

        # ── optionally load synthetic samples (train only) ────────────────────
        if synthetic_data_path is not None and split == "train":
            syn_path = Path(synthetic_data_path)
            syn_file = syn_path / "splits" / f"train_{fold}.txt"
            syn_base = syn_path.parent
            if syn_file.exists():
                n_syn = 0
                with open(syn_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 3:
                            continue
                        raw_img, raw_gt, n = parts[0], parts[1], int(parts[2])
                        img_path = Path(raw_img)
                        gt_path  = Path(raw_gt)
                        if not img_path.exists():
                            img_path = syn_base / raw_img
                        if not gt_path.exists():
                            gt_path = syn_base / raw_gt
                        self.samples.append((str(img_path), str(gt_path), n))
                        n_syn += 1
                print(f"  + {n_syn} synthetic samples from {syn_file}")
            else:
                print(f"  Warning: synthetic split not found: {syn_file}")

        print(f"PageCropDataset [{split}_{fold}]: {len(self.samples)} samples "
              f"(N={sorted(set(n for _,_,n in self.samples))})")

        # ── pre-load and tokenize GT ─────────────────────────────────────────
        # GT files contain raw kern with !!linebreak:original between systems.
        # process_text drops !!linebreak:original, so we split on it first,
        # tokenize each system block separately, and insert <linebreak> manually.
        self._gt_cache = {}   # gt_path → token list
        for _, gt_path, _ in self.samples:
            if gt_path not in self._gt_cache:
                with open(gt_path) as f:
                    kern_lines = f.readlines()

                # Split into per-system blocks
                blocks, current = [], []
                for line in kern_lines:
                    if line.strip() == "!!linebreak:original":
                        blocks.append(current)
                        current = []
                    else:
                        current.append(line)
                if current:
                    blocks.append(current)

                # Tokenize each block and join with <linebreak>
                tokens = ["<bos>"]
                for i, block in enumerate(blocks):
                    if i > 0:
                        tokens.append("<linebreak>")
                    tokens += process_text(lines=block, tokenizer_type=TOKENIZER_TYPE)
                tokens.append("<eos>")

                self._gt_cache[gt_path] = tokens

        # gt_tokens: list of token lists (parallel to self.samples)
        self.gt_tokens = [self._gt_cache[gt] for _, gt, _ in self.samples]

        # ── pre-compute per-N eligible indices ───────────────────────────────
        # Eligibility rule:
        #   Stage 1       → only 1-system crops (strict).
        #   Stage N >= 2  → pages with max_n >= N  contribute their N-system crop
        #                   (strict); pages with 2 <= max_n < N contribute their
        #                   best available crop (they have "maxed out").
        #                   1-system-only pages never appear again after stage 1.
        import re as _re
        page_max_n: dict = {}
        for img_path, _, n in self.samples:
            m = _re.search(r"img_(\d+)_n", img_path)
            if m:
                pid = int(m.group(1))
                page_max_n[pid] = max(page_max_n.get(pid, 0), n)

        sample_pid = []
        for img_path, _, _ in self.samples:
            m = _re.search(r"img_(\d+)_n", img_path)
            sample_pid.append(int(m.group(1)) if m else -1)

        max_n = max(n for _, _, n in self.samples)
        self._eligible_for = {}
        for stage_n in range(1, max_n + 1):
            eligible = []
            for i, (_, _, n) in enumerate(self.samples):
                pmn = page_max_n.get(sample_pid[i], n)
                target = min(stage_n, pmn)
                # 1-system-only pages only appear at stage 1
                if n == target and (pmn >= 2 or stage_n == 1):
                    eligible.append(i)
            self._eligible_for[stage_n] = eligible

    # ── curriculum stage API ─────────────────────────────────────────────────

    def get_stage(self, epoch: int) -> int:
        s = int(self._shared_stage[0])
        if s != -1:
            return s
        stage = (epoch // self.increase_epochs) + self.curriculum_start
        return min(stage, self.curriculum_start + self.num_cl_stages - 1)

    def set_stage_direct(self, stage: int):
        """Override epoch-based stage with a fixed value (for dynamic curriculum).

        Writes to a shared-memory tensor so persistent DataLoader workers see the
        updated stage without needing to be respawned each epoch.
        """
        self._shared_stage[0] = stage

    def get_stage_calculator(self):
        """Return a callable epoch → stage for use by CurriculumSMTTrainer."""
        return self.get_stage

    @property
    def curriculum_stage_beginning(self) -> int:
        return self.get_stage(0)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    # ── vocab API ────────────────────────────────────────────────────────────

    def get_gt(self):
        """Return list of token lists for vocabulary building."""
        return self.gt_tokens

    def set_dictionaries(self, w2i: dict, i2w: dict):
        self.w2i = w2i
        self.i2w = i2w
        self.padding_token = w2i["<pad>"]

    def get_dictionaries(self):
        return self.w2i, self.i2w

    def get_i2w(self):
        return self.i2w

    def vocab_size(self) -> int:
        return len(self.w2i)

    # ── size estimates (for model PE sizing) ─────────────────────────────────

    def get_max_hw(self):
        """
        Return (max_height, max_width) across all samples.

        Uses PIL header-only reads (no pixel decoding) for speed.
        Applies the same resize logic as __getitem__.
        """
        max_h, max_w = 0, 0
        for img_path, _, n in self.samples:
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception:
                continue
            h, w = self._scaled_hw(h, w, n)
            max_h = max(max_h, h)
            max_w = max(max_w, w)
        return max_h, max_w

    def _scaled_hw(self, h: int, w: int, n: int):
        """Return (h, w) after applying the same scaling as __getitem__."""
        if self.fixed_img_height is not None:
            target_h = self.fixed_img_height
        elif self.system_height is not None:
            target_h = n * self.system_height
        else:
            return h, w
        if h != target_h:
            w = max(1, int(round(w * target_h / h)))
            h = target_h
        return h, w

    def get_max_seqlen(self) -> int:
        return max(len(tokens) for tokens in self.gt_tokens)

    # ── dataset interface ────────────────────────────────────────────────────

    def _stage_eligible(self):
        """Return eligible sample indices for the current curriculum stage.

        Result is cached per stage — recomputed only when the stage changes,
        not on every __getitem__ call.
        """
        stage = self.get_stage(self.epoch)
        if stage != self._cached_stage:
            max_n = max(self._eligible_for.keys())
            clamped = min(stage, max_n)
            self._cached_eligible = self._eligible_for.get(clamped, self._eligible_for[max_n])
            self._cached_stage = stage
        return self._cached_eligible

    def __len__(self) -> int:
        # Train: real eligible set size — DataLoader shuffle=True handles ordering.
        # Val:   one N-system crop per page at the current curriculum stage.
        # Test:  all samples (full-page crops only), no stage filtering.
        if self.split == "test":
            return len(self.samples)
        return len(self._stage_eligible())

    def __getitem__(self, index):
        assert self.w2i is not None, "Call set_dictionaries() before iterating."

        if self.split == "test":
            # Iterate through all samples in order.
            idx = index
        else:
            # Train + val: deterministic index into eligible set.
            # DataLoader shuffle=True randomises order for train.
            eligible = self._stage_eligible()
            idx = eligible[index % len(eligible)]

        img_path, gt_path, n = self.samples[idx]
        tokens = self.gt_tokens[idx]

        # Load and optionally resize image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback: white image
            img = np.full((128, 128), 255, dtype=np.uint8)
        h, w = img.shape[:2]
        target_h, target_w = self._scaled_hw(h, w, n)
        if target_h != h:
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        if self.do_augment:
            x = augment(img)
        else:
            x = convert_img_to_tensor(img)

        # Encode GT tokens
        y = torch.from_numpy(
            np.asarray([self.w2i.get(t, self.padding_token) for t in tokens])
        )
        # Teacher forcing errors only during training — val/test use clean decoder input
        if self.split == "train":
            decoder_input = self._apply_teacher_forcing(y)
        else:
            decoder_input = y
        return x, decoder_input, y, img_path

    def _apply_teacher_forcing(self, sequence: torch.Tensor) -> torch.Tensor:
        errored = sequence.clone()
        vocab_size = len(self.w2i)
        for i in range(1, len(sequence)):
            if (random.random() < self.teacher_forcing_error_rate
                    and sequence[i] != self.padding_token):
                errored[i] = random.randint(0, vocab_size - 1)
        return errored
