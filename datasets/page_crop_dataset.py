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
from torch.utils.data import Dataset

from jazzmus.dataset.data_preprocessing import augment, convert_img_to_tensor
from jazzmus.dataset.tokenizer import process_text

TOKENIZER_TYPE = "medium"


@gin.configurable
class PageCropDataset(Dataset):
    """
    Curriculum dataset backed by pre-computed N-system page crops.

    Args:
        data_path:       path to jazzmus_pagecrop/ directory (contains splits/)
        split:           "train", "val", or "test"
        fold:            fold index (int)
        fixed_img_height: target height in pixels when loading images.
                         Set to None to keep original stacked height (N × system_height).
        augment:         apply random augmentation (train only)
        increase_epochs: epochs per curriculum stage
        num_cl_stages:   total number of curriculum stages
        curriculum_start: N value at stage 1 (always 1)
        dataset_length:  virtual dataset size (number of __getitem__ calls per epoch)
    """

    def __init__(
        self,
        data_path,
        split,
        fold,
        fixed_img_height=None,
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
        self.do_augment = augment
        self.increase_epochs = increase_epochs
        self.num_cl_stages = num_cl_stages
        self.curriculum_start = curriculum_start
        self.dataset_length = dataset_length

        self.epoch = 0
        self._direct_stage = None
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
        # At stage S, each page contributes its min(S, page_max_n) crop —
        # i.e. the S-system crop if available, else the full page of that piece.
        # Build a lookup: page_id → max N available
        import re as _re
        page_max_n: dict = {}
        for img_path, _, n in self.samples:
            m = _re.search(r"img_(\d+)_n", img_path)
            if m:
                pid = int(m.group(1))
                page_max_n[pid] = max(page_max_n.get(pid, 0), n)

        # For each sample, record its page id
        sample_page = []
        for img_path, _, _ in self.samples:
            m = _re.search(r"img_(\d+)_n", img_path)
            sample_page.append(int(m.group(1)) if m else -1)

        max_n = max(n for _, _, n in self.samples)
        self._eligible_for = {}
        for stage_n in range(1, max_n + 1):
            eligible = []
            for i, (_, _, n) in enumerate(self.samples):
                pid = sample_page[i]
                target = min(stage_n, page_max_n.get(pid, stage_n))
                if n == target:
                    eligible.append(i)
            self._eligible_for[stage_n] = eligible

    # ── curriculum stage API ─────────────────────────────────────────────────

    def get_stage(self, epoch: int) -> int:
        if self._direct_stage is not None:
            return self._direct_stage
        stage = (epoch // self.increase_epochs) + self.curriculum_start
        return min(stage, self.curriculum_start + self.num_cl_stages - 1)

    def set_stage_direct(self, stage: int):
        """Override epoch-based stage with a fixed value (for dynamic curriculum)."""
        self._direct_stage = stage

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

        Scans image headers without decoding full images for speed.
        Falls back to reading full images if needed.
        """
        max_h, max_w = 0, 0
        for img_path, _, _ in self.samples:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            h, w = img.shape[:2]
            if self.fixed_img_height is not None and h != self.fixed_img_height:
                w = max(1, int(round(w * self.fixed_img_height / h)))
                h = self.fixed_img_height
            max_h = max(max_h, h)
            max_w = max(max_w, w)
        return max_h, max_w

    def get_max_seqlen(self) -> int:
        return max(len(tokens) for tokens in self.gt_tokens)

    # ── dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, index):
        assert self.w2i is not None, "Call set_dictionaries() before iterating."

        # Select eligible sample for current curriculum stage
        stage = self.get_stage(self.epoch)
        max_n = max(self._eligible_for.keys())
        clamped = min(stage, max_n)
        eligible = self._eligible_for.get(clamped, self._eligible_for[max_n])
        idx = random.choice(eligible)

        img_path, gt_path, n = self.samples[idx]
        tokens = self.gt_tokens[idx]

        # Load and optionally resize image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback: white image
            img = np.full((128, 128), 255, dtype=np.uint8)
        if self.fixed_img_height is not None:
            h, w = img.shape[:2]
            if h != self.fixed_img_height:
                new_w = max(1, int(round(w * self.fixed_img_height / h)))
                img = cv2.resize(img, (new_w, self.fixed_img_height),
                                 interpolation=cv2.INTER_LINEAR)

        if self.do_augment:
            x = augment(img)
        else:
            x = convert_img_to_tensor(img)

        # Encode GT tokens
        y = torch.from_numpy(
            np.asarray([self.w2i.get(t, self.padding_token) for t in tokens])
        )
        decoder_input = self._apply_teacher_forcing(y)
        return x, decoder_input, y, img_path

    def _apply_teacher_forcing(self, sequence: torch.Tensor) -> torch.Tensor:
        errored = sequence.clone()
        vocab_size = len(self.w2i)
        for i in range(1, len(sequence)):
            if (random.random() < self.teacher_forcing_error_rate
                    and sequence[i] != self.padding_token):
                errored[i] = random.randint(0, vocab_size - 1)
        return errored
