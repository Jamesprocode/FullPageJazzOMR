"""
PageCropDataset — curriculum dataset for pre-computed N-system page crops.

Reads split files written by data_prep/prepare_pagecrop.py:
    <img_path> <gt_path> <N>   (one entry per line, paths relative to project root)

Curriculum stage is determined from the current epoch:
    stage = min((epoch // increase_epochs) + curriculum_start,
                curriculum_start + num_cl_stages - 1)

Only samples with N <= stage are eligible for sampling.  __len__ returns
the number of unique pages so Lightning sees a stable epoch size.
__getitem__ picks the correct stage crop via index % len(eligible).

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
        final_stage=None,           # stage at which pages use their actual max crop
        synthetic_data_path=None,   # optional: add synthetic train samples
        replay_ratio=0.0,           # fraction of each previous stage to replay during train
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
        self.final_stage = final_stage  # None means never switch to max-crop mode
        self.replay_ratio = replay_ratio

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

        n_real = len(self.samples)

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

        n_syn_loaded = len(self.samples) - n_real
        self._is_synthetic = [False] * n_real + [True] * n_syn_loaded
        print(f"PageCropDataset [{split}_{fold}]: {len(self.samples)} samples "
              f"(real={n_real}, synthetic={n_syn_loaded}, "
              f"N={sorted(set(n for _,_,n in self.samples))})")

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
        # Page ID = (dataset_prefix, numeric_id) so real and synthetic pages
        # with the same img number are not conflated (both start at img_0).
        page_max_n: dict = {}
        for img_path, _, n in self.samples:
            m = _re.search(r"(.*/)img_(\d+)_n", img_path)
            if m:
                pid = (m.group(1), int(m.group(2)))
                page_max_n[pid] = max(page_max_n.get(pid, 0), n)

        sample_pid = []
        for img_path, _, _ in self.samples:
            m = _re.search(r"(.*/)img_(\d+)_n", img_path)
            sample_pid.append((m.group(1), int(m.group(2))) if m else (-1, -1))
        self._page_max_n = page_max_n
        self._sample_pid = sample_pid

        max_n = max(n for _, _, n in self.samples)
        self._eligible_for = {}
        maxedout_counts = {}
        for stage_n in range(1, max_n + 1):
            eligible = []
            n_maxed = 0
            for i, (_, _, n) in enumerate(self.samples):
                pmn = page_max_n.get(sample_pid[i], n)
                # At final_stage+, each page uses its actual max crop (full page)
                if self.final_stage is not None and stage_n >= self.final_stage:
                    target = pmn
                else:
                    target = min(stage_n, pmn)
                if n == target:
                    eligible.append(i)
                    if target < stage_n:   # page contributed max crop, not stage crop
                        n_maxed += 1
            self._eligible_for[stage_n] = eligible
            maxedout_counts[stage_n] = n_maxed

        # Eligible count is the same at every stage (all pages have max_n >= 2).
        self._n_pages = len(self._eligible_for[self.curriculum_start])
        self._maxedout_counts = maxedout_counts
        if len(set(len(v) for v in self._eligible_for.values())) > 1:
            print(f"  WARNING: eligible count varies — possible page ID collision")

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
            from collections import Counter
            max_n = max(self._eligible_for.keys())
            clamped = min(stage, max_n)
            current_eligible = list(self._eligible_for.get(clamped, self._eligible_for[max_n]))

            # ── experience replay: add fraction of each previous stage (train only) ──
            replay_counts = {}
            if self.replay_ratio > 0.0 and clamped > 1 and self.split == "train":
                replay_eligible = []
                for s in range(1, clamped):
                    prev = self._eligible_for.get(s, [])
                    k = max(1, round(self.replay_ratio * len(prev)))
                    sampled = random.sample(prev, min(k, len(prev)))
                    replay_eligible.extend(sampled)
                    replay_counts[s] = len(sampled)
                combined = current_eligible + replay_eligible
                random.shuffle(combined)   # distribute replay evenly so __len__ sampling sees correct ratio
                self._cached_eligible = combined
            else:
                self._cached_eligible = current_eligible

            self._cached_stage = stage
            n_dist = dict(sorted(Counter(self.samples[i][2] for i in self._cached_eligible).items()))

            if replay_counts:
                print(f"  [Dataset/{self.split}] stage={stage}"
                      f"  current={len(current_eligible)}  replay={replay_counts}"
                      f"  total={len(self._cached_eligible)}  n_dist={n_dist}")
            else:
                n_real_el = sum(1 for i in self._cached_eligible if not self._is_synthetic[i])
                n_syn_el  = sum(1 for i in self._cached_eligible if self._is_synthetic[i])
                n_maxed   = self._maxedout_counts.get(clamped, 0)
                print(f"  [Dataset/{self.split}] stage={stage}  eligible={len(self._cached_eligible)}"
                      f"  (real={n_real_el}, synthetic={n_syn_el}, maxed-out={n_maxed})  n_dist={n_dist}")
            # Integrity check: every sample must have n == stage (strict) or n == page_max_n (maxed-out)
            bad = []
            for i in self._cached_eligible:
                _, _, n = self.samples[i]
                pmn = self._page_max_n.get(self._sample_pid[i], n)
                expected = pmn if (self.final_stage is not None and clamped >= self.final_stage) else min(clamped, pmn)
                if n != expected:
                    bad.append((i, n, expected, pmn))
            if bad:
                print(f"  WARNING [{self.split}] stage={stage}: {len(bad)} samples with wrong n:")
                for idx, n_got, n_exp, pmn in bad:
                    print(f"    sample {idx}  path={self.samples[idx][0]}  n={n_got}  expected={n_exp}  page_max_n={pmn}")
            # Val-specific check: warn on any sample that isn't at stage N and isn't maxed-out
            if self.split == "val":
                wrong_val = [
                    (i, self.samples[i][2], self._page_max_n.get(self._sample_pid[i], self.samples[i][2]))
                    for i in self._cached_eligible
                    if self.samples[i][2] != clamped
                    and self.samples[i][2] != self._page_max_n.get(self._sample_pid[i], self.samples[i][2])
                ]
                if wrong_val:
                    print(f"  WARNING [val] stage={stage}: {len(wrong_val)} val samples have wrong system count (not stage-N and not maxed-out):")
                    for idx, n_got, pmn in wrong_val:
                        print(f"    {self.samples[idx][0]}  systems={n_got}  stage={clamped}  page_max={pmn}")
        return self._cached_eligible

    def __len__(self) -> int:
        if self.split == "test":
            return len(self.samples)
        return self._n_pages

    def __getitem__(self, index):
        assert self.w2i is not None, "Call set_dictionaries() before iterating."

        if self.split == "test":
            # Iterate through all samples in order.
            idx = index
        else:
            # Train + val: index into eligible set.
            # DataLoader shuffle=True randomises order for train.
            eligible = self._stage_eligible()
            idx = eligible[index % len(eligible)]  # % is a safety net only

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
