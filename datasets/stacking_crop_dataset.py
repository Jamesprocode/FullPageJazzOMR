"""
StackingCropDataset — curriculum dataset for pre-computed stacked page images.

Unlike PageCropDataset (designed for masking), this dataset uses **strict
n-matching**: at stage N, only samples with exactly n == N are in the current
pool.  There is no "maxed-out" logic — samples at different n values are
independent synthetic images, not different crops of the same page.

Replay adds replay_ratio * count(n==k) samples from each prior stage k < N.

Split file format (same as PageCropDataset):
    <img_path> <gt_path> <N>   (one entry per line)

Usage:
    train_set = StackingCropDataset(data_path, split="train", fold=0)
"""

import random
from collections import Counter, defaultdict
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
class StackingCropDataset(Dataset):
    """
    Curriculum dataset for pre-computed stacked pages with strict n-matching.

    At stage N (train):
        current = all samples with n == N
        replay  = replay_ratio * count(n==k) for each k in 1..N-1
    At stage N (val):
        only samples with n == N (no replay)

    Args:
        data_path:        path to jazzmus_stacked/ directory (contains splits/)
        split:            "train", "val", or "test"
        fold:             fold index (int)
        system_height:    each image is scaled so each system row = system_height px.
                          Target height for an N-system image = N * system_height.
        augment:          apply random augmentation (train only)
        num_cl_stages:    total number of curriculum stages
        curriculum_start: N value at stage 1 (always 1)
        final_stage:      stage at which full pages are included (optional)
        replay_ratio:     fraction of each previous stage to replay during train
        epoch_size:       if set, cap the number of samples drawn per epoch
    """

    def __init__(
        self,
        data_path,
        split,
        fold,
        system_height=256,
        augment=False,
        num_cl_stages=9,
        curriculum_start=1,
        final_stage=None,
        replay_ratio=0.25,
        epoch_size=None,
    ):
        super().__init__()
        self.split = split
        self.fold = fold
        self.system_height = system_height
        self.do_augment = augment
        self.num_cl_stages = num_cl_stages
        self.curriculum_start = curriculum_start
        self.final_stage = final_stage
        self.replay_ratio = replay_ratio
        self.epoch_size = epoch_size

        self.epoch = 0
        self._shared_stage = torch.tensor([-1], dtype=torch.int32).share_memory_()
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

        path_base = data_path.parent

        self.samples = []      # list of (img_path_str, gt_path_str, N)
        with open(split_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                raw_img, raw_gt, n = parts[0], parts[1], int(parts[2])
                img_path = self._resolve(raw_img, path_base)
                gt_path  = self._resolve(raw_gt,  path_base)
                self.samples.append((str(img_path), str(gt_path), n))

        print(f"StackingCropDataset [{split}_{fold}]: {len(self.samples)} samples "
              f"(N={sorted(set(n for _,_,n in self.samples))})")

        # ── pre-load and tokenize GT ─────────────────────────────────────────
        self._gt_cache = {}
        for _, gt_path, _ in self.samples:
            if gt_path not in self._gt_cache:
                with open(gt_path) as f:
                    kern_lines = f.readlines()

                blocks, current = [], []
                for line in kern_lines:
                    if line.strip() == "!!linebreak:original":
                        blocks.append(current)
                        current = []
                    else:
                        current.append(line)
                if current:
                    blocks.append(current)

                tokens = ["<bos>"]
                for i, block in enumerate(blocks):
                    if i > 0:
                        tokens.append("<linebreak>")
                    tokens += process_text(lines=block, tokenizer_type=TOKENIZER_TYPE)
                tokens.append("<eos>")

                self._gt_cache[gt_path] = tokens

        self.gt_tokens = [self._gt_cache[gt] for _, gt, _ in self.samples]

        # ── build per-N index lists (strict matching, no maxed-out) ──────────
        self._per_n: dict[int, list[int]] = defaultdict(list)
        for i, (_, _, n) in enumerate(self.samples):
            self._per_n[n].append(i)

        dist = {n: len(indices) for n, indices in sorted(self._per_n.items())}
        print(f"  Per-N distribution: {dist}")

    # ── path resolution ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve(raw: str, base_dir: Path) -> Path:
        p = Path(raw)
        if p.exists():
            return p
        candidate = base_dir / p
        if candidate.exists():
            return candidate
        if len(p.parts) > 1:
            stripped = base_dir / Path(*p.parts[1:])
            if stripped.exists():
                return stripped
        return candidate

    # ── curriculum stage API ─────────────────────────────────────────────────

    def get_stage(self, epoch: int = None) -> int:
        s = int(self._shared_stage[0])
        if s != -1:
            return s
        return self.curriculum_start

    def set_stage_direct(self, stage: int):
        self._shared_stage[0] = stage

    def get_stage_calculator(self):
        return self.get_stage

    @property
    def curriculum_stage_beginning(self) -> int:
        return self.get_stage(0)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    # ── vocab API ────────────────────────────────────────────────────────────

    def get_gt(self):
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

    # ── size estimates ───────────────────────────────────────────────────────

    def get_max_hw(self):
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
        target_h = n * self.system_height
        if h != target_h:
            w = max(1, int(round(w * target_h / h)))
            h = target_h
        return h, w

    def get_max_seqlen(self) -> int:
        return max(len(tokens) for tokens in self.gt_tokens)

    # ── eligibility (strict n-matching) ──────────────────────────────────────

    def _stage_eligible(self):
        """Return eligible sample indices for the current curriculum stage.

        Strict n-matching: current = n == stage, replay = ratio of each prior n.
        No maxed-out logic — each n-value is an independent pool.
        """
        stage = self.get_stage(self.epoch)
        if stage != self._cached_stage:
            max_n = max(self._per_n.keys())
            clamped = min(stage, max_n)

            # Current stage: only samples with n == clamped
            current_eligible = list(self._per_n.get(clamped, []))

            # Val: strict current stage only, no replay
            if self.split == "val":
                self._cached_eligible = current_eligible
                self._cached_stage = stage
                n_dist = dict(sorted(Counter(
                    self.samples[i][2] for i in self._cached_eligible
                ).items()))
                print(f"  [Dataset/val] stage={stage}  "
                      f"eligible={len(current_eligible)}  n_dist={n_dist}")
                return self._cached_eligible

            # Train: current + explicit replay from prior stages
            replay_counts = {}
            if self.replay_ratio > 0.0 and clamped > 1:
                replay_eligible = []
                for s in range(1, clamped):
                    prev = self._per_n.get(s, [])
                    k = max(1, round(self.replay_ratio * len(prev)))
                    sampled = random.sample(prev, min(k, len(prev)))
                    replay_eligible.extend(sampled)
                    replay_counts[s] = len(sampled)
                combined = current_eligible + replay_eligible
                random.shuffle(combined)
                self._cached_eligible = combined
            else:
                self._cached_eligible = current_eligible

            self._cached_stage = stage
            n_dist = dict(sorted(Counter(
                self.samples[i][2] for i in self._cached_eligible
            ).items()))

            if replay_counts:
                print(f"  [Dataset/train] stage={stage}"
                      f"  current={len(current_eligible)}  replay={replay_counts}"
                      f"  total={len(self._cached_eligible)}  n_dist={n_dist}")
            else:
                print(f"  [Dataset/train] stage={stage}"
                      f"  eligible={len(current_eligible)}  n_dist={n_dist}")

        return self._cached_eligible

    # ── dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        if self.split == "test":
            return len(self.samples)
        eligible = self._stage_eligible()
        n = len(eligible)
        if self.epoch_size is not None and self.split == "train":
            n = min(n, self.epoch_size)
        return n

    def __getitem__(self, index):
        assert self.w2i is not None, "Call set_dictionaries() before iterating."

        if self.split == "test":
            idx = index
        else:
            eligible = self._stage_eligible()
            idx = eligible[index % len(eligible)]

        img_path, gt_path, n = self.samples[idx]
        tokens = self.gt_tokens[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.full((self.system_height, 128), 255, dtype=np.uint8)
            n = 1
        h, w = img.shape[:2]
        target_h, target_w = self._scaled_hw(h, w, n)
        if target_h != h:
            img = cv2.resize(img, (target_w, target_h),
                             interpolation=cv2.INTER_LINEAR)

        if self.do_augment:
            x = augment(img)
        else:
            x = convert_img_to_tensor(img)

        y = torch.from_numpy(
            np.asarray([self.w2i.get(t, self.padding_token) for t in tokens])
        )
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
