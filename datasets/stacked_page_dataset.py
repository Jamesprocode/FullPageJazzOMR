"""
StackedPageDataset — on-the-fly curriculum dataset for full-page jazz OMR.

Epoch composition at stage N:
  - dataset_length  stacked samples at n=N  (current stage)
  - replay_ratio × dataset_length  stacked samples at each prior n=1..N-1
  - all real pagecrop images (every N value)  — only added at final_stage

__len__ is dynamic: it grows each time the curriculum advances, exactly like
PageCropDataset.  DynamicCurriculumAdvancer reads len() after each stage
advance to update Lightning's max_batches.

Uses a shared-memory tensor so stage changes from DynamicCurriculumAdvancer
reach persistent DataLoader workers without respawning.
"""

import random
from pathlib import Path

import cv2
import gin
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.stacking import stack_systems
from jazzmus.dataset.data_preprocessing import augment, convert_img_to_tensor
from jazzmus.dataset.smt_dataset_utils import load_kern
from jazzmus.dataset.tokenizer import process_text

TOKENIZER_TYPE = "medium"


@gin.configurable
class StackedPageDataset(Dataset):
    """
    On-the-fly curriculum dataset: stacks jazzmus_systems images at train time.

    Epoch composition at stage N (train only):
        dataset_length                         stacked samples at n=N
        + replay_ratio×dataset_length          stacked at each n=1..N-1
        + all real pagecrop entries (if N >= final_stage)

    Val always generates exactly n=stage systems, fixed dataset_length samples.

    Args:
        system_data_path: path to jazzmus_systems/ root (contains splits/)
        split:            "train" or "val"  (test uses PageCropDataset)
        fold:             dataset fold index
        system_height:    every system row scaled to this height (px)
        num_cl_stages:    total curriculum stages; also the max N to stack
        final_stage:      from this stage, add all real pagecrop to the epoch
        real_data_path:   path to jazzmus_pagecrop/ for final-stage real mixing
        replay_ratio:     fraction of dataset_length to replay per prior stage
        width_tolerance:  max relative width difference for system pairing
        augment:          random image augmentation (train only)
        dataset_length:   base epoch size (current-stage stacked samples)
    """

    def __init__(
        self,
        system_data_path,
        split,
        fold,
        system_height=256,
        num_cl_stages=9,
        final_stage=9,
        real_data_path=None,
        replay_ratio=0.25,
        width_tolerance=0.15,
        augment=False,
        dataset_length=2000,
    ):
        super().__init__()
        self.split = split
        self.fold = fold
        self.system_height = system_height
        self.num_cl_stages = num_cl_stages
        self.final_stage = final_stage
        self.replay_ratio = replay_ratio
        self.do_augment = augment
        self.width_tolerance = width_tolerance
        self.teacher_forcing_error_rate = 0.2
        self._dataset_length = dataset_length

        # Shared-memory stage tensor — visible to all persistent DataLoader workers
        self._shared_stage = torch.tensor([1], dtype=torch.int32).share_memory_()
        # Per-worker eligible list cache: only rebuilt when stage changes
        self._cached_stage: int  = -1
        self._cached_eligible: list = []

        self.w2i = None
        self.i2w = None
        self.padding_token = 0

        # ── load and resize system images ─────────────────────────────────────
        system_data_path = Path(system_data_path)
        split_file = system_data_path / "splits" / f"{split}_{fold}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        # Paths like "data/jazzmus_systems/jpg/img_0_0.jpg" are relative to
        # the ISMIR-Jazzmus/ root = grandparent of jazzmus_systems/.
        base_dir = system_data_path.parent.parent

        self.system_x: list   = []   # numpy (system_height, W) arrays, pre-resized
        self._raw_gt:  list   = []   # raw kern lines per system
        self._sys_paths: list = []
        n_skipped = 0

        with open(split_file) as f:
            entries = [l.strip().split() for l in f if l.strip()]

        for parts in entries:
            if len(parts) < 2:
                continue
            raw_img, raw_gt = parts[0], parts[1]
            img_path = Path(raw_img) if Path(raw_img).exists() else base_dir / raw_img
            gt_path  = Path(raw_gt)  if Path(raw_gt).exists()  else base_dir / raw_gt
            if not img_path.exists() or not gt_path.exists():
                n_skipped += 1
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                n_skipped += 1
                continue

            h, w = img.shape[:2]
            if h != system_height:
                new_w = max(1, int(round(w * system_height / h)))
                img = cv2.resize(img, (new_w, system_height),
                                 interpolation=cv2.INTER_LINEAR)

            self.system_x.append(img)
            self._raw_gt.append(load_kern(str(gt_path)))
            self._sys_paths.append(str(img_path))

        print(f"StackedPageDataset [{split}_{fold}]: "
              f"{len(self.system_x)} systems loaded  ({n_skipped} skipped)")
        if not self.system_x:
            raise RuntimeError(f"No systems loaded from {split_file}")

        # ── pre-compute per-system width-compatible pools ─────────────────────
        widths = np.array([img.shape[1] for img in self.system_x])
        self._compatible = [
            [j for j in range(len(widths))
             if abs(int(widths[j]) - int(widths[i])) / max(int(widths[i]), 1)
             <= width_tolerance]
            for i in range(len(widths))
        ]
        pool_sizes = [len(c) for c in self._compatible]
        print(f"  Width pools — min: {min(pool_sizes)}  "
              f"median: {int(np.median(pool_sizes))}  max: {max(pool_sizes)}")

        self.system_y = None   # lazy: populated in get_gt() / set_dictionaries()

        # ── index real pagecrop data (all N, train only) ──────────────────────
        # Items: ("real", real_idx) — added to epoch pool at final stage
        self._real_paths: list     = []
        self._real_gt_tokens: list = []
        if real_data_path is not None and split == "train":
            self._load_real_index(Path(real_data_path), fold)

    # ── real-data index ───────────────────────────────────────────────────────

    def _load_real_index(self, real_data_path: Path, fold: int):
        """Load ALL pagecrop train entries (every N) for final-stage mixing."""
        split_file = real_data_path / "splits" / f"train_{fold}.txt"
        if not split_file.exists():
            print(f"  Warning: real split not found at {split_file} — "
                  "final-stage real mixing disabled")
            return
        base_dir = real_data_path.parent
        with open(split_file) as f:
            lines = [l.strip().split() for l in f if l.strip()]
        loaded = 0
        for parts in lines:
            if len(parts) < 3:
                continue
            raw_img, raw_gt = parts[0], parts[1]
            img_path = Path(raw_img) if Path(raw_img).exists() else base_dir / raw_img
            gt_path  = Path(raw_gt)  if Path(raw_gt).exists()  else base_dir / raw_gt
            if not img_path.exists() or not gt_path.exists():
                continue

            kern_lines = load_kern(str(gt_path))
            blocks, cur = [], []
            for line in kern_lines:
                if line.strip() == "!!linebreak:original":
                    blocks.append(cur); cur = []
                else:
                    cur.append(line)
            if cur:
                blocks.append(cur)
            tokens = ["<bos>"]
            for i, blk in enumerate(blocks):
                if i > 0:
                    tokens.append("<linebreak>")
                tokens += process_text(lines=blk, tokenizer_type=TOKENIZER_TYPE)
            tokens.append("<eos>")

            self._real_paths.append(str(img_path))
            self._real_gt_tokens.append(tokens)
            loaded += 1
        print(f"  Final-stage real pool: {loaded} pagecrop entries (all N values)")

    # ── stage API ─────────────────────────────────────────────────────────────

    def set_stage_direct(self, stage: int):
        """Write stage to shared memory — visible to all persistent workers."""
        self._shared_stage[0] = stage

    def get_stage(self) -> int:
        return max(1, int(self._shared_stage[0]))

    # ── eligible list (cached per stage) ─────────────────────────────────────

    def _stage_eligible(self) -> list:
        """Build (and cache) the epoch item list for the current stage.

        Each item is either:
            ("stack", n)   — generate a stacked page with n systems
            ("real",  idx) — return real_paths[idx] (final stage only)

        List is rebuilt only when the stage changes; otherwise the cached
        version is reused (safe for persistent workers since each worker
        maintains its own Python-level cache).
        """
        stage = self.get_stage()
        if stage == self._cached_stage:
            return self._cached_eligible

        n = min(stage, self.num_cl_stages)
        items = []

        # Current-stage stacked samples
        items.extend([("stack", n)] * self._dataset_length)

        # Explicit replay: replay_ratio × dataset_length per prior stage
        if self.split == "train" and self.replay_ratio > 0.0 and stage > 1:
            for k in range(1, stage):
                k_n = min(k, self.num_cl_stages)
                count = max(1, round(self.replay_ratio * self._dataset_length))
                items.extend([("stack", k_n)] * count)
            replay_total = sum(
                max(1, round(self.replay_ratio * self._dataset_length))
                for _ in range(1, stage)
            )
            print(f"  [Dataset/{self.split}] stage={stage}  "
                  f"current={self._dataset_length}  replay={replay_total}  ", end="")

        # Final stage: append ALL real pagecrop entries
        if self.split == "train" and stage >= self.final_stage and self._real_paths:
            for idx in range(len(self._real_paths)):
                items.append(("real", idx))
            print(f"real={len(self._real_paths)}  ", end="")

        random.shuffle(items)

        if self.split == "train":
            print(f"total={len(items)}")

        self._cached_stage    = stage
        self._cached_eligible = items
        return items

    # ── vocabulary API ────────────────────────────────────────────────────────

    def _tokenize_systems(self) -> list:
        return [
            ["<bos>"] + process_text(lines=lines, tokenizer_type=TOKENIZER_TYPE) + ["<eos>"]
            for lines in self._raw_gt
        ]

    def get_gt(self) -> list:
        """Return token lists for vocabulary building.

        Includes all system GTs, all real pagecrop GTs, and a sentinel
        to guarantee <linebreak> appears in the vocab.
        """
        if self.system_y is None:
            self.system_y = self._tokenize_systems()
        result = list(self.system_y)
        result.append(["<bos>", "<linebreak>", "<eos>"])   # ensure <linebreak> in vocab
        result.extend(self._real_gt_tokens)
        return result

    def set_dictionaries(self, w2i: dict, i2w: dict):
        self.w2i = w2i
        self.i2w = i2w
        self.padding_token = w2i["<pad>"]
        if self.system_y is None:
            self.system_y = self._tokenize_systems()

    def get_dictionaries(self):
        return self.w2i, self.i2w

    def vocab_size(self) -> int:
        return len(self.w2i)

    # ── model sizing ──────────────────────────────────────────────────────────

    def get_max_hw(self):
        max_h = self.num_cl_stages * self.system_height
        max_w = max(img.shape[1] for img in self.system_x)
        if self._real_paths:
            from PIL import Image
            for rp in self._real_paths[:50]:
                try:
                    with Image.open(rp) as im:
                        w, h = im.size
                    max_w = max(max_w, int(round(w * max_h / h)))
                except Exception:
                    pass
        return max_h, max_w

    def get_max_seqlen(self) -> int:
        assert self.system_y is not None, "Call get_gt() first"
        max_sys = max(len(gt) for gt in self.system_y)
        est = int(max_sys * self.num_cl_stages * 1.1)
        if self._real_gt_tokens:
            est = max(est, int(max(len(t) for t in self._real_gt_tokens) * 1.1))
        return est

    # ── dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        if self.split != "train":
            return self._dataset_length
        return len(self._stage_eligible())

    def __getitem__(self, index):
        assert self.w2i is not None, "Call set_dictionaries() before iterating."

        if self.split != "train":
            # Val: always stack exactly stage systems, no replay, no real mix
            n = min(self.get_stage(), self.num_cl_stages)
            indices = self._sample_indices(n)
            img, y_tokens, _ = stack_systems(
                self.system_x, self.system_y, n,
                system_height=self.system_height, indices=indices,
            )
            x = convert_img_to_tensor(img)
            y_ids = [self.w2i.get(t, self.padding_token) for t in y_tokens]
            y = torch.from_numpy(np.asarray(y_ids, dtype=np.int64))
            return x, y, y, f"val_stage{self.get_stage()}_n{n}"

        # Train: look up item descriptor from eligible list
        eligible = self._stage_eligible()
        kind, payload = eligible[index % len(eligible)]

        if kind == "real":
            return self._get_real_sample(payload)

        # kind == "stack"
        n = payload
        indices = self._sample_indices(n)
        img, y_tokens, _ = stack_systems(
            self.system_x, self.system_y, n,
            system_height=self.system_height, indices=indices,
        )
        x = augment(img) if self.do_augment else convert_img_to_tensor(img)
        y_ids = [self.w2i.get(t, self.padding_token) for t in y_tokens]
        y = torch.from_numpy(np.asarray(y_ids, dtype=np.int64))
        return x, self._apply_teacher_forcing(y), y, f"stack_stage{self.get_stage()}_n{n}"

    # ── helpers ───────────────────────────────────────────────────────────────

    def _sample_indices(self, n: int) -> list:
        anchor = random.randrange(len(self.system_x))
        pool = self._compatible[anchor]
        if len(pool) < n:
            pool = list(range(len(self.system_x)))
        return [anchor] + random.choices(pool, k=n - 1)

    def _get_real_sample(self, idx: int):
        img = cv2.imread(self._real_paths[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.full((self.system_height, 128), 255, dtype=np.uint8)
        x = augment(img) if self.do_augment else convert_img_to_tensor(img)
        tokens = self._real_gt_tokens[idx]
        y_ids = [self.w2i.get(t, self.padding_token) for t in tokens]
        y = torch.from_numpy(np.asarray(y_ids, dtype=np.int64))
        return x, self._apply_teacher_forcing(y), y, self._real_paths[idx]

    def _apply_teacher_forcing(self, sequence: torch.Tensor) -> torch.Tensor:
        errored = sequence.clone()
        for i in range(1, len(sequence)):
            if (random.random() < self.teacher_forcing_error_rate
                    and sequence[i] != self.padding_token):
                errored[i] = random.randint(0, len(self.w2i) - 1)
        return errored
