"""
FullPageDataset — non-curriculum dataset for whole-page fine-tuning.

Reads 2-column splits (`<img_path> <gt_path>`, no N column) from
jazzmus_fullpage/ and optionally merges jazzmus_fullpage_syn/ for train.

Each sample's effective N is inferred from the GT file by counting
`!!linebreak:original` markers (N = count + 1).  That N is used only
to determine the target image height under the same `system_height`
scaling rule as the other datasets.

No curriculum, no stage, no replay.
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
class FullPageDataset(Dataset):
    """
    Whole-page dataset for fine-tuning.

    Args:
        data_path:            path to jazzmus_fullpage/ (contains splits/)
        split:                "train" | "val" | "test"
        fold:                 fold index
        syn_data_path:        optional path to jazzmus_fullpage_syn/; merged
                              into the train split only.
        system_height:        each system row is scaled to this height;
                              target page height = N_systems * system_height.
        augment:              apply random augmentation (train only)
    """

    def __init__(
        self,
        data_path,
        split: str,
        fold: int,
        syn_data_path: str = None,
        system_height: int = 256,
        augment: bool = False,
    ):
        super().__init__()
        self.split = split
        self.fold = fold
        self.system_height = system_height
        self.do_augment = augment

        self.w2i = None
        self.i2w = None
        self.padding_token = 0
        self.teacher_forcing_error_rate = 0.2

        # ── load real split ──────────────────────────────────────────────────
        data_path = Path(data_path)
        real_base = data_path.parent
        real_split = data_path / "splits" / f"{split}_{fold}.txt"
        if not real_split.exists():
            raise FileNotFoundError(f"Split file not found: {real_split}")

        self.samples = []  # list of (img_path_str, gt_path_str, n_systems)
        self._load_split(real_split, real_base)
        n_real = len(self.samples)

        # ── optionally merge synthetic split (train only) ───────────────────
        n_syn = 0
        if syn_data_path is not None and split == "train":
            syn_path = Path(syn_data_path)
            syn_base = syn_path.parent
            syn_split = syn_path / "splits" / f"{split}_{fold}.txt"
            if syn_split.exists():
                before = len(self.samples)
                self._load_split(syn_split, syn_base)
                n_syn = len(self.samples) - before
            else:
                print(f"  Warning: synthetic split not found: {syn_split}")

        self._is_synthetic = [False] * n_real + [True] * n_syn
        print(f"FullPageDataset [{split}_{fold}]: {len(self.samples)} samples "
              f"(real={n_real}, synthetic={n_syn})")

        # ── tokenize GT ──────────────────────────────────────────────────────
        self._gt_cache = {}
        for _, gt_path, _ in self.samples:
            if gt_path in self._gt_cache:
                continue
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

    # ── split loading ────────────────────────────────────────────────────────

    def _load_split(self, split_file: Path, base_dir: Path):
        with open(split_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                img_path = self._resolve(parts[0], base_dir)
                gt_path  = self._resolve(parts[1], base_dir)
                n = self._count_systems(gt_path)
                self.samples.append((str(img_path), str(gt_path), n))

    @staticmethod
    def _count_systems(gt_path: Path) -> int:
        try:
            with open(gt_path) as f:
                text = f.read()
        except Exception:
            return 1
        return text.count("!!linebreak:original") + 1

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

    # ── vocab API ────────────────────────────────────────────────────────────

    def get_gt(self):
        return self.gt_tokens

    def set_dictionaries(self, w2i: dict, i2w: dict):
        self.w2i = w2i
        self.i2w = i2w
        self.padding_token = w2i["<pad>"]

    def get_dictionaries(self):
        return self.w2i, self.i2w

    def vocab_size(self) -> int:
        return len(self.w2i)

    # ── size estimates (for model PE sizing) ─────────────────────────────────

    def _scaled_hw(self, h: int, w: int, n: int):
        target_h = n * self.system_height
        if h != target_h:
            w = max(1, int(round(w * target_h / h)))
            h = target_h
        return h, w

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

    def get_max_seqlen(self) -> int:
        return max(len(tokens) for tokens in self.gt_tokens)

    # ── dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        assert self.w2i is not None, "Call set_dictionaries() before iterating."

        img_path, gt_path, n = self.samples[index]
        tokens = self.gt_tokens[index]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.full((128, 128), 255, dtype=np.uint8)
        h, w = img.shape[:2]
        target_h, target_w = self._scaled_hw(h, w, n)
        if target_h != h:
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        x = augment(img) if self.do_augment else convert_img_to_tensor(img)

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
