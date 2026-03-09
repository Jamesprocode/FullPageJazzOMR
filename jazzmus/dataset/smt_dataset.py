import re

import cv2
import gin
import numpy as np
import torch

from lightning import LightningDataModule
from rich import progress
from torch.utils.data import Dataset
from torchvision import transforms

from jazzmus.dataset.data_preprocessing import augment, convert_img_to_tensor
from jazzmus.dataset.smt_dataset_utils import check_and_retrieveVocabulary, load_kern
from jazzmus.dataset.tokenizer import process_text

from pathlib import Path


def load_set(
    dataset,
    fold,
    split="train",
    reduce_ratio=1.0,
    fixed_size=None,
    fixed_img_height=256,
    max_fix_img_width=None,
    include_synthetic=False,
):
    x = []
    y = []
    paths = []

    # Read from the given split file

    with open(f"{dataset}/{split}_{fold}.txt") as f:
        lines = f.readlines()
        kern_samples = [line.split(" ")[1].strip() for line in lines]
        img_samples = [line.split(" ")[0].strip() for line in lines]

    if include_synthetic and split == "train":
        img_samples_stems = [Path(img).stem for img in img_samples]
        kern_folder = Path("data/jazzmus_systems_syn/gt")
        counter = 0
        # add the pieces from data/la_result to the dataset
        assert Path("data/jazzmus_systems_syn").exists(), "data/jazzmus_systems_syn does not exist. Create synthetic first!"
        # take only the pieces that are in the train split
        for piece in Path("data/jazzmus_systems_syn").rglob("*.jpg"):
            # if piece.stem.split("_syn")[0] in img_samples_stems:
            img_samples.append(str(piece))
            kern_synpath = kern_folder / Path(str(piece.stem) + ".txt")
            assert kern_synpath.exists(), f"Kern file {kern_synpath} does not exist"
            kern_samples.append(kern_synpath)
            counter += 1
        print(f"Added {counter} synthetic pieces to the dataset")

    print(f"Number of regions in split: {split} SMB dataset: {len(img_samples)}")

    assert len(img_samples) == len(kern_samples), (
        "Number of images and kern files do not match"
    )

    for img_sample, kern_sample in progress.track(zip(img_samples, kern_samples)):
        krn_content = load_kern(kern_sample)

        # read image from path
        img_raw = cv2.imread(img_sample, cv2.IMREAD_GRAYSCALE)
        img = np.array(img_raw)
        if fixed_size is not None:
            width = fixed_size[1]
            height = fixed_size[0]
        elif fixed_img_height is not None:
            # keep the aspect ratio
            width = int(np.ceil(img.shape[1] * fixed_img_height / img.shape[0]))
            height = fixed_img_height
            if max_fix_img_width is not None:
                width = min(width, max_fix_img_width)
        elif img.shape[1] > 3056:
            width = int(np.ceil(3056 * reduce_ratio))
            height = int(np.ceil(max(img.shape[0], 256) * reduce_ratio))
        else:
            width = int(np.ceil(img.shape[1] * reduce_ratio))
            height = int(np.ceil(max(img.shape[0], 256) * reduce_ratio))

        img = cv2.resize(img, (width, height))
        y.append(krn_content)  # list of lines
        # y.append([content + "\n" for content in krn_content.split("\n")])
        x.append(img)
        paths.append(img_sample)

    return x, y, paths


def batch_preparation_img2seq(data):
    images = [sample[0] for sample in data]
    dec_in = [sample[1] for sample in data]
    gt = [sample[2] for sample in data]
    paths = [sample[3] for sample in data]


    # print("Original shapes")
    # for i in images:
    #     print(i.shape)

    max_image_width = max(1000, max([img.shape[2] for img in images]))
    max_image_height = max(32, max([img.shape[1] for img in images]))

    X_train = torch.ones(
        size=[len(images), 1, max_image_height, max_image_width], dtype=torch.float32
    )

    for i, img in enumerate(images):
        _, h, w = img.size()
        X_train[i, :, :h, :w] = img

    max_length_seq = max([len(w) for w in gt])

    decoder_input = torch.zeros(size=[len(dec_in), max_length_seq])
    y = torch.zeros(size=[len(gt), max_length_seq])

    for i, seq in enumerate(dec_in):
        decoder_input[i, 0 : len(seq) - 1] = torch.from_numpy(
            np.asarray([char for char in seq[:-1]])
        )

    for i, seq in enumerate(gt):
        y[i, 0 : len(seq) - 1] = torch.from_numpy(
            np.asarray([char for char in seq[1:]])
        )

    return X_train, decoder_input.long(), y.long(), paths


class OMRIMG2SEQDataset(Dataset):
    def __init__(self, augment=False) -> None:
        self.teacher_forcing_error_rate = 0.2
        self.x = None
        self.y = None
        self.path = None
        self.augment = augment

        super().__init__()

    def apply_teacher_forcing(self, sequence):
        errored_sequence = sequence.clone()
        for token in range(1, len(sequence)):
            if (
                np.random.rand() < self.teacher_forcing_error_rate
                and sequence[token] != self.padding_token
            ):
                errored_sequence[token] = np.random.randint(0, len(self.w2i))

        return errored_sequence

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.augment:
            x = augment(self.x[index])
        else:
            x = convert_img_to_tensor(self.x[index])

        y = torch.from_numpy(np.asarray([self.w2i[token] for token in self.y[index]]))
        decoder_input = self.apply_teacher_forcing(y)
        return x, decoder_input, y, self.path[index]

    def get_max_hw(self):
        m_width = np.max([img.shape[1] for img in self.x])
        m_height = np.max([img.shape[0] for img in self.x])

        return m_height, m_width

    def get_max_seqlen(self):
        return np.max([len(seq) for seq in self.y])

    def vocab_size(self):
        return len(self.w2i)

    def get_gt(self):
        return self.y

    def set_dictionaries(self, w2i, i2w):
        self.w2i = w2i
        self.i2w = i2w
        self.padding_token = w2i["<pad>"]

    def get_dictionaries(self):
        return self.w2i, self.i2w

    def get_i2w(self):
        return self.i2w


@gin.configurable
class GrandStaffSingleSystem(OMRIMG2SEQDataset):
    def __init__(
        self,
        data_path,
        split,
        fold,
        augment=False,
        tokenizer_type="word",
        fixed_img_height=256,
        max_fix_img_width=None,
        include_synthetic=False,
    ) -> None:
        self.augment = augment
        self.teacher_forcing_error_rate = 0.2
        self.fold = fold

        # tokenization
        self.tokenizer_type = tokenizer_type

        # image parameters
        self.fixed_img_height = fixed_img_height

        self.x, self.y, self.path_to_images = load_set(
            data_path,
            split=split,
            fold=self.fold,
            fixed_img_height=self.fixed_img_height,
            max_fix_img_width = max_fix_img_width,
            include_synthetic=include_synthetic,
        )
        self.y = self.preprocess_gt(self.y)
        self.tensorTransform = transforms.ToTensor()
        self.num_sys_gen = 1
        self.fixed_systems_num = False

    def erase_numbers_in_tokens_with_equal(self, tokens):
        return [re.sub(r"(?<=\=)\d+", "", token) for token in tokens]

    def get_width_avgs(self):
        widths = [image.shape[1] for image in self.x]
        return np.average(widths), np.max(widths), np.min(widths)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        path = self.path_to_images[index]

        if self.augment:
            x = augment(x)
        else:
            x = convert_img_to_tensor(x)

        y = torch.from_numpy(np.asarray([self.w2i[token] for token in y]))
        decoder_input = self.apply_teacher_forcing(y)
        return x, decoder_input, y, path

    def __len__(self):
        return len(self.x)

    def preprocess_gt(self, Y):
        for idx, krn in enumerate(Y):

            Y[idx] = (
                ["<bos>"] + process_text(lines=krn, tokenizer_type=self.tokenizer_type) + ["<eos>"]
            )
        return Y


@gin.configurable
class GrandStaffDataset(LightningDataModule):
    def __init__(
        self, data_path="", vocab_name="", batch_size=1, num_workers=4, fold=0
    ) -> None:
        super().__init__()
        print("Initializing datamodule with parameters:")
        print(f"\tData path: {data_path}")
        print(f"\tVocabulary name: {vocab_name}")
        print(f"\tBatch size: {batch_size}")
        print(f"\tNumber of workers: {num_workers}")
        print(f"\tFold: {fold}")
        self.data_path = data_path
        self.vocab_name = vocab_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fold = fold
        self.train_set = GrandStaffSingleSystem(
            data_path=self.data_path,
            split="train",
            augment=True,
            fold=self.fold,
        )
        self.val_set = GrandStaffSingleSystem(
            data_path=self.data_path, split="val", fold=self.fold
        )
        self.test_set = GrandStaffSingleSystem(
            data_path=self.data_path, split="test", fold=self.fold
        )

        w2i, i2w = check_and_retrieveVocabulary(
            [self.train_set.get_gt(), self.val_set.get_gt(), self.test_set.get_gt()],
            "vocab",
            f"{self.vocab_name}",
        )

        self.train_set.set_dictionaries(w2i, i2w)
        self.val_set.set_dictionaries(w2i, i2w)
        self.test_set.set_dictionaries(w2i, i2w)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=batch_preparation_img2seq,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_set,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=batch_preparation_img2seq,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_set,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=batch_preparation_img2seq,
        )
