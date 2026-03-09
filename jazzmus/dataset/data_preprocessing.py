import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from jazzmus.dataset.transforms_custom import *


# import joblib
# MEMORY = joblib.memory.Memory("./joblib_cache", mmap_mode="r", verbose=0)


NUM_CHANNELS = 1
IMG_HEIGHT = 128


RANDOM_THRESHOLD = 0.5
FILTER_APPLY_PROBABILITY = 0.4


def apply_random_filter(x: np.ndarray) -> np.ndarray:
    # Randomly apply different transformations

    if random.random() > RANDOM_THRESHOLD:
        # Rotate
        x = random_rotation(x)

    if random.random() > RANDOM_THRESHOLD:
        # Contrast change
        x = random_contrast(x)

    if random.random() > RANDOM_THRESHOLD:
        # Erosion
        x = random_erosion(x)

    if random.random() > RANDOM_THRESHOLD:
        # Brightness adjustment
        x = random_brightness(x)

    return x


def random_rotation(image: np.ndarray) -> np.ndarray:
    angle = random.uniform(-3, 3)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR
    )
    return rotated_image


def random_contrast(image: np.ndarray) -> np.ndarray:
    alpha = random.uniform(0.5, 1.6)
    beta = random.uniform(-20, 20)
    contrast_adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return contrast_adjusted


def random_erosion(image: np.ndarray) -> np.ndarray:
    kernel_size = random.randint(3, 7)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    iterations = random.randint(1, 3)
    eroded_image = cv2.erode(image, kernel, iterations=iterations)
    return eroded_image


def random_brightness(image: np.ndarray) -> np.ndarray:
    brightness_factor = random.uniform(0.4, 1.5)
    brightness_adjusted = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    return brightness_adjusted


def shrink_image(image):
    return image


def get_image_from_file(path: str, split: str) -> np.ndarray:
    if os.path.exists(path):
        # Open image using Pillow and convert to grayscale
        image = Image.open(path).convert("L")

        # Resize while maintaining aspect ratio
        width = int((IMG_HEIGHT * image.width) / image.height)
        image = image.resize((width, IMG_HEIGHT))

        # Convert to NumPy array
        x = np.array(image, dtype=np.float32)

        # Apply random filter
        if random.random() > FILTER_APPLY_PROBABILITY:
            x = apply_random_filter(x)

        # Normalize to range [0,1]
        x = np.array(x, dtype=np.float32)
        x /= 255.0
    else:
        with open("missing_files.txt", "a") as f:
            f.write(f"{path}\n")
        x = np.zeros((IMG_HEIGHT, 1), dtype=np.float32)

    return x


# @MEMORY.cache
def preprocess_image(path: str, split: str) -> torch.Tensor:
    x = get_image_from_file(path=path, split=split)
    x = np.expand_dims(x, axis=0)
    x = torch.from_numpy(x)
    return x


def pad_batch_images(x):
    max_width = max(x, key=lambda sample: sample.shape[2]).shape[2]
    x = torch.stack([F.pad(i, pad=(0, max_width - i.shape[2])) for i in x], dim=0)
    return x


def pad_batch_transcripts(x, dtype=torch.int32):
    max_length = max(x, key=lambda sample: sample.shape[0]).shape[0]
    x = torch.stack([F.pad(i, pad=(0, max_length - i.shape[0])) for i in x], dim=0)
    x = x.type(dtype=dtype)
    return x


def ctc_batch_preparation(batch):
    x, xl, y, yl = zip(*batch)
    # Zero-pad images to maximum batch image width
    x = pad_batch_images(x)
    xl = torch.tensor(xl, dtype=torch.int32)
    # Zero-pad transcripts to maximum batch transcript length
    y = pad_batch_transcripts(y)
    yl = torch.tensor(yl, dtype=torch.int32)
    return x, xl, y, yl


def augment(image):
    distortion_perspective = np.random.uniform(0, 0.3)

    elastic_dist_magnitude = np.random.randint(1, 20 + 1)
    elastic_dist_kernel = np.random.randint(1, 3 + 1)
    magnitude_w, magnitude_h = (
        (elastic_dist_magnitude, 1)
        if np.random.randint(2) == 0
        else (1, elastic_dist_magnitude)
    )
    kernel_h = np.random.randint(1, 3 + 1)
    kernel_w = np.random.randint(1, 3 + 1)

    br_factor = np.random.uniform(0.7, 1.2)
    ctr_factor = np.random.uniform(0.5, 1.5)

    dilation_erosion = None
    if np.random.randint(2) == 0:
        dilation_erosion = Erosion((kernel_w, kernel_h), 1)
    else:
        dilation_erosion = Dilation((kernel_w, kernel_h), 1)

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomPerspective(
                distortion_scale=distortion_perspective,
                p=1,
                interpolation=Image.BILINEAR,
                fill=255,
            ),
            transforms.RandomApply(
                [
                    ElasticDistortion(
                        grid=(elastic_dist_kernel, elastic_dist_kernel),
                        magnitude=(magnitude_w, magnitude_h),
                        min_sep=(1, 1),
                    )
                ],
                p=0.2,
            ),
            transforms.RandomApply([RandomTransform(16)], p=0.2),
            transforms.RandomApply([dilation_erosion], p=0.2),
            transforms.RandomApply([BrighnessAjust(br_factor)], p=0.2),
            transforms.RandomApply([ContrastAdjust(ctr_factor)], p=0.2),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]
    )

    image = transform(image)

    return image


def convert_img_to_tensor(image, force_one_channel=False):
    transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Grayscale(), transforms.ToTensor()]
    )

    image = transform(image)

    return image


def convert_tensor_to_img(tensor):
    """
    Converts a PyTorch tensor back to a PIL image.
    Assumes the tensor is in (C, H, W) format.
    """
    transform = transforms.ToPILImage()
    image = transform(tensor)
    return image
