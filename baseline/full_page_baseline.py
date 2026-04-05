"""
YOLO-based Staff Segmentation for Jazz Lead Sheets

Simple function to detect and crop staff systems from full-page images.
"""

from typing import List, Tuple
from PIL import Image
import sys
import os

# Add sys_level_imports to path so local jazzmus package is found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sys_level_imports"))

from sys_level_imports.dataset.data_preprocessing import convert_img_to_tensor
from ultralytics import YOLO
import torch
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Union
from sys_level_imports.smt_trainer import SMT_Trainer
from sys_level_imports.dataset.tokenizer import untokenize
from inference import FullPageInference
import cv2
import numpy as np


def detect_skew_angle(image: np.ndarray, max_angle: float = 5.0) -> float:
    """
    Detect skew angle of a music score image using Hough line detection.

    Finds horizontal lines (staff lines) and computes their average angle.

    Args:
        image: Grayscale image as numpy array
        max_angle: Maximum expected rotation in degrees (default ±5°)

    Returns:
        Detected skew angle in degrees (positive = clockwise rotation needed)
    """
    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Hough line detection - look for long horizontal lines (staff lines)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=image.shape[1] // 4,  # At least 1/4 page width
        maxLineGap=10
    )

    if lines is None or len(lines) == 0:
        return 0.0

    # Compute angles of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only consider near-horizontal lines (within max_angle of horizontal)
        if abs(angle) <= max_angle:
            angles.append(angle)

    if not angles:
        return 0.0

    # Return median angle (robust to outliers)
    return np.median(angles)


def deskew_image(image: Image.Image, max_angle: float = 5.0) -> Image.Image:
    """
    Correct rotation/skew in a music score image.

    Args:
        image: PIL Image to deskew
        max_angle: Maximum expected rotation in degrees

    Returns:
        Deskewed PIL Image
    """
    # Convert to numpy for processing
    img_array = np.array(image)

    # Detect skew angle
    angle = detect_skew_angle(img_array, max_angle)

    if abs(angle) < 0.1:  # Skip if nearly straight
        return image

    print(f"  Detected skew: {angle:.2f}° - correcting...")

    # Rotate to correct skew
    # Use PIL for rotation (better quality with anti-aliasing)
    # Negative angle because we want to counter-rotate
    corrected = image.rotate(
        -angle,
        resample=Image.BICUBIC,
        expand=True,  # Expand canvas to fit rotated image
        fillcolor='white'  # Fill new areas with white
    )

    return corrected


def compute_dynamic_boundaries(
    staff_boxes: List[Tuple[float, Tuple[int, int, int, int]]],
    image_height: int,
    top_ratio: float = 0.75,
    bottom_ratio: float = 0.25
) -> List[Tuple[int, int, int, int]]:
    """
    Compute non-overlapping crop boundaries for each staff.

    Gap between systems is split: 70% to system below (for chords),
    30% to system above.

    For first/last systems, use average gap padding instead of extending
    all the way to image edges.

    Args:
        staff_boxes: List of (y_center, (x1, y1, x2, y2)) sorted top-to-bottom
        image_height: Total image height
        top_ratio: Fraction of gap given to system below (default 0.7)
        bottom_ratio: Fraction of gap given to system above (default 0.3)

    Returns:
        List of (x1, crop_top, x2, crop_bottom) for each staff
    """
    n = len(staff_boxes)
    if n == 0:
        return []

    boundaries = []

    # Compute average gap between systems for first/last padding
    gaps = []
    for i in range(1, n):
        _, (_, y1, _, _) = staff_boxes[i]
        _, (_, _, _, prev_y2) = staff_boxes[i - 1]
        gaps.append(y1 - prev_y2)

    # Average gap, or use staff height as fallback for single system
    if gaps:
        avg_gap = sum(gaps) / len(gaps)
    else:
        # Single system: use 20% of staff height as padding
        _, (_, y1, _, y2) = staff_boxes[0]
        avg_gap = (y2 - y1) * 0.4  # 20% top + 20% bottom equivalent

    for i in range(n):
        _, (x1, y1, x2, y2) = staff_boxes[i]

        # Compute top boundary
        if i == 0:
            # First system: use avg_gap * top_ratio as padding (same as middle systems get)
            top_padding = int(avg_gap * top_ratio)
            crop_top = max(0, y1 - top_padding)
        else:
            # Gap between previous system's bottom and this system's top
            _, (_, _, _, prev_y2) = staff_boxes[i - 1]
            gap = y1 - prev_y2
            # This system gets top_ratio (70%) of the gap for its chords
            crop_top = prev_y2 + int(gap * bottom_ratio)

        # Compute bottom boundary
        if i == n - 1:
            # Last system: use avg_gap * bottom_ratio as padding (same as middle systems get)
            bottom_padding = int(avg_gap * bottom_ratio)
            crop_bottom = min(image_height, y2 + bottom_padding)
        else:
            # Gap between this system's bottom and next system's top
            _, (_, next_y1, _, _) = staff_boxes[i + 1]
            gap = next_y1 - y2
            # This system gets bottom_ratio (30%) of the gap
            crop_bottom = y2 + int(gap * bottom_ratio)

        boundaries.append((x1, crop_top, x2, crop_bottom))

    return boundaries


def _iou_1d(a_top, a_bot, b_top, b_bot):
    """1-D IoU of two vertical intervals."""
    inter = max(0, min(a_bot, b_bot) - max(a_top, b_top))
    union = max(a_bot, b_bot) - min(a_top, b_top)
    return inter / union if union > 0 else 0.0


def merge_overlapping_staff_boxes(
    staff_boxes: List[Tuple[float, Tuple[int, int, int, int]]],
    y_overlap_threshold: float = 0.6,
) -> List[Tuple[float, Tuple[int, int, int, int]]]:
    """
    Deduplicate staff detections that overlap vertically (same line detected twice).

    When YOLO detects a partial-width and full-width bbox for the same staff,
    keep the narrower one to be consistent with training data.
    """
    if len(staff_boxes) <= 1:
        return staff_boxes

    keep = []
    used = set()

    for i in range(len(staff_boxes)):
        if i in used:
            continue

        yc_i, (x1_i, y1_i, x2_i, y2_i) = staff_boxes[i]
        width_i = x2_i - x1_i
        best_idx = i
        best_width = width_i

        for j in range(i + 1, len(staff_boxes)):
            if j in used:
                continue
            yc_j, (x1_j, y1_j, x2_j, y2_j) = staff_boxes[j]
            v_iou = _iou_1d(y1_i, y2_i, y1_j, y2_j)
            if v_iou >= y_overlap_threshold:
                width_j = x2_j - x1_j
                if width_j < best_width:
                    used.add(best_idx)
                    best_idx = j
                    best_width = width_j
                else:
                    used.add(j)

        keep.append(staff_boxes[best_idx])
        used.add(best_idx)

    keep.sort(key=lambda x: x[0])
    return keep


def interpolate_missing_systems(
    staff_boxes: List[Tuple[float, Tuple[int, int, int, int]]],
    gap_factor: float = 1.8,
) -> List[Tuple[float, Tuple[int, int, int, int]]]:
    """
    Insert virtual staff boxes where a missing system is detected.

    Uses top-left point distances: compute the median distance between
    consecutive top-left corners (x1, y1). If the distance between two
    consecutive top-left points exceeds gap_factor * median_distance,
    one or more systems are likely missing.

    Per-page statistics so each page gets its own threshold.
    """
    if len(staff_boxes) < 3:
        return staff_boxes

    # Compute distances between consecutive top-left points
    tl_dists = []
    for i in range(len(staff_boxes) - 1):
        _, (x1_cur, y1_cur, _, _) = staff_boxes[i]
        _, (x1_next, y1_next, _, _) = staff_boxes[i + 1]
        dist = ((x1_next - x1_cur) ** 2 + (y1_next - y1_cur) ** 2) ** 0.5
        tl_dists.append(dist)

    median_dist = float(np.median(tl_dists))
    if median_dist <= 0:
        return staff_boxes

    threshold = gap_factor * median_dist

    # Median box dimensions for virtual box sizing
    heights = [y2 - y1 for _, (_, y1, _, y2) in staff_boxes]
    widths = [x2 - x1 for _, (x1, _, x2, _) in staff_boxes]
    median_h = int(np.median(heights))
    median_w = int(np.median(widths))

    new_boxes = [staff_boxes[0]]
    for i in range(len(staff_boxes) - 1):
        _, (x1_cur, y1_cur, x2_cur, _) = staff_boxes[i]
        _, (x1_next, y1_next, x2_next, _) = staff_boxes[i + 1]
        dist = tl_dists[i]

        if dist > threshold:
            n_insert = max(1, int(round(dist / median_dist)) - 1)
            for k in range(1, n_insert + 1):
                frac = k / (n_insert + 1)
                # Interpolate top-left position
                vx1 = int(x1_cur + frac * (x1_next - x1_cur))
                vy1 = int(y1_cur + frac * (y1_next - y1_cur))
                vx2 = vx1 + median_w
                vy2 = vy1 + median_h
                y_center = (vy1 + vy2) / 2
                new_boxes.append((y_center, (vx1, vy1, vx2, vy2)))

        new_boxes.append(staff_boxes[i + 1])

    new_boxes.sort(key=lambda x: x[0])
    n_virtual = len(new_boxes) - len(staff_boxes)
    if n_virtual > 0:
        print(f"  Interpolated {n_virtual} virtual system(s) for missed detections")
    return new_boxes


def segment_staves(
    image_path: str,
    yolo_model_path: str,
    confidence_threshold: float = 0.5,
    top_ratio: float = 0.75,
    bottom_ratio: float = 0.25,
    deskew: bool = False,
    max_skew_angle: float = 5.0
) -> List[Image.Image]:
    """
    Detect staff systems in a full-page image and return cropped regions.

    Uses dynamic boundary computation:
    - No overlapping between crops
    - Gap between systems split 70/30 (favoring top for chord symbols)

    Args:
        image_path: Path to the full-page jazz lead sheet image
        yolo_model_path: Path to YOLO model weights (.pt file)
        confidence_threshold: Minimum confidence for staff detection
        top_ratio: Fraction of gap given to system below for chords (default 0.7)
        bottom_ratio: Fraction of gap given to system above (default 0.3)
        deskew: Whether to correct rotation before processing (default False)
        max_skew_angle: Maximum expected rotation in degrees (default 5.0)

    Returns:
        List of PIL Images, one per detected staff system, sorted top-to-bottom
    """
    # Load YOLO model
    model = YOLO(yolo_model_path)

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Optional: Deskew before YOLO detection
    if deskew:
        image = deskew_image(image, max_skew_angle)

    # Run detection (use image object, not path, in case we deskewed)
    results = model(image, conf=confidence_threshold, verbose=False)
    result = results[0]

    # Extract staff bounding boxes (only "staff" class, exclude others)
    staff_boxes = []
    filtered_counts = {}  # Track filtered classes
    for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
        class_name = result.names[int(cls)].lower()
        if class_name == "staff":
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            y_center = (y1 + y2) / 2
            staff_boxes.append((y_center, (x1, y1, x2, y2)))
        else:
            # Track all filtered classes (empty_staff, title, lyrics, etc.)
            filtered_counts[class_name] = filtered_counts.get(class_name, 0) + 1

    if filtered_counts:
        filtered_str = ", ".join(f"{k}={v}" for k, v in filtered_counts.items())
        print(f"  Filtered out: {filtered_str}")

    # Sort by vertical position (top to bottom)
    staff_boxes.sort(key=lambda x: x[0])

    # Merge duplicate detections (partial + full width on same line)
    n_before = len(staff_boxes)
    staff_boxes = merge_overlapping_staff_boxes(staff_boxes)
    n_merged = n_before - len(staff_boxes)
    if n_merged > 0:
        print(f"  Merged {n_merged} duplicate detection(s)")

    # Interpolate virtual boxes for missed systems (large gaps)
    staff_boxes = interpolate_missing_systems(staff_boxes)

    # Compute dynamic non-overlapping boundaries
    boundaries = compute_dynamic_boundaries(
        staff_boxes, image.height, top_ratio, bottom_ratio
    )

    # Crop each system
    cropped_systems = []
    for x1, crop_top, x2, crop_bottom in boundaries:
        cropped = image.crop((x1, crop_top, x2, crop_bottom))
        cropped_systems.append(cropped)

    print(f"Detected and cropped {len(cropped_systems)} staff systems")
    return cropped_systems

def load_model(checkpoint_path: str, device: str = "cpu") -> SMT_Trainer:
    """
    Load a trained SMT model from checkpoint.

    Args:
        checkpoint_path: Path to the .ckpt file
        device: Device to load model on ('cpu', 'cuda', 'mps')

    Returns:
        Loaded SMT_Trainer model
    """
    print(f"Loading model from: {checkpoint_path}")

    # Load checkpoint
    model = SMT_Trainer.load_from_checkpoint(
        checkpoint_path,
        map_location=device
    )
    model.eval()
    model.to(device)

    print(f"Model loaded successfully on {device}")
    return model


def preprocess_image(image: Image.Image, fixed_img_height: int = 128, max_fix_img_width: int = 1000) -> torch.Tensor:
    """
    Preprocess a staff system image for model input.

    Args:
        image: PIL Image of a staff system
        max_height: Maximum height for the model
        max_width: Maximum width for the model

    Returns:
        Preprocessed tensor (1, 1, H, W)
    """
    # Convert to grayscale
    image = np.array(image.convert('L'))

    # Get original dimensions
    original_height, original_width = image.shape

    # Resize with aspect ratio preservation (matching inference.py logic)
    new_height = fixed_img_height
    new_width = int(np.ceil(original_width * fixed_img_height / original_height))

    # Cap width at max
    if new_width > max_fix_img_width:
        new_width = max_fix_img_width


    image = cv2.resize(image, (new_width, new_height))

    # Convert to tensor using the same pipeline as training
    # This applies: ToPILImage → Grayscale → ToTensor
    img_tensor = convert_img_to_tensor(image)  # Returns (C, H, W) = (1, H, W)
    img_tensor = img_tensor.unsqueeze(0)    # Add batch dimension: (1, 1, H, W)

    # Pad to minimum dimensions using batch_preparation_img2seq logic (lines 105-106)
    # This matches what happens during training when batch_size=1
    pad_height = max(32, new_height)      # At least 32 (from batch_preparation_img2seq)
    pad_width = max(1000, new_width)      # At least 1000 (from batch_preparation_img2seq)

    padded = torch.ones(1, 1, pad_height, pad_width)
    padded[:, :, :new_height, :new_width] = img_tensor


    # # Pad to exact dimensions
    # padded = Image.new('L', (max_width, max_height), color=255)  # white background
    # padded.paste(image, (0, 0))

    # # Convert to tensor and normalize
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     # Invert: black text on white -> white text on black
    #     transforms.Lambda(lambda x: 1 - x),
    # ])

    # tensor = transform(padded)
    # tensor = tensor.unsqueeze(0)  # Add batch dimension: (1, 1, H, W)

    return padded

def recognize_system(
    image: Image.Image,
    model: SMT_Trainer,
    device: str = "cuda"
) -> str:
    """
    Recognize a single staff system and return **kern prediction.

    Args:
        image: PIL Image of a staff system
        model: Loaded SMT_Trainer model
        device: Device for inference

    Returns:
        Predicted **kern string for this system
    """
    # Preprocess
    input_tensor = preprocess_image(image).to(device)

    # Run inference
    with torch.no_grad():
        predicted_sequence, _ = model.model.predict(input=input_tensor[0])

        predicted_tokens = predicted_sequence[0]  # Get the first (and only) sequence

        if isinstance(predicted_tokens[0], int):
            # If tokens are integers, look them up in i2w
            token_strs = [model.model.i2w.get(int(t), "<unk>") for t in predicted_tokens]
        else:
            # If tokens are already strings, use them directly
            token_strs = [str(t) for t in predicted_tokens]     

    # Decode to **kern
    kern_prediction = untokenize(token_strs)

    return kern_prediction

def recognize_systems(
    images: List[Image.Image],
    model: SMT_Trainer,
    device: str = "cpu"
) -> List[str]:
    """
    Recognize multiple staff systems.

    Args:
        images: List of PIL Images (staff systems)
        model: Loaded SMT_Trainer model
        device: Device for inference

    Returns:
        List of **kern predictions, one per system
    """
    predictions = []

    print(f"\nRecognizing {len(images)} systems...")
    for i, image in enumerate(images):
        print(f"  System {i+1}/{len(images)}...", end=" ")
        kern = recognize_system(image, model, device)
        predictions.append(kern)
        print("✓")

    print(f"Recognition complete!")
    return predictions

"""
Kern concatenation logic to merge system-level predictions into full-page **kern.

This module handles merging individual system **kern predictions into a complete
full-page **kern representation, handling headers and linebreaks appropriately.
"""

from typing import List


def concatenate_systems(system_kerns: List[str]) -> str:
    """
    Concatenate system-level **kern predictions into full-page **kern.

    Args:
        system_kerns: List of **kern strings, one per system (top to bottom)

    Returns:
        Full-page **kern string with linebreak markers

    Strategy:
        1. Use headers from the first system
        2. For subsequent systems, strip headers and add linebreak markers
        3. Remove all "b:none" lines
    """
    if not system_kerns:
        return ""

    if len(system_kerns) == 1:
        # Remove b:none lines from single system
        lines = [l for l in system_kerns[0].strip().split('\n') if 'b:none' not in l]
        return '\n'.join(lines)

    # Parse first system (keep all headers, remove b:none)
    lines = system_kerns[0].strip().split('\n')
    full_kern_lines = []

    # Add all lines from first system (except b:none)
    for line in lines:
        if 'b:none' not in line:
            full_kern_lines.append(line)

    # Add linebreak marker after first system
    full_kern_lines.append("!!linebreak:original")

    # Process remaining systems
    for system_kern in system_kerns[1:]:
        lines = system_kern.strip().split('\n')

        # Skip header lines (lines starting with *, !, or **)
        content_started = False
        for line in lines:
            # Skip b:none lines
            if 'b:none' in line:
                continue

            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                continue

            # Skip headers (keep only musical content)
            if stripped.startswith('**') or stripped.startswith('*-'):
                # Skip spine definitions and terminations
                continue
            elif stripped.startswith('*') and not content_started:
                # Skip initial metadata lines
                continue
            elif stripped.startswith('!') and not content_started:
                # Skip initial comments
                continue
            else:
                # This is content
                content_started = True
                full_kern_lines.append(line)

        # Add linebreak marker after this system
        full_kern_lines.append("!!linebreak:original")

    # Remove the last linebreak marker and add proper ending
    if full_kern_lines and full_kern_lines[-1] == "!!linebreak:original":
        full_kern_lines.pop()

    # Add spine terminators at the end
    full_kern_lines.append("*-\t*-")

    return '\n'.join(full_kern_lines)


if __name__ == "__main__":
    from pathlib import Path
    from tqdm import tqdm
    from sys_level_imports.dataset.eval_functions import compute_poliphony_metrics
    from inference import extract_spines
    from chord_metrics import (
        extract_chords_from_mxhm,
        extract_tokens_from_mxhm,
        compute_page_chord_metrics,
        aggregate_page_chord_metrics,
        print_page_chord_metrics,
    )

    checkpint_path = "/home/hice1/jwang3180/scratch/jazzmus/ISMIR-Jazzmus/weights/smt_sys_best/smt_pre_syn_medium.ckpt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_model_path = "/home/hice1/jwang3180/scratch/jazzmus/ISMIR-Jazzmus/yolo_weigths/yolov11s_20241108.pt"
    test_split_file = "/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmuss_Data/jazzmus_pagecrop/splits/test_0.txt"
    data_base_dir = "/home/hice1/jwang3180/scratch/Fullpage Jazzmus/Jazzmuss_Data"

    # Load test split
    with open(test_split_file, 'r') as f:
        test_pairs = [
            (os.path.join(data_base_dir, parts[0]), os.path.join(data_base_dir, parts[1]))
            for line in f.readlines()
            if (parts := line.strip().split()) and len(parts) >= 2
        ]

    print(f"Loaded {len(test_pairs)} test samples")

    # Load model once
    print("Loading model...")
    inference_model = FullPageInference(checkpint_path, device=device)
    print("✓ Model loaded\n")

    # Helper: extract mxhm tokens from a kern chunk (second tab column)
    def _extract_mxhm_tokens(kern_chunk):
        tokens = []
        for line in kern_chunk.strip().split('\n'):
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            # Always take the last column — mxhm is the rightmost spine
            # (polyphonic passages use *^ to split kern into multiple spines,
            #  pushing mxhm from column 2 to column 3+)
            tok = parts[-1].strip()
            if not tok or tok.startswith('**') or tok.startswith('*') or tok.startswith('=') or tok.startswith('!'):
                continue
            tokens.append(tok)
        return tokens

    # Collect all predictions and ground truths
    all_predictions = []
    all_ground_truths = []
    per_sample_metrics = []

    for img_path, gt_path in tqdm(test_pairs, desc="Processing test set"):
        try:
            # Step 1: Segment staves with YOLO
            cropped_systems = segment_staves(
                image_path=img_path,
                yolo_model_path=yolo_model_path,
                confidence_threshold=0.5,
                deskew=True,
                max_skew_angle=10.0
            )

            # Step 2: Recognize each system
            system_kerns = []
            for system_image in cropped_systems:
                # Convert PIL Image to numpy array (grayscale) for inference_model.predict()
                system_array = np.array(system_image.convert('L'))
                result = inference_model.predict(system_array)
                system_kerns.append(result['prediction'])

            # Step 3: Concatenate into full-page **kern
            full_page_kern = concatenate_systems(system_kerns)

            # Step 4: Load ground truth
            with open(gt_path, 'r') as f:
                ground_truth = f.read()

            all_predictions.append(full_page_kern)
            all_ground_truths.append(ground_truth)

            # Compute per-sample metrics
            sample_cer, sample_ser, sample_ler = compute_poliphony_metrics([full_page_kern], [ground_truth])
            per_sample_metrics.append({
                'cer': sample_cer,
                'ser': sample_ser,
                'ler': sample_ler,
                'image': img_path,
                'prediction': full_page_kern,
                'ground_truth': ground_truth,
            })

            # Compute edit distance chord metrics (both per-system and per-page)
            try:
                # Per-system: split by linebreak, extract mxhm per system
                pred_systems = full_page_kern.split('!!linebreak:original')
                gt_systems = ground_truth.split('!!linebreak:original')
                n_systems = min(len(pred_systems), len(gt_systems))

                system_chord_metrics = []
                for sys_idx in range(n_systems):
                    pred_tokens = _extract_mxhm_tokens(pred_systems[sys_idx])
                    gt_tokens = _extract_mxhm_tokens(gt_systems[sys_idx])
                    if pred_tokens and gt_tokens:
                        system_chord_metrics.append(
                            compute_page_chord_metrics(pred_tokens, gt_tokens)
                        )

                if system_chord_metrics:
                    page_agg = aggregate_page_chord_metrics(system_chord_metrics)
                    per_sample_metrics[-1]['per_system_chord_metrics'] = page_agg

                # Per-page: extract mxhm from entire page
                pred_spines = extract_spines(full_page_kern)
                gt_spines = extract_spines(ground_truth)
                if '**mxhm' in pred_spines and '**mxhm' in gt_spines:
                    pred_tokens = extract_tokens_from_mxhm(pred_spines['**mxhm'])
                    gt_tokens = extract_tokens_from_mxhm(gt_spines['**mxhm'])
                    if pred_tokens and gt_tokens:
                        per_sample_metrics[-1]['per_page_chord_metrics'] = \
                            compute_page_chord_metrics(pred_tokens, gt_tokens)
            except Exception:
                pass

        except Exception as e:
            print(f"\n✗ Failed on {img_path}: {e}")
            continue

    # ── Aggregate metrics (matches curriculum training computation) ───────────
    # CER/SER/LER: all predictions passed at once (aggregate, not per-sample mean)
    cer_agg, ser_agg, ler_agg = compute_poliphony_metrics(all_predictions, all_ground_truths)

    # Root SER / Chord SER: pool all per-page edit distances together (same as
    # aggregate_page_chord_metrics in trainer._log_chord_metrics)
    page_level_results = [m['per_page_chord_metrics'] for m in per_sample_metrics if 'per_page_chord_metrics' in m]
    if page_level_results:
        agg_chord = aggregate_page_chord_metrics(page_level_results)
        root_ser  = agg_chord['agg_root_ser']
        chord_ser = agg_chord['agg_ser_no_dots']
    else:
        root_ser  = float('nan')
        chord_ser = float('nan')

    # ── Per-sample average metrics ────────────────────────────────────────────
    cer_mean = np.mean([m['cer'] for m in per_sample_metrics])
    ser_mean = np.mean([m['ser'] for m in per_sample_metrics])
    ler_mean = np.mean([m['ler'] for m in per_sample_metrics])
    cer_std  = np.std([m['cer'] for m in per_sample_metrics])
    ser_std  = np.std([m['ser'] for m in per_sample_metrics])
    ler_std  = np.std([m['ler'] for m in per_sample_metrics])

    # ── Per-system chord metrics ──────────────────────────────────────────────
    all_system_chord_metrics = []
    for m in per_sample_metrics:
        if 'per_system_chord_metrics' in m:
            all_system_chord_metrics.extend(m['per_system_chord_metrics']['_raw_metrics'])
    if all_system_chord_metrics:
        agg_per_system = aggregate_page_chord_metrics(all_system_chord_metrics)
        print_page_chord_metrics(agg_per_system, unit_label="system")

    # ── Per-page chord metrics ────────────────────────────────────────────────
    if page_level_results:
        print_page_chord_metrics(agg_chord, unit_label="page")

    # ── 5 worst / 5 best pages ────────────────────────────────────────────────
    pages_with_chord = [m for m in per_sample_metrics if 'per_system_chord_metrics' in m]
    if pages_with_chord:
        pages_with_chord.sort(
            key=lambda x: x['per_system_chord_metrics']['agg_ser_no_dots'],
            reverse=True
        )
        for label, samples in [("5 WORST", pages_with_chord[:5]),
                               ("5 BEST",  pages_with_chord[-5:][::-1])]:
            print(f"\n{'='*60}")
            print(f"{label} CHORD RECOGNITION PAGES (Root / Chord / Structure Accuracy)")
            print(f"{'='*60}")
            for i, m in enumerate(samples):
                cm = m['per_system_chord_metrics']
                img_name = Path(m['image']).stem
                tgc = cm['total_gt_chords']
                tgt = cm['total_gt_tokens']
                tgr = cm['total_gt_roots']
                root_acc  = max(0.0, 100.0 - cm['agg_root_ser'])
                chord_acc = max(0.0, 100.0 - cm['agg_ser_no_dots'])
                struct_acc = max(0.0, 100.0 - cm['agg_ser_with_dots'])
                print(f"\n  {i+1}. {img_name}  ({cm['n_units']} systems, {tgc} GT chords)")
                print(f"     Root Acc:      {root_acc:.1f}% = 1 - {cm['total_root_errors']}/{tgr}  (S={cm['total_subs_roots']} I={cm['total_ins_roots']} D={cm['total_del_roots']})")
                print(f"     Chord Acc:     {chord_acc:.1f}% = 1 - {cm['total_ed_no_dots']}/{tgc}  (S={cm['total_subs_no_dots']} I={cm['total_ins_no_dots']} D={cm['total_del_no_dots']})")
                print(f"     Structure Acc: {struct_acc:.1f}% = 1 - {cm['total_ed_with_dots']}/{tgt}  (S={cm['total_subs_with_dots']} I={cm['total_ins_with_dots']} D={cm['total_del_with_dots']})")
                pred_sys_chunks = m['prediction'].split('!!linebreak:original')
                gt_sys_chunks   = m['ground_truth'].split('!!linebreak:original')
                n_sys = min(len(pred_sys_chunks), len(gt_sys_chunks))
                for s in range(n_sys):
                    pred_toks = _extract_mxhm_tokens(pred_sys_chunks[s])
                    gt_toks   = _extract_mxhm_tokens(gt_sys_chunks[s])
                    if pred_toks or gt_toks:
                        print(f"     System {s+1}:")
                        print(f"       Pred: {pred_toks}")
                        print(f"       GT:   {gt_toks}")
            print(f"{'='*60}\n")

    # ── Final 5-metric summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL SUMMARY  (aggregate — matches curriculum training metrics)")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(all_predictions)}/{len(test_pairs)}")
    print(f"\n  CER:       {cer_agg:.2f}%   (avg ± std: {cer_mean:.2f}% ± {cer_std:.2f}%)")
    print(f"  SER:       {ser_agg:.2f}%   (avg ± std: {ser_mean:.2f}% ± {ser_std:.2f}%)")
    print(f"  LER:       {ler_agg:.2f}%   (avg ± std: {ler_mean:.2f}% ± {ler_std:.2f}%)")
    print(f"  Root SER:  {root_ser:.2f}%")
    print(f"  Chord SER: {chord_ser:.2f}%")
    print(f"{'='*60}\n")

