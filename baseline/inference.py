"""
Inference pipeline for full-page jazz leadsheet recognition.

Steps:
1. Load trained model checkpoint
2. Process image (resize, normalize)
3. Generate predictions token-by-token
4. Decode to kern format
5. Display/save results
"""

import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys
import os

# Add sys_level_imports to path so local jazzmus package is found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sys_level_imports"))

from sys_level_imports.smt_trainer import SMT_Trainer
from sys_level_imports.dataset.tokenizer import untokenize, process_text
from sys_level_imports.metrics import compute_metrics
from sys_level_imports.dataset.eval_functions import compute_poliphony_metrics
from collections import defaultdict
from sys_level_imports.dataset.data_preprocessing import convert_img_to_tensor

def extract_spines(kern_text):
    """
    Extract individual spines from **kern format.

    Handles both raw and processed (tokenized/untokenized) kern text:
    - Skips special tokens like <bos>, <eos>
    - Finds spine headers (lines starting with **)
    - Extracts content for each spine

    Returns dict with spine name -> content mapping.
    E.g., {'**kern': '...melody...', '**mxhm': '...chords...'}
    """
    lines = kern_text.strip().split('\n')
    spines = {}
    spine_indices = {}

    # Find spine headers (lines starting with **)
    for i, line in enumerate(lines):
        # Remove special tokens from line start if present
        clean_line = line
        if clean_line.startswith('<bos>'):
            clean_line = clean_line[5:].lstrip()  # Remove <bos> and leading whitespace

        if clean_line.startswith('**'):
            parts = clean_line.split('\t')
            for j, part in enumerate(parts):
                if part.startswith('**'):
                    spine_name = part
                    if spine_name not in spines:
                        spines[spine_name] = []
                        spine_indices[spine_name] = j

    # Extract content for each spine
    for line in lines:
        # Remove special tokens from line start if present
        clean_line = line
        if clean_line.startswith('<bos>'):
            clean_line = clean_line[5:].lstrip()
        if clean_line.startswith('<eos>'):
            clean_line = clean_line[5:].lstrip()

        # Skip metadata lines (starting with * = !)
        if clean_line.startswith('*') or clean_line.startswith('=') or clean_line.startswith('!'):
            # In processed text, metadata lines are usually gone, so skip
            continue

        # Data line
        parts = clean_line.split('\t')
        for spine_name, idx in spine_indices.items():
            if idx < len(parts):
                spines[spine_name].append(parts[idx])

    # Join lines back together
    result = {}
    for spine_name, content_list in spines.items():
        result[spine_name] = '\n'.join(content_list)

    return result

def process_ground_truth_from_file(gt_path, model, tokenizer_type="word"):
    """
    Process ground truth from file exactly like training does.

    Training pipeline:
    1. Load raw file
    2. Tokenize with process_text()
    3. Add <bos> and <eos>
    4. Convert to token IDs with w2i
    5. Convert back to strings with i2w
    6. Untokenize to get readable format

    Args:
        gt_path: Path to ground truth kern file
        model: Trained SMTModelForCausalLM with w2i and i2w mappings
        tokenizer_type: "word", "character", or "medium"

    Returns:
        Untokenized readable ground truth string
    """
    # Load raw file
    with open(gt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Tokenize (same as dataset preprocessing)
    tokens = process_text(lines, tokenizer_type=tokenizer_type)

    # Add special tokens (same as dataset preprocessing, line 257 in smt_dataset.py)
    tokens = ["<bos>"] + tokens + ["<eos>"]

    # Convert token strings to token IDs using w2i (same as __getitem__ line 248)
    # Handle unknown tokens by using <unk> if available, or skip them
    token_ids = []
    unk_token_id = model.model.w2i.get("<unk>", None)

    for token in tokens:
        if token in model.model.w2i:
            token_ids.append(model.model.w2i[token])
        elif unk_token_id is not None:
            # Use <unk> token for unknown words
            token_ids.append(unk_token_id)
        else:
            # If no <unk> token, try to handle gracefully by using first available token
            # or raise error with helpful message
            print(f"Warning: Unknown token '{token}' not in vocabulary and no <unk> token available")
            if model.model.w2i:
                # Use first token as fallback
                token_ids.append(next(iter(model.model.w2i.values())))
            else:
                raise ValueError(f"Token '{token}' not in model vocabulary and vocabulary is empty")

    token_ids = torch.tensor(token_ids, dtype=torch.long)

    # Convert back to strings using i2w, excluding the last token (<eos>) like training does (line 194)
    # gt = untokenize([self.model.i2w[token.item()] for token in y_single[:-1]])
    gt_tokens = [model.model.i2w[token.item()] for token in token_ids[:-1]]

    # Untokenize to get readable format
    gt_readable = untokenize(gt_tokens)

    return gt_readable

def filter_chord_spine(chord_spine):
    """
    Filter out dots from chord spine for meaningful evaluation.

    Dots (.) are duration markers and appear very frequently in chord spines,
    inflating accuracy metrics when they're easy to predict. This function
    removes dots to focus evaluation on actual chord symbols (Cmaj7, G7, etc.).

    Args:
        chord_spine: String content from **mxhm spine

    Returns:
        Filtered spine with dots removed, preserving chord symbols
    """
    # Split by newlines, filter out lines that are only dots, and rejoin
    lines = chord_spine.split('\n')
    filtered_lines = []

    for line in lines:
        # Remove dots from each line
        filtered_line = line.replace('.', '')
        # Only keep non-empty lines
        if filtered_line.strip():
            filtered_lines.append(filtered_line)

    return '\n'.join(filtered_lines)


def calculate_spine_metrics(prediction, ground_truth):
    """
    Calculate CER/SER/LER for individual spines and overall.

    Returns dict with metrics for each spine and overall.
    """
    # Extract spines
    pred_spines = extract_spines(prediction)
    gt_spines = extract_spines(ground_truth)

    # Get all spine names
    all_spines = set(pred_spines.keys()) | set(gt_spines.keys())

    results = {}

    # Calculate metrics for each spine
    for spine_name in sorted(all_spines):
        pred_spine = pred_spines.get(spine_name, "")
        gt_spine = gt_spines.get(spine_name, "")

        if not gt_spine:  # Skip if no ground truth for this spine
            print("no gt spine")
            continue

        try:
            cer, ser, ler = compute_poliphony_metrics([pred_spine], [gt_spine])
            results[spine_name] = {
                "cer": cer,
                "ser": ser,
                "ler": ler,
            }
        except Exception as e:
            results[spine_name] = {
                "cer": 100.0,
                "ser": 100.0,
                "ler": 100.0,
                "error": str(e),
            }

    # Calculate overall metrics
    try:
        cer_overall, ser_overall, ler_overall = compute_poliphony_metrics([prediction], [ground_truth])
        results["OVERALL"] = {
            "cer": cer_overall,
            "ser": ser_overall,
            "ler": ler_overall,
        }
    except Exception as e:
        results["OVERALL"] = {
            "cer": 100.0,
            "ser": 100.0,
            "ler": 100.0,
            "error": str(e),
        }

    return results


class FullPageInference:
    """Inference pipeline for full-page jazz leadsheet recognition."""

    def __init__(self, checkpoint_path, device):
        """
        Initialize inference pipeline.

        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run on (cuda or cpu)
        """
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path

        # Load model
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Create model with same config as checkpoint
        # Note: You'll need to load the config from somewhere
        # For now, we'll load it manually
        self.model = SMT_Trainer.load_from_checkpoint(
            checkpoint_path,
            map_location=self.device,
        )
        self.model.eval()
        self.model.to(self.device)

        print("✓ Model loaded successfully")

    def preprocess_image(self, image_path, fixed_img_height=128, max_fix_img_width=1000):
        """
        Preprocess image for inference.

        Matches the dataset preprocessing:
        - Fixed height: 128 (preserve aspect ratio)
        - Variable width: calculated to preserve aspect, capped at 1000

        Args:
            image_path: Path to input image or numpy array
            fixed_img_height: Fixed height for resizing (default 128, matching config)
            max_fix_img_width: Maximum width after resizing (default 1000, matching config)

        Returns:
            torch.Tensor: Preprocessed image with shape (1, 1, H, W)
        """
        # Load image
        if isinstance(image_path, str):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = np.array(image_path)

        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Get original dimensions
        original_height, original_width = img.shape

        # Resize with aspect ratio preservation (matching training logic in smt_dataset.py:72-77)
        new_height = fixed_img_height
        new_width = int(np.ceil(original_width * fixed_img_height / original_height))

        # Cap width at max
        if new_width > max_fix_img_width:
            new_width = max_fix_img_width

        # Resize image
        img = cv2.resize(img, (new_width, new_height))

        # Convert to tensor using the same pipeline as training
        img_tensor = convert_img_to_tensor(img)  # Returns (C, H, W) = (1, H, W)
        img_tensor = img_tensor.unsqueeze(0)      # Add batch dimension: (1, 1, H, W)

        return img_tensor.to(self.device)
    
    def predict(self, image_path, return_probs=False):
        """
        Predict on full-page image.

        Args:
            image_path: Path to input image
            return_probs: Return token probabilities

        Returns:
            dict: Prediction results
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)

        # Generate predictions
        # print("Generating predictions...")
        with torch.no_grad():
            predicted_tokens, logits = self.model.model.predict(input=image_tensor[0])

        # Decode tokens to string
        # predicted_tokens are already strings, no need to convert to int
        if isinstance(predicted_tokens[0], int):
            # If tokens are integers, look them up in i2w
            token_strs = [self.model.model.i2w.get(int(t), "<unk>") for t in predicted_tokens]
        else:
            # If tokens are already strings, use them directly
            token_strs = [str(t) for t in predicted_tokens]
        prediction_str = untokenize(token_strs)
        results = {
            "tokens": token_strs,
            "prediction": prediction_str,
            "num_tokens": len(predicted_tokens),
        }

        if return_probs:
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=5, dim=-1)
            results["logits"] = logits.cpu().numpy()
            results["top_probs"] = top_probs.cpu().numpy()
            results["top_indices"] = top_indices.cpu().numpy()

        return results

    def save_result(self, result, output_path):
        """
        Save prediction result to file.

        Args:
            result: Prediction result dictionary
            output_path: Path to save result
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result['prediction'])

        print(f"✓ Result saved to: {output_path}")

    def display_result(self, result):
        """
        Display prediction result.

        Args:
            result: Prediction result dictionary
        """
        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(f"Number of tokens: {result['num_tokens']}")
        print(f"\nPredicted kern:")
        print(result['prediction'])
        print("=" * 60)

    def score_with_ground_truth(self, prediction, ground_truth_path, per_spine=True, tokenizer_type="medium"):
        """
        Calculate metrics by comparing prediction with ground truth.

        Args:
            prediction: Predicted **kern string
            ground_truth_path: Path to ground truth file
            per_spine: Whether to calculate metrics per individual spine

        Returns:
            dict: Metrics (SER, SEQ-ER, and per-spine if requested)
        """
        # Load ground truth
        ground_truth = process_ground_truth_from_file(ground_truth_path, self.model, tokenizer_type)

        print("\n" + "=" * 60)
        print("ground truth")
        print(ground_truth)
        # Display metrics
        print("\n" + "=" * 60)
        print("SCORING RESULTS")
        print("=" * 60)


        # Calculate per-spine metrics
        spine_metrics = calculate_spine_metrics(prediction, ground_truth)

        print("\nPER-SPINE METRICS:")
        print("-" * 60)
        for spine_name in sorted(spine_metrics.keys()):
            metrics = spine_metrics[spine_name]
            print(f"\n{spine_name}:")
            print(f"  CER: {metrics['cer']:.2f}%")
            print(f"  SER: {metrics['ser']:.2f}%")
            print(f"  LER: {metrics['ler']:.2f}%")

    # Calculate overall metrics only
        cer, ser, ler = compute_poliphony_metrics([prediction], [ground_truth])
        spine_metrics = {"OVERALL": {"cer": cer, "ser": ser, "ler": ler}}
        print(f"CER (Character Error Rate): {cer:.2f}%")  
        print(f"SER (System Error Rate):     {ser:.2f}%")
        print(f"LER (Line Error Rate):      {ler:.2f}%")

        print(f"\nGround truth length: {len(ground_truth)}")
        print(f"Prediction length:   {len(prediction)}")
        print("=" * 60 + "\n")
        

        # spine metrics
        spine_predictions = defaultdict(list)  # spine -> list of predictions
        spine_ground_truths = defaultdict(list)
        pred_spines = extract_spines(prediction)
        gt_spines = extract_spines(ground_truth)
        for spine_name in pred_spines.keys():
            spine_predictions[spine_name].append(pred_spines[spine_name])
            spine_ground_truths[spine_name].append(gt_spines[spine_name])

        if spine_predictions:
            print("\n" + "-" * 70)
            print("PER-SPINE METRICS:")
            print("-" * 70)

        for spine_name in sorted(spine_predictions.keys()):
            pred_list = spine_predictions[spine_name]
            gt_list = spine_ground_truths[spine_name]

            # For chord spine, filter out dots to focus on actual chords
            if spine_name == '**mxhm':
                pred_list = [filter_chord_spine(p) for p in pred_list]
                gt_list = [filter_chord_spine(g) for g in gt_list]
                print(f"\n{spine_name} (Dots Filtered):")
            else:
                print(f"\n{spine_name}:")

            print(f"  Predictions: {len(pred_list)} samples")
            print(f"  Ground truth: {len(gt_list)} samples")

            if not pred_list or not gt_list:
                print(f"  SKIPPED: Empty lists")
                continue

            # Check content length
            total_pred_len = sum(len(p) for p in pred_list)
            total_gt_len = sum(len(g) for g in gt_list)
            if spine_name == '**mxhm':
                print(f"  Pred content length (without dots): {total_pred_len} chars")
                print(f"  GT content length (without dots): {total_gt_len} chars")
            else:
                print(f"  Pred content length: {total_pred_len} chars")
                print(f"  GT content length: {total_gt_len} chars")

            if total_gt_len == 0:
                print(f"  SKIPPED: Empty ground truth content")
                continue

            try:
                cer_spine, ser_spine, ler_spine = compute_poliphony_metrics(pred_list, gt_list)
                print(f"  CER: {cer_spine:.2f}%")
                print(f"  SER: {ser_spine:.2f}%")
                print(f"  LER: {ler_spine:.2f}%")
            except ZeroDivisionError as e:
                print(f"  ERROR: Division by zero - {str(e)}")
            except Exception as e:
                print(f"  ERROR: {str(e)}")

            print("\n" + "=" * 70)
        return spine_metrics


def evaluate_test_set(
    checkpoint_path,
    test_dir,
    split="test",
    fold=0,
    output_dir=None,
    device="cuda",
    tokenizer_type="medium",
):
    """
    Evaluate model on entire test set and compute aggregate metrics.

    Args:
        checkpoint_path: Path to trained model checkpoint
        test_dir: Path to dataset directory with splits
        split: Which split to evaluate (train/val/test)
        fold: Fold number
        output_dir: Optional directory to save predictions
        device: Device to run on
    """
    test_dir = Path(test_dir)
    splits_dir = test_dir / "splits"

    # Read split file
    split_file = splits_dir / f"{split}_{fold}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file, 'r') as f:
        lines = f.readlines()
        img_paths = [line.split()[0] for line in lines]
        gt_paths = [line.split()[1] for line in lines]

    print("\n" + "=" * 70)
    print(f"EVALUATING ON {split.upper()} SET")
    print("=" * 70)
    print(f"Checkpoint:     {checkpoint_path}")
    print(f"Test images:    {len(img_paths)}")
    print(f"Dataset:        {test_dir}")
    print("=" * 70 + "\n")

    # Initialize inference
    print("Loading model...")
    inference = FullPageInference(checkpoint_path, device=device)
    print()

    # Collect all predictions and ground truths
    all_predictions = []
    all_ground_truths = []
    individual_metrics = []
    spine_predictions = defaultdict(list)  # spine -> list of predictions
    spine_ground_truths = defaultdict(list)  # spine -> list of ground truths
    failed = []

    print(f"Running inference on {len(img_paths)} images...\n")
    for idx, (img_path, gt_path) in enumerate(tqdm(zip(img_paths, gt_paths), total=len(img_paths), desc="Evaluating")):
        try:
            # Run inference
            result = inference.predict(img_path)
            prediction = result['prediction']

            # Load ground truth
            ground_truth = process_ground_truth_from_file(gt_path, inference.model, tokenizer_type)

            all_predictions.append(prediction)
            all_ground_truths.append(ground_truth)

            # Calculate per-sample metrics (overall)
            cer, ser, ler = compute_poliphony_metrics([prediction], [ground_truth])
            individual_metrics.append({
                "image": Path(img_path).name,
                "cer": cer,
                "ser": ser,
                "ler": ler,
            })

            # Extract and collect spines for per-spine aggregate metrics
            try:
                pred_spines = extract_spines(prediction)
                gt_spines = extract_spines(ground_truth)

                for spine_name in pred_spines.keys():
                    spine_predictions[spine_name].append(pred_spines[spine_name])
                    spine_ground_truths[spine_name].append(gt_spines[spine_name])
            except Exception as e:
                # If spine extraction fails, skip per-spine metrics for this sample
                pass

            # Save prediction if output dir provided
            if output_dir:
                output_path = Path(output_dir) / (Path(img_path).stem + "_pred.txt")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(prediction)

        except Exception as e:
            failed.append((Path(img_path).name, str(e)))
            continue

    # Calculate aggregate metrics
    print("\n\nCalculating aggregate metrics...\n")
    cer_agg, ser_agg, ler_agg = compute_poliphony_metrics(all_predictions, all_ground_truths)

    # Display results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"✓ Successfully evaluated: {len(all_predictions)}/{len(img_paths)}")
    if failed:
        print(f"✗ Failed: {len(failed)}")
        for filename, reason in failed[:5]:
            print(f"   - {filename}: {reason}")
        if len(failed) > 5:
            print(f"   ... and {len(failed) - 5} more")

    print("\n" + "=" * 70)
    print("AGGREGATE METRICS")
    print("=" * 70)

    print("\nOVERALL (All Spines Combined):")
    print("-" * 70)
    print(f"CER (Character Error Rate):  {cer_agg:.2f}%")
    print(f"SER (Sequence Error Rate):   {ser_agg:.2f}%")
    print(f"LER (Line Error Rate):       {ler_agg:.2f}%")

    # Show per-spine aggregate metrics using same calculation method
    if spine_predictions:
        print("\n" + "-" * 70)
        print("PER-SPINE METRICS:")
        print("-" * 70)

        #kern
        pred_list = spine_predictions['**kern']
        gt_list = spine_ground_truths['**kern']

        print(f"\n**kern:")
        print(f"  Predictions: {len(pred_list)} samples")
        print(f"  Ground truth: {len(gt_list)} samples")


        # Check content length
        total_pred_len = sum(len(p) for p in pred_list)
        total_gt_len = sum(len(g) for g in gt_list)
        print(f"  Pred content length: {total_pred_len} chars")
        print(f"  GT content length: {total_gt_len} chars")


        try:
            cer_kern, ser_kern, ler_kern = compute_poliphony_metrics(pred_list, gt_list)
            print(f"  CER: {cer_kern:.2f}%")
            print(f"  SER: {ser_kern:.2f}%")
            print(f"  LER: {ler_kern:.2f}%")
        except ZeroDivisionError as e:
            print(f"  ERROR: Division by zero - {str(e)}")
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            
        #mxhm (Chord spine - filtered to exclude dots which inflate metrics)
        pred_list_raw = spine_predictions['**mxhm']
        gt_list_raw = spine_ground_truths['**mxhm']

        # Filter out dots from chord spines for meaningful evaluation
        pred_list = [filter_chord_spine(p) for p in pred_list_raw]
        gt_list = [filter_chord_spine(g) for g in gt_list_raw]

        print(f"\n**mxhm (Chord Spine - Dots Filtered):")
        print(f"  Predictions: {len(pred_list)} samples")
        print(f"  Ground truth: {len(gt_list)} samples")


        # Check content length
        total_pred_len = sum(len(p) for p in pred_list)
        total_gt_len = sum(len(g) for g in gt_list)
        print(f"  Pred content length (without dots): {total_pred_len} chars")
        print(f"  GT content length (without dots): {total_gt_len} chars")


        try:
            cer_mxhm, ser_mxhm, ler_mxhm = compute_poliphony_metrics(pred_list, gt_list)
            print(f"  CER: {cer_mxhm:.2f}%")
            print(f"  SER: {ser_mxhm:.2f}%")
            print(f"  LER: {ler_mxhm:.2f}%")
        except ZeroDivisionError as e:
            print(f"  ERROR: Division by zero - {str(e)}")
        except Exception as e:
            print(f"  ERROR: {str(e)}")

    print("\n" + "=" * 70)

    # # Show per-sample statistics
    # if individual_metrics:
    #     print("\nPER-SAMPLE STATISTICS (OVERALL)")
    #     print("-" * 70)
    #     cer_values = [m["cer"] for m in individual_metrics]
    #     ser_values = [m["ser"] for m in individual_metrics]
    #     ler_values = [m["ler"] for m in individual_metrics]

    #     print(f"CER - Mean: {np.mean(cer_values):.2f}%, Std: {np.std(cer_values):.2f}%, Min: {np.min(cer_values):.2f}%, Max: {np.max(cer_values):.2f}%")
    #     print(f"SER - Mean: {np.mean(ser_values):.2f}%, Std: {np.std(ser_values):.2f}%, Min: {np.min(ser_values):.2f}%, Max: {np.max(ser_values):.2f}%")
    #     print(f"LER - Mean: {np.mean(ler_values):.2f}%, Std: {np.std(ler_values):.2f}%, Min: {np.min(ler_values):.2f}%, Max: {np.max(ler_values):.2f}%")
    #     print("=" * 70 + "\n")

    # if output_dir:
    #     print(f"✓ Predictions saved to: {output_dir}\n")

    return {
        "cer": cer_agg,
        "ser": ser_agg,
        "ler": ler_agg,
        "cer_kern": cer_kern,
        "ser_kern": ser_kern,
        "ler_kern": ler_kern,
        "cer_mxhm": cer_mxhm,
        "ser_mxhm": ser_mxhm,
        "ler_mxhm": ler_mxhm,
    }


def infer_tokenizer_type(checkpoint_path):
    """
    Infer tokenizer type from checkpoint path name.

    Extracts the last word after underscore from checkpoint filename.
    Examples:
    - smt_character.ckpt -> "character"
    - smt_word.ckpt -> "word"
    - smt_medium.ckpt -> "medium"

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        str: Inferred tokenizer type, defaults to "medium" if not found
    """
    ckpt_name = Path(checkpoint_path).stem.lower()

    # Extract last word after underscore(s)
    parts = ckpt_name.split('_')
    if len(parts) > 0:
        last_word = parts[-1]
        # Check if it's a valid tokenizer type
        if last_word in ['word', 'character', 'char', 'medium']:
            return 'character' if last_word == 'char' else last_word

    # Default to medium if not found
    return "medium"


def evaluate_multiple_checkpoints(checkpoint_paths, test_dir, split="test", fold=0, output_dir=None, device="cuda", tokenizer_type=None, csv_filename="evaluation_results.csv"):
    """
    Evaluate multiple checkpoints and save results to CSV.

    Args:
        checkpoint_paths: List of paths to checkpoint files, single path, or directory containing checkpoints
        test_dir: Path to test directory
        split: Which split to evaluate (train/val/test)
        fold: Fold number
        output_dir: Directory to save CSV
        device: Device to run on
        tokenizer_type: Tokenizer type to use (if None, inferred from checkpoint path)
        csv_filename: Name of output CSV file

    Returns:
        DataFrame with all results
    """
    # Handle single checkpoint path or directory
    if isinstance(checkpoint_paths, str):
        checkpoint_path = Path(checkpoint_paths)

        # If it's a directory, find all .ckpt files in it
        if checkpoint_path.is_dir():
            checkpoint_paths = sorted(checkpoint_path.glob("**/*.ckpt"))
            print(f"Found {len(checkpoint_paths)} checkpoints in directory: {checkpoint_path}")
        else:
            checkpoint_paths = [checkpoint_path]
    elif not isinstance(checkpoint_paths, list):
        checkpoint_paths = [checkpoint_paths]

    results = []

    print("=" * 70)
    print(f"EVALUATING {len(checkpoint_paths)} CHECKPOINT(S)")
    print("=" * 70)

    for i, ckpt_path in enumerate(checkpoint_paths, 1):
        print(f"\n[{i}/{len(checkpoint_paths)}] Evaluating: {ckpt_path}")
        print("-" * 70)

        try:
            # When evaluating multiple checkpoints, always infer tokenizer from checkpoint name
            # This ensures each checkpoint uses its correct tokenizer type
            ckpt_tokenizer_type = infer_tokenizer_type(ckpt_path)
            print(f"Using tokenizer type: {ckpt_tokenizer_type}")

            # Evaluate this checkpoint
            metrics = evaluate_test_set(
                ckpt_path,
                test_dir,
                split,
                fold,
                output_dir,
                device,
                ckpt_tokenizer_type
            )

            # Add checkpoint name and tokenizer type to metrics
            ckpt_name = Path(ckpt_path).stem
            metrics['checkpoint'] = ckpt_name
            metrics['checkpoint_path'] = str(ckpt_path)
            metrics['tokenizer_type'] = ckpt_tokenizer_type

            results.append(metrics)
            print(f"✓ Successfully evaluated checkpoint")

        except Exception as e:
            print(f"✗ Error evaluating {ckpt_path}: {str(e)}")
            # Infer tokenizer type for error record as well
            ckpt_tokenizer_type_err = infer_tokenizer_type(str(ckpt_path))
            results.append({
                'checkpoint': Path(ckpt_path).stem,
                'checkpoint_path': str(ckpt_path),
                'tokenizer_type': ckpt_tokenizer_type_err,
                'error': str(e)
            })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Reorder columns: checkpoint info first, then metrics
    col_order = ['checkpoint', 'checkpoint_path', 'tokenizer_type']
    metric_cols = [
        'cer', 'ser', 'ler',
        'cer_kern', 'ser_kern', 'ler_kern',
        'cer_mxhm', 'ser_mxhm', 'ler_mxhm'
    ]
    # Only include columns that exist
    col_order = [c for c in col_order if c in df.columns]
    metric_cols = [c for c in metric_cols if c in df.columns]
    col_order.extend(metric_cols)

    # Add error column if present
    if 'error' in df.columns:
        col_order.append('error')

    # Reorder only the columns that exist
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    # Save CSV
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        csv_path = Path(output_dir) / csv_filename
    else:
        csv_path = Path(csv_filename)

    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print(f"✓ Results saved to CSV: {csv_path}")
    print("=" * 70)
    print("\nEvaluation Summary:")
    print(df.to_string(index=False))

    return df


def run_inference(
    checkpoint_path,
    image_path=None,
    output_path=None,
    ground_truth_path=None,
    test_dir=None,
    split="test",
    fold=0,
    device="cuda",
    tokenizer_type="medium",
):
    """
    Run inference on single image or entire test set.

    Args:
        checkpoint_path: Path to trained model checkpoint
        image_path: Path to input image (for single inference)
        output_path: Optional path to save prediction
        ground_truth_path: Optional path to ground truth for scoring
        test_dir: Path to test directory (for batch evaluation)
        split: Which split to evaluate (train/val/test)
        fold: Fold number
        device: Device to run on
    """
    # Batch evaluation mode
    if test_dir:
        result = evaluate_test_set(checkpoint_path, test_dir, split, fold, output_path, device, tokenizer_type)
        print("result", result)
        return

    # Single image mode
    if not image_path:
        raise ValueError("Either image_path or test_dir must be provided")

    # Initialize inference
    inference = FullPageInference(checkpoint_path, device=device)

    # Predict
    result = inference.predict(image_path)

    # Display result
    inference.display_result(result)
    # Score if ground truth provided
    if ground_truth_path:
        inference.score_with_ground_truth(result['prediction'], ground_truth_path, tokenizer_type)



if __name__ == "__main__":
    # ============ CONFIGURATION ============
    # Edit these parameters and run: python inference.py

    # Single image inference
    IMAGE_PATH = "data/jazzmus_systems/jpg/img_0_8.jpg"  # Path to image (e.g., "path/to/image.jpg")
    GROUND_TRUTH_PATH = "data/jazzmus_systems/gt/img_0_8.txt"  # Path to ground truth (e.g., "path/to/gt.txt")
    # IMAGE_PATH = None
    # GROUND_TRUTH_PATH = None

    # Test set evaluation
    # TEST_DIR = "/home/hice1/jwang3180/jazzmus/ISMIR-Jazzmus/data/jazzmus_systems"  # Dataset directory
    TEST_DIR = None

    SPLIT = "test"  # Which split: train/val/test
    FOLD = 0  # Fold number
    TOKENIZER_TYPE = "medium"  # Tokenizer type: "word", "character", or "medium"

    # Model(s)
    CHECKPOINT_PATH = "/home/hice1/jwang3180/scratch/jazzmus/ISMIR-Jazzmus/weights/smt/smt_0.ckpt"

    # For CSV evaluation of multiple checkpoints:
    # Option A: List specific checkpoints
    # CHECKPOINT_PATHS = [
    #     "weights/smt_fullpage/smt_fullpage_0.ckpt",
    #     "weights/smt_fullpage/smt_fullpage_1.ckpt",
    #     "weights/smt_fullpage/smt_fullpage_2.ckpt",
    # ]

    # Option B: Pass directory path - automatically finds all .ckpt files
    # CHECKPOINT_PATHS = "weights/smt_fullpage"

    CHECKPOINT_PATHS = None

    DEVICE = "cuda"  # cuda or cpu

    # Output (optional, leave None to skip saving)
    OUTPUT_DIR = None  # Directory to save predictions/results

    # ============ RUN INFERENCE ============

    # Option 1: Single image or test set evaluation
    if CHECKPOINT_PATHS is None:
        run_inference(
            checkpoint_path=CHECKPOINT_PATH,
            image_path=IMAGE_PATH,
            output_path=OUTPUT_DIR,
            ground_truth_path=GROUND_TRUTH_PATH,
            test_dir=TEST_DIR,
            split=SPLIT,
            fold=FOLD,
            device=DEVICE,
            tokenizer_type=TOKENIZER_TYPE,
        )

    # Option 2: Evaluate multiple checkpoints and save to CSV
    else:
        evaluate_multiple_checkpoints(
            checkpoint_paths=CHECKPOINT_PATHS,
            test_dir=TEST_DIR,
            split=SPLIT,
            fold=FOLD,
            output_dir=OUTPUT_DIR,
            device=DEVICE,
            tokenizer_type=TOKENIZER_TYPE,
            csv_filename="evaluation_results.csv"
        )

