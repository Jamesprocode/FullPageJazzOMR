"""
Core stacking primitive for multi-system image assembly.

Provides stack_systems() and its token-manipulation helpers.
No jazzmus imports — pure numpy / cv2 / Python stdlib so this module
can be imported by any dataset class without circular dependencies.

A <linebreak> token is inserted between systems in the ground-truth
sequence so the model learns to recognise page-line boundaries.
"""

import random

import cv2
import numpy as np


# ── ground-truth helpers ───────────────────────────────────────────────────────

def _strip_header(tokens):
    """Return tokens from the first barline onward (drops kern header lines)."""
    for i, tok in enumerate(tokens):
        if tok.startswith("="):
            return tokens[i:]
    return tokens


def _strip_leading_linebreaks(tokens):
    """Remove !!linebreak:original lines that appear before the first barline.

    System files from the middle of a page include !!linebreak:original in
    their header area (between the kern spine declarations and the first =
    barline).  When used as the first system in a stack (j=0), this must be
    dropped so the stacked page never opens with a linebreak marker.
    """
    lines, cur = [], []
    for tok in tokens:
        cur.append(tok)
        if tok == "<n>":
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)

    result = []
    first_barline_seen = False
    for line in lines:
        content = [t for t in line if t not in ("<t>", "<n>")]
        if first_barline_seen:
            result.extend(line)
        elif any(t.startswith("=") for t in content):
            first_barline_seen = True
            result.extend(line)
        elif any(t == "!!linebreak:original" for t in content):
            pass  # drop — no place before the first barline
        else:
            result.extend(line)
    return result


def _strip_trailing_double_barlines(tokens):
    """Remove trailing double-barline (==) lines from a music token list.

    Individual system files often close their last measure with == (a kern
    final barline).  In a non-last stacked system this is misleading because
    more music follows.
    """
    lines, cur = [], []
    for tok in tokens:
        cur.append(tok)
        if tok == "<n>":
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)

    while lines:
        content = [t for t in lines[-1] if t not in ("<t>", "<n>")]
        if content and all(t.startswith("==") for t in content):
            lines.pop()
        else:
            break
    return [t for line in lines for t in line]


def _split_music_tail(tokens):
    """Split a token sequence into (music, tail).

    Tail = trailing lines that contain no music tokens (only spine
    terminators like *-, !!linebreak:original, and similar).
    A music line is any line with at least one token starting with a
    digit (note duration) or '=' (barline).
    """
    lines, cur = [], []
    for tok in tokens:
        cur.append(tok)
        if tok == "<n>":
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)

    split = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        content = [t for t in lines[i] if t not in ("<t>", "<n>")]
        if any(t[0].isdigit() or t.startswith("=") for t in content if t):
            split = i + 1
            break

    flat = lambda ls: [t for line in ls for t in line]
    return flat(lines[:split]), flat(lines[split:])


# ── main stacking function ─────────────────────────────────────────────────────

def stack_systems(imgs, gts, n, system_height=128, paths=None, indices=None):
    """
    Stack n systems vertically and concatenate their ground-truth tokens.

    If `indices` is provided (e.g. consecutive systems from the same page),
    those exact systems are used in order.  Otherwise n systems are sampled
    randomly from the full pool.

    Systems are normalised to `system_height` pixels tall (preserving aspect
    ratio).  Any remaining width difference inside the stack is padded with
    white (255).  A <linebreak> token is inserted between systems in the GT
    so the model learns to recognise page-line boundaries.

    Args:
        imgs:          list of numpy (H, W) grayscale arrays.
        gts:           list of token lists starting with <bos>, ending with <eos>.
        n:             number of systems to stack.
        system_height: target height in pixels for every row.
        paths:         optional list of file paths for debug logging.
        indices:       optional list of indices to use (skips random sampling).

    Returns:
        stacked (numpy H×W), combined_gt (list of str tokens),
        sampled_paths (list of str | None)
    """
    if indices is None:
        indices = random.sample(range(len(imgs)), min(n, len(imgs)))

    resized = []
    for idx in indices:
        img = imgs[idx]
        h, w = img.shape[:2]
        if h != system_height:
            new_w = max(1, int(round(w * system_height / h)))
            img = cv2.resize(img, (new_w, system_height), interpolation=cv2.INTER_LINEAR)
        resized.append(img)

    max_w = max(img.shape[1] for img in resized)
    padded = [
        np.pad(img, ((0, 0), (0, max_w - img.shape[1])), constant_values=255)
        for img in resized
    ]
    stacked = np.vstack(padded)

    # Concatenate GT:
    #   sys1 : full kern header + music (tail stripped)
    #   sys2…N : <linebreak> + music only (header stripped, tail stripped for non-last)
    #   sysN  : keeps *- spine terminators
    combined = ["<bos>"]
    for j, gt in enumerate([gts[i] for i in indices]):
        inner = gt[1:-1]      # strip <bos> / <eos>
        is_last = (j == n - 1)

        if j > 0:
            inner = _strip_header(inner)
            combined += ["<linebreak>"]

        if is_last:
            combined += inner
        else:
            music, _ = _split_music_tail(inner)
            combined += music
    combined += ["<eos>"]

    sampled_paths = [paths[i] for i in indices] if paths is not None else None
    return stacked, combined, sampled_paths
