"""
Qualitative error analysis: baseline (YOLO + system-level SMT)
vs. replay100 (full-page curriculum) vs. ground truth.

Reads existing predictions in analysis_out/preds/ — does NOT recompute metrics.
For every GT chord and GT kern position, tags how each model did:
  correct / sub_root / sub_qual / sub_ext / sub_total / deleted
For each position, records (baseline_label, replay_label, near_linebreak).

Outputs (under --out_dir):
  shared_errors.md       what BOTH models get wrong
  replay_wins.md         what full-page fixes vs baseline
  replay_losses.md       what baseline still does better
  confusion_pairs.csv    long-form confusion table
  position_labels.csv    raw per-position labels (re-sliceable)
  hotspots/<page>.txt    side-by-side aligned diffs for top pages

Run:
  python analyze_errors.py \
      --preds_dir analysis_out/preds \
      --out_dir analysis_out/error_analysis \
      --linebreak_window 3
"""

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

from jazzmus.dataset.chord_metrics import (
    extract_spines,
    extract_tokens_from_mxhm,
    parse_chord,
)


LINEBREAK_MARK = "!!linebreak:original"


# ── alignment ──────────────────────────────────────────────────────────────────


@dataclass
class GTPos:
    """One GT position after alignment to a prediction."""
    gt_index: int
    gt_token: str
    label: str             # correct | sub_root | sub_qual | sub_ext | sub_total | deleted
    pred_token: Optional[str]


@dataclass
class Insertion:
    """An extra predicted token with no GT counterpart."""
    after_gt_index: int    # sits between gt_index and gt_index+1; -1 = before first GT
    pred_token: str


def align(pred: List[str], gt: List[str]) -> Tuple[List[GTPos], List[Insertion]]:
    """
    Edit-distance alignment with explicit per-GT-index labels.
    Returns (gt_positions, insertions). Each GT position gets a label
    placeholder of either 'matched' (token-equal), 'sub' (substituted),
    or 'deleted' (no aligned pred). Caller refines 'sub' into chord-level
    or kern-level subcategories.
    """
    m, n = len(pred), len(gt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == gt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Backtrack
    ops = []  # list of (op, pred_idx, gt_idx) walked in reverse
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and pred[i - 1] == gt[j - 1]:
            ops.append(("match", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("ins", i - 1, j))   # gt index where this pred sits
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("del", i, j - 1))
            j -= 1
        else:
            break
    ops.reverse()

    gt_positions: List[Optional[GTPos]] = [None] * n
    insertions: List[Insertion] = []
    for op, pi, gi in ops:
        if op == "match":
            gt_positions[gi] = GTPos(gi, gt[gi], "matched", pred[pi])
        elif op == "sub":
            gt_positions[gi] = GTPos(gi, gt[gi], "sub", pred[pi])
        elif op == "del":
            gt_positions[gi] = GTPos(gi, gt[gi], "deleted", None)
        elif op == "ins":
            after = gi - 1   # pred sits before GT index gi
            insertions.append(Insertion(after, pred[pi]))
    return [p for p in gt_positions if p is not None], insertions


# ── chord-level refinement ─────────────────────────────────────────────────────


def refine_chord_label(gt_tok: str, pred_tok: Optional[str], coarse: str) -> str:
    """Refine 'sub' into sub_root / sub_qual / sub_ext / sub_total."""
    if coarse == "matched":
        return "correct"
    if coarse == "deleted":
        return "deleted"
    if pred_tok is None:
        return "sub_total"
    g, p = parse_chord(gt_tok), parse_chord(pred_tok)
    if not g.is_valid or not p.is_valid:
        return "sub_total"
    if g.root != p.root:
        return "sub_root"
    if (g.quality or "") != (p.quality or ""):
        return "sub_qual"
    if (g.extension or "") != (p.extension or ""):
        return "sub_ext"
    return "sub_total"   # something else differed (modifier, bass)


# ── kern-level refinement ──────────────────────────────────────────────────────

import re

_KERN_DUR_RE = re.compile(r"^(\d+)(\.*)")
_KERN_PITCH_RE = re.compile(r"([a-gA-G]+)")
_KERN_ACC_RE = re.compile(r"([#\-]+|n)")


def _strip_kern(tok: str) -> str:
    return tok.replace("[", "").replace("]", "").strip()


def _kern_components(tok: str) -> dict:
    """Decompose a kern token into rough components for diffing."""
    t = _strip_kern(tok)
    is_rest = "r" in t
    is_meta = t.startswith("=") or t.startswith("*") or t.startswith("!")
    dur = None
    dur_dots = 0
    m = _KERN_DUR_RE.match(t)
    if m:
        dur = m.group(1)
        dur_dots = len(m.group(2))
    pitch = None
    pm = _KERN_PITCH_RE.search(t)
    if pm:
        pitch = pm.group(1)
    acc = ""
    am = _KERN_ACC_RE.search(t)
    if am:
        acc = am.group(1)
    # articulation chars: L J k K (beam markers)
    artic = "".join(sorted(c for c in t if c in "LJkK"))
    return dict(
        raw=t, is_rest=is_rest, is_meta=is_meta,
        dur=dur, dur_dots=dur_dots, pitch=pitch, acc=acc, artic=artic,
    )


def refine_kern_label(gt_tok: str, pred_tok: Optional[str], coarse: str) -> str:
    if coarse == "matched":
        return "correct"
    if coarse == "deleted":
        return "deleted"
    if pred_tok is None:
        return "sub_other"
    g, p = _kern_components(gt_tok), _kern_components(pred_tok)
    if g["is_rest"] != p["is_rest"]:
        return "sub_rest"
    if g["pitch"] != p["pitch"]:
        return "sub_pitch"
    if g["dur"] != p["dur"] or g["dur_dots"] != p["dur_dots"]:
        return "sub_duration"
    if g["acc"] != p["acc"]:
        return "sub_accidental"
    if g["artic"] != p["artic"]:
        return "sub_articulation"
    return "sub_other"


# ── linebreak proximity ───────────────────────────────────────────────────────


def gt_token_streams(gt_text: str) -> Tuple[List[str], List[bool], List[str], List[bool]]:
    """
    From raw GT kern text, produce:
      chord_tokens (mxhm chords, dots stripped),  chord_near_lb flags
      kern_tokens (kern column, headers/barlines/meta stripped, dots kept),
                                                  kern_near_lb flags
    A token is near-LB if it lies within window distance of a !!linebreak row;
    actual window is applied later — here we record the absolute token-distance
    to the nearest linebreak.
    """
    # Find spine indices first
    lines = gt_text.strip().split("\n")
    kern_idx = None
    mxhm_idx = None
    for line in lines:
        clean = line[5:].lstrip() if line.startswith("<bos>") else line
        if clean.startswith("**"):
            for i, p in enumerate(clean.split("\t")):
                if p.strip() == "**kern":
                    kern_idx = i
                if p.strip() == "**mxhm":
                    mxhm_idx = i
            break
    if kern_idx is None:
        return [], [], [], []

    chord_tokens: List[str] = []
    chord_lb_dist: List[int] = []          # +inf if no LB seen yet, refined below
    chord_lb_after: List[Optional[int]] = []  # tokens until next LB
    kern_tokens: List[str] = []
    kern_lb_dist: List[int] = []
    kern_lb_after: List[Optional[int]] = []

    # Walk lines once, recording both streams + LB events. A linebreak row
    # is one whose first non-empty parts equals LINEBREAK_MARK.
    chord_pos_of_lb: List[int] = []   # chord-token index at which LB occurs
    kern_pos_of_lb: List[int] = []

    for raw in lines:
        line = raw[5:].lstrip() if raw.startswith("<bos>") else raw
        line = line[5:].lstrip() if line.startswith("<eos>") else line
        s = line.strip()
        if not s:
            continue
        if s.startswith(LINEBREAK_MARK):
            chord_pos_of_lb.append(len(chord_tokens))
            kern_pos_of_lb.append(len(kern_tokens))
            continue
        if s.startswith("*"):
            continue
        parts = line.split("\t")
        # kern token (skip barlines for kern stream too — same treatment as
        # extract_chords_from_mxhm convention; barlines are structural noise)
        if kern_idx < len(parts):
            kt = parts[kern_idx].strip()
            if kt and not kt.startswith("=") and not kt.startswith("!"):
                kern_tokens.append(kt)
        # mxhm chord token (skip dots — chord stream is chord-only like
        # extract_chords_from_mxhm)
        if mxhm_idx is not None and mxhm_idx < len(parts):
            mt = parts[mxhm_idx].strip()
            if mt and mt != "." and not mt.startswith("=") and not mt.startswith("!"):
                chord_tokens.append(mt)

    def dists(n: int, lb_positions: List[int]) -> List[int]:
        if not lb_positions:
            return [10**9] * n
        out = []
        for i in range(n):
            # distance to closest linebreak position (where LB occurs *between* i-1 and i)
            d = min(min(abs(i - lb), abs(i + 1 - lb)) for lb in lb_positions)
            out.append(d)
        return out

    chord_dists = dists(len(chord_tokens), chord_pos_of_lb)
    kern_dists = dists(len(kern_tokens), kern_pos_of_lb)
    return chord_tokens, chord_dists, kern_tokens, kern_dists


def kern_tokens_from_pred(text: str) -> List[str]:
    """Extract the kern column from arbitrary pred text, dropping headers/meta/barlines."""
    lines = text.strip().split("\n")
    kern_idx = None
    for line in lines:
        clean = line[5:].lstrip() if line.startswith("<bos>") else line
        if clean.startswith("**"):
            for i, p in enumerate(clean.split("\t")):
                if p.strip() == "**kern":
                    kern_idx = i
                    break
            break
    if kern_idx is None:
        return []
    out = []
    for raw in lines:
        line = raw[5:].lstrip() if raw.startswith("<bos>") else raw
        line = line[5:].lstrip() if line.startswith("<eos>") else line
        s = line.strip()
        if not s or s.startswith("*") or s.startswith("!"):
            continue
        parts = line.split("\t")
        if kern_idx < len(parts):
            kt = parts[kern_idx].strip()
            if kt and not kt.startswith("="):
                out.append(kt)
    return out


def chord_tokens_from_pred(text: str) -> List[str]:
    """Chord-only mxhm stream (dots dropped)."""
    spines = extract_spines(text)
    if "**mxhm" not in spines:
        return []
    toks = extract_tokens_from_mxhm(spines["**mxhm"])
    return [t for t in toks if t != "."]


# ── per-page processing ───────────────────────────────────────────────────────


@dataclass
class PageRecord:
    page: str
    chord_rows: List[dict]   # one per GT chord position
    kern_rows: List[dict]    # one per GT kern position
    base_chord_inserts: List[Insertion]
    repl_chord_inserts: List[Insertion]
    base_kern_inserts: List[Insertion]
    repl_kern_inserts: List[Insertion]


def process_page(page: str, gt_text: str, base_text: str, repl_text: str,
                 lb_window: int) -> PageRecord:
    gt_chords, gt_chord_dist, gt_kern, gt_kern_dist = gt_token_streams(gt_text)
    base_chords = chord_tokens_from_pred(base_text)
    repl_chords = chord_tokens_from_pred(repl_text)
    base_kern = kern_tokens_from_pred(base_text)
    repl_kern = kern_tokens_from_pred(repl_text)

    # Drop linebreak-marker rows from kern streams (they appear in pred too)
    base_kern = [t for t in base_kern if t != LINEBREAK_MARK and not t.startswith("!")]
    repl_kern = [t for t in repl_kern if t != LINEBREAK_MARK and not t.startswith("!")]
    # GT kern stream already excludes ! lines

    # Align chords
    base_chord_pos, base_chord_ins = align(base_chords, gt_chords)
    repl_chord_pos, repl_chord_ins = align(repl_chords, gt_chords)

    # Align kern
    base_kern_pos, base_kern_ins = align(base_kern, gt_kern)
    repl_kern_pos, repl_kern_ins = align(repl_kern, gt_kern)

    # Index-by-gt-index for fast join
    def idx(positions: List[GTPos]) -> Dict[int, GTPos]:
        return {p.gt_index: p for p in positions}

    bc, rc = idx(base_chord_pos), idx(repl_chord_pos)
    bk, rk = idx(base_kern_pos), idx(repl_kern_pos)

    chord_rows = []
    for gi, gt_tok in enumerate(gt_chords):
        b = bc.get(gi)
        r = rc.get(gi)
        b_label = refine_chord_label(gt_tok, b.pred_token if b else None,
                                     b.label if b else "deleted")
        r_label = refine_chord_label(gt_tok, r.pred_token if r else None,
                                     r.label if r else "deleted")
        chord_rows.append({
            "page": page,
            "level": "chord",
            "gt_index": gi,
            "gt_token": gt_tok,
            "near_lb": gt_chord_dist[gi] <= lb_window,
            "lb_dist": gt_chord_dist[gi],
            "baseline_label": b_label,
            "baseline_pred": (b.pred_token if b else "") or "",
            "replay_label": r_label,
            "replay_pred": (r.pred_token if r else "") or "",
        })

    kern_rows = []
    for gi, gt_tok in enumerate(gt_kern):
        b = bk.get(gi)
        r = rk.get(gi)
        b_label = refine_kern_label(gt_tok, b.pred_token if b else None,
                                    b.label if b else "deleted")
        r_label = refine_kern_label(gt_tok, r.pred_token if r else None,
                                    r.label if r else "deleted")
        kern_rows.append({
            "page": page,
            "level": "kern",
            "gt_index": gi,
            "gt_token": gt_tok,
            "near_lb": gt_kern_dist[gi] <= lb_window,
            "lb_dist": gt_kern_dist[gi],
            "baseline_label": b_label,
            "baseline_pred": (b.pred_token if b else "") or "",
            "replay_label": r_label,
            "replay_pred": (r.pred_token if r else "") or "",
        })

    return PageRecord(
        page=page,
        chord_rows=chord_rows,
        kern_rows=kern_rows,
        base_chord_inserts=base_chord_ins,
        repl_chord_inserts=repl_chord_ins,
        base_kern_inserts=base_kern_ins,
        repl_kern_inserts=repl_kern_ins,
    )


# ── reporting helpers ─────────────────────────────────────────────────────────


def is_error(label: str) -> bool:
    return label != "correct"


def categorize_outcome(b: str, r: str) -> str:
    be, re_ = is_error(b), is_error(r)
    if not be and not re_:
        return "both_correct"
    if be and re_:
        return "both_wrong"
    if be and not re_:
        return "only_baseline_wrong"
    return "only_replay_wrong"


def render_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "(none)\n"
    col_w = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = "| " + " | ".join(h.ljust(col_w[i]) for i, h in enumerate(headers)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in col_w) + "|"
    body = "\n".join(
        "| " + " | ".join(r[i].ljust(col_w[i]) for i in range(len(headers))) + " |"
        for r in rows
    )
    return "\n".join([line, sep, body]) + "\n"


def label_dist(rows: List[dict], key: str) -> Counter:
    return Counter(r[key] for r in rows)


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ── hotspot diff ──────────────────────────────────────────────────────────────


def hotspot_diff(page: str, base_text: str, repl_text: str, gt_text: str) -> str:
    """Render a row-aligned 3-way view by walking GT lines and showing what
    each model produced at the corresponding positions of its own stream
    (best-effort; uses the shared chord+kern alignment to map back)."""
    gt_chords, _, gt_kern, _ = gt_token_streams(gt_text)
    base_chords = chord_tokens_from_pred(base_text)
    repl_chords = chord_tokens_from_pred(repl_text)
    base_kern = [t for t in kern_tokens_from_pred(base_text)
                 if t != LINEBREAK_MARK and not t.startswith("!")]
    repl_kern = [t for t in kern_tokens_from_pred(repl_text)
                 if t != LINEBREAK_MARK and not t.startswith("!")]

    bcp, _ = align(base_chords, gt_chords)
    rcp, _ = align(repl_chords, gt_chords)
    bkp, _ = align(base_kern, gt_kern)
    rkp, _ = align(repl_kern, gt_kern)

    bc = {p.gt_index: p.pred_token for p in bcp}
    rc = {p.gt_index: p.pred_token for p in rcp}
    bk = {p.gt_index: p.pred_token for p in bkp}
    rk = {p.gt_index: p.pred_token for p in rkp}

    out = [f"=== {page} ===", "",
           "CHORD STREAM",
           f"{'idx':>4} {'GT':<14} {'baseline':<14} {'replay':<14}  flags"]
    lb_chord_positions = []
    # Recompute LB positions for display (cheap)
    _, _, _, _ = gt_token_streams(gt_text)
    # find from raw text walking
    chord_count = 0
    for raw in gt_text.strip().split("\n"):
        line = raw[5:].lstrip() if raw.startswith("<bos>") else raw
        s = line.strip()
        if not s:
            continue
        if s.startswith(LINEBREAK_MARK):
            lb_chord_positions.append(chord_count)
            continue
        if s.startswith("*"):
            continue
        parts = line.split("\t")
        # assume **mxhm is 2nd column; if not, skip
        if len(parts) > 1:
            mt = parts[1].strip()
            if mt and mt != "." and not mt.startswith("=") and not mt.startswith("!"):
                chord_count += 1

    for i, g in enumerate(gt_chords):
        b = bc.get(i, "<DEL>") or "<DEL>"
        r = rc.get(i, "<DEL>") or "<DEL>"
        flags = []
        if b != g:
            flags.append("B!")
        if r != g:
            flags.append("R!")
        if i in lb_chord_positions:
            out.append(f"  {'-' * 60}  --- linebreak ---")
        out.append(f"{i:>4} {g:<14} {b:<14} {r:<14}  {' '.join(flags)}")

    out.append("")
    out.append("KERN STREAM")
    out.append(f"{'idx':>4} {'GT':<18} {'baseline':<18} {'replay':<18}  flags")
    for i, g in enumerate(gt_kern):
        b = bk.get(i, "<DEL>") or "<DEL>"
        r = rk.get(i, "<DEL>") or "<DEL>"
        flags = []
        if b != g:
            flags.append("B!")
        if r != g:
            flags.append("R!")
        out.append(f"{i:>4} {g:<18} {b:<18} {r:<18}  {' '.join(flags)}")

    return "\n".join(out) + "\n"


# ── main ──────────────────────────────────────────────────────────────────────


def main(
    preds_dir: str = "analysis_out/preds",
    out_dir: str = "analysis_out/error_analysis",
    per_sample_csv: str = "analysis_out/per_sample_metrics.csv",
    linebreak_window: int = 3,
    top_n: int = 15,
    n_hotspots: int = 5,
):
    preds = Path(preds_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "hotspots").mkdir(exist_ok=True)

    # discover pages
    page_ids = sorted({
        f.name.replace("__gt.kern", "")
        for f in preds.glob("*__gt.kern")
    })
    if not page_ids:
        print(f"No pages found under {preds}")
        return

    print(f"Processing {len(page_ids)} pages...")
    records: List[PageRecord] = []
    for pid in page_ids:
        gt_text = (preds / f"{pid}__gt.kern").read_text()
        base_text = (preds / f"{pid}__baseline.kern").read_text()
        repl_text = (preds / f"{pid}__replay100.kern").read_text()
        records.append(process_page(pid, gt_text, base_text, repl_text, linebreak_window))

    # ── flatten ────────────────────────────────────────────────────────────────
    all_chord_rows = [r for rec in records for r in rec.chord_rows]
    all_kern_rows = [r for rec in records for r in rec.kern_rows]
    for r in all_chord_rows + all_kern_rows:
        r["outcome"] = categorize_outcome(r["baseline_label"], r["replay_label"])
        r["near_lb"] = "1" if r["near_lb"] else "0"

    # ── position_labels.csv ───────────────────────────────────────────────────
    pos_fields = ["page", "level", "gt_index", "gt_token", "near_lb", "lb_dist",
                  "baseline_label", "baseline_pred", "replay_label", "replay_pred",
                  "outcome"]
    write_csv(out / "position_labels.csv", all_chord_rows + all_kern_rows, pos_fields)

    # ── confusion pairs ───────────────────────────────────────────────────────
    confusion_rows = []

    def collect_confusions(rows: List[dict], level: str):
        bc = Counter()
        rc = Counter()
        for r in rows:
            if r["baseline_label"] not in ("correct", "deleted") and r["baseline_pred"]:
                bc[(r["gt_token"], r["baseline_pred"])] += 1
            if r["replay_label"] not in ("correct", "deleted") and r["replay_pred"]:
                rc[(r["gt_token"], r["replay_pred"])] += 1
        # plus deletions: gt_token -> "<DEL>"
        for r in rows:
            if r["baseline_label"] == "deleted":
                bc[(r["gt_token"], "<DEL>")] += 1
            if r["replay_label"] == "deleted":
                rc[(r["gt_token"], "<DEL>")] += 1
        keys = set(bc) | set(rc)
        for k in keys:
            confusion_rows.append({
                "model": "baseline", "level": level,
                "gt": k[0], "pred": k[1],
                "count": bc[k], "also_in_other_model": int(rc[k] > 0),
            })
            confusion_rows.append({
                "model": "replay", "level": level,
                "gt": k[0], "pred": k[1],
                "count": rc[k], "also_in_other_model": int(bc[k] > 0),
            })
        return bc, rc

    chord_bc, chord_rc = collect_confusions(all_chord_rows, "chord")
    kern_bc, kern_rc = collect_confusions(all_kern_rows, "kern")
    write_csv(out / "confusion_pairs.csv",
              [r for r in confusion_rows if r["count"] > 0],
              ["model", "level", "gt", "pred", "count", "also_in_other_model"])

    # ── shared_errors.md ──────────────────────────────────────────────────────
    def shared_section(rows: List[dict], level: str) -> str:
        both_wrong = [r for r in rows if r["outcome"] == "both_wrong"]
        same_pred = sum(1 for r in both_wrong if r["baseline_pred"] == r["replay_pred"])
        diff_pred = len(both_wrong) - same_pred
        # top GT tokens both fail on
        gt_counter = Counter(r["gt_token"] for r in both_wrong)
        rows_t = [[g, str(c)] for g, c in gt_counter.most_common(top_n)]
        # top confusion pairs that occur in BOTH models' confusion sets
        if level == "chord":
            bc, rc = chord_bc, chord_rc
        else:
            bc, rc = kern_bc, kern_rc
        joint = [(k, min(bc[k], rc[k])) for k in set(bc) & set(rc)]
        joint.sort(key=lambda x: -x[1])
        joint_t = [[k[0], k[1], str(c)] for k, c in joint[:top_n]]

        s = [f"## {level.upper()} level"]
        s.append(f"- Total GT positions: {len(rows)}")
        s.append(f"- Both wrong: {len(both_wrong)} "
                 f"(same pred: {same_pred}, divergent pred: {diff_pred})")
        s.append("")
        s.append(f"### Top GT tokens both models miss")
        s.append(render_table(["gt", "count"], rows_t))
        s.append(f"### Top confusion pairs that appear in BOTH models")
        s.append(render_table(["gt", "pred", "min_count"], joint_t))
        return "\n".join(s)

    shared_md = ["# Shared Errors — what BOTH models get wrong",
                 "",
                 "These are mistakes both baseline and full-page commit at the same GT "
                 "position. When both models agree on the *same* wrong answer the GT or "
                 "image is likely the cause; when they diverge they each have their "
                 "own systematic bias.",
                 ""]
    shared_md.append(shared_section(all_chord_rows, "chord"))
    shared_md.append("")
    shared_md.append(shared_section(all_kern_rows, "kern"))
    (out / "shared_errors.md").write_text("\n".join(shared_md))

    # ── replay_wins.md & replay_losses.md ─────────────────────────────────────

    def diff_section(rows: List[dict], level: str, winner: str) -> str:
        # winner = "replay" → list only_baseline_wrong; "baseline" → only_replay_wrong
        target = "only_baseline_wrong" if winner == "replay" else "only_replay_wrong"
        loser_label_key = "baseline_label" if winner == "replay" else "replay_label"
        loser_pred_key = "baseline_pred" if winner == "replay" else "replay_pred"

        target_rows = [r for r in rows if r["outcome"] == target]
        # error-type breakdown
        type_counter = Counter(r[loser_label_key] for r in target_rows)
        type_t = [[k, str(v)] for k, v in type_counter.most_common()]

        # Linebreak distance gradient: at each distance threshold d, what's
        # the error rate of the loser model on positions whose distance to the
        # nearest LB is exactly d? If LB-driven, this should be high at d=0
        # and drop off. Compare to overall error rate as a baseline.
        loser_label_for = lambda r: r[loser_label_key]
        loser_err_total = sum(1 for r in rows if loser_label_for(r) != "correct")
        loser_err_rate_overall = (loser_err_total / len(rows) * 100) if rows else 0.0

        max_d = 6
        gradient_t = []
        for d in range(max_d + 1):
            bucket = [r for r in rows if r["lb_dist"] == d]
            if not bucket:
                continue
            err = sum(1 for r in bucket if loser_label_for(r) != "correct")
            only_loser = sum(1 for r in bucket if r["outcome"] == target)
            gradient_t.append([
                str(d), str(len(bucket)),
                f"{err}",
                f"{err/len(bucket)*100:.1f}%",
                str(only_loser),
                f"{only_loser/len(bucket)*100:.1f}%",
            ])
        # also a "≥max_d+1" bucket
        bucket = [r for r in rows if r["lb_dist"] > max_d]
        if bucket:
            err = sum(1 for r in bucket if loser_label_for(r) != "correct")
            only_loser = sum(1 for r in bucket if r["outcome"] == target)
            gradient_t.append([
                f"≥{max_d+1}", str(len(bucket)),
                f"{err}",
                f"{err/len(bucket)*100:.1f}%",
                str(only_loser),
                f"{only_loser/len(bucket)*100:.1f}%",
            ])

        # top GT tokens the loser misses but the winner gets right
        gt_counter = Counter(r["gt_token"] for r in target_rows)
        gt_t = [[g, str(c)] for g, c in gt_counter.most_common(top_n)]

        # top confusion pairs unique to loser (not in winner's confusion set)
        if level == "chord":
            loser_conf = chord_bc if winner == "replay" else chord_rc
            other_conf = chord_rc if winner == "replay" else chord_bc
        else:
            loser_conf = kern_bc if winner == "replay" else kern_rc
            other_conf = kern_rc if winner == "replay" else kern_bc
        unique_conf = [(k, c) for k, c in loser_conf.items() if other_conf.get(k, 0) == 0]
        unique_conf.sort(key=lambda x: -x[1])
        unique_t = [[k[0], k[1], str(c)] for k, c in unique_conf[:top_n]]

        s = [f"## {level.upper()} level"]
        s.append(f"- Total GT positions: {len(rows)}")
        s.append(f"- Positions where {('only baseline wrong' if winner=='replay' else 'only replay wrong')}: {len(target_rows)}")
        s.append("")
        s.append("### Error-type breakdown")
        s.append(render_table(["error_type", "count"], type_t))

        loser_name = "baseline" if winner == "replay" else "replay"
        s.append(f"### Linebreak distance gradient")
        s.append(f"`lb_dist` = token distance to nearest `!!linebreak`. "
                 f"Compare {loser_name}'s error rate at each distance to the overall "
                 f"rate ({loser_err_rate_overall:.1f}%). If the linebreak hypothesis holds, "
                 f"{loser_name} errors should concentrate at small distances.")
        s.append("")
        s.append(render_table(
            ["lb_dist", "N", f"{loser_name}_err", f"{loser_name}_err_rate",
             f"only_{loser_name}_wrong", f"only_{loser_name}_rate"],
            gradient_t,
        ))
        s.append(f"### Top GT tokens the {('baseline' if winner=='replay' else 'replay')} misses but the other gets right")
        s.append(render_table(["gt", "count"], gt_t))

        s.append(f"### Top confusion pairs unique to {('baseline' if winner=='replay' else 'replay')}")
        s.append(render_table(["gt", "pred", "count"], unique_t))
        return "\n".join(s)

    wins_md = ["# Replay100 Wins — what full-page fixes vs baseline", ""]
    wins_md.append(diff_section(all_chord_rows, "chord", "replay"))
    wins_md.append("")
    wins_md.append(diff_section(all_kern_rows, "kern", "replay"))
    (out / "replay_wins.md").write_text("\n".join(wins_md))

    losses_md = ["# Replay100 Losses — what baseline still does better", ""]
    losses_md.append(diff_section(all_chord_rows, "chord", "baseline"))
    losses_md.append("")
    losses_md.append(diff_section(all_kern_rows, "kern", "baseline"))
    (out / "replay_losses.md").write_text("\n".join(losses_md))

    # ── hotspots ──────────────────────────────────────────────────────────────
    hotspot_pages = []
    try:
        with open(per_sample_csv) as f:
            rdr = list(csv.DictReader(f))
        for row in rdr:
            row["delta_chord_ser"] = float(row["delta_chord_ser"])
        rdr.sort(key=lambda r: r["delta_chord_ser"], reverse=True)
        hotspot_pages.extend([r["page"] for r in rdr[:n_hotspots]])
        hotspot_pages.extend([r["page"] for r in rdr[-n_hotspots:]])
    except Exception as e:
        print(f"  could not load per_sample_csv ({e}); skipping hotspots")

    for pid in hotspot_pages:
        gt_p = preds / f"{pid}__gt.kern"
        base_p = preds / f"{pid}__baseline.kern"
        repl_p = preds / f"{pid}__replay100.kern"
        if not (gt_p.exists() and base_p.exists() and repl_p.exists()):
            continue
        diff = hotspot_diff(pid, base_p.read_text(), repl_p.read_text(), gt_p.read_text())
        (out / "hotspots" / f"{pid}.txt").write_text(diff)

    # ── stdout summary ────────────────────────────────────────────────────────
    def quick_summary(rows: List[dict], name: str) -> str:
        total = len(rows)
        oc = Counter(r["outcome"] for r in rows)
        return (f"  {name:<6}  total={total:5d}  "
                f"both_correct={oc['both_correct']:5d}  "
                f"both_wrong={oc['both_wrong']:5d}  "
                f"only_base_wrong={oc['only_baseline_wrong']:5d}  "
                f"only_repl_wrong={oc['only_replay_wrong']:5d}")

    print("\nGT positions outcome split:")
    print(quick_summary(all_chord_rows, "chord"))
    print(quick_summary(all_kern_rows, "kern"))
    print(f"\nWrote outputs to {out.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preds_dir", default="analysis_out/preds")
    p.add_argument("--out_dir", default="analysis_out/error_analysis")
    p.add_argument("--per_sample_csv", default="analysis_out/per_sample_metrics.csv")
    p.add_argument("--linebreak_window", type=int, default=3)
    p.add_argument("--top_n", type=int, default=15)
    p.add_argument("--n_hotspots", type=int, default=5)
    args = p.parse_args()
    main(**vars(args))
