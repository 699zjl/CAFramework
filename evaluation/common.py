"""
Common utilities: GT loading, answer parsing, F1 computation, task/model discovery.

Reuses evaluate.py patterns for GT loading and label_level_ss_f1 for metrics.
"""

from __future__ import annotations

import json
import logging
import os
import re
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Letters A-Z index (matches evaluate.py)
LETTER_IDX = list(map(chr, range(ord("A"), ord("Z") + 1)))

# Regex for fast letter extraction
_LETTER_RE = re.compile(r"\b([A-E])\b")
_TRAILING_LETTER_RE = re.compile(r"([A-E])[.\):\s]*$")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TaskInfo:
    level: str        # e.g. "L3-lesion_analysis"
    task: str         # e.g. "L3a-lesion_recognition"
    subtask: str      # e.g. "L3a1-lesion_recognition-Haemorrhage" (no .json)
    rel_path: str     # e.g. "L3-lesion_analysis/L3a-lesion_recognition/L3a1-...json"
    gt_path: str      # absolute path to GT json


# ---------------------------------------------------------------------------
# Answer letter extraction (fast, no SentenceSimilarity)
# ---------------------------------------------------------------------------

def extract_letter(raw: str) -> Optional[str]:
    """Extract single answer letter (A-E) from raw model output.

    Tries in order:
    1. Single character
    2. Trailing 'B.' or 'B)'
    3. First word-boundary letter A-E
    Returns None if nothing found.
    """
    if not raw:
        return None
    raw = raw.strip()
    if len(raw) == 1 and raw.upper() in "ABCDE":
        return raw.upper()
    m = _TRAILING_LETTER_RE.search(raw)
    if m:
        return m.group(1).upper()
    m = _LETTER_RE.search(raw.upper())
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# GT loading
# ---------------------------------------------------------------------------

def load_gt(gt_path: str) -> dict:
    """Load FunBench GT file. Returns data dict keyed by image path."""
    if not os.path.exists(gt_path):
        return {}
    with open(gt_path) as f:
        obj = json.load(f)
    return obj.get("data", obj)  # some files have top-level 'data', some don't


def load_answers(path: str) -> Optional[dict]:
    """Load a prediction JSON file. Returns None if missing or empty."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        try:
            d = json.load(f)
        except json.JSONDecodeError:
            logger.warning("JSON decode error: %s", path)
            return None
    if not d:
        return None
    return d


def load_answers_for_task(
    answers_dir: str, model: str, emode: str, task_rel: str
) -> Optional[dict]:
    """Load answers/test/<model>/<emode>/<task_rel>.json"""
    path = os.path.join(answers_dir, model, emode, task_rel)
    return load_answers(path)


# ---------------------------------------------------------------------------
# Task / model discovery
# ---------------------------------------------------------------------------

def discover_models(answers_dir: str) -> list[str]:
    """List model subdirectories under answers_dir."""
    if not os.path.isdir(answers_dir):
        return []
    return sorted(
        d for d in os.listdir(answers_dir)
        if os.path.isdir(os.path.join(answers_dir, d))
    )


def discover_tasks(gt_dir: str, level_filter: Optional[list[str]] = None) -> list[TaskInfo]:
    """Walk FunBench/ and return all task JSON files as TaskInfo objects.

    level_filter: if given, only include levels starting with any of these prefixes
                  e.g. ['L3', 'L4']
    """
    tasks = []
    if not os.path.isdir(gt_dir):
        logger.warning("GT dir not found: %s", gt_dir)
        return tasks

    for level in sorted(os.listdir(gt_dir)):
        level_path = os.path.join(gt_dir, level)
        if not os.path.isdir(level_path):
            continue
        if level_filter and not any(level.startswith(p) for p in level_filter):
            continue

        for entry in sorted(os.listdir(level_path)):
            entry_path = os.path.join(level_path, entry)

            if entry.endswith(".json"):
                # Flat task: FunBench/L1-*/L1a-*.json
                rel = os.path.join(level, entry)
                tasks.append(TaskInfo(
                    level=level,
                    task=entry.replace(".json", ""),
                    subtask=entry.replace(".json", ""),
                    rel_path=rel,
                    gt_path=os.path.join(gt_dir, rel),
                ))
            elif os.path.isdir(entry_path):
                # Task dir: FunBench/L3-*/L3a-*/L3a1-*.json
                task_name = entry
                for subtask_file in sorted(os.listdir(entry_path)):
                    if not subtask_file.endswith(".json"):
                        continue
                    rel = os.path.join(level, task_name, subtask_file)
                    tasks.append(TaskInfo(
                        level=level,
                        task=task_name,
                        subtask=subtask_file.replace(".json", ""),
                        rel_path=rel,
                        gt_path=os.path.join(gt_dir, rel),
                    ))
    return tasks


# ---------------------------------------------------------------------------
# F1 / accuracy computation
# ---------------------------------------------------------------------------

def _fast_hist(label_true, label_pred, n_class=2):
    encoding = n_class * label_true.astype(int) + label_pred.astype(int)
    hist = np.bincount(encoding.flatten(), minlength=n_class ** 2)
    return hist.reshape(n_class, n_class)


def compute_accuracy(
    gt_data: dict,
    pred_data: dict,
    emode: str,
) -> Optional[float]:
    """Compute mean F1 (or sensitivity for binary tasks) for one task.

    gt_data: {img: {gt: "A", raw_data: {options: [...], gt: ...}, E-mode2: ..., E-mode3: ...}}
    pred_data: {img: raw_model_output_str}
    emode: "emode2" or "emode3" — used to pick E-mode2/E-mode3 prompt for options

    Returns float in [0,1] or None if insufficient data.
    """
    emode_key = "E-mode2" if emode == "emode2" else "E-mode3"

    records = []
    for img, gt_entry in gt_data.items():
        if img not in pred_data:
            continue
        gt_letter = gt_entry.get("gt", "")
        if not gt_letter:
            continue
        raw_options = gt_entry.get("raw_data", {}).get("options", [])
        if not raw_options:
            continue

        pred_raw = pred_data[img]
        pred_letter = extract_letter(str(pred_raw)) if pred_raw else None
        if pred_letter is None:
            pred_letter = "X"  # unmapped → wrong

        records.append((gt_letter, pred_letter, raw_options))

    if len(records) < 5:
        return None

    # Build all categories
    all_cats = sorted(set(LETTER_IDX[i] for i in range(len(records[0][2]))))
    if not all_cats:
        return None

    n = len(records)
    nc = len(all_cats)
    cat_idx = {c: i for i, c in enumerate(all_cats)}

    gt_mat = np.zeros((n, nc), dtype=int)
    pred_mat = np.zeros((n, nc), dtype=int)
    mask_mat = np.ones((n, nc), dtype=int)

    for i, (gt_l, pred_l, options) in enumerate(records):
        # GT may be multi-label "A,B"
        for g in gt_l.split(","):
            g = g.strip()
            if g in cat_idx:
                gt_mat[i, cat_idx[g]] = 1
        if pred_l in cat_idx:
            pred_mat[i, cat_idx[pred_l]] = 1

    # Compute F1 per category then average
    f1s = []
    for ci in range(nc):
        gt_c = gt_mat[:, ci]
        pred_c = pred_mat[:, ci]
        tp = int((gt_c * pred_c).sum())
        fp = int(((1 - gt_c) * pred_c).sum())
        fn = int((gt_c * (1 - pred_c)).sum())
        tn = int(((1 - gt_c) * (1 - pred_c)).sum())
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        denom = sens + spec
        f1 = 2 * sens * spec / denom if denom > 0 else 0.0
        f1s.append(f1)

    return float(np.mean(f1s)) if f1s else None


def get_emodes_for_model(answers_dir: str, model: str) -> list[str]:
    """Return available emode subdirs for a model."""
    model_dir = os.path.join(answers_dir, model)
    if not os.path.isdir(model_dir):
        return []
    return sorted(
        d for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d))
    )
