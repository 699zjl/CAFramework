"""
Experiment A: L7 Option-Order Robustness

Tests whether model predictions are stable when answer options are cyclically
rotated (rot1, rot2). Predictions are remapped back to the original option
space via the inverse of option_map before comparison.

Data layout:
  answers_l7/<model>/emode{N}_rot{1,2}/<task_rel>.json   — raw predictions
  answers_l7/<model>/emode{N}_rot{1,2}_meta/<task_rel>.json — per-image meta
  answers/test/<model>/emode{N}/<task_rel>.json           — rot0 (baseline)

Meta file format (per image):
  {
    "rotation": 1,
    "original_gt": "C",
    "rotated_gt": "B",
    "option_map": {"A": "A", "B": "C", "C": "D", ...}  # original -> rotated
  }

Answer remapping:
  inverse_map = {v: k for k, v in option_map.items()}
  pred_original = inverse_map.get(pred_rotated)

Metrics:
  consistency_rate   = fraction of images where rot0 == rot1 == rot2 (original space)
  position_bias_score = 1 - consistency_rate
  accuracy_rot0/1/2  = mean F1 for each rotation
  accuracy_drop      = accuracy_rot0 - mean(accuracy_rot1, accuracy_rot2)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Optional

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.common import (
    TaskInfo,
    compute_accuracy,
    discover_tasks,
    extract_letter,
    load_answers,
    load_answers_for_task,
    load_gt,
)

logger = logging.getLogger(__name__)

# Rotation suffixes
_ROTATIONS = [1, 2]


def _load_meta(l7_dir: str, model: str, emode_base: str, rot: int, task_rel: str) -> Optional[dict]:
    """Load meta file for a rotation. Returns dict keyed by image path."""
    meta_emode = f"{emode_base}_rot{rot}_meta"
    path = os.path.join(l7_dir, model, meta_emode, task_rel)
    return load_answers(path)


def _load_rot_answers(l7_dir: str, model: str, emode_base: str, rot: int, task_rel: str) -> Optional[dict]:
    """Load rotated answer file."""
    rot_emode = f"{emode_base}_rot{rot}"
    path = os.path.join(l7_dir, model, rot_emode, task_rel)
    return load_answers(path)


def _remap_to_original(pred_rotated: Optional[str], option_map: dict) -> Optional[str]:
    """Map a rotated prediction letter back to original option space.

    option_map: original -> rotated
    inverse_map: rotated -> original
    """
    if pred_rotated is None:
        return None
    inverse_map = {v: k for k, v in option_map.items()}
    return inverse_map.get(pred_rotated)


def run_exp_a(
    l7_dir: str,
    answers_dir: str,
    gt_dir: str,
    models: Optional[list[str]] = None,
    base_emodes: Optional[list[str]] = None,
    level_filter: Optional[list[str]] = None,
) -> list[dict]:
    """Run Experiment A: L7 option-order robustness.

    Args:
        l7_dir: Path to answers_l7/
        answers_dir: Path to answers/test/ (for rot0 baseline)
        gt_dir: Path to FunBench/
        models: Model names to evaluate (None = auto-discover from l7_dir)
        base_emodes: Base emodes, e.g. ['emode2', 'emode3'] (None = auto)
        level_filter: Level prefixes to include

    Returns:
        List of result dicts, one per (model, emode, task).
    """
    if not os.path.isdir(l7_dir):
        logger.warning("L7 dir not found: %s", l7_dir)
        return []

    if models is None:
        models = sorted(
            d for d in os.listdir(l7_dir)
            if os.path.isdir(os.path.join(l7_dir, d))
        )

    tasks = discover_tasks(gt_dir, level_filter=level_filter)
    if not tasks:
        logger.warning("No tasks found in %s", gt_dir)
        return []

    results = []

    for model in models:
        model_l7_dir = os.path.join(l7_dir, model)
        if not os.path.isdir(model_l7_dir):
            continue

        # Discover available base emodes from rot1 dirs
        if base_emodes is None:
            avail_emodes = sorted(set(
                d.replace("_rot1", "")
                for d in os.listdir(model_l7_dir)
                if d.endswith("_rot1") and not d.endswith("_meta")
                and os.path.isdir(os.path.join(model_l7_dir, d))
            ))
        else:
            avail_emodes = base_emodes

        for emode_base in avail_emodes:
            logger.info("Exp A: model=%s emode=%s", model, emode_base)
            for task_info in tasks:
                row = _eval_task(
                    l7_dir=l7_dir,
                    answers_dir=answers_dir,
                    model=model,
                    emode_base=emode_base,
                    task_info=task_info,
                )
                if row is not None:
                    results.append(row)

    return results


def _eval_task(
    l7_dir: str,
    answers_dir: str,
    model: str,
    emode_base: str,
    task_info: TaskInfo,
) -> Optional[dict]:
    """Evaluate one (model, emode, task) for L7 robustness."""
    gt_data = load_gt(task_info.gt_path)
    if not gt_data:
        return None

    # Load rot0 (baseline) from answers/test/
    rot0_data = load_answers_for_task(answers_dir, model, emode_base, task_info.rel_path)

    # Load rot1 and rot2 answers + meta
    rot_data: dict[int, Optional[dict]] = {}
    meta_data: dict[int, Optional[dict]] = {}
    for rot in _ROTATIONS:
        rot_data[rot] = _load_rot_answers(l7_dir, model, emode_base, rot, task_info.rel_path)
        meta_data[rot] = _load_meta(l7_dir, model, emode_base, rot, task_info.rel_path)

    # Need at least rot0 or one rotation to proceed
    has_rot0 = bool(rot0_data)
    has_any_rot = any(rot_data[r] for r in _ROTATIONS)
    if not has_rot0 and not has_any_rot:
        return None

    # Build per-image predictions in original option space
    # pred_orig[rot_id][img] = letter in original space (or None)
    pred_orig: dict[int, dict] = {0: {}, 1: {}, 2: {}}

    # rot0: direct extraction
    if rot0_data:
        for img, raw in rot0_data.items():
            pred_orig[0][img] = extract_letter(str(raw)) if raw else None

    # rot1, rot2: extract then remap
    for rot in _ROTATIONS:
        if not rot_data[rot] or not meta_data[rot]:
            continue
        for img, raw in rot_data[rot].items():
            pred_rotated = extract_letter(str(raw)) if raw else None
            meta = meta_data[rot].get(img, {})
            option_map = meta.get("option_map", {}) if isinstance(meta, dict) else {}
            if option_map:
                pred_orig[rot][img] = _remap_to_original(pred_rotated, option_map)
            else:
                # No option_map: treat as identity (no rotation applied)
                pred_orig[rot][img] = pred_rotated

    # Collect images that appear in at least rot0 + one rotation
    all_imgs = set(pred_orig[0].keys())
    for rot in _ROTATIONS:
        all_imgs |= set(pred_orig[rot].keys())

    if len(all_imgs) < 5:
        logger.debug("Skipping task %s (n=%d < 5)", task_info.subtask, len(all_imgs))
        return None

    # Consistency: images where all available rotations agree
    # Only count images that have predictions in ALL available rotations
    available_rots = [0] + [r for r in _ROTATIONS if pred_orig[r]]
    common_imgs = set(pred_orig[available_rots[0]].keys())
    for r in available_rots[1:]:
        common_imgs &= set(pred_orig[r].keys())

    n_common = len(common_imgs)
    consistency_count = 0
    if n_common > 0:
        for img in common_imgs:
            preds = [pred_orig[r].get(img) for r in available_rots]
            # All non-None and all equal
            non_null = [p for p in preds if p is not None]
            if len(non_null) == len(available_rots) and len(set(non_null)) == 1:
                consistency_count += 1

    consistency_rate = consistency_count / n_common if n_common > 0 else None
    position_bias_score = 1.0 - consistency_rate if consistency_rate is not None else None

    # Accuracy (F1) per rotation
    accuracy: dict[int, Optional[float]] = {}
    for rot in [0] + _ROTATIONS:
        if len(pred_orig[rot]) >= 5:
            accuracy[rot] = compute_accuracy(gt_data, pred_orig[rot], emode_base)
        else:
            accuracy[rot] = None

    # Accuracy drop: rot0 - mean(rot1, rot2)
    accuracy_drop = None
    rot_accs = [accuracy[r] for r in _ROTATIONS if accuracy[r] is not None]
    if accuracy[0] is not None and rot_accs:
        accuracy_drop = accuracy[0] - float(np.mean(rot_accs))

    return {
        "model": model,
        "emode": emode_base,
        "level": task_info.level,
        "task": task_info.task,
        "subtask": task_info.subtask,
        "n_common": n_common,
        "n_rot0": len(pred_orig[0]),
        "n_rot1": len(pred_orig[1]),
        "n_rot2": len(pred_orig[2]),
        "accuracy_rot0": accuracy[0],
        "accuracy_rot1": accuracy[1],
        "accuracy_rot2": accuracy[2],
        "consistency_rate": consistency_rate,
        "position_bias_score": position_bias_score,
        "accuracy_drop": accuracy_drop,
    }
