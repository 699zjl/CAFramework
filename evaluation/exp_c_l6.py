"""
Experiment C: L6 Description Dependency

Compares emode2 (with text description) vs emode3 (image only) predictions
on the same images to measure how much models rely on textual descriptions.

Metrics:
  description_gain    = F1_emode2 - F1_emode3
    > 0: description helps (language knowledge compensates for visual weakness)
    < 0: description hurts (text bias overrides visual signal)

  visual_consistency  = fraction of images where emode2 prediction == emode3
  flip_to_correct     = fraction where emode3 wrong but emode2 correct
                        (description rescued the answer)
  flip_to_wrong       = fraction where emode2 wrong but emode3 correct
                        (description misled the model)
"""

from __future__ import annotations

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
    load_answers_for_task,
    load_gt,
)

logger = logging.getLogger(__name__)

# Pairs to compare: (with_description, image_only)
_EMODE_PAIRS = [
    ("emode2", "emode3"),
]


def run_exp_c(
    answers_dir: str,
    gt_dir: str,
    models: Optional[list[str]] = None,
    level_filter: Optional[list[str]] = None,
) -> list[dict]:
    """Run Experiment C: description dependency (emode2 vs emode3).

    Args:
        answers_dir: Path to answers/test/
        gt_dir: Path to FunBench/
        models: Model names to evaluate (None = auto-discover)
        level_filter: Level prefixes to include

    Returns:
        List of result dicts, one per (model, task).
    """
    from evaluation.common import discover_models

    if models is None:
        models = discover_models(answers_dir)

    tasks = discover_tasks(gt_dir, level_filter=level_filter)
    if not tasks:
        logger.warning("No tasks found in %s", gt_dir)
        return []

    results = []

    for model in models:
        logger.info("Exp C: model=%s", model)
        for emode2, emode3 in _EMODE_PAIRS:
            for task_info in tasks:
                row = _eval_task(
                    answers_dir=answers_dir,
                    model=model,
                    emode_desc=emode2,
                    emode_img=emode3,
                    task_info=task_info,
                )
                if row is not None:
                    results.append(row)

    return results


def _eval_task(
    answers_dir: str,
    model: str,
    emode_desc: str,
    emode_img: str,
    task_info: TaskInfo,
) -> Optional[dict]:
    """Evaluate one (model, task) for description dependency."""
    gt_data = load_gt(task_info.gt_path)
    if not gt_data:
        return None

    data_desc = load_answers_for_task(answers_dir, model, emode_desc, task_info.rel_path)
    data_img = load_answers_for_task(answers_dir, model, emode_img, task_info.rel_path)

    if not data_desc and not data_img:
        return None

    # Build per-image prediction dicts
    pred_desc: dict[str, Optional[str]] = {}
    pred_img: dict[str, Optional[str]] = {}

    for img, gt_entry in gt_data.items():
        gt_letter = gt_entry.get("gt", "")
        if not gt_letter:
            continue
        raw_options = gt_entry.get("raw_data", {}).get("options", [])
        if not raw_options:
            continue

        if data_desc and img in data_desc:
            pred_desc[img] = extract_letter(str(data_desc[img])) if data_desc[img] else None
        if data_img and img in data_img:
            pred_img[img] = extract_letter(str(data_img[img])) if data_img[img] else None

    n_desc = len(pred_desc)
    n_img = len(pred_img)
    n_samples = max(n_desc, n_img)

    if n_samples < 5:
        logger.debug("Skipping task %s (n=%d < 5)", task_info.subtask, n_samples)
        return None

    # Accuracy (F1) for each mode
    accuracy_desc = compute_accuracy(gt_data, data_desc or {}, emode_desc) if n_desc >= 5 else None
    accuracy_img = compute_accuracy(gt_data, data_img or {}, emode_img) if n_img >= 5 else None

    description_gain = None
    if accuracy_desc is not None and accuracy_img is not None:
        description_gain = accuracy_desc - accuracy_img

    # Per-image consistency metrics (only on images present in both modes)
    common_imgs = set(pred_desc.keys()) & set(pred_img.keys())
    n_common = len(common_imgs)

    visual_consistency = None
    flip_to_correct = None
    flip_to_wrong = None

    if n_common > 0:
        agree_count = 0
        flip_correct_count = 0
        flip_wrong_count = 0

        for img in common_imgs:
            p2 = pred_desc.get(img)
            p3 = pred_img.get(img)
            gt_entry = gt_data.get(img, {})
            gt_letter = gt_entry.get("gt", "") if isinstance(gt_entry, dict) else ""

            # Visual consistency: same prediction regardless of mode
            if p2 is not None and p3 is not None and p2 == p3:
                agree_count += 1

            # Flip analysis
            correct2 = (p2 == gt_letter) if p2 else False
            correct3 = (p3 == gt_letter) if p3 else False

            # flip_to_correct: emode3 wrong, emode2 correct
            # (description rescued the answer)
            if not correct3 and correct2:
                flip_correct_count += 1

            # flip_to_wrong: emode2 wrong, emode3 correct
            # (description misled the model)
            if not correct2 and correct3:
                flip_wrong_count += 1

        visual_consistency = agree_count / n_common
        flip_to_correct = flip_correct_count / n_common
        flip_to_wrong = flip_wrong_count / n_common

    return {
        "model": model,
        "emode_desc": emode_desc,
        "emode_img": emode_img,
        "level": task_info.level,
        "task": task_info.task,
        "subtask": task_info.subtask,
        "n_desc": n_desc,
        "n_img": n_img,
        "n_common": n_common,
        "accuracy_emode2": accuracy_desc,
        "accuracy_emode3": accuracy_img,
        "description_gain": description_gain,
        "visual_consistency": visual_consistency,
        "flip_to_correct": flip_to_correct,
        "flip_to_wrong": flip_to_wrong,
    }
