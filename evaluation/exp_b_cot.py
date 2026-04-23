"""
Experiment B: CoT vs Direct Answer Accuracy

Compares emode2 (direct) vs emode2_cot (chain-of-thought) predictions.
Also supports emode3 vs emode3_cot pairs.

Metrics:
  - accuracy_direct / accuracy_cot: mean F1 per task
  - cot_gain = accuracy_cot - accuracy_direct
  - cot_score: mean CoTScorer weighted score across samples
  - accuracy_cot_gap: fraction of correct samples with cot_score < 0.5
  - parse_failure_rate: fraction of samples where parse_status == 'failed'
  - 5 dimension scores: finding_coverage, conclusion_consistency,
    reasoning_order, medical_correctness, chain_completeness
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import numpy as np

# Allow imports from project root
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
from cot_evaluation.parse_cot import parse_chain, extract_answer
from cot_evaluation.registry import get_ruleset

logger = logging.getLogger(__name__)

# CoT emode suffix mapping
_COT_EMODE_MAP = {
    "emode2": "emode2_cot",
    "emode3": "emode3_cot",
}

# Dimension names (order matters for output columns)
DIMENSIONS = [
    "finding_coverage",
    "conclusion_consistency",
    "reasoning_order",
    "medical_correctness",
    "chain_completeness",
]


def _extract_cot_letter(raw: str) -> Optional[str]:
    """Extract answer letter from CoT output.

    Tries <answer> block first, then falls back to extract_letter().
    """
    if not raw:
        return None
    parsed = parse_chain(raw)
    chain = parsed["chain"]
    if chain.get("answer"):
        letter = extract_answer(chain["answer"])
        if letter:
            # extract_answer may return 'A,B' — take first
            return letter.split(",")[0].strip()
    return extract_letter(raw)


def _score_cot_sample(
    raw: str,
    gt_letter: str,
    task_info: TaskInfo,
) -> dict:
    """Parse and score a single CoT sample.

    Returns dict with parse_status, predicted_letter, is_correct,
    cot_score, and 5 dimension scores.
    """
    parsed = parse_chain(raw)
    chain = parsed["chain"]
    parse_status = parsed["parse_status"]

    # Extract predicted letter
    pred_letter = None
    if chain.get("answer"):
        pred_letter = extract_answer(chain["answer"])
        if pred_letter:
            pred_letter = pred_letter.split(",")[0].strip()
    if pred_letter is None:
        pred_letter = extract_letter(raw)

    is_correct = (pred_letter is not None and pred_letter == gt_letter)

    # Get ruleset and score
    ruleset = get_ruleset(
        level=task_info.level,
        task=task_info.task,
        subtask=task_info.subtask,
        gt_answer=gt_letter,
    )
    scores = ruleset.score(chain, pred_letter or "")

    return {
        "parse_status": parse_status,
        "pred_letter": pred_letter,
        "is_correct": is_correct,
        "cot_score": scores.get("cot_score") or 0.0,
        "finding_coverage": scores.get("finding_coverage") or 0.0,
        "conclusion_consistency": scores.get("conclusion_consistency") or 0.0,
        "reasoning_order": scores.get("reasoning_order") or 0.0,
        "medical_correctness": scores.get("medical_correctness") or 0.0,
        "chain_completeness": scores.get("chain_completeness") or 0.0,
    }


def run_exp_b(
    answers_dir: str,
    gt_dir: str,
    models: Optional[list[str]] = None,
    base_emodes: Optional[list[str]] = None,
    level_filter: Optional[list[str]] = None,
) -> list[dict]:
    """Run Experiment B: CoT vs Direct comparison.

    Args:
        answers_dir: Path to answers/test/
        gt_dir: Path to FunBench/
        models: List of model names to evaluate (None = auto-discover)
        base_emodes: Base emodes to compare, e.g. ['emode2', 'emode3']
                     (None = try both)
        level_filter: Level prefixes to include, e.g. ['L3', 'L4']

    Returns:
        List of result dicts, one per (model, emode, task).
    """
    from evaluation.common import discover_models

    if models is None:
        models = discover_models(answers_dir)
    if base_emodes is None:
        base_emodes = list(_COT_EMODE_MAP.keys())

    tasks = discover_tasks(gt_dir, level_filter=level_filter)
    if not tasks:
        logger.warning("No tasks found in %s", gt_dir)
        return []

    results = []

    for model in models:
        for base_emode in base_emodes:
            cot_emode = _COT_EMODE_MAP.get(base_emode)
            if cot_emode is None:
                continue

            # Check if CoT data exists for this model/emode
            cot_dir = os.path.join(answers_dir, model, cot_emode)
            if not os.path.isdir(cot_dir):
                logger.debug(
                    "No CoT data for model=%s emode=%s, skipping Exp B",
                    model, cot_emode,
                )
                continue

            logger.info("Exp B: model=%s base=%s cot=%s", model, base_emode, cot_emode)

            for task_info in tasks:
                row = _eval_task_pair(
                    answers_dir=answers_dir,
                    model=model,
                    base_emode=base_emode,
                    cot_emode=cot_emode,
                    task_info=task_info,
                )
                if row is not None:
                    results.append(row)

    return results


def _eval_task_pair(
    answers_dir: str,
    model: str,
    base_emode: str,
    cot_emode: str,
    task_info: TaskInfo,
) -> Optional[dict]:
    """Evaluate one (model, emode_pair, task) combination.

    Returns None if insufficient data.
    """
    gt_data = load_gt(task_info.gt_path)
    if not gt_data:
        return None

    direct_data = load_answers_for_task(answers_dir, model, base_emode, task_info.rel_path)
    cot_data = load_answers_for_task(answers_dir, model, cot_emode, task_info.rel_path)

    if not direct_data and not cot_data:
        return None

    # Collect per-sample records
    cot_records = []
    direct_correct = []
    direct_pred = {}
    cot_pred = {}

    for img, gt_entry in gt_data.items():
        gt_letter = gt_entry.get("gt", "")
        if not gt_letter:
            continue
        raw_options = gt_entry.get("raw_data", {}).get("options", [])
        if not raw_options:
            continue

        # Direct prediction
        if direct_data and img in direct_data:
            raw_direct = direct_data[img]
            pred_d = extract_letter(str(raw_direct)) if raw_direct else None
            direct_pred[img] = pred_d
            direct_correct.append(pred_d == gt_letter if pred_d else False)

        # CoT prediction + scoring
        if cot_data and img in cot_data:
            raw_cot = cot_data[img]
            rec = _score_cot_sample(str(raw_cot) if raw_cot else "", gt_letter, task_info)
            rec["img"] = img
            rec["gt_letter"] = gt_letter
            cot_pred[img] = rec["pred_letter"]
            cot_records.append(rec)

    # Need at least 5 samples in at least one mode
    n_direct = len(direct_pred)
    n_cot = len(cot_records)
    n_samples = max(n_direct, n_cot)

    if n_samples < 5:
        logger.debug(
            "Skipping task %s (n=%d < 5)", task_info.subtask, n_samples
        )
        return None

    # Compute direct accuracy (F1)
    accuracy_direct = None
    if n_direct >= 5:
        accuracy_direct = compute_accuracy(gt_data, direct_data or {}, base_emode)

    # Compute CoT accuracy (F1) using cot_pred letters
    accuracy_cot = None
    if n_cot >= 5:
        accuracy_cot = compute_accuracy(gt_data, cot_pred, cot_emode)

    # CoT aggregate metrics
    cot_score = None
    accuracy_cot_gap = None
    parse_failure_rate = None
    dim_means: dict[str, Optional[float]] = {d: None for d in DIMENSIONS}

    if cot_records:
        cot_scores = [r["cot_score"] for r in cot_records]
        cot_score = float(np.mean(cot_scores))

        # Accuracy-CoT Gap: among correct samples, fraction with cot_score < 0.5
        correct_records = [r for r in cot_records if r["is_correct"]]
        if correct_records:
            gap_count = sum(1 for r in correct_records if r["cot_score"] < 0.5)
            accuracy_cot_gap = gap_count / len(correct_records)
        else:
            accuracy_cot_gap = 0.0

        # Parse failure rate
        failed = sum(1 for r in cot_records if r["parse_status"] == "failed")
        parse_failure_rate = failed / len(cot_records)

        # Dimension means
        for dim in DIMENSIONS:
            vals = [r[dim] for r in cot_records]
            dim_means[dim] = float(np.mean(vals))

    # CoT gain
    cot_gain = None
    if accuracy_direct is not None and accuracy_cot is not None:
        cot_gain = accuracy_cot - accuracy_direct

    return {
        "model": model,
        "emode": base_emode,
        "level": task_info.level,
        "task": task_info.task,
        "subtask": task_info.subtask,
        "n_direct": n_direct,
        "n_cot": n_cot,
        "accuracy_direct": accuracy_direct,
        "accuracy_cot": accuracy_cot,
        "cot_gain": cot_gain,
        "cot_score": cot_score,
        "accuracy_cot_gap": accuracy_cot_gap,
        "parse_failure_rate": parse_failure_rate,
        **{f"dim_{d}": dim_means[d] for d in DIMENSIONS},
    }
