"""
Experiment D: L5 Cross-Task Logical Consistency

Detects medical contradictions between L3a lesion recognition predictions
and L4c DR grading predictions on the same images.

Medical contradiction rules (based on ICDR/ETDRS standards):
  R1: L3a1-Haemorrhage      Yes (A) + L4c No DR (D)
      → Haemorrhage is a DR diagnostic criterion; No DR cannot have haemorrhage
  R2: L3a5-Microaneurysm    Yes (A) + L4c No DR (D)
      → Microaneurysm is the earliest DR lesion; No DR cannot have it
  R3: L3a6-Neovascularization Yes (A) + L4c != Proliferative DR (A)
      → Neovascularization defines Proliferative DR
  R4: L3a2-Hard_exudate     Yes (A) + L4c No DR (D)
      → Hard exudate is a typical DR lesion

L3a option mapping (from GT): A=Yes, B=No
L4c option mapping (from GT): A=Proliferative DR, B=Moderate NPDR,
                               C=Mild NPDR, D=No DR, E=Severe NPDR

Metrics:
  contradiction_rate  = fraction of images with at least one contradiction
  consistency_score   = 1 - contradiction_rate
  rule_R{1-4}_rate    = per-rule contradiction rate
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.common import (
    discover_tasks,
    extract_letter,
    load_answers_for_task,
    load_gt,
)

logger = logging.getLogger(__name__)

# L3a subtask filenames (without .json) that participate in rules
_L3A_SUBTASKS = {
    "R1": "L3a1-lesion_recognition-Haemorrhage",
    "R2": "L3a5-lesion_recognition-Microaneurysm",
    "R3": "L3a6-lesion_recognition-Neovascularization",
    "R4": "L3a2-lesion_recognition-Hard_exudate",
}

# L4c task relative path (flat file, not nested)
_L4C_REL = "L4-disease_diagnosis/L4c-dr_grading.json"

# L3a relative path template
_L3A_REL_TMPL = "L3-lesion_analysis/L3a-lesion_recognition/{subtask}.json"

# L3a: A=Yes (lesion present), B=No
_L3A_YES = "A"

# L4c: D=No DR, A=Proliferative DR
_L4C_NO_DR = "D"
_L4C_PROLIFERATIVE = "A"

# Rule definitions: (rule_id, l3a_subtask_key, l3a_pred_trigger, l4c_pred_trigger)
# l4c_pred_trigger is either a single letter (exact match) or a set of letters
# that constitute a contradiction when l3a says "Yes"
_RULES = [
    ("R1", "R1", _L3A_YES, {_L4C_NO_DR}),           # Haemorrhage + No DR
    ("R2", "R2", _L3A_YES, {_L4C_NO_DR}),           # Microaneurysm + No DR
    ("R3", "R3", _L3A_YES, None),                    # Neovascularization + not Proliferative
    ("R4", "R4", _L3A_YES, {_L4C_NO_DR}),           # Hard exudate + No DR
]


def _is_contradiction(rule_id: str, l3a_pred: Optional[str], l4c_pred: Optional[str]) -> bool:
    """Return True if the (l3a_pred, l4c_pred) pair violates the rule."""
    if l3a_pred is None or l4c_pred is None:
        return False

    if rule_id == "R3":
        # Neovascularization present but NOT Proliferative DR
        return l3a_pred == _L3A_YES and l4c_pred != _L4C_PROLIFERATIVE
    else:
        # Lesion present but No DR
        return l3a_pred == _L3A_YES and l4c_pred == _L4C_NO_DR


def run_exp_d(
    answers_dir: str,
    gt_dir: str,
    models: Optional[list[str]] = None,
    emodes: Optional[list[str]] = None,
) -> list[dict]:
    """Run Experiment D: L5 cross-task logical consistency.

    Args:
        answers_dir: Path to answers/test/
        gt_dir: Path to FunBench/
        models: Model names to evaluate (None = auto-discover)
        emodes: Emodes to evaluate, e.g. ['emode2', 'emode3'] (None = auto)

    Returns:
        List of result dicts, one per (model, emode).
    """
    from evaluation.common import discover_models, get_emodes_for_model

    if models is None:
        models = discover_models(answers_dir)

    results = []

    for model in models:
        model_emodes = emodes
        if model_emodes is None:
            model_emodes = get_emodes_for_model(answers_dir, model)
            # Exclude CoT emodes for this experiment
            model_emodes = [e for e in model_emodes if "cot" not in e.lower()]

        for emode in model_emodes:
            logger.info("Exp D: model=%s emode=%s", model, emode)
            row = _eval_model_emode(
                answers_dir=answers_dir,
                gt_dir=gt_dir,
                model=model,
                emode=emode,
            )
            if row is not None:
                results.append(row)

    return results


def _eval_model_emode(
    answers_dir: str,
    gt_dir: str,
    model: str,
    emode: str,
) -> Optional[dict]:
    """Evaluate cross-task consistency for one (model, emode)."""
    # Load L4c predictions and GT
    l4c_gt_path = os.path.join(gt_dir, _L4C_REL)
    l4c_gt = load_gt(l4c_gt_path)
    l4c_pred_data = load_answers_for_task(answers_dir, model, emode, _L4C_REL)

    if not l4c_gt or not l4c_pred_data:
        logger.debug("No L4c data for model=%s emode=%s", model, emode)
        return None

    # Build L4c predictions: img -> letter
    l4c_pred: dict[str, Optional[str]] = {}
    for img, raw in l4c_pred_data.items():
        l4c_pred[img] = extract_letter(str(raw)) if raw else None

    # Load L3a predictions for each rule
    l3a_preds: dict[str, dict[str, Optional[str]]] = {}  # rule_key -> {img: letter}
    for rule_id, subtask_key in _L3A_SUBTASKS.items():
        rel = _L3A_REL_TMPL.format(subtask=subtask_key)
        pred_data = load_answers_for_task(answers_dir, model, emode, rel)
        if pred_data:
            l3a_preds[rule_id] = {
                img: extract_letter(str(raw)) if raw else None
                for img, raw in pred_data.items()
            }
        else:
            l3a_preds[rule_id] = {}

    # Find images that have both L4c and at least one L3a prediction
    all_imgs_with_l4c = set(l4c_pred.keys())
    if not all_imgs_with_l4c:
        return None

    # Per-rule contradiction counts
    rule_counts: dict[str, int] = {r: 0 for r in _L3A_SUBTASKS}
    rule_totals: dict[str, int] = {r: 0 for r in _L3A_SUBTASKS}

    # Overall contradiction tracking
    img_has_contradiction: dict[str, bool] = {}

    for rule_id in _L3A_SUBTASKS:
        rule_imgs = all_imgs_with_l4c & set(l3a_preds[rule_id].keys())
        rule_totals[rule_id] = len(rule_imgs)

        for img in rule_imgs:
            l3a_p = l3a_preds[rule_id].get(img)
            l4c_p = l4c_pred.get(img)
            contra = _is_contradiction(rule_id, l3a_p, l4c_p)
            if contra:
                rule_counts[rule_id] += 1
                img_has_contradiction[img] = True
            elif img not in img_has_contradiction:
                img_has_contradiction[img] = False

    n_images = len(img_has_contradiction)
    if n_images < 5:
        logger.debug("Skipping model=%s emode=%s (n=%d < 5)", model, emode, n_images)
        return None

    n_contradictions = sum(1 for v in img_has_contradiction.values() if v)
    contradiction_rate = n_contradictions / n_images
    consistency_score = 1.0 - contradiction_rate

    # Per-rule rates
    rule_rates: dict[str, Optional[float]] = {}
    for rule_id in _L3A_SUBTASKS:
        if rule_totals[rule_id] > 0:
            rule_rates[rule_id] = rule_counts[rule_id] / rule_totals[rule_id]
        else:
            rule_rates[rule_id] = None

    return {
        "model": model,
        "emode": emode,
        "n_images": n_images,
        "n_contradictions": n_contradictions,
        "contradiction_rate": contradiction_rate,
        "consistency_score": consistency_score,
        "rule_R1_rate": rule_rates.get("R1"),
        "rule_R2_rate": rule_rates.get("R2"),
        "rule_R3_rate": rule_rates.get("R3"),
        "rule_R4_rate": rule_rates.get("R4"),
        "rule_R1_n": rule_totals.get("R1", 0),
        "rule_R2_n": rule_totals.get("R2", 0),
        "rule_R3_n": rule_totals.get("R3", 0),
        "rule_R4_n": rule_totals.get("R4", 0),
    }
