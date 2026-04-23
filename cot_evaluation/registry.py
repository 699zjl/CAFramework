"""
Registry: maps FunBench task filenames to RuleSets.

Usage:
    from cot_evaluation.registry import get_ruleset

    ruleset = get_ruleset(level="L3-lesion_analysis",
                          task="L3a-lesion_recognition",
                          subtask="L3a5-lesion_recognition-Microaneurysm.json",
                          gt_answer="A")
    scores = ruleset.score(chain, predicted_answer)
"""

from __future__ import annotations
from .base import RuleSet, NullRuleSet
from .l3_rules import get_l3_ruleset
from .l4_rules import (
    BINARY_RULESET,
    MULTI_CONDITION_RULESET,
    get_dr_ruleset,
    get_amd_ruleset,
)

# Tasks that have no applicable medical reasoning rules
_NULL_TASKS = {
    "L1a-coarse_modality_perception",
    "L1b-fine_modality_perception",
    "L2a-od_fovea_positioning",
}

_NULL_RULESET = NullRuleSet()


def _extract_lesion_name(filename: str) -> str:
    """Extract lesion name from subtask filename.

    e.g. 'L3a5-lesion_recognition-Microaneurysm.json' -> 'Microaneurysm'
         'L3b2-lesion_localization-Hard_exudate.json'  -> 'Hard_exudate'
    """
    name = filename.replace(".json", "")
    parts = name.split("-")
    # Last part is the lesion name
    if len(parts) >= 3:
        return parts[-1]
    return name


def get_ruleset(
    level: str,
    task: str,
    subtask: str | None,
    gt_answer: str = "",
) -> RuleSet:
    """Return the appropriate RuleSet for a given FunBench task.

    Args:
        level:    Level directory name, e.g. 'L3-lesion_analysis'
        task:     Task directory or file name, e.g. 'L3a-lesion_recognition'
        subtask:  Subtask filename (with or without .json), e.g.
                  'L3a5-lesion_recognition-Microaneurysm.json', or None
        gt_answer: Ground truth answer letter (used for grade-specific rules)

    Returns:
        Appropriate RuleSet, or NullRuleSet for non-applicable tasks.
    """
    # Normalize: strip .json suffix for matching
    task_key = task.replace(".json", "")
    subtask_key = subtask.replace(".json", "") if subtask else None

    # --- L1: no rules ---
    if level.startswith("L1") or task_key in _NULL_TASKS:
        return _NULL_RULESET

    # --- L2a: no rules (spatial positioning, hard to rule-check) ---
    if "L2a" in task_key or (subtask_key and "L2a" in subtask_key):
        return _NULL_RULESET

    # --- L2b: laterality (basic anatomy rules, use generic RuleSet) ---
    if "L2b" in task_key or (subtask_key and "L2b" in subtask_key):
        from .base import KeywordRule, RuleSet as RS
        from .keywords import ANATOMY_KEYWORDS
        kw_rule = KeywordRule(
            fields=["findings", "reasoning"],
            required=(
                ANATOMY_KEYWORDS["left_eye"]["required"]
                + ANATOMY_KEYWORDS["right_eye"]["required"]
                + ANATOMY_KEYWORDS["optic_disc"]["required"]
            ),
            supporting=(
                ANATOMY_KEYWORDS["left_eye"]["supporting"]
                + ANATOMY_KEYWORDS["right_eye"]["supporting"]
            ),
        )
        return RS(keyword_rule=kw_rule)

    # --- L3: lesion analysis ---
    if level.startswith("L3") or task_key.startswith("L3"):
        if subtask_key:
            lesion_name = _extract_lesion_name(subtask_key)
        else:
            lesion_name = _extract_lesion_name(task_key)
        return get_l3_ruleset(lesion_name)

    # --- L4: disease diagnosis ---
    if level.startswith("L4") or task_key.startswith("L4"):
        fname = subtask_key or task_key

        if "L4a" in fname:
            return BINARY_RULESET

        if "L4b" in fname:
            return MULTI_CONDITION_RULESET

        if "L4c" in fname:
            return get_dr_ruleset(gt_answer)

        if "L4d" in fname:
            return get_amd_ruleset(gt_answer)

    # Fallback: return NullRuleSet for unknown tasks
    return _NULL_RULESET
