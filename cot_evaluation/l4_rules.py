"""
L4 disease diagnosis rule sets.

Covers:
  L4a - binary condition diagnosis (abnormality present/absent)
  L4b - multi-condition diagnosis
  L4c - DR grading (0-4, ICDR standard)
  L4d - AMD categorization
"""

from __future__ import annotations
from .base import KeywordRule, ConsistencyRule, NegationRule, RuleSet, NullRuleSet
from .keywords import DR_GRADE_KEYWORDS, AMD_KEYWORDS, BINARY_DIAGNOSIS_KEYWORDS, MULTI_CONDITION_KEYWORDS


# ---------------------------------------------------------------------------
# L4a: Binary condition diagnosis
# ---------------------------------------------------------------------------

def _make_binary_ruleset() -> RuleSet:
    """RuleSet for binary abnormality detection (Yes/No or A/B)."""
    keyword_rule = KeywordRule(
        fields=["findings", "reasoning"],
        required=(
            BINARY_DIAGNOSIS_KEYWORDS["abnormality_present"]["required"]
            + BINARY_DIAGNOSIS_KEYWORDS["no_abnormality"]["required"]
        ),
        supporting=(
            BINARY_DIAGNOSIS_KEYWORDS["abnormality_present"]["supporting"]
            + BINARY_DIAGNOSIS_KEYWORDS["no_abnormality"]["supporting"]
        ),
    )

    # answer A = abnormality present, answer B = no abnormality
    contradiction_map = {
        "A": BINARY_DIAGNOSIS_KEYWORDS["no_abnormality"]["required"],
        "B": BINARY_DIAGNOSIS_KEYWORDS["abnormality_present"]["required"],
    }
    consistency_rule = ConsistencyRule(contradiction_map=contradiction_map)

    negation_rule = NegationRule(
        fields=["findings", "reasoning"],
        negation_patterns=[
            (r"no\s+abnormality|normal|healthy|no\s+lesion", ["A"]),
            (r"abnormality|lesion|disease|pathology", ["B"]),
        ],
    )

    return RuleSet(
        keyword_rule=keyword_rule,
        consistency_rule=consistency_rule,
        negation_rule=negation_rule,
    )


BINARY_RULESET = _make_binary_ruleset()


# ---------------------------------------------------------------------------
# L4b: Multi-condition diagnosis
# ---------------------------------------------------------------------------

def _make_multi_condition_ruleset() -> RuleSet:
    """RuleSet for multi-condition diagnosis tasks."""
    all_required = []
    all_supporting = []
    for cond in MULTI_CONDITION_KEYWORDS.values():
        all_required.extend(cond["required"])
        all_supporting.extend(cond["supporting"])

    keyword_rule = KeywordRule(
        fields=["findings", "reasoning"],
        required=all_required,
        supporting=all_supporting,
    )
    # Multi-condition: consistency check is less strict (multiple answers possible)
    return RuleSet(keyword_rule=keyword_rule)


MULTI_CONDITION_RULESET = _make_multi_condition_ruleset()


# ---------------------------------------------------------------------------
# L4c: DR Grading (ICDR standard, grades 0-4)
# ---------------------------------------------------------------------------

def _make_dr_ruleset(grade: int) -> RuleSet:
    """Build RuleSet for a specific DR grade.

    Args:
        grade: DR grade 0-4.
    """
    kw = DR_GRADE_KEYWORDS[grade]

    keyword_rule = KeywordRule(
        fields=["findings", "reasoning"],
        required=kw["required"],
        supporting=kw.get("supporting", []),
    )

    # Map answer letters to DR grades: A=0, B=1, C=2, D=3, E=4
    # If answer is grade X, conclusion should not contain forbidden keywords for grade X
    grade_letter = chr(ord("A") + grade)
    contradiction_map = {grade_letter: kw.get("forbidden", [])}
    consistency_rule = ConsistencyRule(contradiction_map=contradiction_map)

    # Negation: if answer says grade >= 1 but chain says "no dr" → wrong
    negation_patterns = []
    if grade >= 1:
        negation_patterns.append((r"no\s+dr|no\s+diabetic\s+retinopathy|normal", [grade_letter]))
    if grade == 0:
        negation_patterns.append((r"neovascularization|proliferative|hemorrhage", [grade_letter]))

    negation_rule = NegationRule(
        fields=["findings", "reasoning"],
        negation_patterns=negation_patterns,
    ) if negation_patterns else None

    return RuleSet(
        keyword_rule=keyword_rule,
        consistency_rule=consistency_rule,
        negation_rule=negation_rule,
    )


DR_RULESETS: dict[int, RuleSet] = {
    grade: _make_dr_ruleset(grade) for grade in range(5)
}

# Grade-agnostic DR ruleset: used when we don't know the GT grade in advance
# Uses all DR keywords combined — for scoring the reasoning quality regardless of grade
def _make_dr_generic_ruleset() -> RuleSet:
    all_required = []
    all_supporting = []
    for grade_kw in DR_GRADE_KEYWORDS.values():
        all_required.extend(grade_kw["required"])
        all_supporting.extend(grade_kw.get("supporting", []))

    keyword_rule = KeywordRule(
        fields=["findings", "reasoning"],
        required=all_required,
        supporting=all_supporting,
    )
    return RuleSet(keyword_rule=keyword_rule)


DR_GENERIC_RULESET = _make_dr_generic_ruleset()


def get_dr_ruleset(gt_answer: str) -> RuleSet:
    """Get DR RuleSet matched to the ground truth grade.

    Args:
        gt_answer: Ground truth answer letter ('A'-'E').

    Returns:
        Grade-specific RuleSet, or generic RuleSet if grade unknown.
    """
    grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    grade = grade_map.get((gt_answer or "").strip().upper())
    if grade is not None:
        return DR_RULESETS[grade]
    return DR_GENERIC_RULESET


# ---------------------------------------------------------------------------
# L4d: AMD Categorization
# ---------------------------------------------------------------------------

def _make_amd_ruleset(category: str) -> RuleSet:
    """Build RuleSet for a specific AMD category."""
    kw = AMD_KEYWORDS[category]

    keyword_rule = KeywordRule(
        fields=["findings", "reasoning"],
        required=kw["required"],
        supporting=kw.get("supporting", []),
    )

    # Map category to answer letter: No_AMD=A, Early=B, Intermediate=C, Late=D
    category_order = ["No_AMD", "Early_AMD", "Intermediate_AMD", "Late_AMD"]
    if category in category_order:
        letter = chr(ord("A") + category_order.index(category))
        contradiction_map = {letter: kw.get("forbidden", [])}
        consistency_rule = ConsistencyRule(contradiction_map=contradiction_map)
    else:
        consistency_rule = None

    return RuleSet(
        keyword_rule=keyword_rule,
        consistency_rule=consistency_rule,
    )


AMD_RULESETS: dict[str, RuleSet] = {
    cat: _make_amd_ruleset(cat) for cat in AMD_KEYWORDS
}

AMD_CATEGORY_ORDER = ["No_AMD", "Early_AMD", "Intermediate_AMD", "Late_AMD"]


def get_amd_ruleset(gt_answer: str) -> RuleSet:
    """Get AMD RuleSet matched to the ground truth category.

    Args:
        gt_answer: Ground truth answer letter ('A'-'D').

    Returns:
        Category-specific RuleSet, or generic RuleSet if unknown.
    """
    idx = ord((gt_answer or "A").strip().upper()) - ord("A")
    if 0 <= idx < len(AMD_CATEGORY_ORDER):
        return AMD_RULESETS[AMD_CATEGORY_ORDER[idx]]
    return RuleSet()
