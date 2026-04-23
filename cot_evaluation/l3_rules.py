"""
L3 lesion analysis rule sets.

Covers all 12 lesion types for tasks:
  L3a - lesion recognition (binary: present/absent)
  L3b - lesion localization
  L3c - lesion size estimation
  L3d - lesion counting
"""

from __future__ import annotations
from .base import KeywordRule, ConsistencyRule, NegationRule, RuleSet
from .keywords import LESION_KEYWORDS


def _make_lesion_ruleset(lesion_name: str) -> RuleSet:
    """Build a RuleSet for a given lesion type.

    Args:
        lesion_name: Key in LESION_KEYWORDS dict.
    """
    kw = LESION_KEYWORDS[lesion_name]
    required = kw["required"]
    supporting = kw["supporting"]

    keyword_rule = KeywordRule(
        fields=["findings", "reasoning"],
        required=required,
        supporting=supporting,
    )

    # For binary recognition tasks: answer A = present, answer B = absent
    # If model says "no <lesion>" in findings but answers A (present) → contradiction
    negation_patterns = [
        (r"no\s+" + r"|no\s+".join(re.escape(r) for r in required), ["A"]),
        (r"absent|not\s+present|not\s+found|not\s+detected", ["A"]),
    ]

    negation_rule = NegationRule(
        fields=["findings", "reasoning"],
        negation_patterns=negation_patterns,
    )

    # Conclusion consistency: if answer is B (absent), conclusion should not
    # contain the lesion's required keywords
    contradiction_map = {
        "B": required,  # answer B = absent, but conclusion mentions lesion → contradiction
    }
    consistency_rule = ConsistencyRule(contradiction_map=contradiction_map)

    return RuleSet(
        keyword_rule=keyword_rule,
        consistency_rule=consistency_rule,
        negation_rule=negation_rule,
    )


import re

# Pre-build RuleSets for all 12 lesion types
LESION_RULESETS: dict[str, RuleSet] = {
    lesion: _make_lesion_ruleset(lesion)
    for lesion in LESION_KEYWORDS
}


def get_l3_ruleset(lesion_name: str) -> RuleSet:
    """Get the RuleSet for a specific lesion type.

    Args:
        lesion_name: Lesion name as it appears in FunBench task filenames.
            e.g. 'Microaneurysm', 'Hard_exudate', 'Haemorrhage'

    Returns:
        RuleSet for the lesion, or a default RuleSet if not found.
    """
    # Normalize: try direct match, then case-insensitive
    if lesion_name in LESION_RULESETS:
        return LESION_RULESETS[lesion_name]

    for key in LESION_RULESETS:
        if key.lower() == lesion_name.lower():
            return LESION_RULESETS[key]

    # Fallback: generic lesion ruleset with no specific keywords
    return RuleSet()
