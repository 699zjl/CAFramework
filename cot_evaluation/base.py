"""
Rule base classes for CoT reasoning chain scoring.

Five scoring dimensions:
  - finding_coverage    (weight 0.35): keyword presence in findings+reasoning
  - reasoning_order     (weight 0.20): field completeness and logical order
  - conclusion_consistency (weight 0.25): conclusion does not contradict answer
  - medical_correctness (weight 0.15): no incorrect negations
  - chain_completeness  (weight 0.05): fraction of non-empty fields
"""

from __future__ import annotations
import re
from typing import Optional

# Canonical field order for a complete reasoning chain
CHAIN_FIELDS = ["observations", "findings", "reasoning", "conclusion", "answer"]

# Default scoring weights
SCORE_WEIGHTS = {
    "finding_coverage": 0.35,
    "reasoning_order": 0.20,
    "conclusion_consistency": 0.25,
    "medical_correctness": 0.15,
    "chain_completeness": 0.05,
}


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace for robust matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


# ---------------------------------------------------------------------------
# Individual Rule Classes
# ---------------------------------------------------------------------------

class KeywordRule:
    """Score based on keyword presence in specified chain fields.

    Args:
        fields: Chain fields to search (e.g. ['findings', 'reasoning']).
        required: At least one required keyword must appear for non-zero score.
        supporting: Additional keywords that increase score.
    """

    def __init__(self, fields: list[str], required: list[str], supporting: list[str]) -> None:
        self.fields = fields
        self.required = [_normalize(k) for k in required]
        self.supporting = [_normalize(k) for k in supporting]

    def score(self, chain: dict) -> float:
        text = " ".join(_normalize(chain.get(f, "") or "") for f in self.fields)

        # Must hit at least one required keyword
        required_hit = any(kw in text for kw in self.required)
        if not required_hit:
            return 0.0

        # Base score from required hit
        base = 0.5

        # Bonus from supporting keywords (up to 0.5 additional)
        if self.supporting:
            support_hits = sum(1 for kw in self.supporting if kw in text)
            bonus = min(0.5, 0.5 * support_hits / len(self.supporting))
        else:
            bonus = 0.5

        return base + bonus


class OrderRule:
    """Score based on field completeness and logical order.

    All CHAIN_FIELDS must be non-empty and appear in the correct sequence.
    Deducts 0.2 per missing or empty field.
    """

    def score(self, chain: dict) -> float:
        score = 1.0
        for field in CHAIN_FIELDS:
            val = chain.get(field, "")
            if not val or not val.strip():
                score -= 0.2
        return max(0.0, score)


class ConsistencyRule:
    """Detect contradiction between conclusion text and answer letter.

    Args:
        contradiction_map: Dict mapping answer letter -> list of forbidden
            keywords that should NOT appear in conclusion if that answer is given.
            Example: {'A': ['neovascularization', 'proliferative']}
    """

    def __init__(self, contradiction_map: dict[str, list[str]]) -> None:
        self.contradiction_map = {
            k: [_normalize(w) for w in v]
            for k, v in contradiction_map.items()
        }

    def score(self, chain: dict, answer: str) -> float:
        conclusion = _normalize(chain.get("conclusion", "") or "")
        answer_letter = (answer or "").strip().upper()

        forbidden = self.contradiction_map.get(answer_letter, [])
        for kw in forbidden:
            if kw in conclusion:
                return 0.0
        return 1.0


class NegationRule:
    """Detect medically incorrect negations.

    Checks whether the chain contains a negation of an expected finding
    while the answer indicates that finding is present.

    Args:
        fields: Chain fields to search.
        negation_patterns: List of (negation_regex, answer_letters_where_wrong).
            Example: (r'no microaneurysm', ['A']) means if chain says
            'no microaneurysm' but answer is 'A' (present), score = 0.
    """

    def __init__(
        self,
        fields: list[str],
        negation_patterns: list[tuple[str, list[str]]],
    ) -> None:
        self.fields = fields
        self.negation_patterns = [
            (re.compile(pat, re.IGNORECASE), letters)
            for pat, letters in negation_patterns
        ]

    def score(self, chain: dict, answer: str) -> float:
        text = " ".join(chain.get(f, "") or "" for f in self.fields)
        answer_letter = (answer or "").strip().upper()

        for pattern, wrong_letters in self.negation_patterns:
            if pattern.search(text) and answer_letter in wrong_letters:
                return 0.0
        return 1.0


# ---------------------------------------------------------------------------
# RuleSet: composes rules into a single scorer
# ---------------------------------------------------------------------------

class RuleSet:
    """Composes multiple rules into a single per-sample scorer.

    Args:
        keyword_rule: KeywordRule for finding_coverage dimension.
        consistency_rule: ConsistencyRule for conclusion_consistency dimension.
        negation_rule: NegationRule for medical_correctness dimension.
    """

    def __init__(
        self,
        keyword_rule: Optional[KeywordRule] = None,
        consistency_rule: Optional[ConsistencyRule] = None,
        negation_rule: Optional[NegationRule] = None,
    ) -> None:
        self.keyword_rule = keyword_rule
        self.consistency_rule = consistency_rule
        self.negation_rule = negation_rule
        self._order_rule = OrderRule()

    def score(self, chain: dict, answer: str) -> dict:
        """Compute all 5 scoring dimensions.

        Returns:
            Dict with keys matching SCORE_WEIGHTS plus 'cot_score'.
        """
        finding_coverage = self.keyword_rule.score(chain) if self.keyword_rule else 0.0
        reasoning_order = self._order_rule.score(chain)
        conclusion_consistency = (
            self.consistency_rule.score(chain, answer) if self.consistency_rule else 1.0
        )
        medical_correctness = (
            self.negation_rule.score(chain, answer) if self.negation_rule else 1.0
        )
        non_empty = sum(1 for f in CHAIN_FIELDS if chain.get(f, "").strip())
        chain_completeness = non_empty / len(CHAIN_FIELDS)

        scores = {
            "finding_coverage": round(finding_coverage, 4),
            "reasoning_order": round(reasoning_order, 4),
            "conclusion_consistency": round(conclusion_consistency, 4),
            "medical_correctness": round(medical_correctness, 4),
            "chain_completeness": round(chain_completeness, 4),
        }
        cot_score = sum(scores[k] * SCORE_WEIGHTS[k] for k in SCORE_WEIGHTS)
        scores["cot_score"] = round(cot_score, 4)
        return scores


class NullRuleSet(RuleSet):
    """Returns None for all scores. Used for tasks without applicable rules (L1, L2a)."""

    def score(self, chain: dict, answer: str) -> dict:
        return {k: None for k in list(SCORE_WEIGHTS.keys()) + ["cot_score"]}
