"""
XML-based CoT reasoning chain parser.

Extracts 5 structured fields from model output:
  <observations>, <findings>, <reasoning>, <conclusion>, <answer>

parse_status:
  complete  - all 5 fields present and non-empty
  partial   - 1-4 fields present
  failed    - no fields found (model ignored format)
"""

from __future__ import annotations
import re
from typing import Optional

# Canonical field names matching the CoT prompt template
CHAIN_FIELDS = ["observations", "findings", "reasoning", "conclusion", "answer"]

# Regex to extract content between XML tags (non-greedy, DOTALL for multiline)
_TAG_PATTERN = re.compile(
    r"<({field})>(.*?)<\/\1>",
    re.DOTALL | re.IGNORECASE,
)

# Regex to extract a single answer letter from the answer block
_ANSWER_LETTER_PATTERN = re.compile(r"\b([A-Z])\b")


def parse_chain(raw_output: str) -> dict:
    """Parse raw model output into a structured reasoning chain.

    Args:
        raw_output: Raw string output from the MLLM.

    Returns:
        Dict with keys:
          chain: dict with 5 field texts (empty string if missing)
          parse_status: 'complete' | 'partial' | 'failed'
    """
    chain: dict[str, str] = {f: "" for f in CHAIN_FIELDS}
    found_count = 0

    for field in CHAIN_FIELDS:
        pattern = re.compile(
            r"<" + re.escape(field) + r">(.*?)<\/" + re.escape(field) + r">",
            re.DOTALL | re.IGNORECASE,
        )
        match = pattern.search(raw_output)
        if match:
            content = match.group(1).strip()
            if content:
                chain[field] = content
                found_count += 1

    if found_count == len(CHAIN_FIELDS):
        parse_status = "complete"
    elif found_count > 0:
        parse_status = "partial"
    else:
        parse_status = "failed"

    return {"chain": chain, "parse_status": parse_status}


def extract_answer(answer_block: str) -> Optional[str]:
    """Extract the answer letter(s) from the answer field text.

    Handles formats like:
      'A', 'A.', 'A,B', 'The answer is A', 'A and C'

    Args:
        answer_block: Text content of the <answer> field.

    Returns:
        Uppercase letter string (e.g. 'A', 'A,C') or None if not found.
    """
    if not answer_block:
        return None

    text = answer_block.strip()

    # Try direct single letter match first
    if len(text) == 1 and text.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return text.upper()

    # Try letter followed by punctuation: 'A.' or 'A)'
    m = re.match(r"^([A-Za-z])[.\):]", text)
    if m:
        return m.group(1).upper()

    # Extract all capital letters that look like answer options
    letters = _ANSWER_LETTER_PATTERN.findall(text.upper())
    # Filter to plausible answer letters (A-F range for typical MCQ)
    letters = [l for l in letters if l in "ABCDEF"]
    if letters:
        return ",".join(sorted(set(letters)))

    return None


def build_sample_record(
    img_name: str,
    task: str,
    emode: str,
    gt_answer: str,
    raw_output: str,
) -> dict:
    """Build a complete per-sample record with parsed chain and placeholder scores.

    Args:
        img_name: Image file path (relative).
        task: Task identifier, e.g. 'L4c-dr_grading'.
        emode: Evaluation mode, e.g. 'emode3'.
        gt_answer: Ground truth answer letter.
        raw_output: Raw model output string.

    Returns:
        Complete sample record dict ready for scoring.
    """
    parsed = parse_chain(raw_output)
    chain = parsed["chain"]
    parse_status = parsed["parse_status"]

    predicted_answer = extract_answer(chain["answer"]) if chain["answer"] else None

    return {
        "img_name": img_name,
        "task": task,
        "emode": emode,
        "gt_answer": gt_answer,
        "predicted_answer": predicted_answer,
        "raw_output": raw_output,
        "chain": chain,
        "parse_status": parse_status,
        "scores": {
            "finding_coverage": 0.0,
            "reasoning_order": 0.0,
            "conclusion_consistency": 0.0,
            "medical_correctness": 0.0,
            "chain_completeness": 0.0,
        },
        "cot_score": 0.0,
    }
