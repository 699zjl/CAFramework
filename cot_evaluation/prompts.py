"""
CoT prompt templates for FunBench-CoT evaluation.

Strategy: wrap the original FunBench prompt with a reasoning structure
preamble, keeping the original question and options intact.

Two modes:
  E-mode2: original prompt includes clinical description
  E-mode3: image-only, no description

The CoT template instructs the model to fill 5 XML-tagged fields before
giving the final answer, enabling structured parsing downstream.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CoT preamble injected before the original FunBench prompt
# ---------------------------------------------------------------------------

_COT_PREAMBLE = """\
You are an expert ophthalmologist analyzing a fundus image.
Before answering, reason step by step using the structure below.

<observations>
List specific visual features you observe in the image (colors, shapes, locations, sizes).
</observations>

<findings>
Identify any lesions, anatomical structures, or abnormalities based on your observations.
</findings>

<reasoning>
Apply medical knowledge to interpret your findings. Reference relevant clinical criteria if applicable.
</reasoning>

<conclusion>
State your final interpretation in one sentence.
</conclusion>

<answer>
"""

_COT_SUFFIX = "</answer>"


def build_cot_prompt(original_prompt: str) -> str:
    """Wrap an original FunBench prompt with CoT reasoning structure.

    The original prompt already contains the question, options, and
    (for E-mode2) the clinical description. We prepend the CoT preamble
    and append the closing </answer> tag.

    Args:
        original_prompt: The original E-mode2 or E-mode3 prompt string
                         from the FunBench task JSON.

    Returns:
        CoT-wrapped prompt string ready for model inference.
    """
    # Replace the original instruction line with our CoT preamble
    # Original starts with "Please choose the most suitable option..."
    # We keep everything from "Description:" or "The question is:" onward
    lines = original_prompt.strip().split("\n")

    # Find where the actual content starts (after the instruction line)
    content_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Description:") or stripped.startswith("The question is:"):
            content_start = i
            break
        # If no such line found, keep all lines
        if i == len(lines) - 1:
            content_start = 0

    content_lines = lines[content_start:]
    content = "\n".join(content_lines).strip()

    return _COT_PREAMBLE + content + "\n" + _COT_SUFFIX


def get_cot_prompts(text_data_item: dict) -> dict:
    """Build CoT prompts for both E-mode2 and E-mode3 from a task data item.

    Args:
        text_data_item: Single image entry from FunBench task JSON, e.g.:
            {
                'E-mode2': '...original prompt...',
                'E-mode3': '...original prompt...',
                'gt': 'A',
                'raw_data': {...}
            }

    Returns:
        Dict with keys 'emode2_cot' and 'emode3_cot' (None if original is None).
    """
    emode2_orig = text_data_item.get("E-mode2")
    emode3_orig = text_data_item.get("E-mode3")

    return {
        "emode2_cot": build_cot_prompt(emode2_orig) if emode2_orig else None,
        "emode3_cot": build_cot_prompt(emode3_orig) if emode3_orig else None,
    }
