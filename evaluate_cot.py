"""
FunBench-CoT: Evaluation script.

Scores CoT reasoning chains using the medical rule library and computes:
  - Standard accuracy metrics (F1/Sensitivity/Specificity) via evaluate.py
  - CoT-Score (5-dimension weighted score)
  - Accuracy-CoT Gap (correct answer but poor reasoning)
  - parse_failure_rate

Output:
  experiments/<model>_<timestamp>/results_cot.xlsx
  experiments/<model>_<timestamp>/summary.json

Usage:
    .venv3.10.20/bin/python evaluate_cot.py \
        --pred_root answers/test/Qwen2.5-VL-7B-Instruct/emode3_cot \
        --emode emode3 \
        --model_name Qwen2.5-VL-7B-Instruct
"""

import os
import json
import argparse
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm

import numpy as np
from openpyxl import Workbook

from evaluate import SentenceSimilarity, label_level_ss_f1, fast_hist, letter_idx
from cot_evaluation.parse_cot import parse_chain, extract_answer
from cot_evaluation.registry import get_ruleset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task_id(level: str, task: str, subtask: str | None) -> str:
    return f"{level}/{task}/{subtask}" if subtask else f"{level}/{task}"


def _is_correct(pred_letters: list[str], gt_letters: list[str]) -> bool:
    return set(pred_letters) == set(gt_letters)


# ---------------------------------------------------------------------------
# Per-task scoring
# ---------------------------------------------------------------------------

def score_task(
    gt_data: dict,
    pred_data: dict,
    level: str,
    task: str,
    subtask: str | None,
    similarity_processor: SentenceSimilarity,
) -> dict:
    """Score a single task's CoT predictions.

    Returns a dict with per-sample records and aggregate metrics.
    """
    samples = []
    cot_scores = []
    correct_flags = []
    parse_statuses = []

    for img_name in gt_data:
        if img_name not in pred_data:
            continue

        gt_entry = gt_data[img_name]
        gt_letter = gt_entry["gt"]
        options_raw = gt_entry["raw_data"]["options"]
        gt_raw = gt_entry["raw_data"]["gt"]

        multi_label = isinstance(gt_raw, list)
        gt_letters = gt_letter.split(",") if multi_label else [gt_letter]

        bin_mapping = set(options_raw) == {"Yes", "No"}
        if bin_mapping:
            if task == "L3a-lesion_recognition":
                bin_infos = subtask.split("-")[-1][:-5] if subtask else "lesion"
            else:
                bin_infos = "abnormality"
        else:
            bin_infos = None

        raw_output = pred_data[img_name]

        # Parse CoT chain
        parsed = parse_chain(raw_output)
        chain = parsed["chain"]
        parse_status = parsed["parse_status"]

        # Extract predicted answer from chain
        predicted_answer = extract_answer(chain["answer"]) if chain["answer"] else None

        # If parse failed, fall back to full raw output for answer extraction
        answer_text = chain["answer"] if chain["answer"] else raw_output

        # Use SentenceSimilarity to map to option letter (same as evaluate.py)
        pred_letters = similarity_processor.process(
            answers=answer_text,
            options=options_raw,
            multiple_answers=multi_label,
            bin_mapping=bin_mapping,
            bin_infos=bin_infos,
        )

        correct = _is_correct(pred_letters, gt_letters)

        # Get RuleSet and score the chain
        ruleset = get_ruleset(
            level=level,
            task=task,
            subtask=subtask,
            gt_answer=gt_letter,
        )
        scores = ruleset.score(chain, predicted_answer or "")
        cot_score = scores.get("cot_score")

        sample = {
            "img_name": img_name,
            "gt_answer": gt_letter,
            "predicted_answer": predicted_answer,
            "pred_letters": pred_letters,
            "correct": correct,
            "parse_status": parse_status,
            "scores": scores,
            "cot_score": cot_score,
        }
        samples.append(sample)

        if cot_score is not None:
            cot_scores.append(cot_score)
        correct_flags.append(correct)
        parse_statuses.append(parse_status)

    n = len(samples)
    if n == 0:
        return {"samples": [], "metrics": None}

    n_correct = sum(correct_flags)
    accuracy = n_correct / n

    valid_cot = [s for s in cot_scores]
    mean_cot_score = float(np.mean(valid_cot)) if valid_cot else None

    # Accuracy-CoT Gap: correct answer but CoT-Score < 0.5
    gap_samples = [
        s for s in samples
        if s["correct"] and s["cot_score"] is not None and s["cot_score"] < 0.5
    ]
    accuracy_cot_gap = len(gap_samples) / n_correct if n_correct > 0 else 0.0

    parse_failure_rate = parse_statuses.count("failed") / n

    # Per-dimension means
    dim_means = {}
    for dim in ["finding_coverage", "reasoning_order", "conclusion_consistency",
                "medical_correctness", "chain_completeness"]:
        vals = [s["scores"].get(dim) for s in samples if s["scores"].get(dim) is not None]
        dim_means[dim] = float(np.mean(vals)) if vals else None

    metrics = {
        "n_samples": n,
        "accuracy": round(accuracy, 4),
        "mean_cot_score": round(mean_cot_score, 4) if mean_cot_score is not None else None,
        "accuracy_cot_gap": round(accuracy_cot_gap, 4),
        "parse_failure_rate": round(parse_failure_rate, 4),
        **{f"mean_{k}": round(v, 4) if v is not None else None for k, v in dim_means.items()},
    }

    return {"samples": samples, "metrics": metrics}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    gt_root: str,
    pred_root: str,
    emode: str,
    model_name: str,
    out_dir: str,
) -> None:

    os.makedirs(out_dir, exist_ok=True)
    similarity_processor = SentenceSimilarity()

    # Discover tasks
    task_infos = []
    for level in sorted(os.listdir(gt_root)):
        if level.startswith(".") or level == "README.md":
            continue
        for task in sorted(os.listdir(os.path.join(gt_root, level))):
            if task.startswith("."):
                continue
            task_dir = os.path.join(gt_root, level, task)
            if os.path.isdir(task_dir):
                for subtask in sorted(os.listdir(task_dir)):
                    if subtask.startswith("."):
                        continue
                    task_infos.append((level, task, subtask))
            else:
                task_infos.append((level, task, None))

    all_metrics: dict = {}
    all_samples: dict = {}

    for count, (level, task, subtask) in enumerate(task_infos):
        task_name = subtask if subtask else task
        print(f"{count + 1}/{len(task_infos)} evaluating: {task_name}")

        # GT path
        if subtask is None:
            gt_path = os.path.join(gt_root, level, task)
            pred_path = os.path.join(pred_root, level, task)
        else:
            gt_path = os.path.join(gt_root, level, task, subtask)
            pred_path = os.path.join(pred_root, level, task, subtask)

        if not os.path.exists(pred_path):
            print(f"  [SKIP] no predictions: {pred_path}")
            continue

        with open(gt_path) as f:
            gt_data = json.load(f)["data"]
        with open(pred_path) as f:
            pred_data = json.load(f)

        result = score_task(
            gt_data=gt_data,
            pred_data=pred_data,
            level=level,
            task=task,
            subtask=subtask,
            similarity_processor=similarity_processor,
        )

        tid = _task_id(level, task, subtask)
        all_metrics[tid] = result["metrics"]
        all_samples[tid] = result["samples"]

        if result["metrics"]:
            m = result["metrics"]
            print(
                f"  acc={m['accuracy']:.3f} | cot={m['mean_cot_score']} | "
                f"gap={m['accuracy_cot_gap']:.3f} | parse_fail={m['parse_failure_rate']:.3f}"
            )

    # Save summary.json
    summary = {
        "model": model_name,
        "emode": emode,
        "timestamp": datetime.now().isoformat(),
        "tasks": all_metrics,
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSummary saved: {summary_path}")

    # Save results_cot.xlsx
    xlsx_path = os.path.join(out_dir, "results_cot.xlsx")
    _save_xlsx(all_metrics, xlsx_path)
    print(f"Results saved: {xlsx_path}")


def _save_xlsx(all_metrics: dict, out_path: str) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "CoT Metrics"

    headers = [
        "Task", "N", "Accuracy", "CoT-Score", "Accuracy-CoT Gap",
        "Parse Fail Rate", "Finding Coverage", "Reasoning Order",
        "Conclusion Consistency", "Medical Correctness", "Chain Completeness",
    ]
    ws.append(headers)

    for tid, metrics in all_metrics.items():
        if metrics is None:
            continue
        row = [
            tid,
            metrics.get("n_samples"),
            metrics.get("accuracy"),
            metrics.get("mean_cot_score"),
            metrics.get("accuracy_cot_gap"),
            metrics.get("parse_failure_rate"),
            metrics.get("mean_finding_coverage"),
            metrics.get("mean_reasoning_order"),
            metrics.get("mean_conclusion_consistency"),
            metrics.get("mean_medical_correctness"),
            metrics.get("mean_chain_completeness"),
        ]
        ws.append(row)

    wb.save(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FunBench-CoT evaluation")
    parser.add_argument(
        "--gt_root", default="FunBench", help="FunBench ground truth root"
    )
    parser.add_argument(
        "--pred_root",
        default="answers/test/Qwen2.5-VL-7B-Instruct/emode3_cot",
        help="CoT prediction root (emode3_cot or emode2_cot directory)",
    )
    parser.add_argument(
        "--emode", default="emode3", choices=["emode2", "emode3"],
        help="Evaluation mode being scored"
    )
    parser.add_argument(
        "--model_name", default="Qwen2.5-VL-7B-Instruct", help="Model name for logging"
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory. Defaults to experiments/<model>_<timestamp>/",
    )
    args = parser.parse_args()

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = os.path.join("experiments", f"{args.model_name}_{ts}")

    main(
        gt_root=args.gt_root,
        pred_root=args.pred_root,
        emode=args.emode,
        model_name=args.model_name,
        out_dir=args.out_dir,
    )
