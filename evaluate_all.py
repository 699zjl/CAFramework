#!/usr/bin/env python3
"""
evaluate_all.py — FunBench-Consistency comprehensive evaluation entry point.

Usage:
    .venv3.10.20/bin/python3 evaluate_all.py \\
        --answers_dir answers/test \\
        --l7_dir answers_l7 \\
        --gt_dir FunBench \\
        --out_dir cot_evaluation \\
        --experiments A,B,C,D

Outputs:
    <out_dir>/results_<timestamp>.xlsx
    <out_dir>/figures/fig{1-5}_*.png

Graceful degradation:
    Missing data → skip with warning, never crash.
    Tasks with < 5 samples → skipped silently.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_all")

# Ensure project root is on path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _run_exp_a(args) -> list[dict]:
    if not os.path.isdir(args.l7_dir):
        logger.warning("Exp A skipped: l7_dir not found: %s", args.l7_dir)
        return []
    from evaluation.exp_a_l7 import run_exp_a
    logger.info("=== Running Experiment A: L7 Option-Order Robustness ===")
    results = run_exp_a(
        l7_dir=args.l7_dir,
        answers_dir=args.answers_dir,
        gt_dir=args.gt_dir,
    )
    logger.info("Exp A: %d task-level results", len(results))
    return results


def _run_exp_b(args) -> list[dict]:
    from evaluation.exp_b_cot import run_exp_b
    logger.info("=== Running Experiment B: CoT vs Direct ===")
    results = run_exp_b(
        answers_dir=args.answers_dir,
        gt_dir=args.gt_dir,
    )
    logger.info("Exp B: %d task-level results", len(results))
    return results


def _run_exp_c(args) -> list[dict]:
    from evaluation.exp_c_l6 import run_exp_c
    logger.info("=== Running Experiment C: Description Dependency ===")
    results = run_exp_c(
        answers_dir=args.answers_dir,
        gt_dir=args.gt_dir,
    )
    logger.info("Exp C: %d task-level results", len(results))
    return results


def _run_exp_d(args) -> list[dict]:
    from evaluation.exp_d_l5 import run_exp_d
    logger.info("=== Running Experiment D: Cross-Task Consistency ===")
    results = run_exp_d(
        answers_dir=args.answers_dir,
        gt_dir=args.gt_dir,
    )
    logger.info("Exp D: %d model-level results", len(results))
    return results


def _print_summary(exp_a, exp_b, exp_c, exp_d):
    import numpy as np

    def _mean_metric(data, key):
        vals = [r[key] for r in data if r.get(key) is not None]
        return f"{np.mean(vals):.4f}" if vals else "N/A"

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if exp_a:
        print(f"\n[Exp A] L7 Option-Order Robustness ({len(exp_a)} tasks)")
        print(f"  Mean consistency_rate:    {_mean_metric(exp_a, 'consistency_rate')}")
        print(f"  Mean position_bias_score: {_mean_metric(exp_a, 'position_bias_score')}")
        print(f"  Mean accuracy_drop:       {_mean_metric(exp_a, 'accuracy_drop')}")

    if exp_b:
        print(f"\n[Exp B] CoT vs Direct ({len(exp_b)} tasks)")
        print(f"  Mean accuracy_direct:  {_mean_metric(exp_b, 'accuracy_direct')}")
        print(f"  Mean accuracy_cot:     {_mean_metric(exp_b, 'accuracy_cot')}")
        print(f"  Mean cot_gain:         {_mean_metric(exp_b, 'cot_gain')}")
        print(f"  Mean cot_score:        {_mean_metric(exp_b, 'cot_score')}")
        print(f"  Mean accuracy_cot_gap: {_mean_metric(exp_b, 'accuracy_cot_gap')}")

    if exp_c:
        print(f"\n[Exp C] Description Dependency ({len(exp_c)} tasks)")
        print(f"  Mean accuracy_emode2:   {_mean_metric(exp_c, 'accuracy_emode2')}")
        print(f"  Mean accuracy_emode3:   {_mean_metric(exp_c, 'accuracy_emode3')}")
        print(f"  Mean description_gain:  {_mean_metric(exp_c, 'description_gain')}")
        print(f"  Mean visual_consistency:{_mean_metric(exp_c, 'visual_consistency')}")

    if exp_d:
        print(f"\n[Exp D] Cross-Task Consistency ({len(exp_d)} model-emodes)")
        print(f"  Mean contradiction_rate: {_mean_metric(exp_d, 'contradiction_rate')}")
        print(f"  Mean consistency_score:  {_mean_metric(exp_d, 'consistency_score')}")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="FunBench-Consistency comprehensive evaluation"
    )
    parser.add_argument(
        "--answers_dir", default="answers/test",
        help="Path to answers/test/ directory (default: answers/test)"
    )
    parser.add_argument(
        "--l7_dir", default="answers_l7",
        help="Path to answers_l7/ directory (default: answers_l7)"
    )
    parser.add_argument(
        "--gt_dir", default="FunBench",
        help="Path to FunBench GT directory (default: FunBench)"
    )
    parser.add_argument(
        "--out_dir", default="cot_evaluation",
        help="Output directory for Excel and figures (default: cot_evaluation)"
    )
    parser.add_argument(
        "--experiments", default="A,B,C,D",
        help="Comma-separated experiments to run: A,B,C,D (default: A,B,C,D)"
    )
    parser.add_argument(
        "--models", default=None,
        help="Comma-separated model names to evaluate (default: auto-discover)"
    )
    parser.add_argument(
        "--no_excel", action="store_true",
        help="Skip Excel output"
    )
    parser.add_argument(
        "--no_figures", action="store_true",
        help="Skip figure output"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse experiment list
    exps = {e.strip().upper() for e in args.experiments.split(",")}

    # Parse model list
    if args.models:
        args._models = [m.strip() for m in args.models.split(",")]
    else:
        args._models = None

    # Resolve paths relative to script location
    for attr in ("answers_dir", "l7_dir", "gt_dir", "out_dir"):
        val = getattr(args, attr)
        if not os.path.isabs(val):
            setattr(args, attr, os.path.join(_ROOT, val))

    logger.info("answers_dir: %s", args.answers_dir)
    logger.info("l7_dir:      %s", args.l7_dir)
    logger.info("gt_dir:      %s", args.gt_dir)
    logger.info("out_dir:     %s", args.out_dir)
    logger.info("experiments: %s", sorted(exps))

    # Run experiments
    exp_a: list[dict] = []
    exp_b: list[dict] = []
    exp_c: list[dict] = []
    exp_d: list[dict] = []

    if "A" in exps:
        try:
            exp_a = _run_exp_a(args)
        except Exception as e:
            logger.error("Exp A failed: %s", e, exc_info=True)

    if "B" in exps:
        try:
            exp_b = _run_exp_b(args)
        except Exception as e:
            logger.error("Exp B failed: %s", e, exc_info=True)

    if "C" in exps:
        try:
            exp_c = _run_exp_c(args)
        except Exception as e:
            logger.error("Exp C failed: %s", e, exc_info=True)

    if "D" in exps:
        try:
            exp_d = _run_exp_d(args)
        except Exception as e:
            logger.error("Exp D failed: %s", e, exc_info=True)

    # Print console summary
    _print_summary(exp_a, exp_b, exp_c, exp_d)

    # Write Excel
    if not args.no_excel:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        xlsx_path = os.path.join(args.out_dir, f"results_{timestamp}.xlsx")
        try:
            from evaluation.excel_writer import write_excel
            ok = write_excel(xlsx_path, exp_a, exp_b, exp_c, exp_d)
            if ok:
                print(f"Excel: {xlsx_path}")
        except Exception as e:
            logger.error("Excel write failed: %s", e, exc_info=True)

    # Write figures
    if not args.no_figures:
        fig_dir = os.path.join(args.out_dir, "figures")
        try:
            from evaluation.figure_writer import write_figures
            written = write_figures(fig_dir, exp_a, exp_b, exp_c, exp_d)
            for p in written:
                print(f"Figure: {p}")
        except Exception as e:
            logger.error("Figure write failed: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
