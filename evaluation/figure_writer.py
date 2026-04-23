"""
Figure writer: generates 5 matplotlib figures from experiment results.

fig1_position_bias.png       — Exp A: position bias score by model × emode
fig2_cot_gain_by_level.png   — Exp B: CoT gain by task level × model
fig3_accuracy_cot_scatter.png — Exp B: accuracy_direct vs cot_score scatter
fig4_description_dependency.png — Exp C: description gain by model × level
fig5_l5_contradiction.png    — Exp D: contradiction rate by model × emode
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    logger.warning("matplotlib not installed — figure output disabled")

# Color palette (colorblind-friendly)
_COLORS = ["#4472C4", "#ED7D31", "#A9D18E", "#FF0000", "#7030A0",
           "#00B0F0", "#FFC000", "#70AD47", "#C00000", "#002060"]


def _model_colors(models: list[str]) -> dict[str, str]:
    return {m: _COLORS[i % len(_COLORS)] for i, m in enumerate(sorted(set(models)))}


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s", path)


# ---------------------------------------------------------------------------
# Fig 1: Position Bias Score (Exp A)
# ---------------------------------------------------------------------------

def fig1_position_bias(exp_a: list[dict], out_path: str) -> bool:
    if not _HAS_MPL:
        return False
    if not exp_a:
        logger.warning("No Exp A data — skipping fig1")
        return False

    # Aggregate: mean position_bias_score per (model, emode)
    agg: dict[tuple, list] = defaultdict(list)
    for r in exp_a:
        if r.get("position_bias_score") is not None:
            agg[(r["model"], r["emode"])].append(r["position_bias_score"])

    if not agg:
        logger.warning("No position_bias_score values — skipping fig1")
        return False

    models = sorted(set(k[0] for k in agg))
    emodes = sorted(set(k[1] for k in agg))
    x = np.arange(len(models))
    width = 0.8 / max(len(emodes), 1)

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.5), 5))
    for i, emode in enumerate(emodes):
        vals = [np.mean(agg.get((m, emode), [np.nan])) for m in models]
        offset = (i - len(emodes) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      label=emode, color=_COLORS[i % len(_COLORS)])
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Position Bias Score (1 - Consistency Rate)")
    ax.set_title("Fig 1: Option-Order Position Bias by Model (Exp A)")
    ax.set_ylim(0, min(1.1, ax.get_ylim()[1] * 1.15))
    ax.legend(title="E-mode", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)
    return True


# ---------------------------------------------------------------------------
# Fig 2: CoT Gain by Task Level (Exp B)
# ---------------------------------------------------------------------------

def fig2_cot_gain_by_level(exp_b: list[dict], out_path: str) -> bool:
    if not _HAS_MPL:
        return False
    if not exp_b:
        logger.warning("No Exp B data — skipping fig2")
        return False

    # Extract level prefix (L1, L2, L3, L4) from level field
    def _level_prefix(level: str) -> str:
        for prefix in ["L1", "L2", "L3", "L4"]:
            if level.startswith(prefix):
                return prefix
        return level[:2]

    agg: dict[tuple, list] = defaultdict(list)
    for r in exp_b:
        if r.get("cot_gain") is not None:
            prefix = _level_prefix(r.get("level", ""))
            agg[(r["model"], prefix)].append(r["cot_gain"])

    if not agg:
        logger.warning("No cot_gain values — skipping fig2")
        return False

    models = sorted(set(k[0] for k in agg))
    levels = sorted(set(k[1] for k in agg))
    x = np.arange(len(levels))
    width = 0.8 / max(len(models), 1)
    colors = _model_colors(models)

    fig, ax = plt.subplots(figsize=(max(6, len(levels) * 2), 5))
    for i, model in enumerate(models):
        vals = [np.mean(agg.get((model, lv), [np.nan])) for lv in levels]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      label=model, color=colors[model])
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.002 if v >= 0 else -0.012),
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=10)
    ax.set_ylabel("CoT Gain (F1_cot - F1_direct)")
    ax.set_title("Fig 2: CoT Gain by Task Level (Exp B)")
    ax.legend(title="Model", fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)
    return True


# ---------------------------------------------------------------------------
# Fig 3: Accuracy vs CoT Score scatter (Exp B)
# ---------------------------------------------------------------------------

def fig3_accuracy_cot_scatter(exp_b: list[dict], out_path: str) -> bool:
    if not _HAS_MPL:
        return False
    if not exp_b:
        logger.warning("No Exp B data — skipping fig3")
        return False

    models = sorted(set(r["model"] for r in exp_b))
    colors = _model_colors(models)

    fig, ax = plt.subplots(figsize=(7, 5))
    for model in models:
        xs = [r["accuracy_direct"] for r in exp_b
              if r["model"] == model
              and r.get("accuracy_direct") is not None
              and r.get("cot_score") is not None]
        ys = [r["cot_score"] for r in exp_b
              if r["model"] == model
              and r.get("accuracy_direct") is not None
              and r.get("cot_score") is not None]
        if xs:
            ax.scatter(xs, ys, label=model, color=colors[model], alpha=0.6, s=30)

    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", label="CoT threshold=0.5")
    ax.set_xlabel("Direct Answer Accuracy (F1)")
    ax.set_ylabel("CoT Score")
    ax.set_title("Fig 3: Direct Accuracy vs CoT Reasoning Quality (Exp B)")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)
    return True


# ---------------------------------------------------------------------------
# Fig 4: Description Dependency (Exp C)
# ---------------------------------------------------------------------------

def fig4_description_dependency(exp_c: list[dict], out_path: str) -> bool:
    if not _HAS_MPL:
        return False
    if not exp_c:
        logger.warning("No Exp C data — skipping fig4")
        return False

    def _level_prefix(level: str) -> str:
        for prefix in ["L1", "L2", "L3", "L4"]:
            if level.startswith(prefix):
                return prefix
        return level[:2]

    agg: dict[tuple, list] = defaultdict(list)
    for r in exp_c:
        if r.get("description_gain") is not None:
            prefix = _level_prefix(r.get("level", ""))
            agg[(r["model"], prefix)].append(r["description_gain"])

    if not agg:
        logger.warning("No description_gain values — skipping fig4")
        return False

    models = sorted(set(k[0] for k in agg))
    levels = sorted(set(k[1] for k in agg))
    x = np.arange(len(levels))
    width = 0.8 / max(len(models), 1)
    colors = _model_colors(models)

    fig, ax = plt.subplots(figsize=(max(6, len(levels) * 2), 5))
    for i, model in enumerate(models):
        vals = [np.mean(agg.get((model, lv), [np.nan])) for lv in levels]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      label=model, color=colors[model])
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.002 if v >= 0 else -0.012),
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=10)
    ax.set_ylabel("Description Gain (F1_emode2 - F1_emode3)")
    ax.set_title("Fig 4: Description Dependency by Task Level (Exp C)")
    ax.legend(title="Model", fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)
    return True


# ---------------------------------------------------------------------------
# Fig 5: L5 Contradiction Rate (Exp D)
# ---------------------------------------------------------------------------

def fig5_l5_contradiction(exp_d: list[dict], out_path: str) -> bool:
    if not _HAS_MPL:
        return False
    if not exp_d:
        logger.warning("No Exp D data — skipping fig5")
        return False

    models = sorted(set(r["model"] for r in exp_d))
    emodes = sorted(set(r["emode"] for r in exp_d))
    x = np.arange(len(models))
    width = 0.8 / max(len(emodes), 1)

    # Also plot per-rule breakdown as stacked or grouped
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overall contradiction rate
    ax = axes[0]
    for i, emode in enumerate(emodes):
        vals = []
        for model in models:
            matching = [r for r in exp_d if r["model"] == model and r["emode"] == emode]
            vals.append(matching[0]["contradiction_rate"] if matching else np.nan)
        offset = (i - len(emodes) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      label=emode, color=_COLORS[i % len(_COLORS)])
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Contradiction Rate")
    ax.set_title("Overall Contradiction Rate")
    ax.set_ylim(0, 1.05)
    ax.legend(title="E-mode", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Right: per-rule rates (emode2 only, or first available)
    ax2 = axes[1]
    rules = ["R1", "R2", "R3", "R4"]
    rule_labels = ["R1\n(出血+无DR)", "R2\n(微动脉瘤+无DR)",
                   "R3\n(新生血管+非增殖)", "R4\n(硬渗+无DR)"]
    colors = _model_colors(models)
    x2 = np.arange(len(rules))
    width2 = 0.8 / max(len(models), 1)

    # Use first emode for per-rule breakdown
    first_emode = emodes[0] if emodes else None
    for i, model in enumerate(models):
        matching = [r for r in exp_d
                    if r["model"] == model and r["emode"] == (first_emode or r["emode"])]
        if not matching:
            continue
        row = matching[0]
        vals2 = [row.get(f"rule_{r}_rate") or 0.0 for r in rules]
        offset2 = (i - len(models) / 2 + 0.5) * width2
        ax2.bar(x2 + offset2, vals2, width2 * 0.9, label=model, color=colors[model])

    ax2.set_xticks(x2)
    ax2.set_xticklabels(rule_labels, fontsize=8)
    ax2.set_ylabel("Per-Rule Contradiction Rate")
    ax2.set_title(f"Per-Rule Breakdown ({first_emode})")
    ax2.set_ylim(0, 1.05)
    ax2.legend(title="Model", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Fig 5: L5 Cross-Task Logical Consistency (Exp D)", fontsize=12)
    fig.tight_layout()
    _save(fig, out_path)
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_figures(
    out_dir: str,
    exp_a: list[dict],
    exp_b: list[dict],
    exp_c: list[dict],
    exp_d: list[dict],
) -> list[str]:
    """Generate all 5 figures. Returns list of successfully written paths."""
    if not _HAS_MPL:
        logger.error("matplotlib not installed — cannot write figures")
        return []

    os.makedirs(out_dir, exist_ok=True)
    written = []

    figs = [
        (fig1_position_bias,        exp_a, "fig1_position_bias.png"),
        (fig2_cot_gain_by_level,    exp_b, "fig2_cot_gain_by_level.png"),
        (fig3_accuracy_cot_scatter, exp_b, "fig3_accuracy_cot_scatter.png"),
        (fig4_description_dependency, exp_c, "fig4_description_dependency.png"),
        (fig5_l5_contradiction,     exp_d, "fig5_l5_contradiction.png"),
    ]

    for fn, data, fname in figs:
        path = os.path.join(out_dir, fname)
        try:
            ok = fn(data, path)
            if ok:
                written.append(path)
        except Exception as e:
            logger.warning("Failed to generate %s: %s", fname, e)

    return written
