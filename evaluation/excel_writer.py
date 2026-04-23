"""
Excel writer: produces a multi-sheet workbook from experiment results.

Sheets:
  1. 总览        — core metrics summary across all models × experiments
  2. 实验A_L7鲁棒性  — full Exp A data
  3. 实验B_CoT      — full Exp B data
  4. 实验C_L6描述依赖 — full Exp C data
  5. 实验D_L5一致性  — full Exp D data
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    _HAS_OPENPYXL = True
except ImportError:
    _HAS_OPENPYXL = False
    logger.warning("openpyxl not installed — Excel output disabled")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_HEADER_FILL = None
_HEADER_FONT = None

def _init_styles():
    global _HEADER_FILL, _HEADER_FONT
    if _HAS_OPENPYXL and _HEADER_FILL is None:
        _HEADER_FILL = PatternFill("solid", fgColor="4472C4")
        _HEADER_FONT = Font(bold=True, color="FFFFFF")


def _write_header(ws, headers: list[str], row: int = 1):
    _init_styles()
    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=row, column=col, value=h)
        if _HAS_OPENPYXL:
            cell.fill = _HEADER_FILL
            cell.font = _HEADER_FONT
            cell.alignment = Alignment(horizontal="center", wrap_text=True)


def _write_rows(ws, rows: list[list[Any]], start_row: int = 2):
    for r, row_data in enumerate(rows, start_row):
        for c, val in enumerate(row_data, 1):
            cell = ws.cell(row=r, column=c, value=val)
            if isinstance(val, float):
                cell.number_format = "0.0000"


def _auto_width(ws):
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            try:
                max_len = max(max_len, len(str(cell.value or "")))
            except Exception:
                pass
        ws.column_dimensions[col_letter].width = min(max(max_len + 2, 8), 40)


def _fmt(val: Any, decimals: int = 4) -> Any:
    """Format float for display; pass through None and non-float."""
    if val is None:
        return ""
    if isinstance(val, float):
        return round(val, decimals)
    return val


# ---------------------------------------------------------------------------
# Sheet builders
# ---------------------------------------------------------------------------

def _build_overview(ws, exp_a: list[dict], exp_b: list[dict],
                    exp_c: list[dict], exp_d: list[dict]):
    """Sheet 1: one row per (model, emode) with key metrics from each exp."""
    headers = [
        "模型", "评测模式",
        # Exp A
        "A: 一致性率", "A: 位置偏见分", "A: 准确率下降",
        # Exp B
        "B: 直接准确率", "B: CoT准确率", "B: CoT增益", "B: CoT质量分", "B: 答对推错率",
        # Exp C
        "C: emode2准确率", "C: emode3准确率", "C: 描述增益", "C: 视觉一致性",
        # Exp D
        "D: 矛盾率", "D: 一致性分",
    ]
    _write_header(ws, headers)

    # Aggregate by (model, emode)
    from collections import defaultdict
    agg: dict[tuple, dict] = defaultdict(dict)

    for r in exp_a:
        key = (r["model"], r["emode"])
        d = agg[key]
        # Average across tasks
        for metric in ("consistency_rate", "position_bias_score", "accuracy_drop"):
            vals = d.get(f"_a_{metric}", [])
            if r.get(metric) is not None:
                vals.append(r[metric])
            d[f"_a_{metric}"] = vals

    for r in exp_b:
        key = (r["model"], r["emode"])
        d = agg[key]
        for metric in ("accuracy_direct", "accuracy_cot", "cot_gain", "cot_score", "accuracy_cot_gap"):
            vals = d.get(f"_b_{metric}", [])
            if r.get(metric) is not None:
                vals.append(r[metric])
            d[f"_b_{metric}"] = vals

    for r in exp_c:
        key = (r["model"], r.get("emode_desc", "emode2"))
        d = agg[key]
        for metric in ("accuracy_emode2", "accuracy_emode3", "description_gain", "visual_consistency"):
            vals = d.get(f"_c_{metric}", [])
            if r.get(metric) is not None:
                vals.append(r[metric])
            d[f"_c_{metric}"] = vals

    for r in exp_d:
        key = (r["model"], r["emode"])
        d = agg[key]
        for metric in ("contradiction_rate", "consistency_score"):
            vals = d.get(f"_d_{metric}", [])
            if r.get(metric) is not None:
                vals.append(r[metric])
            d[f"_d_{metric}"] = vals

    import numpy as np

    def _mean(lst):
        return float(np.mean(lst)) if lst else None

    rows = []
    for (model, emode), d in sorted(agg.items()):
        rows.append([
            model, emode,
            _fmt(_mean(d.get("_a_consistency_rate", []))),
            _fmt(_mean(d.get("_a_position_bias_score", []))),
            _fmt(_mean(d.get("_a_accuracy_drop", []))),
            _fmt(_mean(d.get("_b_accuracy_direct", []))),
            _fmt(_mean(d.get("_b_accuracy_cot", []))),
            _fmt(_mean(d.get("_b_cot_gain", []))),
            _fmt(_mean(d.get("_b_cot_score", []))),
            _fmt(_mean(d.get("_b_accuracy_cot_gap", []))),
            _fmt(_mean(d.get("_c_accuracy_emode2", []))),
            _fmt(_mean(d.get("_c_accuracy_emode3", []))),
            _fmt(_mean(d.get("_c_description_gain", []))),
            _fmt(_mean(d.get("_c_visual_consistency", []))),
            _fmt(_mean(d.get("_d_contradiction_rate", []))),
            _fmt(_mean(d.get("_d_consistency_score", []))),
        ])

    _write_rows(ws, rows)
    _auto_width(ws)


def _build_exp_a(ws, data: list[dict]):
    headers = [
        "模型", "评测模式", "层级", "任务", "子任务",
        "公共样本数", "rot0样本数", "rot1样本数", "rot2样本数",
        "准确率_rot0", "准确率_rot1", "准确率_rot2",
        "一致性率", "位置偏见分", "准确率下降",
    ]
    _write_header(ws, headers)
    rows = []
    for r in data:
        rows.append([
            r["model"], r["emode"], r["level"], r["task"], r["subtask"],
            r.get("n_common"), r.get("n_rot0"), r.get("n_rot1"), r.get("n_rot2"),
            _fmt(r.get("accuracy_rot0")), _fmt(r.get("accuracy_rot1")), _fmt(r.get("accuracy_rot2")),
            _fmt(r.get("consistency_rate")), _fmt(r.get("position_bias_score")),
            _fmt(r.get("accuracy_drop")),
        ])
    _write_rows(ws, rows)
    _auto_width(ws)


def _build_exp_b(ws, data: list[dict]):
    headers = [
        "模型", "评测模式", "层级", "任务", "子任务",
        "直接样本数", "CoT样本数",
        "直接准确率", "CoT准确率", "CoT增益",
        "CoT质量分", "答对推错率", "解析失败率",
        "发现覆盖率", "结论一致性", "推理顺序", "医学正确性", "链完整性",
    ]
    _write_header(ws, headers)
    rows = []
    for r in data:
        rows.append([
            r["model"], r["emode"], r["level"], r["task"], r["subtask"],
            r.get("n_direct"), r.get("n_cot"),
            _fmt(r.get("accuracy_direct")), _fmt(r.get("accuracy_cot")),
            _fmt(r.get("cot_gain")), _fmt(r.get("cot_score")),
            _fmt(r.get("accuracy_cot_gap")), _fmt(r.get("parse_failure_rate")),
            _fmt(r.get("dim_finding_coverage")),
            _fmt(r.get("dim_conclusion_consistency")),
            _fmt(r.get("dim_reasoning_order")),
            _fmt(r.get("dim_medical_correctness")),
            _fmt(r.get("dim_chain_completeness")),
        ])
    _write_rows(ws, rows)
    _auto_width(ws)


def _build_exp_c(ws, data: list[dict]):
    headers = [
        "模型", "描述模式", "图像模式", "层级", "任务", "子任务",
        "描述样本数", "图像样本数", "公共样本数",
        "emode2准确率", "emode3准确率", "描述增益",
        "视觉一致性", "描述救回率", "描述误导率",
    ]
    _write_header(ws, headers)
    rows = []
    for r in data:
        rows.append([
            r["model"], r.get("emode_desc", "emode2"), r.get("emode_img", "emode3"),
            r["level"], r["task"], r["subtask"],
            r.get("n_desc"), r.get("n_img"), r.get("n_common"),
            _fmt(r.get("accuracy_emode2")), _fmt(r.get("accuracy_emode3")),
            _fmt(r.get("description_gain")), _fmt(r.get("visual_consistency")),
            _fmt(r.get("flip_to_correct")), _fmt(r.get("flip_to_wrong")),
        ])
    _write_rows(ws, rows)
    _auto_width(ws)


def _build_exp_d(ws, data: list[dict]):
    headers = [
        "模型", "评测模式", "图像数", "矛盾数",
        "矛盾率", "一致性分",
        "R1矛盾率(出血+无DR)", "R2矛盾率(微动脉瘤+无DR)",
        "R3矛盾率(新生血管+非增殖期)", "R4矛盾率(硬渗+无DR)",
        "R1样本数", "R2样本数", "R3样本数", "R4样本数",
    ]
    _write_header(ws, headers)
    rows = []
    for r in data:
        rows.append([
            r["model"], r["emode"],
            r.get("n_images"), r.get("n_contradictions"),
            _fmt(r.get("contradiction_rate")), _fmt(r.get("consistency_score")),
            _fmt(r.get("rule_R1_rate")), _fmt(r.get("rule_R2_rate")),
            _fmt(r.get("rule_R3_rate")), _fmt(r.get("rule_R4_rate")),
            r.get("rule_R1_n"), r.get("rule_R2_n"),
            r.get("rule_R3_n"), r.get("rule_R4_n"),
        ])
    _write_rows(ws, rows)
    _auto_width(ws)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_excel(
    out_path: str,
    exp_a: list[dict],
    exp_b: list[dict],
    exp_c: list[dict],
    exp_d: list[dict],
) -> bool:
    """Write all experiment results to a multi-sheet Excel file.

    Returns True on success, False if openpyxl is unavailable.
    """
    if not _HAS_OPENPYXL:
        logger.error("openpyxl not installed — cannot write Excel")
        return False

    wb = openpyxl.Workbook()

    # Sheet 1: overview
    ws_overview = wb.active
    ws_overview.title = "总览"
    _build_overview(ws_overview, exp_a, exp_b, exp_c, exp_d)

    # Sheet 2: Exp A
    ws_a = wb.create_sheet("实验A_L7鲁棒性")
    if exp_a:
        _build_exp_a(ws_a, exp_a)
    else:
        ws_a.cell(1, 1, "暂无数据（实验A未完成）")

    # Sheet 3: Exp B
    ws_b = wb.create_sheet("实验B_CoT")
    if exp_b:
        _build_exp_b(ws_b, exp_b)
    else:
        ws_b.cell(1, 1, "暂无数据（实验B未完成）")

    # Sheet 4: Exp C
    ws_c = wb.create_sheet("实验C_L6描述依赖")
    if exp_c:
        _build_exp_c(ws_c, exp_c)
    else:
        ws_c.cell(1, 1, "暂无数据（实验C未完成）")

    # Sheet 5: Exp D
    ws_d = wb.create_sheet("实验D_L5一致性")
    if exp_d:
        _build_exp_d(ws_d, exp_d)
    else:
        ws_d.cell(1, 1, "暂无数据（实验D未完成）")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    wb.save(out_path)
    logger.info("Excel saved: %s", out_path)
    return True
