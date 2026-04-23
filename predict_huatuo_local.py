"""
Local inference for HuatuoGPT-Vision-7B on FunBench.

Requires HuatuoGPT-Vision repo cloned at /root/shared-nvme/code/HuatuoGPT-Vision/

Usage:
    CUDA_VISIBLE_DEVICES=1 \
    .venv3.10.20/bin/python predict_huatuo_local.py \
        --model_path /root/shared-nvme/code/.cache/hub/models--FreedomIntelligence--HuatuoGPT-Vision-7B/snapshots/34dfcdbb7728ff38da865839f342b88c4cf6ef39 \
        --tasks L1,L2,L3,L4 --emode both --max_new_tokens 512 \
        > logs/predict_huatuo_L1234_both_direct.log 2>&1 &
"""

import argparse
import json
import os
import sys

from tqdm import tqdm

# Add HuatuoGPT-Vision repo to path
HUATUO_REPO = "/root/shared-nvme/code/HuatuoGPT-Vision"
sys.path.insert(0, HUATUO_REPO)

from cot_evaluation.prompts import get_cot_prompts


def _should_process_task(level, task, subtask, task_filter):
    if not task_filter:
        return True
    candidates = [level, task]
    if subtask:
        candidates.append(subtask)
    for f in task_filter:
        for c in candidates:
            if c.upper().startswith(f.upper()):
                return True
    return False


def _model_name_from_path(model_path):
    parts = model_path.rstrip(os.sep).split(os.sep)
    for p in parts:
        if p.startswith("models--"):
            return p.replace("models--", "").replace("--", "/").split("/")[-1]
    return parts[-1]


def main(
    model_path,
    img_root,
    text_root,
    out_root,
    set_name,
    task_filter,
    emode_filter,
    max_new_tokens,
    use_cot,
    device,
):
    from cli import HuatuoChatbot

    model_name = _model_name_from_path(model_path)
    out_root = os.path.join(out_root, set_name, model_name)

    print(f"Model path: {model_path}")
    print(f"Prompt mode: {'cot' if use_cot else 'direct'}")
    print(f"Output root: {out_root}")

    bot = HuatuoChatbot(model_path, device=device)
    bot.gen_kwargs["max_new_tokens"] = max_new_tokens

    task_infos = []
    for level in sorted(os.listdir(text_root)):
        if level.startswith(".") or level == "README.md":
            continue
        for task in sorted(os.listdir(os.path.join(text_root, level))):
            if task.startswith("."):
                continue
            task_dir = os.path.join(text_root, level, task)
            if os.path.isdir(task_dir):
                for subtask in sorted(os.listdir(task_dir)):
                    if not subtask.startswith("."):
                        task_infos.append((level, task, subtask))
            else:
                task_infos.append((level, task, None))

    if task_filter:
        task_infos = [
            t for t in task_infos
            if _should_process_task(t[0], t[1], t[2], task_filter)
        ]

    print(f"Tasks to process: {len(task_infos)}")

    for idx, (level, task, subtask) in enumerate(task_infos, start=1):
        task_name = subtask if subtask else task
        task_path = (
            os.path.join(text_root, level, task, subtask)
            if subtask else os.path.join(text_root, level, task)
        )
        print(f"{idx}/{len(task_infos)} processing: {task_name}")

        with open(task_path) as f:
            text_data = json.load(f)["data"]

        out_data_emode2 = {}
        out_data_emode3 = {}
        err_emode2 = err_emode3 = 0

        for img_name in tqdm(text_data):
            img_path = os.path.join(img_root, img_name)
            if not os.path.exists(img_path):
                continue

            if use_cot:
                prompts = get_cot_prompts(text_data[img_name])
                prompt_e2 = prompts.get("emode2_cot")
                prompt_e3 = prompts.get("emode3_cot")
            else:
                prompt_e2 = text_data[img_name].get("E-mode2")
                prompt_e3 = text_data[img_name].get("E-mode3")

            if emode_filter in ("emode2", "both") and prompt_e2:
                try:
                    bot.clear_history()
                    out_data_emode2[img_name] = bot.inference(prompt_e2, [img_path])[0]
                except Exception as e:
                    err_emode2 += 1
                    out_data_emode2[img_name] = f"[ERROR] {type(e).__name__}: {e}"

            if emode_filter in ("emode3", "both") and prompt_e3:
                try:
                    bot.clear_history()
                    out_data_emode3[img_name] = bot.inference(prompt_e3, [img_path])[0]
                except Exception as e:
                    err_emode3 += 1
                    out_data_emode3[img_name] = f"[ERROR] {type(e).__name__}: {e}"

        emode2_dir = "emode2_cot" if use_cot else "emode2"
        emode3_dir = "emode3_cot" if use_cot else "emode3"
        if subtask:
            out_path_e2 = os.path.join(out_root, emode2_dir, level, task, subtask)
            out_path_e3 = os.path.join(out_root, emode3_dir, level, task, subtask)
        else:
            out_path_e2 = os.path.join(out_root, emode2_dir, level, task)
            out_path_e3 = os.path.join(out_root, emode3_dir, level, task)

        if out_data_emode2 and emode_filter in ("emode2", "both"):
            os.makedirs(os.path.dirname(out_path_e2), exist_ok=True)
            with open(out_path_e2, "w") as f:
                json.dump(out_data_emode2, f, indent=2, ensure_ascii=False)

        if out_data_emode3 and emode_filter in ("emode3", "both"):
            os.makedirs(os.path.dirname(out_path_e3), exist_ok=True)
            with open(out_path_e3, "w") as f:
                json.dump(out_data_emode3, f, indent=2, ensure_ascii=False)

        print(f"  Done: {task_name} | emode2_err={err_emode2} | emode3_err={err_emode3}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--img_root", default="datasets_preprocessed")
    parser.add_argument("--text_root", default="FunBench")
    parser.add_argument("--set_name", default="test")
    parser.add_argument("--out_root", default="answers")
    parser.add_argument("--tasks", default="L1,L2,L3,L4")
    parser.add_argument("--emode", default="both", choices=["emode2", "emode3", "both"])
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    task_filter = [t.strip() for t in args.tasks.split(",") if t.strip()]
    main(
        model_path=args.model_path,
        img_root=args.img_root,
        text_root=args.text_root,
        out_root=args.out_root,
        set_name=args.set_name,
        task_filter=task_filter,
        emode_filter=args.emode,
        max_new_tokens=args.max_new_tokens,
        use_cot=args.cot,
        device=args.device,
    )
