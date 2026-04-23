"""
FunBench-CoT: Chain-of-Thought inference script.

Mirrors predict.py structure exactly. Replaces the original FunBench prompt
with a CoT-wrapped version that elicits structured reasoning chains.

Output format: JSON per task, same directory structure as predict.py but
under emode2_cot/ and emode3_cot/ instead of emode2/ and emode3/.

Usage:
    TRANSFORMERS_OFFLINE=1 HF_HOME=.cache/huggingface \
    CUDA_VISIBLE_DEVICES=0,1 \
    .venv3.10.20/bin/python predict_cot.py \
        --model_path /root/shared-nvme/code/.cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/... \
        --tasks L3,L4 \
        --emode emode3
"""

import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", ".cache/huggingface")

import argparse
import json
from tqdm import tqdm

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from cot_evaluation.prompts import get_cot_prompts


class Predictor:

    def __init__(self, model_path: str, size: int = 224) -> None:
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        self.size = size

    def generate(self, query: str, img_path: str, max_new_tokens: int = 1024) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                        "resized_height": self.size,
                        "resized_width": self.size,
                    },
                    {"type": "text", "text": query},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(next(self.model.parameters()).device)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        return output_text[0]


def _should_process_task(level: str, task: str, subtask: str | None, task_filter: list[str]) -> bool:
    """Check if a task matches the filter list.

    Matches against level directory, task name, or subtask name.
    e.g. 'L4' matches L4-disease_diagnosis/*
         'L4c' matches L4-disease_diagnosis/L4c-dr_grading*
         'L3a5' matches L3-lesion_analysis/L3a-lesion_recognition/L3a5-*
    """
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


def main(
    img_root: str,
    text_root: str,
    set_name: str,
    model_path: str,
    out_root: str,
    task_filter: list[str],
    emode_filter: str,
    max_new_tokens: int,
) -> None:

    # Extract model name from path: prefer "models--Org--Name" segment over snapshot hash
    path_parts = model_path.rstrip(os.sep).split(os.sep)
    model_name = path_parts[-1]
    for part in path_parts:
        if part.startswith("models--"):
            model_name = part.replace("models--", "").replace("--", "/").split("/")[-1]
            break
    out_root = os.path.join(out_root, set_name, model_name)

    predictor = Predictor(model_path=model_path, size=224)

    # Discover all tasks (same logic as predict.py)
    task_infos = []
    for level in os.listdir(text_root):
        if level.startswith(".") or level == "README.md":
            continue
        for task in os.listdir(os.path.join(text_root, level)):
            if task.startswith("."):
                continue
            task_dir = os.path.join(text_root, level, task)
            if os.path.isdir(task_dir):
                for subtask in os.listdir(task_dir):
                    if subtask.startswith("."):
                        continue
                    task_infos.append((level, task, subtask))
            else:
                task_infos.append((level, task, None))

    task_infos = sorted(task_infos, key=lambda x: x[1])
    task_infos = sorted(task_infos, key=lambda x: x[0])

    # Apply task filter
    if task_filter:
        task_infos = [
            (level, task, subtask) for level, task, subtask in task_infos
            if _should_process_task(level, task, subtask, task_filter)
        ]
    print(f"Tasks to process: {len(task_infos)}")

    for count, task_info in enumerate(task_infos):
        level, task, subtask = task_info

        if subtask is None:
            task_path = os.path.join(text_root, level, task)
        else:
            task_path = os.path.join(text_root, level, task, subtask)

        task_name = subtask if subtask else task
        print(f"{count + 1}/{len(task_infos)} processing: {task_name}")

        with open(task_path) as fin:
            text_data = json.load(fin)["data"]

        out_data_emode2: dict = {}
        out_data_emode3: dict = {}
        processed_count = 0
        parse_fail_emode2 = 0
        parse_fail_emode3 = 0

        pbar = tqdm(text_data)
        for img_name in pbar:
            img_path = os.path.join(img_root, img_name)
            if not os.path.exists(img_path):
                continue

            processed_count += 1
            cot_prompts = get_cot_prompts(text_data[img_name])

            if emode_filter in ("emode2", "both"):
                prompt_emode2 = cot_prompts["emode2_cot"]
                if prompt_emode2 is not None:
                    raw = predictor.generate(
                        query=prompt_emode2,
                        img_path=img_path,
                        max_new_tokens=max_new_tokens,
                    )
                    out_data_emode2[img_name] = raw
                    if "<observations>" not in raw:
                        parse_fail_emode2 += 1

            if emode_filter in ("emode3", "both"):
                prompt_emode3 = cot_prompts["emode3_cot"]
                if prompt_emode3 is not None:
                    raw = predictor.generate(
                        query=prompt_emode3,
                        img_path=img_path,
                        max_new_tokens=max_new_tokens,
                    )
                    out_data_emode3[img_name] = raw
                    if "<observations>" not in raw:
                        parse_fail_emode3 += 1

        print(
            f"  Done: {task_name} | processed={processed_count} | "
            f"parse_fail_emode2={parse_fail_emode2} | parse_fail_emode3={parse_fail_emode3}"
        )

        # Save outputs
        if subtask is None:
            out_path_emode2 = os.path.join(out_root, "emode2_cot", level, task)
            out_path_emode3 = os.path.join(out_root, "emode3_cot", level, task)
        else:
            out_path_emode2 = os.path.join(out_root, "emode2_cot", level, task, subtask)
            out_path_emode3 = os.path.join(out_root, "emode3_cot", level, task, subtask)

        if out_data_emode2 and emode_filter in ("emode2", "both"):
            os.makedirs(os.path.dirname(out_path_emode2), exist_ok=True)
            with open(out_path_emode2, "w") as fout:
                fout.write(json.dumps(out_data_emode2, indent=4))

        if out_data_emode3 and emode_filter in ("emode3", "both"):
            os.makedirs(os.path.dirname(out_path_emode3), exist_ok=True)
            with open(out_path_emode3, "w") as fout:
                fout.write(json.dumps(out_data_emode3, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FunBench-CoT inference")
    parser.add_argument(
        "--model_path",
        default="/root/shared-nvme/code/.cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct",
        help="Path to local model directory",
    )
    parser.add_argument(
        "--img_root", default="datasets_preprocessed", help="Preprocessed image root"
    )
    parser.add_argument(
        "--text_root", default="FunBench", help="FunBench task JSON root"
    )
    parser.add_argument("--set_name", default="test", help="Dataset split name")
    parser.add_argument("--out_root", default="answers", help="Output root directory")
    parser.add_argument(
        "--tasks",
        default="L3,L4",
        help="Comma-separated task level prefixes to process, e.g. 'L3,L4' or 'L4c'",
    )
    parser.add_argument(
        "--emode",
        default="emode3",
        choices=["emode2", "emode3", "both"],
        help="Evaluation mode to run",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Max new tokens for CoT generation (higher than direct answer)",
    )
    args = parser.parse_args()

    task_filter = [t.strip() for t in args.tasks.split(",") if t.strip()]

    main(
        img_root=args.img_root,
        text_root=args.text_root,
        set_name=args.set_name,
        model_path=args.model_path,
        out_root=args.out_root,
        task_filter=task_filter,
        emode_filter=args.emode,
        max_new_tokens=args.max_new_tokens,
    )
