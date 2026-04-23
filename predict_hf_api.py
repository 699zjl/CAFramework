"""
FunBench inference via cloud API backends (HF Inference API or Together AI).

Mirrors predict_cot.py structure exactly. Use this for comparison experiments
with models that don't need local download.

Backends:
  --backend hf        HuggingFace Inference API (requires HF_TOKEN)
  --backend together  Together AI (requires TOGETHER_API_KEY, more stable for vision)

Supported models (examples):
  google/gemma-4-31b-it                              (Together)
  meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo    (Together)
  google/gemma-4-31B-it:together                     (HF router → Together)

Output goes to:  answers/<set_name>/<model_short_name>/emode2/ and emode3/
(same layout as predict_cot.py so evaluate.py works unchanged)

Usage:
    # Together AI (recommended for large vision models)
    export TOGETHER_API_KEY=xxx
    python predict_hf_api.py --backend together \
        --model google/gemma-4-31b-it --tasks L4c --emode emode3

    # HuggingFace Inference API
    export HF_TOKEN=hf_xxx
    python predict_hf_api.py --backend hf \
        --model google/gemma-4-31B-it:together --tasks L4c --emode emode3
"""

import argparse
import json
import logging
import os
from tqdm import tqdm

from cot_evaluation.prompts import get_cot_prompts
from predictors.hf_api import HFAPIPredictor
from predictors.together_api import TogetherAPIPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def _should_process_task(
    level: str, task: str, subtask: str | None, task_filter: list[str]
) -> bool:
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
    model_id: str,
    out_root: str,
    task_filter: list[str],
    emode_filter: str,
    max_new_tokens: int,
    img_size: int,
    use_cot: bool,
    backend: str,
) -> None:
    # Short model name for output directory, e.g. "gemma-4-31b-it"
    model_short = model_id.split("/")[-1].split(":")[0]
    out_root = os.path.join(out_root, set_name, model_short)

    if backend == "together":
        predictor = TogetherAPIPredictor(model_id=model_id, size=img_size)
    else:
        predictor = HFAPIPredictor(model_id=model_id, size=img_size)
    print(f"Backend: {backend}  Model: {model_id}")
    print(f"Output root: {out_root}")

    # Discover tasks (identical logic to predict_cot.py)
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
                    if not subtask.startswith("."):
                        task_infos.append((level, task, subtask))
            else:
                task_infos.append((level, task, None))

    task_infos = sorted(task_infos, key=lambda x: x[1])
    task_infos = sorted(task_infos, key=lambda x: x[0])

    if task_filter:
        task_infos = [
            t for t in task_infos
            if _should_process_task(t[0], t[1], t[2], task_filter)
        ]
    print(f"Tasks to process: {len(task_infos)}")

    # Subdirectory suffix: _cot when using CoT prompts
    emode2_dir = "emode2_cot" if use_cot else "emode2"
    emode3_dir = "emode3_cot" if use_cot else "emode3"

    for count, (level, task, subtask) in enumerate(task_infos):
        task_path = (
            os.path.join(text_root, level, task, subtask)
            if subtask
            else os.path.join(text_root, level, task)
        )
        task_name = subtask if subtask else task
        print(f"{count + 1}/{len(task_infos)} processing: {task_name}")

        with open(task_path) as fin:
            text_data = json.load(fin)["data"]

        out_data_emode2: dict = {}
        out_data_emode3: dict = {}
        processed_count = 0

        for img_name in tqdm(text_data):
            img_path = os.path.join(img_root, img_name)
            if not os.path.exists(img_path):
                continue
            processed_count += 1

            if use_cot:
                prompts = get_cot_prompts(text_data[img_name])
                prompt_e2 = prompts.get("emode2_cot")
                prompt_e3 = prompts.get("emode3_cot")
            else:
                prompt_e2 = text_data[img_name].get("E-mode2")
                prompt_e3 = text_data[img_name].get("E-mode3")

            if emode_filter in ("emode2", "both") and prompt_e2:
                out_data_emode2[img_name] = predictor.generate(
                    query=prompt_e2, img_path=img_path, max_new_tokens=max_new_tokens
                )

            if emode_filter in ("emode3", "both") and prompt_e3:
                out_data_emode3[img_name] = predictor.generate(
                    query=prompt_e3, img_path=img_path, max_new_tokens=max_new_tokens
                )

        print(f"  Done: {task_name} | processed={processed_count}/{len(text_data)}")

        # Build output paths
        if subtask:
            base_e2 = os.path.join(out_root, emode2_dir, level, task, subtask)
            base_e3 = os.path.join(out_root, emode3_dir, level, task, subtask)
        else:
            base_e2 = os.path.join(out_root, emode2_dir, level, task)
            base_e3 = os.path.join(out_root, emode3_dir, level, task)

        if out_data_emode2 and emode_filter in ("emode2", "both"):
            os.makedirs(os.path.dirname(base_e2), exist_ok=True)
            with open(base_e2, "w") as fout:
                fout.write(json.dumps(out_data_emode2, indent=4))

        if out_data_emode3 and emode_filter in ("emode3", "both"):
            os.makedirs(os.path.dirname(base_e3), exist_ok=True)
            with open(base_e3, "w") as fout:
                fout.write(json.dumps(out_data_emode3, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FunBench via cloud API (HF or Together)")
    parser.add_argument(
        "--backend",
        default="hf",
        choices=["hf", "together"],
        help="API backend: 'hf' (HF Inference API) or 'together' (Together AI)",
    )
    parser.add_argument(
        "--model",
        default="google/gemma-4-31B-it:together",
        help="Model ID, e.g. google/gemma-4-31B-it:together (HF) or google/gemma-4-31b-it (Together)",
    )
    parser.add_argument("--img_root", default="datasets_preprocessed")
    parser.add_argument("--text_root", default="FunBench")
    parser.add_argument("--set_name", default="test")
    parser.add_argument("--out_root", default="answers")
    parser.add_argument(
        "--tasks",
        default="L3,L4",
        help="Comma-separated task prefixes, e.g. 'L3,L4' or 'L4c'",
    )
    parser.add_argument(
        "--emode",
        default="emode3",
        choices=["emode2", "emode3", "both"],
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Use CoT-wrapped prompts (requires cot_evaluation.prompts)",
    )
    args = parser.parse_args()

    main(
        img_root=args.img_root,
        text_root=args.text_root,
        set_name=args.set_name,
        model_id=args.model,
        out_root=args.out_root,
        task_filter=[t.strip() for t in args.tasks.split(",") if t.strip()],
        emode_filter=args.emode,
        max_new_tokens=args.max_new_tokens,
        img_size=args.img_size,
        use_cot=args.cot,
        backend=args.backend,
    )
