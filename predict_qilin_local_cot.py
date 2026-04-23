"""
Standalone local CoT inference for LLaVA-style models (e.g. Qilin-Med-VL-Chat).

This script is intentionally independent from existing predict*.py files so it
does not affect current pipelines.

Example:
    TRANSFORMERS_OFFLINE=1 HF_HOME=/root/shared-nvme/code/.cache/huggingface \
    CUDA_VISIBLE_DEVICES=1 \
    .venv3.10.20/bin/python predict_qilin_local_cot.py \
      --model_path /root/shared-nvme/code/.cache/hub/models--williamliu--Qilin-Med-VL-Chat/snapshots/386ff83b4469a09e179cf2fd293fa9769b6ad50d \
      --tasks L1,L2,L3,L4 \
      --emode both \
      --max_new_tokens 2048 \
      > logs/predict_williamliu_Qilin-Med-VL-Chat_both_cot.log 2>&1 &
"""

import argparse
import json
import os

from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor, CLIPImageProcessor, AutoTokenizer

from cot_evaluation.prompts import get_cot_prompts


os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", ".cache/huggingface")


class LocalLlavaPredictor:
    def __init__(self, model_path: str, image_size: int = 336, processor_path: str | None = None) -> None:
        processor_source = processor_path or model_path
        # Try AutoProcessor first; if it fails (missing preprocessor_config.json),
        # build LlavaProcessor manually from tokenizer + CLIPImageProcessor.
        try:
            self.processor = AutoProcessor.from_pretrained(
                processor_source,
                trust_remote_code=True,
            )
        except OSError:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            image_processor = CLIPImageProcessor(
                size={"shortest_edge": image_size},
                crop_size={"height": image_size, "width": image_size},
            )
            self.processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
        self.image_size = image_size

    def _build_prompt(self, query: str) -> str:
        return f"USER: <image>\n{query}\nASSISTANT:"

    def generate(self, query: str, img_path: str, max_new_tokens: int = 1024) -> str:
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
        prompt = self._build_prompt(query)

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {
            k: (v.to(device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        if text.startswith(prompt):
            return text[len(prompt):].strip()
        return text.strip()


def _should_process_task(level: str, task: str, subtask: str | None, task_filter: list[str]) -> bool:
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


def _model_name_from_path(model_path: str) -> str:
    parts = model_path.rstrip(os.sep).split(os.sep)
    name = parts[-1]
    for p in parts:
        if p.startswith("models--"):
            return p.replace("models--", "").replace("--", "/").split("/")[-1]
    return name


def main(
    model_path: str,
    img_root: str,
    text_root: str,
    out_root: str,
    set_name: str,
    task_filter: list[str],
    emode_filter: str,
    max_new_tokens: int,
    image_size: int,
    use_cot: bool,
    processor_path: str | None,
) -> None:
    model_name = _model_name_from_path(model_path)
    out_root = os.path.join(out_root, set_name, model_name)

    predictor = LocalLlavaPredictor(
        model_path=model_path,
        image_size=image_size,
        processor_path=processor_path,
    )

    task_infos: list[tuple[str, str, str | None]] = []
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

    print(f"Model path: {model_path}")
    print(f"Prompt mode: {'cot' if use_cot else 'direct'}")
    print(f"Output root: {out_root}")
    if processor_path:
        print(f"Processor path: {processor_path}")
    print(f"Tasks to process: {len(task_infos)}")

    for idx, (level, task, subtask) in enumerate(task_infos, start=1):
        task_name = subtask if subtask else task
        task_path = (
            os.path.join(text_root, level, task, subtask)
            if subtask
            else os.path.join(text_root, level, task)
        )
        print(f"{idx}/{len(task_infos)} processing: {task_name}")

        with open(task_path) as f:
            text_data = json.load(f)["data"]

        out_data_emode2: dict[str, str] = {}
        out_data_emode3: dict[str, str] = {}
        err_emode2 = 0
        err_emode3 = 0

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
                    out_data_emode2[img_name] = predictor.generate(
                        query=prompt_e2,
                        img_path=img_path,
                        max_new_tokens=max_new_tokens,
                    )
                except Exception as e:
                    err_emode2 += 1
                    out_data_emode2[img_name] = f"[ERROR] {type(e).__name__}: {e}"

            if emode_filter in ("emode3", "both") and prompt_e3:
                try:
                    out_data_emode3[img_name] = predictor.generate(
                        query=prompt_e3,
                        img_path=img_path,
                        max_new_tokens=max_new_tokens,
                    )
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
                f.write(json.dumps(out_data_emode2, indent=2, ensure_ascii=False))

        if out_data_emode3 and emode_filter in ("emode3", "both"):
            os.makedirs(os.path.dirname(out_path_e3), exist_ok=True)
            with open(out_path_e3, "w") as f:
                f.write(json.dumps(out_data_emode3, indent=2, ensure_ascii=False))

        print(
            f"  Done: {task_name} | emode2_err={err_emode2} | emode3_err={err_emode3}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone local CoT inference for Qilin/LLaVA")
    parser.add_argument(
        "--model_path",
        required=True,
        help="Local model path, ideally a snapshots/<hash> directory",
    )
    parser.add_argument("--img_root", default="datasets_preprocessed")
    parser.add_argument("--text_root", default="FunBench")
    parser.add_argument("--set_name", default="test")
    parser.add_argument("--out_root", default="answers")
    parser.add_argument("--tasks", default="L1,L2,L3,L4")
    parser.add_argument(
        "--emode",
        default="both",
        choices=["emode2", "emode3", "both"],
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument(
        "--processor_path",
        default=None,
        help="Optional processor path or repo ID when local model lacks preprocessor_config.json",
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Use CoT prompts. If omitted, use original E-mode prompts.",
    )
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        img_root=args.img_root,
        text_root=args.text_root,
        out_root=args.out_root,
        set_name=args.set_name,
        task_filter=[t.strip() for t in args.tasks.split(",") if t.strip()],
        emode_filter=args.emode,
        max_new_tokens=args.max_new_tokens,
        image_size=args.image_size,
        use_cot=args.cot,
        processor_path=args.processor_path,
    )
