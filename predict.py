"""
This is an example code of FunBench using Qwen2.5-VL (https://github.com/QwenLM/Qwen2.5-VL).
Please follow the instructions of Qwen2.5-VL to install the requiments.
For other MLLMs, you need to implement custom Predictor.
"""

import os
import json
from tqdm import tqdm

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"


class Predictor(object):

    def __init__(self, model_path, size=224) -> None:
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
        self.size = size
        
    def generate(self, query, img_path):
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
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            )
        inputs = inputs.to("cuda")
        with torch.inference_mode():
            # for Qwen, you may set the configs in generation_config.json in the model file to avoid randomness
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )            
         
        return output_text[0]

    
def main(img_root, text_root, set_name, model_path, out_root):

    out_root = os.path.join(out_root, set_name, model_path.rstrip(os.sep).split(os.sep)[-1])

    predictor = Predictor(model_path=model_path)

    # load all the tasks
    task_infos = []
    for level in os.listdir(text_root):
        if level.startswith('.') or level == 'README.md':
            continue
        for task in os.listdir(os.path.join(text_root, level)):
            if task.startswith('.'):
                continue
            if os.path.isdir(os.path.join(text_root, level, task)):
                for subtask in os.listdir(os.path.join(text_root, level, task)):
                    if subtask.startswith('.'):
                        continue    
                    task_infos.append((level, task, subtask))
            else:
                task_infos.append((level, task, None))

    # sorted by level and task
    task_infos = sorted(task_infos, key=lambda x: x[1])
    task_infos = sorted(task_infos, key=lambda x: x[0])


    for count, task_info in enumerate(task_infos):
        level, task, subtask = task_info
        if subtask is None:
            task_path = os.path.join(text_root, level, task)
        else:
            task_path = os.path.join(text_root, level, task, subtask)
        
        if subtask is None:
            print('{} of {}, processing: {}'.format(count + 1, len(task_infos), task))
        else:
            print('{} of {}, processing: {}'.format(count + 1, len(task_infos), subtask))
        with open(task_path) as fin:
            text_data = json.load(fin)['data']
        
        out_data_emode2 = {}
        out_data_emode3 = {}

        for img_name in tqdm(text_data):
            img_path = os.path.join(img_root, img_name)
            text_prompt_emode2 = text_data[img_name]['E-mode2']
            text_prompt_emode3 = text_data[img_name]['E-mode3']
            
            if text_prompt_emode2 is not None:
                out_emode2 = predictor.generate(query=text_prompt_emode2, img_path=img_path)
                out_data_emode2[img_name] = out_emode2
            
            out_emode3 = predictor.generate(query=text_prompt_emode3, img_path=img_path)
            out_data_emode3[img_name] = out_emode3
        
        if subtask is None:
            out_path_emode2 = os.path.join(out_root, 'emode2', level, task)
            out_path_emode3 = os.path.join(out_root, 'emode3', level, task)
        else:
            out_path_emode2 = os.path.join(out_root, 'emode2', level, task, subtask)
            out_path_emode3 = os.path.join(out_root, 'emode3', level, task, subtask)
        
        
        if len(out_data_emode2) > 0:
            os.makedirs(os.path.dirname(out_path_emode2), exist_ok=True)
            with open(out_path_emode2, 'w') as fout:
                fout.write(json.dumps(out_data_emode2, indent=4))
        os.makedirs(os.path.dirname(out_path_emode3), exist_ok=True)
        with open(out_path_emode3, 'w') as fout:
            fout.write(json.dumps(out_data_emode3, indent=4))

    
if __name__ == '__main__':
    
    img_root = 'datasets_preprocessed'
    text_root = 'FunBench' 

    set_name = 'test'
    model_path = 'Qwen/Qwen2.5-VL-7B-Instruct'  
    out_root = 'answers'
    main(img_root, text_root, set_name, model_path, out_root)          
            
