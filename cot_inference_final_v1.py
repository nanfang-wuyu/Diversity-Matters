# Import necessary libraries
import os
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, \
    LlavaOnevisionForConditionalGeneration, MllamaForConditionalGeneration, LlavaForConditionalGeneration, \
        BitsAndBytesConfig, AutoModelForImageTextToText, Gemma3ForConditionalGeneration
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import requests
from datasets import load_dataset, load_from_disk
import matplotlib.pyplot as plt
import numpy as np
import random
import json
import re
from huggingface_hub import login
from dotenv import load_dotenv, find_dotenv
import pickle
from tqdm import tqdm
import time
import sys
import argparse
import os
import re
import numpy as np
from Levenshtein import distance
from sentence_transformers import SentenceTransformer, util
from qwen_vl_utils import process_vision_info
from torch.cuda.amp import autocast
import ast


# for text similarity
SentBERT = SentenceTransformer("all-MiniLM-L6-v2")  


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

def resize_with_pillow(image, min_size=28):
    w, h = image.size
    if w < min_size or h < min_size:
        print(f"Resize starts: w{w}, h{h}")
        scale = max(min_size / w, min_size / h)
        scale = round(scale, 2)
        new_size = (int(w * scale), int(h * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        w, h = image.size
        print(f"Resize ends: w{w}, h{h}")
    return image

def resize_image(img):

    target_width, target_height = 640, 640
    # Calculate the target size (maximum width and height).
    if target_width and target_height:
        max_size = (target_width, target_height)
    elif target_width:
        max_size = (target_width, img.height)
    elif target_height:
        max_size = (img.width, target_height)

    img.thumbnail(max_size)

    return img

def load_model(model_name, device):
    """
    Load the model based on the provided model name.
    """
    if model_name == "llama":
        model = MllamaForConditionalGeneration.from_pretrained(
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config, 
            low_cpu_mem_usage=True,
            # device_map="auto",
            # local_files_only=True,
        )
    elif model_name == "molmo":
        model = AutoModelForCausalLM.from_pretrained(
            'allenai/Molmo-7B-D-0924',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config, 
            low_cpu_mem_usage=True,
            # device_map="auto",
        )
    elif model_name == "pixtral":
        model = LlavaForConditionalGeneration.from_pretrained(
            "mistral-community/pixtral-12b",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config, 
            low_cpu_mem_usage=True,
            # device_map="auto",
        )
    elif model_name == "gemma3_4b":
        model = Gemma3ForConditionalGeneration.from_pretrained(
            "google/gemma-3-4b-it", 
            torch_dtype=torch.bfloat16,
            # quantization_config=quantization_config, 
            # low_cpu_mem_usage=True,
            attn_implementation="eager",
            # device_map="auto"
        )
    elif model_name == "gemma3_12b":
        model = Gemma3ForConditionalGeneration.from_pretrained(
            "google/gemma-3-12b-it", 
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config, 
            # low_cpu_mem_usage=True,
            attn_implementation="eager",
            # device_map="auto"
        )
    elif model_name == "gemma3_27b":
        model = Gemma3ForConditionalGeneration.from_pretrained(
            "google/gemma-3-12b-it", 
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config, 
            # low_cpu_mem_usage=True,
            attn_implementation="eager",
            # device_map="auto"
        )
    elif model_name == "qwen_3b":
        model = AutoModelForImageTextToText.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config, 
            low_cpu_mem_usage=True,
            # attn_implementation="flash_attention_2", # enabling flash_attention_2 for better acceleration and memory saving
            )
    elif model_name == "qwen_7b":
        model = AutoModelForImageTextToText.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config, 
            low_cpu_mem_usage=True,
            # attn_implementation="flash_attention_2", # enabling flash_attention_2 for better acceleration and memory saving
            )
    elif model_name == "qwen_32b":
        model = AutoModelForImageTextToText.from_pretrained(
            "Qwen/Qwen2.5-VL-32B-Instruct",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config, 
            low_cpu_mem_usage=True,
            trust_remote_code=True
            # attn_implementation="flash_attention_2", # enabling flash_attention_2 for better acceleration and memory saving
            )
    elif model_name == "qwen_72b":
        model = AutoModelForImageTextToText.from_pretrained(
            "Qwen/Qwen2.5-VL-72B-Instruct",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config, 
            low_cpu_mem_usage=True,
            trust_remote_code=True
            # attn_implementation="flash_attention_2", # enabling flash_attention_2 for better acceleration and memory saving
            )
    elif model_name == "internvl3_14b":
        model = AutoModel.from_pretrained(
            "OpenGVLab/InternVL3-14B", 
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config, 
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval()
    else:
        raise ValueError(f"Model {model_name} not supported.")
    
    model.to(device)
    return model



def load_processor(model_name):
    """
    Load the processor based on the provided model name.
    """
    if model_name == "llama":
        processor = AutoProcessor.from_pretrained(
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            padding_side='left',
            )
    elif model_name == "molmo":
        processor = AutoProcessor.from_pretrained(
                            'allenai/Molmo-7B-D-0924',
                            trust_remote_code=True,
                            torch_dtype='auto',
                            device_map='auto'
            )
    elif model_name == "pixtral":
        processor = AutoProcessor.from_pretrained("mistral-community/pixtral-12b")
    elif model_name == "gemma3_4b":
        processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it", use_fast=True)
    elif model_name == "gemma3_12b":
        processor = AutoProcessor.from_pretrained("google/gemma-3-12b-it", use_fast=True)
    elif model_name == "gemma3_27b":
        processor = AutoProcessor.from_pretrained("google/gemma-3-27b-it", use_fast=True)
    elif model_name == "qwen_3b":
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", 
                                                  max_pixels = 1280*28*28,
                                                  min_pixels = 256*28*28)
    elif model_name == "qwen_7b":
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", 
                                                  max_pixels = 1280*28*28,
                                                  min_pixels = 256*28*28)
    elif model_name == "qwen_32b":
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct")
    elif model_name == "qwen_72b":
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")
    elif model_name == "internvl3_14b":
        processor = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-14B", trust_remote_code=True, use_fast=False)
    else:
        raise ValueError(f"Processor for {model_name} not supported.")   
    
    return processor



def load_dataset_(name):
    """
    Load the dataset based on the provided name.
    """
    if name == "VLQA":
        return load_dataset("VLQA", "testmini", split='test')
    elif name == "TQA":
        return load_dataset("TQA", split='test')
    elif name == "MathVista":
        return load_dataset("AI4Math/MathVista", split="testmini") # or split="test" no gt answer
    elif name == "AI2D_no_mask":
        return load_dataset("lmms-lab/ai2d-no-mask", split="test") # version with no mask
    elif name == "ScienceQA":
        return load_dataset("lmms-lab/ScienceQA-IMG", split="test") # or validation, train
    elif name == "SFE":
        return load_dataset("PrismaX/SFE", split="test")
    elif name == "MathVision":
        return load_dataset("MathLLMs/MathVision", split="test")
    elif name == "Test":
        return load_from_disk("mathvista_testmini_subset")
    elif name == "MMMU":
        return load_dataset("lmms-lab/MMMU", split="validation")
    elif name == "MMStar":
        return load_dataset("Lin-Chen/MMStar", split="val") 
    else:
        raise ValueError(f"Dataset {name} not supported.")

def option_recode(options: list, type):
    if type == 'number':
        return "\n".join([f'[{i}] {op}' for i, op in enumerate(options)])
    elif type == 'character':
        chara_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        return "\n".join([f'[{ch}] {op}' for ch, op in zip(chara_list[:len(options)], options)])
    elif type == 'nidx':
        return "\n".join([f'{op}' for i, op in enumerate(options)])
    else:
        raise ValueError("Invalid type")
    
def option_idx_recode(options: list, type):
    if type == 'number':
        return ", ".join([f'[{i}]' for i, op in enumerate(options)])
    elif type == 'character':
        chara_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        return ", ".join([f'[{ch}]' for ch, op in zip(chara_list[:len(options)], options)])
    else:
        raise ValueError("Invalid type")


def SBERT_similarity(sent1, sent2, nlp):
    emb1 = nlp.encode(sent1, convert_to_tensor=True)
    emb2 = nlp.encode(sent2, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()


def Answer_Extraction_Pipeline(input, opts, debug=False, mode='sim', thre=0.8, using_pixtral = False):
    # mode: if mode is 'sim', then always calculate the most similar answer and return. 
    #       Otherwise return -1 if the possibility is under a threshold $p$.


    # Input: (Full Response) Response without CoT / Second response with CoT: 
    resp = input

    # Step1 (New Response): 
    new_resp = resp.strip().split("assistant")[-1]

    if using_pixtral:
        new_resp = resp.split('**Answer:**')[-1].strip()

    charas = ['A', 'B', 'C', 'D', 'E']

    try:
        return charas[:len(opts)].index(re.sub(r"[\* \n\\]", "", new_resp).upper())
    except ValueError:
        pass

    # Step2 (Head or Tail): 
    def extract_head(text):
        sentences = re.split(r'(?<=[.?!])\s+', text.strip())  # Split by sentence-ending punctuation and also new line
        return sentences[0] if sentences else None
    def extract_tail(text):
        sentences = re.split(r'(?<=[.?!])\s+|\n', text.strip())  # Split by sentence-ending punctuation and also new line
        return sentences[-1] if sentences else None
    
    # def extract_head_and_tail(text):
    #     sentences = re.split(r'(?<=[.?!])\s+|\n', text.strip())  # Split by sentence-ending punctuation and also new line
    #     return sentences[0], sentences[-1] if sentences else None
    # head, tail = extract_head_and_tail(new_resp)
    head, tail = extract_head(new_resp), extract_tail(new_resp)
    if debug:
        print(f"Intitial head and tail: \nHead: {head}\nTail: {tail}")

    # Step 2.4 (Find Matched Option)
    def find_option(options, sentence):
        # Use word boundaries to ensure exact match
        pattern = r'\b(' + '|'.join(re.escape(opt) for opt in sorted(options, key=len, reverse=True)) + r')\b'
        match = re.search(pattern, sentence, re.IGNORECASE)
        choice = match.group(0).lower() if match else None
        options_lower = [opt.lower() for opt in options]
        if choice and choice in options_lower:
            return options_lower.index(choice), True
        else:
            return sentence, False
    
    idx, match = find_option(opts, tail)
    if match:
        if debug:
            print(f"Match option to: {opts[idx]}")
        return idx
    
    idx, match = find_option(opts, head)
    if match:
        if debug:
            print(f"Match option to: {opts[idx]}")
        return idx
    
    # Step2.5/2.6 (*Answer*)
    def extract_after_Answer(text):
        match = re.search(r'(\*?\*?(Answer|Conclusion)\*?\*?)[:：]?\s*([\s\S]+?)([.?!]\s|$)', text, re.IGNORECASE)
        return re.sub(r"[\*\n]", "", match.group(3).strip()) if match else text 

    head = extract_after_Answer(head)
    tail = extract_after_Answer(tail)
    
    # Step2.8 (is /is: [answer])
    def extract_after_is(text):
        match = re.search(r'\bis[:]? \s*(.+)', text)    
        return match.group(1) if match else text
    head = extract_after_is(head)
    tail = extract_after_is(tail)

    try:
        return charas[:len(opts)].index(re.sub(r"[\* \n\\]", "", head).upper())
    except ValueError:
        pass

    try:
        return charas[:len(opts)].index(re.sub(r"[\* \n\\]", "", tail).upper())
    except ValueError:
        pass


    # Step3 (Similarity): 
    sims_head = [SBERT_similarity(opt, head, SentBERT) for opt in opts]
    sims_tail = [SBERT_similarity(opt, tail, SentBERT) for opt in opts]


    # Step4 (Argmax): 
    ans_in_tail = np.max(sims_tail) > np.max(sims_head)
    if debug:
        print("Answer is in tail:", tail) if ans_in_tail else print("Answer is in head:", head)
        print(f"NewHead: {head}\nNewTail: {tail}\nOpts: {opts}")
        print(f"Similarity Scores:\nHead's: {sims_head}\nTail's: {sims_tail}")

    # Output: Index of the most correlated option
    output = np.argmax(sims_tail) if ans_in_tail else np.argmax(sims_head)
    confidence = np.max(sims_tail+sims_head)
    if mode != 'sim' and confidence < thre:
        print(f"{np.max(sims_tail+sims_head):.2f}, {tail}")
        return -1

    return output

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

def load_image(image_file, input_size=448, max_num=12):
    if type(image_file) == str:
        image = Image.open(image_file).convert('RGB')
    else:
        image = image_file
    
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    def build_transform(input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform
    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def inference_cot_singlechoice(ds, ds_name, model, model_name, processor, device, batch_size=1, debug=True, divided=True, code_type='character', setting='ncot_force', max_new_tokens = 128):

    wrong_idxs = []

    if 'options' in ds.column_names:
        option_s = 'options'
    elif 'choices' in ds.column_names:
        option_s = 'choices'
    else:
        raise ValueError("No options provided, check if it's a multichoice type dataset")

    conversations = []

    cot_responses = []

    settings = {
            'ncot_force': "Among the given options, ONLY SIMPLY choose the correct option in a single sentence or a single word. No preamble, no explanation. ",
            'zero_shot': "Among the given options, the answer is: ",
            'cot_trigger':"Among the given options, the answer is: (Let's think step by step, and give the answer at the end of your thought with **Answer:**.) ",
            }

    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Question: {sample['question']}" + "\nOptions:\n" + option_recode(sample[option_s], 'nidx') + "\n\n" + settings[setting]}, # for force-not cot

                ],
            },
        ]
        for sample in ds
    ]

    if ds_name in ['MMStar']:
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"Question: {sample['question']}" + "\n\n" + settings[setting]}, 

                    ],
                },
            ]
            for sample in ds
        ]

    if model_name in ['molmo', 'internvl3_14b']:
        prompts = [f"Question: {sample['question']}" + "\nOptions:\n" + option_recode(sample[option_s], 'nidx') + "\n\n" + settings[setting] for sample in ds]
        if ds_name in ["MMStar"]:
            prompts = [f"Question: {sample['question']}" + "\n\n" + settings[setting] for sample in ds]

    elif 'gemma' in model_name:
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": resize_image(sample['image'])},
                        {"type": "text", "text": f"Question: {sample['question']}" + "\nOptions:\n" + option_recode(sample[option_s], 'nidx') + "\n\n" + settings[setting]}, # for force-not cot

                    ],
                },
            ]
            for sample in ds
        ]

        if ds_name in ['MMStar']:
            conversations = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": resize_image(sample['image'])},
                            {"type": "text", "text": f"Question: {sample['question']}" + "\n\n" + settings[setting]}, 

                        ],
                    },
                ]
                for sample in ds
            ]
        prompts = [processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16) for conv in conversations]
    else:
        prompts = [processor.apply_chat_template(conv, add_generation_prompt=True) for conv in conversations]
    
    predictions = []
    corr_map = []

    total_failed_count = 0

    corr = 0
    total = 0
    acc = 0.0

    lst_responses = []
    
    for i in tqdm(range(0, len(ds), batch_size), desc="Processing Batches"):
        if i == 0:
            print("Example Prompt:", prompts[0])
        batch_prompts = prompts[i:i+batch_size]
        batch_images = ds[i:i+batch_size]["image"]
        
        if model_name == 'molmo':
            inputs = processor.process(
                images=batch_images,
                text=batch_prompts[0],
            )

            inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                output = model.generate_from_batch(
                    inputs,
                    GenerationConfig(max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>"),
                    tokenizer=processor.tokenizer,
                    do_sample=True,
                )

                generated_tokens = output[0,inputs['input_ids'].size(1):]
                responses = [processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)]

        if model_name == 'internvl3_14b':
            pixel_values = load_image(batch_images[0], max_num=12).to(torch.bfloat16)
            responses = [model.chat(processor, pixel_values.to(device), batch_prompts[0], dict(
                            do_sample=True,
                            temperature=0.8,
                            top_k=50,
                            top_p=0.95,
                            max_new_tokens=max_new_tokens,
                        ))]
        else:
            if 'gemma' in model_name:
                inputs = prompts[i]
            else:
                if 'qwen' in model_name:
                    batch_images = resize_with_pillow(batch_images[0])
                inputs = processor(
                    text=batch_prompts[0], 
                    images=batch_images, 
                    return_tensors="pt", 
                ).to(device)
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                with torch.no_grad():
                    generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
                    responses = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        lst_responses.append(responses[0])

        pre_corr = corr
        
        
        gts = ds[i:i+batch_size]["answer"]
        opts = ds[i:i+batch_size][option_s]
        if i == 0:
            print(gts, opts)
        
        ans_idx = Answer_Extraction_Pipeline(responses[0], opts[0], debug, using_pixtral = model_name=='pixtral') if ds_name != "MMStar" else -1
        # predictions.append(opts[0][ans_idx])
        preds = [ans_idx]

        if ds_name in ["AI2D_no_mask", "ScienceQA", "TQA"]: # answer is an index
            corr_list = [int(gt) == int(pred) for pred, gt in zip(preds, gts)]
            corr += sum(corr_list)
            corr_map.extend(corr_list)
        elif ds_name in ["MathVision", "MMMU", "MMStar"]: # answer is a character index
            chara_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
            corr_list = [chara_list.index(gt) == int(pred) for pred, gt in zip(preds, gts)]
            corr += sum(corr_list)
            corr_map.extend(corr_list)

        else: # answer is a str
            for pred, gt, op in zip(preds, gts, opts):
                if pred < len(op):
                    corr += (op[pred] == gt)
                    corr_map.append(op[pred] == gt)
                else:
                    corr_map.append(0)

        total += batch_size
        acc = round(corr / total, 4)

            
        if debug:
            print(f"Batch {i//batch_size}'s response:\n" + str(responses) + "\n------------------------------------------------------------------------------\n")
            
            if corr - pre_corr < batch_size:
                print("Wrong Answer!" + "\n------------------------------------------------------------------------------\n")
                wrong_idxs.append(i)
                print(preds, gts, opts, "\n------------------------------------------------------------------------------\n")


        if i % 10*batch_size == 0:
            print(f"Step {total}, Accuracy: {acc}", file=sys.stderr)


        if i % 10*batch_size == 0:
            print(f"Step {total}", file=sys.stderr)

    print("Failed ratio: ", round(total_failed_count / len(ds), 4))
    print("Final accuracy: ", acc)
    print("Wrong indexes: ", wrong_idxs)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/responses_{model_name}_{ds_name}_{setting}_{timestamp}.pkl", "wb") as f:
        pickle.dump(lst_responses, f)
    return predictions, cot_responses, corr_map      

def parse_args(default_config):
    parser = argparse.ArgumentParser(description="Parse configuration parameters.")

    # Define arguments with default values from JSON
    parser.add_argument("--dataset_name", type=str, default=default_config["dataset_name"])
    parser.add_argument("--model_name", type=str, default=default_config["model_name"])
    parser.add_argument("--batch_size", type=int, default=default_config["batch_size"])
    parser.add_argument("--max_new_tokens", type=int, default=default_config["max_new_tokens"])
    parser.add_argument("--num_workers", type=int, default=default_config["num_workers"])
    parser.add_argument("--use_cuda", action="store_true" if default_config["use_cuda"] else "store_false")
    parser.add_argument("--seed", type=int, default=default_config["seed"])
    parser.add_argument("--debug", action="store_true" if default_config["debug"] else "store_false")
    parser.add_argument("--cot", action="store_true" if default_config["cot"] else "store_false")
    parser.add_argument("--code_type", type=str, default=default_config["code_type"])
    parser.add_argument("--setting", type=str, default=default_config["setting"])
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--ds_name_list", type=str, nargs="+", default=default_config["ds_name_list"])

    return parser.parse_args()



def main():

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    with open("config.json", "r") as f:
        config = json.load(f)

    args = parse_args(config)
    config = vars(args)

    device = torch.device("cuda" if config["use_cuda"] else "cpu")
    print(f"Using device: {device}")
    print(torch.cuda.is_available())

    print("Configuration Settings:")
    for key, value in config.items():
        print(f"{key}: {value}")

    # _ = load_dotenv(find_dotenv('.env'))
    # login(os.getenv('hug'))


    
    processor = load_processor(config["model_name"])
    model = load_model(config["model_name"], device) 

    model.eval()
   
    model.generation_config.update(
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        max_new_tokens=config['max_new_tokens'],
    )
    
    ds_name = config['dataset_name']
    model_name = config["model_name"]
    print(ds_name, model_name)

    for i in range(config['loops']):
        torch_seed = torch.seed()  # Sets and returns a new random seed
        torch.manual_seed(torch_seed)

        np_seed = np.random.randint(0, 2**32 - 1)
        np.random.seed(np_seed)

        py_seed = random.randint(0, 2**32 - 1)
        random.seed(py_seed)
        
        ds = load_dataset_(ds_name)

        if ds_name in ["MathVista", "Test", "MathVision"]:
            if 'decoded_image' in ds.column_names:
                ds = ds.rename_columns({'image': 'image_path', 'decoded_image': 'image'})
            if 'question_type' in ds.column_names: 
                ds_multi_choice = ds.filter(lambda x: x["question_type"] == "multi_choice")
            else:
                ds_multi_choice = ds.filter(lambda x: x["options"] != [] and x["options"] is not None)
            print(f"There are {len(ds_multi_choice)} / {len(ds)} samples in {ds_name} are multi-choice type.")
            results = inference_cot_singlechoice(ds_multi_choice, ds_name, model, config['model_name'], processor, device, config['batch_size'], config['debug'], config['cot'], config['code_type'], config['setting'], config['max_new_tokens'])

        elif ds_name == "MMMU":
            ds_multi_choice = ds.filter(lambda x: x["question_type"] == "multiple-choice" and x['image_2'] is None)
            ds_multi_choice = ds_multi_choice.rename_columns({'image_1': 'image'})
            print(f"There are {len(ds_multi_choice)} / {len(ds)} samples in {ds_name} are multi-choice type.")
            def parse_options(example):
                options_str = example['options']
                example['options'] = ast.literal_eval(options_str) if isinstance(options_str, str) else options_str
                return example

            ds_multi_choice = ds_multi_choice.map(parse_options)
            results = inference_cot_singlechoice(ds_multi_choice, ds_name, model, config['model_name'], processor, device, config['batch_size'], config['debug'], config['cot'], config['code_type'], config['setting'], config['max_new_tokens']) 
        elif ds_name == "MMStar":
            ds_multi_choice = ds.add_column('options', [[] for _ in range(len(ds))])
            results = inference_cot_singlechoice(ds_multi_choice, ds_name, model, config['model_name'], processor, device, config['batch_size'], config['debug'], config['cot'], config['code_type'], config['setting'], config['max_new_tokens']) 
        elif ds_name == "SFE":
            ds_multi_choice = ds.filter(lambda x: x["question_type"] == "mcq" and len(x["images"]) == 1 )
            ds_multi_choice = ds_multi_choice.rename_columns({'images': 'image'})
            results = inference_cot_singlechoice(ds_multi_choice, ds_name, model, config['model_name'], processor, device, config['batch_size'], config['debug'], config['cot'], config['code_type'], config['setting'], config['max_new_tokens']) 


        else:
            results = inference_cot_singlechoice(ds, ds_name, model, config['model_name'], processor, device, config['batch_size'], config['debug'], config['cot'], config['code_type'], config['setting'], config['max_new_tokens']) 
    
    print("Configuration Settings:")
    for key, value in config.items():
        print(f"{key}: {value}")




if __name__ == "__main__":
    main()