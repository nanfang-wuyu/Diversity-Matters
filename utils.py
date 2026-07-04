

import os
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, \
    LlavaOnevisionForConditionalGeneration, MllamaForConditionalGeneration, LlavaForConditionalGeneration, \
        BitsAndBytesConfig, AutoModelForImageTextToText
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

# SentBERT = SentenceTransformer("all-MiniLM-L6-v2")  
SentBERT = None





def load_model(model_name, device):
    """
    Load the model based on the provided model name.
    """
    if model_name == "llama":
        model = MllamaForConditionalGeneration.from_pretrained(
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            torch_dtype=torch.float16,
            quantization_config=quantization_config, 
            low_cpu_mem_usage=True,
            # device_map="auto",
            # local_files_only=True,
        )
    elif model_name == "molmo":
        model = AutoModelForCausalLM.from_pretrained(
            'allenai/Molmo-7B-D-0924',
            trust_remote_code=True,
            torch_dtype=torch.float16,
            quantization_config=quantization_config, 
            low_cpu_mem_usage=True,
            # device_map="auto",
        )
    elif model_name == "llava":
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            "llava-hf/llava-onevision-qwen2-7b-ov-hf", 
            # model_path,
            torch_dtype=torch.float16,
            quantization_config=quantization_config, 
            low_cpu_mem_usage=True,
            # device_map="auto",
            use_safetensors=True,

        )
    elif model_name == "pixtral":
        model = LlavaForConditionalGeneration.from_pretrained(
            "mistral-community/pixtral-12b",
            torch_dtype=torch.float16,
            quantization_config=quantization_config, 
            low_cpu_mem_usage=True,
            # device_map="auto",
        )
    elif model_name == "gemma":
        model = Gemma3ForConditionalGeneration.from_pretrained(
            "google/gemma-3-4b-it", 
            torch_dtype=torch.float16,
            quantization_config=quantization_config, 
            low_cpu_mem_usage=True,
            # device_map="auto"
        )
    elif model_name == "qwen":
        model = AutoModelForImageTextToText.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.float16,
            quantization_config=quantization_config, 
            low_cpu_mem_usage=True,
            # attn_implementation="flash_attention_2", # enabling flash_attention_2 for better acceleration and memory saving
            )
    elif model_name == "qwen-32":
        model = AutoModelForImageTextToText.from_pretrained(
            "Qwen/Qwen2.5-VL-32B-Instruct",
            torch_dtype=torch.float16,
            quantization_config=quantization_config, 
            low_cpu_mem_usage=True,
            trust_remote_code=True
            # attn_implementation="flash_attention_2", # enabling flash_attention_2 for better acceleration and memory saving
            )
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
    elif model_name == "llava":
        processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")
    elif model_name == "pixtral":
        processor = AutoProcessor.from_pretrained("mistral-community/pixtral-12b")
    elif model_name == "gemma":
        processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it", padding_side="left")
    elif model_name == "qwen":
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", 
                                                  max_pixels = 1280*28*28,
                                                  min_pixels = 256*28*28)
    elif model_name == "qwen-32":
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct")
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
    elif name == "FlowLearn":
        return load_dataset("flowlearn_test", split='test')
    elif name == "ChartQA":
        return load_dataset("HuggingFaceM4/ChartQA", split='test')
    elif name == "CountBench":
        return load_dataset("vikhyatk/CountBenchQA", split='test')
    elif name == "HallusionBench":
        return load_dataset("lmms-lab/HallusionBench", split='image')
    elif name == "MathVista":
        ds = load_dataset("AI4Math/MathVista", split="testmini") # or split="test" no gt answer
        ds = ds.filter(lambda x: x["question_type"] == "multi_choice")
        return ds
    elif name == "VisualWebBench":
        return load_dataset("visualwebbench/VisualWebBench", "webqa", split="test")
    elif name == "InfographicVQA":
        return load_dataset("lmms-lab/DocVQA", "InfographicVQA", split="validation") # or split="test"
    elif name == "AI2D_no_mask":
        return load_dataset("lmms-lab/ai2d-no-mask", split="test") # version with no mask
    elif name == "ScienceQA":
        return load_dataset("lmms-lab/ScienceQA-IMG", split="test") # or validation, train
    elif name == "SFE":
        return load_dataset("PrismaX/SFE", split="test")
    elif name == "MathVision":
        ds = load_dataset("MathLLMs/MathVision", split="test")
        ds = ds.filter(lambda x: x["options"] != [] and x["options"] is not None)
        return ds
    elif name == "Test":
        return load_from_disk("mathvista_testmini_subset")
    elif name == "MMMU":
        ds = load_dataset("lmms-lab/MMMU", split="validation")
        import ast
        ds_multi_choice = ds.filter(lambda x: x["question_type"] == "multiple-choice" and x['image_2'] is None)
        ds_multi_choice = ds_multi_choice.rename_columns({'image_1': 'image'})
        def parse_options(example):
            options_str = example['options']
            example['options'] = ast.literal_eval(options_str) if isinstance(options_str, str) else options_str
            return example

        ds_multi_choice = ds_multi_choice.map(parse_options)
        return ds_multi_choice
    elif name == "MMStar":
        # ds = load_dataset("Lin-Chen/MMStar", split="val") 
        from datasets import load_from_disk
        ds = load_from_disk("MMStar_with_options")
        return ds
    else:
        raise ValueError(f"Dataset {name} not supported.")



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
        return sentences[0].strip() if sentences else None
    def extract_tail(text):
        sentences = re.split(r'(?<=[.?!])\s+|\n', text.strip())  # Split by sentence-ending punctuation and also new line
        return ' '.join(sentences[-2:]).strip() if sentences else None

    # def extract_head_and_tail(text):
    #     sentences = re.split(r'(?<=[.?!])\s+|\n', text.strip())  # Split by sentence-ending punctuation and also new line
    #     return sentences[0], sentences[-1] if sentences else None
    # head, tail = extract_head_and_tail(new_resp)
    head, tail = extract_head(new_resp), extract_tail(new_resp)
    if debug:
        print(f"Intitial head and tail: \nHead: {head}\nTail: {tail}")

    
    # Step2.5/2.6 (*Answer*)
    def extract_after_Answer(text):
        match = re.search(r'(\*?\*?(Answer|Conclusion)\*?\*?)[:：]?\s*([\s\S]+?)([.?!]\s|$)', text, re.IGNORECASE)
        return re.sub(r"[\*\n]", "", match.group(3).strip()).strip() if match else text 

    head = extract_after_Answer(head)
    tail = extract_after_Answer(tail)
    
    
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

    # TODO: add a check here
    


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
    # if confidence < thre:
    #     # print(f"Confidence: {confidence:.2f}\nTail: {tail}\nOptions: {opts}")
    #     return -1
    # elif confidence >= 0.9 and confidence <= 0.95:
    #     print(f"Confidence: {confidence:.2f}\nTail: {tail}\nOptions: {opts}\nBest Match: {output}")
    return output



import pickle
import glob
import os

def load_latest_response(prefix, num=1, debug=True):
    files = sorted(glob.glob(f"{prefix}*.pkl"), key=os.path.getmtime, reverse=True)
    
    if not files:
        print("No matching files found.")
        return None

    latest_file = files[:num]
    loaded = []
    for file in latest_file:
        if debug:
            print(f"Loading: {file}")

        with open(file, 'rb') as f:
            loaded.append(pickle.load(f))
    
    return (loaded, latest_file)

def clean_responses(cot_trigger_responses, mode='pixtral', setting='cot_trigger'):
    cot_trigger_responses = [text.split('assistant')[-1].strip() for text in cot_trigger_responses]
    if setting == 'cot_trigger' and len(cot_trigger_responses[0]) < 20:
        print('Warning, you may use the wrong setting')
    if mode == "pixtral":
        if setting == 'cot_trigger':
            cot_trigger_responses = [text.split('**Answer:**.)')[-1].strip() for text in cot_trigger_responses]
        else:
            cot_trigger_responses = [text.split(", no explanation.")[-1].strip() for text in cot_trigger_responses]
    elif 'gemma' in mode:
        cot_trigger_responses = [text.split('\nmodel\n')[-1].strip() for text in cot_trigger_responses]

    return cot_trigger_responses


def extract_all_answer_idxs(responses, ds, option_s, debug=False, mode='sim', thre=0.8):
    ds_options = ds[option_s]
    lst_AE = [Answer_Extraction_Pipeline(resp, opts, debug, mode, thre) for resp, opts in zip(responses, ds_options)]
    return lst_AE


def cal_acc(lst_AE, ds, ds_name, option_s):
                
    if ds_name in ["MathVista", "VLQA"]:
        answers = [ds[option_s][i][idx] if idx != -1 else -1 for i, idx in enumerate(lst_AE) ]
        match_list = np.array(answers) == np.array(ds['answer'])  
    elif ds_name in ["MathVision", "MMMU", "MMStar"]:
        chara_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        answers = [chara_list[idx] if idx != -1 else -1 for idx in lst_AE]
        match_list = np.array(answers) == np.array(ds['answer'])      
    else:
        match_list = np.array(lst_AE) == np.array(ds['answer'])
    
    corr_indices = np.where(match_list)[0]
    wrg_indices = np.where(~match_list)[0]
    return round(np.mean(match_list), 4), match_list, corr_indices, wrg_indices

def responses_post_processing(responses, ds, ds_name):
    # Although answer extraction and indices extraction have been done in training, since Answer Extraction Pipeline might change
    # it's necessary to do post processing again.
    # > 1. First extract all answer indices, then calculate the accuracy
    # > 2. Using the correct / wrong indices, split the responses
    option_s = dataset_option(ds_name)
    # Step 1
    lst_AE = extract_all_answer_idxs(responses, ds, option_s, debug=False, mode='sim', thre=0.9)
    acc, match_list, corr_indices, wrg_indices = cal_acc(lst_AE, ds, ds_name, option_s)

    # Step 2
    corr_responses = [responses[i] for i in corr_indices]
    wrg_responses = [responses[i] for i in wrg_indices]

    return lst_AE, acc, match_list, corr_indices, wrg_indices, corr_responses, wrg_responses

def dataset_option(ds_name):
    if ds_name in ["MathVista", "ScienceQA"]:
        return 'choices'
    else:
        return 'options'

def postprocess_responses(ds, DATASET_NAME, MODEL_NAME, cot_responses, file_names, force_reprocess=False):
    lst_AE_all = []
    match_list_all = []
    for i, cot in enumerate(cot_responses):
        # print(f"Processing {MODEL_NAME} {file_names[i]} ...")
        if os.path.exists(f"data/n_vs_one/{DATASET_NAME}/{MODEL_NAME}/PostProcessing/postprocess_{file_names[i]}.json") and not force_reprocess:
            with open(f"data/n_vs_one/{DATASET_NAME}/{MODEL_NAME}/PostProcessing/postprocess_{file_names[i]}.json", "r", encoding="utf-8") as f:
                data_json = f.read()
            data = json.loads(data_json)
            lst_AE_all.append(data['lst_AE'])
            match_list_all.append(data['match_list'])
            continue
        lst_AE, acc, match_list, corr_indices, wrg_indices, corr_responses, wrg_responses = responses_post_processing(cot, ds, DATASET_NAME)
        # Store each processed data as a JSON string (for later analysis or debugging)
        # Convert numpy types to native Python types for JSON serialization
        def to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            return obj

        data_json = json.dumps({
            'lst_AE': [to_serializable(x) for x in lst_AE],
            'acc': to_serializable(acc),
            'match_list': to_serializable(match_list),
            'corr_indices': to_serializable(corr_indices),
            'wrg_indices': to_serializable(wrg_indices),
            'corr_responses': corr_responses,
            'wrg_responses': wrg_responses
        }, ensure_ascii=False)
        # Save each JSON string to a file in the specified directory
        os.makedirs(f"data/n_vs_one/{DATASET_NAME}/{MODEL_NAME}/PostProcessing", exist_ok=True)
        with open(f"data/n_vs_one/{DATASET_NAME}/{MODEL_NAME}/PostProcessing/postprocess_{file_names[i]}.json", "w", encoding="utf-8") as f:
            f.write(data_json)
        lst_AE_all.append(lst_AE)
        match_list_all.append(match_list)

    return lst_AE_all, match_list_all