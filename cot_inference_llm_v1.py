import argparse
import sys

from tqdm import tqdm
from dashscope import Generation
import dashscope
dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1/"
import random
import time
import pickle
from datasets import load_dataset
import os


def option_recode(options: list, type):
    if type == 'number':
        return "\n".join([f'[{i}] {op}' for i, op in enumerate(options)])
    elif type == 'character':
        chara_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
        return "\n".join([f'[{ch}] {op}' for ch, op in zip(chara_list[:len(options)], options)])
    elif type == 'nidx':
        return "\n".join([f'{op}' for i, op in enumerate(options)])
    else:
        raise ValueError("Invalid type")


def run_once(run_id: int = 0, start: int = 0, limit: int = 50, model: str = None, output_root: str = "output"):
    # ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    # ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test").filter(lambda x: x['category'] == 'math')
    # ds_name = "ARC-Easy"
    ds_name = "ARC-Challenge"
    # ds_name = "MMLU-Pro-Math"
    
    option_s = "options" if "options" in ds.column_names else "choices"

    model_list = [
        "qwen3-235b-a22b-thinking-2507",
        "qwen3-30b-a3b-thinking-2507",
        "qwen3-4b",
    ]
    if model is None:
        model = model_list[0]

    setting = "Among the given options, the answer is: (Let's think step by step, and give the answer at the end of your thought with **Answer:**.) "

    responses = []
    end = min(start + limit, len(ds))
    for i in tqdm(range(start, end), desc=f"Run {run_id} Processing"):
        if i == 1:
            print("Processing second sample")
        sample = ds[i]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Question: {sample['question']}" + "\nOptions:\n" + option_recode(
                            sample[option_s], 'character'
                        )
                        + "\n\n" + setting,
                    },
                ],
            },
        ]
        seed = random.randint(0, 2**32 - 1)
        completion = Generation.call(
            api_key="your-api-key",
            model=model,
            messages=messages,
            result_format="message",
            enable_thinking=True,
            stream=True,
            incremental_output=True,
            generation={
                "temperature": 0.8,
                "top_k": 50,
                "top_p": 0.95,
                "max_new_tokens": 1024,
                "seed": seed,
            },
        )

        reply = ""
        try:
            for chunk in completion:
                if chunk.output.choices[0].message.content != "":
                    reply += chunk.output.choices[0].message.content
        except Exception as e:
            print(f"Error during generation: {e}")
        responses.append(reply)
        if i % 150 == 0 and i > start:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(output_root, f"{model}", f"run_{run_id}")
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(
                output_dir,
                f"responses_{model}_{ds_name}_run{run_id}_{timestamp}_{start}_{i}.pkl",
            )
            with open(out_path, "wb") as f:
                pickle.dump(responses, f)
            print(f"Run {run_id} running , saved to {out_path}")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_root, f"{model}", f"run_{run_id}")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir,
        f"responses_{model}_{ds_name}_run{run_id}_{timestamp}_{start+1}_{i+1}.pkl",
    )
    with open(out_path, "wb") as f:
        pickle.dump(responses, f)
    print(f"Run {run_id} finished, saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, default=0, help="ID of this run (used in output path)")
    parser.add_argument("--start", type=int, default=0, help="start index in dataset")
    parser.add_argument("--limit", type=int, default=50, help="number of samples to process")
    parser.add_argument("--model", type=str, default=None, help="model name override")
    parser.add_argument("--outdir", type=str, default="output", help="output root dir")
    args = parser.parse_args()

    try:
        run_once(
            run_id=args.run_id,
            start=args.start,
            limit=args.limit,
            model=args.model,
            output_root=args.outdir,
        )
    except Exception as e:
        print(f"Run {args.run_id} failed: {e}", file=sys.stderr)
        raise
