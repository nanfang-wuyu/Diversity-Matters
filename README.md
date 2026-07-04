# Diversity Matters: Revisiting Test-Time Compute in Vision-Language Models

Code release for our ICML 2026 paper on test-time compute (TTC) for vision-language models (VLMs).

[ICML 2026 poster page](https://icml.cc/virtual/2026/poster/63569)

> **Paper status:** Accepted to ICML 2026  
> **Authors:** Yijie Tong, Yifan Hou, Shaobo Cui, Antoine Bosselut, Mrinmaya Sachan  
> **Primary area:** Deep Learning / Foundation Models  
> **Keywords:** vision-language model, test-time compute, chain-of-thought, multi-model ensemble, reasoning

## Overview

Test-time compute strategies have become a lightweight way to improve reasoning in large language models by spending more inference-time computation. This repository contains the scripts and notebooks used in our systematic study of whether such strategies transfer to visual reasoning with VLMs.

We study two broad TTC paradigms:

1. **Feature-based scoring of chain-of-thought traces**, where generated rationales are ranked or selected using cues such as length or pivot words.
2. **Confidence-based aggregation**, where multiple predictions are combined through voting or confidence estimates.

Our main finding is that repeated sampling from a single VLM provides limited benefit for visual reasoning because the samples tend to share the same visual interpretation errors. In contrast, multi-model ensembles provide stronger diversity through architectural, training-data, and scale differences. We therefore propose **Entropy-based Test-Time Compute (ETTC)**, which selects predictions using predictive entropy rather than treating every model vote equally.

ETTC reduces to majority voting in the single-model setting, but in multi-model ensembles it can prioritize more confident models and avoid being dominated by correlated errors from weaker models.

## Paper

**Diversity Matters: Revisiting Test-Time Compute in Vision-Language Models**  
Yijie Tong, Yifan Hou, Shaobo Cui, Antoine Bosselut, Mrinmaya Sachan  
International Conference on Machine Learning (ICML), 2026

## Abstract

Test-time compute (TTC) strategies have emerged as a lightweight approach to boost reasoning in large language models, but their applicability to vision-language models (VLMs) remains unclear. We present a systematic study of TTC for visual reasoning across seven open-source VLMs and six benchmarks, revisiting two paradigms: (i) feature-based scoring of chain-of-thought (CoT) traces and (ii) confidence-based aggregation via majority voting (MV). In the single-model setting, feature cues (e.g., length, pivot words) fail to improve accuracy, while MV yields only modest, CoT-dependent gains. To explain this limitation, we theoretically show that the voting method's effectiveness depends on prediction diversity: when outputs are highly correlated, the benefit of voting vanishes. In contrast, multi-model ensembles introduce stronger diversity through architectural differences, training data, and scale, making them both more realistic and more promising for TTC. However, MV treats all models equally, leaving it vulnerable to correlated errors from weaker models. To address this, we propose Entropy-based TTC, which selects the most confident prediction based on predictive entropy. Our method reduces to MV in the single-model case but, in ensembles, leverages confidence disparities to prioritize stronger models. We prove that our method theoretically outperforms MV under mild dependence assumptions, and empirically show that it consistently surpasses both MV and the best individual model across diverse visual reasoning benchmarks. This demonstrates that smaller models can enhance, rather than hinder, larger ones when combined appropriately, unlocking ensemble gains not achievable with existing TTC strategies.

## Lay Summary

When text-based AI models are given multiple chances to answer a question and vote on the final result, their accuracy often improves dramatically. This work investigates whether the same idea works for vision-language models: AI systems that must reason over both images and text.

We find that asking a single visual AI model to answer the same question multiple times barely helps. If the model misunderstands the image, its repeated answers usually share the same visual mistake. Real improvement requires diversity, which is better achieved by asking several different models.

However, simple majority voting can fail when a group of weaker models outvotes a stronger one. ETTC addresses this by estimating how confident each model's prediction is and selecting the answer with the lowest predictive entropy. This lets smaller and cheaper models assist larger models without dragging down the final ensemble.

## Repository Structure

| File | Description |
| --- | --- |
| `cot_inference_final_v1.py` | VLM inference script for generating multiple CoT or direct-answer samples on visual reasoning benchmarks. |
| `cot_inference_llm_v1.py` | LLM inference script for text-only multiple-choice reasoning experiments. |
| `utils.py` | Shared utilities for loading models, loading datasets, extracting answers, post-processing responses, and computing accuracy. |
| `final_comparison_V2.ipynb` | Main analysis notebook for comparing single-model and multi-model TTC strategies. |
| `final_comparison_V5.ipynb` | Focused analysis notebook for the LLM + entropy strategy. |
| `final_correlation.ipynb` | Correlation analysis notebook studying prediction dependence across repeated samples and models. |

## Method Summary

The experiments follow this high-level workflow:

1. **Generate predictions** from VLMs or LLMs under direct-answer and CoT prompting.
2. **Extract answer choices** from free-form model responses using post-processing utilities.
3. **Estimate per-question uncertainty** from repeated predictions.
4. **Compare TTC strategies**, including individual model accuracy, repeated-sampling majority voting, multi-model majority voting, and entropy-based selection.
5. **Analyze diversity and correlation** to explain when voting helps or fails.

## Supported Models and Benchmarks

The scripts include support for multiple open-source VLM families, including Llama-Vision, Molmo, Pixtral, Gemma, Qwen2.5-VL, and InternVL-style models. Dataset-loading utilities cover several visual reasoning benchmarks, including MathVista, AI2D, ScienceQA, MathVision, MMMU, MMStar, ChartQA-style tasks, and related multiple-choice VQA datasets.

Some datasets and model checkpoints require separate access through Hugging Face or the original dataset providers. Please make sure you comply with each dataset and model license.

## Environment Setup

We recommend using a fresh Python environment with GPU-enabled PyTorch.

```bash
conda create -n ettc python=3.10
conda activate ettc
```

Install the core dependencies:

```bash
pip install torch transformers datasets accelerate bitsandbytes
pip install pillow matplotlib numpy scipy pandas scikit-learn tqdm
pip install python-dotenv huggingface-hub python-Levenshtein sentence-transformers
pip install qwen-vl-utils dashscope jupyter
```

Depending on the selected model, you may need additional model-specific dependencies such as `flash-attn`, `torchvision`, or remote-code support from Hugging Face.

## Running Inference

### VLM inference

`cot_inference_final_v1.py` reads experiment settings from `config.json` if present and also exposes command-line arguments. A typical run is:

```bash
python cot_inference_final_v1.py \
  --dataset_name MathVista \
  --model_name qwen_7b \
  --batch_size 1 \
  --max_new_tokens 1024 \
  --use_cuda \
  --setting cot_trigger \
  --loops 16
```

Useful settings include:

- `--setting ncot_force`: direct answer without explanation.
- `--setting zero_shot`: zero-shot answer prompt.
- `--setting cot_trigger`: chain-of-thought prompt with an explicit final answer marker.
- `--loops`: number of repeated sampling rounds.

Outputs are saved under:

```text
output/<model_name>/responses_<model_name>_<dataset_name>_<setting>_<timestamp>.pkl
```

### LLM inference

`cot_inference_llm_v1.py` runs text-only experiments with DashScope-compatible Qwen models:

```bash
python cot_inference_llm_v1.py \
  --run-id 0 \
  --start 0 \
  --limit 100 \
  --model qwen3-235b-a22b-thinking-2507 \
  --outdir output
```

For public use, store API credentials in environment variables and avoid committing private keys.

## Analysis

After collecting response files, use the notebooks to reproduce the analysis:

1. Open `final_comparison_V2.ipynb` for the main TTC comparison.
2. Use `final_correlation.ipynb` to inspect prediction correlation and diversity.
3. Use `final_comparison_V5.ipynb` for the focused LLM entropy analysis.

The notebooks assume that generated response files and post-processed JSON files are available in the expected local output directories. If you reorganize outputs, update the corresponding paths in the notebooks.

## Reproducibility Notes

- Many experiments rely on stochastic decoding, so exact numbers may vary across hardware, package versions, and random seeds.
- Large VLMs may require substantial GPU memory; the scripts use 4-bit quantization for several models.
- Some checkpoints require Hugging Face authentication or acceptance of model-specific license terms.
- Some notebooks were used as research analysis workbooks and may contain local paths or cached intermediate results.
- Generated model responses are not included in this lightweight code release.

## Citation

If this repository is useful for your research, please cite our ICML 2026 paper. The BibTeX entry will be added after the official proceedings entry is available.

```bibtex
@inproceedings{tong2026diversity,
  title     = {Diversity Matters: Revisiting Test-Time Compute in Vision-Language Models},
  author    = {Tong, Yijie and Hou, Yifan and Cui, Shaobo and Bosselut, Antoine and Sachan, Mrinmaya},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```

## License

Please check the repository license before using this code. Dataset and model licenses are governed by their original providers.

## Contact

For questions about the paper or code release, please open a GitHub issue or contact the authors listed on the ICML paper page.
