# Steering LLM Thinking with Budget Guidance

[[Demo](YOUR_DEMO_LINK)] [[Paper](https://arxiv.org/abs/2506.xxxxx)] [[Hugging Face Models](https://huggingface.co/collections/senfu/budget-guidance-6844426427e777c8bc04a5ce)]


![method](figures/method.jpg)

This repository contains the official code for **Budget Guidance**, a lightweight and non-invasive method for controlling the reasoning length of large language models (LLMs). It enables **budget-conditioned** generation without fine-tuning the LLM, and achieves strong performance across a wide range of reasoning benchmarks.




## Table of Contents

- [News](#news)
- [Installation](#installation)
- [Model Checkpoints](#model-checkpoints)
- [Training](#training)
  - [Data Augmentation](#data-augmentation)
  - [Train the Predictor](#train-the-predictor)
- [Evaluation](#evaluation)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)
- [License](#license)
- [Contributing](#contributing)


## News

* June 2025: Code and model checkpoints released.
* June 2025: Paper released on arXiv.


## Installation

```bash
# Create environment
conda create -n bg python=3.10
conda activate bg

# Install dependencies
pip install torch
pip install flash-attn --no-build-isolation

# Install modified transformers
cd 3rdparty/transformers && pip install -e .

# For training
cd training && pip install -e .
cd 3rdparty/trl && pip install -e .

# For evaluation
cd evaluation/lm-evaluation-harness && pip install -e .[math,vllm]
```

## Model Checkpoints

| Model | Link |
|-------|------|
| DeepSeek-R1-Distill-Qwen-7B | [🤗 Hugging Face](https://huggingface.co/senfu/DeepSeek-R1-Distill-Qwen-7B-BG) |
| DeepSeek-R1-Distill-Qwen-32B | [🤗 Hugging Face](https://huggingface.co/senfu/DeepSeek-R1-Distill-Qwen-32B-BG) |
| Qwen3-8B-BG | [🤗 Hugging Face](https://huggingface.co/senfu/Qwen3-8B-BG) |


## Training

### Data Augmentation

First, apply the data augmentation technique described in our paper:

```bash
cd training
python run_data_augmentation.py
```

### Train the Predictor
Then, start training:

```bash
bash train.sh
```


## Evaluation

We use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) as the evaluation framework.  
For evaluating reasoning quality under a thinking budget, we employ an external LLM (e.g., Azure OpenAI GPT-4o-mini) as the judge.

Example: to evaluate **DeepSeek-R1-Distill-Qwen-7B** on **MATH-500** with a thinking budget of 1000 tokens:

```bash
cd evaluation
export MODEL_PATH=senfu/DeepSeek-R1-Distill-Qwen-7B-BG
export TOKENIZER=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
export THINKING_BUDGET=1000

# Azure OpenAI API setup
export API_KEY_NAME=YOUR_AZURE_OPENAI_API
export API_ENDPOINT=YOUR_AZURE_API_ENDPOINT
export PROCESSOR=gpt-4o-mini

# Run evaluation
accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=$MODEL_PATH,tokenizer=$TOKENIZER,dtype=bfloat16 \
    --seed 0 \
    --tasks openai_math \
    --batch_size 1 \
    --apply_chat_template \
    --output_path results \
    --log_samples \
    --gen_kwargs "max_gen_toks=32768,token_budget=$THINKING_BUDGET"
```


## Acknowledgement

We gratefully acknowledge the following open-source projects:

- [s1](https://github.com/simplescaling/s1): Evaluation codebase adaptation.
- [open-r1](https://github.com/huggingface/open-r1): Training codebase adaptation.


## Citation

If you find our work helpful, please consider citing:

```bibtex
@misc{li2025budgetguidance,
  title        = {Steering LLM Thinking with Budget Guidance},
  author       = {Junyan Li and Wenshuo Zhao and Yang Zhang and Chuang Gan},
  year         = {2025},
  eprint       = {2506.xxxxx},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2506.xxxxx}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.


## Contributing

We welcome contributions to Budget Guidance!  
If you have suggestions, bug reports, or would like to contribute improvements, feel free to open an issue or submit a pull request.

