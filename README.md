# Budget Guidance


## Installation

```python
conda create -n bg python=3.10
conda activate bg
pip install torch
pip install flash-attn --no-build-isolation
cd 3rdparty/transformers && pip install -e .
# For training
cd training && pip install -e .
cd 3rdparty/trl && pip install -e .
# For evaluation
cd evaluation/lm-evaluation-harness && pip install -e .[math,vllm]
```


## Training

First, apply the data augmentation technique mentioned in the paper:

```python
cd training
python run_data_augmentation.py
```

Then, train the predictor:

```python
bash train.sh
```

## Evaluation

We use [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness) as our evaluation codebase. Also, we use Azure OpenAI API to provide LLM as the evaluation judge. For example, to evaluate DeepSeek-R1-Distill-Qwen-7B model on MATH-500 with a thinking budget of 1000, simple run:

```python
cd evaluation
export MODEL_PATH=senfu/DeepSeek-R1-Distill-Qwen-7B-BG
export TOKENIZER=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
export THINKING_BUDGET=1000
export API_KEY_NAME=YOUR_AZURE_OPENAI_API
export API_ENDPOINT=YOUR_AZURE_API_ENDPOINT
export PROCESSOR=gpt-4o-mini
accelerate launch -m lm_eval --model hf --model_args pretrained=$MODEL_PATH,tokenizer=$TOKENIZER,dtype=bfloat16 --seed 0 --tasks openai_math --batch_size 1 --apply_chat_template --output_path results --log_samples --gen_kwargs "max_gen_toks=32768,token_budget=$THINKING_BUDGET"
```

## Acknowledgement

[s1](https://github.com/simplescaling/s1): We adapt their codebase for evaluation.

[open-r1](https://github.com/huggingface/open-r1) We adapt their codebase for training.

## Citation

If our work is useful or relevant to your research, please kindly recognize our contributions by citing our paper:

```
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
