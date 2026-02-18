#!/bin/bash

# Generate evaluation commands with varying max_tokens_thinking values
for max_tokens_thinking in 4500 5000 5500 6000 6500 7000; do
    output_dir="bf1122/bf_thinking_${max_tokens_thinking}"
    
    echo "Running evaluation with max_tokens_thinking=${max_tokens_thinking}"
    echo "Output directory: ${output_dir}"
    
    lm_eval \
        --model vllm \
        --model_args pretrained=Qwen/Qwen3-30B-A3B,tokenizer=Qwen/Qwen3-30B-A3B,dtype=bfloat16,max_model_len=32000 \
        --seed 0 \
        --tasks openai_math \
        --batch_size auto \
        --apply_chat_template \
        --output_path "${output_dir}" \
        --log_samples \
        --predict_only \
        --gen_kwargs "max_gen_toks=30000,max_tokens_thinking=${max_tokens_thinking},thinking_start=<think>,thinking_end=</think>,until_thinking=</think>"
    
    echo "Completed: ${output_dir}"
    echo "---"
done
