MODEL_PATH=/proj/inf-scaling/efficient_long_ctx/test-time/BudgetGuidance/training/output/DS-7B-code
TOKENIZER=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

for THINKING_BUDGET in 1000 1500 2000 2500 3000 4000; do
    echo "Running evaluation with THINKING_BUDGET=$THINKING_BUDGET"
    accelerate launch -m lm_eval \
        --model hf \
        --model_args pretrained=$MODEL_PATH,tokenizer=$TOKENIZER,dtype=bfloat16,attn_implementation=flash_attention_2 \
        --seed 0 \
        --tasks openai_math \
        --batch_size 1 \
        --apply_chat_template \
        --output_path ds7b_code_math_budget_$THINKING_BUDGET \
        --log_samples \
        --predict_only \
        --gen_kwargs "max_gen_toks=8192,token_budget=$THINKING_BUDGET"
done
