MODEL_PATH=/proj/inf-scaling/efficient_long_ctx/test-time/BudgetGuidance/training/output/Qwen3-30B-A3B-fused
TOKENIZER=Qwen/Qwen3-30B-A3B
THINKING_BUDGET=4000
accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=$MODEL_PATH,tokenizer=$TOKENIZER,dtype=bfloat16,attn_implementation=flash_attention_2 \
    --seed 0 \
    --tasks openai_math \
    --batch_size 1 \
    --apply_chat_template \
    --output_path moe_openai_math \
    --log_samples \
    --predict_only \
    --gen_kwargs "max_gen_toks=8192,token_budget=$THINKING_BUDGET"
