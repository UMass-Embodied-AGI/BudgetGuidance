lm_eval \
    --model vllm \
    --model_args pretrained=Qwen/Qwen3-30B-A3B,tokenizer=Qwen/Qwen3-30B-A3B,dtype=bfloat16,max_model_len=32000 \
    --seed 0 \
    --tasks openai_math \
    --batch_size auto \
    --apply_chat_template \
    --output_path bf1124 \
    --log_samples \
    --predict_only \
    --gen_kwargs "max_gen_toks=30000,max_tokens_thinking=1700,thinking_start=<think>,thinking_end=</think>,until_thinking=</think>"


# lm_eval \
#     --model vllm \
#     --model_args pretrained=Qwen/Qwen3-30B-A3B,tokenizer=Qwen/Qwen3-30B-A3B,dtype=bfloat16,max_model_len=32000 \
#     --seed 0 \
#     --tasks openai_math \
#     --batch_size auto \
#     --apply_chat_template \
#     --output_path bf \
#     --log_samples \
#     --predict_only \
#     --gen_kwargs "max_gen_toks=30000,max_tokens_thinking=3000,thinking_start=<think>,thinking_end=</think>,until_thinking=</think>"



# lm_eval \
#     --model vllm \
#     --model_args pretrained=Qwen/Qwen3-30B-A3B,tokenizer=Qwen/Qwen3-30B-A3B,dtype=bfloat16,max_model_len=32000 \
#     --seed 0 \
#     --tasks aime24_nofigures \
#     --batch_size auto \
#     --apply_chat_template \
#     --output_path bf \
#     --log_samples \
#     --predict_only \
#     --gen_kwargs "max_gen_toks=30000,max_tokens_thinking=5000,thinking_start=<think>,thinking_end=</think>,until_thinking=</think>"


# lm_eval \
#     --model vllm \
#     --model_args pretrained=Qwen/Qwen3-30B-A3B,tokenizer=Qwen/Qwen3-30B-A3B,dtype=bfloat16,max_model_len=32000 \
#     --seed 0 \
#     --tasks amc \
#     --batch_size auto \
#     --apply_chat_template \
#     --output_path baseline \
#     --log_samples \
#     --predict_only \
#     --gen_kwargs "max_gen_toks=30000,max_tokens_thinking=2000,thinking_start=<think>,thinking_end=</think>,until_thinking=</think>"


# lm_eval \
#     --model vllm \
#     --model_args pretrained=Qwen/Qwen3-30B-A3B,tokenizer=Qwen/Qwen3-30B-A3B,dtype=bfloat16,max_model_len=32000 \
#     --seed 0 \
#     --tasks olympiadbench_math \
#     --batch_size auto \
#     --apply_chat_template \
#     --output_path baseline \
#     --log_samples \
#     --predict_only \
#     --gen_kwargs "max_gen_toks=30000,max_tokens_thinking=2000,thinking_start=<think>,thinking_end=</think>,until_thinking=</think>"
