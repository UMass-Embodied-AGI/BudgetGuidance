export WANDB_API_KEY=7a637fdf546ac2d41295c4c256f5fab6b7d1b647
export WANDB_PROJECT=ttc

accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/sft.py \
    --config recipes/Qwen3-30B-A3B.yaml
