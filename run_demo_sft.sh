#!/bin/bash

# 要执行的 Python 脚本路径
APP_HOME=$(dirname "$0")
cd "$APP_HOME"
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python gradio_demo.py \
    --model_type auto \
    --base_model bigscience/bloomz-560m \
    --lora_model outputs-sft-v1/checkpoint-250 \
    --cache_dir ./cache \
    --share 
    