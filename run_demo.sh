CUDA_VISIBLE_DEVICES=0 python gradio_demo.py \
    --model_type auto \
    --base_model Qwen/Qwen2.5-7B \
    --lora_model outputs-pt-qwen-v1/checkpoint-1250 \
    --cache_dir ./cache \
    --share 
    