# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

pip install gradio>=3.50.2
"""
import argparse
from threading import Thread

import gradio as gr
import torch
from peft import PeftModel
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaForCausalLM,
    GenerationConfig,
    TextIteratorStreamer,
)

from template import get_conv_template

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, AutoTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='auto', type=str)
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_model', default="", type=str, help="If None, perform inference on the base model")
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--template_name', default="qwen", type=str,
                        help="Prompt template name, eg: alpaca, vicuna, baichuan2, chatglm2 etc.")
    parser.add_argument('--system_prompt', default="你是一个专业的宠物医生,帮助解答各类宠物疾病问题,尽可能全面准确,字数在30到500之间。", type=str)
    parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    parser.add_argument('--share', action='store_true', help='Share gradio')
    parser.add_argument('--cache_dir', default='cache', help='cache_dir')
    parser.add_argument('--port', default=8082, type=int, help='Port of gradio demo')
    args = parser.parse_args()
    print(args)
    load_type = 'auto'
    if torch.cuda.is_available() and not args.only_cpu:
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path, trust_remote_code=True,cache_dir=args.cache_dir)
    base_model = model_class.from_pretrained(
        args.base_model,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        trust_remote_code=True,
        cache_dir=args.cache_dir
    )
    try:
        base_model.generation_config = GenerationConfig.from_pretrained(args.base_model, trust_remote_code=True)
    except OSError:
        print("Failed to load generation config, use default.")
    if args.resize_emb:
        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(tokenizer)
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
        if model_vocab_size != tokenzier_vocab_size:
            print("Resize model embeddings to fit tokenizer")
            base_model.resize_token_embeddings(tokenzier_vocab_size)
    if args.lora_model:
        model = PeftModel.from_pretrained(base_model, args.lora_model, torch_dtype=load_type, device_map='auto')
        print("loaded lora model")
    else:
        model = base_model
    if device == torch.device('cpu'):
        model.float()
    model.eval()
    prompt_template = get_conv_template(args.template_name)
    system_prompt = args.system_prompt
    stop_str = tokenizer.eos_token if tokenizer.eos_token else prompt_template.stop_str

    def predict(message, history):
        """Generate answer from prompt with GPT and stream the output"""
        history_messages = history + [[message, ""]]
        prompt = prompt_template.get_prompt(messages=history_messages, system_prompt=system_prompt)
        streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        input_ids = tokenizer(prompt).input_ids
        context_len = 2048
        max_new_tokens = 512
        max_src_len = context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]
        generation_kwargs = dict(
            input_ids=torch.as_tensor([input_ids]).to(device),
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            num_beams=1,
            repetition_penalty=1.0,
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        partial_message = ""
        for new_token in streamer:
            if new_token != stop_str:
                partial_message += new_token
                yield partial_message
    CSS = """
        .gradio-container { 
            height: 100vh !important; 
            display: flex; 
            flex-direction: column; 
        }
        #component-0 {
            height: 100% !important; 
            display: flex; 
            flex-direction: column;
        }
        .wrap {
            flex-grow: 1; 
            display: flex; 
            flex-direction: column; 
        }
        #chatbot {
            flex-grow: 1; 
            overflow-y: auto; 
            max-height: calc(100vh - 180px);  /* Ensure the chat area can scroll */
        }
        #textbox {
            width: 100% !important; 
        }
    """
    
    gr.ChatInterface(
        predict,
        chatbot=gr.Chatbot(),
        textbox=gr.Textbox(placeholder="Ask me question", lines=4, scale=9),
        title="医宠 (PetMedialGPT)",
        description="输入宠物的症状描述，医宠大模型为您解答",
        theme="soft",
        css = CSS
    ).queue().launch(share=args.share, inbrowser=True, server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    main()
