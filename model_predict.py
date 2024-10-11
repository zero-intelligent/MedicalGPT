from types import SimpleNamespace
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
)

from template import get_conv_template

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, AutoTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}


args = SimpleNamespace(
    model_type = "auto",
    base_model = "Qwen/Qwen2.5-7B",
    lora_model = "outputs-sft-qwen-v1/checkpoint-250",
    tokenizer_path = "",
    template_name = "qwen",
    system_prompt = "你是一个专业的宠物医生,帮助解答各类宠物疾病问题,尽可能全面准确,字数在30到500之间。",
    only_cpu = "",
    resize_emb = "",
    cache_dir = "./cache",
)

print(args)
load_type = 'auto'
if torch.cuda.is_available() and not args.only_cpu:
    device = torch.device(0)
else:
    device = torch.device('cpu')

if not args.tokenizer_path:
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
model.to(device)
model.eval()
prompt_template = get_conv_template(args.template_name)
system_prompt = args.system_prompt
stop_str = tokenizer.eos_token if tokenizer.eos_token else prompt_template.stop_str

def predict(message, history=[]):
    """Generate complete answer synchronously from the prompt with GPT"""
    # 合并历史对话和当前消息，构造完整的对话上下文
    history_messages = history + [[message, ""]]
    prompt = prompt_template.get_prompt(messages=history_messages, system_prompt=system_prompt)

    # Tokenize 输入，并返回 attention_mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    context_len = 2048
    max_new_tokens = 512

    # 设置生成参数，添加 pad_token_id 和 attention_mask
    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,  # 传递 attention_mask
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.eos_token_id  # 设置 pad_token_id
    )

    # 同步生成完整的文本
    output_ids = model.generate(**generation_kwargs)

    # 解码输出的token，并去除特殊字符
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    generated_text = generated_text.split('assistant\n')[-1]
    
    print(f"question:{message},history={history}\nanswer:{generated_text}")
    return generated_text