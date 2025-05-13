from transformers import (
    AutoTokenizer,
    AutoConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
)
import torch

model_path = "meta-llama/Meta-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
)
model = model.eval()

prompt = "I am a boy"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")


outputs = model(
  input_ids,
  past_key_values=None,
  use_cache=True,
)

breakpoint()

print("done")