from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

model_path = "meta-llama/Meta-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="flash_attention_2"
)
device = model.hf_device_map.get("model.embed_tokens", "cuda:0")
model.eval()

bsz = 1
seq_len = 16384
input_ids = torch.zeros((bsz, seq_len), dtype=torch.int64).to(device)
start_time = time.time()
torch.cuda.synchronize()
#warmup
outputs = model.generate(
    input_ids,
    max_new_tokens=1,
    past_key_values=None,
    use_cache=True,
)
torch.cuda.synchronize()
end_time = time.time()
print(f"warmup prefill:{end_time - start_time}s, bsz:{bsz}, seqlen:{seq_len}")

bsz = 1
seq_len = 16384
input_ids = torch.zeros((bsz, seq_len), dtype=torch.int64).to(device)
start_time = time.time()
torch.cuda.synchronize()
#warmup
outputs = model.generate(
    input_ids,
    max_new_tokens=1,
    past_key_values=None,
    use_cache=True,
)
torch.cuda.synchronize()
end_time = time.time()
print(f"prefill:{end_time - start_time}s, bsz:{bsz}, seqlen:{seq_len}")
bsz = 2
seq_len = 16384
input_ids = torch.zeros((bsz, seq_len), dtype=torch.int64).to(device)
start_time = time.time()
torch.cuda.synchronize()
#warmup
outputs = model.generate(
    input_ids,
    max_new_tokens=1,
    past_key_values=None,
    use_cache=True,
)
torch.cuda.synchronize()
end_time = time.time()
print(f"prefill:{end_time - start_time}s, bsz:{bsz}, seqlen:{seq_len}")
bsz = 4
seq_len = 16384
input_ids = torch.zeros((bsz, seq_len), dtype=torch.int64).to(device)
start_time = time.time()
torch.cuda.synchronize()
#warmup
outputs = model.generate(
    input_ids,
    max_new_tokens=1,
    past_key_values=None,
    use_cache=True,
)
torch.cuda.synchronize()
end_time = time.time()
print(f"prefill:{end_time - start_time}s, bsz:{bsz}, seqlen:{seq_len}")
bsz = 8
seq_len = 16384
input_ids = torch.zeros((bsz, seq_len), dtype=torch.int64).to(device)
start_time = time.time()
torch.cuda.synchronize()
#warmup
outputs = model.generate(
    input_ids,
    max_new_tokens=1,
    past_key_values=None,
    use_cache=True,
)
torch.cuda.synchronize()
end_time = time.time()
print(f"prefill:{end_time - start_time}s, bsz:{bsz}, seqlen:{seq_len}")


print("done")
