import torch
from datasets import load_dataset
import os, math
import importlib
import yaml
from torch.utils.data import TensorDataset
from tqdm import tqdm
import json
from MagicDec.Data.preprocess_longbench import preprocess_longbenchv2, preprocess_longbenchv1
# from nemo.collections.asr.parts.utils.manifest_utils import read_manifest

def convert_c4_dataset(tokenizer, file_path=None):
    dataset = load_dataset("allenai/c4", "en")
    # dataset = load_dataset("json", data_files="Data/c4_small.json", split="train")
    # dataset = load_dataset("json", data_files=file_path, split="train")
    def tokenize_function(examples):
            input_ids = torch.Tensor(examples['input_ids'])
            labels = input_ids.clone()
            if tokenizer.pad_token_id is not None:
                 labels[labels == tokenizer.pad_token_id] = -100
            ret = {
                "input_ids": input_ids,
                "labels": labels
            }
            return ret
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['input_tokens'])
    dataset.set_format(type='torch', columns=['input_ids', "labels"])
    return dataset

def convert_wiki_dataset(tokenizer, seq_len = 256):
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[0:2000]")
    def tokenize_function(examples):
            return tokenizer(examples["text"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return dataset

# def convert_cnn_dataset_old(tokenizer, seq_len = 256):
#     breakpoint()
#     dataset = load_dataset("cnn_dailymail", "1.0.0", split="test[0:2000]")
#     def tokenize_function(examples):
#             return tokenizer(examples["article"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
#     dataset = dataset.map(tokenize_function, batched=True, remove_columns=['article'])
#     dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
#     return dataset

def convert_cnn_dataset(tokenizer, seq_len = 256):
    breakpoint()
    dataset = load_dataset("cnn_dailymail", "1.0.0", split="test[0:2000]")
    tokenized_prompts = []
    for i in tqdm(range(0,50)):
        prompt = dataset[i]['article']
        tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
        tokenized_prompt = tokenized_prompt.split(seq_len, dim=-1)[:-1]
        
        for i in range(len(tokenized_prompt)):
            tokenized_prompt[i][:, 0] = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
            tokenized_prompts.append(tokenized_prompt[i])
    data = torch.cat(tokenized_prompts, dim=0)
    return TensorDataset(data)

def convert_pg19_dataset(tokenizer, seq_len = 4096, end = 20):
    datasetparent = "Data/pg19/"
    d_files = os.listdir(datasetparent)
    dataset = load_dataset("json", data_files = [datasetparent + name for name in d_files], split = "train")
    tokenized_prompts = []
    for i in tqdm(range(0,50)):
        prompt = dataset[i]['text']
        tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")[:,8000:]
        tokenized_prompt = tokenized_prompt.split(seq_len, dim=-1)[:-1]
        
        for i in range(len(tokenized_prompt)):
            tokenized_prompt[i][:, 0] = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
            tokenized_prompts.append(tokenized_prompt[i])
    data = torch.cat(tokenized_prompts, dim=0).repeat(end,1)
    return TensorDataset(data)

def convert_longbench_v1_dataset(tokenizer, task=None, is_under_32k=False):
    prompts = []
    if task is None:
        split_list = ["gov_report", "qmsum", "multi_news", "lcc", "repobench-p"]
        split_tag = ["gov_report", "qmsum", "multi_news", "lcc", "repobench-p"]
    else:
        split_list = [task]
        split_tag = [task]

    for split, tag in zip(split_list, split_tag):
        if is_under_32k:
            file_path = f"Data/longbenchv1/{tag}_under_32K.jsonl"
        else:
            file_path = f"Data/longbenchv1/{tag}.jsonl"
        if not os.path.exists(file_path):
            preprocess_longbenchv1(split, tag)
        dataset = [json.loads(line) for line in open(file_path).readlines()]

        for i in tqdm(range(len(dataset))):
            # prompts.append(dataset[i]['instruction'])
            # 1) tokenize *without* padding
            tokenized_prompt = tokenizer.encode(dataset[i]['instruction'], return_tensors="pt")
            # calculate pad length to make prompt length = 128 x k + 32
            # this satisfy assert (args.prefix_len - args.window_size) % 128 == 0 in selfspec_benchmark.py
            L = tokenized_prompt.shape[1]
            k = max((L - 32 + 127) // 128, 0)
            target_len = 128 * k + 32
            pad_len = target_len - L
            # 2) left-pad
            pad_id = tokenizer.pad_token_id
            # padded_prompt = [pad_id] * pad_len + tokenized_prompt
            padded_prompt = torch.nn.functional.pad(tokenized_prompt, (pad_len, 0), value=pad_id)
            # 3) convert to tensors
            prompts.append(padded_prompt)

    return prompts

    # breakpoint()
    # # 1) first pass: encode *without* padding to get lengths
    # tok_kwargs = dict(truncation=True, add_special_tokens=True)
    # first = tokenizer(prompts, **tok_kwargs)
    # lengths = [len(ids) for ids in first["input_ids"]]
    # max_len = max(lengths)

    # # 2) compute target = smallest (128 * k + 32) ≥ max_len
    # if max_len <= 32:
    #     target_len = 32
    # else:
    #     target_len = math.ceil((max_len - 32) / 128) * 128 + 32

    # # 3) second pass: pad *on the left* up to target_len
    # tokenizer.padding_side = "left"
    # enc = tokenizer(
    #     prompts,
    #     padding="max_length",
    #     max_length=target_len,
    #     truncation=True,
    #     return_tensors="pt",
    # )

    # return TensorDataset(enc["input_ids"])

def convert_longbench_v2_dataset(tokenizer, seq_len = 4096):
    tokenized_prompts = []

    # split_list=["Single-Document QA","Multi-Document QA","Long In-context Learning"]
    # split_tag=["SQA","MQA","LongICL"]
    split_list=["Long In-context Learning"]
    split_tag=["LongICL"]
    for split, tag in zip(split_list, split_tag):
        file_path = f"Data/longbenchv2/{tag}_over_64K.jsonl"
        if not os.path.exists(file_path):
            preprocess_longbenchv2(split, tag)
        dataset = [json.loads(line) for line in open(file_path).readlines()]
        for i in tqdm(range(0,50)):
            prompt = dataset[i]['instruction']
            tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
            tokenized_prompt = tokenized_prompt.split(seq_len, dim=-1)[:-1]
            
            for i in range(len(tokenized_prompt)):
                tokenized_prompt[i][:, 0] = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
                tokenized_prompts.append(tokenized_prompt[i])

    data = torch.cat(tokenized_prompts, dim=0)
    return TensorDataset(data)

def convert_longbench_v2_sum_dataset(tokenizer, seq_len = 4096):
    tokenized_prompts = []

    # split_list=["Single-Document QA","Multi-Document QA","Long In-context Learning"]
    # split_tag=["SQA","MQA","LongICL"]
    split_list=["Long In-context Learning"]
    split_tag=["LongICL"]
    for split, tag in zip(split_list, split_tag):
        file_path = f"Data/longbenchv2/{tag}_over_64K_sum.jsonl"
        if not os.path.exists(file_path):
            preprocess_longbenchv2(split, tag)
        dataset = [json.loads(line) for line in open(file_path).readlines()]
        for i in tqdm(range(0,50)):
            prompt = dataset[i]['instruction']
            tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
            tokenized_prompt = tokenized_prompt.split(seq_len, dim=-1)[:-1]
            
            for i in range(len(tokenized_prompt)):
                tokenized_prompt[i][:, 0] = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
                tokenized_prompts.append(tokenized_prompt[i])

    data = torch.cat(tokenized_prompts, dim=0)
    return TensorDataset(data)

# def convert_ruler_dataset(tokenizer, task, model_name, seq_len = 4096, subset = "validation"):
#     curr_folder = os.path.dirname(os.path.abspath(__file__))
#     try:
#         module = importlib.import_module(f"MagicDec.Data.Ruler.synthetic.constants")
#     except ImportError:
#         print(f"Module MagicDec.Data.Ruler.synthetic.constants not found.")

#     tasks_base = module.TASKS
#     with open(os.path.join(curr_folder, f"Ruler/synthetic.yaml"), "r") as f:
#         tasks_customized = yaml.safe_load(f)

#     if task not in tasks_customized:
#         raise ValueError(f'{task} is not found in config_tasks.yaml')
        
#     config = tasks_customized.get(task)
#     config.update(tasks_base[config['task']])
    
#     root_task = tasks_customized[task]['task']
#     suffix = tasks_base[root_task]['template'].split('{context}')[-1]

#     task_file = os.path.join(curr_folder, "Ruler/benchmark_root", model_name, "data", task, f"{subset}.jsonl")
    
#     data = read_manifest(task_file)

#     tokenized_prompts = []
#     tokenized_suffix = tokenizer.encode(suffix, return_tensors="pt")[:, 1:] # remove the bos token
#     suffix_len = tokenized_suffix.shape[-1]
#     print("Total number of prompts", len(data))
#     for i in range(len(data)):
#         prompt = data[i]['input'][:-len(suffix)]
#         input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=seq_len - suffix_len, padding="max_length")
#         assert input_ids.shape[-1] == seq_len - suffix_len
#         tokenized_prompts.append(torch.cat([input_ids[:, :seq_len - suffix_len], tokenized_suffix], dim=-1))
#     data = torch.cat(tokenized_prompts, dim=0)
#     return TensorDataset(data)

# if __name__ == "__main__":
#     from transformers import LlamaTokenizer, DataCollatorForLanguageModeling
#     from torch.utils.data import DataLoader, TensorDataset
#     from tqdm import tqdm
#     tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
#     tokenizer.pad_token = tokenizer.eos_token
#     dataset = convert_pg19_dataset(tokenizer=tokenizer, seq_len=4096)

#     dataloader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True)
#     num_eval_steps = len(dataloader)
#     for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
#         input_ids = batch[0]
    