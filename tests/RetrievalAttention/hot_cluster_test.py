import time
import torch
import sys
sys.path.append("..")
from pathlib import Path
import torch.distributed as dist
from MagicDec.Engine.utils import setup_seed, cuda_graph_for_sampling_argmax_batch, sampling_argmax_batch
from MagicDec.Data.data_converter import convert_pg19_dataset, convert_c4_dataset, convert_wiki_dataset, convert_cnn_dataset, convert_longbench_v2_dataset, convert_longbench_v2_sum_dataset, convert_longbench_v1_dataset
from transformers import AutoTokenizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import argparse
# from MagicDec.Engine.SnapKV.backend import LMBackend
from MagicDec.Engine.RetrievalAttention.backend import LMBackend_Retro
from datasets import load_dataset

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MagicDec.Engine.RetrievalAttention.benchmark.config import generate_config, parse_attn_args
import json

parser = argparse.ArgumentParser(description='Process model configuration and partitions.')
parser.add_argument('--model_name', type=str, default="llama-3.1-8b", help='model name')
parser.add_argument('--dataset', type=str, default="pg19", help='Dataset name.')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')

parser.add_argument('--gamma', type=int, default=7, help='start')

parser.add_argument('--B', type=int, default=45, help='Batch size.')
parser.add_argument('--prefix_len', type=int, default=32800, help='Prefix length')

parser.add_argument('--seed', type=int, default=123, help='Random seed.')

parser.add_argument('--printoutput', action='store_true', help='Whether to compile the model.')
parser.add_argument('--benchmark', action='store_true', help='Whether to compile the model.')
parser.add_argument('--task', type=str, default="gov_report", help='for longbenchv1.')
parser.add_argument("--num_examples", type=int, default=-1, help="num of example to evaluate. -1 for all.")
parser.add_argument("--attn_type", type=str, default="Full_Flash_Attn",                                                     \
                    choices=["Full_Flash_Attn", "RetroInfer"],                          \
                    help="Attention method")
parser.add_argument("--budget_ratio", type=float, default=0.018, help="ratio of budget")
parser.add_argument("--estimate_ratio", type=float, default=0.25, help="ratio of estimated clusters for RetriveInfer")
parser.add_argument("--profile_clustering", action='store_true', help="profile ")

args = parser.parse_args()

# Init model parallelism
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
global print
from MagicDec.Engine.tp import init_dist
global_group = None
rank = 0

setup_seed(args.seed)
print(f"Using device={DEVICE}")

DTYPE = torch.bfloat16
BATCH_SIZE = args.B
benchmark = args.benchmark

target_dec_len = args.gamma + 1
draft_dec_len = 1

# Load target model
engine = LMBackend_Retro(dtype=DTYPE, device=DEVICE, dec_len=target_dec_len, draft_dec_len=draft_dec_len)

model2path = json.load(open("/home/juchanlee/MagicDec/Engine/RetrievalAttention/benchmark/LongBench/config/model2path.json", "r"))
model2maxlen = json.load(open("/home/juchanlee/MagicDec/Engine/RetrievalAttention/benchmark/LongBench/config/model2maxlen.json", "r"))
dataset2prompt = json.load(open("/home/juchanlee/MagicDec/Engine/RetrievalAttention/benchmark/LongBench/config/dataset2prompt.json", "r"))
dataset2maxlen = json.load(open("/home/juchanlee/MagicDec/Engine/RetrievalAttention/benchmark/LongBench/config/dataset2maxlen.json", "r"))

MODEL = args.model_name.split("/")[-1]
TASK = args.task

num_examples = args.num_examples
attn_type = args.attn_type
device = "auto"
dtype = torch.bfloat16
model_path = model2path[args.model_name]
max_length = model2maxlen[MODEL]
prompt_format = dataset2prompt[TASK]

engine.load_model(model_path, max_length, dtype, device, BATCH_SIZE)
vocab_size = engine.model.config.vocab_size
if args.compile:
    engine.compile()

# Load dataset
# tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer = engine.model.tokenizer
eot_1 = tokenizer.eos_token_id
if tokenizer.unk_token_id is not None:
    eot_2 = tokenizer.unk_token_id
else:
    eot_2 = tokenizer.encode("<|eot_id|>")[-1]
print(f"eot_1: {eot_1}, eot_2: {eot_2}")

if args.dataset == "pg19":
  dataset = convert_pg19_dataset(tokenizer=engine.model.tokenizer, seq_len=args.prefix_len)
# elif args.dataset == "c4":
#     dataset = convert_c4_dataset(tokenizer=tokenizer, seq_len=args.prefix_len)
# elif args.dataset == "wiki":
#     dataset = convert_wiki_dataset(tokenizer=tokenizer, seq_len=args.prefix_len)
# elif args.dataset == "cnn":
#     dataset = convert_cnn_dataset(tokenizer=tokenizer, seq_len=args.prefix_len)
# elif args.dataset == "longbenchv1":
#     dataset = convert_longbench_v1_dataset(tokenizer=tokenizer, task=args.task, is_under_32k=False)
# elif args.dataset == "longbenchv1-32k":
#     dataset = convert_longbench_v1_dataset(tokenizer=tokenizer, task=args.task, is_under_32k=True)
# elif args.dataset == "longbenchv2":
#     dataset = convert_longbench_v2_dataset(tokenizer=tokenizer, seq_len=args.prefix_len)
# elif args.dataset == "longbenchv2_sum":
#     dataset = convert_longbench_v2_sum_dataset(tokenizer=tokenizer, seq_len=args.prefix_len)
# elif args.dataset.startswith("ruler"):
#     dataset = convert_ruler_dataset(tokenizer=tokenizer, task=args.dataset.split(":")[1], model_name=args.model_name, seq_len=args.prefix_len)
elif args.dataset == "longbenchv1":
    dataset = load_dataset('THUDM/LongBench', TASK, split='test')
else:
    raise ValueError(f"Unknown dataset {args.dataset}")

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
if args.dataset == "pg19":
  num_eval_steps = min(10, len(dataloader))
else:
  num_eval_steps = len(dataloader)

total_time = 0.0
num_gen_tokens = 0
target_steps = 0
if benchmark:
    draft_time = 0.0
    target_time = 0.0
    verify_loop = 0.0

# initialize global counters
total_spec_tokens = 0
total_acc_tokens  = 0


num_eval_steps = 1
# for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
for step, batch in tqdm(enumerate(dataset), total=num_eval_steps):
    if step >= num_eval_steps:
        break
    # input_ids = batch[0].to(DEVICE)
    input_ids = engine.preprocess_input(batch, prompt_format, args.attn_type, model_path, args.budget_ratio, args.estimate_ratio, args.dataset, args.prefix_len)
    terminal = False
    tokens_buffer= torch.zeros((BATCH_SIZE, args.gamma+1), device=DEVICE).long()
    verified_tokens = torch.zeros(BATCH_SIZE, max_length+1, device=DEVICE).long()
    verified_tokens[:, :input_ids.shape[1]] = input_ids
    num_nodes = torch.zeros(BATCH_SIZE,device=DEVICE).long()
    num_nodes += input_ids.shape[1]
    input_len = num_nodes.max()

    tokens_buffer[:, 0] = torch.LongTensor(engine.encode(input_ids)[0])
    torch.cuda.synchronize()
    start = time.perf_counter()
                    
    draft_outputs, draft_logits, draft_top1_top2_diff = engine.speculate(tokens_buffer[:, 0].view(-1,1), args.gamma, profile_clustering=args.profile_clustering, profile_hot_clustering=True)