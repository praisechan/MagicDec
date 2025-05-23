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
parser.add_argument('--gamma1', type=int, default=10, help='start')
parser.add_argument('--gamma2', type=int, default=20, help='start')
parser.add_argument("--budget1", type=float, default=0.05, help="ratio of budget")
parser.add_argument("--budget2", type=float, default=0.25, help="ratio of budget")
parser.add_argument("--estimate_ratio", type=float, default=0.25, help="ratio of estimated clusters for RetriveInfer")

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

target_dec_len = args.gamma1 + 1
draft_dec_len = 1

# Load target model
engine = LMBackend_Retro(dtype=DTYPE, device=DEVICE, dec_len=target_dec_len, draft_dec_len=draft_dec_len)

model2path = json.load(open("Engine/RetrievalAttention/benchmark/LongBench/config/model2path.json", "r"))
model2maxlen = json.load(open("Engine/RetrievalAttention/benchmark/LongBench/config/model2maxlen.json", "r"))
dataset2prompt = json.load(open("Engine/RetrievalAttention/benchmark/LongBench/config/dataset2prompt.json", "r"))
dataset2maxlen = json.load(open("Engine/RetrievalAttention/benchmark/LongBench/config/dataset2maxlen.json", "r"))

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
verify_steps = 0
settle_steps = 0

if benchmark:
    draft_time = 0.0
    target_time = 0.0
    verify_loop = 0.0

# initialize global counters
total_spec_tokens = 0
total_acc_tokens  = 0


for step, batch in tqdm(enumerate(dataset), total=num_eval_steps):
    if step >= num_eval_steps:
        break
    # input_ids = batch[0].to(DEVICE)
    input_ids = engine.preprocess_input(batch, prompt_format, args.attn_type, model_path, args.budget1, args.budget2, args.estimate_ratio, args.dataset, args.prefix_len)
    terminal = False
    tokens_buffer= torch.zeros((BATCH_SIZE, args.gamma1+1), device=DEVICE).long()

    num_nodes = torch.zeros(BATCH_SIZE,device=DEVICE).long()
    num_nodes += input_ids.shape[1]
    input_len = num_nodes.max()

    tokens_buffer[:, 0] = torch.LongTensor(engine.encode(input_ids)[0])
    torch.cuda.synchronize()
    start = time.perf_counter()

    # record unsettled_tokens
    num_unsettled_tokens = 0
    called_verify = 0
    while terminal == False:

        settled = False
        verified = False

        # Draft speculation
        tokens_buffer[:,1:1+args.gamma1] = torch.LongTensor(engine.speculate(tokens_buffer[:, 0].view(-1,1), args.gamma1))


        if (num_unsettled_tokens + args.gamma1 >= args.gamma2) or (called_verify > 5):
            
            settled = True

            target_tokens = torch.LongTensor(engine.settle(cached_tokens_buffer.view(-1,1), num_unsettled_tokens+args.gamma1+1)).to(DEVICE) #TODO: verify stage should be batch-fashion, but this verify() is auto-regressive. 
            settle_steps += 1
    

            input_from_start = torch.concat((engine.input_tokens[:, :engine.verified_cachelength], tokens_buffer), dim=1)
            draft_tokens = input_from_start[:, -(num_unsettled_tokens+args.gamma1):]
            flag_accept_matrix = (target_tokens[:, :num_unsettled_tokens+args.gamma1] == draft_tokens)  # shape: (BATCH_SIZE, gamma)
            breakpoint()
            eot_condition = ((draft_tokens == eot_1) | (draft_tokens == eot_2))  # shape: (BATCH_SIZE, gamma)

            # Compute accept_flags by considering both the acceptance condition and EOT tokens
            accept_flags_int = (flag_accept_matrix & (~eot_condition)).int()
            accept_flags_cumprod = torch.cumprod(accept_flags_int, dim=1)
            accept_flags_matrix = accept_flags_cumprod.bool()

             # Compute the number of accepted tokens
            accept_nums = accept_flags_matrix.sum(dim=1, keepdim=True)  # shape: (BATCH_SIZE, 1)
            
            positions_buffer = torch.arange(num_unsettled_tokens + args.gamma1, device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
            mask_buffer = positions_buffer<accept_nums.view(-1,1)

            # Get the bonus tokens
            indices = accept_nums
            bonus_tokens = target_tokens.gather(1, indices)
            num_nodes += accept_nums.flatten()
            
            # Check for termination conditions

            # 1: eot in accepted tokens
            condition = (eot_condition & accept_flags_matrix).any(dim=1, keepdim=True)
            if condition.any():
                terminal = True

            if (bonus_tokens == eot_1).any() or (bonus_tokens == eot_2).any():
                terminal = True

            # 2: reach max tokens
            num_gen_token_max = 80
            if args.dataset == "longbenchv1" or args.dataset == "longbenchv1-32k":
                #longbenchv1 does not have fixed prefix len
                if num_nodes.max() - input_len >= num_gen_token_max:
                    terminal = True
            else:
                # Check Number of Nodes + Bonus Token <= max_target_token
                if num_nodes.max() - args.prefix_len >= num_gen_token_max:
                    terminal = True
            # Put Bonus tokens to the tokens buffer, and prepare the variables for next itr
            if not terminal:
                
                breakpoint()
                accepted_tokens = torch.concat((cached_tokens_buffer, draft_tokens[mask_buffer].view(1,-1)), dim=1)

                engine.update_settled_kv(accepted_tokens)

                tokens_buffer[:, :1] = bonus_tokens

            # reset counters
            num_unsettled_tokens = 0
            called_verify = 0

            print(f"settlement accepted tokens: {accept_nums.flatten().item()}")
            print(f"total unsettled tokens: {num_unsettled_tokens}")
            print(f"num_nodes: {num_nodes.item()}")
            
        else:

            verified = True

            if called_verify == 0:
                cached_tokens_buffer = tokens_buffer[:, 0].clone()

            target_tokens = torch.LongTensor(engine.verify(tokens_buffer[:, 0].view(-1,1), args.gamma1+1)).to(DEVICE) #TODO: verify stage should be batch-fashion, but this verify() is auto-regressive. 
            verify_steps +=1
            called_verify += 1

            draft_tokens = tokens_buffer[:, 1:args.gamma1+1]
            flag_accept_matrix = (target_tokens[:, :args.gamma1] == draft_tokens)  # shape: (BATCH_SIZE, gamma)

            eot_condition = ((draft_tokens == eot_1) | (draft_tokens == eot_2))  # shape: (BATCH_SIZE, gamma)

            # Compute accept_flags by considering both the acceptance condition and EOT tokens
            accept_flags_int = (flag_accept_matrix & (~eot_condition)).int()
            accept_flags_cumprod = torch.cumprod(accept_flags_int, dim=1)
            accept_flags_matrix = accept_flags_cumprod.bool()

            # Compute the number of accepted tokens
            accept_nums = accept_flags_matrix.sum(dim=1, keepdim=True)  # shape: (BATCH_SIZE, 1)
            num_unsettled_tokens += accept_nums.flatten().item() + 1 # consider bonus tokens
            
            positions_buffer = torch.arange(args.gamma1, device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
            mask_buffer = positions_buffer<accept_nums.view(-1,1)

            # Get the bonus tokens
            indices = accept_nums
            bonus_tokens = target_tokens.gather(1, indices)

            # Check for termination conditions

            # 1: eot in accepted tokens
            condition = (eot_condition & accept_flags_matrix).any(dim=1, keepdim=True)
            if condition.any():
                terminal = True

            if (bonus_tokens == eot_1).any() or (bonus_tokens == eot_2).any():
                terminal = True

            # 2: reach max tokens
            num_gen_token_max = 80
            if args.dataset == "longbenchv1" or args.dataset == "longbenchv1-32k":
                #longbenchv1 does not have fixed prefix len
                if num_nodes.max() - input_len >= num_gen_token_max:
                    terminal = True
            else:
                # Check Number of Nodes + Bonus Token <= max_target_token
                # if num_nodes.max() + 1 >= args.prefix_len + gen_len:
                # if num_nodes.max() + 1 + args.gamma > MAX_LEN_TARGET:
                if num_nodes.max() - args.prefix_len >= num_gen_token_max:
                    terminal = True
            # Put Bonus tokens to the tokens buffer, and prepare the variables for next itr
            if not terminal:
                
                # get accepted token and re-decode to set draft cache (Quest)
                accepted_tokens = torch.concat((tokens_buffer[:, :1], draft_tokens[mask_buffer].view(1,-1)), dim=1)
               
                engine.update_verified_kv(accepted_tokens)

                tokens_buffer[:, :1] = bonus_tokens

                breakpoint()

            print(f"verification accepted tokens: {accept_nums.flatten().item()} + 1 bonus token")
            print(f"total unsettled tokens: {num_unsettled_tokens}")
            print(f"num_nodes: {num_nodes.item()}")

        if terminal:
            num_nodes += 1


    torch.cuda.synchronize()
    end=time.perf_counter()
    total_time += end-start
    num_gen_tokens += (num_nodes.sum() - (input_ids.shape[1] + 1) * BATCH_SIZE)
    # if args.printoutput:
    #     for i in range(BATCH_SIZE):
    #         print("Sequence ", i)
    #         print(tokenizer.decode(output[i, args.prefix_len:num_nodes[i]]))
    print("total time :{:.5f}s, time per iter :{:.5f}s, decoding step: {}, large model step: {}".format(total_time, total_time / target_steps, num_gen_tokens, target_steps))
    if benchmark:
        print("target time :{:.5f}s, draft time :{:.5f}s, verify loop : {}, avg generate len per sentence: {}".format(target_time/target_steps, draft_time / target_steps, verify_loop/target_steps, num_gen_tokens/target_steps/BATCH_SIZE))
    if step < 5:   # TODO: revert to 10?
        total_time = 0.0
        num_gen_tokens = 0
        target_steps = 0
        if benchmark:
            draft_time = 0.0
            target_time = 0.0
            verify_loop = 0.0
    # if use_tp:
    #     dist.barrier()

print(f"Final tokens per second :{num_gen_tokens/total_time}")

# print acceptance rate
if total_spec_tokens > 0:
    accept_rate_total = total_acc_tokens / total_spec_tokens
    print(f"Draft acceptance rate: {accept_rate_total*100:.2f}% "
          f"({total_acc_tokens} accepted of {total_spec_tokens} speculated)")
    import math

    def find_alpha(gamma, accept_rate_total, tol=1e-8, max_iter=100):
        """
        Solve for alpha in (0,1) such that
            (1 - alpha^(gamma+1)) / (1 - alpha) == gamma * accept_rate_total
        using the bisection method.
        """
        def f(alpha):
            # avoid division by zero at alpha=1
            return (1 - alpha**(gamma+1)) / (1 - alpha) -1 - gamma * accept_rate_total

        # initial bracket [low, high]
        low, high = 0.0, 1.0 - 1e-15
        f_low, f_high = f(low), f(high)

        if f_low * f_high > 0:
            raise ValueError(
                "f(0) and f(1) have the same sign; no guaranteed root in (0,1). "
                f"f(0)={f_low}, f(1-)={f_high}"
            )

        for i in range(max_iter):
            mid = (low + high) / 2
            f_mid = f(mid)

            # Check for convergence
            if abs(f_mid) < tol or (high - low)/2 < tol:
                return mid

            # Narrow the bracket
            if f_low * f_mid <= 0:
                high, f_high = mid, f_mid
            else:
                low, f_low = mid, f_mid

        # return best estimate after max_iter
        return (low + high) / 2

    accept_rate_per_token = find_alpha(args.gamma, accept_rate_total)
    print(f"Found alpha = {accept_rate_per_token:.8f}")


import os, csv
CSV_PATH = f"MagicDec/output/RetroInfer/{MODEL}_{args.dataset}_acceptance_rates.csv"
# if the file doesn't yet exist, write the header
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["attn_type", "prefix_len","budget_ratio", "gamma", "task", "accept_rate_total", "accept_rate_per_token"])
        
# append to CSV
with open(CSV_PATH, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        args.attn_type,
        args.prefix_len,
        args.budget_ratio,
        args.gamma,
        args.task,
        f"{accept_rate_total:.4f}",
        f"{accept_rate_per_token:.4f}"
    ])
# if rank == 0:
#     with open("result.txt", "a") as file:
#         file.write("total time :{:.5f}s, time per iter :{:.5f}s, decoding step: {}, large model step: {}, avg latency: {} \n".format(total_time, total_time / target_steps, num_gen_tokens, target_steps, total_time / num_gen_tokens * BATCH_SIZE))
#         file.write("target time :{:.5f}s, draft time :{:.5f}s, verify loop : {}, avg generate len per sentence: {} \n".format(target_time/target_steps, draft_time / target_steps, verify_loop/target_steps, num_gen_tokens/target_steps/BATCH_SIZE))