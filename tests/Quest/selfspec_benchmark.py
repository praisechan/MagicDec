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
from MagicDec.Engine.SnapKV.backend import LMBackend
from MagicDec.Engine.Quest.backend import LMBackend_Quest

parser = argparse.ArgumentParser(description='Process model configuration and partitions.')
parser.add_argument('--model', type=Path, default=Path("/scratch/models/meta-llama/Meta-Llama-3.1-8B/model.pth"), help='model')
parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3.1-8B", help='model name')
parser.add_argument('--dataset', type=str, default="pg19", help='Dataset name.')
parser.add_argument('--draft_budget', type=int, default=4097, help='Dataset end index.')
parser.add_argument('--rank_group', nargs='+', type=int, help='Target group of ranks')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')

parser.add_argument('--gamma', type=int, default=7, help='start')

parser.add_argument('--B', type=int, default=45, help='Batch size.')
parser.add_argument('--prefix_len', type=int, default=32800, help='Prefix length')
parser.add_argument('--max_len', type=int, default=32896, help='Generate length')
parser.add_argument('--window_size', type=int, default=32, help='Generate length')
parser.add_argument('--chunk_size', type=int, default=16, help='Chunk size')
parser.add_argument('--latest_k', type=int, default=16, help='in Quest, force to see latest k tokens')

parser.add_argument('--seed', type=int, default=123, help='Random seed.')

parser.add_argument('--printoutput', action='store_true', help='Whether to compile the model.')
parser.add_argument('--benchmark', action='store_true', help='Whether to compile the model.')
parser.add_argument('--task', type=str, default=None, help='for longbenchv1.')

args = parser.parse_args()
assert args.prefix_len < args.max_len
assert (args.prefix_len - args.window_size) % 128 == 0
assert args.max_len % 128 == 0
# assert (args.max_len + 127) // 128 == args.prefix_len // 128 + 1
assert (args.draft_budget - 1) % 128 == 0

# Init model parallelism
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
global print
from MagicDec.Engine.tp import init_dist
use_tp = len(args.rank_group) > 1
global_group = None
rank = 0
if use_tp:
    rank, global_group = init_dist()
    if rank != args.rank_group[0]:
        print = lambda *args, **kwargs: None

# if rank == 0:
#     with open("result.txt", "a") as file:
#         file.write(f"SnapKV-Selfspec: Prefix:{args.prefix_len}; Bsz:{args.B}; Gamma:{args.gamma}; Draft budget:{args.draft_budget}\n")

setup_seed(args.seed)
print(f"Using device={DEVICE}")

MAX_LEN_TARGET = args.max_len
if args.dataset == "longbenchv1": 
    MAX_LEN_TARGET = 65664
if args.dataset == "longbenchv1-32k":
    MAX_LEN_TARGET = 49125
DTYPE = torch.bfloat16
BATCH_SIZE = args.B
benchmark = args.benchmark
checkpoint_path = args.model

target_dec_len = args.gamma + 1
draft_dec_len = 1

# Load target model
engine = LMBackend_Quest(dtype=DTYPE, device=DEVICE, dec_len=target_dec_len, draft_dec_len=draft_dec_len)
engine.load_model(args.model_name)
engine.load_draft_model(args.model_name, args.draft_budget, args.chunk_size, BATCH_SIZE, MAX_LEN_TARGET, args.latest_k)
vocab_size = engine.model.config.vocab_size
if args.compile:
    engine.compile()
# engine.setup_caches(max_batch_size=BATCH_SIZE, max_seq_length=MAX_LEN_TARGET, draft_budget=args.draft_budget, window_size=args.window_size)

# Load dataset
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token
eot_1 = tokenizer.eos_token_id
if tokenizer.unk_token_id is not None:
    eot_2 = tokenizer.unk_token_id
else:
    eot_2 = tokenizer.encode("<|eot_id|>")[-1]
print(f"eot_1: {eot_1}, eot_2: {eot_2}")

if args.dataset == "pg19":
    dataset = convert_pg19_dataset(tokenizer=tokenizer, seq_len=args.prefix_len)
elif args.dataset == "c4":
    dataset = convert_c4_dataset(tokenizer=tokenizer, seq_len=args.prefix_len)
elif args.dataset == "wiki":
    dataset = convert_wiki_dataset(tokenizer=tokenizer, seq_len=args.prefix_len)
elif args.dataset == "cnn":
    dataset = convert_cnn_dataset(tokenizer=tokenizer, seq_len=args.prefix_len)
elif args.dataset == "longbenchv1":
    dataset = convert_longbench_v1_dataset(tokenizer=tokenizer, task=args.task, is_under_32k=False)
elif args.dataset == "longbenchv1-32k":
    dataset = convert_longbench_v1_dataset(tokenizer=tokenizer, task=args.task, is_under_32k=True)
elif args.dataset == "longbenchv2":
    dataset = convert_longbench_v2_dataset(tokenizer=tokenizer, seq_len=args.prefix_len)
elif args.dataset == "longbenchv2_sum":
    dataset = convert_longbench_v2_sum_dataset(tokenizer=tokenizer, seq_len=args.prefix_len)
# elif args.dataset.startswith("ruler"):
#     dataset = convert_ruler_dataset(tokenizer=tokenizer, task=args.dataset.split(":")[1], model_name=args.model_name, seq_len=args.prefix_len)
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

# for step, batch in tqdm(enumerate(dataloader)):
for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
    if step >= num_eval_steps:
        break
    # if step == 35:
    #     breakpoint()
    input_ids = batch[0].to(DEVICE)
    terminal = False
    tokens_buffer= torch.zeros((BATCH_SIZE, args.gamma+1), device=DEVICE).long()
    output = torch.zeros(BATCH_SIZE, MAX_LEN_TARGET+1, device=DEVICE).long()
    output[:, :input_ids.shape[1]] = input_ids
    num_nodes = torch.zeros(BATCH_SIZE,device=DEVICE).long()
    num_nodes += input_ids.shape[1]
    input_len = num_nodes.max()
    tokens_buffer[:, :1] = engine.encode(input_ids=input_ids)[:,-1:]
    torch.cuda.synchronize()
    start = time.perf_counter()
    while terminal == False:

        # Draft speculation
        if benchmark:
            torch.cuda.synchronize()
            t1 = time.time()

        tokens_buffer[:,1:1+args.gamma] = engine.speculate(tokens_buffer[:, 0].view(-1,1), BATCH_SIZE, args.gamma)

        if benchmark:
            torch.cuda.synchronize()
            t2 = time.time()
            draft_time+=t2-t1

        # Target Verification
        target_tokens = engine.verify(tokens_buffer)

        if benchmark:
            torch.cuda.synchronize()
            t3 = time.time()
            target_time+=t3-t2

        target_steps+=1

        # Verification
        # Vectorized Verify Loop
        draft_tokens = tokens_buffer[:, 1:args.gamma+1]
        flag_accept_matrix = (target_tokens[:, :args.gamma] == draft_tokens)  # shape: (BATCH_SIZE, gamma)
        eot_condition = ((draft_tokens == eot_1) | (draft_tokens == eot_2))  # shape: (BATCH_SIZE, gamma)

        # Compute accept_flags by considering both the acceptance condition and EOT tokens
        accept_flags_int = (flag_accept_matrix & (~eot_condition)).int()
        accept_flags_cumprod = torch.cumprod(accept_flags_int, dim=1)
        accept_flags_matrix = accept_flags_cumprod.bool()

        # Compute the number of accepted tokens
        accept_nums = accept_flags_matrix.sum(dim=1, keepdim=True) + 1  # shape: (BATCH_SIZE, 1)

        #############################Added for acceptance rate#####################
        # how many draft tokens _in total_ got fully accepted this iteration?
        # accept_flags_matrix.sum() is the total across the batch
        accepted_this_iter = int(accept_flags_matrix.sum().item())

        # record total speculations: BATCH_SIZE * gamma
        speculated_this_iter = BATCH_SIZE * args.gamma
        total_spec_tokens += speculated_this_iter
        total_acc_tokens  += accepted_this_iter
        ##########################################################################
        
        # Check for termination conditions
        condition = (eot_condition & accept_flags_matrix).any(dim=1, keepdim=True)
        if condition.any():
            terminal = True
        
        # # Rollback the memory length
        # engine.cachelens = engine.cachelens - args.gamma - 1
        # engine.paged_kv_last_page_len = engine.paged_kv_last_page_len - args.gamma - 1
        # engine.draft_cachelens = engine.draft_cachelens - args.gamma -1
        # engine.draft_paged_kv_last_page_len = engine.draft_paged_kv_last_page_len - args.gamma -1

        # Put the accepted tokens to output
        # positions = torch.arange(output.shape[1], device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
        # mask = (positions < (engine.cachelens.view(-1,1) + accept_nums)) & (positions >= engine.cachelens.view(-1, 1))
        positions_buffer = torch.arange(args.gamma+1, device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
        mask_buffer = positions_buffer<accept_nums.view(-1,1)
        # output[mask] = tokens_buffer[mask_buffer]

        # # Set the cache length to the accepted length
        # engine.cachelens += accept_nums.flatten().to(torch.int32)
        # engine.paged_kv_last_page_len += accept_nums.flatten().to(torch.int32)
        # engine.draft_cachelens += accept_nums.flatten().to(torch.int32)
        # engine.draft_paged_kv_last_page_len += accept_nums.flatten().to(torch.int32)
        
        # Get the bonus tokens
        indices = accept_nums - 1
        bonus_tokens = target_tokens.gather(1, indices)
        if (bonus_tokens == eot_1).any() or (bonus_tokens == eot_2).any():
            terminal = True
        num_nodes += accept_nums.flatten()

        # Check for termination conditions with accepted token number
        # num_gen_token_max = 16
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
            tokens_buffer[:, :1] = bonus_tokens

        if not terminal:
            # get accepted token and re-decode to set draft cache (Quest)
            accepted_tokens = tokens_buffer[mask_buffer].view(1,-1)
            engine.draft_kv_update(accepted_tokens)


        if not terminal:
            if benchmark:
                torch.cuda.synchronize()
                t4 = time.time()
                verify_loop += t4-t3
        else:
            # for i in range(BATCH_SIZE):
            #     output[i, num_nodes[i]] = bonus_tokens[i]
            num_nodes += 1
            if benchmark:
                torch.cuda.synchronize()
                t4 = time.time()
                verify_loop += t4-t3

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
    if use_tp:
        dist.barrier()

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
model_name = args.model_name.split("/", 1)[1]
CSV_PATH = f"/home/juchanlee/MagicDec/output/{model_name}_{args.dataset}_acceptance_rates.csv"
# if the file doesn't yet exist, write the header
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prefix_len", "draft_budget", "gamma", "task", "accept_rate_total", "accept_rate_per_token"])
        
# append to CSV
with open(CSV_PATH, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        args.prefix_len,
        args.draft_budget,
        args.gamma,
        args.task,
        f"{accept_rate_total:.4f}"
        f"{accept_rate_per_token:.4f}"
    ])
# if rank == 0:
#     with open("result.txt", "a") as file:
#         file.write("total time :{:.5f}s, time per iter :{:.5f}s, decoding step: {}, large model step: {}, avg latency: {} \n".format(total_time, total_time / target_steps, num_gen_tokens, target_steps, total_time / num_gen_tokens * BATCH_SIZE))
#         file.write("target time :{:.5f}s, draft time :{:.5f}s, verify loop : {}, avg generate len per sentence: {} \n".format(target_time/target_steps, draft_time / target_steps, verify_loop/target_steps, num_gen_tokens/target_steps/BATCH_SIZE))