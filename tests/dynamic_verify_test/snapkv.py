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
import numpy as np
import pandas as pd
import os

# Process histogram data
def create_histogram_data(data, bins=50, range_min=None, range_max=None):
    """Create histogram data from a list of values"""
    if not data:
        return np.zeros(bins), []
    
    # Flatten the data if it contains nested lists
    flat_data = []
    for item in data:
        if isinstance(item, (list, tuple)):
            # Handle list/tuple of tensors or values
            for sub_item in item:
                if torch.is_tensor(sub_item):
                    flat_data.extend(sub_item.cpu().float().numpy().flatten())
                else:
                    flat_data.append(float(sub_item))
        elif torch.is_tensor(item):
            flat_data.extend(item.cpu().float().numpy().flatten())
        else:
            flat_data.append(float(item))
    
    flat_data = np.array(flat_data)
    
    # Set range if not provided
    if range_min is None:
        range_min = float(np.min(flat_data))
    if range_max is None:
        range_max = float(np.max(flat_data))
    
    # Create histogram
    hist, bin_edges = np.histogram(flat_data, bins=bins, range=(range_min, range_max))
    
    # Create range labels for column headers
    range_labels = []
    for i in range(len(bin_edges)-1):
        range_labels.append(f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}")
    
    return hist, range_labels

parser = argparse.ArgumentParser(description='Process model configuration and partitions.')
parser.add_argument('--model', type=Path, default=Path("/scratch/models/meta-llama/Meta-Llama-3.1-8B/model.pth"), help='model')
parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3.1-8B", help='model name')
parser.add_argument('--dataset', type=str, default="pg19", help='Dataset name.')
parser.add_argument('--draft_budget', type=int, default=4097, help='Dataset end index.')
parser.add_argument('--rank_group', nargs='+', type=int, help='Target group of ranks')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')

parser.add_argument('--gamma', type=int, default=7, help='start')

parser.add_argument('--B', type=int, default=1, help='Batch size.')
parser.add_argument('--prefix_len', type=int, default=32800, help='Prefix length')
parser.add_argument('--max_len', type=int, default=32896, help='Generate length')
parser.add_argument('--window_size', type=int, default=32, help='Generate length')

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
BATCH_SIZE = args.B # only support 1 for now
if BATCH_SIZE > 1:
    print("Warning: BATCH_SIZE > 1 is not supported in this script, setting BATCH_SIZE to 1.")
    BATCH_SIZE = 1
benchmark = args.benchmark
checkpoint_path = args.model

target_dec_len = args.gamma + 1
draft_dec_len = 1

# Load target model
engine = LMBackend(dtype=DTYPE, device=DEVICE, dec_len=target_dec_len, draft_dec_len=draft_dec_len)
engine.load_model(checkpoint_path, use_tp=use_tp, rank_group = args.rank_group, group=global_group)
vocab_size = engine.model.config.vocab_size
if args.compile:
    engine.compile()
engine.setup_caches(max_batch_size=BATCH_SIZE, max_seq_length=MAX_LEN_TARGET, draft_budget=args.draft_budget, window_size=args.window_size)

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

# Initialize draft history storage
draft_history = []

# initialize global counters
total_spec_tokens = 0
total_acc_tokens  = 0


draft_top1_top2_diff_data = []
reject_token_top1_top2_diff_data = []

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

    # Initialize trace for this step
    step_trace = {
        'step': step,
        'input_ids': None,
        'draft_iter': {
            'draft_tokens': [],
            'accept_flags_matrix': [],
            'draft_top1_top2_diff': []
        }
    }    
    step_trace['input_ids'] = input_ids.clone().detach().cpu()  # Store on CPU to save memory
    
    while terminal == False:

        # Draft speculation
        if benchmark:
            torch.cuda.synchronize()
            t1 = time.time()

        draft_logits =[None]*args.gamma
        draft_top1_top2_diff =[None]*args.gamma
        for i in range(args.gamma):
            draft_output, draft_logits[i], draft_top1_top2_diff[i] = engine.speculate(tokens_buffer[:, i].view(-1,1))
            tokens_buffer[:,i+1:i+2] =  draft_output
            # tokens_buffer[:,i+1:i+2] = engine.speculate(tokens_buffer[:, i].view(-1,1))        
        
        if benchmark:
            torch.cuda.synchronize()
            t2 = time.time()
            draft_time+=t2-t1

        # Target Verification
        target_outputs, target_logits = engine.verify(tokens_buffer)
        target_tokens = target_outputs
        # target_tokens = engine.verify(tokens_buffer)


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
        
        # if accepted_this_iter != args.gamma:
        #   print(draft_logits[accepted_this_iter][0])
        #   print(target_logits[0][accepted_this_iter])
        ##########################################################################
        
        # Check for termination conditions
        condition = (eot_condition & accept_flags_matrix).any(dim=1, keepdim=True)
        if condition.any():
            terminal = True
        
        # Rollback the memory length
        engine.cachelens = engine.cachelens - args.gamma - 1
        engine.paged_kv_last_page_len = engine.paged_kv_last_page_len - args.gamma - 1
        engine.draft_cachelens = engine.draft_cachelens - args.gamma -1
        engine.draft_paged_kv_last_page_len = engine.draft_paged_kv_last_page_len - args.gamma -1

        # Put the accepted tokens to output
        positions = torch.arange(output.shape[1], device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
        mask = (positions < (engine.cachelens.view(-1,1) + accept_nums)) & (positions >= engine.cachelens.view(-1, 1))
        positions_buffer = torch.arange(args.gamma+1, device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
        mask_buffer = positions_buffer<accept_nums.view(-1,1)
        output[mask] = tokens_buffer[mask_buffer]

        # Set the cache length to the accepted length
        engine.cachelens += accept_nums.flatten().to(torch.int32)
        engine.paged_kv_last_page_len += accept_nums.flatten().to(torch.int32)
        engine.draft_cachelens += accept_nums.flatten().to(torch.int32)
        engine.draft_paged_kv_last_page_len += accept_nums.flatten().to(torch.int32)
        
        # Get the bonus tokens
        indices = accept_nums - 1
        bonus_tokens = target_tokens.gather(1, indices)
        if (bonus_tokens == eot_1).any() or (bonus_tokens == eot_2).any():
            terminal = True
        num_nodes += accept_nums.flatten()
        
        # record reject token's probabilty
        reject_token_idx = accept_nums - 1
        if reject_token_idx < args.gamma:
            # if all accepted, no reject token
            reject_token_top1_top2_diff = draft_top1_top2_diff[reject_token_idx]
            reject_token_top1_top2_diff_data.append(reject_token_top1_top2_diff)
        # Get the draft top1-top2 difference
        draft_top1_top2_diff_data.append(draft_top1_top2_diff[:reject_token_idx+1])

        # Record the draft tokens, accept flags, and top1-top2 diff for this step
        step_trace['draft_iter']['draft_tokens'].append(draft_tokens.clone().detach().cpu())
        step_trace['draft_iter']['accept_flags_matrix'].append(accept_flags_matrix.clone().detach().cpu())
        step_trace['draft_iter']['draft_top1_top2_diff'].append([x.clone().detach().cpu() if torch.is_tensor(x) else x for x in draft_top1_top2_diff])
   

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
            if benchmark:
                torch.cuda.synchronize()
                t4 = time.time()
                verify_loop += t4-t3
        else:
            for i in range(BATCH_SIZE):
                output[i, num_nodes[i]] = bonus_tokens[i]
            num_nodes += 1
            if benchmark:
                torch.cuda.synchronize()
                t4 = time.time()
                verify_loop += t4-t3

    # Add the step trace to draft history
    draft_history.append(step_trace)
    
    torch.cuda.synchronize()
    end=time.perf_counter()
    total_time += end-start
    num_gen_tokens += (num_nodes.sum() - (input_ids.shape[1] + 1) * BATCH_SIZE)
    if args.printoutput:
        for i in range(BATCH_SIZE):
            print("Sequence ", i)
            print(tokenizer.decode(output[i, args.prefix_len:num_nodes[i]]))
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

# print(f"Final tokens per second :{num_gen_tokens/total_time}")

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


# import os, csv
# model_name = args.model_name.split("/", 1)[1]
# CSV_PATH = f"/home/juchanlee/MagicDec/output/{model_name}_{args.dataset}_acceptance_rates.csv"
# # if the file doesn't yet exist, write the header
# if not os.path.exists(CSV_PATH):
#     with open(CSV_PATH, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["prefix_len", "draft_budget", "gamma", "task", "accept_rate_total", "accept_rate_per_token"])
        
# # append to CSV
# with open(CSV_PATH, "a", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow([
#         args.prefix_len,
#         args.draft_budget,
#         args.gamma,
#         args.task,
#         f"{accept_rate_total:.4f}"
#         f"{accept_rate_per_token:.4f}"
#     ])
# if rank == 0:
#     with open("result.txt", "a") as file:
#         file.write("total time :{:.5f}s, time per iter :{:.5f}s, decoding step: {}, large model step: {}, avg latency: {} \n".format(total_time, total_time / target_steps, num_gen_tokens, target_steps, total_time / num_gen_tokens * BATCH_SIZE))
#         file.write("target time :{:.5f}s, draft time :{:.5f}s, verify loop : {}, avg generate len per sentence: {} \n".format(target_time/target_steps, draft_time / target_steps, verify_loop/target_steps, num_gen_tokens/target_steps/BATCH_SIZE))

# Set histogram parameters
HIST_BINS = 10  # Reduced for readability, adjust as needed
HIST_RANGE_MIN = 0
HIST_RANGE_MAX = 1

print(f"Creating histograms for draft_top1_top2_diff_data (length: {len(draft_top1_top2_diff_data)})")
print(f"Creating histograms for reject_token_top1_top2_diff_data (length: {len(reject_token_top1_top2_diff_data)})")

# Create histogram data
draft_hist, range_labels = create_histogram_data(
    draft_top1_top2_diff_data, 
    bins=HIST_BINS, 
    range_min=HIST_RANGE_MIN, 
    range_max=HIST_RANGE_MAX
)

reject_hist, _ = create_histogram_data(
    reject_token_top1_top2_diff_data, 
    bins=HIST_BINS, 
    range_min=HIST_RANGE_MIN, 
    range_max=HIST_RANGE_MAX
)

# Prepare experiment identifier
experiment_info = f"snapkv_{args.model_name.split('/', 1)[1]}_{args.dataset}_prefix{args.prefix_len}_gamma{args.gamma}_budget{args.draft_budget}"
if args.task:
    experiment_info += f"_task{args.task}"

# Create data for CSV - Draft histogram
draft_row = {'experiment': f"{experiment_info}_draft"}
for i, range_label in enumerate(range_labels):
    draft_row[range_label] = draft_hist[i]

# Create data for CSV - Reject histogram  
reject_row = {'experiment': f"{experiment_info}_reject"}
for i, range_label in enumerate(range_labels):
    reject_row[range_label] = reject_hist[i]

# Create DataFrame
df_new = pd.DataFrame([draft_row, reject_row])

# Save histogram data to CSV (append mode)
model_name = args.model_name.split("/", 1)[1]
HISTOGRAM_CSV_PATH = f"/home/juchanlee/MagicDec/output/snapkv_{model_name}_{args.dataset}_histogram_data.csv"

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(HISTOGRAM_CSV_PATH), exist_ok=True)

# Check if file exists to determine if we need headers
file_exists = os.path.exists(HISTOGRAM_CSV_PATH)

# If file exists, read it and append new data
if file_exists:
    df_existing = pd.read_csv(HISTOGRAM_CSV_PATH)
    # Ensure all columns are present in both dataframes
    all_columns = list(set(df_existing.columns) | set(df_new.columns))
    df_existing = df_existing.reindex(columns=all_columns, fill_value=0)
    df_new = df_new.reindex(columns=all_columns, fill_value=0)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
else:
    df_combined = df_new

# Define desired order once
desired_columns = ['experiment'] + sorted(range_labels, key=lambda x: float(x.split('-')[0]))

# Apply this order to ALL DataFrames
df_new = df_new.reindex(columns=desired_columns, fill_value=0)
df_existing = df_existing.reindex(columns=desired_columns, fill_value=0) 
df_combined = df_combined.reindex(columns=desired_columns, fill_value=0)

# Save the combined data
df_combined.to_csv(HISTOGRAM_CSV_PATH, index=False)
print(f"Histogram data saved to: {HISTOGRAM_CSV_PATH}")

# Print summary
print(f"\nAdded 2 rows (draft and reject) to CSV")
print(f"Draft total count: {draft_hist.sum()}")
print(f"Reject total count: {reject_hist.sum()}")
print(f"Column headers: {range_labels}")


# Save draft history to .pt file
model_name = args.model_name.split("/", 1)[1]
experiment_info = f"{model_name}_{args.dataset}_prefix{args.prefix_len}_gamma{args.gamma}_budget{args.draft_budget}"
if args.task:
    experiment_info += f"_task{args.task}"

# Create output directory if it doesn't exist
output_dir = f"/home/juchanlee/MagicDec/output/draft_histories"
os.makedirs(output_dir, exist_ok=True)

draft_history_path = f"{output_dir}/{experiment_info}_draft_history.pt"

# Save the complete draft history
torch.save(draft_history, draft_history_path)
print(f"Draft history saved to: {draft_history_path}")

# Optional: Save metadata separately
metadata = {
    'experiment_config': {
        'model_name': args.model_name,
        'dataset': args.dataset,
        'prefix_len': args.prefix_len,
        'gamma': args.gamma,
        'draft_budget': args.draft_budget,
        'task': args.task,
        'batch_size': args.B,
        'max_len': args.max_len,
        'window_size': args.window_size,
        'seed': args.seed
    },
    'total_steps': len(draft_history),
    'total_spec_tokens': total_spec_tokens,
    'total_acc_tokens': total_acc_tokens
}

metadata_path = f"{output_dir}/{experiment_info}_metadata.pt"
torch.save(metadata, metadata_path)
print(f"Metadata saved to: {metadata_path}")