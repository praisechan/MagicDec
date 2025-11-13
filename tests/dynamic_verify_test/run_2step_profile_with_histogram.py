import time
import torch
import sys
sys.path.append("..")
from pathlib import Path
import torch.distributed as dist
import csv
import os
from datetime import datetime
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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
parser.add_argument('--model_name', type=str, default="llama-3.1-8b", help='model name')
parser.add_argument('--dataset', type=str, default="pg19", help='Dataset name.')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')

parser.add_argument('--B', type=int, default=1, help='Batch size.')
parser.add_argument('--prefix_len', type=int, default=32800, help='Prefix length')

parser.add_argument('--seed', type=int, default=123, help='Random seed.')

parser.add_argument('--printoutput', action='store_true', help='Whether to compile the model.')
parser.add_argument('--benchmark', action='store_true', help='Whether to compile the model.')
parser.add_argument('--task', type=str, default="gov_report", help='for longbenchv1.')
parser.add_argument("--num_examples", type=int, default=-1, help="num of example to evaluate. -1 for all.")
parser.add_argument("--attn_type", type=str, default="RetroInfer", help="Attention method")
parser.add_argument('--gamma1', type=int, default=10, help='start')
parser.add_argument('--gamma2', type=int, default=20, help='start')
parser.add_argument("--budget1", type=float, default=0.05, help="ratio of budget")
parser.add_argument("--budget2", type=float, default=0.25, help="ratio of budget")
parser.add_argument("--budget2_low", type=float, default=0.1, help="lower ratio of budget for verification when confidence is high")
parser.add_argument("--confidence_threshold", type=float, default=0.5, help="threshold for top1_top2_diff to use lower budget")
parser.add_argument("--enable_dynamic_budget", action='store_true', help="enable dynamic budget adjustment based on confidence")
parser.add_argument("--estimate_ratio", type=float, default=0.25, help="ratio of estimated clusters for RetriveInfer")

# Histogram configuration parameters
parser.add_argument("--hist_num_bins", type=int, default=10, help="number of bins for confidence change histogram")
parser.add_argument("--hist_bin_width", type=float, default=0.1, help="width of each bin for confidence change histogram")
parser.add_argument("--hist_center", type=float, default=0.5, help="center value for histogram ranges (typically 0.0)")
parser.add_argument("--hist_statistics_bins", type=int, default=50, help="number of bins for histogram data in statistics CSV files")

# Intermediate verification budget parameters
parser.add_argument("--intermediate_budgets", nargs='+', type=float, default=[0.1, 0.25, 0.4], help="list of intermediate budget ratios for verification (e.g., 0.1 0.25 0.4)")
parser.add_argument("--enable_intermediate_verify", action='store_true', help="enable intermediate verification with multiple budgets")

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
model_path = model2path[MODEL]
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
  # dataset = load_dataset('emozilla/pg19', split='test')
elif args.dataset == "longbenchv1":
    dataset = load_dataset('THUDM/LongBench', TASK, split='test')
else:
    raise ValueError(f"Unknown dataset {args.dataset}")

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
if args.dataset == "pg19":
#   num_eval_steps = min(10, len(dataloader))
  num_eval_steps = min(300, len(dataloader))
else:
  num_eval_steps = len(dataloader)

num_gen_token_max = 100
num_gen_tokens = 0

# Store these for dynamic budget adjustment
current_model_path = model_path
current_attn_type = args.attn_type

# CSV logging setup
# log_dir = "logs"
profile_dir = f"/home/juchanlee/MagicDec/profile/histogram_profile/{MODEL}_{args.dataset}_{args.prefix_len}"
log_dir = profile_dir

os.makedirs(log_dir, exist_ok=True)

# Simple filenames without timestamp/counter
step_log_file = os.path.join(log_dir, "step_log.csv")
accumulated_log_file = os.path.join(log_dir, "accumulated_log.csv")

# Initialize step-wise CSV file with headers
step_headers = [
    "step", "dataset", "prefix_len", "gamma1", "gamma2", "budget1", "budget2", 
    "budget2_low", "confidence_threshold", "enable_dynamic_budget", "speculate_calls", "verify_calls", 
    "settle_calls", "budget_switches_step", "tokens_generated", "min_confidence", 
    "avg_confidence"
]

# Initialize accumulated CSV file with headers
accumulated_headers = [
    "step", "dataset", "prefix_len", "gamma1", "gamma2", "budget1", "budget2", 
    "budget2_low", "confidence_threshold", "enable_dynamic_budget", "total_speculate_calls", "total_verify_calls", 
    "total_settle_calls", "total_budget_switches", "total_tokens_generated"
]

# Create step-wise log file only if it doesn't exist
if not os.path.exists(step_log_file):
    with open(step_log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(step_headers)

# Create accumulated log file only if it doesn't exist
if not os.path.exists(accumulated_log_file):
    with open(accumulated_log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(accumulated_headers)

print(f"Logging to: {step_log_file}")
print(f"Accumulated logging to: {accumulated_log_file}")

# Global counters for accumulated statistics
total_speculate_calls = 0
total_verify_calls = 0
total_settle_calls = 0
total_budget_switches = 0
total_tokens_generated = 0

# Initialize data storage for histogram analysis
draft_top1_top2_diff_data = []
reject_token_top1_top2_diff_data = []
draft_top1_logits_data = []
reject_token_top1_logits_data = []

# Initialize intermediate verify data storage
intermediate_reject_data = {}
intermediate_reject_top1_logits_data = {}
if args.enable_intermediate_verify:
    for budget in args.intermediate_budgets:
        intermediate_reject_data[f"budget_{budget:.2f}"] = []
        intermediate_reject_top1_logits_data[f"budget_{budget:.2f}"] = []
actual_step = 0
for step, batch in tqdm(enumerate(dataset), total=num_eval_steps):
    if actual_step >= num_eval_steps:
        break
    input_ids = engine.preprocess_input(batch, prompt_format, args.attn_type, model_path, args.budget1, args.estimate_ratio, args.dataset, args.prefix_len)
    if input_ids is None:
        print(f"Skipping step {step} due to empty input_ids.")
        continue
    actual_step += 1 # increment actual step count only if input_ids is valid
    
    # Initialize step-wise counters
    step_speculate_calls = 0
    step_verify_calls = 0
    step_settle_calls = 0
    step_budget_switches = 0
    step_confidences = []  # Store confidence values for this step
    
    terminal = False
    tokens_buffer= torch.zeros((BATCH_SIZE, args.gamma1+1), device=DEVICE).long()

    num_nodes = torch.zeros(BATCH_SIZE,device=DEVICE).long()
    num_nodes += input_ids.shape[1]
    input_len = num_nodes.max()

    tokens_buffer[:, 0] = torch.LongTensor(engine.encode(input_ids)[0])
    torch.cuda.synchronize()
    start = time.perf_counter()

    while not terminal:
        settled = False
        verified = False

        # Draft speculation
        draft_outputs, top3_logits, top1_top2_diff = engine.speculate(tokens_buffer[:, :1], args.gamma1, profile_clustering=False, profile_hot_cluster_selection_ratio=False, generate_name=f"{profile_dir}/speculate_{step}_{step_speculate_calls}")
        tokens_buffer[:,1:1+args.gamma1] = torch.LongTensor(draft_outputs)
        step_speculate_calls += args.gamma1
        
        # Store draft confidences for histogram analysis
        top1_logits_tensor = [torch.tensor([logits[0][0][0]]) for logits in top3_logits]
        min_confidence = torch.min(torch.tensor(top1_logits_tensor))
        avg_confidence = torch.mean(torch.tensor(top1_logits_tensor))

        if top1_top2_diff is not None:
            # Convert to list of tensors or values for compatibility with create_histogram_data
            if isinstance(top1_top2_diff, (list, tuple)):
                draft_top1_top2_diff_data.extend(top1_top2_diff)
            else:
                draft_top1_top2_diff_data.append(top1_top2_diff)
        
        # Store draft top1_logits for histogram analysis
        if top1_logits_tensor is not None:
            if isinstance(top1_logits_tensor, (list, tuple)):
                draft_top1_logits_data.extend(top1_logits_tensor)
            else:
                draft_top1_logits_data.append(top1_logits_tensor)
        
        # Settle
        # in run_2step, settle is always called after speculate
        settled = True

        settle_outputs, settle_logits = engine.verify(tokens_buffer[:, :1], args.gamma1+1)
        target_tokens = torch.LongTensor(settle_outputs).to(DEVICE) #TODO: verify stage should be batch-fashion, but this verify() is auto-regressive.
        
        step_settle_calls += 1

        draft_tokens = tokens_buffer[:, 1:args.gamma1+1]
        flag_accept_matrix = (target_tokens[:, :args.gamma1] == draft_tokens)
        eot_condition = ((draft_tokens == eot_1) | (draft_tokens == eot_2))

        accept_flags_int = (flag_accept_matrix & (~eot_condition)).int()
        accept_flags_cumprod = torch.cumprod(accept_flags_int, dim=1)
        accept_flags_matrix = accept_flags_cumprod.bool()
        accept_nums = accept_flags_matrix.sum(dim=1, keepdim=True)

        # Record reject token's top1-top2 difference for histogram analysis
        reject_token_idx = accept_nums.flatten().item()
        if reject_token_idx < args.gamma1 and top1_top2_diff is not None:
            # If not all tokens were accepted, record the reject token's confidence
            if isinstance(top1_top2_diff, (list, tuple)) and len(top1_top2_diff) > reject_token_idx:
                reject_token_top1_top2_diff_data.append(top1_top2_diff[reject_token_idx])
        
        # Record reject token's top1_logits for histogram analysis
        if reject_token_idx < args.gamma1 and top1_logits_tensor is not None:
            if isinstance(top1_logits_tensor, (list, tuple)) and len(top1_logits_tensor) > reject_token_idx:
                reject_token_top1_logits_data.append(top1_logits_tensor[reject_token_idx])

        # Intermediate verification with multiple budgets
        if args.enable_intermediate_verify and reject_token_idx < args.gamma1:
            # Save original input length for reset_attn_config_for_speculate
            original_input_len = input_ids.shape[1]
            
            for budget in args.intermediate_budgets:
                # Reset attention config with intermediate budget
                engine.reset_attn_config_for_speculate(
                    model_path, 
                    original_input_len, 
                    args.attn_type, 
                    budget, 
                    args.estimate_ratio
                )
                
                # Perform intermediate verify using speculate with higher budget
                try:
                    intermediate_outputs, intermediate_logits, intermediate_top1_top2_diff = engine.speculate(
                        tokens_buffer[:, :1], 
                        args.gamma1, 
                        profile_clustering=False,
                        profile_hot_cluster_selection_ratio=False,
                        generate_name=None
                    )
                    
                    # Compare intermediate results with original draft tokens
                    intermediate_tokens = torch.LongTensor(intermediate_outputs).to(DEVICE)
                    intermediate_target_tokens = intermediate_tokens[:, :args.gamma1]  # Exclude the first token
                    
                    # Calculate acceptance for intermediate verification
                    intermediate_flag_accept_matrix = (intermediate_target_tokens == draft_tokens)
                    intermediate_accept_flags_int = (intermediate_flag_accept_matrix & (~eot_condition)).int()
                    intermediate_accept_flags_cumprod = torch.cumprod(intermediate_accept_flags_int, dim=1)
                    intermediate_accept_flags_matrix = intermediate_accept_flags_cumprod.bool()
                    intermediate_accept_nums = intermediate_accept_flags_matrix.sum(dim=1, keepdim=True)
                    intermediate_reject_token_idx = intermediate_accept_nums.flatten().item()
                    
                    # Only record if intermediate rejection index matches real rejection index
                    if (intermediate_reject_token_idx == reject_token_idx and 
                        intermediate_top1_top2_diff is not None and
                        isinstance(intermediate_top1_top2_diff, (list, tuple)) and
                        len(intermediate_top1_top2_diff) > reject_token_idx):
                        
                        budget_key = f"budget_{budget:.2f}"
                        intermediate_reject_data[budget_key].append(top1_top2_diff[reject_token_idx])
                        
                        # Store intermediate top1_logits if available
                        if (intermediate_logits is not None and 
                            len(intermediate_logits) > reject_token_idx):
                            intermediate_top1_logits_tensor = [torch.tensor([logits[0][0][0]]) for logits in intermediate_logits]
                            if len(intermediate_top1_logits_tensor) > reject_token_idx:
                                intermediate_reject_top1_logits_data[budget_key].append(intermediate_top1_logits_tensor[reject_token_idx])
                        
                except Exception as e:
                    print(f"Warning: Intermediate verify failed for budget {budget}: {e}")
                    continue
            
            # Reset attention config back to original budget for next speculation
            engine.reset_attn_config_for_speculate(
                model_path, 
                original_input_len, 
                args.attn_type, 
                args.budget1, 
                args.estimate_ratio
            )

        positions_buffer = torch.arange(args.gamma1, device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
        mask_buffer = positions_buffer < accept_nums.view(-1,1)
        indices = accept_nums
        bonus_tokens = target_tokens.gather(1, indices)
        num_nodes += (accept_nums.flatten() + 1)

        # Check for termination conditions
        condition = (eot_condition & accept_flags_matrix).any(dim=1, keepdim=True)
        if condition.any() or (bonus_tokens == eot_1).any() or (bonus_tokens == eot_2).any():
            terminal = True

        if args.dataset == "longbenchv1" or args.dataset == "longbenchv1-32k":
            if num_nodes.max() - input_len >= num_gen_token_max:
                terminal = True
        else:
            if num_nodes.max() - args.prefix_len >= num_gen_token_max:
                terminal = True

        # get accepted token and re-decode to set draft cache
        accepted_tokens = torch.concat((tokens_buffer[:, :1], draft_tokens[mask_buffer].view(1,-1)), dim=1)
        engine.update_verified_kv(accepted_tokens)
        tokens_buffer[:, :1] = bonus_tokens        

        print(f"settlement accepted tokens: {accept_nums.flatten().item()} + 1 bonus_token")

        eot_condition = ((target_tokens == eot_1) | (target_tokens == eot_2))
        if True in eot_condition:
            eot_index = (eot_condition.view(-1) == True).nonzero(as_tuple=True)[0].item()
            engine.verified_cachelength = engine.verified_cachelength - accept_nums.flatten().item() + eot_index
            num_nodes = num_nodes - accept_nums.flatten().item() + eot_index

    num_gen_tokens = engine.verified_cachelength - input_len

    output = engine.input_tokens[:, input_len:engine.verified_cachelength][0]
    decoded_output = engine.model.tokenizer.decode(output, skip_special_tokens=True)
    
    # # Calculate confidence statistics for this step
    min_confidence_step = float(min(step_confidences)) if step_confidences else 0.0
    avg_confidence_step = float(sum(step_confidences) / len(step_confidences)) if step_confidences else 0.0
    
    # Update global counters
    total_speculate_calls += step_speculate_calls
    total_verify_calls += step_verify_calls
    total_settle_calls += step_settle_calls
    total_budget_switches += step_budget_switches
    total_tokens_generated += num_gen_tokens
    
    # Log step-wise data
    step_data = [
        step, args.dataset, args.prefix_len, args.gamma1, args.gamma2, 
        args.budget1, args.budget2, args.budget2_low, args.confidence_threshold, args.enable_dynamic_budget,
        step_speculate_calls, step_verify_calls, step_settle_calls, 
        step_budget_switches, num_gen_tokens, min_confidence_step, avg_confidence_step
    ]
    
    with open(step_log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(step_data)
    
    # Print dynamic budget statistics
    print(f"\n=== Step {step} Statistics ===")
    print(f"Dynamic budget enabled: {args.enable_dynamic_budget}")
    print(f"Speculate calls: {step_speculate_calls}")
    print(f"Verify calls: {step_verify_calls}")
    print(f"Settle calls: {step_settle_calls}")
    print(f"Budget switches: {step_budget_switches}")
    print(f"Tokens generated: {num_gen_tokens}")
    print(f"Min confidence: {min_confidence_step:.3f}")
    print(f"Avg confidence: {avg_confidence_step:.3f}")
    
    print(f"\n=== Accumulated Statistics (up to step {step}) ===")
    print(f"Total speculate calls: {total_speculate_calls}")
    print(f"Total verify calls: {total_verify_calls}")
    print(f"Total settle calls: {total_settle_calls}")
    print(f"Total budget switches: {total_budget_switches}")
    print(f"Total tokens generated: {total_tokens_generated}")
    
    if args.printoutput:
        print(f"Generated output: {decoded_output}")

    # Cleanup GPU memory after each step to prevent OOM
    print(f"Step {step} completed. Cleaning up GPU memory...")
    engine.cleanup()
    torch.cuda.empty_cache()

# After all steps are completed, store the final accumulated data
# Accumulated output should aggregate across all steps in this run only
final_accumulated_data = [
    step, args.dataset, args.prefix_len, args.gamma1, args.gamma2,
    args.budget1, args.budget2, args.budget2_low, args.confidence_threshold, args.enable_dynamic_budget,
    total_speculate_calls, total_verify_calls, total_settle_calls,
    total_budget_switches, total_tokens_generated
]

with open(accumulated_log_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(final_accumulated_data)

print(f"\n=== Final Accumulated Statistics ===")
print(f"Total speculate calls: {total_speculate_calls}")
print(f"Total verify calls: {total_verify_calls}")
print(f"Total settle calls: {total_settle_calls}")
print(f"Total budget switches: {total_budget_switches}")
print(f"Total tokens generated: {total_tokens_generated}")

# Save histogram data to CSV file
save_histogram_history = True
if save_histogram_history:
    # Set histogram parameters
    HIST_BINS = args.hist_statistics_bins  # Use configurable bins
    HIST_RANGE_MIN = 0
    HIST_RANGE_MAX = 1

    print(f"Creating histograms for draft_top1_top2_diff_data (length: {len(draft_top1_top2_diff_data)})")
    print(f"Creating histograms for reject_token_top1_top2_diff_data (length: {len(reject_token_top1_top2_diff_data)})")
    print(f"Creating histograms for draft_top1_logits_data (length: {len(draft_top1_logits_data)})")
    print(f"Creating histograms for reject_token_top1_logits_data (length: {len(reject_token_top1_logits_data)})")
    
    # Print intermediate verify data statistics
    if args.enable_intermediate_verify:
        for budget_key, data in intermediate_reject_data.items():
            print(f"Creating histograms for {budget_key}_reject_data (length: {len(data)})")
        for budget_key, data in intermediate_reject_top1_logits_data.items():
            print(f"Creating histograms for {budget_key}_reject_top1_logits_data (length: {len(data)})")

    # Create histogram data for top1_top2_diff
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

    # Create histogram data for top1_logits
    draft_top1_logits_hist, range_labels_logits = create_histogram_data(
        draft_top1_logits_data, 
        bins=HIST_BINS, 
        range_min=HIST_RANGE_MIN, 
        range_max=HIST_RANGE_MAX
    )

    reject_top1_logits_hist, _ = create_histogram_data(
        reject_token_top1_logits_data, 
        bins=HIST_BINS, 
        range_min=HIST_RANGE_MIN, 
        range_max=HIST_RANGE_MAX
    )

    # Create intermediate verify histograms
    intermediate_hists = {}
    intermediate_top1_logits_hists = {}
    if args.enable_intermediate_verify:
        for budget_key, data in intermediate_reject_data.items():
            intermediate_hists[budget_key], _ = create_histogram_data(
                data, 
                bins=HIST_BINS, 
                range_min=HIST_RANGE_MIN, 
                range_max=HIST_RANGE_MAX
            )
        for budget_key, data in intermediate_reject_top1_logits_data.items():
            intermediate_top1_logits_hists[budget_key], _ = create_histogram_data(
                data, 
                bins=HIST_BINS, 
                range_min=HIST_RANGE_MIN, 
                range_max=HIST_RANGE_MAX
            )

    # Prepare experiment identifier
    experiment_info = f"run2step_{MODEL}_{args.dataset}_prefix{args.prefix_len}_gamma1_{args.gamma1}_budget{args.budget1}"
    if args.task:
        experiment_info += f"_task{args.task}"

    # Create data for CSV - Draft histogram (top1_top2_diff)
    draft_row = {'experiment': f"{experiment_info}_draft"}
    for i, range_label in enumerate(range_labels):
        draft_row[range_label] = draft_hist[i]

    # Create data for CSV - Reject histogram (top1_top2_diff)
    reject_row = {'experiment': f"{experiment_info}_reject"}
    for i, range_label in enumerate(range_labels):
        reject_row[range_label] = reject_hist[i]

    # Create data for CSV - Draft histogram (top1_logits)
    draft_top1_logits_row = {'experiment': f"{experiment_info}_draft_top1_logits"}
    for i, range_label in enumerate(range_labels_logits):
        draft_top1_logits_row[range_label] = draft_top1_logits_hist[i]

    # Create data for CSV - Reject histogram (top1_logits)
    reject_top1_logits_row = {'experiment': f"{experiment_info}_reject_top1_logits"}
    for i, range_label in enumerate(range_labels_logits):
        reject_top1_logits_row[range_label] = reject_top1_logits_hist[i]

    # Create data for intermediate verify histograms
    rows_list = [draft_row, reject_row, draft_top1_logits_row, reject_top1_logits_row]
    if args.enable_intermediate_verify:
        for budget_key, hist_data in intermediate_hists.items():
            intermediate_row = {'experiment': f"{experiment_info}_{budget_key}_reject"}
            for i, range_label in enumerate(range_labels):
                intermediate_row[range_label] = hist_data[i]
            rows_list.append(intermediate_row)
        for budget_key, hist_data in intermediate_top1_logits_hists.items():
            intermediate_row = {'experiment': f"{experiment_info}_{budget_key}_reject_top1_logits"}
            for i, range_label in enumerate(range_labels_logits):
                intermediate_row[range_label] = hist_data[i]
            rows_list.append(intermediate_row)

    # Create DataFrame
    df_new = pd.DataFrame(rows_list)

    # Save histogram data to CSV (append mode)
    # HISTOGRAM_CSV_PATH = f"/home/juchanlee/MagicDec/figure/confidence/run2step_{MODEL}_{args.dataset}_histogram_data_{args.prefix_len}_new_step60to200.csv"
    HISTOGRAM_CSV_PATH = f"/home/juchanlee/MagicDec/profile/confidence/run2step_{MODEL}_{args.dataset}_histogram_data_{args.prefix_len}.csv"

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
    if file_exists:
        df_existing = df_existing.reindex(columns=desired_columns, fill_value=0) 
    df_combined = df_combined.reindex(columns=desired_columns, fill_value=0)

    # Save the combined data
    df_combined.to_csv(HISTOGRAM_CSV_PATH, index=False)
    print(f"Histogram data saved to: {HISTOGRAM_CSV_PATH}")

    # Print summary
    total_rows_added = 4  # draft, reject, draft_top1_logits, reject_top1_logits
    print(f"\nAdded {total_rows_added} rows (draft, reject, draft_top1_logits, reject_top1_logits) to CSV")
    print(f"Draft top1_top2_diff total count: {draft_hist.sum()}")
    print(f"Reject top1_top2_diff total count: {reject_hist.sum()}")
    print(f"Draft top1_logits total count: {draft_top1_logits_hist.sum()}")
    print(f"Reject top1_logits total count: {reject_top1_logits_hist.sum()}")
    
    # Print intermediate verify summaries
    if args.enable_intermediate_verify:
        for budget_key, hist_data in intermediate_hists.items():
            total_rows_added += 1
            print(f"{budget_key}_reject total count: {hist_data.sum()}")
        for budget_key, hist_data in intermediate_top1_logits_hists.items():
            total_rows_added += 1
            print(f"{budget_key}_reject_top1_logits total count: {hist_data.sum()}")
        print(f"Total rows added including intermediate verify: {total_rows_added}")
    
    print(f"Column headers: {range_labels}")