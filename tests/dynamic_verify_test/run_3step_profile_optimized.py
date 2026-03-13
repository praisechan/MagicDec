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
from MagicDec.Engine.RetrievalAttention.backend_for_3stage import LMBackend_Retro
from datasets import load_dataset

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MagicDec.Engine.RetrievalAttention.benchmark.config import generate_config, parse_attn_args
import json
import numpy as np
import matplotlib.pyplot as plt

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
parser.add_argument("--budget2_high", type=float, default=0.4, help="higher ratio of budget for verification when confidence is very low")
parser.add_argument("--confidence_threshold", type=float, default=0.5, help="threshold for top1_top2_diff to use lower budget")
parser.add_argument("--confidence_threshold_low", type=float, default=0.1, help="threshold for top1_top2_diff to use higher budget")
parser.add_argument("--enable_dynamic_budget", action='store_true', help="enable dynamic budget adjustment based on confidence")
parser.add_argument("--estimate_ratio", type=float, default=0.25, help="ratio of estimated clusters for RetriveInfer")

# Histogram configuration parameters
parser.add_argument("--hist_num_bins", type=int, default=10, help="number of bins for confidence change histogram")
parser.add_argument("--hist_bin_width", type=float, default=0.1, help="width of each bin for confidence change histogram")
parser.add_argument("--hist_center", type=float, default=0.5, help="center value for histogram ranges (typically 0.0)")
parser.add_argument("--hist_statistics_bins", type=int, default=50, help="number of bins for histogram data in statistics CSV files")
parser.add_argument("--num_eval_steps", type=int, default=None, help="number of evaluation steps to run. If not provided, uses dataset-specific defaults.")

args = parser.parse_args()

# Initialize stage outputs recording
stage_outputs_data = []

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
    dataset = load_dataset('THUDM/LongBench', TASK, split='test', trust_remote_code=True)
else:
    raise ValueError(f"Unknown dataset {args.dataset}")

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
if args.num_eval_steps is not None:
  num_eval_steps = args.num_eval_steps
elif args.dataset == "pg19":
  num_eval_steps = min(10, len(dataloader))
else:
  num_eval_steps = len(dataloader)

num_gen_token_max = 100
num_gen_tokens = 0

# Store these for dynamic budget adjustment
current_model_path = model_path
current_attn_type = args.attn_type

# CSV logging setup
# log_dir = "logs"
# profile_dir = f"/home/juchanlee/MagicDec/profile/data_{args.budget1}/{MODEL}_{args.dataset}_{args.prefix_len}"
profile_dir = f"/home/juchanlee/MagicDec/profile/ICCAD/static_profile"
# profile_dir = f"/home/juchanlee/MagicDec/profile/temp/{MODEL}_{args.dataset}_{args.prefix_len}"
log_dir = profile_dir

os.makedirs(log_dir, exist_ok=True)

# Simple filenames without timestamp/counter
step_log_file = os.path.join(log_dir, "step_log.csv")
accumulated_log_file = os.path.join(log_dir, "accumulated_log.csv")

# Initialize step-wise CSV file with headers
step_headers = [
    "step", "dataset", "prefix_len", "gamma1", "gamma2", "budget1", "budget2", 
    "budget2_low", "budget2_high", "confidence_threshold", "confidence_threshold_low", "enable_dynamic_budget", "speculate_calls", "verify_calls", 
    "settle_calls", "budget_switches_step", "budget_switches_low_step", "budget_switches_high_step", "tokens_generated", "min_confidence", 
    "avg_confidence"
]

# Initialize accumulated CSV file with headers
accumulated_headers = [
    "step", "dataset", "prefix_len", "gamma1", "gamma2", "budget1", "budget2", 
    "budget2_low", "budget2_high", "confidence_threshold", "confidence_threshold_low", "enable_dynamic_budget", "total_speculate_calls", "total_verify_calls", 
    "total_settle_calls", "total_budget_switches", "total_budget_switches_low", "total_budget_switches_high", "total_tokens_generated"
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
total_budget_switches_low = 0
total_budget_switches_high = 0
total_tokens_generated = 0

actual_step = 0
for step, batch in tqdm(enumerate(dataset), total=num_eval_steps):
    # if step < 100:
    #   continue  
    if actual_step >= num_eval_steps:
        break
    input_ids = engine.preprocess_input(batch, prompt_format, args.attn_type, model_path, args.budget1, args.budget2, args.estimate_ratio, args.dataset, args.prefix_len)
    if input_ids is None:
        print(f"Skipping step {step} due to empty input_ids.")
        continue
    actual_step += 1 # increment actual step count only if input_ids is valid
    
    # Initialize step-wise counters
    step_speculate_calls = 0
    step_verify_calls = 0
    step_settle_calls = 0
    step_budget_switches = 0
    step_budget_switches_low = 0
    step_budget_switches_high = 0
    step_confidences = []  # Store confidence values for this step
    
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
    while not terminal:
        settled = False
        verified = False

        # Draft speculation
        draft_outputs, draft_logits, top1_top2_diff = engine.speculate(tokens_buffer[:, :1], args.gamma1, profile_clustering=False, profile_hot_cluster_selection_ratio=False, generate_name=f"{profile_dir}/speculate_{step}_{step_speculate_calls}")
        tokens_buffer[:,1:1+args.gamma1] = torch.LongTensor(draft_outputs)
        step_speculate_calls += args.gamma1
        
        # Record draft outputs
        stage_outputs_data.append({
            "stage": "draft",
            "outputs": draft_outputs.tolist() if hasattr(draft_outputs, 'tolist') else list(draft_outputs)
        })
      
        
        # Dynamic budget adjustment based on confidence
        # If all tokens have high confidence (top1_top2_diff > threshold), use lower budget
        current_budget = args.budget2  # default budget
        budget_switched = False  # Track if budget was switched for this speculation
        pass_verify = False  # Track if budget was switched to low for this speculation
        verify_budget = None

        if args.enable_dynamic_budget and top1_top2_diff is not None and len(top1_top2_diff) > 0:
            min_confidence = torch.min(torch.tensor(top1_top2_diff))
            avg_confidence = torch.mean(torch.tensor(top1_top2_diff))
            # Convert tensor values to floats for storage
            step_confidences.extend([float(x) for x in top1_top2_diff])  # Store all confidence values as floats
            
            if min_confidence < args.confidence_threshold_low:
                # Very low confidence: use higher budget for verification
                budget_switched = True
                step_budget_switches += 1
                step_budget_switches_high += 1
                verify_budget = args.budget2_high
                print(f"Very low confidence detected (min_diff={min_confidence:.3f}), using higher verification budget: {verify_budget}")
            elif min_confidence > args.confidence_threshold:
                # High confidence: use lower budget for verification
                budget_switched = True
                step_budget_switches += 1
                step_budget_switches_low += 1
                verify_budget = args.budget2_low
                pass_verify = True
                print(f"High confidence detected (min_diff={min_confidence:.3f}), using lower verification budget: {verify_budget}")
            else:
                verify_budget = args.budget2
                print(f"Medium confidence detected (min_diff={min_confidence:.3f}), using default verification budget: {verify_budget}")
        else:
            # Dynamic budget disabled or no confidence data available - use original budget
            if args.enable_dynamic_budget:
                print("No confidence data available, using default budget")
            else:
                print("Dynamic budget disabled, using default budget")
            verify_budget = args.budget2            
            # Still collect confidence data for logging if available
            if top1_top2_diff is not None and len(top1_top2_diff) > 0:
                step_confidences.extend([float(x) for x in top1_top2_diff])
            
        engine.update_verification_budget(
            budget_ratio=verify_budget,
            estimate_ratio=args.estimate_ratio,
            model_path=current_model_path,
            seq_len=input_ids.shape[1],
            attn_type=current_attn_type
        )

        # Always call verify after speculate
        if called_verify == 0:
            cached_tokens_buffer = tokens_buffer[:, 0].clone() # bonus token from settle

        if pass_verify:
            # Skip verification and accept all drafted tokens
            step_verify_calls += 1
            called_verify += 1
            
            target_tokens = tokens_buffer[:, 1:args.gamma1+1]
            draft_tokens = tokens_buffer[:, 1:args.gamma1]

            flag_accept_matrix = (target_tokens[:, :args.gamma1-1] == draft_tokens) # use gamma1-1 because last token is bonus token
            eot_condition = ((draft_tokens == eot_1) | (draft_tokens == eot_2))

            accept_flags_int = (flag_accept_matrix & (~eot_condition)).int()
            accept_flags_cumprod = torch.cumprod(accept_flags_int, dim=1)
            accept_flags_matrix = accept_flags_cumprod.bool()
            accept_nums = accept_flags_matrix.sum(dim=1, keepdim=True)
            num_unsettled_tokens += accept_nums.flatten().item() + 1

            bonus_tokens = target_tokens[:,-1]

            # Check for termination conditions
            condition = (eot_condition & accept_flags_matrix).any(dim=1, keepdim=True)
            if condition.any() or (bonus_tokens == eot_1).any() or (bonus_tokens == eot_2).any():
                terminal = True

            # get accepted token and re-decode to set draft cache
            # accepted_tokens = torch.concat((tokens_buffer[:, :1], draft_tokens[mask_buffer].view(1,-1)), dim=1)
            accepted_tokens = tokens_buffer[:, :args.gamma1]
            engine.update_verified_kv(accepted_tokens)
            tokens_buffer[:, :1] = bonus_tokens

            print(f"verification accepted tokens: {accept_nums.flatten().item()} + 1 bonus token")
            print(f"total unsettled tokens: {num_unsettled_tokens}")

        else:
            verify_outputs, verify_logits, verify_top1_top2_diff = engine.verify(tokens_buffer[:, :1], args.gamma1+1, use_first_kv=True, profile_clustering=False, profile_hot_cluster_selection_ratio=False, generate_name=f"{profile_dir}/verify_{step}_{step_verify_calls}")
            target_tokens = torch.LongTensor(verify_outputs).to(DEVICE) #TODO: verify stage should be batch-fashion, but this verify() is auto-regressive.

            # Record verify outputs
            stage_outputs_data.append({
                "stage": "verify",
                "outputs": verify_outputs.tolist() if hasattr(verify_outputs, 'tolist') else list(verify_outputs)
            })

            step_verify_calls += 1
            called_verify += 1

            draft_tokens = tokens_buffer[:, 1:args.gamma1+1]
            flag_accept_matrix = (target_tokens[:, :args.gamma1] == draft_tokens)
            eot_condition = ((draft_tokens == eot_1) | (draft_tokens == eot_2))

            accept_flags_int = (flag_accept_matrix & (~eot_condition)).int()
            accept_flags_cumprod = torch.cumprod(accept_flags_int, dim=1)
            accept_flags_matrix = accept_flags_cumprod.bool()
            accept_nums = accept_flags_matrix.sum(dim=1, keepdim=True)
            num_unsettled_tokens += accept_nums.flatten().item() + 1

            # Analyze confidence changes for verify stage
            num_accepted = accept_nums.flatten().item()

            positions_buffer = torch.arange(args.gamma1, device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
            mask_buffer = positions_buffer < accept_nums.view(-1,1)
            indices = accept_nums
            bonus_tokens = target_tokens.gather(1, indices)

            # Check for termination conditions
            condition = (eot_condition & accept_flags_matrix).any(dim=1, keepdim=True)
            if condition.any() or (bonus_tokens == eot_1).any() or (bonus_tokens == eot_2).any():
                terminal = True

            # if args.dataset == "longbenchv1" or args.dataset == "longbenchv1-32k":
            #     if num_nodes.max() - input_len >= num_gen_token_max:
            #         terminal = True
            # else:
            #     if num_nodes.max() - args.prefix_len >= num_gen_token_max:
            #         terminal = True

            # get accepted token and re-decode to set draft cache
            accepted_tokens = torch.concat((tokens_buffer[:, :1], draft_tokens[mask_buffer].view(1,-1)), dim=1)
            engine.update_verified_kv(accepted_tokens)
            tokens_buffer[:, :1] = bonus_tokens

            print(f"verification accepted tokens: {accept_nums.flatten().item()} + 1 bonus token")
            print(f"total unsettled tokens: {num_unsettled_tokens}")

        # Now, after verify, check if we need to settle
        if num_unsettled_tokens >= args.gamma2 or called_verify > 2 * (args.gamma2 / args.gamma1) or terminal:
            # Settle
            settled = True

            if not terminal:
                # bonus tokens is the last token from verify
                engine.update_verified_kv(tokens_buffer[:,:1])
            else:
                print("Terminal")

            settle_outputs, settle_logits, settle_top1_top2_diff = engine.settle(cached_tokens_buffer.view(-1,1), num_unsettled_tokens+1)
            target_tokens = torch.LongTensor(settle_outputs).to(DEVICE) #TODO: verify stage should be batch-fashion, but this verify() is auto-regressive.
            
            # Record settle outputs
            stage_outputs_data.append({
                "stage": "settle",
                "outputs": settle_outputs.tolist() if hasattr(settle_outputs, 'tolist') else list(settle_outputs)
            })
            
            step_settle_calls += 1

            # input_from_start = torch.concat((engine.input_tokens[:, :engine.verified_cachelength], tokens_buffer), dim=1)
            input_from_start = engine.input_tokens[:, :engine.verified_cachelength]
            draft_tokens = input_from_start[:, -(num_unsettled_tokens):]
            flag_accept_matrix = (target_tokens[:, :num_unsettled_tokens] == draft_tokens)
            eot_condition = ((draft_tokens == eot_1) | (draft_tokens == eot_2))
            accept_flags_int = (flag_accept_matrix & (~eot_condition)).int()
            accept_flags_cumprod = torch.cumprod(accept_flags_int, dim=1)
            accept_flags_matrix = accept_flags_cumprod.bool()
            accept_nums = accept_flags_matrix.sum(dim=1, keepdim=True)
            
            # For settlement, we need to analyze confidence changes for rejected tokens
            # We use the accumulated draft and verify confidences from the speculation cycles
            settle_accepted = accept_nums.flatten().item()
                        
            positions_buffer = torch.arange(num_unsettled_tokens, device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
            mask_buffer = positions_buffer < accept_nums.view(-1,1)
            indices = accept_nums
            bonus_tokens = target_tokens.gather(1, indices)
            num_nodes += (accept_nums.flatten() + 1)

            # Check for termination conditions again
            condition = (eot_condition & accept_flags_matrix).any(dim=1, keepdim=True)
            if condition.any() or (bonus_tokens == eot_1).any() or (bonus_tokens == eot_2).any():
                terminal = True

            if args.dataset == "longbenchv1" or args.dataset == "longbenchv1-32k":
                if num_nodes.max() - input_len >= num_gen_token_max:
                    terminal = True
            else:
                if num_nodes.max() - args.prefix_len >= num_gen_token_max:
                    terminal = True

            accepted_tokens = torch.concat((cached_tokens_buffer.view(1,-1), draft_tokens[mask_buffer].view(1,-1)), dim=1)
            engine.update_settled_kv(accepted_tokens)
            tokens_buffer[:, :1] = bonus_tokens

            # reset counters
            num_unsettled_tokens = 0
            called_verify = 0

            print(f"settlement accepted tokens: {accept_nums.flatten().item()} + 1 bonus_token")
            print(f"total unsettled tokens: {num_unsettled_tokens}")

            eot_condition = ((target_tokens == eot_1) | (target_tokens == eot_2))
            if True in eot_condition:
                eot_index = (eot_condition.view(-1) == True).nonzero(as_tuple=True)[0][0].item()
                engine.settled_cachelength = engine.settled_cachelength - accept_nums.flatten().item() + eot_index
                num_nodes = num_nodes - accept_nums.flatten().item() + eot_index

    num_gen_tokens = engine.settled_cachelength - input_len

    output = engine.settled_input_tokens[:, input_len:engine.settled_cachelength][0]
    decoded_output = engine.model.tokenizer.decode(output, skip_special_tokens=True)
    
    # Calculate confidence statistics for this step
    min_confidence_step = float(min(step_confidences)) if step_confidences else 0.0
    avg_confidence_step = float(sum(step_confidences) / len(step_confidences)) if step_confidences else 0.0
    
    # Update global counters
    total_speculate_calls += step_speculate_calls
    total_verify_calls += step_verify_calls
    total_settle_calls += step_settle_calls
    total_budget_switches += step_budget_switches
    total_budget_switches_low += step_budget_switches_low
    total_budget_switches_high += step_budget_switches_high
    total_tokens_generated += num_gen_tokens
    
    # Log step-wise data
    step_data = [
        step, args.dataset, args.prefix_len, args.gamma1, args.gamma2, 
        args.budget1, args.budget2, args.budget2_low, args.budget2_high, args.confidence_threshold, args.confidence_threshold_low, args.enable_dynamic_budget,
        step_speculate_calls, step_verify_calls, step_settle_calls, 
        step_budget_switches, step_budget_switches_low, step_budget_switches_high, num_gen_tokens, min_confidence_step, avg_confidence_step
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
    print(f"Budget switches: {step_budget_switches} (Low: {step_budget_switches_low}, High: {step_budget_switches_high})")
    print(f"Tokens generated: {num_gen_tokens}")
    print(f"Min confidence: {min_confidence_step:.3f}")
    print(f"Avg confidence: {avg_confidence_step:.3f}")
    
    print(f"\n=== Accumulated Statistics (up to step {step}) ===")
    print(f"Total speculate calls: {total_speculate_calls}")
    print(f"Total verify calls: {total_verify_calls}")
    print(f"Total settle calls: {total_settle_calls}")
    print(f"Total budget switches: {total_budget_switches} (Low: {total_budget_switches_low}, High: {total_budget_switches_high})")
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
    args.budget1, args.budget2, args.budget2_low, args.budget2_high, args.confidence_threshold, args.confidence_threshold_low, args.enable_dynamic_budget,
    total_speculate_calls, total_verify_calls, total_settle_calls,
    total_budget_switches, total_budget_switches_low, total_budget_switches_high, total_tokens_generated
]

with open(accumulated_log_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(final_accumulated_data)

print(f"\n=== Final Accumulated Statistics ===")
print(f"Total speculate calls: {total_speculate_calls}")
print(f"Total verify calls: {total_verify_calls}")
print(f"Total settle calls: {total_settle_calls}")
print(f"Total budget switches: {total_budget_switches} (Low: {total_budget_switches_low}, High: {total_budget_switches_high})")
print(f"Total tokens generated: {total_tokens_generated}")

# Save confidence analysis results
print(f"\n=== Saving Confidence Analysis Results ===")

# Create filename prefix with model configuration
filename_prefix = f"{MODEL}_{args.dataset}_prefix{args.prefix_len}_gamma1{args.gamma1}_budget1{args.budget1}_budget2{args.budget2}"

# Save stage outputs data to JSON file
print(f"\n=== Saving Stage Outputs Data ===")
stage_outputs_file = os.path.join(log_dir, "stage_outputs.json")
with open(stage_outputs_file, 'w') as f:
    json.dump(stage_outputs_data, f, indent=2)
print(f"Stage outputs saved to: {stage_outputs_file}")
print(f"Total stage entries recorded: {len(stage_outputs_data)}")
