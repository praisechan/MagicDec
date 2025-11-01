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
from datasets import load_dataset
# from MagicDec.Engine.SnapKV.backend import LMBackend
from MagicDec.Engine.RetrievalAttention.backend_for_3stage_dynamic_budget import LMBackend_Retro
from datasets import load_dataset
from confidence_analyzer import KLConfidenceAnalyzer

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
parser.add_argument("--confidence_threshold", type=float, default=0.5, help="threshold for top1_top2_diff to use lower budget")
parser.add_argument("--enable_dynamic_budget", action='store_true', help="enable dynamic budget adjustment based on confidence")
parser.add_argument("--kl_threshold", type=float, default=0.08, help="threshold for KL divergence to trigger re-verification with larger budget")
parser.add_argument("--enable_extended_verification", action='store_true', help="enable extended verification mode when KL divergence exceeds threshold")
parser.add_argument("--estimate_ratio", type=float, default=0.25, help="ratio of estimated clusters for RetriveInfer")

# Histogram configuration parameters
parser.add_argument("--hist_num_bins", type=int, default=10, help="number of bins for confidence change histogram")
parser.add_argument("--hist_bin_width", type=float, default=0.1, help="width of each bin for confidence change histogram")
parser.add_argument("--hist_center", type=float, default=0.5, help="center value for histogram ranges (typically 0.5 for confidence)")
parser.add_argument("--hist_statistics_bins", type=int, default=50, help="number of bins for histogram data in statistics CSV files")

args = parser.parse_args()

# Initialize KL confidence analyzer with configurable histogram parameters
kl_confidence_analyzer = KLConfidenceAnalyzer(
    num_bins=args.hist_num_bins, 
    bin_width=args.hist_bin_width, 
    center=args.hist_center
)

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
  dataset = load_dataset('emozilla/pg19', split='test')
elif args.dataset == "longbenchv1":
    dataset = load_dataset('THUDM/LongBench', TASK, split='test')
else:
    raise ValueError(f"Unknown dataset {args.dataset}")

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
if args.dataset == "pg19":
  num_eval_steps = min(10, len(dataloader))
else:
  num_eval_steps = len(dataloader)

num_gen_token_max = 100
num_gen_tokens = 0

# Store these for dynamic budget adjustment
current_model_path = model_path
current_attn_type = args.attn_type

# CSV logging setup
log_dir = "logs_kl_confidence"
os.makedirs(log_dir, exist_ok=True)

# Simple filenames without timestamp/counter
step_log_file = os.path.join(log_dir, "step_log_kl_confidence.csv")
accumulated_log_file = os.path.join(log_dir, "accumulated_log_kl_confidence.csv")
rejected_tokens_log_file = os.path.join(log_dir, "rejected_tokens_settle_log.csv")

# Initialize step-wise CSV file with headers
step_headers = [
    "step", "dataset", "prefix_len", "gamma1", "gamma2", "budget1", "budget2", 
    "budget2_low", "confidence_threshold", "enable_dynamic_budget", "kl_threshold", "enable_extended_verification", "speculate_calls", "verify_calls", 
    "settle_calls", "budget_switches_step", "tokens_generated", "min_confidence", 
    "avg_confidence"
]

# Initialize accumulated CSV file with headers
accumulated_headers = [
    "step", "dataset", "prefix_len", "gamma1", "gamma2", "budget1", "budget2", 
    "budget2_low", "confidence_threshold", "enable_dynamic_budget", "kl_threshold", "enable_extended_verification", "total_speculate_calls", "total_verify_calls", 
    "total_settle_calls", "total_budget_switches", "total_tokens_generated"
]

# Initialize rejected tokens log file with headers
rejected_tokens_headers = [
    "step", "settle_call_number", "rejected_token_position", "kl_divergence", "draft_confidence", 
    "confidence_bin", "dataset", "prefix_len", "gamma1", "gamma2", "budget1", "budget2"
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

# Create rejected tokens log file only if it doesn't exist
if not os.path.exists(rejected_tokens_log_file):
    with open(rejected_tokens_log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(rejected_tokens_headers)

print(f"Logging to: {step_log_file}")
print(f"Accumulated logging to: {accumulated_log_file}")
print(f"Rejected tokens logging to: {rejected_tokens_log_file}")

# Global counters for accumulated statistics
total_speculate_calls = 0
total_verify_calls = 0
total_settle_calls = 0
total_budget_switches = 0
total_tokens_generated = 0

actual_step = 0
for step, batch in tqdm(enumerate(dataset), total=num_eval_steps):
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
    step_confidences = []  # Store confidence values for this step
    step_settle_call_number = 0  # Track settle calls within this step
    
    # input_ids = batch[0].to(DEVICE)
    terminal = False
    # Expand buffer to handle extended verification (up to 2x gamma1)
    max_verify_length = args.gamma1 * 2 + 1
    tokens_buffer= torch.zeros((BATCH_SIZE, max_verify_length), device=DEVICE).long()

    num_nodes = torch.zeros(BATCH_SIZE,device=DEVICE).long()
    num_nodes += input_ids.shape[1]
    input_len = num_nodes.max()

    tokens_buffer[:, 0] = torch.LongTensor(engine.encode(input_ids)[0])
    torch.cuda.synchronize()
    start = time.perf_counter()

    # record unsettled_tokens
    num_unsettled_tokens = 0
    called_verify = 0
    # Extended verification state tracking
    accumulated_draft_tokens = 0  # Number of drafted tokens accumulated for extended verification
    use_extended_verification = False  # Flag to use extended verification with higher budget
    last_kl_exceeded = False  # Track if last verification exceeded KL threshold
    while not terminal:
        settled = False
        verified = False

        # Draft speculation
        draft_outputs, draft_logits, top1_top2_diff = engine.speculate(tokens_buffer[:, :1], args.gamma1)
        
        # Determine where to place new draft tokens in buffer
        if use_extended_verification and accumulated_draft_tokens > 0:
            # Place new drafts after accumulated ones
            start_pos = accumulated_draft_tokens + 1
            end_pos = start_pos + args.gamma1
            tokens_buffer[:, start_pos:end_pos] = torch.LongTensor(draft_outputs)
            print(f"Placed {args.gamma1} new draft tokens at positions {start_pos} to {end_pos-1}")
        else:
            # Normal placement
            tokens_buffer[:,1:1+args.gamma1] = torch.LongTensor(draft_outputs)
        
        step_speculate_calls += args.gamma1
        
        # Store draft data for KL analysis
        kl_confidence_analyzer.store_draft_data(draft_logits, top1_top2_diff)
        
        # Dynamic budget adjustment based on confidence
        # If all tokens have high confidence (top1_top2_diff > threshold), use lower budget
        current_budget = args.budget2  # default budget
        budget_switched = False  # Track if budget was switched for this speculation
        
        # Determine verification length and budget based on KL threshold from previous cycle
        verify_length = args.gamma1 + 1  # Default verification length
        verification_budget = current_budget  # Default budget

        if args.enable_dynamic_budget and top1_top2_diff is not None and len(top1_top2_diff) > 0:
            min_confidence = torch.min(torch.tensor(top1_top2_diff))
            avg_confidence = torch.mean(torch.tensor(top1_top2_diff))
            # Convert tensor values to floats for storage
            step_confidences.extend([float(x) for x in top1_top2_diff])  # Store all confidence values as floats
                        
            if use_extended_verification:
                # Use extended verification with accumulated tokens and higher budget
                verify_length = min(accumulated_draft_tokens + args.gamma1 + 1, args.gamma1 * 2 + 1)  # Cap at 2x gamma1
                verification_budget = max(args.budget2, current_budget * 1.5)  # Use higher budget
                
                # Update engine with higher budget for extended verification
                engine.update_verification_budget(
                    budget_ratio=verification_budget, 
                    estimate_ratio=args.estimate_ratio,
                    model_path=current_model_path,
                    seq_len=input_ids.shape[1],
                    attn_type=current_attn_type
                )
                print(f"Extended verification: length={verify_length}, budget={verification_budget} (was {current_budget})")
            else:
                if min_confidence > args.confidence_threshold:                
                    # High confidence: use lower budget for verification
                    current_budget = args.budget2_low
                    budget_switched = True
                    step_budget_switches += 1
                    engine.update_verification_budget(
                        budget_ratio=current_budget, 
                        estimate_ratio=args.estimate_ratio,
                        model_path=current_model_path,
                        seq_len=input_ids.shape[1],
                        attn_type=current_attn_type
                    )
                    print(f"High confidence detected (min_diff={min_confidence:.3f}), using lower verification budget: {current_budget}")
                else:
                    # Low confidence: use original budget for verification
                    engine.update_verification_budget(
                        budget_ratio=current_budget, 
                        estimate_ratio=args.estimate_ratio,
                        model_path=current_model_path,
                        seq_len=input_ids.shape[1],
                        attn_type=current_attn_type
                    )
                    print(f"Low confidence detected (min_diff={min_confidence:.3f}), using original verification budget: {current_budget}")
        else:
            # Dynamic budget disabled or no confidence data available - use original budget
            if args.enable_dynamic_budget:
                print(f"Dynamic budget enabled but no confidence data available - using original budget: {current_budget}")
            else:
                print(f"Dynamic budget disabled - using original budget: {current_budget}")
            
            # Still collect confidence data for logging if available
            if top1_top2_diff is not None and len(top1_top2_diff) > 0:
                step_confidences.extend([float(x) for x in top1_top2_diff])
            
            engine.update_verification_budget(
                budget_ratio=current_budget, 
                estimate_ratio=args.estimate_ratio,
                model_path=current_model_path,
                seq_len=input_ids.shape[1],
                attn_type=current_attn_type
            )

        # Always call verify after speculate
        if called_verify == 0:
            cached_tokens_buffer = tokens_buffer[:, 0].clone()
        
        verify_outputs, verify_logits, verify_top1_top2_diff = engine.verify(tokens_buffer[:, :1], verify_length)
        target_tokens = torch.LongTensor(verify_outputs).to(DEVICE) #TODO: verify stage should be batch-fashion, but this verify() is auto-regressive.

        step_verify_calls += 1
        called_verify += 1
        
        # Store verify data for KL analysis
        kl_confidence_analyzer.store_verify_data(verify_logits, verify_top1_top2_diff)

        # Check KL divergence values and determine if extended verification is needed for next cycle
        max_kl_divergence = 0.0
        kl_threshold_exceeded = False
        
        if (kl_confidence_analyzer.current_draft_logits is not None and 
            kl_confidence_analyzer.current_verify_logits is not None):
            # Calculate KL divergences for all tokens
            kl_divergences = []
            min_len = min(len(kl_confidence_analyzer.current_draft_logits), 
                         len(kl_confidence_analyzer.current_verify_logits))
            
            for i in range(min_len):
                draft_logits = kl_confidence_analyzer.current_draft_logits[i]
                verify_logits = kl_confidence_analyzer.current_verify_logits[i]
                kl_div = kl_confidence_analyzer.compute_kl_divergence(draft_logits, verify_logits)
                kl_divergences.append(kl_div)
            
            if kl_divergences:
                max_kl_divergence = max(kl_divergences)
                kl_threshold_exceeded = max_kl_divergence > args.kl_threshold
                print(f"KL divergence check: max={max_kl_divergence:.4f}, threshold={args.kl_threshold}, exceeded={kl_threshold_exceeded}")

        # Handle verification results based on verification length used
        if use_extended_verification:
            # Extended verification was used, process more tokens
            draft_tokens = tokens_buffer[:, 1:verify_length]
            flag_accept_matrix = (target_tokens[:, :verify_length-1] == draft_tokens)
            print(f"Extended verification processed {verify_length-1} draft tokens")
        else:
            # Normal verification
            draft_tokens = tokens_buffer[:, 1:args.gamma1+1]
            flag_accept_matrix = (target_tokens[:, :args.gamma1] == draft_tokens)
        
        eot_condition = ((draft_tokens == eot_1) | (draft_tokens == eot_2))

        accept_flags_int = (flag_accept_matrix & (~eot_condition)).int()
        accept_flags_cumprod = torch.cumprod(accept_flags_int, dim=1)
        accept_flags_matrix = accept_flags_cumprod.bool()
        accept_nums = accept_flags_matrix.sum(dim=1, keepdim=True)
        num_unsettled_tokens += accept_nums.flatten().item() + 1

        verify_tokens_count = verify_length - 1 if use_extended_verification else args.gamma1

        # Update extended verification state for next cycle
        if kl_threshold_exceeded and not terminal and args.enable_extended_verification:
            # KL threshold exceeded, prepare for extended verification next time
            if not use_extended_verification:
                # First time exceeding threshold, start accumulating
                accumulated_draft_tokens = args.gamma1
                use_extended_verification = True
                print(f"KL threshold exceeded, enabling extended verification for next cycle")
            else:
                # Already in extended mode, continue accumulating but don't exceed buffer
                # max_additional = max_verify_length - accumulated_draft_tokens - 1 - args.gamma1
                # if max_additional > 0:
                #     accumulated_draft_tokens += args.gamma1
                #     print(f"KL threshold still exceeded, accumulated tokens: {accumulated_draft_tokens}")
                # else:
                #     print(f"Cannot accumulate more tokens, buffer limit reached")
                #     use_extended_verification = False
                #     accumulated_draft_tokens = 0
                use_extended_verification = False
                accumulated_draft_tokens = 0
        else:
            # KL threshold not exceeded, terminal condition, or extended verification disabled - reset extended verification
            if use_extended_verification:
                if not args.enable_extended_verification:
                    print(f"Extended verification disabled by argument, disabling extended verification")
                else:
                    print(f"KL threshold no longer exceeded, disabling extended verification")
            accumulated_draft_tokens = 0
            use_extended_verification = False
        
        last_kl_exceeded = kl_threshold_exceeded
        
        # Adjust position calculation based on verification length used
        positions_buffer = torch.arange(verify_tokens_count, device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
        mask_buffer = positions_buffer < accept_nums.view(-1,1)
        indices = accept_nums
        bonus_tokens = target_tokens.gather(1, indices)

        # Analyze KL divergences for verify stage & Accumulate data for later settle analysis
        num_accepted = accept_nums.flatten().item()
        print(f"Verify analysis: {num_accepted} tokens accepted, draft_logits_len={len(kl_confidence_analyzer.current_draft_logits) if kl_confidence_analyzer.current_draft_logits else 0}, verify_logits_len={len(kl_confidence_analyzer.current_verify_logits) if kl_confidence_analyzer.current_verify_logits else 0}")
        
        kl_confidence_analyzer.analyze_all_tokens(num_accepted)
        kl_confidence_analyzer.accumulate_data_after_verify(num_accepted)
        print(f"Accumulated data now: {len(kl_confidence_analyzer.accumulated_draft_logits)} tokens")

        # Check for termination conditions
        condition = (eot_condition & accept_flags_matrix).any(dim=1, keepdim=True)
        if condition.any() or (bonus_tokens == eot_1).any() or (bonus_tokens == eot_2).any():
            terminal = True

        # get accepted token and re-decode to set draft cache
        accepted_tokens = torch.concat((tokens_buffer[:, :1], draft_tokens[mask_buffer].view(1,-1)), dim=1)

        if num_unsettled_tokens >= args.gamma2 or called_verify > 2 * (args.gamma2 / args.gamma1) or terminal:
            engine.update_verified_kv(accepted_tokens)
        else:
            if use_extended_verification:
                engine.update_draft_kv_only(accepted_tokens)
            else:
                engine.update_verified_kv(accepted_tokens)
  
        tokens_buffer[:, :1] = bonus_tokens

        print(f"verification accepted tokens: {accept_nums.flatten().item()} + 1 bonus token")
        print(f"total unsettled tokens: {num_unsettled_tokens}")

        # Now, after verify, check if we need to settle
        if num_unsettled_tokens >= args.gamma2 or called_verify > 2 * (args.gamma2 / args.gamma1) or terminal:
            # Settle
            settled = True
            
            # for sanity
            use_extended_verification = False
            accumulated_draft_tokens = 0

            if not terminal:
                # bonus tokens is the last token from verify
                engine.update_verified_kv(tokens_buffer[:,:1])
            else:
                print("Terminal")

            settle_outputs, settle_logits, settle_top1_top2_diff = engine.settle(cached_tokens_buffer.view(-1,1), num_unsettled_tokens+1)
            target_tokens = torch.LongTensor(settle_outputs).to(DEVICE) #TODO: verify stage should be batch-fashion, but this verify() is auto-regressive.
            
            step_settle_calls += 1
            step_settle_call_number += 1

            # input_from_start = torch.concat((engine.input_tokens[:, :engine.verified_cachelength], tokens_buffer), dim=1)
            input_from_start = engine.input_tokens[:, :engine.verified_cachelength]
            draft_tokens = input_from_start[:, -(num_unsettled_tokens):]
            flag_accept_matrix = (target_tokens[:, :num_unsettled_tokens] == draft_tokens)
            eot_condition = ((draft_tokens == eot_1) | (draft_tokens == eot_2))
            accept_flags_int = (flag_accept_matrix & (~eot_condition)).int()
            accept_flags_cumprod = torch.cumprod(accept_flags_int, dim=1)
            accept_flags_matrix = accept_flags_cumprod.bool()
            accept_nums = accept_flags_matrix.sum(dim=1, keepdim=True)
            
            # For settlement, we need to analyze KL divergences for rejected tokens
            # We use the accumulated draft and verify data from the speculation cycles
            settle_accepted = accept_nums.flatten().item()
            
            # Analyze rejected tokens in settle stage using accumulated data
            kl_confidence_analyzer.analyze_rejected_tokens_settle(
                settle_accepted, 
                step=step, 
                settle_call_number=step_settle_call_number,
                log_file_path=rejected_tokens_log_file,
                dataset=args.dataset,
                prefix_len=args.prefix_len,
                gamma1=args.gamma1,
                gamma2=args.gamma2,
                budget1=args.budget1,
                budget2=args.budget2
            )
            
            print(f"Settle analysis: {settle_accepted} tokens accepted out of {len(kl_confidence_analyzer.accumulated_draft_logits)} accumulated tokens")
            
            positions_buffer = torch.arange(num_unsettled_tokens, device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
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

            accepted_tokens = torch.concat((cached_tokens_buffer.view(1,-1), draft_tokens[mask_buffer].view(1,-1)), dim=1)
            engine.update_settled_kv(accepted_tokens)
            tokens_buffer[:, :1] = bonus_tokens

            # Reset accumulated data after settlement
            kl_confidence_analyzer.reset_accumulated_data()

            print(f"settlement accepted tokens: {accept_nums.flatten().item()} + 1 bonus token")
            
            # record unsettled_tokens
            num_unsettled_tokens = 0
            called_verify = 0

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
    total_tokens_generated += num_gen_tokens
    
    # Log step-wise data
    step_data = [
        step, args.dataset, args.prefix_len, args.gamma1, args.gamma2, 
        args.budget1, args.budget2, args.budget2_low, args.confidence_threshold, args.enable_dynamic_budget, args.kl_threshold, args.enable_extended_verification,
        step_speculate_calls, step_verify_calls, step_settle_calls, 
        step_budget_switches, num_gen_tokens, min_confidence_step, avg_confidence_step
    ]
    
    with open(step_log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(step_data)
    
    # Print dynamic budget statistics
    print(f"\n=== Step {step} Statistics ===")
    print(f"Dynamic budget enabled: {args.enable_dynamic_budget}")
    print(f"KL threshold for re-verification: {args.kl_threshold}")
    print(f"Extended verification enabled: {args.enable_extended_verification}")
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

# After all steps are completed, store the final accumulated data
# Accumulated output should aggregate across all steps in this run only
final_accumulated_data = [
    step, args.dataset, args.prefix_len, args.gamma1, args.gamma2,
    args.budget1, args.budget2, args.budget2_low, args.confidence_threshold, args.enable_dynamic_budget, args.kl_threshold, args.enable_extended_verification,
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

# Save KL confidence analysis results
print(f"\n=== Saving KL Confidence Analysis Results ===")

# Create filename prefix with model configuration
filename_prefix = f"{MODEL}_{args.dataset}_prefix{args.prefix_len}_gamma1{args.gamma1}_budget1{args.budget1}_budget2{args.budget2}_klthresh{args.kl_threshold}_extverify{args.enable_extended_verification}"

kl_confidence_analyzer.save_histograms("kl_confidence_analysis", filename_prefix)
kl_confidence_analyzer.save_statistics("kl_confidence_analysis", filename_prefix, args.hist_statistics_bins)
print(f"KL confidence analysis saved to 'kl_confidence_analysis' directory with prefix: {filename_prefix}")

# Print summary of collected data
print(f"\n=== KL Confidence Analysis Summary ===")
print(f"Histogram configuration: {args.hist_num_bins} bins, width={args.hist_bin_width}, center={args.hist_center}")
print(f"Bin range: [{kl_confidence_analyzer.bin_ranges[0][0]:.2f}, {kl_confidence_analyzer.bin_ranges[-1][1]:.2f})")
total_all_tokens = sum(len(data) for data in kl_confidence_analyzer.all_tokens_kl_data.values())
total_rejected_tokens = sum(len(data) for data in kl_confidence_analyzer.rejected_tokens_kl_data.values())
print(f"Total tokens analyzed: {total_all_tokens}")
print(f"Total rejected tokens analyzed: {total_rejected_tokens}")
print(f"Total rejected token pairs recorded: {len(kl_confidence_analyzer.rejected_tokens_pairs)}")

for i in range(min(args.hist_num_bins, 20)):  # Limit output to first 20 bins for readability
    all_count = len(kl_confidence_analyzer.all_tokens_kl_data[f"bin_{i}"])
    rejected_count = len(kl_confidence_analyzer.rejected_tokens_kl_data[f"bin_{i}"])
    bin_start, bin_end = kl_confidence_analyzer.bin_ranges[i]
    print(f"Bin {i} ([{bin_start:.2f}, {bin_end:.2f})): All tokens: {all_count}, Rejected tokens: {rejected_count}")

if args.hist_num_bins > 20:
    print(f"... (showing first 20 bins out of {args.hist_num_bins} total bins)")
