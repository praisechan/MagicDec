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
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MagicDec.Engine.RetrievalAttention.benchmark.config import generate_config, parse_attn_args
import json

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
elif args.dataset == "longbenchv1":
    dataset = load_dataset('THUDM/LongBench', TASK, split='test')
else:
    raise ValueError(f"Unknown dataset {args.dataset}")

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
if args.dataset == "pg19":
  num_eval_steps = min(10, len(dataloader))
else:
  num_eval_steps = len(dataloader)

num_gen_token_max = 80
num_gen_tokens = 0
verify_steps = 0
settle_steps = 0
budget_switches = 0  # Track how many times we switch budget

# Store these for dynamic budget adjustment
current_model_path = model_path
current_attn_type = args.attn_type

# Confidence analysis data structures
confidence_ranges = [(i/10, (i+1)/10) for i in range(10)]  # 0-0.1, 0.1-0.2, ..., 0.9-1.0
confidence_changes_by_range = {f"{r[0]:.1f}-{r[1]:.1f}": [] for r in confidence_ranges}
rejected_confidence_changes_by_range = {f"{r[0]:.1f}-{r[1]:.1f}": [] for r in confidence_ranges}

def get_confidence_range_key(confidence_value):
    """Get the range key for a given confidence value"""
    if confidence_value < 0:
        confidence_value = 0
    if confidence_value >= 1:
        confidence_value = 0.999
    
    range_idx = int(confidence_value * 10)
    if range_idx >= 10:
        range_idx = 9
    
    return f"{range_idx/10:.1f}-{(range_idx+1)/10:.1f}"

def record_confidence_changes(draft_confidences, verify_confidences, accept_mask=None, is_rejected_only=False):
    """Record confidence changes between draft and verify models"""
    if draft_confidences is None or verify_confidences is None:
        return
    
    # Convert to numpy arrays for easier handling
    draft_conf = np.array(draft_confidences)
    verify_conf = np.array(verify_confidences)
    
    # Ensure we only process up to the minimum length to avoid meaningless tokens
    min_len = min(len(draft_conf), len(verify_conf))
    draft_conf = draft_conf[:min_len]
    verify_conf = verify_conf[:min_len]
    
    if accept_mask is not None:
        # For rejected tokens, we only care about tokens that were actually processed
        # but need to identify which ones were rejected
        accept_mask = np.array(accept_mask[:min_len])
        
        if is_rejected_only:
            # Find the first rejected token and only process up to that point
            if not accept_mask.all():
                first_reject_idx = np.where(~accept_mask)[0][0]
                # Only record the rejected token itself
                if first_reject_idx < len(draft_conf):
                    draft_val = float(draft_conf[first_reject_idx])
                    verify_val = float(verify_conf[first_reject_idx])
                    confidence_change = draft_val - verify_val
                    
                    range_key = get_confidence_range_key(draft_val)
                    rejected_confidence_changes_by_range[range_key].append(confidence_change)
                    print(f"Rejected token - Draft conf: {draft_val:.3f}, Verify conf: {verify_val:.3f}, Change: {confidence_change:.3f}")
        else:
            # For all tokens, record up to the first rejected token (excluding it)
            if not accept_mask.all():
                first_reject_idx = np.where(~accept_mask)[0][0]
                valid_tokens = min_len if first_reject_idx == 0 else first_reject_idx
            else:
                valid_tokens = min_len
            
            for i in range(valid_tokens):
                draft_val = float(draft_conf[i])
                verify_val = float(verify_conf[i])
                confidence_change = draft_val - verify_val
                
                range_key = get_confidence_range_key(draft_val)
                confidence_changes_by_range[range_key].append(confidence_change)
    else:
        # No accept mask provided, record all tokens
        for i in range(min_len):
            draft_val = float(draft_conf[i])
            verify_val = float(verify_conf[i])
            confidence_change = draft_val - verify_val
            
            range_key = get_confidence_range_key(draft_val)
            confidence_changes_by_range[range_key].append(confidence_change)

# CSV logging setup
log_dir = "logs"
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

for step, batch in tqdm(enumerate(dataset), total=num_eval_steps):
    if step >= num_eval_steps:
        break
    
    # Initialize step-wise counters
    step_speculate_calls = 0
    step_verify_calls = 0
    step_settle_calls = 0
    step_budget_switches = 0
    step_confidences = []  # Store confidence values for this step
    
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
    
    # Store draft confidences for comparison with verify confidences
    stored_draft_confidences = None
    
    while terminal == False:

        settled = False
        verified = False

        # Draft speculation
        draft_outputs, draft_logits, top1_top2_diff = engine.speculate(tokens_buffer[:, 0].view(-1,1), args.gamma1)
        tokens_buffer[:,1:1+args.gamma1] = torch.LongTensor(draft_outputs)
        step_speculate_calls += 1
        
        # Store draft confidences for later comparison
        stored_draft_confidences = top1_top2_diff.copy() if top1_top2_diff is not None else None
        
        # Dynamic budget adjustment based on confidence
        # If all tokens have high confidence (top1_top2_diff > threshold), use lower budget
        current_budget = args.budget2  # default budget
        budget_switched = False  # Track if budget was switched for this speculation
        
        if args.enable_dynamic_budget and top1_top2_diff is not None and len(top1_top2_diff) > 0:
            min_confidence = torch.min(torch.tensor(top1_top2_diff))
            avg_confidence = torch.mean(torch.tensor(top1_top2_diff))
            # Convert tensor values to floats for storage
            step_confidences.extend([float(x) for x in top1_top2_diff])  # Store all confidence values as floats
            
            if min_confidence > args.confidence_threshold:
                # High confidence: use lower budget for verification
                current_budget = args.budget2_low
                budget_switched = True
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
                print("No confidence data available, using default budget")
            else:
                print("Dynamic budget disabled, using default budget")
            
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

        # Check if we can settle or verify
        # if (num_unsettled_tokens + args.gamma1 >= args.gamma2) or (called_verify > 5):
        if num_unsettled_tokens + args.gamma1 >= args.gamma2:
            # If we have enough unsettled tokens or have called verify too many times, settle
            settled = True

            settle_outputs, settle_logits, settle_top1_top2_diff = engine.settle(cached_tokens_buffer.view(-1,1), num_unsettled_tokens+args.gamma1+1)
            target_tokens = torch.LongTensor(settle_outputs).to(DEVICE) #TODO: verify stage should be batch-fashion, but this verify() is auto-regressive.
            step_settle_calls += 1
            
            # Record confidence changes for settlement (use cached draft confidences vs settle confidences)
            if stored_draft_confidences is not None and settle_top1_top2_diff is not None:
                # For settlement, we need to compare with the original draft confidences from when tokens were first speculated
                # This is more complex as we need to track which tokens correspond to which speculation
                print(f"Settlement - comparing draft vs settle confidences")
    

            input_from_start = torch.concat((engine.input_tokens[:, :engine.verified_cachelength], tokens_buffer), dim=1)
            draft_tokens = input_from_start[:, -(num_unsettled_tokens+args.gamma1):]
            flag_accept_matrix = (target_tokens[:, :num_unsettled_tokens+args.gamma1] == draft_tokens)  # shape: (BATCH_SIZE, gamma)

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
            num_nodes += (accept_nums.flatten() + 1)
            
            # Check for termination conditions

            # 1: eot in accepted tokens
            condition = (eot_condition & accept_flags_matrix).any(dim=1, keepdim=True)
            if condition.any():
                terminal = True

            if (bonus_tokens == eot_1).any() or (bonus_tokens == eot_2).any():
                terminal = True

            # 2: reach max tokens
            if args.dataset == "longbenchv1" or args.dataset == "longbenchv1-32k":
                #longbenchv1 does not have fixed prefix len
                if num_nodes.max() - input_len >= num_gen_token_max:
                    terminal = True
            else:
                # Check Number of Nodes + Bonus Token <= max_target_token
                if num_nodes.max() - args.prefix_len >= num_gen_token_max:
                    terminal = True
            # Put Bonus tokens to the tokens buffer, and prepare the variables for next itr
        

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
                eot_index = (eot_condition.view(-1) == True).nonzero(as_tuple=True)[0].item()
                engine.settled_cachelength = engine.settled_cachelength - accept_nums + eot_index

                num_nodes = num_nodes - accept_nums + eot_index
            
        else:
            # If not settled, we need to verify
            verified = True

            if called_verify == 0:
                cached_tokens_buffer = tokens_buffer[:, 0].clone()

            verify_outputs, verify_logits, verify_top1_top2_diff = engine.verify(tokens_buffer[:, 0].view(-1,1), args.gamma1+1)
            target_tokens = torch.LongTensor(verify_outputs).to(DEVICE) #TODO: verify stage should be batch-fashion, but this verify() is auto-regressive.
            step_verify_calls += 1
            called_verify += 1
            
            # Count budget switches only when verify is executed with switched budget
            if budget_switched:
                step_budget_switches += 1

            draft_tokens = tokens_buffer[:, 1:args.gamma1+1]
            flag_accept_matrix = (target_tokens[:, :args.gamma1] == draft_tokens)  # shape: (BATCH_SIZE, gamma)

            eot_condition = ((draft_tokens == eot_1) | (draft_tokens == eot_2))  # shape: (BATCH_SIZE, gamma)

            # Compute accept_flags by considering both the acceptance condition and EOT tokens
            accept_flags_int = (flag_accept_matrix & (~eot_condition)).int()
            accept_flags_cumprod = torch.cumprod(accept_flags_int, dim=1)
            accept_flags_matrix = accept_flags_cumprod.bool()

            # Compute the number of accepted tokens
            accept_nums = accept_flags_matrix.sum(dim=1, keepdim=True)  # shape: (BATCH_SIZE, 1)
            
            # Record confidence changes for verification
            if stored_draft_confidences is not None and verify_top1_top2_diff is not None:
                # Convert accept_flags_matrix to numpy boolean array for the first batch
                accept_mask = accept_flags_matrix[0].cpu().numpy()  # Shape: (gamma1,)
                
                print(f"Verification - Draft confidences: {len(stored_draft_confidences)}, Verify confidences: {len(verify_top1_top2_diff)}")
                print(f"Accept mask: {accept_mask}")
                
                # Record all token confidence changes (up to first rejection)
                record_confidence_changes(
                    stored_draft_confidences, 
                    verify_top1_top2_diff, 
                    accept_mask, 
                    is_rejected_only=False
                )
                
                # Record rejected token confidence changes
                record_confidence_changes(
                    stored_draft_confidences, 
                    verify_top1_top2_diff, 
                    accept_mask, 
                    is_rejected_only=True
                )
            
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

                
            # get accepted token and re-decode to set draft cache (Quest)
            accepted_tokens = torch.concat((tokens_buffer[:, :1], draft_tokens[mask_buffer].view(1,-1)), dim=1)
            engine.update_verified_kv(accepted_tokens)

            tokens_buffer[:, :1] = bonus_tokens

            print(f"verification accepted tokens: {accept_nums.flatten().item()} + 1 bonus token")
            print(f"total unsettled tokens: {num_unsettled_tokens}")

            # if terminal -> fast track to settle
            if terminal:
                print("Terminal occured in verification: Fast Track to Settlement")
                settled = True

                target_tokens = torch.LongTensor(engine.settle(cached_tokens_buffer.view(-1,1), num_unsettled_tokens+1)).to(DEVICE) #TODO: verify stage should be batch-fashion, but this verify() is auto-regressive. 
                step_settle_calls += 1

                input_from_start = torch.concat((engine.input_tokens[:, :engine.verified_cachelength], tokens_buffer), dim=1)
                draft_tokens = input_from_start[:, -(num_unsettled_tokens):]
                flag_accept_matrix = (target_tokens[:, :num_unsettled_tokens] == draft_tokens)  # shape: (BATCH_SIZE, gamma)

                eot_condition = ((draft_tokens == eot_1) | (draft_tokens == eot_2))  # shape: (BATCH_SIZE, gamma)

                # Compute accept_flags by considering both the acceptance condition and EOT tokens
                accept_flags_int = (flag_accept_matrix & (~eot_condition)).int()
                accept_flags_cumprod = torch.cumprod(accept_flags_int, dim=1)
                accept_flags_matrix = accept_flags_cumprod.bool()

                # Compute the number of accepted tokens
                accept_nums = accept_flags_matrix.sum(dim=1, keepdim=True)  # shape: (BATCH_SIZE, 1)
                
                positions_buffer = torch.arange(num_unsettled_tokens, device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
                mask_buffer = positions_buffer<accept_nums.view(-1,1)

                # Get the bonus tokens
                indices = accept_nums
                bonus_tokens = target_tokens.gather(1, indices)
                num_nodes += (accept_nums.flatten() + 1)
                
                # Check for termination conditions

                # 1: eot in accepted tokens
                condition = (eot_condition & accept_flags_matrix).any(dim=1, keepdim=True)
                if condition.any():
                    terminal = True

                if (bonus_tokens == eot_1).any() or (bonus_tokens == eot_2).any():
                    terminal = True

                # 2: reach max tokens
                if args.dataset == "longbenchv1" or args.dataset == "longbenchv1-32k":
                    #longbenchv1 does not have fixed prefix len
                    if num_nodes.max() - input_len >= num_gen_token_max:
                        terminal = True
                else:
                    # Check Number of Nodes + Bonus Token <= max_target_token
                    if num_nodes.max() - args.prefix_len >= num_gen_token_max:
                        terminal = True
                # Put Bonus tokens to the tokens buffer, and prepare the variables for next itr


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
                    eot_index = (eot_condition.view(-1) == True).nonzero(as_tuple=True)[0].item()
                    engine.settled_cachelength = engine.settled_cachelength - accept_nums + eot_index

                    num_nodes = num_nodes - accept_nums + eot_index


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

# Generate histograms for confidence changes
def create_histograms():
    """Create histograms for confidence changes by range"""
    # Create output directory for plots
    plots_dir = "confidence_analysis_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create histograms for all tokens
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Confidence Changes (Draft - Verify) for All Tokens by Draft Confidence Range')
    
    for i, (range_key, changes) in enumerate(confidence_changes_by_range.items()):
        row = i // 5
        col = i % 5
        ax = axes[row, col]
        
        if changes:
            ax.hist(changes, bins=20, alpha=0.7, edgecolor='black')
            ax.set_title(f'Range {range_key}\n(n={len(changes)})')
            ax.set_xlabel('Confidence Change')
            ax.set_ylabel('Frequency')
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            
            # Add statistics
            mean_change = np.mean(changes)
            ax.text(0.05, 0.95, f'Mean: {mean_change:.3f}', transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Range {range_key}\n(n=0)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'all_tokens_confidence_changes_by_range.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create histograms for rejected tokens only
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Confidence Changes (Draft - Verify) for Rejected Tokens by Draft Confidence Range')
    
    for i, (range_key, changes) in enumerate(rejected_confidence_changes_by_range.items()):
        row = i // 5
        col = i % 5
        ax = axes[row, col]
        
        if changes:
            ax.hist(changes, bins=20, alpha=0.7, edgecolor='black', color='red')
            ax.set_title(f'Range {range_key}\n(n={len(changes)})')
            ax.set_xlabel('Confidence Change')
            ax.set_ylabel('Frequency')
            ax.axvline(x=0, color='darkred', linestyle='--', alpha=0.5)
            
            # Add statistics
            mean_change = np.mean(changes)
            ax.text(0.05, 0.95, f'Mean: {mean_change:.3f}', transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Range {range_key}\n(n=0)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'rejected_tokens_confidence_changes_by_range.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save raw data to CSV files
    # All tokens data
    all_tokens_data = []
    for range_key, changes in confidence_changes_by_range.items():
        for change in changes:
            all_tokens_data.append([range_key, change])
    
    with open(os.path.join(plots_dir, 'all_tokens_confidence_changes.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['draft_confidence_range', 'confidence_change'])
        writer.writerows(all_tokens_data)
    
    # Rejected tokens data
    rejected_tokens_data = []
    for range_key, changes in rejected_confidence_changes_by_range.items():
        for change in changes:
            rejected_tokens_data.append([range_key, change])
    
    with open(os.path.join(plots_dir, 'rejected_tokens_confidence_changes.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['draft_confidence_range', 'confidence_change'])
        writer.writerows(rejected_tokens_data)
    
    # Print summary statistics
    print(f"\n=== Confidence Change Analysis Summary ===")
    print(f"Results saved to: {plots_dir}/")
    
    print(f"\nAll tokens confidence changes by range:")
    for range_key, changes in confidence_changes_by_range.items():
        if changes:
            mean_change = np.mean(changes)
            std_change = np.std(changes)
            print(f"  {range_key}: n={len(changes)}, mean={mean_change:.3f}, std={std_change:.3f}")
        else:
            print(f"  {range_key}: n=0")
    
    print(f"\nRejected tokens confidence changes by range:")
    for range_key, changes in rejected_confidence_changes_by_range.items():
        if changes:
            mean_change = np.mean(changes)
            std_change = np.std(changes)
            print(f"  {range_key}: n={len(changes)}, mean={mean_change:.3f}, std={std_change:.3f}")
        else:
            print(f"  {range_key}: n=0")

# Create the histograms and analysis
create_histograms()
