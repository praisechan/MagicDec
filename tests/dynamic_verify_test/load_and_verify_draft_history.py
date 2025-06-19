import time
import torch
import sys
sys.path.append("..")
from pathlib import Path
import torch.distributed as dist
from MagicDec.Engine.utils import setup_seed
from MagicDec.Data.data_converter import convert_pg19_dataset, convert_c4_dataset, convert_wiki_dataset, convert_cnn_dataset, convert_longbench_v2_dataset, convert_longbench_v2_sum_dataset, convert_longbench_v1_dataset
from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse
from MagicDec.Engine.RetrievalAttention.backend import LMBackend_Retro
from MagicDec.Engine.RetrievalAttention.benchmark.config import generate_config, parse_attn_args
import json
import os
import pandas as pd
import numpy as np

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

parser = argparse.ArgumentParser(description='Load draft history and verify with RetroInfer')
parser.add_argument('--draft_history_path', type=str, required=True, help='Path to the draft history .pt file')
parser.add_argument('--model_name', type=str, default="llama-3.1-8b", help='model name')
parser.add_argument('--dataset', type=str, default="pg19", help='Dataset name.')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
parser.add_argument('--gamma', type=int, default=16, help='gamma value')
parser.add_argument('--B', type=int, default=1, help='Batch size.')
parser.add_argument('--prefix_len', type=int, default=32800, help='Prefix length')
parser.add_argument('--seed', type=int, default=123, help='Random seed.')
parser.add_argument('--task', type=str, default="gov_report", help='for longbenchv1.')
parser.add_argument("--attn_type", type=str, default="RetroInfer", help="Attention method")
parser.add_argument("--budget_ratio", type=float, default=0.018, help="ratio of budget")
parser.add_argument("--estimate_ratio", type=float, default=0.25, help="ratio of estimated clusters for RetriveInfer")
parser.add_argument("--profile_clustering", action='store_true', help="profile clustering")

args = parser.parse_args()

# Init model parallelism
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
global print

setup_seed(args.seed)
print(f"Using device={DEVICE}")

DTYPE = torch.bfloat16
BATCH_SIZE = args.B

target_dec_len = args.gamma + 1
draft_dec_len = 1

# Load target model
engine = LMBackend_Retro(dtype=DTYPE, device=DEVICE, dec_len=target_dec_len, draft_dec_len=draft_dec_len)

model2path = json.load(open("/home/juchanlee/MagicDec/Engine/RetrievalAttention/benchmark/LongBench/config/model2path.json", "r"))
model2maxlen = json.load(open("/home/juchanlee/MagicDec/Engine/RetrievalAttention/benchmark/LongBench/config/model2maxlen.json", "r"))
dataset2prompt = json.load(open("/home/juchanlee/MagicDec/Engine/RetrievalAttention/benchmark/LongBench/config/dataset2prompt.json", "r"))

MODEL = args.model_name.split("/")[-1]
TASK = args.task

model_path = model2path[MODEL]
max_length = model2maxlen[MODEL]
prompt_format = dataset2prompt[TASK]

engine.load_model(model_path, max_length, DTYPE, "auto", BATCH_SIZE)
vocab_size = engine.model.config.vocab_size
if args.compile:
    engine.compile()

# Load draft history
print(f"Loading draft history from: {args.draft_history_path}")
draft_history = torch.load(args.draft_history_path, map_location='cpu')

# Load metadata if available
metadata_path = args.draft_history_path.replace('_draft_history.pt', '_metadata.pt')
if os.path.exists(metadata_path):
    metadata = torch.load(metadata_path, map_location='cpu')
    print(f"Loaded metadata: {metadata['experiment_config']}")

tokenizer = engine.model.tokenizer
eot_1 = tokenizer.eos_token_id
if tokenizer.unk_token_id is not None:
    eot_2 = tokenizer.unk_token_id
else:
    eot_2 = tokenizer.encode("<|eot_id|>")[-1]
print(f"eot_1: {eot_1}, eot_2: {eot_2}")

# Initialize storage for RetroInfer results
retroinfer_draft_top1_top2_diff_data = []
total_comparisons = 0
matching_predictions = 0

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

if args.dataset == "pg19":
  num_eval_steps = min(10, len(dataset))
else:
  num_eval_steps = len(dataset)

def preprocess_input_for_retroinfer(input_ids, prompt_format, attn_type, model_path, budget_ratio, estimate_ratio, dataset):
    # inputs = None
    # if dataset == "longbenchv1":
    #   # prompt = prompt_format.format(**data)
    #   # inputs = self.model.tokenizer([prompt], return_tensors="pt", padding=True)
    #   input_ids = inputs.input_ids
    #   self.attention_masks = inputs.attention_mask

    # if dataset == "pg19":
    #   input_ids = data[0].unsqueeze(0) # already preprocessed in convert_pg19_dataset()
    #   self.attention_masks = torch.ones_like(input_ids)
    breakpoint()
    engine.attention_masks = torch.ones_like(input_ids)
    engine.attn_config = generate_config(
        model_path, 
        input_ids.shape[1], 
        attn_type,
        budget_ratio=budget_ratio,
        estimate_ratio=estimate_ratio,
    )
    return input_ids

# Process each step in draft history
BUDGET_RATIOS = [0.2, 0.1]  # High and low budget ratios

# Initialize storage for analysis results
similarity_results = []
bonus_token_comparison_results = []
first_reject_analysis = []

for step_idx, step_data in enumerate(draft_history):
    concatenated_accepted_tokens = None
    if step_idx >= num_eval_steps:
        break
    print(f"Processing step {step_idx + 1}/{len(draft_history)}")
    
    original_input_ids = step_data['input_ids'].to(DEVICE)
    first_token = step_data['first_token'].to(DEVICE)
    original_input_ids = torch.cat((original_input_ids, first_token), dim=1)

    try:
        processed_input_ids = preprocess_input_for_retroinfer(
            original_input_ids,
            prompt_format, 
            args.attn_type, 
            model_path, 
            args.budget_ratio, 
            args.estimate_ratio, 
            args.dataset
        )
    except Exception as e:
        print(f"Warning: Could not preprocess input for step {step_idx}, using original input_ids: {e}")
        processed_input_ids = original_input_ids
    
    # Process each draft iteration in this step
    for iter_idx, draft_iter in enumerate(step_data['draft_iter']['draft_tokens']):
        print(f"  Processing draft iteration {iter_idx + 1}")
        
        draft_tokens = draft_iter.to(DEVICE)  # Shape: (batch_size, gamma)
        original_accept_flags = step_data['draft_iter']['accept_flags_matrix'][iter_idx].to(DEVICE)
        original_accepted_tokens = step_data['draft_iter']['accepted_tokens'][iter_idx].to(DEVICE)
        original_top1_top2_diff = step_data['draft_iter']['draft_top1_top2_diff'][iter_idx]

        accept_matrix_results = {}
        predicted_outputs_results = {}
        target_outputs_results = {}
        rejected_info_results = {}

        for budget_idx, budget_ratio in enumerate(BUDGET_RATIOS):
            # Update engine config for each budget_ratio
            engine.attn_config = generate_config(
                model_path, 
                processed_input_ids.shape[1], 
                args.attn_type,
                budget_ratio=budget_ratio,
                estimate_ratio=args.estimate_ratio,
            )
            try:
                # Get both predicted outputs and target outputs for bonus token comparison
                predicted_outputs, predicted_logits, retroinfer_top1_top2_diff = engine.speculate(
                    concatenated_accepted_tokens, 
                    args.gamma, 
                    profile_clustering=args.profile_clustering, 
                )
                
                # For bonus token, we need target verification (similar to snapkv.py)
                tokens_buffer = torch.zeros((BATCH_SIZE, args.gamma+1), device=DEVICE).long()
                tokens_buffer[:, 0] = concatenated_accepted_tokens[:, -1]  # Last accepted token
                tokens_buffer[:, 1:] = predicted_outputs[:, :args.gamma]
                
                target_outputs, target_logits = engine.verify(tokens_buffer)
                
                predicted_tokens = predicted_outputs[:, :draft_tokens.shape[1]]
                predicted_outputs_results[budget_ratio] = predicted_tokens
                target_outputs_results[budget_ratio] = target_outputs
                
                # Build accept_matrix: 1 if predicted == draft, else 0
                accept_matrix = (predicted_tokens == draft_tokens).int().cpu().tolist()[0]
                accept_matrix_results[budget_ratio] = accept_matrix

                # Find rejected tokens and record info
                rejected_indices = [i for i, accepted in enumerate(accept_matrix) if not accepted]
                rejected_info = []
                for idx in rejected_indices:
                    info = {
                        'index': idx,
                        'original_top1_top2_diff': float(original_top1_top2_diff[idx]),
                        'retroinfer_top1_top2_diff': float(retroinfer_top1_top2_diff[idx]) if isinstance(retroinfer_top1_top2_diff, (list, np.ndarray)) else float(retroinfer_top1_top2_diff)
                    }
                    rejected_info.append(info)
                rejected_info_results[budget_ratio] = rejected_info

            except Exception as e:
                print(f"    Warning: Error during speculation for budget_ratio={budget_ratio}: {e}")
                accept_matrix_results[budget_ratio] = [0] * args.gamma
                predicted_outputs_results[budget_ratio] = torch.zeros((1, args.gamma), device=DEVICE, dtype=torch.long)
                target_outputs_results[budget_ratio] = torch.zeros((1, args.gamma), device=DEVICE, dtype=torch.long)
                rejected_info_results[budget_ratio] = []

        # 1. Calculate similarity between accept_matrix_results
        if len(BUDGET_RATIOS) == 2:
            budget_1, budget_2 = BUDGET_RATIOS
            accept_matrix_1 = accept_matrix_results[budget_1]
            accept_matrix_2 = accept_matrix_results[budget_2]
            
            equal_decisions = sum(1 for a1, a2 in zip(accept_matrix_1, accept_matrix_2) if a1 == a2)
            total_decisions = len(accept_matrix_1)
            similarity = equal_decisions / total_decisions if total_decisions > 0 else 0
            
            similarity_results.append({
                'step': step_idx,
                'iter': iter_idx,
                'budget_1': budget_1,
                'budget_2': budget_2,
                'equal_decisions': equal_decisions,
                'total_decisions': total_decisions,
                'similarity': similarity
            })

        # 2. For reject cases, find first reject and compare bonus tokens
        for budget_ratio in BUDGET_RATIOS:
            accept_matrix = accept_matrix_results[budget_ratio]
            predicted_tokens = predicted_outputs_results[budget_ratio]
            target_outputs = target_outputs_results[budget_ratio]
            
            # Find first reject index
            first_reject_idx = None
            for i, accepted in enumerate(accept_matrix):
                if not accepted:
                    first_reject_idx = i
                    break
            
            if first_reject_idx is not None:
                # Get bonus token (from target verification)
                bonus_token = target_outputs[0, first_reject_idx].item()
                
                first_reject_analysis.append({
                    'step': step_idx,
                    'iter': iter_idx,
                    'budget_ratio': budget_ratio,
                    'first_reject_idx': first_reject_idx,
                    'bonus_token': bonus_token,
                    'original_top1_top2_diff': float(original_top1_top2_diff[first_reject_idx])
                })

        # 3. Compare bonus tokens between budget ratios for reject cases
        if len(BUDGET_RATIOS) == 2:
            budget_1, budget_2 = BUDGET_RATIOS
            
            # Get first reject analysis for both budgets
            budget_1_reject = None
            budget_2_reject = None
            
            for analysis in first_reject_analysis:
                if (analysis['step'] == step_idx and analysis['iter'] == iter_idx):
                    if analysis['budget_ratio'] == budget_1:
                        budget_1_reject = analysis
                    elif analysis['budget_ratio'] == budget_2:
                        budget_2_reject = analysis
            
            # If both have rejects, compare bonus tokens
            if budget_1_reject and budget_2_reject:
                same_bonus_token = budget_1_reject['bonus_token'] == budget_2_reject['bonus_token']
                
                bonus_token_comparison_results.append({
                    'step': step_idx,
                    'iter': iter_idx,
                    'budget_1': budget_1,
                    'budget_2': budget_2,
                    'budget_1_reject_idx': budget_1_reject['first_reject_idx'],
                    'budget_2_reject_idx': budget_2_reject['first_reject_idx'],
                    'budget_1_bonus_token': budget_1_reject['bonus_token'],
                    'budget_2_bonus_token': budget_2_reject['bonus_token'],
                    'same_bonus_token': same_bonus_token,
                    'budget_1_original_top1_top2_diff': budget_1_reject['original_top1_top2_diff'],
                    'budget_2_original_top1_top2_diff': budget_2_reject['original_top1_top2_diff']
                })

        concatenated_accepted_tokens = torch.cat((processed_input_ids, original_accepted_tokens), dim=1)

# Print analysis results
print(f"\n=== ANALYSIS RESULTS ===")

# Similarity analysis
if similarity_results:
    total_similarity = sum(r['similarity'] for r in similarity_results)
    avg_similarity = total_similarity / len(similarity_results)
    print(f"Average accept_matrix similarity between budget {BUDGET_RATIOS[0]} and {BUDGET_RATIOS[1]}: {avg_similarity:.4f}")
    
    print(f"Similarity details:")
    for result in similarity_results[:5]:  # Show first 5 as example
        print(f"  Step {result['step']}, Iter {result['iter']}: {result['equal_decisions']}/{result['total_decisions']} = {result['similarity']:.4f}")

# First reject analysis
reject_count_by_budget = {}
for budget_ratio in BUDGET_RATIOS:
    count = sum(1 for r in first_reject_analysis if r['budget_ratio'] == budget_ratio)
    reject_count_by_budget[budget_ratio] = count
    print(f"Budget {budget_ratio}: {count} first rejects")

# Bonus token comparison
if bonus_token_comparison_results:
    same_bonus_count = sum(1 for r in bonus_token_comparison_results if r['same_bonus_token'])
    total_comparisons = len(bonus_token_comparison_results)
    print(f"Bonus token matches: {same_bonus_count}/{total_comparisons} = {same_bonus_count/total_comparisons:.4f}")
    
    print(f"Bonus token comparison details:")
    for result in bonus_token_comparison_results[:5]:  # Show first 5 as example
        print(f"  Step {result['step']}, Iter {result['iter']}: Budget {result['budget_1']} token {result['budget_1_bonus_token']} vs Budget {result['budget_2']} token {result['budget_2_bonus_token']} - Same: {result['same_bonus_token']}")

# Save detailed results
output_dir = "/home/juchanlee/MagicDec/output/budget_comparison"
os.makedirs(output_dir, exist_ok=True)

# Save similarity results
similarity_df = pd.DataFrame(similarity_results)
similarity_path = f"{output_dir}/similarity_results.csv"
similarity_df.to_csv(similarity_path, index=False)
print(f"Similarity results saved to: {similarity_path}")

# Save first reject analysis
first_reject_df = pd.DataFrame(first_reject_analysis)
first_reject_path = f"{output_dir}/first_reject_analysis.csv"
first_reject_df.to_csv(first_reject_path, index=False)
print(f"First reject analysis saved to: {first_reject_path}")

# Save bonus token comparison
if bonus_token_comparison_results:
    bonus_comparison_df = pd.DataFrame(bonus_token_comparison_results)
    bonus_comparison_path = f"{output_dir}/bonus_token_comparison.csv"
    bonus_comparison_df.to_csv(bonus_comparison_path, index=False)
    print(f"Bonus token comparison saved to: {bonus_comparison_path}")

# ...existing code...