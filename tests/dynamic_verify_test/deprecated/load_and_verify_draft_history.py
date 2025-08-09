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
parser.add_argument('--draft_budget', type=int, default=4097, help='Dataset end index.')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')

parser.add_argument('--gamma', type=int, default=16, help='gamma value')

parser.add_argument('--B', type=int, default=1, help='Batch size.')
parser.add_argument('--prefix_len', type=int, default=32800, help='Prefix length')

parser.add_argument('--seed', type=int, default=123, help='Random seed.')

parser.add_argument('--task', type=str, default="gov_report", help='for longbenchv1.')
parser.add_argument("--attn_type", type=str, default="RetroInfer", help="Attention method")
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
file_name = f"{MODEL}_{args.dataset}_prefix{args.prefix_len}_gamma{args.gamma}_budget{args.draft_budget}_draft_history.pt"
args.draft_history_path = Path(args.draft_history_path) / file_name
print(f"Loading draft history from: {args.draft_history_path}")
draft_history = torch.load(args.draft_history_path, map_location='cpu')

# # Load metadata if available
# metadata_path = args.draft_history_path.replace('_draft_history.pt', '_metadata.pt')
# if os.path.exists(metadata_path):
#     metadata = torch.load(metadata_path, map_location='cpu')
#     print(f"Loaded metadata: {metadata['experiment_config']}")

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
  dataset = load_dataset('emozilla/pg19', split='test')
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
  num_eval_steps = 10
else:
  num_eval_steps = len(dataset)

# Process each step in draft history
BUDGET_RATIOS = [0.2, 0.1, 0.05, 0.02]  # High and low budget ratios

# Initialize storage for analysis results
similarity_results = []
bonus_token_comparison_results = []
first_reject_analysis = []
top1_top2_diff_range_similarity_results = []  # New storage for range-based similarity analysis

for step_idx, step_data in enumerate(draft_history):
    if step_idx >= num_eval_steps:
        break
    print(f"Processing step {step_idx}/{len(draft_history)}")
    if step_idx != 8:
        continue
    
    original_input_ids = step_data['input_ids'].to(DEVICE)
    first_token = step_data['first_token'].to(DEVICE)
    concatenated_accepted_tokens = torch.cat((original_input_ids, first_token), dim=1)
    
    # Process each draft iteration in this step
    for iter_idx, draft_iter in enumerate(step_data['draft_iter']['draft_tokens']):
        print(f"  Processing draft iteration {iter_idx}")
        if iter_idx == 6:
          breakpoint()
        
        draft_tokens = draft_iter.to(DEVICE)  # Shape: (batch_size, gamma)
        original_top1_top2_diff = step_data['draft_iter']['draft_top1_top2_diff'][iter_idx]

        accept_matrix_results = {}
        predicted_outputs_results = {}
        rejected_info_results = {}

        # Run speculation for both budget ratios
        for budget_ratio in BUDGET_RATIOS:
            # Update engine config for each budget_ratio
            engine.attention_masks = torch.ones_like(concatenated_accepted_tokens)
            engine.attn_config = generate_config(
                model_path, 
                concatenated_accepted_tokens.shape[1], 
                args.attn_type,
                budget_ratio=budget_ratio,
                estimate_ratio=args.estimate_ratio,
            )
            
            try:
                # Get predicted outputs using engine.speculate (these are target outputs)
                predicted_outputs, predicted_logits, retroinfer_top1_top2_diff = engine.speculate(
                    concatenated_accepted_tokens, 
                    args.gamma, 
                    profile_clustering=args.profile_clustering, 
                )
                
                # predicted_outputs is a 2D list, convert to tensor and slice
                predicted_tokens = torch.tensor(predicted_outputs, device=DEVICE)[:, :draft_tokens.shape[1]]
                predicted_outputs_results[budget_ratio] = predicted_tokens
                
                # Build accept_matrix: 1 if predicted == draft, else 0
                accept_matrix = (predicted_tokens == draft_tokens).int().cpu().tolist()[0]
                accept_matrix_results[budget_ratio] = accept_matrix

                # Find first rejected token and record info
                first_reject_idx = None
                for i, accepted in enumerate(accept_matrix):
                    if not accepted:
                        first_reject_idx = i
                        break
                
                if first_reject_idx is not None:
                    # Bonus token is the predicted token at first reject position
                    bonus_token = predicted_tokens[0, first_reject_idx].item()
                    
                    rejected_info = {
                        'index': first_reject_idx,
                        'bonus_token': bonus_token,
                        'original_top1_top2_diff': float(original_top1_top2_diff[first_reject_idx])
                    }
                    rejected_info_results[budget_ratio] = rejected_info

            except Exception as e:
                print(f"    Warning: Error during speculation for budget_ratio={budget_ratio}: {e}")
                accept_matrix_results[budget_ratio] = [0] * args.gamma
                predicted_outputs_results[budget_ratio] = torch.zeros((1, args.gamma), device=DEVICE, dtype=torch.long)
                rejected_info_results[budget_ratio] = None

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

        # 2. Store first reject analysis for each budget
        for budget_ratio in BUDGET_RATIOS:
            if budget_ratio in rejected_info_results and rejected_info_results[budget_ratio] is not None:
                reject_info = rejected_info_results[budget_ratio]
                first_reject_analysis.append({
                    'step': step_idx,
                    'iter': iter_idx,
                    'budget_ratio': budget_ratio,
                    'first_reject_idx': reject_info['index'],
                    'bonus_token': reject_info['bonus_token'],
                    'original_top1_top2_diff': reject_info['original_top1_top2_diff']
                })

        # 3. Compare bonus tokens between budget ratios for reject cases
        if len(BUDGET_RATIOS) == 2:
            budget_1, budget_2 = BUDGET_RATIOS
            
            if (budget_1 in rejected_info_results and rejected_info_results[budget_1] is not None and
                budget_2 in rejected_info_results and rejected_info_results[budget_2] is not None):
                
                budget_1_info = rejected_info_results[budget_1]
                budget_2_info = rejected_info_results[budget_2]
                
                same_bonus_token = budget_1_info['bonus_token'] == budget_2_info['bonus_token']
                
                bonus_token_comparison_results.append({
                    'step': step_idx,
                    'iter': iter_idx,
                    'budget_1': budget_1,
                    'budget_2': budget_2,
                    'budget_1_reject_idx': budget_1_info['index'],
                    'budget_2_reject_idx': budget_2_info['index'],
                    'budget_1_bonus_token': budget_1_info['bonus_token'],
                    'budget_2_bonus_token': budget_2_info['bonus_token'],
                    'same_bonus_token': same_bonus_token,
                    'budget_1_original_top1_top2_diff': budget_1_info['original_top1_top2_diff'],
                    'budget_2_original_top1_top2_diff': budget_2_info['original_top1_top2_diff']
                })

        # Update concatenated_accepted_tokens for next iteration
        # This should be done based on the original verification results, not our RetroInfer results
        if 'accepted_tokens' in step_data['draft_iter'] and iter_idx < len(step_data['draft_iter']['accepted_tokens']):
            accepted_tokens = step_data['draft_iter']['accepted_tokens'][iter_idx].to(DEVICE)
            if accepted_tokens.numel() > 0:
                concatenated_accepted_tokens = torch.cat((concatenated_accepted_tokens, accepted_tokens), dim=1)

        # 4. Calculate similarity between target model's accept_matrix and each budget ratio's accept_matrix
        # across different ranges of top1_top2_diff
        if 'accept_flags_matrix' in step_data['draft_iter'] and iter_idx < len(step_data['draft_iter']['accept_flags_matrix']):
            target_accept_matrix = step_data['draft_iter']['accept_flags_matrix'][iter_idx].cpu().tolist()
            if isinstance(target_accept_matrix, list) and len(target_accept_matrix) > 0:
                # If it's a batch, take the first element
                target_accept_matrix = target_accept_matrix[0] if isinstance(target_accept_matrix[0], list) else target_accept_matrix
            
            # Define top1_top2_diff ranges (0.1 unit intervals)
            diff_ranges = [(i * 0.1, (i + 1) * 0.1) for i in range(10)]  # 0.0-0.1, 0.1-0.2, ..., 0.9-1.0
            
            for budget_ratio in BUDGET_RATIOS:
                if budget_ratio in accept_matrix_results:
                    retroinfer_accept_matrix = accept_matrix_results[budget_ratio]
                    
                    # Ensure both matrices have the same length
                    min_len = min(len(target_accept_matrix), len(retroinfer_accept_matrix), len(original_top1_top2_diff))
                    
                    # Calculate similarity for each top1_top2_diff range
                    for range_min, range_max in diff_ranges:
                        # Find indices where top1_top2_diff falls in this range
                        indices_in_range = []
                        for i in range(min_len):
                            diff_val = float(original_top1_top2_diff[i])
                            if range_min <= diff_val < range_max or (range_max == 1.0 and diff_val == 1.0):
                                indices_in_range.append(i)
                        
                        if indices_in_range:  # Only calculate if there are tokens in this range
                            # Calculate similarity for tokens in this range
                            equal_decisions = sum(1 for i in indices_in_range 
                                                if target_accept_matrix[i] == retroinfer_accept_matrix[i])
                            total_decisions = len(indices_in_range)
                            similarity = equal_decisions / total_decisions if total_decisions > 0 else 0
                            
                            top1_top2_diff_range_similarity_results.append({
                                'step': step_idx,
                                'iter': iter_idx,
                                'budget_ratio': budget_ratio,
                                'range_min': range_min,
                                'range_max': range_max,
                                'range_label': f"{range_min:.1f}-{range_max:.1f}",
                                'tokens_in_range': total_decisions,
                                'equal_decisions': equal_decisions,
                                'similarity': similarity
                            })

print("\n=== ANALYSIS RESULTS ===")

# Print similarity analysis
print(f"\n--- Accept Matrix Similarity Analysis ---")
if similarity_results:
    avg_similarity = sum(r['similarity'] for r in similarity_results) / len(similarity_results)
    print(f"Average similarity between budget ratios {BUDGET_RATIOS[0]} and {BUDGET_RATIOS[1]}: {avg_similarity:.4f}")
    
    # Count by similarity ranges
    high_similarity = sum(1 for r in similarity_results if r['similarity'] >= 0.8)
    medium_similarity = sum(1 for r in similarity_results if 0.5 <= r['similarity'] < 0.8)
    low_similarity = sum(1 for r in similarity_results if r['similarity'] < 0.5)
    
    print(f"High similarity (>=0.8): {high_similarity}/{len(similarity_results)} ({high_similarity/len(similarity_results)*100:.1f}%)")
    print(f"Medium similarity (0.5-0.8): {medium_similarity}/{len(similarity_results)} ({medium_similarity/len(similarity_results)*100:.1f}%)")
    print(f"Low similarity (<0.5): {low_similarity}/{len(similarity_results)} ({low_similarity/len(similarity_results)*100:.1f}%)")

# Print bonus token comparison
print(f"\n--- Bonus Token Comparison Analysis ---")
if bonus_token_comparison_results:
    same_bonus_count = sum(1 for r in bonus_token_comparison_results if r['same_bonus_token'])
    total_bonus_comparisons = len(bonus_token_comparison_results)
    print(f"Same bonus tokens: {same_bonus_count}/{total_bonus_comparisons} ({same_bonus_count/total_bonus_comparisons*100:.1f}%)")

# Print first reject analysis
print(f"\n--- First Reject Analysis ---")
for budget_ratio in BUDGET_RATIOS:
    budget_rejects = [r for r in first_reject_analysis if r['budget_ratio'] == budget_ratio]
    if budget_rejects:
        avg_reject_idx = sum(r['first_reject_idx'] for r in budget_rejects) / len(budget_rejects)
        avg_top1_top2_diff = sum(r['original_top1_top2_diff'] for r in budget_rejects) / len(budget_rejects)
        print(f"Budget {budget_ratio}: {len(budget_rejects)} rejections, avg first reject index: {avg_reject_idx:.2f}, avg top1-top2 diff: {avg_top1_top2_diff:.4f}")

# # Print top1_top2_diff range similarity analysis
# print(f"\n--- Target vs RetroInfer Accept Matrix Similarity by Top1-Top2 Diff Range ---")
# if top1_top2_diff_range_similarity_results:
#     # Group by budget ratio and range
#     for budget_ratio in BUDGET_RATIOS:
#         print(f"\nBudget ratio {budget_ratio}:")
#         budget_results = [r for r in top1_top2_diff_range_similarity_results if r['budget_ratio'] == budget_ratio]
        
#         if budget_results:
#             # Group by range
#             range_groups = {}
#             for result in budget_results:
#                 range_label = result['range_label']
#                 if range_label not in range_groups:
#                     range_groups[range_label] = []
#                 range_groups[range_label].append(result)
            
#             # Calculate average similarity for each range
#             for range_label in sorted(range_groups.keys()):
#                 range_results = range_groups[range_label]
#                 total_tokens = sum(r['tokens_in_range'] for r in range_results)
#                 total_equal = sum(r['equal_decisions'] for r in range_results)
#                 avg_similarity = total_equal / total_tokens if total_tokens > 0 else 0
                
#                 print(f"  Range {range_label}: {len(range_results)} comparisons, {total_tokens} total tokens, similarity: {avg_similarity:.4f}")
# else:
#     print("No top1_top2_diff range similarity data available")

# Create accumulated results for CSV export
print(f"\n--- Creating Accumulated Results ---")
accumulated_range_similarity_results = []

if top1_top2_diff_range_similarity_results:
    # Define all possible ranges
    diff_ranges = [(i * 0.1, (i + 1) * 0.1) for i in range(10)]
    
    for budget_ratio in BUDGET_RATIOS:
        budget_results = [r for r in top1_top2_diff_range_similarity_results if r['budget_ratio'] == budget_ratio]
        
        if budget_results:
            # Initialize row data
            row_data = {
                'dataset': args.dataset,
                'prefix_len': args.prefix_len,
                'gamma': args.gamma,
                'draft_budget': args.draft_budget,
                'budget_ratio': budget_ratio,
            }
            
            # Group by range and accumulate
            range_groups = {}
            for result in budget_results:
                range_label = result['range_label']
                if range_label not in range_groups:
                    range_groups[range_label] = {'total_tokens': 0, 'total_equal': 0}
                range_groups[range_label]['total_tokens'] += result['tokens_in_range']
                range_groups[range_label]['total_equal'] += result['equal_decisions']
            
            # Add similarity for each range to the row
            for range_min, range_max in diff_ranges:
                range_label = f"{range_min:.1f}-{range_max:.1f}"
                if range_label in range_groups:
                    group = range_groups[range_label]
                    similarity = group['total_equal'] / group['total_tokens'] if group['total_tokens'] > 0 else 0
                    row_data[f'similarity_{range_label}'] = similarity
                    row_data[f'tokens_{range_label}'] = group['total_tokens']
                else:
                    row_data[f'similarity_{range_label}'] = 0
                    row_data[f'tokens_{range_label}'] = 0
            
            accumulated_range_similarity_results.append(row_data)

# Save results to files
results_dir = Path(args.draft_history_path).parent
similarity_df = pd.DataFrame(similarity_results)
bonus_df = pd.DataFrame(bonus_token_comparison_results) 
reject_df = pd.DataFrame(first_reject_analysis)
# range_similarity_df = pd.DataFrame(top1_top2_diff_range_similarity_results)
accumulated_range_similarity_df = pd.DataFrame(accumulated_range_similarity_results)

# For accumulated_range_similarity_analysis.csv, append to existing file or create with header
accumulated_csv_path = results_dir / "accumulated_range_similarity_analysis.csv"
if accumulated_csv_path.exists():
    # Append without header
    accumulated_range_similarity_df.to_csv(accumulated_csv_path, mode='a', header=False, index=False)
else:
    # Create new file with header
    accumulated_range_similarity_df.to_csv(accumulated_csv_path, index=False)

# Other files - overwrite (or you can change these to append too if needed)
similarity_df.to_csv(results_dir / "similarity_analysis.csv", index=False)
bonus_df.to_csv(results_dir / "bonus_token_analysis.csv", index=False)
reject_df.to_csv(results_dir / "first_reject_analysis.csv", index=False)

print(f"\nResults saved to {results_dir}")
print("- similarity_analysis.csv")
print("- bonus_token_analysis.csv") 
print("- first_reject_analysis.csv")
print("- accumulated_range_similarity_analysis.csv (appended)" if accumulated_csv_path.exists() else "- accumulated_range_similarity_analysis.csv (created)")