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

model_path = model2path[args.model_name]
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

print(f"Processing {len(draft_history)} steps from draft history...")

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


def preprocess_input_for_retroinfer(self, input_ids, prompt_format, attn_type, model_path, budget_ratio, estimate_ratio, dataset):
    # inputs = None
    # if dataset == "longbenchv1":
    #   # prompt = prompt_format.format(**data)
    #   # inputs = self.model.tokenizer([prompt], return_tensors="pt", padding=True)
    #   input_ids = inputs.input_ids
    #   self.attention_masks = inputs.attention_mask

    # if dataset == "pg19":
    #   input_ids = data[0].unsqueeze(0) # already preprocessed in convert_pg19_dataset()
    #   self.attention_masks = torch.ones_like(input_ids)
    self.attention_masks = torch.ones_like(input_ids)
    self.attn_config = generate_config(
        model_path, 
        input_ids.shape[1], 
        attn_type,
        budget_ratio=budget_ratio,
        estimate_ratio=estimate_ratio,
    )
    return input_ids

input_ids=[None]*10
for step, batch in tqdm(enumerate(dataset), total=10):
    input_ids[step] = engine.preprocess_input(batch, prompt_format, args.attn_type, model_path, args.budget_ratio, args.estimate_ratio, args.dataset, args.prefix_len)

# Process each step in draft history
for step_idx, step_data in enumerate(draft_history):
    print(f"Processing step {step_idx + 1}/{len(draft_history)}")
    
    # Get original input_ids
    original_input_ids = step_data['input_ids'].to(DEVICE)
    
    # Process the input through RetroInfer preprocessing
    try:
        processed_input_ids = engine.preprocess_input_for_retroinfer(
            {'input_ids': original_input_ids.squeeze(0)},  # Remove batch dimension for preprocessing
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
        original_top1_top2_diff = step_data['draft_iter']['draft_top1_top2_diff'][iter_idx]
        
        # Run RetroInfer speculation for each draft token position
        retroinfer_top1_top2_diffs = []
        
        # Start with the first token from draft_tokens
        current_token = draft_tokens[:, 0].view(-1, 1)  # First draft token
        
        try:
            # Run RetroInfer speculation
            draft_outputs, draft_logits, retroinfer_top1_top2_diff = engine.speculate(
                current_token, 
                args.gamma, 
                profile_clustering=args.profile_clustering, 
                profile_hot_cluster_selection_ratio=True
            )
            
            # Store RetroInfer top1-top2 differences
            if isinstance(retroinfer_top1_top2_diff, list):
                retroinfer_top1_top2_diffs.extend(retroinfer_top1_top2_diff)
            else:
                retroinfer_top1_top2_diffs.append(retroinfer_top1_top2_diff)
            
            # Compare predictions (optional analysis)
            if draft_outputs.shape[1] >= draft_tokens.shape[1]:
                predicted_tokens = draft_outputs[:, :draft_tokens.shape[1]]
                matches = (predicted_tokens == draft_tokens).sum().item()
                total_tokens = draft_tokens.numel()
                matching_predictions += matches
                total_comparisons += total_tokens
                
                print(f"    Token prediction match rate: {matches}/{total_tokens} = {matches/total_tokens*100:.1f}%")
            
        except Exception as e:
            print(f"    Warning: Error during speculation for step {step_idx}, iter {iter_idx}: {e}")
            # Use placeholder data if speculation fails
            retroinfer_top1_top2_diffs = [0.0] * args.gamma
        
        # Store the RetroInfer results
        retroinfer_draft_top1_top2_diff_data.append(retroinfer_top1_top2_diffs)

print(f"\nCompleted processing all draft history steps.")
if total_comparisons > 0:
    overall_match_rate = matching_predictions / total_comparisons
    print(f"Overall token prediction match rate: {matching_predictions}/{total_comparisons} = {overall_match_rate*100:.2f}%")

# Create histogram data for RetroInfer results
HIST_BINS = 10
HIST_RANGE_MIN = 0
HIST_RANGE_MAX = 1

print(f"Creating histogram for RetroInfer draft_top1_top2_diff_data (length: {len(retroinfer_draft_top1_top2_diff_data)})")

retroinfer_draft_hist, range_labels = create_histogram_data(
    retroinfer_draft_top1_top2_diff_data, 
    bins=HIST_BINS, 
    range_min=HIST_RANGE_MIN, 
    range_max=HIST_RANGE_MAX
)

# Prepare experiment identifier
base_name = os.path.basename(args.draft_history_path).replace('_draft_history.pt', '')
experiment_info = f"retroinfer_{base_name}"

# Create data for CSV
retroinfer_row = {'experiment': f"{experiment_info}_draft"}
for i, range_label in enumerate(range_labels):
    retroinfer_row[range_label] = retroinfer_draft_hist[i]

# Create DataFrame
df_new = pd.DataFrame([retroinfer_row])

# Define the desired column order: experiment first, then sorted range labels
desired_columns = ['experiment'] + sorted(range_labels, key=lambda x: float(x.split('-')[0]))

# Apply consistent ordering
df_new = df_new.reindex(columns=desired_columns, fill_value=0)

# Save histogram data to CSV
model_name = args.model_name.split("/", 1)[1]
HISTOGRAM_CSV_PATH = f"/home/juchanlee/MagicDec/output/retroinfer_{model_name}_{args.dataset}_histogram_data.csv"

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(HISTOGRAM_CSV_PATH), exist_ok=True)

# Check if file exists to determine if we need headers
file_exists = os.path.exists(HISTOGRAM_CSV_PATH)

# If file exists, read it and append new data
if file_exists:
    df_existing = pd.read_csv(HISTOGRAM_CSV_PATH)
    # Ensure consistent column order for both dataframes
    df_existing = df_existing.reindex(columns=desired_columns, fill_value=0)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
else:
    df_combined = df_new

# Ensure final DataFrame has the correct column order
df_combined = df_combined.reindex(columns=desired_columns, fill_value=0)

# Save the combined data
df_combined.to_csv(HISTOGRAM_CSV_PATH, index=False)
print(f"RetroInfer histogram data saved to: {HISTOGRAM_CSV_PATH}")

# Print summary
print(f"\nAdded 1 row (RetroInfer draft) to CSV")
print(f"RetroInfer draft total count: {retroinfer_draft_hist.sum()}")
print(f"Column headers: {range_labels}")