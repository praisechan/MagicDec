import time
import torch
import torch.nn.functional as F
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

# Add KL divergence analysis class with confidence binning
class KLConfidenceAnalyzer:
    def __init__(self, num_bins=10, bin_width=0.1, center=0.0):
        self.num_bins = num_bins
        self.bin_width = bin_width
        self.center = center
        
        # Calculate bin ranges centered around the center value
        # For example, with center=0.0, bin_width=0.1, num_bins=20:
        # Bins will be: [-1.0, -0.9), [-0.9, -0.8), ..., [0.9, 1.0)
        half_bins = num_bins // 2
        start_value = center - half_bins * bin_width
        
        self.bin_ranges = []
        self.bin_centers = []
        for i in range(num_bins):
            bin_start = start_value + i * bin_width
            bin_end = bin_start + bin_width
            self.bin_ranges.append((bin_start, bin_end))
            self.bin_centers.append((bin_start + bin_end) / 2)
        
        # For all tokens - KL divergences binned by confidence
        self.all_tokens_kl_data = {f"bin_{i}": [] for i in range(num_bins)}
        
        # For rejected tokens only - store both KL divergence and confidence pairs
        self.rejected_tokens_kl_data = {f"bin_{i}": [] for i in range(num_bins)}
        self.rejected_tokens_pairs = []  # Store (kl_divergence, confidence) pairs
        
        # Temporary storage for current speculation cycle
        self.current_draft_logits = None
        self.current_verify_logits = None
        self.current_draft_confidences = None
        self.current_verify_confidences = None
        
        # Storage for accumulated unsettled tokens (between settlement cycles)
        self.accumulated_draft_logits = []
        self.accumulated_verify_logits = []
        self.accumulated_draft_confidences = []
        self.accumulated_verify_confidences = []
        
    def compute_kl_divergence(self, logits_p, logits_q):
        """
        Compute KL divergence between two logit distributions.
        KL(P||Q) = sum(P * log(P/Q))
        
        Args:
            logits_p: Source distribution logits [1, 1, vocab_size]
            logits_q: Target distribution logits [1, 1, vocab_size]
        Returns:
            KL divergence value (scalar)
        """
        # Convert logits to log probabilities for numerical stability
        log_p = F.log_softmax(logits_p.squeeze(), dim=-1)  # [vocab_size]
        log_q = F.log_softmax(logits_q.squeeze(), dim=-1)  # [vocab_size]
        
        # Use PyTorch's built-in KL divergence which is more numerically stable
        # kl_div expects log probabilities for the first argument and log probabilities for the second
        kl_div = F.kl_div(log_q, log_p, log_target=True, reduction='sum')
        
        return kl_div.item()
        
    def get_bin_index(self, confidence):
        """Get the bin index for a given confidence value"""
        # Find which bin this confidence value falls into
        for i, (bin_start, bin_end) in enumerate(self.bin_ranges):
            if bin_start <= confidence < bin_end:
                return i
        
        # Handle edge cases: values outside the range
        if confidence < self.bin_ranges[0][0]:
            return 0  # Put in first bin
        else:
            return self.num_bins - 1  # Put in last bin
    
    def store_draft_data(self, draft_logits, draft_top1_top2_diff):
        """Store logits and confidences from speculation"""
        if draft_logits is not None:
            self.current_draft_logits = draft_logits
        else:
            self.current_draft_logits = None
            
        if draft_top1_top2_diff is not None:
            self.current_draft_confidences = [float(x) for x in draft_top1_top2_diff]
        else:
            self.current_draft_confidences = None
    
    def store_verify_data(self, verify_logits, verify_top1_top2_diff):
        """Store logits and confidences from verification"""
        if verify_logits is not None:
            self.current_verify_logits = verify_logits
        else:
            self.current_verify_logits = None
            
        if verify_top1_top2_diff is not None:
            self.current_verify_confidences = [float(x) for x in verify_top1_top2_diff]
        else:
            self.current_verify_confidences = None
    
    def accumulate_data_after_verify(self, num_accepted_tokens):
        """Accumulate data after each verify call for later settle analysis"""
        if (self.current_draft_logits is None or 
            self.current_verify_logits is None or
            self.current_draft_confidences is None):
            return
        
        # Only accumulate tokens up to accepted position (plus one for the bonus token)
        # We include the bonus token because it will be part of the unsettled tokens
        max_tokens_to_accumulate = num_accepted_tokens + 1
        
        # Add accepted tokens + bonus token to accumulated storage
        for i in range(max_tokens_to_accumulate):
            if i >= len(self.current_draft_confidences):
                self.accumulated_draft_logits.append(None)
                self.accumulated_draft_confidences.append(None)
            else:
                self.accumulated_draft_logits.append(self.current_draft_logits[i])
                self.accumulated_draft_confidences.append(self.current_draft_confidences[i])
            
            if i < len(self.current_verify_logits):
                self.accumulated_verify_logits.append(self.current_verify_logits[i])
            else:
                self.accumulated_verify_logits.append(None)
                
            if i < len(self.current_verify_confidences):
                self.accumulated_verify_confidences.append(self.current_verify_confidences[i])
            else:
                self.accumulated_verify_confidences.append(None)
    
    def analyze_all_tokens(self, num_accepted_tokens):
        """Analyze KL divergences for all tokens up to accepted position, binned by draft confidence"""
        if (self.current_draft_logits is None or 
            self.current_verify_logits is None or
            self.current_draft_confidences is None):
            return
        
        # Only analyze tokens up to the number of accepted tokens
        min_len = min(len(self.current_draft_logits), 
                     len(self.current_verify_logits),
                     len(self.current_draft_confidences),
                     num_accepted_tokens)
        
        for i in range(min_len):
            draft_logits = self.current_draft_logits[i]
            verify_logits = self.current_verify_logits[i]
            draft_conf = self.current_draft_confidences[i]
            
            # Get bin based on draft confidence
            bin_idx = self.get_bin_index(draft_conf)
            
            # Compute KL divergence between draft and verify logits
            kl_div = self.compute_kl_divergence(draft_logits, verify_logits)
            
            self.all_tokens_kl_data[f"bin_{bin_idx}"].append(kl_div)
    
    def analyze_rejected_tokens_settle(self, num_accepted_tokens):
        """Analyze KL divergences for rejected tokens in settle stage using accumulated data"""
        if (len(self.accumulated_draft_logits) == 0 or 
            len(self.accumulated_verify_logits) == 0 or
            len(self.accumulated_draft_confidences) == 0):
            return
        
        # Find the first rejected token (if any)
        max_tokens = min(len(self.accumulated_draft_logits), 
                        len(self.accumulated_verify_logits),
                        len(self.accumulated_draft_confidences))
        
        if num_accepted_tokens < max_tokens:
            # There is a rejected token at position num_accepted_tokens
            rejected_idx = num_accepted_tokens
            
            draft_logits = self.accumulated_draft_logits[rejected_idx]
            verify_logits = self.accumulated_verify_logits[rejected_idx]
            draft_conf = self.accumulated_draft_confidences[rejected_idx]
            
            if draft_logits is None or verify_logits is None or draft_conf is None:
                print(f"Warning: No data available for rejected token at index {rejected_idx}")
                return
            
            # Get bin based on draft confidence
            bin_idx = self.get_bin_index(draft_conf)
            
            # Compute KL divergence between draft and verify logits for rejected token
            kl_div = self.compute_kl_divergence(draft_logits, verify_logits)
            
            self.rejected_tokens_kl_data[f"bin_{bin_idx}"].append(kl_div)
            
            # Store the pair for raw data output
            self.rejected_tokens_pairs.append((kl_div, draft_conf))
        
    def get_accumulated_stats(self):
        """Get statistics about accumulated data for debugging"""
        return {
            'draft_logits_count': len(self.accumulated_draft_logits),
            'verify_logits_count': len(self.accumulated_verify_logits),
            'draft_conf_count': len(self.accumulated_draft_confidences),
            'verify_conf_count': len(self.accumulated_verify_confidences),
            'current_draft_count': len(self.current_draft_logits) if self.current_draft_logits else 0,
            'current_verify_count': len(self.current_verify_logits) if self.current_verify_logits else 0
        }
        
    def reset_accumulated_data(self):
        """Reset accumulated storage after settlement"""
        self.current_draft_logits = None
        self.current_verify_logits = None
        self.current_draft_confidences = None
        self.current_verify_confidences = None

        self.accumulated_draft_logits = []
        self.accumulated_verify_logits = []
        self.accumulated_draft_confidences = []
        self.accumulated_verify_confidences = []
    
    def save_histograms(self, output_dir="kl_confidence_analysis", filename_prefix=""):
        """Save histograms for each confidence bin"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all tokens histograms for each bin
        for i, (bin_key, data) in enumerate(self.all_tokens_kl_data.items()):
            if len(data) > 0:
                plt.figure(figsize=(10, 6))
                plt.hist(data, bins=50, alpha=0.7, edgecolor='black')
                bin_start, bin_end = self.bin_ranges[i]
                plt.title(f'All Tokens - KL Divergence Distribution\nConfidence Bin {i}: [{bin_start:.2f}, {bin_end:.2f})')
                plt.xlabel('KL Divergence (Draft || Verify)')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                filename = f"{filename_prefix}_all_tokens_{bin_key}_kl.png" if filename_prefix else f'all_tokens_{bin_key}_kl.png'
                plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
                plt.close()
        
        # Save rejected tokens histograms for each bin
        for i, (bin_key, data) in enumerate(self.rejected_tokens_kl_data.items()):
            if len(data) > 0:
                plt.figure(figsize=(10, 6))
                plt.hist(data, bins=50, alpha=0.7, edgecolor='black', color='orange')
                bin_start, bin_end = self.bin_ranges[i]
                plt.title(f'Rejected Tokens - KL Divergence Distribution\nConfidence Bin {i}: [{bin_start:.2f}, {bin_end:.2f})')
                plt.xlabel('KL Divergence (Draft || Verify)')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                filename = f"{filename_prefix}_rejected_tokens_{bin_key}_kl.png" if filename_prefix else f'rejected_tokens_{bin_key}_kl.png'
                plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
                plt.close()
    
    def save_statistics(self, output_dir="kl_confidence_analysis", filename_prefix="", num_histogram_bins=50):
        """Save detailed statistics including histogram data to CSV"""
        os.makedirs(output_dir, exist_ok=True)
        
        # All tokens statistics with histogram data for each bin
        for i, (bin_key, data) in enumerate(self.all_tokens_kl_data.items()):
            if len(data) > 0:
                # Create histogram data
                hist_counts, hist_edges = np.histogram(data, bins=num_histogram_bins)
                hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
                
                # Save histogram data
                hist_filename = f"{filename_prefix}_all_tokens_{bin_key}_histogram.csv" if filename_prefix else f'all_tokens_{bin_key}_histogram.csv'
                with open(os.path.join(output_dir, hist_filename), 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Bin_Center', 'Count', 'Bin_Start', 'Bin_End'])
                    for j in range(len(hist_counts)):
                        writer.writerow([hist_centers[j], hist_counts[j], hist_edges[j], hist_edges[j+1]])
                
                # Save raw data
                raw_filename = f"{filename_prefix}_all_tokens_{bin_key}_raw.csv" if filename_prefix else f'all_tokens_{bin_key}_raw.csv'
                with open(os.path.join(output_dir, raw_filename), 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['KL_Divergence'])
                    for value in data:
                        writer.writerow([value])
        
        # All tokens summary statistics
        all_tokens_filename = f"{filename_prefix}_all_tokens_stats.csv" if filename_prefix else 'all_tokens_stats.csv'
        with open(os.path.join(output_dir, all_tokens_filename), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Bin', 'Range', 'Count', 'Mean', 'Std', 'Min', 'Max', 'Percentile_25', 'Percentile_50', 'Percentile_75'])
            
            for i, (bin_key, data) in enumerate(self.all_tokens_kl_data.items()):
                bin_start, bin_end = self.bin_ranges[i]
                bin_range_str = f"[{bin_start:.2f}, {bin_end:.2f})"
                
                if len(data) > 0:
                    data_array = np.array(data)
                    writer.writerow([
                        i, bin_range_str, len(data), np.mean(data_array), np.std(data_array),
                        np.min(data_array), np.max(data_array),
                        np.percentile(data_array, 25), np.percentile(data_array, 50), np.percentile(data_array, 75)
                    ])
                else:
                    writer.writerow([i, bin_range_str, 0, 0, 0, 0, 0, 0, 0, 0])
        
        # Rejected tokens statistics with histogram data for each bin
        for i, (bin_key, data) in enumerate(self.rejected_tokens_kl_data.items()):
            if len(data) > 0:
                # Create histogram data
                hist_counts, hist_edges = np.histogram(data, bins=num_histogram_bins)
                hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
                
                # Save histogram data
                hist_filename = f"{filename_prefix}_rejected_tokens_{bin_key}_histogram.csv" if filename_prefix else f'rejected_tokens_{bin_key}_histogram.csv'
                with open(os.path.join(output_dir, hist_filename), 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Bin_Center', 'Count', 'Bin_Start', 'Bin_End'])
                    for j in range(len(hist_counts)):
                        writer.writerow([hist_centers[j], hist_counts[j], hist_edges[j], hist_edges[j+1]])
                
                # Save raw data
                raw_filename = f"{filename_prefix}_rejected_tokens_{bin_key}_raw.csv" if filename_prefix else f'rejected_tokens_{bin_key}_raw.csv'
                with open(os.path.join(output_dir, raw_filename), 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['KL_Divergence'])
                    for value in data:
                        writer.writerow([value])
        
        # Save rejected tokens pairs (KL divergence, confidence) in a single file
        if len(self.rejected_tokens_pairs) > 0:
            pairs_filename = f"{filename_prefix}_rejected_tokens_raw.csv" if filename_prefix else 'rejected_tokens_raw.csv'
            with open(os.path.join(output_dir, pairs_filename), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['KL_Divergence', 'Draft_Confidence'])
                for kl_div, confidence in self.rejected_tokens_pairs:
                    writer.writerow([kl_div, confidence])
        
        # Rejected tokens summary statistics
        rejected_tokens_filename = f"{filename_prefix}_rejected_tokens_stats.csv" if filename_prefix else 'rejected_tokens_stats.csv'
        with open(os.path.join(output_dir, rejected_tokens_filename), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Bin', 'Range', 'Count', 'Mean', 'Std', 'Min', 'Max', 'Percentile_25', 'Percentile_50', 'Percentile_75'])
            
            for i, (bin_key, data) in enumerate(self.rejected_tokens_kl_data.items()):
                bin_start, bin_end = self.bin_ranges[i]
                bin_range_str = f"[{bin_start:.2f}, {bin_end:.2f})"
                
                if len(data) > 0:
                    data_array = np.array(data)
                    writer.writerow([
                        i, bin_range_str, len(data), np.mean(data_array), np.std(data_array),
                        np.min(data_array), np.max(data_array),
                        np.percentile(data_array, 25), np.percentile(data_array, 50), np.percentile(data_array, 75)
                    ])
                else:
                    writer.writerow([i, bin_range_str, 0, 0, 0, 0, 0, 0, 0, 0])

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

num_gen_token_max = 100
num_gen_tokens = 0

# Store these for dynamic budget adjustment
current_model_path = model_path
current_attn_type = args.attn_type

# CSV logging setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Simple filenames without timestamp/counter
step_log_file = os.path.join(log_dir, "step_log_kl_confidence.csv")
accumulated_log_file = os.path.join(log_dir, "accumulated_log_kl_confidence.csv")

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
    while not terminal:
        settled = False
        verified = False

        # Draft speculation
        draft_outputs, draft_logits, top1_top2_diff = engine.speculate(tokens_buffer[:, :1], args.gamma1)
        tokens_buffer[:,1:1+args.gamma1] = torch.LongTensor(draft_outputs)
        step_speculate_calls += args.gamma1
        
        # Store draft data for KL analysis
        kl_confidence_analyzer.store_draft_data(draft_logits, top1_top2_diff)
        
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

        verify_outputs, verify_logits, verify_top1_top2_diff = engine.verify(tokens_buffer[:, :1], args.gamma1+1)
        target_tokens = torch.LongTensor(verify_outputs).to(DEVICE) #TODO: verify stage should be batch-fashion, but this verify() is auto-regressive.

        step_verify_calls += 1
        called_verify += 1
        
        # Store verify data for KL analysis
        kl_confidence_analyzer.store_verify_data(verify_logits, verify_top1_top2_diff)

        draft_tokens = tokens_buffer[:, 1:args.gamma1+1]
        flag_accept_matrix = (target_tokens[:, :args.gamma1] == draft_tokens)
        eot_condition = ((draft_tokens == eot_1) | (draft_tokens == eot_2))

        accept_flags_int = (flag_accept_matrix & (~eot_condition)).int()
        accept_flags_cumprod = torch.cumprod(accept_flags_int, dim=1)
        accept_flags_matrix = accept_flags_cumprod.bool()
        accept_nums = accept_flags_matrix.sum(dim=1, keepdim=True)
        num_unsettled_tokens += accept_nums.flatten().item() + 1

        # Analyze KL divergences for verify stage
        num_accepted = accept_nums.flatten().item()
        print(f"Verify analysis: {num_accepted} tokens accepted, draft_logits_len={len(kl_confidence_analyzer.current_draft_logits) if kl_confidence_analyzer.current_draft_logits else 0}, verify_logits_len={len(kl_confidence_analyzer.current_verify_logits) if kl_confidence_analyzer.current_verify_logits else 0}")
        
        kl_confidence_analyzer.analyze_all_tokens(num_accepted)
        
        # Accumulate data for later settle analysis
        kl_confidence_analyzer.accumulate_data_after_verify(num_accepted)
        print(f"Accumulated data now: {len(kl_confidence_analyzer.accumulated_draft_logits)} tokens")

        positions_buffer = torch.arange(args.gamma1, device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
        mask_buffer = positions_buffer < accept_nums.view(-1,1)
        indices = accept_nums
        bonus_tokens = target_tokens.gather(1, indices)

        # Check for termination conditions
        condition = (eot_condition & accept_flags_matrix).any(dim=1, keepdim=True)
        if condition.any() or (bonus_tokens == eot_1).any() or (bonus_tokens == eot_2).any():
            terminal = True

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
            
            # For settlement, we need to analyze KL divergences for rejected tokens
            # We use the accumulated draft and verify data from the speculation cycles
            settle_accepted = accept_nums.flatten().item()
            
            # Analyze rejected tokens in settle stage using accumulated data
            kl_confidence_analyzer.analyze_rejected_tokens_settle(settle_accepted)
            
            print(f"Settle analysis: {settle_accepted} tokens accepted out of {len(kl_confidence_analyzer.accumulated_draft_logits)} accumulated tokens")
            
            positions_buffer = torch.arange(num_unsettled_tokens, device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
            mask_buffer = positions_buffer < accept_nums.view(-1,1)
            indices = accept_nums
            bonus_tokens = target_tokens.gather(1, indices)

            # Check for termination conditions
            condition = (eot_condition & accept_flags_matrix).any(dim=1, keepdim=True)
            if condition.any() or (bonus_tokens == eot_1).any() or (bonus_tokens == eot_2).any():
                terminal = True

            # get accepted token and re-decode to set draft cache
            accepted_tokens = torch.concat((cached_tokens_buffer.view(1,-1), draft_tokens[mask_buffer].view(1,-1)), dim=1)
            bonus_tokens = target_tokens.gather(1, indices)
            # load settled tokens to input cache.
            engine.update_settled_kv(accepted_tokens)
            # after settle, the new start is the bonus tokens
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

# Save KL confidence analysis results
print(f"\n=== Saving KL Confidence Analysis Results ===")

# Create filename prefix with model configuration
filename_prefix = f"{MODEL}_{args.dataset}_prefix{args.prefix_len}_gamma1{args.gamma1}_budget1{args.budget1}_budget2{args.budget2}"

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
