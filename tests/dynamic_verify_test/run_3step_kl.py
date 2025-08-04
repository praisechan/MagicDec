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
from datasets import load_dataset
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

# Add KL divergence analysis class
class KLAnalyzer:
    def __init__(self, num_bins=10, bin_width=0.1, center=0.0):
        # Keep parameters for compatibility but won't use bins
        self.num_bins = num_bins
        self.bin_width = bin_width
        self.center = center
        
        # Store KL divergences without binning
        # For all tokens
        self.all_tokens_data = []
        
        # For rejected tokens only
        self.rejected_tokens_data = []
        
        # Temporary storage for current speculation cycle
        self.current_draft_logits = None
        self.current_verify_logits = None
        
        # Storage for accumulated unsettled tokens (between settlement cycles)
        self.accumulated_draft_logits = []
        self.accumulated_verify_logits = []
        
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
        
    def store_draft_logits(self, draft_logits):
        """Store logits from speculation"""
        if draft_logits is not None:
            self.current_draft_logits = draft_logits
        else:
            self.current_draft_logits = None
    
    def store_verify_logits(self, verify_logits):
        """Store logits from verification"""
        if verify_logits is not None:
            self.current_verify_logits = verify_logits
        else:
            self.current_verify_logits = None
    
    def accumulate_kl_after_verify(self, num_accepted_tokens):
        """Accumulate logits after each verify call for later settle analysis"""
        if (self.current_draft_logits is None or 
            self.current_verify_logits is None):
            return
        
        # Only accumulate tokens up to accepted position (plus one for the bonus token)
        max_tokens_to_accumulate = num_accepted_tokens + 1        
        for i in range(max_tokens_to_accumulate):
            if i == max_tokens_to_accumulate-1:
                self.accumulated_draft_logits.append(None)
                self.accumulated_verify_logits.append(self.current_verify_logits[i])            
            else:
                self.accumulated_draft_logits.append(self.current_draft_logits[i])
                self.accumulated_verify_logits.append(self.current_verify_logits[i])
    
    def analyze_all_tokens(self, num_accepted_tokens):
        """Analyze KL divergences for all tokens up to accepted position"""
        if (self.current_draft_logits is None or 
            self.current_verify_logits is None):
            return
        
        # Only analyze tokens up to the number of accepted tokens
        min_len = min(len(self.current_draft_logits), 
                     len(self.current_verify_logits),
                     num_accepted_tokens)
        
        for i in range(min_len):
            # Compute KL divergence between draft and verify logits
            kl_div = self.compute_kl_divergence(self.current_draft_logits[i], self.current_verify_logits[i])
            self.all_tokens_data.append(kl_div)
    
    def analyze_rejected_tokens_settle(self, num_accepted_tokens):
        """Analyze KL divergences for rejected tokens in settle stage using accumulated data"""
        if (len(self.accumulated_draft_logits) == 0 or 
            len(self.accumulated_verify_logits) == 0):
            return
        
        # Find the first rejected token (if any)
        max_tokens = min(len(self.accumulated_draft_logits), 
                        len(self.accumulated_verify_logits))
        
        if num_accepted_tokens < max_tokens:
            # There is a rejected token at position num_accepted_tokens
            rejected_idx = num_accepted_tokens
            
            draft_logits = self.accumulated_draft_logits[rejected_idx]
            verify_logits = self.accumulated_verify_logits[rejected_idx]
            
            if draft_logits is None or verify_logits is None:
                print(f"Warning: No logits available for rejected token at index {rejected_idx}")
                return

            # Compute KL divergence between draft and verify logits for rejected token
            kl_div = self.compute_kl_divergence(draft_logits, verify_logits)
            self.rejected_tokens_data.append(kl_div)
        
    def get_accumulated_stats(self):
        """Get statistics about accumulated data for debugging"""
        return {
            'draft_logits_count': len(self.accumulated_draft_logits),
            'verify_logits_count': len(self.accumulated_verify_logits),
            'current_draft_count': len(self.current_draft_logits) if self.current_draft_logits else 0,
            'current_verify_count': len(self.current_verify_logits) if self.current_verify_logits else 0
        }
        
    def reset_accumulated_data(self):
        """Reset accumulated storage after settlement"""
        self.current_draft_logits = None
        self.current_verify_logits = None

        self.accumulated_draft_logits = []
        self.accumulated_verify_logits = []
    
    def save_histograms(self, output_dir="kl_analysis", filename_prefix=""):
        """Save histograms for KL divergences"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all tokens histogram
        if len(self.all_tokens_data) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(self.all_tokens_data, bins=50, alpha=0.7, edgecolor='black')
            plt.title('All Tokens - KL Divergence Distribution')
            plt.xlabel('KL Divergence (Draft || Verify)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            filename = f"{filename_prefix}_all_tokens_kl.png" if filename_prefix else 'all_tokens_kl.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Save rejected tokens histogram
        if len(self.rejected_tokens_data) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(self.rejected_tokens_data, bins=50, alpha=0.7, edgecolor='black', color='orange')
            plt.title('Rejected Tokens - KL Divergence Distribution')
            plt.xlabel('KL Divergence (Draft || Verify)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            filename = f"{filename_prefix}_rejected_tokens_kl.png" if filename_prefix else 'rejected_tokens_kl.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_statistics(self, output_dir="kl_analysis", filename_prefix="", num_histogram_bins=50):
        """Save detailed statistics including histogram data to CSV"""
        os.makedirs(output_dir, exist_ok=True)
        
        # All tokens statistics
        if len(self.all_tokens_data) > 0:
            # Create histogram data
            hist_counts, hist_edges = np.histogram(self.all_tokens_data, bins=num_histogram_bins)
            hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
            
            # Save histogram data
            hist_filename = f"{filename_prefix}_all_tokens_histogram.csv" if filename_prefix else 'all_tokens_histogram.csv'
            with open(os.path.join(output_dir, hist_filename), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Bin_Center', 'Count', 'Bin_Start', 'Bin_End'])
                for j in range(len(hist_counts)):
                    writer.writerow([hist_centers[j], hist_counts[j], hist_edges[j], hist_edges[j+1]])
            
            # Save raw data
            raw_filename = f"{filename_prefix}_all_tokens_raw.csv" if filename_prefix else 'all_tokens_raw.csv'
            with open(os.path.join(output_dir, raw_filename), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['KL_Divergence'])
                for value in self.all_tokens_data:
                    writer.writerow([value])
        
        # All tokens summary statistics
        all_tokens_filename = f"{filename_prefix}_all_tokens_stats.csv" if filename_prefix else 'all_tokens_stats.csv'
        with open(os.path.join(output_dir, all_tokens_filename), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Count', 'Mean', 'Std', 'Min', 'Max', 'Percentile_25', 'Percentile_50', 'Percentile_75'])
            
            if len(self.all_tokens_data) > 0:
                data_array = np.array(self.all_tokens_data)
                writer.writerow([
                    len(self.all_tokens_data), np.mean(data_array), np.std(data_array),
                    np.min(data_array), np.max(data_array),
                    np.percentile(data_array, 25), np.percentile(data_array, 50), np.percentile(data_array, 75)
                ])
            else:
                writer.writerow([0, 0, 0, 0, 0, 0, 0, 0])
        
        # Rejected tokens statistics
        if len(self.rejected_tokens_data) > 0:
            # Create histogram data
            hist_counts, hist_edges = np.histogram(self.rejected_tokens_data, bins=num_histogram_bins)
            hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
            
            # Save histogram data
            hist_filename = f"{filename_prefix}_rejected_tokens_histogram.csv" if filename_prefix else 'rejected_tokens_histogram.csv'
            with open(os.path.join(output_dir, hist_filename), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Bin_Center', 'Count', 'Bin_Start', 'Bin_End'])
                for j in range(len(hist_counts)):
                    writer.writerow([hist_centers[j], hist_counts[j], hist_edges[j], hist_edges[j+1]])
            
            # Save raw data
            raw_filename = f"{filename_prefix}_rejected_tokens_raw.csv" if filename_prefix else 'rejected_tokens_raw.csv'
            with open(os.path.join(output_dir, raw_filename), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['KL_Divergence'])
                for value in self.rejected_tokens_data:
                    writer.writerow([value])
        
        # Rejected tokens summary statistics
        rejected_tokens_filename = f"{filename_prefix}_rejected_tokens_stats.csv" if filename_prefix else 'rejected_tokens_stats.csv'
        with open(os.path.join(output_dir, rejected_tokens_filename), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Count', 'Mean', 'Std', 'Min', 'Max', 'Percentile_25', 'Percentile_50', 'Percentile_75'])
            
            if len(self.rejected_tokens_data) > 0:
                data_array = np.array(self.rejected_tokens_data)
                writer.writerow([
                    len(self.rejected_tokens_data), np.mean(data_array), np.std(data_array),
                    np.min(data_array), np.max(data_array),
                    np.percentile(data_array, 25), np.percentile(data_array, 50), np.percentile(data_array, 75)
                ])
            else:
                writer.writerow([0, 0, 0, 0, 0, 0, 0, 0])

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

# Histogram configuration parameters for KL divergence
parser.add_argument("--hist_num_bins", type=int, default=10, help="number of bins for KL divergence histogram")
parser.add_argument("--hist_bin_width", type=float, default=0.1, help="width of each bin for KL divergence histogram")
parser.add_argument("--hist_center", type=float, default=0.5, help="center value for histogram ranges (for KL divergence, typically positive)")
parser.add_argument("--hist_statistics_bins", type=int, default=50, help="number of bins for histogram data in statistics CSV files")

args = parser.parse_args()

# Initialize KL analyzer with configurable histogram parameters
kl_analyzer = KLAnalyzer(
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
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Simple filenames without timestamp/counter
step_log_file = os.path.join(log_dir, "step_log_kl.csv")
accumulated_log_file = os.path.join(log_dir, "accumulated_log_kl.csv")

# Initialize step-wise CSV file with headers
step_headers = [
    "step", "dataset", "prefix_len", "gamma1", "gamma2", "budget1", "budget2", 
    "budget2_low", "confidence_threshold", "enable_dynamic_budget", "speculate_calls", "verify_calls", 
    "settle_calls", "budget_switches_step", "tokens_generated", "min_kl_divergence", 
    "avg_kl_divergence"
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
    step_start_all_tokens_count = len(kl_analyzer.all_tokens_data)  # Track starting count for step statistics
    
    # input_ids = batch[0].to(DEVICE)
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
        
        # Store draft logits for KL analysis
        kl_analyzer.store_draft_logits(draft_logits)
        
        # Dynamic budget adjustment based on confidence (keeping original logic)
        current_budget = args.budget2  # default budget
        budget_switched = False  # Track if budget was switched for this speculation
        
        if args.enable_dynamic_budget and top1_top2_diff is not None and len(top1_top2_diff) > 0:
            min_confidence = torch.min(torch.tensor(top1_top2_diff))
            avg_confidence = torch.mean(torch.tensor(top1_top2_diff))
            
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
        
        # Store verify logits for KL analysis
        kl_analyzer.store_verify_logits(verify_logits)

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
        print(f"Verify analysis: {num_accepted} tokens accepted, draft_logits_len={len(kl_analyzer.current_draft_logits) if kl_analyzer.current_draft_logits else 0}, verify_logits_len={len(kl_analyzer.current_verify_logits) if kl_analyzer.current_verify_logits else 0}")
        
        kl_analyzer.analyze_all_tokens(num_accepted)
        
        # Accumulate KL divergences for later settle analysis
        kl_analyzer.accumulate_kl_after_verify(num_accepted)
        print(f"Accumulated logits now: {len(kl_analyzer.accumulated_draft_logits)} tokens")

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
            # We use the accumulated draft and verify KL divergences from the speculation cycles
            settle_accepted = accept_nums.flatten().item()
            
            # Analyze rejected tokens in settle stage using accumulated data
            kl_analyzer.analyze_rejected_tokens_settle(settle_accepted)
            
            print(f"Settle analysis: {settle_accepted} tokens accepted out of {len(kl_analyzer.accumulated_draft_logits)} accumulated tokens")
            
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
            
            # Reset KL analyzer data after settlement
            kl_analyzer.reset_accumulated_data()

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
    
    # Calculate KL divergence statistics for this step using all_tokens_data
    step_end_all_tokens_count = len(kl_analyzer.all_tokens_data)
    step_kl_data = kl_analyzer.all_tokens_data[step_start_all_tokens_count:step_end_all_tokens_count]
    min_kl_step = float(min(step_kl_data)) if step_kl_data else 0.0
    avg_kl_step = float(sum(step_kl_data) / len(step_kl_data)) if step_kl_data else 0.0
    
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
        step_budget_switches, num_gen_tokens, min_kl_step, avg_kl_step
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
    print(f"Min KL divergence: {min_kl_step:.6f}")
    print(f"Avg KL divergence: {avg_kl_step:.6f}")
    
    print(f"\n=== Accumulated Statistics (up to step {step}) ===")
    print(f"Total speculate calls: {total_speculate_calls}")
    print(f"Total verify calls: {total_verify_calls}")
    print(f"Total settle calls: {total_settle_calls}")
    print(f"Total budget switches: {total_budget_switches}")
    print(f"Total tokens generated: {total_tokens_generated}")
    
    if args.printoutput:
        print(f"Generated output: {decoded_output}")

# After all steps are completed, store the final accumulated data
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

# Save KL divergence analysis results
print(f"\n=== Saving KL Divergence Analysis Results ===")

# Create filename prefix with model configuration
filename_prefix = f"{MODEL}_{args.dataset}_prefix{args.prefix_len}_gamma1{args.gamma1}_budget1{args.budget1}_budget2{args.budget2}"

kl_analyzer.save_histograms("kl_analysis", filename_prefix)
kl_analyzer.save_statistics("kl_analysis", filename_prefix, args.hist_statistics_bins)
print(f"KL divergence analysis saved to 'kl_analysis' directory with prefix: {filename_prefix}")

# Print summary of collected data
print(f"\n=== KL Divergence Analysis Summary ===")
print(f"Histogram configuration: {args.hist_num_bins} bins, width={args.hist_bin_width}, center={args.hist_center} (Note: No binning used)")
total_all_tokens = len(kl_analyzer.all_tokens_data)
total_rejected_tokens = len(kl_analyzer.rejected_tokens_data)
print(f"Total tokens analyzed: {total_all_tokens}")
print(f"Total rejected tokens analyzed: {total_rejected_tokens}")

if total_all_tokens > 0:
    all_tokens_array = np.array(kl_analyzer.all_tokens_data)
    print(f"All tokens KL divergence - Mean: {np.mean(all_tokens_array):.6f}, Std: {np.std(all_tokens_array):.6f}, Min: {np.min(all_tokens_array):.6f}, Max: {np.max(all_tokens_array):.6f}")

if total_rejected_tokens > 0:
    rejected_tokens_array = np.array(kl_analyzer.rejected_tokens_data)
    print(f"Rejected tokens KL divergence - Mean: {np.mean(rejected_tokens_array):.6f}, Std: {np.std(rejected_tokens_array):.6f}, Min: {np.min(rejected_tokens_array):.6f}, Max: {np.max(rejected_tokens_array):.6f}")
