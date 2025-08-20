import time
import torch
import sys
sys.path.append("..")
from pathlib import Path
import torch.distributed as dist
import csv
import os
import json
import numpy as np
import itertools
from datetime import datetime
from MagicDec.Engine.utils import setup_seed, cuda_graph_for_sampling_argmax_batch, sampling_argmax_batch
from MagicDec.Data.data_converter import convert_pg19_dataset, convert_c4_dataset, convert_wiki_dataset, convert_cnn_dataset, convert_longbench_v2_dataset, convert_longbench_v2_sum_dataset, convert_longbench_v1_dataset
from transformers import AutoTokenizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import argparse
from MagicDec.Engine.RetrievalAttention.backend_for_3stage_dynamic_budget import LMBackend_Retro
from datasets import load_dataset
from confidence_analyzer import KLConfidenceAnalyzer, KLConfidenceAnalyzer_temp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MagicDec.Engine.RetrievalAttention.benchmark.config import generate_config, parse_attn_args


class KLThresholdOptimizer:
    def __init__(self, base_args):
        self.base_args = base_args
        self.optimization_results = []
        self.best_config = None
        self.best_score = float('-inf')
        
        # Define search space for KL thresholds
        self.threshold_candidates = [0.02, 0.03, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2]
        
        # Setup logging
        self.setup_optimization_logging()
    
    def setup_optimization_logging(self):
        """Setup logging for optimization results"""
        MODEL = self.base_args.model_name.split("/")[-1]
        self.opt_log_dir = f"/home/juchanlee/MagicDec/profile/kl_threshold_optimization_cluster32/{MODEL}_{self.base_args.dataset}_{self.base_args.prefix_len}"
        os.makedirs(self.opt_log_dir, exist_ok=True)
        
        self.opt_log_file = os.path.join(self.opt_log_dir, "kl_threshold_optimization.csv")
        self.best_config_file = os.path.join(self.opt_log_dir, "best_kl_thresholds.json")
        
        # Initialize optimization log file
        opt_headers = [
            "experiment_id", "bin_0", "bin_1", "bin_2", "bin_3", "bin_4", 
            "bin_5", "bin_6", "bin_7", "bin_8", "bin_9",
            "total_speculate_calls", "total_verify_calls", "total_settle_calls",
            "total_budget_switches_high", "total_budget_switches_low", 
            "total_tokens_generated", "acceptance_rate", "efficiency_score",
            "avg_confidence", "std_confidence", "runtime_seconds"
        ]
        
        if not os.path.exists(self.opt_log_file):
            with open(self.opt_log_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(opt_headers)
    
    def generate_threshold_configurations(self, strategy="smart_grid"):
        """Generate KL threshold configurations to test"""
        if strategy == "coarse_grid":
            # Coarse grid search - test a subset of combinations
            base_values = [0.03, 0.06, 0.1, 0.12]
            configs = []
            
            # Generate configurations with increasing thresholds
            for low_thresh in [0.03, 0.05]:
                for mid_thresh in [0.06, 0.08, 0.1]:
                    for high_thresh in [0.1, 0.12, 0.15]:
                        if low_thresh <= mid_thresh <= high_thresh:
                            config = {
                                0: low_thresh, 1: low_thresh, 2: mid_thresh,
                                3: mid_thresh, 4: mid_thresh, 5: mid_thresh,
                                6: high_thresh, 7: high_thresh, 8: high_thresh, 9: high_thresh
                            }
                            configs.append(config)
            
            return configs[:20]  # Limit to 20 configurations for initial search
        
        elif strategy == "smart_grid":
            # Smart grid based on intuition that higher bins need higher thresholds
            configs = []
            
            # Strategy 1: Linear increase
            for start in [0.02, 0.03, 0.05]:
                for step in [0.01, 0.02, 0.03]:
                    config = {}
                    for i in range(10):
                        config[i] = min(start + i * step, 0.2)
                    configs.append(config)
            
            # Strategy 2: Exponential-like increase
            for base in [0.03, 0.05]:
                for factor in [1.2, 1.3, 1.5]:
                    config = {}
                    current = base
                    for i in range(10):
                        config[i] = min(current, 0.2)
                        current *= factor
                    configs.append(config)
            
            # Strategy 3: Step functions
            step_configs = [
                # Conservative (lower thresholds)
                {0: 0.02, 1: 0.03, 2: 0.04, 3: 0.06, 4: 0.06, 5: 0.08, 6: 0.1, 7: 0.1, 8: 0.12, 9: 0.12},
                # Moderate 
                {0: 0.03, 1: 0.05, 2: 0.06, 3: 0.08, 4: 0.08, 5: 0.1, 6: 0.12, 7: 0.12, 8: 0.15, 9: 0.15},
                # Aggressive (higher thresholds)
                {0: 0.05, 1: 0.06, 2: 0.08, 3: 0.1, 4: 0.1, 5: 0.12, 6: 0.15, 7: 0.15, 8: 0.18, 9: 0.2},
            ]
            configs.extend(step_configs)
            
            return configs[:15]  # Limit to manageable number
        
        elif strategy == "random_search":
            # Random search for exploration
            configs = []
            np.random.seed(42)
            
            for _ in range(10):
                config = {}
                # Generate random thresholds with constraint that they generally increase
                base_values = np.sort(np.random.uniform(0.02, 0.15, 4))
                bins_per_value = [3, 3, 2, 2]  # How many bins to assign each value
                
                bin_idx = 0
                for val_idx, (value, count) in enumerate(zip(base_values, bins_per_value)):
                    for _ in range(count):
                        if bin_idx < 10:
                            # Add some noise
                            noise = np.random.uniform(-0.01, 0.01)
                            config[bin_idx] = max(0.02, min(0.2, value + noise))
                            bin_idx += 1
                
                configs.append(config)
            
            return configs
    
    def calculate_performance_score(self, results):
        """Calculate a performance score for a configuration"""
        # Extract key metrics
        speculate_calls = results['total_speculate_calls']
        verify_calls = results['total_verify_calls'] 
        settle_calls = results['total_settle_calls']
        budget_switches_high = results['total_budget_switches_high']
        budget_switches_low = results['total_budget_switches_low']
        tokens_generated = results['total_tokens_generated']
        
        if tokens_generated == 0:
            return 0.0, float('inf')  # Avoid division by zero, worst efficiency score
        
        # Calculate total budget spent
        # Speculate calls use budget1 (default: 2%)
        speculate_budget = speculate_calls * self.base_args.budget1
        
        # Verify calls breakdown:
        # - Normal verify calls use budget2 (default: 25%)
        # - High budget switches use budget2_high (default: 40%)  
        # - Low budget switches use budget2_low (default: 10%)
        normal_verify_calls = verify_calls - budget_switches_high - budget_switches_low
        verify_budget = (normal_verify_calls * self.base_args.budget2 + 
                        budget_switches_high * self.base_args.budget2_high + 
                        budget_switches_low * self.base_args.budget2_low)
        
        # Settle calls use 100% budget
        settle_budget = settle_calls * 1.0
        
        total_budget_spent = speculate_budget + verify_budget + settle_budget
        
        # Efficiency score = total budget spent per token (lower is better)
        efficiency_score = total_budget_spent / tokens_generated
        
        # Acceptance rate (tokens generated per total calls) - for reference
        total_calls = speculate_calls + verify_calls + settle_calls
        acceptance_rate = tokens_generated / max(total_calls, 1)
        
        return acceptance_rate, efficiency_score
    
    def run_single_experiment(self, kl_thresholds_config, experiment_id):
        """Run a single experiment with given KL thresholds"""
        print(f"\n=== Experiment {experiment_id} ===")
        print(f"Testing KL thresholds: {kl_thresholds_config}")
        
        start_time = time.time()
        
        # Create analyzer with this configuration
        kl_confidence_analyzer = KLConfidenceAnalyzer_temp(
            num_bins=self.base_args.hist_num_bins, 
            bin_width=self.base_args.hist_bin_width, 
            center=self.base_args.hist_center,
            kl_threshold=self.base_args.kl_threshold,
            bin_kl_thresholds=kl_thresholds_config
        )
        
        # Run the main experiment (simplified version)
        results = self.run_experiment_with_config(kl_confidence_analyzer, experiment_id)
        
        runtime = time.time() - start_time
        
        # Calculate performance metrics
        acceptance_rate, efficiency_score = self.calculate_performance_score(results)
        
        # Log results
        experiment_data = [
            experiment_id,
            kl_thresholds_config.get(0, 0), kl_thresholds_config.get(1, 0), 
            kl_thresholds_config.get(2, 0), kl_thresholds_config.get(3, 0),
            kl_thresholds_config.get(4, 0), kl_thresholds_config.get(5, 0),
            kl_thresholds_config.get(6, 0), kl_thresholds_config.get(7, 0),
            kl_thresholds_config.get(8, 0), kl_thresholds_config.get(9, 0),
            results['total_speculate_calls'], results['total_verify_calls'], 
            results['total_settle_calls'], results['total_budget_switches_high'],
            results['total_budget_switches_low'], results['total_tokens_generated'],
            acceptance_rate, efficiency_score, results['avg_confidence'], 
            results['std_confidence'], runtime
        ]
        
        with open(self.opt_log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(experiment_data)
        
        # Update best configuration
        if efficiency_score < self.best_score or self.best_score == float('-inf'):
            self.best_score = efficiency_score
            self.best_config = kl_thresholds_config.copy()
            
            # Save best configuration
            with open(self.best_config_file, 'w') as f:
                json.dump({
                    'best_kl_thresholds': self.best_config,
                    'best_score': float(self.best_score),
                    'experiment_id': experiment_id,
                    'acceptance_rate': float(acceptance_rate),
                    'efficiency_score': float(efficiency_score),
                    'budget_breakdown': {
                        'speculate_budget': float(results['total_speculate_calls'] * self.base_args.budget1),
                        'verify_budget_normal': float((results['total_verify_calls'] - results['total_budget_switches_high'] - results['total_budget_switches_low']) * self.base_args.budget2),
                        'verify_budget_high': float(results['total_budget_switches_high'] * self.base_args.budget2_high),
                        'verify_budget_low': float(results['total_budget_switches_low'] * self.base_args.budget2_low),
                        'settle_budget': float(results['total_settle_calls'] * 1.0),
                        'total_budget': float(efficiency_score * results['total_tokens_generated'])
                    }
                }, f, indent=2)
        
        print(f"Acceptance rate: {acceptance_rate:.4f}, Efficiency score: {efficiency_score:.4f}")
        print(f"Runtime: {runtime:.2f} seconds")
        
        return results, efficiency_score
    
    def run_experiment_with_config(self, kl_confidence_analyzer, experiment_id):
        """Run the main experiment logic with given analyzer - wrapper for run_3step_kl_confidence.py"""
        
        # Initialize engine and data (copied from run_3step_kl_confidence.py)
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        DTYPE = torch.bfloat16
        BATCH_SIZE = self.base_args.B
        benchmark = self.base_args.benchmark if hasattr(self.base_args, 'benchmark') else False
        
        target_dec_len = self.base_args.gamma1 + 1
        draft_dec_len = 1
        
        # Load target model
        engine = LMBackend_Retro(dtype=DTYPE, device=DEVICE, dec_len=target_dec_len, draft_dec_len=draft_dec_len)
        
        model2path = json.load(open("Engine/RetrievalAttention/benchmark/LongBench/config/model2path.json", "r"))
        model2maxlen = json.load(open("Engine/RetrievalAttention/benchmark/LongBench/config/model2maxlen.json", "r"))
        dataset2prompt = json.load(open("Engine/RetrievalAttention/benchmark/LongBench/config/dataset2prompt.json", "r"))
        dataset2maxlen = json.load(open("Engine/RetrievalAttention/benchmark/LongBench/config/dataset2maxlen.json", "r"))
        
        MODEL = self.base_args.model_name.split("/")[-1]
        TASK = self.base_args.task
        
        num_examples = self.base_args.num_examples if hasattr(self.base_args, 'num_examples') else -1
        attn_type = self.base_args.attn_type
        device = "auto"
        dtype = torch.bfloat16
        model_path = model2path[MODEL]
        max_length = model2maxlen[MODEL]
        prompt_format = dataset2prompt[TASK]
        
        engine.load_model(model_path, max_length, dtype, device, BATCH_SIZE)
        vocab_size = engine.model.config.vocab_size
        
        # Load dataset
        tokenizer = engine.model.tokenizer
        eot_1 = tokenizer.eos_token_id
        if tokenizer.unk_token_id is not None:
            eot_2 = tokenizer.unk_token_id
        else:
            eot_2 = tokenizer.encode("<|eot_id|>")[-1]
        
        if self.base_args.dataset == "pg19":
            dataset = convert_pg19_dataset(tokenizer=engine.model.tokenizer, seq_len=self.base_args.prefix_len)
        elif self.base_args.dataset == "longbenchv1":
            dataset = load_dataset('THUDM/LongBench', TASK, split='test')
        else:
            raise ValueError(f"Unknown dataset {self.base_args.dataset}")
        
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
        if self.base_args.dataset == "pg19":
            num_eval_steps = min(3, len(dataloader))  # Use only 3 examples for optimization
        else:
            num_eval_steps = min(3, len(dataloader))  # Use only 3 examples for optimization
        
        num_gen_token_max = 100
        num_gen_tokens = 0
        
        # Store these for dynamic budget adjustment
        current_model_path = model_path
        current_attn_type = self.base_args.attn_type
        
        # Accumulated statistics
        total_speculate_calls = 0
        total_verify_calls = 0
        total_settle_calls = 0
        total_budget_switches_high = 0
        total_budget_switches_low = 0
        total_tokens_generated = 0
        all_confidences = []
        
        # Main loop - copied from run_3step_kl_confidence.py
        actual_step = 0
        for step, batch in tqdm(enumerate(dataset), total=num_eval_steps, desc=f"Experiment {experiment_id}"):
            if actual_step >= num_eval_steps:
                break
            input_ids = engine.preprocess_input(batch, prompt_format, self.base_args.attn_type, model_path, self.base_args.budget1, self.base_args.budget2, self.base_args.estimate_ratio, self.base_args.dataset, self.base_args.prefix_len)
            if input_ids is None:
                print(f"Skipping step {step} due to empty input_ids.")
                continue
            actual_step += 1 # increment actual step count only if input_ids is valid

            # Initialize step-wise counters
            step_speculate_calls = 0
            step_verify_calls = 0
            step_settle_calls = 0
            step_budget_switches_high = 0
            step_budget_switches_low = 0
            step_confidences = []  # Store confidence values for this step
            
            terminal = False
            tokens_buffer= torch.zeros((BATCH_SIZE, self.base_args.gamma1+1), device=DEVICE).long()

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
            use_extended_verification = False  # Flag to use extended verification with higher budget
            previous_num_accepted = 0  # Number of tokens accepted in the previous verify stage
            previous_accepted_tokens = tokens_buffer[:, :1].clone()  # Store the last accepted token for extended verification
            
            while not terminal:
                settled = False
                verified = False

                # Draft speculation
                draft_outputs, draft_logits, top1_top2_diff = engine.speculate(tokens_buffer[:, :1], self.base_args.gamma1, profile_clustering=False, profile_hot_cluster_selection_ratio=False, generate_name=f"optimize_exp_{experiment_id}/speculate_{step}_{step_speculate_calls}")
                tokens_buffer[:,1:1+self.base_args.gamma1] = torch.LongTensor(draft_outputs)
                step_speculate_calls += self.base_args.gamma1
                
                # Store draft data for KL analysis
                kl_confidence_analyzer.store_draft_data(draft_logits, top1_top2_diff, use_extended_verification)
                
                # Dynamic budget adjustment based on confidence
                # If all tokens have high confidence (top1_top2_diff > threshold), use lower budget
                current_budget = self.base_args.budget2  # default budget
                budget_switched = False  # Track if budget was switched for this speculation
                if self.base_args.enable_dynamic_budget and top1_top2_diff is not None and len(top1_top2_diff) > 0:
                    min_confidence = torch.min(torch.tensor(top1_top2_diff))
                    avg_confidence = torch.mean(torch.tensor(top1_top2_diff))
                    # Convert tensor values to floats for storage
                    step_confidences.extend([float(x) for x in top1_top2_diff])  # Store all confidence values as floats
                                
                    if use_extended_verification:
                        # Use extended verification with accumulated tokens and higher budget
                        verification_budget = self.base_args.budget2_high # Use higher budget
                        step_budget_switches_high += 1

                        # Update engine with higher budget for extended verification
                        engine.update_verification_budget(
                            budget_ratio=verification_budget, 
                            estimate_ratio=self.base_args.estimate_ratio,
                            model_path=current_model_path,
                            seq_len=input_ids.shape[1],
                            attn_type=current_attn_type
                        )
                    else:
                        if min_confidence > self.base_args.confidence_threshold:                
                            # High confidence: use lower budget for verification
                            current_budget = self.base_args.budget2_low
                            budget_switched = True
                            step_budget_switches_low += 1
                            engine.update_verification_budget(
                                budget_ratio=current_budget, 
                                estimate_ratio=self.base_args.estimate_ratio,
                                model_path=current_model_path,
                                seq_len=input_ids.shape[1],
                                attn_type=current_attn_type
                            )
                        else:
                            # Low confidence: use original budget for verification
                            engine.update_verification_budget(
                                budget_ratio=current_budget, 
                                estimate_ratio=self.base_args.estimate_ratio,
                                model_path=current_model_path,
                                seq_len=input_ids.shape[1],
                                attn_type=current_attn_type
                            )
                else:
                    # Dynamic budget disabled or no confidence data available - use original budget
                    # Still collect confidence data for logging if available
                    if top1_top2_diff is not None and len(top1_top2_diff) > 0:
                        step_confidences.extend([float(x) for x in top1_top2_diff])
                    
                    engine.update_verification_budget(
                        budget_ratio=current_budget, 
                        estimate_ratio=self.base_args.estimate_ratio,
                        model_path=current_model_path,
                        seq_len=input_ids.shape[1],
                        attn_type=current_attn_type
                    )

                # Always call verify after speculate
                if called_verify == 0:
                    cached_tokens_buffer = tokens_buffer[:, 0].clone() # bonus token from settle

                verify_outputs, verify_logits, verify_top1_top2_diff = engine.verify_dynamic(tokens_buffer[:, :1], self.base_args.gamma1+1, use_first_kv=True, profile_clustering=False, profile_hot_cluster_selection_ratio=False, generate_name=f"optimize_exp_{experiment_id}/verify_{step}_{step_verify_calls}", use_extended_verification=use_extended_verification, previous_num_accepted=previous_num_accepted)
                target_tokens = torch.LongTensor(verify_outputs).to(DEVICE)

                step_verify_calls += 1
                called_verify += 1
                
                # Store verify data for KL analysis
                kl_confidence_analyzer.store_verify_data(verify_logits, verify_top1_top2_diff)
                
                # Handle verification results based on verification length used
                if use_extended_verification:
                    # Extended verification
                    verify_tokens_count = previous_num_accepted + self.base_args.gamma1 + 1
                    draft_tokens = torch.concat((previous_accepted_tokens, tokens_buffer[:, 0:self.base_args.gamma1+1]), dim=1)
                    
                    # Ensure we don't slice beyond target_tokens bounds
                    actual_target_len = target_tokens.shape[1]
                    effective_verify_count = min(verify_tokens_count, actual_target_len, draft_tokens.shape[1])
                    
                    flag_accept_matrix = (target_tokens[:, :effective_verify_count] == draft_tokens[:, :effective_verify_count])
                    verify_tokens_count = effective_verify_count  # Update for later use
                else:
                    # Normal verification
                    verify_tokens_count = self.base_args.gamma1
                    draft_tokens = tokens_buffer[:, 1:self.base_args.gamma1+1]
                    flag_accept_matrix = (target_tokens[:, :self.base_args.gamma1] == draft_tokens)
                
                eot_condition = ((draft_tokens == eot_1) | (draft_tokens == eot_2))

                accept_flags_int = (flag_accept_matrix & (~eot_condition)).int()
                accept_flags_cumprod = torch.cumprod(accept_flags_int, dim=1)
                accept_flags_matrix = accept_flags_cumprod.bool()
                accept_nums = accept_flags_matrix.sum(dim=1, keepdim=True)
                # Analyze KL divergences for verify stage & Accumulate data for later settle analysis
                num_accepted = accept_nums.flatten().item()
                
                # Adjust position calculation based on verification length used
                positions_buffer = torch.arange(verify_tokens_count, device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
                mask_buffer = positions_buffer < accept_nums.view(-1,1)
                indices = accept_nums
                
                # CRITICAL FIX: Ensure indices don't exceed target_tokens bounds
                max_valid_index = target_tokens.shape[1] - 1
                if indices.max() > max_valid_index:
                    indices = torch.clamp(indices, 0, max_valid_index)
                
                bonus_tokens = target_tokens.gather(1, indices)

                # Check for termination conditions
                condition = (eot_condition & accept_flags_matrix).any(dim=1, keepdim=True)
                if condition.any() or (bonus_tokens == eot_1).any() or (bonus_tokens == eot_2).any():
                    terminal = True

                if use_extended_verification:
                    accepted_tokens = draft_tokens[mask_buffer].view(1,-1)
                else:
                    accepted_tokens = torch.concat((tokens_buffer[:, :1], draft_tokens[mask_buffer].view(1,-1)), dim=1)
                engine.update_verified_kv_dynamic(accepted_tokens, use_extended_verification=use_extended_verification, previous_num_accepted=previous_num_accepted)
                tokens_buffer[:, :1] = bonus_tokens
                
                # Record the last accepted token for extended verification
                previous_num_accepted = num_accepted  # Store for next verify stage
                previous_accepted_tokens = accepted_tokens[:,1:].clone() # Exclude bonus token from previous verify

                # Update extended verification state for next cycle
                kl_threshold_exceeded = kl_confidence_analyzer.analyze_all_tokens(num_accepted)
                if kl_threshold_exceeded and not terminal and self.base_args.enable_extended_verification:
                    # KL threshold exceeded, prepare for extended verification next time
                    if not use_extended_verification:
                        # First time exceeding threshold, start accumulating
                        use_extended_verification = True
                        num_unsettled_tokens += 0 # add only the bonus token, rest will be accumulated in next verify
                    else:
                        # Already in extended mode, continue accumulating but don't exceed buffer
                        use_extended_verification = False
                        num_unsettled_tokens += accept_nums.flatten().item() + 1
                else:
                    # KL threshold not exceeded, terminal condition, or extended verification disabled - reset extended verification
                    use_extended_verification = False
                    num_unsettled_tokens += accept_nums.flatten().item() + 1

                # Now, after verify, check if we need to settle
                if num_unsettled_tokens >= self.base_args.gamma2 or called_verify > 2 * (self.base_args.gamma2 / self.base_args.gamma1) or terminal:
                    # Settle
                    settled = True
                    
                    # for sanity
                    use_extended_verification = False

                    if not terminal:
                        # bonus tokens is the last token from verify
                        engine.update_verified_kv(tokens_buffer[:,:1])
                    else:
                        pass  # Terminal

                    settle_outputs, settle_logits, settle_top1_top2_diff = engine.settle(cached_tokens_buffer.view(-1,1), num_unsettled_tokens+1)
                    target_tokens = torch.LongTensor(settle_outputs).to(DEVICE)
                    
                    step_settle_calls += 1

                    input_from_start = engine.input_tokens[:, :engine.verified_cachelength]
                    draft_tokens = input_from_start[:, -(num_unsettled_tokens):]
                    flag_accept_matrix = (target_tokens[:, :num_unsettled_tokens] == draft_tokens)
                    eot_condition = ((draft_tokens == eot_1) | (draft_tokens == eot_2))
                    accept_flags_int = (flag_accept_matrix & (~eot_condition)).int()
                    accept_flags_cumprod = torch.cumprod(accept_flags_int, dim=1)
                    accept_flags_matrix = accept_flags_cumprod.bool()
                    accept_nums = accept_flags_matrix.sum(dim=1, keepdim=True)
                    
                    settle_accepted = accept_nums.flatten().item()
                    
                    positions_buffer = torch.arange(num_unsettled_tokens, device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
                    mask_buffer = positions_buffer < accept_nums.view(-1,1)
                    indices = accept_nums
                    
                    # CRITICAL FIX: Ensure indices don't exceed target_tokens bounds in settle section
                    max_valid_index = target_tokens.shape[1] - 1
                    if indices.max() > max_valid_index:
                        indices = torch.clamp(indices, 0, max_valid_index)
                    
                    bonus_tokens = target_tokens.gather(1, indices)
                    num_nodes += (accept_nums.flatten() + 1)

                    # Check for termination conditions
                    condition = (eot_condition & accept_flags_matrix).any(dim=1, keepdim=True)
                    if condition.any() or (bonus_tokens == eot_1).any() or (bonus_tokens == eot_2).any():
                        terminal = True

                    if self.base_args.dataset == "longbenchv1" or self.base_args.dataset == "longbenchv1-32k":
                        if num_nodes.max() - input_len >= num_gen_token_max:
                            terminal = True
                    else:
                        if num_nodes.max() - self.base_args.prefix_len >= num_gen_token_max:
                            terminal = True

                    accepted_tokens = torch.concat((cached_tokens_buffer.view(1,-1), draft_tokens[mask_buffer].view(1,-1)), dim=1)
                    engine.update_settled_kv(accepted_tokens)
                    tokens_buffer[:, :1] = bonus_tokens
                    
                    # record unsettled_tokens
                    num_unsettled_tokens = 0
                    called_verify = 0

            num_gen_tokens = engine.settled_cachelength - input_len
            
            # Update step counters
            total_speculate_calls += step_speculate_calls
            total_verify_calls += step_verify_calls
            total_settle_calls += step_settle_calls
            total_budget_switches_high += step_budget_switches_high
            total_budget_switches_low += step_budget_switches_low
            total_tokens_generated += num_gen_tokens
            all_confidences.extend(step_confidences)
        
        # Calculate final metrics
        avg_confidence = float(np.mean(all_confidences)) if all_confidences else 0.0
        std_confidence = float(np.std(all_confidences)) if all_confidences else 0.0
        
        return {
            'total_speculate_calls': total_speculate_calls,
            'total_verify_calls': total_verify_calls,
            'total_settle_calls': total_settle_calls,
            'total_budget_switches_high': total_budget_switches_high,
            'total_budget_switches_low': total_budget_switches_low,
            'total_tokens_generated': total_tokens_generated,
            'avg_confidence': avg_confidence,
            'std_confidence': std_confidence
        }
    
    def optimize(self, strategy="smart_grid"):
        """Run the optimization process"""
        print(f"Starting KL threshold optimization with strategy: {strategy}")
        
        # Generate configurations to test
        configs = self.generate_threshold_configurations(strategy)
        
        print(f"Testing {len(configs)} configurations...")
        
        results = []
        for i, config in enumerate(configs):
            result, score = self.run_single_experiment(config, i)
            results.append((config, result, score))
        
        # Sort by efficiency score (lower is better)
        results.sort(key=lambda x: x[2])
        
        print(f"\n=== Optimization Complete ===")
        print(f"Best configuration (Experiment ID: varies):")
        print(f"KL Thresholds: {results[0][0]}")
        print(f"Efficiency Score (budget per token): {results[0][2]:.4f}")
        print(f"Budget breakdown for best config:")
        
        # Calculate and display budget breakdown for best result
        best_result = results[0][1]
        speculate_budget = best_result['total_speculate_calls'] * self.base_args.budget1
        normal_verify = best_result['total_verify_calls'] - best_result['total_budget_switches_high'] - best_result['total_budget_switches_low']
        verify_budget_normal = normal_verify * self.base_args.budget2
        verify_budget_high = best_result['total_budget_switches_high'] * self.base_args.budget2_high
        verify_budget_low = best_result['total_budget_switches_low'] * self.base_args.budget2_low
        settle_budget = best_result['total_settle_calls'] * 1.0
        total_budget = speculate_budget + verify_budget_normal + verify_budget_high + verify_budget_low + settle_budget
        
        print(f"  Speculate: {best_result['total_speculate_calls']} calls × {self.base_args.budget1} = {speculate_budget:.2f}")
        print(f"  Verify (normal): {normal_verify} calls × {self.base_args.budget2} = {verify_budget_normal:.2f}")
        print(f"  Verify (high): {best_result['total_budget_switches_high']} calls × {self.base_args.budget2_high} = {verify_budget_high:.2f}")
        print(f"  Verify (low): {best_result['total_budget_switches_low']} calls × {self.base_args.budget2_low} = {verify_budget_low:.2f}")
        print(f"  Settle: {best_result['total_settle_calls']} calls × 1.0 = {settle_budget:.2f}")
        print(f"  Total budget: {total_budget:.2f} for {best_result['total_tokens_generated']} tokens")
        
        # Save top 5 configurations
        top_configs_file = os.path.join(self.opt_log_dir, "top_configurations.json")
        top_configs = []
        for i, (config, result, score) in enumerate(results[:5]):
            # Convert all tensor values to native Python types
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    serializable_result[key] = value.item() if value.numel() == 1 else value.tolist()
                elif isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_result[key] = value.item()
                else:
                    serializable_result[key] = value
            
            top_configs.append({
                'rank': i + 1,
                'kl_thresholds': config,
                'efficiency_score': float(score),
                'results': serializable_result
            })
        
        with open(top_configs_file, 'w') as f:
            json.dump(top_configs, f, indent=2)
        
        return results[0][0]  # Return best configuration


def main():
    parser = argparse.ArgumentParser(description='Optimize KL thresholds for dynamic budget adjustment.')
    parser.add_argument('--model_name', type=str, default="qwen2.5-14b", help='model name')
    parser.add_argument('--dataset', type=str, default="pg19", help='Dataset name.')
    parser.add_argument('--B', type=int, default=1, help='Batch size.')
    parser.add_argument('--prefix_len', type=int, default=32800, help='Prefix length')
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--task', type=str, default="gov_report", help='for longbenchv1.')
    parser.add_argument('--gamma1', type=int, default=4, help='start')
    parser.add_argument('--gamma2', type=int, default=32, help='start')
    parser.add_argument("--budget1", type=float, default=0.02, help="ratio of budget")
    parser.add_argument("--budget2", type=float, default=0.25, help="ratio of budget")
    parser.add_argument("--budget2_low", type=float, default=0.1, help="lower ratio of budget")
    parser.add_argument("--budget2_high", type=float, default=0.4, help="upper ratio of budget")
    parser.add_argument("--confidence_threshold", type=float, default=0.4, help="threshold for confidence")
    parser.add_argument("--enable_dynamic_budget", action='store_true', help="enable dynamic budget")
    parser.add_argument("--kl_threshold", type=float, default=0.1, help="threshold for KL divergence")
    parser.add_argument("--enable_extended_verification", action='store_true', help="enable extended verification")
    parser.add_argument("--estimate_ratio", type=float, default=0.25, help="ratio of estimated clusters")
    parser.add_argument("--attn_type", type=str, default="RetroInfer", help="Attention method")
    
    # Histogram configuration
    parser.add_argument("--hist_num_bins", type=int, default=10, help="number of bins")
    parser.add_argument("--hist_bin_width", type=float, default=0.1, help="width of each bin")
    parser.add_argument("--hist_center", type=float, default=0.5, help="center value")
    
    # Optimization specific
    parser.add_argument("--optimization_strategy", type=str, default="smart_grid", 
                       choices=["coarse_grid", "smart_grid", "random_search"],
                       help="Strategy for threshold optimization")
    
    args = parser.parse_args()
    
    # Set up random seed
    setup_seed(args.seed)
    
    # Initialize optimizer
    optimizer = KLThresholdOptimizer(args)
    
    # Run optimization
    best_config = optimizer.optimize(strategy=args.optimization_strategy)
    
    print(f"\nOptimal KL thresholds found:")
    for bin_id, threshold in best_config.items():
        print(f"  Bin {bin_id}: {threshold:.3f}")


if __name__ == "__main__":
    main()