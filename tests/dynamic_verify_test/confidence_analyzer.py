import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Add KL divergence analysis class with confidence binning
class KLConfidenceAnalyzer_temp:
    def __init__(self, num_bins=10, bin_width=0.1, center=0.0, kl_threshold=0.1, bin_kl_thresholds=None, topk=100):
        self.num_bins = num_bins
        self.bin_width = bin_width
        self.center = center
        self.kl_threshold = kl_threshold
        self.topk = topk

        # Set up bin-specific KL thresholds
        if bin_kl_thresholds is not None:
            if isinstance(bin_kl_thresholds, dict):
                self.bin_kl_thresholds = bin_kl_thresholds
            elif isinstance(bin_kl_thresholds, list):
                # Convert list to dict with bin indices as keys
                self.bin_kl_thresholds = {i: threshold for i, threshold in enumerate(bin_kl_thresholds)}
            else:
                raise ValueError("bin_kl_thresholds must be a dict or list")
        else:
            # Use default threshold for all bins
            self.bin_kl_thresholds = {i: kl_threshold for i in range(num_bins)}

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
        
        self.step_idx = 0  # Track the current step index for logging

        # Temporary storage for current speculation cycle
        self.current_draft_logits = None
        self.current_verify_logits = None
        self.current_draft_confidences = None
        self.current_verify_confidences = None
        
        # Storage for accumulated unsettled tokens (between settlement cycles)
        self.temp_accumulated_draft_logits = []
        self.temp_accumulated_verify_logits = []
        self.temp_accumulated_draft_confidences = []
        self.temp_accumulated_verify_confidences = []
        self.temp_accumulated_kl_divergences = []
        self.temp_accumulated_tokens = []
        self.temp_accumulated_reject_flags = []  # True if token was rejected, False if accepted

        self.accumulated_draft_logits = []
        self.accumulated_verify_logits = []
        self.accumulated_draft_confidences = []
        self.accumulated_verify_confidences = []
        self.accumulated_kl_divergences = []
        self.accumulated_tokens = []
        self.accumulated_reject_flags = []  # True if token was rejected, False if accepted
        self.accumulated_step_idx = []  # Store step index for each token
        
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
    
    def compute_topk_kl_divergence(self, logits_p, logits_q, k=100):
        """
        Compute KL divergence between two logit distributions using only top-k tokens.
        Select top-k tokens based on logits, then apply softmax to get probabilities.
        
        Args:
            logits_p: Source distribution logits [1, 1, vocab_size]
            logits_q: Target distribution logits [1, 1, vocab_size]
            k: Number of top tokens to keep (default: 100)
        Returns:
            KL divergence value (scalar)
        """
        # Squeeze to get [vocab_size] tensors
        logits_p_squeezed = logits_p.squeeze()  # [vocab_size]
        logits_q_squeezed = logits_q.squeeze()  # [vocab_size]
        
        # Get top-k indices for each distribution independently based on LOGITS
        topk_values_p, topk_indices_p = torch.topk(logits_p_squeezed, k, dim=-1)
        topk_values_q, topk_indices_q = torch.topk(logits_q_squeezed, k, dim=-1)
        
        # Create masks for top-k tokens
        mask_p = torch.zeros_like(logits_p_squeezed, dtype=torch.bool)
        mask_q = torch.zeros_like(logits_q_squeezed, dtype=torch.bool)
        
        mask_p[topk_indices_p] = True
        mask_q[topk_indices_q] = True
        
        # Zero out non-top-k logits by setting them to a very negative value
        # This ensures they have near-zero probability after softmax
        filtered_logits_p = logits_p_squeezed.clone()
        filtered_logits_q = logits_q_squeezed.clone()
        
        filtered_logits_p[~mask_p] = -float('inf')
        filtered_logits_q[~mask_q] = -float('inf')
        
        # Now apply softmax to get properly normalized probabilities
        prob_p = F.softmax(filtered_logits_p, dim=-1)  # [vocab_size]
        prob_q = F.softmax(filtered_logits_q, dim=-1)  # [vocab_size]
        
        # Convert to log probabilities for KL divergence computation
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        log_p = torch.log(prob_p + eps)
        log_q = torch.log(prob_q + eps)
        
        # Use PyTorch's built-in KL divergence
        # kl_div expects log probabilities for both arguments when log_target=True
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
    
    def get_bin_kl_threshold(self, bin_idx):
        """Get the KL threshold for a specific bin"""
        return self.bin_kl_thresholds.get(bin_idx, self.kl_threshold)
    
    def store_draft_data(self, draft_logits, draft_top1_top2_diff, use_extended_verification=False):
        if not use_extended_verification:
            """Store logits and confidences from speculation"""
            self.current_draft_logits = draft_logits
            self.current_draft_confidences = [float(x) for x in draft_top1_top2_diff]
    
    def store_verify_data(self, verify_outputs, verify_logits, verify_top1_top2_diff, num_accepted_tokens, use_extended_verification=False):
        if not use_extended_verification:
            """Store logits and confidences from verification"""
            self.current_verify_logits = verify_logits
            self.current_verify_confidences = [float(x) for x in verify_top1_top2_diff]

            """Analyze KL divergences for all tokens up to accepted position, binned by draft confidence"""
            if (self.current_draft_logits is None or 
                self.current_verify_logits is None or
                self.current_draft_confidences is None):
                raise ValueError("Draft and verify logits/confidences must be set before analysis")
            
            # Only analyze tokens up to the number of accepted tokens
            min_len = min(len(self.current_draft_logits), 
                        len(self.current_verify_logits),
                        len(self.current_draft_confidences),
                        num_accepted_tokens)

            kl_divergences = []
            bin_threshold_exceeded = []  # Track which bins exceeded their thresholds
            kl_threshold_exceeded = False
            if min_len > 0:
                for i in range(min_len):
                    draft_logits = self.current_draft_logits[i]
                    verify_logits = self.current_verify_logits[i]
                    draft_conf = self.current_draft_confidences[i]
                    
                    # Get bin based on draft confidence
                    bin_idx = self.get_bin_index(draft_conf)
                    
                    # Get bin-specific threshold
                    bin_threshold = self.get_bin_kl_threshold(bin_idx)
                    
                    # Compute KL divergence between draft and verify logits
                    kl_div = self.compute_topk_kl_divergence(draft_logits, verify_logits, self.topk)
                    
                    self.all_tokens_kl_data[f"bin_{bin_idx}"].append(kl_div)
                    
                    kl_divergences.append(kl_div)
                    # Use bin-specific thresholds instead of global threshold
                    # The threshold is exceeded if any token exceeds its bin-specific threshold
                    if kl_div > bin_threshold:
                        print(f"KL divergence exceeded bin threshold: {kl_div:.4f} > {bin_threshold:.4f} (draft_conf={draft_conf:.4f}, bin={bin_idx})")
                        bin_threshold_exceeded.append(True)
                    else:
                        bin_threshold_exceeded.append(False)

                    # store statistics for profile
                    self.temp_accumulated_draft_confidences.append(self.current_draft_confidences[i])
                    self.temp_accumulated_verify_confidences.append(self.current_verify_confidences[i])
                    self.temp_accumulated_draft_logits.append(self.current_draft_logits[i])
                    self.temp_accumulated_verify_logits.append(self.current_verify_logits[i])
                    self.temp_accumulated_kl_divergences.append(kl_div)
                    self.temp_accumulated_tokens.append(verify_outputs[i])
                    self.temp_accumulated_reject_flags.append(0)  # Initialize to 0 (accepted by default)

                max_kl_divergence = max(kl_divergences)
                kl_threshold_exceeded = any(bin_threshold_exceeded)  # True if any bin threshold was exceeded
                print(f"KL divergence check: max={max_kl_divergence:.4f}, any_bin_threshold_exceeded={kl_threshold_exceeded}")
            
            # store bonus tokens' statistics for profile
            self.temp_accumulated_draft_confidences.append(None)
            self.temp_accumulated_verify_confidences.append(self.current_verify_confidences[min_len])
            self.temp_accumulated_draft_logits.append(None)
            self.temp_accumulated_verify_logits.append(self.current_verify_logits[min_len])
            self.temp_accumulated_kl_divergences.append(None)
            self.temp_accumulated_tokens.append(verify_outputs[min_len])
            self.temp_accumulated_reject_flags.append(0)  # Initialize to 0 (accepted by default)

            
            return kl_threshold_exceeded
        else:
            return False  # No analysis for extended verification
    
    def analyze_rejected_tokens_settle(self, num_accepted_tokens, last_accepted_token):
        """
        Mark rejected tokens in accumulated_reject_flags during settle stage.
        The rejected token is the one right after the last accepted token.
        
        Args:
            num_accepted_tokens: Number of tokens accepted in settle stage
        """
        for i in range(num_accepted_tokens):
            self.accumulated_draft_confidences.append(self.temp_accumulated_draft_confidences[i])
            self.accumulated_verify_confidences.append(self.temp_accumulated_verify_confidences[i])
            self.accumulated_draft_logits.append(self.temp_accumulated_draft_logits[i])
            self.accumulated_verify_logits.append(self.temp_accumulated_verify_logits[i])
            self.accumulated_kl_divergences.append(self.temp_accumulated_kl_divergences[i])
            self.accumulated_tokens.append(self.temp_accumulated_tokens[i])
            self.accumulated_reject_flags.append(0)
            self.accumulated_step_idx.append(self.step_idx)  # Store current step index
        

        if num_accepted_tokens >= len(self.temp_accumulated_tokens):
            print(f"All {len(self.temp_accumulated_tokens)} accumulated tokens were accepted in settle stage")
            # return
        else:
            print(f"{num_accepted_tokens} out of {len(self.temp_accumulated_tokens)} accumulated tokens were accepted in settle stage")
            # store rejected token
            self.accumulated_draft_confidences.append(self.temp_accumulated_draft_confidences[num_accepted_tokens])
            self.accumulated_verify_confidences.append(self.temp_accumulated_verify_confidences[num_accepted_tokens])
            self.accumulated_draft_logits.append(self.temp_accumulated_draft_logits[num_accepted_tokens])
            self.accumulated_verify_logits.append(self.temp_accumulated_verify_logits[num_accepted_tokens])
            self.accumulated_kl_divergences.append(self.temp_accumulated_kl_divergences[num_accepted_tokens])
            self.accumulated_tokens.append(self.temp_accumulated_tokens[num_accepted_tokens])
            self.accumulated_reject_flags.append(1)  # Mark the rejected token
            self.accumulated_step_idx.append(self.step_idx)  # Store current step index

        # Sanity check
        if num_accepted_tokens != 0 and self.temp_accumulated_tokens[num_accepted_tokens-1] != last_accepted_token:
            raise ValueError("Last accepted token does not match the expected token")

        # reset temporary accumulators
        self.temp_accumulated_draft_confidences = []
        self.temp_accumulated_verify_confidences = []
        self.temp_accumulated_draft_logits = []
        self.temp_accumulated_verify_logits = []
        self.temp_accumulated_kl_divergences = []
        self.temp_accumulated_tokens = []
        self.temp_accumulated_reject_flags = []

    def raise_step_idx(self):
        self.step_idx += 1
    
    def save_histograms(self, output_dir, filename_prefix):
        """
        Save histograms and raw data for KL divergences of all tokens and rejected tokens.
        Creates separate histograms per confidence bin as well.
        
        Args:
            output_dir: Directory to save the output files
            filename_prefix: Prefix for the output filenames
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for analysis
        all_kl_divs = []
        rejected_kl_divs = []
        all_confidences = []
        rejected_confidences = []
        
        # Separate data by bins
        bin_all_kl_divs = {f"bin_{i}": [] for i in range(self.num_bins)}
        bin_rejected_kl_divs = {f"bin_{i}": [] for i in range(self.num_bins)}
        
        # Process accumulated data
        for i in range(len(self.accumulated_kl_divergences)):
            if self.accumulated_kl_divergences[i] is not None and self.accumulated_draft_confidences[i] is not None:
                kl_div = self.accumulated_kl_divergences[i]
                confidence = self.accumulated_draft_confidences[i]
                is_rejected = self.accumulated_reject_flags[i] == 1
                
                all_kl_divs.append(kl_div)
                all_confidences.append(confidence)
                
                # Get bin index
                bin_idx = self.get_bin_index(confidence)
                bin_all_kl_divs[f"bin_{bin_idx}"].append(kl_div)
                
                if is_rejected:
                    rejected_kl_divs.append(kl_div)
                    rejected_confidences.append(confidence)
                    bin_rejected_kl_divs[f"bin_{bin_idx}"].append(kl_div)
        
        # Save raw data to CSV
        self._save_raw_data_csv(output_dir, filename_prefix)
        
        # Create and save overall histograms
        self._create_overall_histograms(output_dir, filename_prefix, all_kl_divs, rejected_kl_divs)
        
        # Create and save per-bin histograms
        self._create_per_bin_histograms(output_dir, filename_prefix, bin_all_kl_divs, bin_rejected_kl_divs)
        
        print(f"Histograms and data saved to {output_dir} with prefix {filename_prefix}")
        print(f"Total tokens analyzed: {len(all_kl_divs)}")
        print(f"Total rejected tokens: {len(rejected_kl_divs)}")
    
    def _save_raw_data_csv(self, output_dir, filename_prefix):
        """Save all accumulated raw data to CSV"""
        raw_data_file = os.path.join(output_dir, f"{filename_prefix}_raw_data.csv")
        
        headers = [
            "step_idx", "token_id", "draft_confidence", "verify_confidence", 
            "kl_divergence", "token_value", "is_rejected", "confidence_bin"
        ]
        
        with open(raw_data_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for i in range(len(self.accumulated_step_idx)):
                # Skip entries with None values
                if (self.accumulated_kl_divergences[i] is not None and 
                    self.accumulated_draft_confidences[i] is not None):
                    
                    confidence = self.accumulated_draft_confidences[i]
                    bin_idx = self.get_bin_index(confidence)
                    
                    row = [
                        self.accumulated_step_idx[i],
                        i,
                        confidence,
                        self.accumulated_verify_confidences[i],
                        self.accumulated_kl_divergences[i],
                        self.accumulated_tokens[i],
                        self.accumulated_reject_flags[i],
                        bin_idx
                    ]
                    writer.writerow(row)
    
    def _create_overall_histograms(self, output_dir, filename_prefix, all_kl_divs, rejected_kl_divs):
        """Create overall histograms for all tokens and rejected tokens"""
        if len(all_kl_divs) == 0:
            print("No KL divergence data to plot")
            return
        
        # Create histogram data
        bins = 50
        
        # All tokens histogram
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(all_kl_divs, bins=bins, alpha=0.7, color='blue', edgecolor='black')
        plt.title(f'All Tokens KL Divergence Distribution\n(n={len(all_kl_divs)})')
        plt.xlabel('KL Divergence')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Rejected tokens histogram
        plt.subplot(1, 2, 2)
        if len(rejected_kl_divs) > 0:
            plt.hist(rejected_kl_divs, bins=bins, alpha=0.7, color='red', edgecolor='black')
            plt.title(f'Rejected Tokens KL Divergence Distribution\n(n={len(rejected_kl_divs)})')
        else:
            plt.title('Rejected Tokens KL Divergence Distribution\n(n=0)')
        plt.xlabel('KL Divergence')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(output_dir, f"{filename_prefix}_overall_kl_histogram.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save histogram data to CSV
        hist_data_file = os.path.join(output_dir, f"{filename_prefix}_overall_histogram_data.csv")
        
        # Calculate histogram data
        all_counts, all_bin_edges = np.histogram(all_kl_divs, bins=bins)
        if len(rejected_kl_divs) > 0:
            rejected_counts, rejected_bin_edges = np.histogram(rejected_kl_divs, bins=bins)
        else:
            rejected_counts = np.zeros(bins)
            rejected_bin_edges = all_bin_edges
        
        with open(hist_data_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['bin_start', 'bin_end', 'all_tokens_count', 'rejected_tokens_count'])
            
            for i in range(bins):
                writer.writerow([
                    all_bin_edges[i], 
                    all_bin_edges[i+1], 
                    all_counts[i], 
                    rejected_counts[i]
                ])
    
    def _create_per_bin_histograms(self, output_dir, filename_prefix, bin_all_kl_divs, bin_rejected_kl_divs):
        """Create separate histograms for each confidence bin"""
        # Calculate number of subplots needed
        rows = (self.num_bins + 1) // 2
        cols = 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        bin_hist_data = []
        
        for i in range(self.num_bins):
            row = i // cols
            col = i % cols
            
            if rows > 1:
                ax = axes[row, col]
            else:
                ax = axes[col] if cols > 1 else axes
            
            bin_all = bin_all_kl_divs[f"bin_{i}"]
            bin_rejected = bin_rejected_kl_divs[f"bin_{i}"]
            
            bin_start, bin_end = self.bin_ranges[i]
            
            if len(bin_all) > 0:
                # Plot histogram for this bin
                ax.hist(bin_all, bins=20, alpha=0.7, color='blue', label=f'All (n={len(bin_all)})', edgecolor='black')
                
                if len(bin_rejected) > 0:
                    ax.hist(bin_rejected, bins=20, alpha=0.7, color='red', label=f'Rejected (n={len(bin_rejected)})', edgecolor='black')
                
                ax.set_title(f'Bin {i}: Confidence [{bin_start:.1f}, {bin_end:.1f})')
                ax.set_xlabel('KL Divergence')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Store histogram data
                if len(bin_all) > 0:
                    all_counts, all_bin_edges = np.histogram(bin_all, bins=20)
                    if len(bin_rejected) > 0:
                        rejected_counts, _ = np.histogram(bin_rejected, bins=all_bin_edges)
                    else:
                        rejected_counts = np.zeros(20)
                    
                    bin_hist_data.append({
                        'confidence_bin': i,
                        'confidence_range': f"[{bin_start:.1f}, {bin_end:.1f})",
                        'all_counts': all_counts,
                        'rejected_counts': rejected_counts,
                        'bin_edges': all_bin_edges,
                        'total_all': len(bin_all),
                        'total_rejected': len(bin_rejected)
                    })
            else:
                ax.set_title(f'Bin {i}: Confidence [{bin_start:.1f}, {bin_end:.1f}) - No Data')
                ax.set_xlabel('KL Divergence')
                ax.set_ylabel('Frequency')
        
        # Hide empty subplots
        total_subplots = rows * cols
        for i in range(self.num_bins, total_subplots):
            row = i // cols
            col = i % cols
            if rows > 1:
                axes[row, col].set_visible(False)
            else:
                if cols > 1 and i < len(axes):
                    axes[col].set_visible(False)
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(output_dir, f"{filename_prefix}_per_bin_kl_histograms.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save per-bin histogram data to CSV
        bin_hist_file = os.path.join(output_dir, f"{filename_prefix}_per_bin_histogram_data.csv")
        
        with open(bin_hist_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'confidence_bin', 'confidence_range', 'kl_bin_start', 'kl_bin_end', 
                'all_tokens_count', 'rejected_tokens_count', 'total_all_in_bin', 'total_rejected_in_bin'
            ])
            
            for bin_data in bin_hist_data:
                for i in range(len(bin_data['all_counts'])):
                    writer.writerow([
                        bin_data['confidence_bin'],
                        bin_data['confidence_range'],
                        bin_data['bin_edges'][i],
                        bin_data['bin_edges'][i+1],
                        bin_data['all_counts'][i],
                        bin_data['rejected_counts'][i],
                        bin_data['total_all'],
                        bin_data['total_rejected']
                    ])