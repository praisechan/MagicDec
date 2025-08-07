import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

class ConfidenceAnalyzer:
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
        
        # For all tokens
        self.all_tokens_data = {f"bin_{i}": [] for i in range(num_bins)}
        
        # For rejected tokens only
        self.rejected_tokens_data = {f"bin_{i}": [] for i in range(num_bins)}
        
        # Temporary storage for current speculation cycle
        self.current_draft_confidences = None
        self.current_verify_confidences = None
        
        # Storage for accumulated unsettled tokens (between settlement cycles)
        self.accumulated_draft_confidences = []
        self.accumulated_verify_confidences = []
        
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
    
    def store_draft_confidences(self, draft_top1_top2_diff):
        """Store confidences from speculation"""
        if draft_top1_top2_diff is not None:
            self.current_draft_confidences = [float(x) for x in draft_top1_top2_diff]
        else:
            self.current_draft_confidences = None
    
    def store_verify_confidences(self, verify_top1_top2_diff):
        """Store confidences from verification"""
        if verify_top1_top2_diff is not None:
            self.current_verify_confidences = [float(x) for x in verify_top1_top2_diff]
        else:
            self.current_verify_confidences = None
    
    def accumulate_confidences_after_verify(self, num_accepted_tokens):
        """Accumulate draft and verify confidences after each verify call for later settle analysis"""
        if (self.current_draft_confidences is None or 
            self.current_verify_confidences is None):
            return
        
        # Only accumulate tokens up to accepted position (plus one for the bonus token)
        # We include the bonus token because it will be part of the unsettled tokens
        max_tokens_to_accumulate = num_accepted_tokens + 1
        # max_available = min(len(self.current_draft_confidences), 
        #                    len(self.current_verify_confidences))
        # tokens_to_accumulate = min(max_tokens_to_accumulate, len(self.current_verify_confidences))
        
        # Add accepted tokens + bonus token to accumulated storage
        for i in range(max_tokens_to_accumulate):
            if i >= len(self.current_draft_confidences):
                self.accumulated_draft_confidences.append(None)
                self.accumulated_verify_confidences.append(self.current_verify_confidences[i])
            else:
                self.accumulated_draft_confidences.append(self.current_draft_confidences[i])
                self.accumulated_verify_confidences.append(self.current_verify_confidences[i])
    
    def analyze_all_tokens(self, num_accepted_tokens):
        """Analyze confidence changes for all tokens up to accepted position"""
        if (self.current_draft_confidences is None or 
            self.current_verify_confidences is None):
            return
        
        # Only analyze tokens up to the number of accepted tokens
        min_len = min(len(self.current_draft_confidences), 
                     len(self.current_verify_confidences),
                     num_accepted_tokens)
        
        for i in range(min_len):
            draft_conf = self.current_draft_confidences[i]
            verify_conf = self.current_verify_confidences[i]
            
            # Get bin based on draft confidence
            bin_idx = self.get_bin_index(draft_conf)
            
            # Calculate confidence change (draft - verify)
            conf_change = draft_conf - verify_conf
            
            self.all_tokens_data[f"bin_{bin_idx}"].append(conf_change)
    
    def analyze_rejected_tokens_settle(self, num_accepted_tokens):
        """Analyze confidence changes for rejected tokens in settle stage using accumulated data"""
        if (len(self.accumulated_draft_confidences) == 0 or 
            len(self.accumulated_verify_confidences) == 0):
            return
        
        # Find the first rejected token (if any)
        max_tokens = min(len(self.accumulated_draft_confidences), 
                        len(self.accumulated_verify_confidences))
        
        if num_accepted_tokens < max_tokens:
            # There is a rejected token at position num_accepted_tokens
            rejected_idx = num_accepted_tokens
            
            draft_conf = self.accumulated_draft_confidences[rejected_idx]
            verify_conf = self.accumulated_verify_confidences[rejected_idx]
            
            if draft_conf is None:
                # If draft confidence is None, we cannot analyze this token
                return
            
            # Get bin based on draft confidence
            bin_idx = self.get_bin_index(draft_conf)
            
            # Calculate confidence change (draft - verify)
            conf_change = draft_conf - verify_conf
            
            self.rejected_tokens_data[f"bin_{bin_idx}"].append(conf_change)
        
    def get_accumulated_stats(self):
        """Get statistics about accumulated data for debugging"""
        return {
            'draft_count': len(self.accumulated_draft_confidences),
            'verify_count': len(self.accumulated_verify_confidences),
            'current_draft_count': len(self.current_draft_confidences) if self.current_draft_confidences else 0,
            'current_verify_count': len(self.current_verify_confidences) if self.current_verify_confidences else 0
        }
        
    def reset_accumulated_data(self):
        """Reset accumulated storage after settlement"""
        self.current_draft_confidences = None
        self.current_verify_confidences = None

        self.accumulated_draft_confidences = []
        self.accumulated_verify_confidences = []
    
    def save_histograms(self, output_dir="confidence_analysis", filename_prefix=""):
        """Save histograms for each bin"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save all tokens histograms
        for i, (bin_key, data) in enumerate(self.all_tokens_data.items()):
            if len(data) > 0:
                plt.figure(figsize=(10, 6))
                plt.hist(data, bins=50, alpha=0.7, edgecolor='black')
                plt.title(f'All Tokens - Confidence Change Distribution\nBin {i}: [{self.bin_ranges[i][0]:.2f}, {self.bin_ranges[i][1]:.2f})')
                plt.xlabel('Draft Confidence - Verify Confidence')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Change')
                plt.legend()
                filename = f"{filename_prefix}_all_tokens_bin_{i}.png" if filename_prefix else f'all_tokens_bin_{i}.png'
                plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
                plt.close()
        
        # Save rejected tokens histograms
        for i, (bin_key, data) in enumerate(self.rejected_tokens_data.items()):
            if len(data) > 0:
                plt.figure(figsize=(10, 6))
                plt.hist(data, bins=50, alpha=0.7, edgecolor='black', color='orange')
                plt.title(f'Rejected Tokens - Confidence Change Distribution\nBin {i}: [{self.bin_ranges[i][0]:.2f}, {self.bin_ranges[i][1]:.2f})')
                plt.xlabel('Draft Confidence - Verify Confidence')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Change')
                plt.legend()
                filename = f"{filename_prefix}_rejected_tokens_bin_{i}.png" if filename_prefix else f'rejected_tokens_bin_{i}.png'
                plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
                plt.close()
    
    def save_statistics(self, output_dir="confidence_analysis", filename_prefix="", num_histogram_bins=50):
        """Save detailed statistics including histogram data to CSV"""
        os.makedirs(output_dir, exist_ok=True)
        
        # All tokens statistics with histogram data for each bin
        for i, (bin_key, data) in enumerate(self.all_tokens_data.items()):
            if len(data) > 0:
                # Create histogram data
                hist_counts, hist_edges = np.histogram(data, bins=num_histogram_bins)
                hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
                
                # Save individual bin histogram data
                bin_filename = f"{filename_prefix}_all_tokens_bin_{i}_histogram.csv" if filename_prefix else f'all_tokens_bin_{i}_histogram.csv'
                with open(os.path.join(output_dir, bin_filename), 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Bin_Center', 'Count', 'Bin_Start', 'Bin_End'])
                    for j in range(len(hist_counts)):
                        writer.writerow([hist_centers[j], hist_counts[j], hist_edges[j], hist_edges[j+1]])
                
                # Save raw data for this bin
                raw_filename = f"{filename_prefix}_all_tokens_bin_{i}_raw.csv" if filename_prefix else f'all_tokens_bin_{i}_raw.csv'
                with open(os.path.join(output_dir, raw_filename), 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Confidence_Change'])
                    for value in data:
                        writer.writerow([value])
        
        # All tokens summary statistics
        all_tokens_filename = f"{filename_prefix}_all_tokens_stats.csv" if filename_prefix else 'all_tokens_stats.csv'
        with open(os.path.join(output_dir, all_tokens_filename), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Bin', 'Range', 'Count', 'Mean', 'Std', 'Min', 'Max', 'Percentile_25', 'Percentile_50', 'Percentile_75'])
            
            for i, (bin_key, data) in enumerate(self.all_tokens_data.items()):
                if len(data) > 0:
                    data_array = np.array(data)
                    writer.writerow([
                        i, f"[{self.bin_ranges[i][0]:.2f}, {self.bin_ranges[i][1]:.2f})",
                        len(data), np.mean(data_array), np.std(data_array),
                        np.min(data_array), np.max(data_array),
                        np.percentile(data_array, 25), np.percentile(data_array, 50), np.percentile(data_array, 75)
                    ])
                else:
                    writer.writerow([
                        i, f"[{self.bin_ranges[i][0]:.2f}, {self.bin_ranges[i][1]:.2f})",
                        0, 0, 0, 0, 0, 0, 0, 0
                    ])
        
        # Rejected tokens statistics with histogram data for each bin
        for i, (bin_key, data) in enumerate(self.rejected_tokens_data.items()):
            if len(data) > 0:
                # Create histogram data
                hist_counts, hist_edges = np.histogram(data, bins=num_histogram_bins)
                hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
                
                # Save individual bin histogram data
                bin_filename = f"{filename_prefix}_rejected_tokens_bin_{i}_histogram.csv" if filename_prefix else f'rejected_tokens_bin_{i}_histogram.csv'
                with open(os.path.join(output_dir, bin_filename), 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Bin_Center', 'Count', 'Bin_Start', 'Bin_End'])
                    for j in range(len(hist_counts)):
                        writer.writerow([hist_centers[j], hist_counts[j], hist_edges[j], hist_edges[j+1]])
                
                # Save raw data for this bin
                raw_filename = f"{filename_prefix}_rejected_tokens_bin_{i}_raw.csv" if filename_prefix else f'rejected_tokens_bin_{i}_raw.csv'
                with open(os.path.join(output_dir, raw_filename), 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Confidence_Change'])
                    for value in data:
                        writer.writerow([value])
        
        # Rejected tokens summary statistics
        rejected_tokens_filename = f"{filename_prefix}_rejected_tokens_stats.csv" if filename_prefix else 'rejected_tokens_stats.csv'
        with open(os.path.join(output_dir, rejected_tokens_filename), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Bin', 'Range', 'Count', 'Mean', 'Std', 'Min', 'Max', 'Percentile_25', 'Percentile_50', 'Percentile_75'])
            
            for i, (bin_key, data) in enumerate(self.rejected_tokens_data.items()):
                if len(data) > 0:
                    data_array = np.array(data)
                    writer.writerow([
                        i, f"[{self.bin_ranges[i][0]:.2f}, {self.bin_ranges[i][1]:.2f})",
                        len(data), np.mean(data_array), np.std(data_array),
                        np.min(data_array), np.max(data_array),
                        np.percentile(data_array, 25), np.percentile(data_array, 50), np.percentile(data_array, 75)
                    ])
                else:
                    writer.writerow([
                        i, f"[{self.bin_ranges[i][0]:.2f}, {self.bin_ranges[i][1]:.2f})",
                        0, 0, 0, 0, 0, 0, 0, 0
                    ])

# Add KL divergence analysis class with confidence binning
class KLConfidenceAnalyzer:
    def __init__(self, num_bins=10, bin_width=0.1, center=0.0, kl_threshold=0.1):
        self.num_bins = num_bins
        self.bin_width = bin_width
        self.center = center
        self.kl_threshold = kl_threshold

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
    
    def store_draft_data(self, draft_logits, draft_top1_top2_diff, use_extended_verification=False):
        """Store logits and confidences from speculation"""
        if draft_logits is not None:
            self.current_draft_logits = draft_logits
        else:
            self.current_draft_logits = None
            
        if draft_top1_top2_diff is not None:
            self.current_draft_confidences = [float(x) for x in draft_top1_top2_diff]
        else:
            self.current_draft_confidences = None
    
    def store_verify_data(self, verify_logits, verify_top1_top2_diff, use_extended_verification=False):
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
            if i == max_tokens_to_accumulate-1:
                self.accumulated_draft_logits.append(None)
                self.accumulated_draft_confidences.append(None)
            else:
                self.accumulated_draft_logits.append(self.current_draft_logits[i])
                self.accumulated_draft_confidences.append(self.current_draft_confidences[i])
            
            self.accumulated_verify_logits.append(self.current_verify_logits[i])
            self.accumulated_verify_confidences.append(self.current_verify_confidences[i])
    
    def analyze_all_tokens(self, num_accepted_tokens, use_extended_verification=False):
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

        kl_divergences = []
        kl_threshold_exceeded = False
        if min_len > 0:
            for i in range(min_len):
                draft_logits = self.current_draft_logits[i]
                verify_logits = self.current_verify_logits[i]
                draft_conf = self.current_draft_confidences[i]
                
                # Get bin based on draft confidence
                bin_idx = self.get_bin_index(draft_conf)
                
                # Compute KL divergence between draft and verify logits
                kl_div = self.compute_kl_divergence(draft_logits, verify_logits)
                
                self.all_tokens_kl_data[f"bin_{bin_idx}"].append(kl_div)

                
                kl_divergences.append(kl_div)
            
            max_kl_divergence = max(kl_divergences)
            kl_threshold_exceeded = max_kl_divergence > self.kl_threshold
            print(f"KL divergence check: max={max_kl_divergence:.4f}, threshold={self.kl_threshold}, exceeded={kl_threshold_exceeded}")
        
        return kl_threshold_exceeded
    
    def analyze_rejected_tokens_settle(self, num_accepted_tokens, step=None, settle_call_number=None, 
                                      log_file_path=None, dataset=None, prefix_len=None, 
                                      gamma1=None, gamma2=None, budget1=None, budget2=None):
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
            
            # Log the rejected token details if log file path is provided
            if log_file_path is not None and all(v is not None for v in [step, settle_call_number, dataset, prefix_len, gamma1, gamma2, budget1, budget2]):
                rejected_token_data = [
                    step, settle_call_number, rejected_idx, kl_div, draft_conf, 
                    bin_idx, dataset, prefix_len, gamma1, gamma2, budget1, budget2
                ]
                
                with open(log_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(rejected_token_data)
                
                print(f"Logged rejected token: step={step}, settle_call={settle_call_number}, position={rejected_idx}, kl_div={kl_div:.4f}, confidence={draft_conf:.4f}, bin={bin_idx}")
        
        
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

# Add KL divergence analysis class with confidence binning
class KLConfidenceAnalyzer_temp:
    def __init__(self, num_bins=10, bin_width=0.1, center=0.0, kl_threshold=0.1):
        self.num_bins = num_bins
        self.bin_width = bin_width
        self.center = center
        self.kl_threshold = kl_threshold

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
    
    def store_draft_data(self, draft_logits, draft_top1_top2_diff, use_extended_verification=False):
        if not use_extended_verification:
            """Store logits and confidences from speculation"""
            if draft_logits is not None:
                self.current_draft_logits = draft_logits
            else:
                self.current_draft_logits = None
                
            if draft_top1_top2_diff is not None:
                self.current_draft_confidences = [float(x) for x in draft_top1_top2_diff]
            else:
                self.current_draft_confidences = None
    
    def store_verify_data(self, verify_logits, verify_top1_top2_diff, use_extended_verification=False):
        if not use_extended_verification:
          """Store logits and confidences from verification"""
          if verify_logits is not None:
              self.current_verify_logits = verify_logits
          else:
              self.current_verify_logits = None
              
          if verify_top1_top2_diff is not None:
              self.current_verify_confidences = [float(x) for x in verify_top1_top2_diff]
          else:
              self.current_verify_confidences = None

    def analyze_all_tokens(self, num_accepted_tokens, use_extended_verification=False):
        if not use_extended_verification:
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

            kl_divergences = []
            kl_threshold_exceeded = False
            if min_len > 0:
                for i in range(min_len):
                    draft_logits = self.current_draft_logits[i]
                    verify_logits = self.current_verify_logits[i]
                    draft_conf = self.current_draft_confidences[i]
                    
                    # Get bin based on draft confidence
                    bin_idx = self.get_bin_index(draft_conf)
                    
                    # Compute KL divergence between draft and verify logits
                    kl_div = self.compute_kl_divergence(draft_logits, verify_logits)
                    
                    self.all_tokens_kl_data[f"bin_{bin_idx}"].append(kl_div)
                    
                    kl_divergences.append(kl_div)
                    if kl_div > self.kl_threshold:
                        print(f"KL divergence exceeded threshold: {kl_div:.4f} > {self.kl_threshold:.4f} (draft_conf={draft_conf:.4f})")
                
                max_kl_divergence = max(kl_divergences)
                kl_threshold_exceeded = max_kl_divergence > self.kl_threshold
                print(f"KL divergence check: max={max_kl_divergence:.4f}, threshold={self.kl_threshold}, exceeded={kl_threshold_exceeded}")

            #TODO: decide kl_threshold_exceeded based kl_divergence and draft confidence
            #TODO: you should set each kl threshold for each bin
            #TODO: this is done by calibration set
            
            return kl_threshold_exceeded
        else:
            return False  # No analysis for extended verification
