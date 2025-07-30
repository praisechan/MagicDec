import os
import csv
import numpy as np
import matplotlib.pyplot as plt


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
