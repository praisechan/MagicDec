import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import os

def load_draft_history(filepath):
    """Load and inspect draft history from .pt file"""
    draft_history = torch.load(filepath, map_location='cpu')
    
    print(f"Loaded draft history with {len(draft_history)} steps")
    breakpoint()
    # Print structure of first step as example
    if draft_history:
        first_step = draft_history[0]
        print(f"Step structure:")
        print(f"  - step: {first_step['step']}")
        print(f"  - input_ids shape: {first_step['input_ids'].shape}")
        print(f"  - draft_iter keys: {list(first_step['draft_iter'].keys())}")
        print(f"  - number of draft iterations: {len(first_step['draft_iter']['draft_tokens'])}")
        
        if first_step['draft_iter']['draft_tokens']:
            print(f"  - first draft_tokens shape: {first_step['draft_iter']['draft_tokens'][0].shape}")
    
    return draft_history

def load_metadata(filepath):
    """Load metadata from .pt file"""
    try:
        metadata = torch.load(filepath, map_location='cpu')
        print(f"Loaded metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        return metadata
    except FileNotFoundError:
        print(f"Metadata file not found: {filepath}")
        return None

def extract_all_draft_tokens(draft_history):
    """Extract all draft tokens from all steps"""
    all_draft_tokens = []
    for step_data in draft_history:
        for draft_iter in step_data['draft_iter']['draft_tokens']:
            all_draft_tokens.append(draft_iter)
    return all_draft_tokens

def extract_acceptance_rates(draft_history):
    """Extract acceptance rates per step"""
    step_acceptance_rates = []
    for step_data in draft_history:
        step_accepts = []
        for accept_flags in step_data['draft_iter']['accept_flags_matrix']:
            # Calculate acceptance rate for this iteration
            acceptance_rate = accept_flags.float().mean().item()
            step_accepts.append(acceptance_rate)
        step_acceptance_rates.append(step_accepts)
    return step_acceptance_rates

def extract_top1_top2_differences(draft_history):
    """Extract top1-top2 probability differences"""
    all_differences = []
    for step_data in draft_history:
        for diff_list in step_data['draft_iter']['draft_top1_top2_diff']:
            for diff in diff_list:
                if torch.is_tensor(diff):
                    all_differences.extend(diff.cpu().float().numpy().flatten())
                else:
                    all_differences.append(float(diff))
    return np.array(all_differences)

def analyze_draft_history(draft_history_path, metadata_path=None, output_dir=None):
    """Comprehensive analysis of draft history"""
    
    # Load data
    draft_history = load_draft_history(draft_history_path)
    metadata = load_metadata(metadata_path) if metadata_path else None
    
    # Extract data
    all_draft_tokens = extract_all_draft_tokens(draft_history)
    acceptance_rates = extract_acceptance_rates(draft_history)
    top1_top2_diffs = extract_top1_top2_differences(draft_history)
    
    # Analysis results
    results = {
        'total_steps': len(draft_history),
        'total_draft_iterations': sum(len(step['draft_iter']['draft_tokens']) for step in draft_history),
        'avg_iterations_per_step': np.mean([len(step['draft_iter']['draft_tokens']) for step in draft_history]),
        'acceptance_rate_stats': {
            'mean': np.mean([rate for step_rates in acceptance_rates for rate in step_rates]),
            'std': np.std([rate for step_rates in acceptance_rates for rate in step_rates]),
            'min': np.min([rate for step_rates in acceptance_rates for rate in step_rates]),
            'max': np.max([rate for step_rates in acceptance_rates for rate in step_rates])
        },
        'top1_top2_diff_stats': {
            'mean': np.mean(top1_top2_diffs),
            'std': np.std(top1_top2_diffs),
            'min': np.min(top1_top2_diffs),
            'max': np.max(top1_top2_diffs)
        }
    }
    
    # Print analysis
    print("\n=== DRAFT HISTORY ANALYSIS ===")
    print(f"Total steps: {results['total_steps']}")
    print(f"Total draft iterations: {results['total_draft_iterations']}")
    print(f"Average iterations per step: {results['avg_iterations_per_step']:.2f}")
    
    print(f"\nAcceptance Rate Statistics:")
    print(f"  Mean: {results['acceptance_rate_stats']['mean']:.4f}")
    print(f"  Std:  {results['acceptance_rate_stats']['std']:.4f}")
    print(f"  Min:  {results['acceptance_rate_stats']['min']:.4f}")
    print(f"  Max:  {results['acceptance_rate_stats']['max']:.4f}")
    
    print(f"\nTop1-Top2 Difference Statistics:")
    print(f"  Mean: {results['top1_top2_diff_stats']['mean']:.4f}")
    print(f"  Std:  {results['top1_top2_diff_stats']['std']:.4f}")
    print(f"  Min:  {results['top1_top2_diff_stats']['min']:.4f}")
    print(f"  Max:  {results['top1_top2_diff_stats']['max']:.4f}")
    
    # Create visualizations if output directory is provided
    if output_dir:
        create_visualizations(acceptance_rates, top1_top2_diffs, results, output_dir)
    
    return results, draft_history, metadata

def create_visualizations(acceptance_rates, top1_top2_diffs, results, output_dir):
    """Create and save visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Flatten acceptance rates
    flat_acceptance_rates = [rate for step_rates in acceptance_rates for rate in step_rates]
    
    # 1. Acceptance rate histogram
    plt.figure(figsize=(10, 6))
    plt.hist(flat_acceptance_rates, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Acceptance Rate')
    plt.ylabel('Frequency')
    plt.title('Distribution of Acceptance Rates')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/acceptance_rate_histogram.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Top1-Top2 difference histogram
    plt.figure(figsize=(10, 6))
    plt.hist(top1_top2_diffs, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Top1-Top2 Probability Difference')
    plt.ylabel('Frequency')
    plt.title('Distribution of Top1-Top2 Probability Differences')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/top1_top2_diff_histogram.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Acceptance rate over time
    plt.figure(figsize=(12, 6))
    step_means = [np.mean(step_rates) for step_rates in acceptance_rates]
    plt.plot(step_means, marker='o', markersize=3)
    plt.xlabel('Step')
    plt.ylabel('Mean Acceptance Rate')
    plt.title('Acceptance Rate Over Time')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/acceptance_rate_over_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Analyze draft history from .pt files')
    parser.add_argument('--draft_history_path', type=str, required=True,
                       help='Path to draft history .pt file')
    parser.add_argument('--metadata_path', type=str, default=None,
                       help='Path to metadata .pt file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save analysis results and visualizations')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.draft_history_path):
        print(f"Error: Draft history file not found: {args.draft_history_path}")
        return
    
    # If metadata path not provided, try to infer it
    if args.metadata_path is None:
        base_path = args.draft_history_path.replace('_draft_history.pt', '_metadata.pt')
        if os.path.exists(base_path):
            args.metadata_path = base_path
            print(f"Found metadata file: {args.metadata_path}")
    
    # Run analysis
    results, draft_history, metadata = analyze_draft_history(
        args.draft_history_path, 
        args.metadata_path, 
        args.output_dir
    )
    
    # Save results if output directory provided
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save analysis results as JSON
        import json
        with open(f"{args.output_dir}/analysis_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Analysis results saved to: {args.output_dir}/analysis_results.json")

if __name__ == "__main__":
    main()