#!/usr/bin/env python
import argparse
import os
import math
import torch
from typing import List, Dict, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SSD_internal import (
    LayerData,
    HeadData,
    ClusterData,
)
from utils import compute_pages_per_cluster
from tqdm import tqdm
import csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute hot cluster hit ratio analysis"
    )
    parser.add_argument('--page_size_bytes', type=int, required=True)
    parser.add_argument('--vector_bytes', type=int, default=2)
    parser.add_argument('--profiling_dir', type=str, required=True,
                        help='Directory with layer-wise profiling .pt files')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='KV heads per layer')
    parser.add_argument('--cluster_size', type=int, default=16, required=True,
                        help='cluster size')
    parser.add_argument('--dram_capacity_gb', type=float, required=True,
                        help='DRAM capacity in GB for automatic hot cluster ratio calculation')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Batch size for profiling')
    parser.add_argument('--budget_ratio', type=str, default=0.25, required=True,
                        help='budget ratio')
    parser.add_argument('--prefix_len', type=str, default=16385, required=True)
    parser.add_argument('--head_dim', type=int, default=128,
                        help='Dimension per KV head')
    parser.add_argument('--constrained', action='store_true',
                        help='Constrain cluster size to certain value')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name for profiling directory')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name for profiling directory')
    parser.add_argument('--generate_name', type=str, required=True,
                        help='Generate name (e.g., speculate_0_0, verify_1_2)')
    parser.add_argument('--layer_num', type=int, default=32,
                        help='Number of layers to process')
    parser.add_argument('--csv_path', type=str,
                        help='Output CSV file path for results')
    
    return parser.parse_args()


def calculate_dram_page_count(
    dram_capacity_gb: float,
    batch_size: int,
    page_size_bytes: int
) -> int:
    """
    Calculate total number of pages available in DRAM.
    
    Args:
        dram_capacity_gb: DRAM capacity in GB
        batch_size: Batch size for profiling
        page_size_bytes: Size of each page in bytes
    
    Returns:
        total_pages: Total number of pages available in DRAM
    """
    print(f"\n=== CALCULATING DRAM PAGE COUNT ===")
    
    # Convert DRAM capacity to bytes
    dram_capacity_bytes = dram_capacity_gb * (1024 ** 3) / batch_size  # GB to bytes, divided by batch size
    
    # Calculate total available pages in DRAM
    total_available_pages = int(dram_capacity_bytes / page_size_bytes)
    
    print(f"DRAM Capacity: {dram_capacity_gb} GB")
    print(f"Batch Size: {batch_size}")
    print(f"Effective DRAM per batch: {dram_capacity_bytes:,} bytes")
    print(f"Page Size: {page_size_bytes:,} bytes")
    print(f"Total Available Pages: {total_available_pages:,}")
    print(f"==========================================")
    
    return total_available_pages

def load_profiling_layer(
    args,
    layer_idx: int,
    generate_name: str = None,
    step_idx: int = 0,
    dram_page_count: int = 0
) -> LayerData:
    """Load profiling data for a specific layer, similar to simulator.py"""
    if generate_name is None:
        raise ValueError("generate_name must be specified")
    
    # Try different cluster sizes in case of naming inconsistencies
    cluster_sizes_to_try = [args.cluster_size]
    
    # Add alternative cluster sizes based on common patterns
    if args.cluster_size == 16:
        cluster_sizes_to_try.extend([17, 15])
    elif args.cluster_size == 32:
        cluster_sizes_to_try.extend([33, 31])
    elif args.cluster_size == 64:
        cluster_sizes_to_try.extend([65, 63])
    elif args.cluster_size == 8:
        cluster_sizes_to_try.extend([9, 7])
    else:
        # For any other cluster size, try ±1
        cluster_sizes_to_try.extend([args.cluster_size + 1, args.cluster_size - 1])
    
    profiling_dir = None
    cluster_sizes = None
    successful_cluster_size = None
    
    # Try each cluster size until one works
    for cluster_size_attempt in cluster_sizes_to_try:
        try:
            profiling_dir = args.profiling_dir + f"{args.model_name}_{args.dataset}_{args.prefix_len}/{generate_name}/data_{args.budget_ratio}KV_clustersize_{cluster_size_attempt}"
            cluster_sizes = torch.load(
                os.path.join(profiling_dir, f"cluster_size_{layer_idx}.pt"), map_location='cpu'
            )
            successful_cluster_size = cluster_size_attempt
            if cluster_size_attempt != args.cluster_size:
                print(f"Note: Using cluster_size_{cluster_size_attempt} instead of {args.cluster_size} for {generate_name}")
            break
        except FileNotFoundError:
            if cluster_size_attempt == cluster_sizes_to_try[-1]:  # Only print on last attempt
                print(f"Warning: Could not find profiling data for cluster sizes: {cluster_sizes_to_try}")
            continue
    
    if cluster_sizes is None:
        raise FileNotFoundError(f"Could not load profiling data for any cluster size variants: {cluster_sizes_to_try}")
    
    # Load selected clusters and softmax scores
    selected_list = torch.load(
        os.path.join(profiling_dir, f"selected_cI_step{step_idx}_layer{layer_idx}.pt"), map_location='cpu'
    )

    # softmax sum should be <train_step_idx>_0, because hot cluster is selected only for decoding step 0.
    # Extract number after first underscore from generate_name (e.g., "speculate_0_84" -> 0)
    generate_parts = generate_name.split('_')
    if len(generate_parts) >= 2:
      train_step_idx = int(generate_parts[1])
    else:
      train_step_idx = 0
    profiling_dir_for_softmax = args.profiling_dir + f"{args.model_name}_{args.dataset}_{args.prefix_len}/speculate_{train_step_idx}_0/data_{args.budget_ratio}KV_clustersize_{cluster_size_attempt}"
    softmax_sum = torch.load(
        os.path.join(profiling_dir_for_softmax, f"softmax_sum_{layer_idx}.pt"), map_location='cpu'
    )

    # Create HeadData objects first to use compute_pages_per_cluster
    heads: List[HeadData] = []
    for head_idx in range(args.num_heads):
        clusters = [ClusterData(cid, int(size.item()))
                    for cid, size in enumerate(cluster_sizes[head_idx])]
        superclusters = None
        selected = [int(cid.item()) for cid in selected_list[head_idx]]
        
        # Create temporary HeadData to compute pages per cluster
        temp_head = HeadData(head_idx, clusters, superclusters, selected, [], softmax_sum[head_idx])
        
        # Get pages per cluster for this head
        pages_map = compute_pages_per_cluster(
            temp_head,
            args.page_size_bytes,
            args.vector_bytes,
            args.head_dim,
            args.constrained,
            args.cluster_size
        )
        
        # Calculate hot clusters based on DRAM page budget and softmax scores
        head_softmax_scores = softmax_sum[head_idx]
        num_clusters = len(head_softmax_scores)
        
        # Create list of (cluster_id, softmax_score, pages) tuples
        cluster_info = []
        for cid in range(num_clusters):
            cluster_info.append((cid, head_softmax_scores[cid].item(), pages_map.get(cid, 0)))
        
        # Sort by softmax score (descending)
        cluster_info.sort(key=lambda x: x[1], reverse=True)
        
        # Greedily select clusters that fit within DRAM page budget
        hot_cluster_ids = []
        total_pages_used = 0
        pages_per_head = dram_page_count // args.num_heads // args.layer_num  # Equal distribution across heads
        
        for cid, score, pages in cluster_info:
            if total_pages_used + pages <= pages_per_head:
                hot_cluster_ids.append(cid)
                total_pages_used += pages
            else:
                break  # Can't fit any more clusters
        
        print(f"Head {head_idx}: Selected {len(hot_cluster_ids)} hot clusters using {total_pages_used}/{pages_per_head} pages")
        
        heads.append(HeadData(head_idx, clusters, superclusters, selected, hot_cluster_ids, softmax_sum[head_idx]))
    
    return LayerData(layer_idx, heads)


def compute_hot_cluster_hit_ratio(
    args,
    layer: LayerData
) -> Tuple[Dict[int, Dict], Dict]:
    """
    Compute hot cluster hit ratio analysis for a layer.
    
    Returns:
        head_stats: Dict[head_idx, stats] where stats contains:
            - num_hot_clusters: number of hot clusters identified
            - num_selected_clusters: number of selected clusters
            - num_hot_selected: number of hot clusters that are selected (hits)
            - hot_hit_ratio: ratio of hot clusters that are selected
            - hot_pages: total pages in hot clusters
            - selected_pages: total pages in selected clusters
            - hot_selected_pages: pages in hot clusters that are selected
            - total_pages: total pages across all clusters
            - hot_page_hit_ratio: ratio of hot cluster pages that are selected
        layer_stats: aggregated statistics for the entire layer
    """
    head_stats = {}
    
    # Aggregate statistics across all heads
    total_hot_clusters = 0
    total_selected_clusters = 0
    total_hot_selected = 0
    total_hot_pages = 0
    total_selected_pages = 0
    total_hot_selected_pages = 0
    total_all_pages = 0
    
    for head in layer.heads:
        head_idx = head.head_index
        
        # Get pages per cluster for this head
        pages_map = compute_pages_per_cluster(
            head,
            args.page_size_bytes,
            args.vector_bytes,
            args.head_dim,
            args.constrained,
            args.cluster_size
        )
        
        # Identify hot and selected clusters
        hot_cluster_ids = set(head.hot_cluster_ids) if head.hot_cluster_ids else set()
        selected_cluster_ids = set(head.selected_cluster_ids)
        hot_selected_ids = hot_cluster_ids.intersection(selected_cluster_ids)
        
        # Count clusters
        num_hot_clusters = len(hot_cluster_ids)
        num_selected_clusters = len(selected_cluster_ids)
        num_hot_selected = len(hot_selected_ids)
        
        # Calculate pages
        hot_pages = sum(pages_map[cid] for cid in hot_cluster_ids if cid in pages_map)
        selected_pages = sum(pages_map[cid] for cid in selected_cluster_ids if cid in pages_map)
        hot_selected_pages = sum(pages_map[cid] for cid in hot_selected_ids if cid in pages_map)
        total_pages = sum(pages_map.values())
        
        # Calculate ratios
        hot_hit_ratio = num_hot_selected / num_hot_clusters if num_hot_clusters > 0 else 0.0
        hot_page_hit_ratio = hot_selected_pages / hot_pages if hot_pages > 0 else 0.0
        
        # Store per-head statistics
        head_stats[head_idx] = {
            'num_hot_clusters': num_hot_clusters,
            'num_selected_clusters': num_selected_clusters,
            'num_hot_selected': num_hot_selected,
            'hot_hit_ratio': hot_hit_ratio,
            'hot_pages': hot_pages,
            'selected_pages': selected_pages,
            'hot_selected_pages': hot_selected_pages,
            'total_pages': total_pages,
            'hot_page_hit_ratio': hot_page_hit_ratio,
        }
        
        # Accumulate for layer statistics
        total_hot_clusters += num_hot_clusters
        total_selected_clusters += num_selected_clusters
        total_hot_selected += num_hot_selected
        total_hot_pages += hot_pages
        total_selected_pages += selected_pages
        total_hot_selected_pages += hot_selected_pages
        total_all_pages += total_pages
    
    # Calculate layer-wide statistics
    layer_stats = {
        'layer_idx': layer.layer_index,
        'total_hot_clusters': total_hot_clusters,
        'total_selected_clusters': total_selected_clusters,
        'total_hot_selected': total_hot_selected,
        'avg_hot_hit_ratio': total_hot_selected / total_hot_clusters if total_hot_clusters > 0 else 0.0,
        'total_hot_pages': total_hot_pages,
        'total_selected_pages': total_selected_pages,
        'total_hot_selected_pages': total_hot_selected_pages,
        'total_all_pages': total_all_pages,
        'avg_hot_page_hit_ratio': total_hot_selected_pages / total_hot_pages if total_hot_pages > 0 else 0.0,
        'avg_hot_pages_per_head': total_hot_pages / args.num_heads,
        'avg_selected_pages_per_head': total_selected_pages / args.num_heads,
        'avg_hot_selected_pages_per_head': total_hot_selected_pages / args.num_heads,
        'avg_total_pages_per_head': total_all_pages / args.num_heads,
    }
    
    return head_stats, layer_stats


def main():
    args = parse_args()

    if args.model_name =="qwen2.5-14b":
        args.layer_num = 48
    elif args.model_name =="qwen2.5-32b":
        args.layer_num = 64
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    # Calculate DRAM page count for hot cluster selection
    dram_page_count = calculate_dram_page_count(
        dram_capacity_gb=args.dram_capacity_gb,
        batch_size=args.batch_size,
        page_size_bytes=args.page_size_bytes,
    )
    
    # Determine the number of steps based on the generate_name pattern
    if "speculate" in args.generate_name:
        max_steps = 4  # speculate has steps 0, 1, 2, 3
    else:  # verify
        max_steps = 1  # verify typically has only one step (step 0)
    
    step_aggregated_results = []
    
    # Process each step
    for step_idx in range(max_steps):
        print(f"\n=== Processing Step {step_idx} ===")
        step_results = []
        
        for layer_idx in tqdm(range(args.layer_num), desc=f"Processing layers for step {step_idx}"):
            try:
                # Load layer data with the pre-calculated hot cluster count
                layer = load_profiling_layer(args, layer_idx, args.generate_name, step_idx, dram_page_count)
                
                # Compute hot cluster hit ratio
                head_stats, layer_stats = compute_hot_cluster_hit_ratio(args, layer)
                
                # Add step information to layer stats
                layer_stats['step_idx'] = step_idx
                layer_stats['step_name'] = f"{args.generate_name}_step{step_idx}"
                
                step_results.append({
                    'layer_stats': layer_stats,
                    'head_stats': head_stats
                })
                
                # Print summary for this layer
                print(f"Layer {layer_idx}: Hot hit ratio = {layer_stats['avg_hot_hit_ratio']:.3f}, "
                      f"Hot page hit ratio = {layer_stats['avg_hot_page_hit_ratio']:.3f}, "
                      f"Avg hot pages/head = {layer_stats['avg_hot_pages_per_head']:.1f}")
                
            except Exception as e:
                print(f"Error processing layer {layer_idx}, step {step_idx}: {e}")
                continue
        
        # Calculate step-level aggregated statistics
        if step_results:
            # Aggregate across all layers in this step
            total_hot_clusters = sum(r['layer_stats']['total_hot_clusters'] for r in step_results)
            total_selected_clusters = sum(r['layer_stats']['total_selected_clusters'] for r in step_results)
            total_hot_selected = sum(r['layer_stats']['total_hot_selected'] for r in step_results)
            total_hot_pages = sum(r['layer_stats']['total_hot_pages'] for r in step_results)
            total_selected_pages = sum(r['layer_stats']['total_selected_pages'] for r in step_results)
            total_hot_selected_pages = sum(r['layer_stats']['total_hot_selected_pages'] for r in step_results)
            total_all_pages = sum(r['layer_stats']['total_all_pages'] for r in step_results)
            
            # Calculate step-level averages
            avg_hot_hit_ratio = sum(r['layer_stats']['avg_hot_hit_ratio'] for r in step_results) / len(step_results)
            avg_hot_page_hit_ratio = sum(r['layer_stats']['avg_hot_page_hit_ratio'] for r in step_results) / len(step_results)
            avg_hot_pages_per_head = sum(r['layer_stats']['avg_hot_pages_per_head'] for r in step_results) / len(step_results)
            avg_selected_pages_per_head = sum(r['layer_stats']['avg_selected_pages_per_head'] for r in step_results) / len(step_results)
            avg_hot_selected_pages_per_head = sum(r['layer_stats']['avg_hot_selected_pages_per_head'] for r in step_results) / len(step_results)
            avg_total_pages_per_head = sum(r['layer_stats']['avg_total_pages_per_head'] for r in step_results) / len(step_results)
            
            # Store step-level aggregated result
            step_aggregated_result = {
                'step_name': f"{args.generate_name}_step{step_idx}",
                'step_idx': step_idx,
                'num_layers_processed': len(step_results),
                'total_hot_clusters': total_hot_clusters,
                'total_selected_clusters': total_selected_clusters,
                'total_hot_selected': total_hot_selected,
                'avg_hot_hit_ratio': avg_hot_hit_ratio,
                'total_hot_pages': total_hot_pages,
                'total_selected_pages': total_selected_pages,
                'total_hot_selected_pages': total_hot_selected_pages,
                'total_all_pages': total_all_pages,
                'avg_hot_page_hit_ratio': avg_hot_page_hit_ratio,
                'avg_hot_pages_per_head': avg_hot_pages_per_head,
                'avg_selected_pages_per_head': avg_selected_pages_per_head,
                'avg_hot_selected_pages_per_head': avg_hot_selected_pages_per_head,
                'avg_total_pages_per_head': avg_total_pages_per_head,
            }
            
            step_aggregated_results.append(step_aggregated_result)
            
            print(f"\nStep {step_idx} Summary:")
            print(f"  Average hot cluster hit ratio: {avg_hot_hit_ratio:.3f}")
            print(f"  Average hot page hit ratio: {avg_hot_page_hit_ratio:.3f}")
            print(f"  Average hot pages per head: {avg_hot_pages_per_head:.1f}")
            print(f"  Average total pages per head: {avg_total_pages_per_head:.1f}")
    
    # Save results to CSV - only step-level aggregated results
    if args.csv_path:
        csv_path = args.csv_path + ".csv"
    else:
        # Use hot cluster count for filename
        csv_path = f"hot_cluster_hit_ratio_{args.model_name}_{args.dataset}_budget{args.budget_ratio}_dram{args.dram_capacity_gb}GB_batch{args.batch_size}.csv"

    print(f"\nSaving step-level results to: {csv_path}")
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(csv_path)
    
    # Define fieldnames
    fieldnames = [
        'step_name', 'step_idx', 'num_layers_processed', 'dram_page_count', 'dram_capacity_gb', 'budget_ratio', 'batch_size',
        'total_hot_clusters', 'total_selected_clusters', 'total_hot_selected',
        'avg_hot_hit_ratio', 'total_hot_pages', 'total_selected_pages', 
        'total_hot_selected_pages', 'total_all_pages', 'avg_hot_page_hit_ratio',
        'avg_hot_pages_per_head', 'avg_selected_pages_per_head', 
        'avg_hot_selected_pages_per_head', 'avg_total_pages_per_head'
    ]
    
    # Open file in append mode
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header only if file didn't exist before
        if not file_exists:
            writer.writeheader()
        
        for step_result in step_aggregated_results:
            row = {
                'dram_page_count': dram_page_count,
                'dram_capacity_gb': args.dram_capacity_gb,
                'budget_ratio': args.budget_ratio,
                'batch_size': args.batch_size,
                **step_result
            }
            writer.writerow(row)
    
    # Print overall summary
    if step_aggregated_results:
        overall_avg_hot_hit_ratio = sum(r['avg_hot_hit_ratio'] for r in step_aggregated_results) / len(step_aggregated_results)
        overall_avg_hot_page_hit_ratio = sum(r['avg_hot_page_hit_ratio'] for r in step_aggregated_results) / len(step_aggregated_results)
        overall_avg_hot_pages_per_head = sum(r['avg_hot_pages_per_head'] for r in step_aggregated_results) / len(step_aggregated_results)
        overall_avg_total_pages_per_head = sum(r['avg_total_pages_per_head'] for r in step_aggregated_results) / len(step_aggregated_results)
        
        print(f"\n=== OVERALL SUMMARY ===")
        print(f"DRAM capacity: {args.dram_capacity_gb} GB")
        print(f"Calculated DRAM page count for hot clusters: {dram_page_count}")
        print(f"Budget ratio: {args.budget_ratio}")
        print(f"Processed {len(step_aggregated_results)} steps")
        print(f"Overall average hot cluster hit ratio: {overall_avg_hot_hit_ratio:.3f}")
        print(f"Overall average hot page hit ratio: {overall_avg_hot_page_hit_ratio:.3f}")
        print(f"Overall average hot pages per head: {overall_avg_hot_pages_per_head:.1f}")
        print(f"Overall average total pages per head: {overall_avg_total_pages_per_head:.1f}")


if __name__ == '__main__':
    main()
