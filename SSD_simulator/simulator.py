#!/usr/bin/env python
import argparse
import os
import math
import torch
from typing import List
from SSD_internal import (
    Plane,
    Chip,
    LayerData,
    HeadData,
    ClusterData,
    SuperclusterData
)
from utils import (
    compute_pages_per_cluster,
    balance_values
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os, csv

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and cluster modes for layer 0"
    )
    parser.add_argument('--num_channels', type=int, required=True)
    parser.add_argument('--chips_per_channel', type=int, required=True)
    parser.add_argument('--dies_per_chip', type=int, required=True)
    parser.add_argument('--planes_per_die', type=int, required=True)
    parser.add_argument('--page_size_bytes', type=int, required=True)
    parser.add_argument('--vector_bytes', type=int, default=2)
    parser.add_argument('--flash_read_latency_us', type=int, default=30)
    parser.add_argument('--profiling_dir', type=str, required=True,
                        help='Directory with layer-wise profiling .pt files')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='KV heads per layer')
    parser.add_argument('--cluster_size', type=int, default=16, required=True,
                        help='cluster size')
    parser.add_argument('--hot_cluster_ratio', type=float, default=0.01, required=True,
                        help='hot cluster ratio')
    parser.add_argument('--budget_ratio', type=str, default=0.25, required=True,
                        help='budget ratio')
    parser.add_argument('--prefix_len', type=str, default=16385, required=True)
    parser.add_argument('--window_size', type=int, default=16, required=True,
                        help='observation window size')
    parser.add_argument('--num_replica', type=int, default=4, required=True,
                        help='observation window size')
    parser.add_argument('--head_dim', type=int, default=128,
                        help='Dimension per KV head')
    parser.add_argument('--csv_path', type=str,
                        help='csv path')
    parser.add_argument('--hot_cluster_duplicate', action='store_true',
                        help='Duplicate hot cluster and reduce load imbalance by balance_values')
    parser.add_argument('--hotness_aware_layout', action='store_true',
                        help='Sort in hotness order and layout')
    parser.add_argument('--constrained', action='store_true',
                        help='Constrain cluster size to certain value')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name for profiling directory')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name for profiling directory')
    parser.add_argument('--generate_name', type=str, required=True,
                        help='Generate name (e.g., speculate_0_0, verify_1_2)')
    parser.add_argument('--step_idx', type=int, default=0,
                        help='Step index for multi-step data')
    parser.add_argument('--layer_num', type=int, default=32,
                        help='Number of layers to process')
    
    # simulation config
    parser.add_argument('--max_latency_calculate', action='store_true',
                    help='whether accumulate pages_per_plane value for each layer or not')

    return parser.parse_args()


def load_profiling_layer(
    args,
    layer_idx: int,
    generate_name: str = None,
    step_idx: int = 0
) -> LayerData:
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
            profiling_dir = args.profiling_dir + f"/{args.model_name}_{args.dataset}_{args.prefix_len}/{generate_name}/data_{args.budget_ratio}KV_clustersize_{cluster_size_attempt}"
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
    
    # unchanged loader for .pt files
    # superclusters_list = torch.load(
    #     os.path.join(profiling_dir, f"superclusters_{layer_idx}.pt"), map_location='cpu'
    # )
    # supercluster_size = torch.load(
    #     os.path.join(profiling_dir, f"supercluster_size_{layer_idx}.pt"), map_location='cpu'
    # )
    # selected_list = torch.load(
    #     os.path.join(profiling_dir, f"cI_of_selected_superclusters_{layer_idx}.pt"), map_location='cpu'
    # )
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
    # profiling_dir_for_softmax = args.profiling_dir + f"{args.model_name}_{args.dataset}_{args.prefix_len}/speculate_{train_step_idx}_0/data_{args.budget_ratio}KV_clustersize_{cluster_size_attempt}"
    # softmax_sum = torch.load(
    #     os.path.join(profiling_dir_for_softmax, f"softmax_sum_{layer_idx}.pt"), map_location='cpu'
    # )
    softmax_sum = torch.load(
        os.path.join(profiling_dir, f"softmax_sum_{layer_idx}.pt"), map_location='cpu'
    )

    # Calculate hot clusters based on softmax_sum scores
    hot_cluster_list = None
    if args.hot_cluster_duplicate:
        hot_cluster_list = []
        for head_idx in range(args.num_heads):
            head_softmax_scores = softmax_sum[head_idx]
            num_clusters = len(head_softmax_scores)
            num_hot_clusters = max(1, int(num_clusters * args.hot_cluster_ratio))
            
            # Get indices of top hot clusters based on softmax scores
            _, hot_cluster_indices = torch.topk(head_softmax_scores, num_hot_clusters, largest=True)
            hot_cluster_list.append(hot_cluster_indices.tolist())
        
        # Convert to tensor format similar to the original hot_cluster_list
        hot_cluster_list = torch.tensor(hot_cluster_list)

    heads: List[HeadData] = []
    for head_idx in range(args.num_heads):
        clusters = [ClusterData(cid, int(size.item()))
                    for cid, size in enumerate(cluster_sizes[head_idx])]
        # superclusters = [SuperclusterData(
        #                      sc_id,
        #                      [int(cid.item()) for cid in ids],
        #                      supercluster_size[head_idx][sc_id]
        #                  ) for sc_id, ids in enumerate(superclusters_list[head_idx])]
        superclusters=None
        selected = [int(cid.item()) for cid in selected_list[head_idx]]
        # hot_cluster = [int(cid.item()) for cid in hot_cluster_list[head_idx]] if args.hot_cluster_duplicate else None
        hot_cluster = hot_cluster_list[head_idx].tolist() if args.hot_cluster_duplicate else None
        
        heads.append(HeadData(head_idx, clusters, superclusters, selected, hot_cluster, softmax_sum[head_idx]))
    return LayerData(layer_idx, heads)


def build_chips(args, layer: LayerData, mode: str, pages_per_head_data: List = None) -> List[Chip]:
    chips: List[Chip] = []
    for channel_id in range(args.num_channels):
        for chip_id in range(args.chips_per_channel):
            chip = Chip(
                channel_id=channel_id,
                chip_id=chip_id,
                dies_per_chip=args.dies_per_chip,
                planes_per_die=args.planes_per_die,
                chips_per_channel=args.chips_per_channel,
                num_channels=args.num_channels,
                hotness_aware_layout = args.hotness_aware_layout,
                hot_cluster_duplicate = args.hot_cluster_duplicate
            )
            for head in layer.heads:
                pages_map = compute_pages_per_cluster(
                    head,
                    args.page_size_bytes,
                    args.vector_bytes,
                    args.head_dim,
                    args.constrained,
                    args.cluster_size
                )
                
                # Calculate and print pages per head
                pages_per_head = sum(pages_map.values())
                # print(f"Layer {layer.layer_index}, Head {head.head_index}: {pages_per_head} pages")
                
                # Collect pages per head data if list is provided
                if pages_per_head_data is not None:
                    pages_per_head_data.append(pages_per_head)
                
                chip.layout_clusters(
                    head_idx=head.head_index,
                    pages_per_cluster=pages_map,
                    superclusters=head.superclusters,
                    hot_cluster_ids=head.hot_cluster_ids,
                    softmax_sum=head.softmax_sum,
                    mode=mode,
                    num_replica=args.num_replica
                )
            chips.append(chip)
    return chips

def get_plane_reads_per_layer(layer: LayerData, args, mode: str, pages_per_head_data: List = None) -> List[int]:
    chips = build_chips(args, layer, mode, pages_per_head_data)
    total_planes = sum(len(chip.planes) for chip in chips)

    # plane_reads = [0] * total_planes
    # # count per head, per plane
    # for head in layer.heads:
    #     for chip in chips:
    #         for plane in chip.planes:
    #             reads,_ = plane.simulate_access(
    #                 head.head_index,
    #                 head.selected_cluster_ids,
    #                 hot_cluster_ids=head.hot_cluster_ids,
    #                 mode=mode
    #             )
    #             plane_reads[plane.global_plane_id] += reads

    # plane_reads = [0] * total_planes

    reads_per_head = []
    for head in layer.heads:
        for chip in chips:
            plane_reads = chip.simulate_access(
                    head.head_index,
                    head.selected_cluster_ids,
                    hot_cluster_ids=head.hot_cluster_ids,
                    mode=mode
            )
        reads_per_head.append(plane_reads)

    # calculate max "page read per plane" for each head
    reads_per_layer = [sum(col) for col in zip(*reads_per_head)]
    total_latency = max(reads_per_layer)
    ideal_total_latency = sum(reads_per_layer) / len(reads_per_layer)

    return reads_per_layer, total_latency, ideal_total_latency


def get_plane_reads_per_head(layer: LayerData, args, mode: str, pages_per_head_data: List = None) -> List[List[int]]:
    chips = build_chips(args, layer, mode, pages_per_head_data)
    reads_per_head = []

    for head in layer.heads:
        for chip in chips:
            plane_reads = chip.simulate_access(
                    head.head_index,
                    head.selected_cluster_ids,
                    hot_cluster_ids=head.hot_cluster_ids,
                    mode=mode
            )

        reads_per_head.append(plane_reads)        
        
    # calculate max "page read per plane" for each head
    total_latency_per_layer=0
    ideal_total_latency_per_layer=0
    for i in range(len(reads_per_head)):
      total_latency_per_layer += max(reads_per_head[i])
      ideal_total_latency_per_layer += sum(reads_per_head[i]) / len(reads_per_head[i])

    max_head_latency = max(max(reads) for reads in reads_per_head) # get max head latency among all heads

    return reads_per_head, total_latency_per_layer, ideal_total_latency_per_layer, max_head_latency


def main():
    args = parse_args() 
    if args.num_channels != 1 or args.chips_per_channel != 1 or args.dies_per_chip != 1:
      raise ValueError("Hot cluster calculation only support single chip distribution yet")
    if args.hot_cluster_duplicate and args.num_replica is None:
      raise ValueError("num_replica should be given")
  
    layers_to_plot = range(args.layer_num)
    # layers_to_plot = [0, 5, 10, 15, 20]
    # layers_to_plot = [0]
    # modes = ['baseline', 'supercluster']
    modes = ['baseline']
    if 'supercluster' in modes:
      raise ValueError("supercluster does not support hot cluster mode yet")
    
    # List to collect all pages per head data
    all_pages_per_head_data = []
    
    # List to collect all reads per head values for histogram
    all_reads_per_head_values = []
    
    # List to collect raw data per head for box plots
    reads_per_head_raw_data = {}  # Structure: {(layer_idx, step_idx, head_idx): [reads_per_plane_list]}
    
    if args.max_latency_calculate:
      # Determine the number of steps based on the generate_name pattern
      if "speculate" in args.generate_name:
          max_steps = 4  # speculate has steps 0, 1, 2, 3
      else:  # verify
          max_steps = 1  # verify typically has only one step (step 0)
      
      total_latency = 0
      overlap_total_latency = 0
      ideal_total_latency = 0
      
      # Process each step
      for step_idx in range(max_steps):
          step_total_latency = 0
          step_overlap_total_latency = 0
          step_ideal_total_latency = 0
          max_head_latencies = []  # Collect max_head_latency for each layer
          
          for layer_idx in tqdm(layers_to_plot, desc=f"Processing step {step_idx}"):
              layer = load_profiling_layer(args, layer_idx, args.generate_name, step_idx)
              mode = 'baseline'  # only baseline mode for max latency calculation
              plane_reads_overlap, overlap_latency_per_layer, overlap_ideal_latency_per_layer = get_plane_reads_per_layer(layer, args, mode, all_pages_per_head_data)
              reads_per_head, latency_per_layer, ideal_latency_per_layer, max_head_latency = get_plane_reads_per_head(layer, args, mode, all_pages_per_head_data)
              
              # Collect all reads per head values for histogram
              for head_reads in reads_per_head:
                  all_reads_per_head_values.extend(head_reads)
              
              # Collect raw data per head for box plots
              for head_idx, head_reads in enumerate(reads_per_head):
                  key = (layer_idx, step_idx, head_idx)
                  reads_per_head_raw_data[key] = head_reads.copy()
                  
              step_total_latency += latency_per_layer
              step_overlap_total_latency += overlap_latency_per_layer
              step_ideal_total_latency += ideal_latency_per_layer
              max_head_latencies.append(max_head_latency)
          
          # Sum all max_head_latencies
          total_max_head_latency = sum(max_head_latencies)
          
          total_latency += step_total_latency
          overlap_total_latency += step_overlap_total_latency
          ideal_total_latency += step_ideal_total_latency
          
          print(f"Step {step_idx} - total latency: {step_total_latency}")
          print(f"Step {step_idx} - total latency(head overlap): {step_overlap_total_latency}")
          print(f"Step {step_idx} - ideal total latency: {step_ideal_total_latency}")
          print(f"Step {step_idx} - total max head latency: {total_max_head_latency}")

          # Write step-specific results to CSV
          import re

          if args.csv_path:
              CSV_PATH = args.csv_path + ".csv"
          else:
              CSV_PATH = f"/home/juchanlee/MagicDec/SSD_simulator/output/latency_budget{args.budget_ratio}_replica{args.num_replica}.csv"
          
          # if the file doesn't yet exist, write the header
          if not os.path.exists(CSV_PATH):
              with open(CSV_PATH, "w", newline="") as f:
                  writer = csv.writer(f)
                  # Create header with layer-specific max_head_latency columns
                  header = ["step_name", "prefix_len", "plane_num", "budget_ratio", "cluster_size", "window_size", "num_replica", "hot_cluster_ratio", "hot_cluster_duplication", "hotness_aware_layout", "step_total_latency", "step_overlap_latency", "step_ideal_latency"]
                  # Add columns for each layer's max_head_latency
                  for layer_idx in range(args.layer_num):
                      header.append(f"layer_{layer_idx}_max_head_latency")
                  header.append("total_max_head_latency")
                  writer.writerow(header)

          # append step results to CSV
          step_name = f"{args.generate_name}_step{step_idx}"
          with open(CSV_PATH, "a", newline="") as f:
              writer = csv.writer(f)
              row = [
                  step_name,
                  args.prefix_len,
                  args.planes_per_die,
                  args.budget_ratio,
                  args.cluster_size,
                  args.window_size,
                  args.num_replica,
                  args.hot_cluster_ratio,
                  args.hot_cluster_duplicate,
                  args.hotness_aware_layout,
                  step_total_latency,
                  step_overlap_total_latency,
                  step_ideal_total_latency,
              ]
              # Add each layer's max_head_latency
              row.extend(max_head_latencies)
              # Add total max_head_latency
              row.append(total_max_head_latency)
              writer.writerow(row)
      
      print(f"Total accumulated latency: {total_latency}")
      print(f"Total accumulated latency(head overlap): {overlap_total_latency}")
      print(f"Total accumulated ideal latency: {ideal_total_latency}")
      
      # Calculate and print average pages per head across all heads and all layers
      if all_pages_per_head_data:
          # Remove duplicates that might occur due to calling both get_plane_reads_per_layer and get_plane_reads_per_head
          # Since both functions call build_chips, we might have duplicates, so let's deduplicate by taking every other element
          unique_pages_data = all_pages_per_head_data[::2] if len(all_pages_per_head_data) > len(layers_to_plot) * args.num_heads * max_steps else all_pages_per_head_data
          
          total_pages = sum(unique_pages_data)
          average_pages_per_head = total_pages / len(unique_pages_data)
          
          print(f"\n=== PAGES PER HEAD SUMMARY ===")
          print(f"Total heads processed: {len(unique_pages_data)}")
          print(f"Total pages across all heads: {total_pages}")
          print(f"Average pages per head: {average_pages_per_head:.2f}")
          print(f"Min pages per head: {min(unique_pages_data)}")
          print(f"Max pages per head: {max(unique_pages_data)}")
      
      # Create histogram of reads per head
      if all_reads_per_head_values:
          print(f"\n=== READS PER HEAD HISTOGRAM ===")
          print(f"Total number of plane reads collected: {len(all_reads_per_head_values)}")
          print(f"Min reads per plane: {min(all_reads_per_head_values)}")
          print(f"Max reads per plane: {max(all_reads_per_head_values)}")
          print(f"Average reads per plane: {np.mean(all_reads_per_head_values):.2f}")
          
          # Create histogram
          plt.figure(figsize=(12, 8))
          
          # Use integer bins since read counts are discrete integers
          min_reads = min(all_reads_per_head_values)
          max_reads = max(all_reads_per_head_values)
          # Create bins for each integer value from min to max
          bins = list(range(min_reads, max_reads + 2))  # +2 to include max_reads in a bin
          
          counts, bin_edges, patches = plt.hist(all_reads_per_head_values, bins=bins, edgecolor='black', alpha=0.7)
          
          plt.xlabel('Number of Reads per Plane', fontsize=14)
          plt.ylabel('Number of Planes', fontsize=14)
          plt.title('Histogram of Reads per Head across All Layers and Steps', fontsize=16)
          plt.grid(True, alpha=0.3)
          
          # Add some statistics as text on the plot
          plt.axvline(np.mean(all_reads_per_head_values), color='red', linestyle='--', 
                     label=f'Mean: {np.mean(all_reads_per_head_values):.1f}')
          plt.axvline(np.median(all_reads_per_head_values), color='orange', linestyle='--', 
                     label=f'Median: {np.median(all_reads_per_head_values):.1f}')
          plt.legend()
          
          # Save the histogram
          histogram_filename = f'reads_per_head_{args.model_name}_{args.prefix_len}_histogram_budget{args.budget_ratio}_replica{args.num_replica}.png'
          plt.tight_layout()
          plt.savefig(histogram_filename, dpi=300, bbox_inches='tight')
          print(f"Histogram saved as: {histogram_filename}")
          plt.show()
          plt.close()
          
          # Save histogram data as CSV with integer bins
          csv_filename = f'reads_per_head_{args.model_name}_{args.prefix_len}_histogram_data_budget{args.budget_ratio}_replica{args.num_replica}.csv'

          # Count occurrences of each integer read value
          from collections import Counter
          read_counts = Counter(all_reads_per_head_values)
          
          # Prepare CSV data with integer read counts
          with open(csv_filename, 'w', newline='') as csvfile:
              csv_writer = csv.writer(csvfile)
              # Write header
              csv_writer.writerow(['reads_count', 'num_planes', 'percentage'])
              
              total_count = len(all_reads_per_head_values)
              # Write data for each read count from min to max
              for read_count in range(min_reads, max_reads + 1):
                  num_planes = read_counts.get(read_count, 0)  # 0 if this read count doesn't exist
                  percentage = (num_planes / total_count) * 100
                  csv_writer.writerow([read_count, num_planes, f'{percentage:.2f}'])
          
          # Also save raw data as CSV
          raw_data_filename = f'reads_per_head_raw_data_{args.model_name}_{args.prefix_len}_budget{args.budget_ratio}_replica{args.num_replica}.csv'
          with open(raw_data_filename, 'w', newline='') as csvfile:
              csv_writer = csv.writer(csvfile)
              csv_writer.writerow(['plane_index', 'reads_count'])
              for i, reads in enumerate(all_reads_per_head_values):
                  csv_writer.writerow([i, reads])
          
          print(f"Histogram CSV data saved as: {csv_filename}")
          print(f"Raw reads data CSV saved as: {raw_data_filename}")
          
          # Print CSV summary
          print(f"\n=== CSV DATA SUMMARY ===")
          print(f"Read count range: {min_reads} to {max_reads}")
          print(f"Total unique read counts: {len(read_counts)}")
          print(f"Raw data points: {len(all_reads_per_head_values)}")
          most_common_read_count, most_common_count = read_counts.most_common(1)[0]
          print(f"Most common read count: {most_common_read_count} reads ({most_common_count} planes, {(most_common_count/total_count)*100:.1f}%)")
      
      # Save raw data per head as CSV and create box plots
      if reads_per_head_raw_data:
          print(f"\n=== READS PER HEAD RAW DATA ===")
          
          # Save raw data per head to CSV
          raw_heads_csv_filename = f'reads_per_head_by_head_{args.model_name}_{args.prefix_len}_budget{args.budget_ratio}_replica{args.num_replica}.csv'
          with open(raw_heads_csv_filename, 'w', newline='') as csvfile:
              csv_writer = csv.writer(csvfile)
              # Write header
              csv_writer.writerow(['layer_idx', 'step_idx', 'head_idx', 'plane_idx', 'reads_count'])
              
              # Write data for each head
              for (layer_idx, step_idx, head_idx), reads_list in reads_per_head_raw_data.items():
                  for plane_idx, reads_count in enumerate(reads_list):
                      csv_writer.writerow([layer_idx, step_idx, head_idx, plane_idx, reads_count])
          
          print(f"Raw heads data CSV saved as: {raw_heads_csv_filename}")
          
          # Create box plots for head 0 of layers 8, 16, 24, 32
          target_layers = [8, 16, 24, 32]  # Note: layer indices are 0-based, so layer 8 is actually layer_idx 7
          target_layers_0_based = [l-1 for l in target_layers if l-1 < args.layer_num]  # Convert to 0-based and filter
          target_head = 0
          
          # Collect data for box plots
          box_plot_data = []
          box_plot_labels = []
          
          for layer_idx in target_layers_0_based:
              layer_data = []
              # Collect data from all steps for this layer and head
              for step_idx in range(max_steps):
                  key = (layer_idx, step_idx, target_head)
                  if key in reads_per_head_raw_data:
                      layer_data.extend(reads_per_head_raw_data[key])
              
              if layer_data:  # Only add if we have data
                  box_plot_data.append(layer_data)
                  box_plot_labels.append(f'Layer {layer_idx + 1}')  # Convert back to 1-based for labels
                  print(f"Layer {layer_idx + 1}, Head {target_head}: {len(layer_data)} planes, "
                        f"range [{min(layer_data)}-{max(layer_data)}], mean {np.mean(layer_data):.1f}")
          
          if box_plot_data:
              # Create box plot
              plt.figure(figsize=(12, 8))
              
              box_plot = plt.boxplot(box_plot_data, labels=box_plot_labels, patch_artist=True)
              
              # Customize colors
              colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
              for patch, color in zip(box_plot['boxes'], colors[:len(box_plot_data)]):
                  patch.set_facecolor(color)
                  patch.set_alpha(0.7)
              
              plt.xlabel('Layer', fontsize=14)
              plt.ylabel('Number of Reads per Plane', fontsize=14)
              plt.title(f'Box Plot of Reads per Plane for Head {target_head} across Selected Layers', fontsize=16)
              plt.grid(True, alpha=0.3, axis='y')
              
              # Add some statistics text
              plt.figtext(0.02, 0.02, f'Budget ratio: {args.budget_ratio}, Replica: {args.num_replica}', 
                         fontsize=10, ha='left')
              
              # Save box plot
              box_plot_filename = f'reads_per_head_boxplot_head{target_head}_{args.model_name}_{args.prefix_len}_budget{args.budget_ratio}_replica{args.num_replica}.png'
              plt.tight_layout()
              plt.savefig(box_plot_filename, dpi=300, bbox_inches='tight')
              print(f"Box plot saved as: {box_plot_filename}")
              plt.show()
              plt.close()
              
              # Save box plot data summary as CSV
              box_plot_summary_csv = f'reads_per_head_boxplot_summary_head{target_head}_budget{args.budget_ratio}_replica{args.num_replica}.csv'
              with open(box_plot_summary_csv, 'w', newline='') as csvfile:
                  csv_writer = csv.writer(csvfile)
                  csv_writer.writerow(['layer', 'num_planes', 'min_reads', 'q1_reads', 'median_reads', 
                                     'q3_reads', 'max_reads', 'mean_reads', 'std_reads'])
                  
                  for i, (layer_label, data) in enumerate(zip(box_plot_labels, box_plot_data)):
                      q1, median, q3 = np.percentile(data, [25, 50, 75])
                      csv_writer.writerow([
                          layer_label,
                          len(data),
                          min(data),
                          f'{q1:.2f}',
                          f'{median:.2f}',
                          f'{q3:.2f}',
                          max(data),
                          f'{np.mean(data):.2f}',
                          f'{np.std(data):.2f}'
                      ])
              
              print(f"Box plot summary CSV saved as: {box_plot_summary_csv}")
              
              print(f"\n=== BOX PLOT SUMMARY ===")
              print(f"Target layers: {target_layers} (requested)")
              print(f"Available layers: {[l+1 for l in target_layers_0_based]} (0-based: {target_layers_0_based})")
              print(f"Target head: {target_head}")
              print(f"Total data points plotted: {sum(len(d) for d in box_plot_data)}")
          else:
              print(f"No data found for head {target_head} in target layers {target_layers}")
              print(f"Available layers: {sorted(set(k[0] for k in reads_per_head_raw_data.keys()))}")
              print(f"Available heads: {sorted(set(k[2] for k in reads_per_head_raw_data.keys()))}")
      
    # else:          
    #   data_per_head = []
    #   data_per_layer = []
    #   labels_per_head = []
    #   labels_per_layer = []
  
    #   for layer_idx in layers_to_plot:
    #       layer = load_profiling_layer(args, layer_idx)
    #       for mode in modes:
    #           import numpy as np
    #           page_reads, _, _= get_plane_reads_per_head(layer, args, mode)
    #           reads_per_head = np.array(page_reads).flatten().tolist()
    #           data_per_head.append(reads_per_head)
    #           labels_per_head.append(f"L{layer_idx}-{mode}")

    #           reads_per_layer = get_plane_reads_per_layer(layer, args, mode)
    #           data_per_layer.append(reads_per_layer)
    #           labels_per_layer.append(f"L{layer_idx}-{mode}")

    #   # violin plot
    #   # plt.figure(figsize=(6, 6))
    #   plt.figure(figsize=(12, 6))
    #   plt.violinplot(data_per_head, showmeans=True)
    #   plt.xticks(range(1, len(labels_per_head) + 1), labels_per_head, fontsize=14)
    #   plt.xlabel(f'CWDP {args.num_channels}-{args.chips_per_channel}-{args.dies_per_chip}-{args.planes_per_die}', fontsize=14)
    #   plt.ylabel('Total page reads per plane', fontsize=14)
    #   plt.title('Page Reads per Plane (Head 0~7 in Layer 0, 10, 20)', fontsize=14)
    #   plt.tight_layout()
    #   violin_filename = f'cluster_{args.cluster_size}_allhead_CWDP{args.num_channels}-{args.chips_per_channel}-{args.dies_per_chip}-{args.planes_per_die}.png'
    #   plt.savefig(violin_filename)
    #   plt.close()

    #   # plt.figure(figsize=(6, 6))
    #   plt.figure(figsize=(12, 6))
    #   plt.violinplot(data_per_layer, showmeans=True)
    #   plt.xticks(range(1, len(labels_per_layer) + 1), labels_per_layer, fontsize=14)
    #   plt.xlabel(f'CWDP {args.num_channels}-{args.chips_per_channel}-{args.dies_per_chip}-{args.planes_per_die}', fontsize=14)
    #   plt.ylabel('Total page reads per plane', fontsize=14)
    #   plt.title('Page Reads per Plane (Layer 0, 10, 20)', fontsize=14)
    #   plt.tight_layout()
    #   violin_filename = f'cluster_{args.cluster_size}_layer_CWDP{args.num_channels}-{args.chips_per_channel}-{args.dies_per_chip}-{args.planes_per_die}.png'
    #   plt.savefig(violin_filename)
    #   plt.close()

if __name__ == '__main__':
    main()
