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
    parser.add_argument('--max_concurrent_heads', type=int, default=None,
                    help='Maximum number of heads that can execute concurrently. If None, no constraint is applied.')

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


def get_constrained_overlap_latency(reads_per_head: List[List[int]], max_concurrent_heads: int, plane_reads_overlap: List[int]) -> tuple:
    """
    Calculate latency with constraint on number of concurrent heads.
    
    The key insight: each plane can only read one page at a time. When multiple heads
    overlap, their reads on different planes can execute concurrently. The constraint
    limits how many heads can have active plane reads at any given time.
    
    Args:
        reads_per_head: List of reads per plane for each head. 
                       reads_per_head[head_idx][plane_idx] = number of reads
        max_concurrent_heads: Maximum number of heads that can execute concurrently
        plane_reads_overlap: Total reads per plane across all heads (for reference)
    
    Returns:
        tuple: (total_latency, head_schedules)
            - total_latency: Total cycles needed with overlap constraint
            - head_schedules: List of (head_idx, start_cycle, end_cycle) tuples
    """
    num_heads = len(reads_per_head)
    num_planes = len(reads_per_head[0]) if num_heads > 0 else 0
    
    # Track per-plane state: when each plane will be free
    plane_free_time = [0] * num_planes
    
    # Track per-head state: (start_cycle, end_cycle) 
    head_schedules = []
    
    # Track which heads are currently active (have started but not finished)
    active_heads = set()  # Set of head indices
    
    # For each head, track remaining reads per plane
    remaining_reads = [[reads for reads in head_reads] for head_reads in reads_per_head]
    
    # Track when each head starts and finishes
    head_start_time = [None] * num_heads
    head_finish_time = [None] * num_heads
    
    current_cycle = 0
    heads_to_schedule = list(range(num_heads))
    
    # Simulate cycle by cycle
    while heads_to_schedule or active_heads:
        # Remove finished heads from active set
        finished_heads = [h for h in active_heads if all(remaining_reads[h][p] == 0 for p in range(num_planes))]
        for h in finished_heads:
            active_heads.remove(h)
            head_finish_time[h] = current_cycle
        
        # Try to start new heads if we have capacity
        while heads_to_schedule and len(active_heads) < max_concurrent_heads:
            new_head = heads_to_schedule.pop(0)
            active_heads.add(new_head)
            head_start_time[new_head] = current_cycle
        
        # For each active head, schedule reads on available planes
        if active_heads:
            for plane_idx in range(num_planes):
                if plane_free_time[plane_idx] <= current_cycle:
                    # Find an active head that still has reads on this plane
                    for head_idx in active_heads:
                        if remaining_reads[head_idx][plane_idx] > 0:
                            # Schedule one read for this head on this plane
                            remaining_reads[head_idx][plane_idx] -= 1
                            plane_free_time[plane_idx] = current_cycle + 1
                            break
        
        # Advance to next cycle
        current_cycle += 1
        
        # Safety check to prevent infinite loop
        if current_cycle > 1000000:
            raise RuntimeError("Simulation exceeded maximum cycles - possible infinite loop")
    
    # Build head schedules
    for head_idx in range(num_heads):
        head_schedules.append((head_idx, head_start_time[head_idx], head_finish_time[head_idx]))
    
    # Total latency is when the last head finishes
    total_latency = max(head_finish_time)
    
    return total_latency, head_schedules


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
      total_constrained_overlap_latency = 0  # New: accumulate constrained overlap latency
      
      # Process each step
      for step_idx in range(max_steps):
          step_total_latency = 0
          step_overlap_total_latency = 0
          step_ideal_total_latency = 0
          step_constrained_overlap_latency = 0  # New: constrained overlap latency
          max_head_latencies = []  # Collect max_head_latency for each layer
          
          for layer_idx in tqdm(layers_to_plot, desc=f"Processing step {step_idx}"):
              layer = load_profiling_layer(args, layer_idx, args.generate_name, step_idx)
              mode = 'baseline'  # only baseline mode for max latency calculation
              plane_reads_overlap, overlap_latency_per_layer, overlap_ideal_latency_per_layer = get_plane_reads_per_layer(layer, args, mode, all_pages_per_head_data)
              reads_per_head, latency_per_layer, ideal_latency_per_layer, max_head_latency = get_plane_reads_per_head(layer, args, mode, all_pages_per_head_data)
              
              # Calculate constrained overlap latency if max_concurrent_heads is specified
              if args.max_concurrent_heads is not None:
                  constrained_latency, _ = get_constrained_overlap_latency(reads_per_head, args.max_concurrent_heads, plane_reads_overlap)
                  step_constrained_overlap_latency += constrained_latency
              
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
          if args.max_concurrent_heads is not None:
              total_constrained_overlap_latency += step_constrained_overlap_latency
          
          print(f"Step {step_idx} - total latency: {step_total_latency}")
          print(f"Step {step_idx} - total latency(head overlap): {step_overlap_total_latency}")
          print(f"Step {step_idx} - ideal total latency: {step_ideal_total_latency}")
          if args.max_concurrent_heads is not None:
              print(f"Step {step_idx} - constrained overlap latency (max {args.max_concurrent_heads} concurrent heads): {step_constrained_overlap_latency}")
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
                  header = ["step_name", "prefix_len", "plane_num", "budget_ratio", "cluster_size", "window_size", "num_replica", "hot_cluster_ratio", "hot_cluster_duplication", "hotness_aware_layout", "max_concurrent_heads", "step_total_latency", "step_overlap_latency", "step_ideal_latency", "step_constrained_overlap_latency"]
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
                  args.max_concurrent_heads if args.max_concurrent_heads is not None else "N/A",
                  step_total_latency,
                  step_overlap_total_latency,
                  step_ideal_total_latency,
                  step_constrained_overlap_latency if args.max_concurrent_heads is not None else "N/A",
              ]
              # Add each layer's max_head_latency
              row.extend(max_head_latencies)
              # Add total max_head_latency
              row.append(total_max_head_latency)
              writer.writerow(row)
      
      print(f"Total accumulated latency: {total_latency}")
      print(f"Total accumulated latency(head overlap): {overlap_total_latency}")
      print(f"Total accumulated ideal latency: {ideal_total_latency}")
      if args.max_concurrent_heads is not None:
          print(f"Total accumulated constrained overlap latency (max {args.max_concurrent_heads} concurrent heads): {total_constrained_overlap_latency}")
      
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
