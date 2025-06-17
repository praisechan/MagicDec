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
    parser.add_argument('--window_size', type=int, default=16, required=True,
                        help='observation window size')
    parser.add_argument('--num_replica', type=int, default=4, required=True,
                        help='observation window size')
    parser.add_argument('--head_dim', type=int, default=128,
                        help='Dimension per KV head')
    parser.add_argument('--hot_cluster_duplicate', action='store_true',
                        help='Duplicate hot cluster and reduce load imbalance by balance_values')
    parser.add_argument('--hotness_aware_layout', action='store_true',
                        help='Sort in hotness order and layout')
    parser.add_argument('--constrained', action='store_true',
                        help='Constrain cluster size to certain value')
    
    # simulation config
    parser.add_argument('--max_latency_calculate', action='store_true',
                    help='whether accumulate pages_per_plane value for each layer or not')

    return parser.parse_args()


def load_profiling_layer(
    profiling_dir: str,
    layer_idx: int,
    num_heads: int,
    hot_cluster_duplicate,
    hot_cluster_ratio,
    window_size
) -> LayerData:
    # unchanged loader for .pt files
    cluster_sizes = torch.load(
        os.path.join(profiling_dir, f"cluster_size_{layer_idx}.pt"), map_location='cpu'
    )
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
        os.path.join(profiling_dir, f"selected_cI_step0_layer{layer_idx}.pt"), map_location='cpu'
    )
    softmax_sum = torch.load(
        os.path.join(profiling_dir, f"softmax_sum_{layer_idx}.pt"), map_location='cpu'
    )

    if hot_cluster_duplicate:
        if window_size==16:
            hot_cluster_list = torch.load(
                os.path.join(profiling_dir, f"hot_cluster_{hot_cluster_ratio}_{layer_idx}.pt"), map_location='cpu'
            )
        else:
            hot_cluster_list = torch.load(
                os.path.join(profiling_dir, f"hot_cluster_window{window_size}_{hot_cluster_ratio}_{layer_idx}.pt"), map_location='cpu'
            )
    else:    
        hot_cluster_list = None
    
    heads: List[HeadData] = []
    for head_idx in range(num_heads):
        clusters = [ClusterData(cid, int(size.item()))
                    for cid, size in enumerate(cluster_sizes[head_idx])]
        # superclusters = [SuperclusterData(
        #                      sc_id,
        #                      [int(cid.item()) for cid in ids],
        #                      supercluster_size[head_idx][sc_id]
        #                  ) for sc_id, ids in enumerate(superclusters_list[head_idx])]
        superclusters=None
        selected = [int(cid.item()) for cid in selected_list[head_idx]]
        hot_cluster = [int(cid.item()) for cid in hot_cluster_list[head_idx]] if hot_cluster_duplicate else None
        
        heads.append(HeadData(head_idx, clusters, superclusters, selected, hot_cluster, softmax_sum[head_idx]))
    return LayerData(layer_idx, heads)


def build_chips(args, layer: LayerData, mode: str) -> List[Chip]:
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

def get_plane_reads_per_layer(layer: LayerData, args, mode: str) -> List[int]:
    chips = build_chips(args, layer, mode)
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


def get_plane_reads_per_head(layer: LayerData, args, mode: str) -> List[List[int]]:
    chips = build_chips(args, layer, mode)
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

    return reads_per_head, total_latency_per_layer, ideal_total_latency_per_layer


def main():
    args = parse_args() 
    if args.num_channels != 1 or args.chips_per_channel != 1 or args.dies_per_chip != 1:
      raise ValueError("Hot cluster calculation only support single chip distribution yet")
    if args.hot_cluster_duplicate and args.num_replica is None:
      raise ValueError("num_replica should be given")
  
    layers_to_plot = range(32)
    # layers_to_plot = [0, 5, 10, 15, 20]
    # layers_to_plot = [0]
    # modes = ['baseline', 'supercluster']
    modes = ['baseline']
    if 'supercluster' in modes:
      raise ValueError("supercluster does not support hot cluster mode yet")
    
    if args.max_latency_calculate:
      total_latency = 0
      overlap_total_latency = 0
      ideal_total_latency =0 
      for layer_idx in tqdm(layers_to_plot):
          layer = load_profiling_layer(args.profiling_dir, layer_idx, args.num_heads, args.hot_cluster_duplicate, args.hot_cluster_ratio, args.window_size)
          for mode in modes:
              plane_reads_overlap, overlap_latency_per_layer, overlap_ideal_latency_per_layer = get_plane_reads_per_layer(layer, args, mode)
              reads_per_head, latency_per_layer, ideal_latency_per_layer = get_plane_reads_per_head(layer, args, mode)
              total_latency += latency_per_layer
              overlap_total_latency += overlap_latency_per_layer
              ideal_total_latency +=ideal_latency_per_layer
              # for each head, compute imbalance = max(reads) - min(reads)
              # for head_reads in reads_per_head:
              #     imbalance = max(head_reads) - min(head_reads)
              #     max_min_values.append(imbalance)    
      print(f"total latency: {total_latency}")
      print(f"total latency(head overlap): {overlap_total_latency}")
      print(f"ideal total latency: {ideal_total_latency}")
      
      import re
      prefix_len = re.search(r'KV_(\d+)', args.profiling_dir).group(1)
      budget_ratio = re.search(r'_(\d+\.\d+)KV_', args.profiling_dir).group(1)
      CSV_PATH = f"/home/juchanlee/MagicDec/SSD_simulator/output/latency_hotness_aware_layout_num_replica.csv"
      # if the file doesn't yet exist, write the header
      if not os.path.exists(CSV_PATH):
          with open(CSV_PATH, "w", newline="") as f:
              writer = csv.writer(f)
              writer.writerow(["prefix_len", "plane_num", "budget_ratio", "cluster_size", "window_size", "num_replica", "hot_cluster_ratio", "hot_cluster_duplication", "hotness_aware_layout", "total latency", "ideal latency"])

      # append to CSV
      with open(CSV_PATH, "a", newline="") as f:
          writer = csv.writer(f)
          writer.writerow([
              prefix_len,
              args.planes_per_die,
              budget_ratio,
              args.cluster_size,
              args.window_size,
              args.num_replica,
              args.hot_cluster_ratio,
              args.hot_cluster_duplicate,
              args.hotness_aware_layout,
              total_latency,
              ideal_total_latency,
          ])      
      
    else:          
      data_per_head = []
      data_per_layer = []
      labels_per_head = []
      labels_per_layer = []
  
      for layer_idx in layers_to_plot:
          layer = load_profiling_layer(args.profiling_dir, layer_idx, args.num_heads, args.hot_cluster_duplicate, args.hot_cluster_ratio)
          for mode in modes:
              import numpy as np
              page_reads, _, _= get_plane_reads_per_head(layer, args, mode)
              reads_per_head = np.array(page_reads).flatten().tolist()
              data_per_head.append(reads_per_head)
              labels_per_head.append(f"L{layer_idx}-{mode}")

              reads_per_layer = get_plane_reads_per_layer(layer, args, mode)
              data_per_layer.append(reads_per_layer)
              labels_per_layer.append(f"L{layer_idx}-{mode}")

      # violin plot
      # plt.figure(figsize=(6, 6))
      plt.figure(figsize=(12, 6))
      plt.violinplot(data_per_head, showmeans=True)
      plt.xticks(range(1, len(labels_per_head) + 1), labels_per_head, fontsize=14)
      plt.xlabel(f'CWDP {args.num_channels}-{args.chips_per_channel}-{args.dies_per_chip}-{args.planes_per_die}', fontsize=14)
      plt.ylabel('Total page reads per plane', fontsize=14)
      plt.title('Page Reads per Plane (Head 0~7 in Layer 0, 10, 20)', fontsize=14)
      plt.tight_layout()
      violin_filename = f'cluster_{args.cluster_size}_allhead_CWDP{args.num_channels}-{args.chips_per_channel}-{args.dies_per_chip}-{args.planes_per_die}.png'
      plt.savefig(violin_filename)
      plt.close()

      # plt.figure(figsize=(6, 6))
      plt.figure(figsize=(12, 6))
      plt.violinplot(data_per_layer, showmeans=True)
      plt.xticks(range(1, len(labels_per_layer) + 1), labels_per_layer, fontsize=14)
      plt.xlabel(f'CWDP {args.num_channels}-{args.chips_per_channel}-{args.dies_per_chip}-{args.planes_per_die}', fontsize=14)
      plt.ylabel('Total page reads per plane', fontsize=14)
      plt.title('Page Reads per Plane (Layer 0, 10, 20)', fontsize=14)
      plt.tight_layout()
      violin_filename = f'cluster_{args.cluster_size}_layer_CWDP{args.num_channels}-{args.chips_per_channel}-{args.dies_per_chip}-{args.planes_per_die}.png'
      plt.savefig(violin_filename)
      plt.close()

if __name__ == '__main__':
    main()
