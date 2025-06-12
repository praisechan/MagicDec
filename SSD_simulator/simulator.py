#!/usr/bin/env python
import argparse
import os
import torch
from typing import List
from SSD_internal import (
    Plane,
    Chip,
    compute_pages_per_cluster,
    LayerData,
    HeadData,
    ClusterData,
    SuperclusterData
)
import matplotlib.pyplot as plt


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
    parser.add_argument('--head_dim', type=int, default=128,
                        help='Dimension per KV head')
    parser.add_argument('--constrained', action='store_true',
                        help='Constrain cluster size to certain value')
    
    # simulation config
    parser.add_argument('--max_latency_calculate', action='store_true',
                    help='whether accumulate pages_per_plane value for each layer or not')

    return parser.parse_args()


def load_profiling_layer(
    profiling_dir: str,
    layer_idx: int,
    num_heads: int
) -> LayerData:
    # unchanged loader for .pt files
    cluster_sizes = torch.load(
        os.path.join(profiling_dir, f"cluster_size_{layer_idx}.pt"), map_location='cpu'
    )
    superclusters_list = torch.load(
        os.path.join(profiling_dir, f"superclusters_{layer_idx}.pt"), map_location='cpu'
    )
    supercluster_size = torch.load(
        os.path.join(profiling_dir, f"supercluster_size_{layer_idx}.pt"), map_location='cpu'
    )
    # selected_list = torch.load(
    #     os.path.join(profiling_dir, f"cI_of_selected_superclusters_{layer_idx}.pt"), map_location='cpu'
    # )
    selected_list = torch.load(
        os.path.join(profiling_dir, f"selected_cI_{layer_idx}.pt"), map_location='cpu'
    )
    heads: List[HeadData] = []
    for head_idx in range(num_heads):
        clusters = [ClusterData(cid, int(size.item()))
                    for cid, size in enumerate(cluster_sizes[head_idx])]
        superclusters = [SuperclusterData(
                             sc_id,
                             [int(cid.item()) for cid in ids],
                             supercluster_size[head_idx][sc_id]
                         ) for sc_id, ids in enumerate(superclusters_list[head_idx])]
        selected = [int(cid.item()) for cid in selected_list[head_idx]]
        heads.append(HeadData(head_idx, clusters, superclusters, selected))
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
                num_channels=args.num_channels
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
                chip.assign_clusters(
                    head_idx=head.head_index,
                    pages_per_cluster=pages_map,
                    superclusters=head.superclusters,
                    mode=mode
                )
            chips.append(chip)
    return chips

def get_plane_reads(layer: LayerData, args, mode: str) -> List[int]:
    chips = build_chips(args, layer, mode)
    total_planes = sum(len(chip.planes) for chip in chips)
    plane_reads = [0] * total_planes
    # count per head, per plane
    for head in layer.heads:
        idx = 0
        for chip in chips:
            for plane in chip.planes:
                reads = plane.simulate_access(
                    head.head_index,
                    head.selected_cluster_ids,
                    mode
                )
                plane_reads[idx] += reads
                idx += 1
    return plane_reads

def get_plane_reads_per_head(layer: LayerData, args, mode: str) -> List[List[int]]:
    chips = build_chips(args, layer, mode)
    reads_per_head = []
    for head in layer.heads:
        plane_reads = []
        for chip in chips:
            for plane in chip.planes:
                r = plane.simulate_access(
                    head.head_index,
                    head.selected_cluster_ids,
                    mode
                )
                plane_reads.append(r)
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
    layers_to_plot = range(32)
    # layers_to_plot = [0, 5, 10, 15, 20]
    # layers_to_plot = [0]
    # modes = ['baseline', 'cluster']
    modes = ['baseline']
    # modes = ['cluster']
    if args.max_latency_calculate:
      data = []
      labels = []
      max_min_values = []
      max_min_values = []
      total_latency = 0
      ideal_total_latency =0 
      for layer_idx in layers_to_plot:
          layer = load_profiling_layer(args.profiling_dir, layer_idx, args.num_heads)
          for mode in modes:
              reads_per_head, latency_per_layer, ideal_latency_per_layer = get_plane_reads_per_head(layer, args, mode)
              data.append(latency_per_layer)
              total_latency += latency_per_layer
              ideal_total_latency +=ideal_latency_per_layer
              # for each head, compute imbalance = max(reads) - min(reads)
              # for head_reads in reads_per_head:
              #     imbalance = max(head_reads) - min(head_reads)
              #     max_min_values.append(imbalance)    
      print(f"total latency: {total_latency}")
      print(f"ideal total latency: {ideal_total_latency}")
      # plt.figure(figsize=(10,6))
      # # you can tweak bins= and range= to control the x‐axis units and limits
      # plt.hist(max_min_values, bins=30, edgecolor='black')
      # plt.xlabel('Max-Min page reads per head', fontsize=14)
      # plt.ylabel('Number of heads', fontsize=14)
      # plt.title('Distribution of per-head load imbalance', fontsize=16)
      # plt.tight_layout()
      # plt.savefig('head_load_imbalance_histogram.png')
      # plt.close()
      # print("Histogram saved to head_load_imbalance_histogram.png")
      
    else:          
      data_per_head = []
      data_per_layer = []
      labels_per_head = []
      labels_per_layer = []
  
      for layer_idx in layers_to_plot:
          layer = load_profiling_layer(args.profiling_dir, layer_idx, args.num_heads)
          for mode in modes:
              import numpy as np
              page_reads, _, _= get_plane_reads_per_head(layer, args, mode)
              reads_per_head = np.array(page_reads).flatten().tolist()
              data_per_head.append(reads_per_head)
              labels_per_head.append(f"L{layer_idx}-{mode}")

              reads_per_layer = get_plane_reads(layer, args, mode)
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
