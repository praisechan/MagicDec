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
    parser.add_argument('--head_dim', type=int, default=128,
                        help='Dimension per KV head')
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
                    args.head_dim
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


def main():
    args = parse_args()
    layers_to_plot = [0, 10, 20]
    modes = ['baseline', 'cluster']
    data = []
    labels = []
    for layer_idx in layers_to_plot:
        layer = load_profiling_layer(args.profiling_dir, layer_idx, args.num_heads)
        for mode in modes:
            reads = get_plane_reads(layer, args, mode)
            data.append(reads)
            labels.append(f"L{layer_idx}-{mode}")

    # violin plot for selected layers
    plt.figure(figsize=(12, 6))
    parts = plt.violinplot(data, showmeans=True)
    plt.xticks(range(1, len(labels) + 1), labels, fontsize=14)
    plt.ylabel('Total page reads per plane', fontsize=14)
    plt.title('Page Reads per Plane (Layer 0, 10, 20)', fontsize=14)
    plt.tight_layout()
    violin_filename = 'layers0_10_20_violin.png'
    plt.savefig(violin_filename)
    plt.close()
    print(f"Violin plot saved to {violin_filename}")

if __name__ == '__main__':
    main()
