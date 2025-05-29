#!/usr/bin/env python
import argparse
import os
import torch
from typing import List, Tuple, Union
from SSD_internal import (
    Plane,
    Chip,
    compute_pages_per_cluster,
    LayerData,
    HeadData,
    ClusterData,
    SuperclusterData
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analytical simulator for in-flash attention layer"
    )
    parser.add_argument('--mode', choices=['baseline', 'cluster_independent', 'cluster_superblock'], required=True)
    parser.add_argument('--num_channels', type=int, required=True)
    parser.add_argument('--chips_per_channel', type=int, required=True)
    parser.add_argument('--dies_per_chip', type=int, required=True)
    parser.add_argument('--planes_per_die', type=int, required=True)
    parser.add_argument('--page_size_bytes', type=int, required=True)
    parser.add_argument('--vector_bytes', type=int, default=2)
    parser.add_argument('--flash_read_latency_us', type=int, default=30)
    parser.add_argument('--profiling_dir', type=str, required=True,
                        help='Directory with layer-wise profiling .pt files')
    parser.add_argument('--num_layers', type=int, default=32,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='KV heads per layer')
    parser.add_argument('--head_dim', type=int, default=128,
                        help='Dimension per KV head')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save results as CSV')
    return parser.parse_args()


def load_profiling_layer(
    profiling_dir: str,
    layer_idx: int,
    num_heads: int
) -> LayerData:
    # Load per-layer profiling data from .pt files
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
        superclusters = [SuperclusterData(sc_id, [int(cid.item()) for cid in ids], supercluster_size[head_idx][sc_id])
                         for sc_id, ids in enumerate(superclusters_list[head_idx])]
        selected = [int(cid.item()) for cid in selected_list[head_idx]]
        heads.append(HeadData(head_idx, clusters, superclusters, selected))
    return LayerData(layer_idx, heads)


def build_chips(args, layer: LayerData) -> List[Chip]:
    chips: List[Chip] = []
    for channel_id in range(args.num_channels):
        for chip_id in range(args.chips_per_channel):
            chip = Chip(
                channel_id=channel_id,
                chip_id=chip_id,
                dies_per_chip=args.dies_per_chip,
                planes_per_die=args.planes_per_die,
                chips_per_channel = args.chips_per_channel,
                num_channels = args.num_channels
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
                    mode=args.mode
                )
            chips.append(chip)
    return chips


def simulate_layer(layer: LayerData, args) -> Tuple[int, int]:
    # Build SSD hierarchy and assign clusters
    chips = build_chips(args, layer)
    head_latencies = []
    for head in layer.heads:
        if args.mode in ('baseline', 'cluster_independent'):
            # sum per-plane page reads
            # chip_pages = [
            #     sum(
            #         plane.simulate_access(
            #             head.head_index,
            #             head.selected_cluster_ids,
            #             args.mode
            #         ) for plane in chip.planes
            #     ) for chip in chips
            # ]
            page_reads = [
                [
                    plane.simulate_access(
                        head.head_index,
                        head.selected_cluster_ids,
                        args.mode
                    ) for plane in chip.planes
                 ] for chip in chips
            ]
            latency_us = max(max(page_reads)) * args.flash_read_latency_us
        else:
            # supercluster_superblock: one read per superpage if any cluster selected
            chip_pages = [
                chip.simulate_access(
                    head.head_index,
                    head.selected_cluster_ids,
                    args.mode
                ) for chip in chips
            ]
            latency_us = max(chip_pages) * args.flash_read_latency_us
        head_latencies.append(latency_us)
    return layer.layer_index, max(head_latencies) if head_latencies else (layer.layer_index, 0)


def main():
    args = parse_args()
    results: List[Tuple[int, int]] = []
    for layer_idx in range(args.num_layers):
        layer = load_profiling_layer(
            args.profiling_dir, layer_idx, args.num_heads
        )
        idx, lat = simulate_layer(layer, args)
        results.append((idx, lat))

    # Output results
    if args.output_csv:
        import csv
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['layer_index', 'latency_us'])
            writer.writerows(results)
        print(f"Results saved to {args.output_csv}")
    else:
        print('layer_index,latency_us')
        for idx, lat in results:
            print(f"{idx},{lat}")

if __name__ == '__main__':
    main()
