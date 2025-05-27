#!/usr/bin/env python
import argparse
import os
import torch
from typing import List, Tuple, Union
from SSD_internal import (
    Plane, Chip,
    compute_pages_per_cluster,
    LayerData, HeadData, ClusterData, SuperclusterData
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
    parser.add_argument('--flash_read_latency_ns', type=int, default=30)
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
    cluster_sizes = torch.load(
        os.path.join(profiling_dir, f"cluster_size_{layer_idx}.pt"), map_location='cpu'
    )
    superclusters_list = torch.load(
        os.path.join(profiling_dir, f"superclusters_{layer_idx}.pt"), map_location='cpu'
    )
    selected_list = torch.load(
        os.path.join(profiling_dir, f"selected_cI_{layer_idx}.pt"), map_location='cpu'
    )

    heads: List[HeadData] = []
    for head_idx in range(num_heads):
        clusters = [ClusterData(cid, int(size.item()))
                    for cid, size in enumerate(cluster_sizes[head_idx])]
        superclusters = [SuperclusterData(sc_id, [int(cid.item()) for cid in ids])
                         for sc_id, ids in enumerate(superclusters_list[head_idx])]
        selected = [int(cid.item()) for cid in selected_list[head_idx]]
        heads.append(HeadData(head_idx, clusters, superclusters, selected))
    return LayerData(layer_idx, heads)


def layout_structures(
    layer: LayerData,
    args: argparse.Namespace
) -> Union[List[Plane], List[Chip]]:
    num_planes = (
        args.num_channels * args.chips_per_channel *
        args.dies_per_chip * args.planes_per_die
    )
    if args.mode == 'cluster_superblock':
        total_chips = args.num_channels * args.chips_per_channel
        planes_per_chip = args.dies_per_chip * args.planes_per_die
        chips: List[Chip] = []
        for chip_id in range(total_chips):
            base = chip_id * planes_per_chip
            planes = [Plane(pid, planes_per_chip)
                      for pid in range(base, base + planes_per_chip)]
            chip = Chip(chip_id, planes)
            for head in layer.heads:
                pages_map = compute_pages_per_cluster(
                    head, args.page_size_bytes, args.vector_bytes, args.head_dim
                )
                chip.assign_clusters(head.head_index, pages_map, head.superclusters)
            chips.append(chip)
        return chips

    planes = [Plane(pid, num_planes) for pid in range(num_planes)]
    for head in layer.heads:
        pages_map = compute_pages_per_cluster(
            head, args.page_size_bytes, args.vector_bytes, args.head_dim
        )
        for pl in planes:
            pl.assign_clusters(head.head_index, pages_map, head.superclusters, args.mode)
    return planes


def simulate_layer(
    layer: LayerData,
    structures: Union[List[Plane], List[Chip]],
    mode: str,
    flash_read_latency_ns: int
) -> Tuple[int, int]:
    head_latencies: List[int] = []
    for head in layer.heads:
        if mode == 'cluster_superblock':
            chip_pages = [
                c.simulate_access(head.head_index, head.selected_cluster_ids, mode)
                for c in structures
            ]
            latency = sum(p * flash_read_latency_ns for p in chip_pages)
        else:
            total_pages = sum(
                pl.simulate_access(head.head_index, head.selected_cluster_ids, mode)
                for pl in structures
            )
            latency = total_pages * flash_read_latency_ns
        head_latencies.append(latency)
    return layer.layer_index, (max(head_latencies) if head_latencies else 0)


def main():
    args = parse_args()
    results: List[Tuple[int, int]] = []
    for layer_idx in range(args.num_layers):
        layer = load_profiling_layer(
            args.profiling_dir, layer_idx, args.num_heads
        )
        structures = layout_structures(layer, args)
        idx, lat = simulate_layer(
            layer, structures, args.mode, args.flash_read_latency_ns
        )
        results.append((idx, lat))

    if args.output_csv:
        import csv
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['layer_index', 'latency_ns'])
            writer.writerows(results)
        print(f"Results saved to {args.output_csv}")
    else:
        print('layer_index,latency_ns')
        for idx, lat in results:
            print(f"{idx},{lat}")

if __name__ == '__main__':
    main()
