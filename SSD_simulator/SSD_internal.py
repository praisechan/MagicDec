import math
from dataclasses import dataclass
from typing import List, Dict, Tuple
import torch

@dataclass
class ClusterData:
    cluster_id: int
    cluster_size_vectors: int # how many tokens are included in a cluster

@dataclass
class SuperclusterData:
    supercluster_id: int
    cluster_ids: List[int]
    supercluster_size: int  # actual number of clusters, rest are zero-padding

@dataclass
class HeadData:
    head_index: int
    clusters: List[ClusterData]
    superclusters: List[SuperclusterData]
    selected_cluster_ids: List[int]
    hot_cluster_ids: List[int]
    softmax_sum: torch.Tensor

@dataclass
class LayerData:
    layer_index: int
    heads: List[HeadData]

class Plane:
    """
    Represents a single physical plane within a die.
    """
    def __init__(
        self,
        channel_id: int,
        chip_id: int,
        die_id: int,
        plane_id: int,
        total_planes: int,
        dies_per_chip: int, 
        planes_per_die: int, 
        chips_per_channel: int, 
        num_channels: int,
        hotness_aware_layout,
        hot_cluster_duplicate,
    ):
        self.channel_id = channel_id
        self.chip_id = chip_id
        self.die_id = die_id
        self.local_plane_id = plane_id # plane id in a chip
        self.global_plane_id = plane_id + die_id * planes_per_die \
                              + chip_id * dies_per_chip * planes_per_die \
                              + channel_id * chips_per_channel * dies_per_chip * planes_per_die # plane index in total planes
        self.total_planes = total_planes
        # maps (head_idx, cluster_id) -> list of page indices on this plane
        self.cluster_to_pages: Dict[Tuple[int, int], List[int]] = {}
        self.hot_cluster_to_pages: Dict[Tuple[int, int], List[int]] = {}
        self.hotness_aware_layout = hotness_aware_layout
        self.hot_cluster_duplicate = hot_cluster_duplicate

    def assign_clusters(
        self,
        head_idx: int,
        pages_per_cluster: Dict[int, int],
        superclusters: List[SuperclusterData],
        hot_cluster_ids: List[int],
        sorted_cids,        
        mode: str
    ) -> None:
        cluster_cnt = 0
        # Baseline: serial assign by cluster index ordering
        # Hotness-aware layout: water-falling algorithm based on hotness
        if mode == 'baseline':
            cluster_list = sorted_cids if self.hotness_aware_layout else sorted(pages_per_cluster)
            assign_strategy = "round-robin"
            if assign_strategy == "round-robin":
                # round-robin layout based on cluster id
                offset = 0
                for cid in cluster_list:
                    if self.hot_cluster_duplicate:
                        if cid in hot_cluster_ids:
                            continue # if hot cluster, do not have to assign to plane, because every plane has the same hot clusters.

                    count = pages_per_cluster[cid]

                    # compute the window of 'count' pages starting at 'offset'
                    start = offset % self.total_planes
                    end   = (offset + count) % self.total_planes
                    plane_id = self.global_plane_id

                    included = False
                    if start < end:
                        # simple contiguous range
                        if start <= plane_id < end:
                            included = True
                    else:
                        # wrapped-around range
                        if plane_id >= start or plane_id < end:
                            included = True

                    if included:
                        self.cluster_to_pages[(head_idx, cid)] = [True]

                    offset += count

            elif assign_strategy == "zigzag":
                # Snake(zig-zag) assignment
                def zigzag(offset, total_planes):
                    multiple = offset // total_planes
                    # if offset is even mutiple of total_planes, forward
                    if multiple % 2 == 0:
                        return offset % total_planes
                    else:
                        return total_planes - 1 - offset % total_planes
                
                offset = 0
                for cid in cluster_list:
                    if self.hot_cluster_duplicate:
                        if cid in hot_cluster_ids:
                            continue # if hot cluster, do not have to assign to plane, because every plane has the same hot clusters.

                    count = pages_per_cluster[cid]
                    included = False
                    for i in range(offset, offset+count):
                        if zigzag(i, self.total_planes) == self.global_plane_id:
                            included = True

                    if included:
                        self.cluster_to_pages[(head_idx, cid)] = [True]

                    offset += count

        # # Supercluster modes: assign in supercluster order, round-robin like baseline
        # elif mode == 'supercluster':
        #     offset = 0
        #     for sc in sorted(superclusters, key=lambda x: x.supercluster_id):
        #         # only consider actual clusters, ignore zero-padding
        #         valid_cids = sc.cluster_ids[:sc.supercluster_size]
        #         for cid in valid_cids:
        #             count = pages_per_cluster[cid]
        #             # round-robin distribution across planes
        #             pages = [i - offset
        #                      for i in range(offset, offset + count)
        #                      if i % self.total_planes == self.global_plane_id]
        #             if pages != []:
        #                 self.cluster_to_pages[(head_idx, cid)] = pages
        #             offset += count
        #         cluster_cnt += sc.supercluster_size
        # else:
        #     pass

    def simulate_access(
        self,
        head_idx: int,
        selected_clusters: List[int],
        hot_cluster_ids,
        mode: str
    ) -> int:
        sum_output = 0
        for cid in selected_clusters:
            if self.hot_cluster_duplicate:
                if cid not in hot_cluster_ids:
                    sum_output += len(self.cluster_to_pages.get((head_idx, cid), []))
            else:
                sum_output += len(self.cluster_to_pages.get((head_idx, cid), []))

        # sum_output = sum(
        #     len(self.cluster_to_pages.get((head_idx, cid), []))
        #     for cid in selected_clusters
        # )
        return sum_output

class Chip:
    """
    Groups planes for cluster_superblock mode; simulates superpage reads per supercluster.
    """
    def __init__(
        self,
        channel_id: int,
        chip_id: int,
        dies_per_chip: int,
        planes_per_die: int,
        chips_per_channel: int,
        num_channels: int,
        hotness_aware_layout,
        hot_cluster_duplicate,
    ):
        self.channel_id = channel_id
        self.chip_id = chip_id
        self.dies_per_chip = dies_per_chip
        self.planes_per_die = planes_per_die
        self.total_planes = dies_per_chip * planes_per_die * chips_per_channel * num_channels
        self.planes: List[Plane] = [
            Plane(channel_id, chip_id, die, plane, self.total_planes, dies_per_chip, planes_per_die, chips_per_channel, num_channels, hotness_aware_layout, hot_cluster_duplicate)
            for die in range(dies_per_chip)
            for plane in range(planes_per_die)
        ]
        self.hotness_aware_layout = hotness_aware_layout
        self.hot_cluster_duplicate = hot_cluster_duplicate

    def assign_clusters(
        self,
        head_idx: int,
        pages_per_cluster: Dict[int, int],
        superclusters: List[SuperclusterData],
        hot_cluster_ids: List[int],
        softmax_sum,
        mode: str
    ) -> None:
        
        sorted_cids = None
        if self.hotness_aware_layout:
            # Sort cluster IDs by descending hotness score
            sorted_cids = sorted(
                pages_per_cluster.keys(),
                key=lambda cid: softmax_sum[cid],
                reverse=True
            )

        # delegate to planes for baseline and independent layouts
        for plane in self.planes:
            plane.assign_clusters(head_idx, pages_per_cluster, superclusters, hot_cluster_ids, sorted_cids, mode)