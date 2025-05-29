import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class ClusterData:
    cluster_id: int
    cluster_size_vectors: int

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
        num_channels: int
    ):
        self.channel_id = channel_id
        self.chip_id = chip_id
        self.die_id = die_id
        self.plane_id = plane_id
        self.total_plane_id = plane_id + die_id * planes_per_die \
                              + chip_id * dies_per_chip * planes_per_die \
                              + channel_id * chips_per_channel * dies_per_chip * planes_per_die # plane index in total planes
        self.total_planes = total_planes
        # maps (head_idx, cluster_id) -> list of page indices on this plane
        self.cluster_to_pages: Dict[Tuple[int, int], List[int]] = {}

    def assign_clusters(
        self,
        head_idx: int,
        pages_per_cluster: Dict[int, int],
        superclusters: List[SuperclusterData],
        mode: str
    ) -> None:
        cluster_cnt = 0
        # Baseline: serial assign by cluster index ordering
        if mode == 'baseline':
            offset = 0
            for cid in sorted(pages_per_cluster):
                count = pages_per_cluster[cid]
                pages = [i - offset for i in range(offset, offset + count)
                         if i % self.total_planes == self.total_plane_id]
                if pages != []:
                    self.cluster_to_pages[(head_idx, cid)] = pages
                offset += count

        # Supercluster modes: assign in supercluster order, round-robin like baseline
        elif mode in ('cluster_independent', 'cluster_superblock'):
            offset = 0
            for sc in sorted(superclusters, key=lambda x: x.supercluster_id):
                # only consider actual clusters, ignore zero-padding
                valid_cids = sc.cluster_ids[:sc.supercluster_size]
                for cid in valid_cids:
                    count = pages_per_cluster[cid]
                    # round-robin distribution across planes
                    pages = [i - offset
                             for i in range(offset, offset + count)
                             if i % self.total_planes == self.total_plane_id]
                    if pages != []:
                        self.cluster_to_pages[(head_idx, cid)] = pages
                    offset += count
                cluster_cnt += sc.supercluster_size
        else:
            # chip-level modes do not assign here
            pass
        print(len(self.cluster_to_pages))

    def simulate_access(
        self,
        head_idx: int,
        selected_clusters: List[int],
        mode: str
    ) -> int:
        if mode in ('baseline', 'cluster_independent'):
            sum_output = sum(
                len(self.cluster_to_pages.get((head_idx, cid), []))
                for cid in selected_clusters
            )
            return sum_output
        raise ValueError(f"Plane cannot simulate mode {mode}")

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
    ):
        self.channel_id = channel_id
        self.chip_id = chip_id
        self.dies_per_chip = dies_per_chip
        self.planes_per_die = planes_per_die
        self.total_planes = dies_per_chip * planes_per_die * chips_per_channel * num_channels
        self.planes: List[Plane] = [
            Plane(channel_id, chip_id, die, plane, self.total_planes, dies_per_chip, planes_per_die, chips_per_channel, num_channels)
            for die in range(dies_per_chip)
            for plane in range(planes_per_die)
        ]
        # maps (head_idx, supercluster_id) -> pages per plane for each supercluster
        self.supercluster_to_pages_per_plane: Dict[Tuple[int, int], int] = {}
        # maps (head_idx, cluster_id) -> supercluster_id
        self.cluster_to_super: Dict[Tuple[int, int], int] = {}

    def assign_clusters(
        self,
        head_idx: int,
        pages_per_cluster: Dict[int, int],
        superclusters: List[SuperclusterData],
        mode: str
    ) -> None:
        # record supercluster to plane page counts and map clusters
        for sc in superclusters:
            valid_cids = sc.cluster_ids[:sc.supercluster_size]
            total = sum(pages_per_cluster[cid] for cid in valid_cids)
            per_plane = math.ceil(total / self.total_planes)
            self.supercluster_to_pages_per_plane[(head_idx, sc.supercluster_id)] = per_plane
            for cid in valid_cids:
                self.cluster_to_super[(head_idx, cid)] = sc.supercluster_id
        # delegate to planes for baseline and independent layouts
        for plane in self.planes:
            plane.assign_clusters(head_idx, pages_per_cluster, superclusters, mode)

    def simulate_access(
        self,
        head_idx: int,
        selected_clusters: List[int],
        mode: str
    ) -> int:
        if mode == 'cluster_superblock':
            touched = {
                self.cluster_to_super[(head_idx, cid)]
                for cid in selected_clusters
                if (head_idx, cid) in self.cluster_to_super
            }
            return sum(
                self.supercluster_to_pages_per_plane[(head_idx, sc)]
                for sc in touched
            )
        raise ValueError(f"Chip cannot simulate mode {mode}")


def compute_pages_per_cluster(
    head: HeadData,
    page_size_bytes: int,
    vector_bytes: int,
    head_dim: int
) -> Dict[int, int]:
    return {
        c.cluster_id:
        math.ceil(c.cluster_size_vectors * head_dim * vector_bytes / page_size_bytes)
        for c in head.clusters
    }
