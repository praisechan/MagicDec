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
    Represents a single physical plane. Handles both baseline and cluster_independent layouts.
    """
    def __init__(self, plane_id: int, total_planes: int):
        self.plane_id = plane_id
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
        # Baseline: stripe each cluster's pages evenly across planes
        if mode == 'baseline':
            for cid, pcount in pages_per_cluster.items():
                pages = [i for i in range(pcount)
                         if i % self.total_planes == self.plane_id]
                self.cluster_to_pages[(head_idx, cid)] = pages

        # Cluster_independent: layout superclusters in superpage units, then stripe
        elif mode == 'cluster_independent':
            for sc in superclusters:
                # total pages in this supercluster
                total_pages = sum(pages_per_cluster[cid] for cid in sc.cluster_ids)
                # generate global page indices [0..total_pages-1]
                # then pick those that map to this plane
                plane_global_pages = [pid for pid in range(total_pages)
                                       if pid % self.total_planes == self.plane_id]
                # assign per-cluster pages based on cumulative ranges
                offset = 0
                for cid in sc.cluster_ids:
                    count = pages_per_cluster[cid]
                    # pages within this cluster range that reside on this plane
                    pages = [pid - offset for pid in plane_global_pages
                             if offset <= pid < offset + count]
                    key = (head_idx, cid)
                    self.cluster_to_pages.setdefault(key, []).extend(pages)
                    offset += count
        # cluster_superblock is handled by Chip

    def simulate_access(
        self,
        head_idx: int,
        selected_clusters: List[int],
        mode: str
    ) -> int:
        if mode in ('baseline', 'cluster_independent'):
            # count pages touched per selected cluster
            return sum(
                len(self.cluster_to_pages.get((head_idx, cid), []))
                for cid in selected_clusters
            )
        raise ValueError(f"Plane cannot simulate mode {mode}")

class Chip:
    """
    Groups planes for cluster_superblock mode; simulates superpage reads per supercluster.
    """
    def __init__(
        self,
        chip_id: int,
        planes: List[Plane]
    ):
        self.chip_id = chip_id
        self.planes = planes
        self.total_planes = len(planes)
        # maps (head_idx, supercluster_id) -> pages per plane (i.e., superpages count)
        self.supercluster_to_pages_per_plane: Dict[Tuple[int,int], int] = {}
        # maps (head_idx, cluster_id) -> supercluster_id
        self.cluster_to_super: Dict[Tuple[int,int], int] = {}

    def assign_clusters(
        self,
        head_idx: int,
        pages_per_cluster: Dict[int, int],
        superclusters: List[SuperclusterData]
    ) -> None:
        for sc in superclusters:
            total = sum(pages_per_cluster[cid] for cid in sc.cluster_ids)
            per_plane = math.ceil(total / self.total_planes)
            self.supercluster_to_pages_per_plane[(head_idx, sc.supercluster_id)] = per_plane
            for cid in sc.cluster_ids:
                self.cluster_to_super[(head_idx, cid)] = sc.supercluster_id

    def simulate_access(
        self,
        head_idx: int,
        selected_clusters: List[int],
        mode: str
    ) -> int:
        if mode == 'cluster_superblock':
            # one read per touched supercluster
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
