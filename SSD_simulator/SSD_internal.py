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
    Represents a single physical plane for baseline and cluster_independent modes.
    """
    def __init__(self, plane_id: int, total_planes: int):
        self.plane_id = plane_id
        self.total_planes = total_planes
        self.cluster_to_pages: Dict[Tuple[int, int], List[int]] = {}

    def assign_clusters(
        self,
        head_idx: int,
        pages_per_cluster: Dict[int, int],
        superclusters: List[SuperclusterData],  # unused for baseline/independent
        mode: str
    ) -> None:
        if mode in ('baseline', 'cluster_independent'):
            for cid, pcount in pages_per_cluster.items():
                pages = [i for i in range(pcount)
                         if i % self.total_planes == self.plane_id]
                self.cluster_to_pages[(head_idx, cid)] = pages

    def simulate_access(
        self,
        head_idx: int,
        selected_clusters: List[int],
        mode: str
    ) -> int:
        if mode in ('baseline', 'cluster_independent'):
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
        self.supercluster_to_pages_per_plane: Dict[Tuple[int,int], int] = {}
        self.cluster_to_super: Dict[Tuple[int,int], int] = {}

    def assign_clusters(
        self,
        head_idx: int,
        pages_per_cluster: Dict[int, int],
        superclusters: List[SuperclusterData]
    ) -> None:
        for sc in superclusters:
            total_pages = sum(pages_per_cluster[cid] for cid in sc.cluster_ids)
            per_plane = math.ceil(total_pages / self.total_planes)
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
    head_dim: int,
) -> Dict[int, int]:
    return {
        c.cluster_id:
        math.ceil(c.cluster_size_vectors * head_dim * vector_bytes / page_size_bytes)
        for c in head.clusters
    }