import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable
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
        self.softmax_sum = []

    def layout_clusters(
        self,
        head_idx: int,
        pages_per_cluster: Dict[int, int],
        superclusters: List[SuperclusterData],
        hot_cluster_ids: List[int],
        sorted_cids,        
        mode: str,
        softmax_sum
    ) -> None:
        self.softmax_sum.append(softmax_sum)
        cluster_cnt = 0
        # Baseline: serial assign by cluster index ordering
        # Hotness-aware layout: water-falling algorithm based on hotness
        if mode == 'baseline':
            cluster_list = sorted_cids if self.hotness_aware_layout else sorted(pages_per_cluster)
            layout_strategy = "round-robin"
            if layout_strategy == "round-robin":
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

            elif layout_strategy == "zigzag":
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
    def layout_hot_clusters(
        self,
        head_idx: int,
        pages_per_cluster: Dict[int, int],
        superclusters: List[SuperclusterData],
        hot_cluster_ids: List[int],
        sorted_cids,
        mode,
        num_replica:int = 4
    ) -> None:
        # Baseline: serial assign by cluster index ordering
        # Hotness-aware layout: water-falling algorithm based on hotness
            layout_strategy = "round-robin"
            if layout_strategy == "round-robin":
                # round-robin layout based on cluster id
                for replica_set_idx in range(num_replica + 1): # num_replica = 1 means no replication
                  offset = int(replica_set_idx * self.total_planes / num_replica)
                  for cid in hot_cluster_ids:
                      count = pages_per_cluster[cid]
                      pages=[]

                      # compute the window of 'count' pages starting at 'offset'
                      start = offset % self.total_planes
                      end   = (offset + count) % self.total_planes
                      plane_id = self.global_plane_id

                      included = False
                      if start < end:
                          # simple contiguous range
                          if start <= plane_id < end:
                              included = True
                              pages.append(plane_id - start)
                      else:
                          # wrapped-around range
                          if plane_id >= start:
                              included = True
                              pages.append(plane_id - start)
                          if plane_id < end:
                              included = True
                              pages.append(count - (end - plane_id))

                      if included:
                          self.hot_cluster_to_pages.setdefault((head_idx, cid), []).extend(pages)

                      offset += count
            # if layout_strategy == "zigzag":
            #     # Snake(zig-zag) assignment
            #     def zigzag(offset, total_planes):
            #         multiple = offset // total_planes
            #         # if offset is even mutiple of total_planes, forward
            #         if multiple % 2 == 0:
            #             return offset % total_planes
            #         else:
            #             return total_planes - 1 - offset % total_planes
                
            #     for replica_set_idx in range(num_replica):
            #       offset = int(replica_set_idx * self.total_planes / num_replica)
            #       for cid in hot_cluster_ids:
            #           count = pages_per_cluster[cid]
            #           pages=[]
            #           included = False
            #           for i in range(offset, offset+count):
            #               if zigzag(i, self.total_planes) == self.global_plane_id:
            #                   included = True
            #                   pages.append(i - offset)
            #               if included:
            #                   self.hot_cluster_to_pages[(head_idx, cid)] = pages


            #           offset += count

    def simulate_access(
        self,
        head_idx: int,
        selected_clusters: List[int],
        hot_cluster_ids,
        mode: str
    ) -> int:
        sum_output = 0
        softmax_aggregation = 0
        for cid in selected_clusters:
            if self.hot_cluster_duplicate: 
                if cid not in hot_cluster_ids:
                    # in hot cluster duplcate mode, we handle hot cluster separately
                    sum_output += len(self.cluster_to_pages.get((head_idx, cid), []))
            else:
                sum_output += len(self.cluster_to_pages.get((head_idx, cid), []))
                
                if self.cluster_to_pages.get((head_idx, cid), []):
                  softmax_aggregation += self.softmax_sum[head_idx][cid]
                  

        # sum_output = sum(
        #     len(self.cluster_to_pages.get((head_idx, cid), []))
        #     for cid in selected_clusters
        # )
        return sum_output, softmax_aggregation

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
        self.pages_per_cluster = [] # each head has its own pages_per_cluster
        self.softmax_sum = []
        
    def layout_clusters(
        self,
        head_idx: int,
        pages_per_cluster: Dict[int, int],
        superclusters: List[SuperclusterData],
        hot_cluster_ids: List[int],
        softmax_sum,
        mode: str,
        num_replica
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
            plane.layout_clusters(head_idx, pages_per_cluster, superclusters, hot_cluster_ids, sorted_cids, mode, softmax_sum)
            if self.hot_cluster_duplicate:
                plane.layout_hot_clusters(head_idx, pages_per_cluster, superclusters, hot_cluster_ids, sorted_cids, mode, num_replica)
        
        self.pages_per_cluster.append(pages_per_cluster)
        self.softmax_sum.append(softmax_sum)

    def simulate_access(
        self,
        head_idx: int,
        selected_clusters: List[int],
        hot_cluster_ids,
        mode: str
    ) -> int:
        plane_reads = []      
        for plane in self.planes:
          r, _ = plane.simulate_access(
              head_idx,
              selected_clusters,
              hot_cluster_ids=hot_cluster_ids,
              mode=mode
          )                    
          plane_reads.append(r)

        ###### All-to-All duplicate ######
        # if every plane shares all hot clusters
        # # reduce load imbalance with hot cluster pages
        # if args.hot_cluster_duplicate:
        #     selected_hot_cluster = []
        #     for hot_cid in head.hot_cluster_ids:
        #       if hot_cid in head.selected_cluster_ids: 
        #           selected_hot_cluster.append(hot_cid)
        #     # calculate number of pages of selected hot clusters
        #     num_selected_hot_cluster_pages = 0
        #     for cid in selected_hot_cluster:
        #         num_selected_hot_cluster_pages += math.ceil(head.clusters[cid].cluster_size_vectors * args.head_dim * args.vector_bytes / args.page_size_bytes)

        #     plane_reads = balance_values(plane_reads, num_selected_hot_cluster_pages)
        
        ###### Practical duplicate ######
        if self.hot_cluster_duplicate:
            plane_reads = self.simulate_hot_cluster_access_greedy(head_idx, selected_clusters, hot_cluster_ids, plane_reads)
            # temp_plane_reads = plane_reads.copy()  # make a copy to avoid modifying original during greedy allocation
            # plane_reads = self.simulate_hot_cluster_access_optimal(head_idx, selected_clusters, hot_cluster_ids, plane_reads)
                
        return plane_reads

    def simulate_hot_cluster_access_greedy(
        self,
        head_idx: int,
        selected_clusters: List[int],
        hot_cluster_ids,
        plane_reads: List[int],
    ) -> int:
        """
        For each hot cluster in selected_clusters, allocate its pages one by one
        to the plane (among those that hold that cluster) which currently has the
        fewest reads, bumping that plane's read-count each time. Finally return
        the maximum reads across all planes (i.e. the worst-case plane load).
        """
        # 1. restrict to just the “hot” ones
        selected_hot = [c for c in hot_cluster_ids if c in selected_clusters]

        for cluster in selected_hot:
            num_pages = self.pages_per_cluster[head_idx][cluster]
            
            # for each page‐offset in this cluster
            for offset in range(num_pages):
                # gather planes that have this cluster at this offset
                candidate_planes = []
                for i, plane in enumerate(self.planes):
                    # plane.hot_cluster_to_pages[cluster] is the list of offsets it stores
                    pages_on_plane = plane.hot_cluster_to_pages.get((head_idx, cluster), [])
                    if offset in pages_on_plane:
                        candidate_planes.append(i)

                if not candidate_planes:
                    # cluster not on any plane? skip or raise depending on your policy
                    continue

                # pick the least‐loaded plane among those
                best = min(candidate_planes, key=lambda p: plane_reads[p])
                plane_reads[best] += 1

        return plane_reads

    # pages_per_cluster and self.planes should be available in your class

    def simulate_hot_cluster_access_optimal_old(
        self,
        head_idx: int,
        selected_clusters: List[int],
        hot_cluster_ids: List[int],
        initial_plane_reads: List[int],
    ) -> Tuple[List[int], int]:
        """
        Assign hot-cluster pages to planes so as to minimize the maximum
        loaded reads on any plane (makespan), taking an existing starting load
        into account. Uses binary search + feasibility ILP for optimality.

        Returns:
          plane_reads: List of final loads per plane (initial + assigned hot pages).
          makespan: the minimized maximum load.
        """
        import pulp
        from typing import List, Any, Tuple

        # 1. Filter to selected hot clusters
        selected_hot = [c for c in hot_cluster_ids if c in selected_clusters]

        # 2. Build tasks: list of (cluster, offset)
        tasks: List[Tuple[int, int]] = []
        for cluster in selected_hot:
            num_pages = self.pages_per_cluster[head_idx][cluster]
            for offset in range(num_pages):
                tasks.append((cluster, offset))

        num_planes = len(initial_plane_reads)
        total_hot = len(tasks)

        # 3. Precompute candidate planes per task
        candidate_planes = {
            t_idx: [p_idx for p_idx, plane in enumerate(self.planes)
                    if tasks[t_idx][1] in plane.hot_cluster_to_pages.get((head_idx, tasks[t_idx][0]), [])]
            for t_idx in range(total_hot)
        }

        # 4. Bounds for makespan
        base_max = max(initial_plane_reads)
        avg_lb = (sum(initial_plane_reads) + total_hot + num_planes - 1) // num_planes
        lower = max(base_max, avg_lb)
        upper = base_max + total_hot

        best_solution = None
        # 5. Binary search on makespan
        while lower < upper:
            mid = (lower + upper) // 2
            # Feasibility ILP: can we assign all tasks so plane_load <= mid?
            prob = pulp.LpProblem("feasibility_check", pulp.LpStatusOptimal)
            # decision vars x[t_idx][p_idx]
            x = {
                (t_idx, p_idx): pulp.LpVariable(f"x_{t_idx}_{p_idx}", cat="Binary")
                for t_idx, planes_list in candidate_planes.items()
                for p_idx in planes_list
            }
            # each task assigned exactly once
            for t_idx, planes_list in candidate_planes.items():
                prob += pulp.lpSum(x[(t_idx, p)] for p in planes_list) == 1, f"assign_{t_idx}"
            # load constraints
            for p_idx in range(num_planes):
                prob += (
                    pulp.lpSum(x[(t_idx, p_idx)]
                              for t_idx in candidate_planes
                              if (t_idx, p_idx) in x)
                    + initial_plane_reads[p_idx]
                    <= mid,
                    f"load_{p_idx}"
                )
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
            if pulp.LpStatus[prob.status] == 'Optimal':
                upper = mid
                # save solution values
                best_solution = {key: var.value() for key, var in x.items()}
            else:
                lower = mid + 1

        # 6. Build final plane_reads from best_solution
        plane_reads = initial_plane_reads.copy()
        if best_solution:
            for (t_idx, p_idx), val in best_solution.items():
                if val > 0.5:
                    plane_reads[p_idx] += 1

        makespan = lower
        return plane_reads

    def simulate_hot_cluster_access_optimal(
        self,
        head_idx: int,
        selected_clusters: List[int],
        hot_cluster_ids: Iterable[int],
        plane_reads: List[int],
    ) -> List[int]:
        """
        Assign each hot-cluster page to one of its candidate planes
        so as to minimize the maximum load (makespan) across all planes,
        then return the final plane_reads distribution.
        Uses a binary search over possible max-load T and a max-flow check
        for feasibility under candidate-plane constraints.
        After finding the minimal T, reconstructs the assignment to update
        plane_reads and returns the resulting list.
        """
        import networkx as nx
        from typing import List, Iterable
        # 1. Filter only the hot clusters in the selection
        selected_hot = [c for c in hot_cluster_ids if c in selected_clusters]

        # 2. Build a list of tasks: for each page-offset, the eligible planes
        tasks: List[List[int]] = []
        for cluster in selected_hot:
            num_pages = self.pages_per_cluster[head_idx][cluster]
            for offset in range(num_pages):
                eligible = []
                for pi, plane in enumerate(self.planes):
                    pages_on_plane = plane.hot_cluster_to_pages.get((head_idx, cluster), [])
                    if offset in pages_on_plane:
                        eligible.append(pi)
                if eligible:
                    tasks.append(eligible)
        num_tasks = len(tasks)
        # if no tasks, nothing to assign
        if num_tasks == 0:
            return plane_reads

        # keep a copy of the original reads
        initial_reads = list(plane_reads)

        # 3. Binary search on the target makespan T
        lo = max(initial_reads)
        hi = lo + num_tasks  # worst case: all tasks to one plane
        best_T = hi

        while lo <= hi:
            mid = (lo + hi) // 2
            # build flow network
            G = nx.DiGraph()
            src, sink = 'src', 'sink'
            G.add_node(src); G.add_node(sink)
            # source -> task nodes
            for ti in range(num_tasks):
                tnode = f"t{ti}"
                G.add_edge(src, tnode, capacity=1)
                # task -> plane nodes
                for p in tasks[ti]:
                    G.add_edge(tnode, f"p{p}", capacity=1)
            # plane -> sink with capacity = mid - initial_read
            for p, base in enumerate(initial_reads):
                cap = max(mid - base, 0)
                G.add_edge(f"p{p}", sink, capacity=cap)

            flow_val, _ = nx.maximum_flow(G, src, sink)
            if flow_val == num_tasks:
                best_T = mid
                hi = mid - 1
            else:
                lo = mid + 1

        # 4. Reconstruct assignment at best_T
        # reset reads
        plane_reads[:] = initial_reads
        # build final flow network
        G = nx.DiGraph()
        src, sink = 'src', 'sink'
        G.add_node(src); G.add_node(sink)
        for ti in range(num_tasks):
            tnode = f"t{ti}"
            G.add_edge(src, tnode, capacity=1)
            for p in tasks[ti]:
                G.add_edge(tnode, f"p{p}", capacity=1)
        for p, base in enumerate(initial_reads):
            cap = max(best_T - base, 0)
            G.add_edge(f"p{p}", sink, capacity=cap)

        _, flow_dict = nx.maximum_flow(G, src, sink)
        # for each task, find which plane got the flow
        for ti in range(num_tasks):
            tnode = f"t{ti}"
            for p in tasks[ti]:
                if flow_dict.get(tnode, {}).get(f"p{p}", 0) == 1:
                    plane_reads[p] += 1
                    break

        return plane_reads
