
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple
from SSD_internal import HeadData

def compute_pages_per_cluster(
    head: HeadData,
    page_size_bytes: int,
    vector_bytes: int,
    head_dim: int,
    constrained: bool,
    cluster_size: int,
) -> Dict[int, int]:
    if constrained:
        pages = {c.cluster_id: int(cluster_size * head_dim * vector_bytes / page_size_bytes) for c in head.clusters}  
    else:
        pages = {
            c.cluster_id:
            math.ceil(c.cluster_size_vectors * head_dim * vector_bytes / page_size_bytes)
            for c in head.clusters
        }  
  
    return pages


def balance_values(values, budget):
    """
    Evenly distribute a budget among list elements to reduce imbalance.

    Args:
        values (List[int]): Original list of integer values.
        budget (int): Total amount to distribute.

    Returns:
        List[int]: New list with budget distributed to minimize range.
    """
    n = len(values)
    # Work on a list of (value, original_index)
    paired = sorted((v, i) for i, v in enumerate(values))
    result = list(values)
    i = 1
    # Iterate over unique levels
    while i <= n and budget > 0:
        # Number of lowest elements considered
        k = i
        # Current level value
        level = paired[0][0]
        # Next level value or infinity
        next_val = paired[i][0] if i < n else None
        if next_val is not None:
            diff = next_val - level
            cost_for_level = diff * k
            if budget >= cost_for_level:
                # Raise all k elements up to next_val
                for j in range(k):
                    _, idx = paired[j]
                    result[idx] += diff
                    paired[j] = (next_val, idx)
                budget -= cost_for_level
                i += 1
                continue
        # Can't reach the next level (or no next level)
        # Distribute remaining budget evenly among k elements
        q, r = divmod(budget, k)
        for j in range(k):
            _, idx = paired[j]
            result[idx] += q + (1 if j < r else 0)
        break

    return result