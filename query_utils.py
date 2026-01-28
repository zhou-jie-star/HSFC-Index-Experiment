import bisect
import timeit
import numpy as np
import random
from typing import List, Tuple
from sfc_utils import calc_sfc_position_multidim, get_dimension_count


class QueryWorkloadManager:
    """Query Workload Manager"""

    def __init__(self, train_ratio: float = 0.1, random_state: int = 42):
        self.train_ratio = train_ratio
        random.seed(random_state)
        np.random.seed(random_state)

    def split_queries(self, queries: List) -> Tuple[List, List]:
        queries_copy = queries.copy()
        random.shuffle(queries_copy)
        n_train = int(len(queries_copy) * self.train_ratio)
        return queries_copy[:n_train], queries_copy[n_train:]


def convert_query_to_multidim_format(query, n_dims):
    """Standardize query format"""
    if hasattr(query, 'min_point') and hasattr(query, 'max_point'):
        min_coords = tuple(query.min_point)
        max_coords = tuple(query.max_point)
    elif isinstance(query, tuple) and len(query) == 2:
        min_coords = tuple(query[0])
        max_coords = tuple(query[1])
    else:
        raise ValueError(f"Unsupported query format: {type(query)}")
    return min_coords, max_coords


def query_process_multidim(data_points, query, theta, theta_values, curve):
    """Execute single SFC query"""
    query_min, query_max = query
    n_dims = len(query_min)

    # Simple Min/Max SFC range estimation (Actual HSFC uses Query Splitting)
    # Calculate corners
    corners = []
    for i in range(2 ** n_dims):
        corner = []
        for d in range(n_dims):
            corner.append(query_max[d] if (i >> d) & 1 else query_min[d])
        corners.append(calc_sfc_position_multidim(tuple(corner), theta))

    start_point = min(corners)
    end_point = max(corners)

    # Binary Search
    start_index = bisect.bisect_left(curve, (start_point, *([0] * n_dims)))
    end_index = bisect.bisect_right(curve, (end_point, *([float('inf')] * n_dims)))

    num_points = 0
    false_pos = 0

    # Scan
    for i in range(start_index, end_index):
        coords = curve[i][1:n_dims + 1]
        in_range = True
        for d in range(n_dims):
            if not (query_min[d] <= coords[d] <= query_max[d]):
                in_range = False
                break
        if in_range:
            num_points += 1
        else:
            false_pos += 1

    return num_points, false_pos


def objective_function_with_queries_multidim(data_points, region_min, region_max, theta, theta_values, curve, queries):
    """Objective Function: Total Query Time"""
    total_time = 0
    for query in queries:
        t = timeit.timeit(lambda: query_process_multidim(data_points, query, theta, theta_values, curve), number=1)
        total_time += t
    return total_time


def evaluate_sfc_performance_multidim(data_points, region_min, region_max, theta, theta_values, curve, queries):
    """Evaluation Function"""
    total_time = 0
    total_points = 0
    total_fp = 0

    for query in queries:
        start = timeit.default_timer()
        pts, fp = query_process_multidim(data_points, query, theta, theta_values, curve)
        end = timeit.default_timer()
        total_time += (end - start)
        total_points += pts
        total_fp += fp

    count = len(queries)
    return {
        'total_time': total_time,
        'avg_time': total_time / count if count > 0 else 0,
        'total_points': total_points,
        'false_positives': total_fp,
        'false_positive_rate': total_fp / max(1, total_points + total_fp),
        'query_count': count
    }


class QuerySplitter:
    """
    [Core of HSFC] Recursive Query Splitter
    Reference: Query Splitting Strategy
    """

    def __init__(self, k_maxsplit: int = 3):
        self.k_maxsplit = k_maxsplit

    def get_weight_at_bit(self, theta_dim, bit_idx):
        if bit_idx < 0 or bit_idx >= len(theta_dim): return 0
        return int(theta_dim[bit_idx])

    def get_v(self, query, delta):
        q_min, q_max = query
        dim_min, dim_max = int(q_min[delta]), int(q_max[delta])
        if dim_min >= dim_max: return -1, -1
        xor_val = dim_min ^ dim_max
        if xor_val == 0: return -1, -1
        msb_idx = xor_val.bit_length() - 1
        v_candidate = (dim_max >> msb_idx) << msb_idx
        if v_candidate <= dim_min or v_candidate > dim_max: return -1, -1
        return v_candidate, msb_idx

    def recursive_query_splitting(self, query, theta):
        q_splits = []
        n_dims = len(query[0])

        def split(q, k):
            q_min, q_max = q
            if k == 0:
                q_splits.append(q)
                return

            # Find best split dimension
            best_weight = -1
            best_delta = None
            best_v = None

            for delta in range(n_dims):
                v, idx = self.get_v(q, delta)
                if v != -1:
                    w = self.get_weight_at_bit(theta[delta], idx)
                    if w > best_weight:
                        best_weight = w
                        best_delta = delta
                        best_v = v

            if best_delta is None:
                q_splits.append(q)
                return

            # Execute split
            max1 = list(q_max)
            max1[best_delta] = int(best_v) - 1
            min2 = list(q_min)
            min2[best_delta] = int(best_v)

            q1 = (tuple(q_min), tuple(max1))
            q2 = (tuple(min2), tuple(q_max))

            split(q1, k - 1)
            split(q2, k - 1)

        split(query, self.k_maxsplit)
        return q_splits


def apply_query_splitting(queries, theta, k_maxsplit=3, debug=False):
    """Apply query splitting"""
    splitter = QuerySplitter(k_maxsplit)
    split_queries = []
    for q in queries:
        try:
            split_queries.extend(splitter.recursive_query_splitting(q, theta))
        except:
            split_queries.append(q)
    return split_queries