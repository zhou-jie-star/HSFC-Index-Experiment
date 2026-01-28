import json
import numpy as np
from typing import List, Tuple, Dict, Optional
import copy


class Query:
    """Multi-dimensional Query Range Class"""

    def __init__(self, min_point: List[float], max_point: List[float], weight: float = 1.0):
        self.min_point = np.array(min_point)
        self.max_point = np.array(max_point)
        self.weight = weight
        self.n_dims = len(self.min_point)

        # Validate dimension consistency
        if len(self.max_point) != self.n_dims:
            raise ValueError(f"min_point and max_point dimension mismatch: {self.n_dims} vs {len(self.max_point)}")

    def intersects_region(self, region_min: np.ndarray, region_max: np.ndarray) -> bool:
        """Determine if query intersects multi-dimensional region"""
        # Validate dimensions
        if len(region_min) != self.n_dims or len(region_max) != self.n_dims:
            raise ValueError(f"Region dimension ({len(region_min)}, {len(region_max)}) does not match query dimension ({self.n_dims})")

        # Check for overlap in each dimension
        return not np.any(self.max_point < region_min) and not np.any(self.min_point > region_max)

    def get_overlap_volume(self, region_min: np.ndarray, region_max: np.ndarray) -> float:
        """Calculate overlap volume between query and region (multi-dimensional)"""
        if not self.intersects_region(region_min, region_max):
            return 0.0

        # Calculate overlap length for each dimension
        overlap_min = np.maximum(self.min_point, region_min)
        overlap_max = np.minimum(self.max_point, region_max)

        overlap_sizes = overlap_max - overlap_min

        # Multi-dimensional volume is the product of lengths in all dimensions
        volume = np.prod(overlap_sizes)
        return volume

    def __repr__(self):
        return f"Query(min={self.min_point}, max={self.max_point}, dims={self.n_dims}, weight={self.weight})"


class TreeNode:
    """Multi-dimensional Binary Tree Node"""

    def __init__(self, points: List[np.ndarray], region_min: np.ndarray, region_max: np.ndarray,
                 depth: int = 0, queries: List[Query] = None):
        self.points = points
        self.region_min = region_min.copy()
        self.region_max = region_max.copy()
        self.depth = depth
        self.queries = queries if queries is not None else []
        self.n_dims = len(region_min)

        # Validate dimension consistency
        if len(region_max) != self.n_dims:
            raise ValueError(f"region_min and region_max dimension mismatch: {self.n_dims} vs {len(region_max)}")

        # Split information
        self.split_dim = None
        self.split_value = None

        # Child nodes
        self.left_child = None
        self.right_child = None

        # Is leaf node
        self.is_leaf = True

        # Cost information
        self.cost_info = None

        # SFC Optimization Results
        self.sfc_theta = None
        self.sfc_curve = None
        self.sfc_performance = None

    def get_region_volume(self) -> float:
        """Calculate multi-dimensional volume of the region"""
        region_sizes = self.region_max - self.region_min
        return np.prod(region_sizes)

    def __repr__(self):
        return f"TreeNode(points={len(self.points)}, depth={self.depth}, dims={self.n_dims}, " \
               f"region=[{self.region_min}, {self.region_max}], is_leaf={self.is_leaf})"


class MultiLevelSpatialPartitioning:
    """Multi-dimensional Multi-level Bucket Spatial Partitioning based on Query Workload"""

    def __init__(self, min_points: int = 10, max_depth: int = 20,
                 num_buckets: int = 50, bucket_levels: int = 3,
                 penalty: float = 2.0, balance_weight_base: float = 1.0,
                 alpha_start: float = 0.3, alpha_end: float = 0.8,
                 balance_threshold_base: float = 0.4, balance_threshold_decay: float = 0.05):
        self.min_points = min_points
        self.max_depth = max_depth
        self.num_buckets = num_buckets
        self.bucket_levels = bucket_levels
        self.penalty = penalty
        self.balance_weight_base = balance_weight_base
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.balance_threshold_base = balance_threshold_base
        self.balance_threshold_decay = balance_threshold_decay

    def compute_variance_multidim(self, points: List[np.ndarray]) -> np.ndarray:
        """Calculate variance for each dimension (multi-dimensional)"""
        if not points:
            return np.array([])

        points_array = np.array(points)
        return np.var(points_array, axis=0)

    def select_split_dimension_multidim(self, points: List[np.ndarray]) -> int:
        """Select split dimension using max variance method (multi-dimensional)"""
        variances = self.compute_variance_multidim(points)
        if len(variances) == 0:
            return 0
        return np.argmax(variances)

    def select_split_dimension(self, points: List[np.ndarray]) -> int:
        """Backward compatible split dimension selection"""
        return self.select_split_dimension_multidim(points)

    def create_buckets(self, min_val: float, max_val: float, num_buckets: int) -> Tuple[np.ndarray, float]:
        """Create buckets within specified range"""
        if min_val == max_val:
            return np.array([min_val]), 0
        bucket_width = (max_val - min_val) / num_buckets
        bucket_centers = np.linspace(min_val + bucket_width / 2, max_val - bucket_width / 2, num_buckets)
        return bucket_centers, bucket_width

    def get_depth_weights(self, depth: int) -> Tuple[float, float]:
        """Calculate query weight and balance weight based on depth"""
        progress = min(1.0, depth / self.max_depth)
        alpha = self.alpha_start + progress * (self.alpha_end - self.alpha_start)
        beta = 1.0 - alpha
        return alpha, beta

    def get_balance_threshold(self, depth: int) -> float:
        """Calculate balance threshold based on depth"""
        return max(0.1, self.balance_threshold_base - self.balance_threshold_decay * depth)

    def compute_query_cost_multidim(self, split_value: float, split_dim: int, relevant_queries: List[Query]) -> float:
        """Calculate multi-dimensional query cost"""
        if not relevant_queries:
            return 0.0

        cross_penalty = 0.0
        total_weight = 0.0

        for query in relevant_queries:
            weight = getattr(query, 'weight', 1.0)
            total_weight += weight

            # Check if query crosses the split line on the split dimension
            q_low = query.min_point[split_dim]
            q_high = query.max_point[split_dim]

            if q_low <= split_value < q_high:
                cross_penalty += weight * self.penalty

        if total_weight > 0:
            return cross_penalty / total_weight
        return 0.0

    def compute_query_cost(self, split_value: float, split_dim: int, relevant_queries: List[Query]) -> float:
        """Backward compatible query cost calculation"""
        return self.compute_query_cost_multidim(split_value, split_dim, relevant_queries)

    def compute_balance_cost(self, left_points_count: int, total_points_count: int) -> float:
        """Calculate balance cost"""
        if total_points_count == 0:
            return 0.0

        p_left = left_points_count / total_points_count
        p_right = 1.0 - p_left

        if p_left == 0 or p_right == 0:
            return 1.0

        entropy = -(p_left * np.log2(p_left) + p_right * np.log2(p_right))
        max_entropy = 1.0

        return 1.0 - (entropy / max_entropy)

    def compute_total_cost(self, split_value: float, split_dim: int, points: List[np.ndarray],
                           relevant_queries: List[Query], depth: int) -> Tuple[float, Dict]:
        """Calculate total cost"""
        left_points_count = sum(1 for p in points if p[split_dim] <= split_value)
        total_points_count = len(points)

        balance_threshold = self.get_balance_threshold(depth)
        p_left = left_points_count / total_points_count if total_points_count > 0 else 0.5
        p_right = 1.0 - p_left

        if min(p_left, p_right) < balance_threshold:
            cost_details = {
                'query_cost': float('inf'),
                'balance_cost': float('inf'),
                'total_cost': float('inf'),
                'alpha': 0,
                'beta': 0,
                'balance_violated': True,
                'left_ratio': p_left,
                'balance_threshold': balance_threshold
            }
            return float('inf'), cost_details

        query_cost = self.compute_query_cost_multidim(split_value, split_dim, relevant_queries)
        balance_cost = self.compute_balance_cost(left_points_count, total_points_count)

        alpha, beta = self.get_depth_weights(depth)
        total_cost = alpha * query_cost + beta * balance_cost

        cost_details = {
            'query_cost': query_cost,
            'balance_cost': balance_cost,
            'total_cost': total_cost,
            'alpha': alpha,
            'beta': beta,
            'balance_violated': False,
            'left_ratio': p_left,
            'balance_threshold': balance_threshold
        }

        return total_cost, cost_details

    def compute_bucket_costs(self, bucket_centers: np.ndarray, bucket_width: float,
                             split_dim: int, points: List[np.ndarray],
                             relevant_queries: List[Query], depth: int) -> Tuple[np.ndarray, List[Dict]]:
        """Calculate cost for each bucket center as a split point"""
        costs = np.zeros(len(bucket_centers))
        cost_details_list = []

        for i, split_value in enumerate(bucket_centers):
            total_cost, cost_details = self.compute_total_cost(
                split_value, split_dim, points, relevant_queries, depth
            )
            costs[i] = total_cost
            cost_details_list.append(cost_details)

        return costs, cost_details_list

    def multi_level_bucket_search(self, min_val: float, max_val: float, split_dim: int,
                                  points: List[np.ndarray], relevant_queries: List[Query],
                                  depth: int, level: int) -> Tuple[float, float, int, Dict]:
        """Multi-level bucket search for optimal split point"""
        if level <= 0 or min_val == max_val:
            mid_value = (min_val + max_val) / 2
            _, cost_details = self.compute_total_cost(mid_value, split_dim, points, relevant_queries, depth)
            return mid_value, cost_details['total_cost'], 1, cost_details

        bucket_centers, bucket_width = self.create_buckets(min_val, max_val, self.num_buckets)

        if len(bucket_centers) == 1:
            _, cost_details = self.compute_total_cost(bucket_centers[0], split_dim, points, relevant_queries, depth)
            return bucket_centers[0], cost_details['total_cost'], 1, cost_details

        costs, cost_details_list = self.compute_bucket_costs(
            bucket_centers, bucket_width, split_dim, points, relevant_queries, depth
        )

        best_bucket_idx = np.argmin(costs)
        best_cost = costs[best_bucket_idx]
        best_split_value = bucket_centers[best_bucket_idx]
        best_cost_details = cost_details_list[best_bucket_idx]
        total_buckets = len(bucket_centers)

        if level > 1:
            bucket_min = max(best_split_value - bucket_width / 2, min_val)
            bucket_max = min(best_split_value + bucket_width / 2, max_val)

            refined_split_value, refined_cost, sub_buckets, refined_cost_details = self.multi_level_bucket_search(
                bucket_min, bucket_max, split_dim, points, relevant_queries, depth, level - 1
            )

            total_buckets += sub_buckets

            if refined_cost < best_cost:
                best_split_value = refined_split_value
                best_cost = refined_cost
                best_cost_details = refined_cost_details

        return best_split_value, best_cost, total_buckets, best_cost_details

    def select_split_value(self, points: List[np.ndarray], split_dim: int,
                           relevant_queries: List[Query], depth: int) -> Optional[Tuple[float, Dict]]:
        """Select split point (Modified: force integer split point)"""
        if not points:
            return None

        values = np.array([point[split_dim] for point in points])
        min_val, max_val = np.min(values), np.max(values)

        if min_val == max_val:
            return int(min_val), {"buckets_evaluated": 1, "levels_used": 1, "cost_details": None}

        # Original logic: find optimal float split point
        best_split_value, best_cost, total_buckets, best_cost_details = self.multi_level_bucket_search(
            min_val, max_val, split_dim, points, relevant_queries, depth, self.bucket_levels
        )

        # New: Convert float split point to integer
        integer_split_value, integer_cost_details = self.convert_to_integer_split(
            best_split_value, split_dim, points, relevant_queries, depth, min_val, max_val
        )

        search_info = {
            "buckets_evaluated": total_buckets,
            "levels_used": self.bucket_levels,
            "best_cost": integer_cost_details['total_cost'],
            "value_range": (min_val, max_val),
            "cost_details": integer_cost_details,
            "original_float_split": best_split_value,
            "integer_split": integer_split_value
        }

        return integer_split_value, search_info

    def convert_to_integer_split(self, float_split_value: float, split_dim: int,
                                 points: List[np.ndarray], relevant_queries: List[Query],
                                 depth: int, min_val: float, max_val: float) -> Tuple[int, Dict]:
        """Convert float split point to integer (using floor only)"""

        # Generate floor candidate only
        candidates = []
        floor_val = int(np.floor(float_split_value))

        # Ensure candidate is within valid range
        if min_val <= floor_val <= max_val:
            candidates.append(floor_val)
        else:
            # If floor is out of range, use closest valid integer
            if float_split_value < min_val:
                candidates = [int(np.ceil(min_val))]
            else:
                candidates = [int(np.floor(max_val))]

        # Evaluate floor candidate
        best_int_split = candidates[0]
        best_cost_details = None

        # Check if this candidate effectively splits the data
        left_count = sum(1 for p in points if p[split_dim] <= best_int_split)
        right_count = len(points) - left_count

        # Calculate cost if split is valid
        if left_count > 0 and right_count > 0:
            _, best_cost_details = self.compute_total_cost(
                float(best_int_split), split_dim, points, relevant_queries, depth
            )

        # If floor candidate cannot split effectively, fallback to search for closest valid integer
        if best_cost_details is None:
            for offset in range(1, int(max_val - min_val) + 1):
                # Prioritize left search (smaller integers) consistent with floor
                for candidate in [int(float_split_value) - offset, int(float_split_value) + offset]:
                    if min_val <= candidate <= max_val:
                        left_count = sum(1 for p in points if p[split_dim] <= candidate)
                        right_count = len(points) - left_count
                        if left_count > 0 and right_count > 0:
                            _, best_cost_details = self.compute_total_cost(
                                float(candidate), split_dim, points, relevant_queries, depth
                            )
                            best_int_split = candidate
                            break
                if best_cost_details is not None:
                    break

        return best_int_split, best_cost_details

    def filter_queries_for_region_multidim(self, queries: List[Query], region_min: np.ndarray,
                                           region_max: np.ndarray) -> List[Query]:
        """Filter queries intersecting with current multi-dimensional region"""
        relevant_queries = []
        for query in queries:
            try:
                if query.intersects_region(region_min, region_max):
                    relevant_queries.append(query)
            except ValueError as e:
                # Handle dimension mismatch
                print(f"Warning: Query dimension mismatch, skipping: {e}")
                continue
        return relevant_queries

    def filter_queries_for_region(self, queries: List[Query], region_min: np.ndarray,
                                  region_max: np.ndarray) -> List[Query]:
        """Backward compatible query filtering"""
        return self.filter_queries_for_region_multidim(queries, region_min, region_max)

    def split_points_multidim(self, points: List[np.ndarray], split_dim: int,
                              split_value: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Split multi-dimensional data points based on split value"""
        left_points = []
        right_points = []

        for point in points:
            if point[split_dim] <= split_value:
                left_points.append(point)
            else:
                right_points.append(point)

        return left_points, right_points

    def split_points(self, points: List[np.ndarray], split_dim: int,
                     split_value: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Backward compatible data point splitting"""
        return self.split_points_multidim(points, split_dim, split_value)

    def should_stop_splitting(self, node: TreeNode) -> bool:
        """Determine if splitting should stop"""
        return (len(node.points) <= self.min_points or node.depth >= self.max_depth)

    def build_tree(self, points: List[np.ndarray], queries: List[Query],
                   region_min: np.ndarray = None, region_max: np.ndarray = None,
                   depth: int = 0, debug: bool = False) -> TreeNode:
        """Recursively build multi-dimensional spatial partitioning tree"""
        if region_min is None or region_max is None:
            points_array = np.array(points)
            region_min = np.min(points_array, axis=0)
            region_max = np.max(points_array, axis=0)

        if debug:
            n_dims = len(region_min)
            print(f"{'  ' * depth}Building {n_dims}D node at depth {depth}, points: {len(points)}")

        relevant_queries = self.filter_queries_for_region_multidim(queries, region_min, region_max)
        node = TreeNode(points, region_min, region_max, depth, relevant_queries)

        if self.should_stop_splitting(node):
            if debug:
                print(f"{'  ' * depth}Leaf node created")
            return node

        split_dim = self.select_split_dimension_multidim(points)
        split_result = self.select_split_value(points, split_dim, relevant_queries, depth)

        if split_result is None or split_result[1]["best_cost"] == float('inf'):
            if debug:
                print(f"{'  ' * depth}Cannot find valid split, creating leaf")
            return node

        split_value, search_info = split_result
        left_points, right_points = self.split_points_multidim(points, split_dim, split_value)

        if len(left_points) == 0 or len(right_points) == 0:
            if debug:
                print(f"{'  ' * depth}Split results in empty child, creating leaf")
            return node

        node.split_dim = split_dim
        node.split_value = split_value
        node.is_leaf = False
        node.cost_info = search_info["cost_details"]
        node.search_info = search_info

        # Create sub-regions
        left_region_min = region_min.copy()
        left_region_max = region_max.copy()
        left_region_max[split_dim] = split_value

        right_region_min = region_min.copy()
        right_region_max = region_max.copy()
        # right_region_min[split_dim] = split_value
        right_region_min[split_dim] = split_value+1

        # Recursively build subtrees
        node.left_child = self.build_tree(left_points, queries, left_region_min,
                                          left_region_max, depth + 1, debug)
        node.right_child = self.build_tree(right_points, queries, right_region_min,
                                           right_region_max, depth + 1, debug)

        return node

    def collect_leaf_nodes(self, node: TreeNode) -> List[TreeNode]:
        """Collect all leaf nodes"""
        if node is None:
            return []

        if node.is_leaf:
            return [node]

        leaves = []
        if node.left_child is not None:
            leaves.extend(self.collect_leaf_nodes(node.left_child))
        if node.right_child is not None:
            leaves.extend(self.collect_leaf_nodes(node.right_child))
        return leaves

    def print_tree(self, node: TreeNode, prefix: str = ""):
        """Print multi-dimensional tree structure"""
        if node is None:
            return

        print(f"{prefix}Node: depth={node.depth}, points={len(node.points)}, dims={node.n_dims}, "
              f"region=[{node.region_min}, {node.region_max}]")

        if not node.is_leaf:
            print(f"{prefix}  Split: dim={node.split_dim}, value={node.split_value:.2f}")
            if hasattr(node, 'search_info'):
                info = node.search_info
                print(f"{prefix}  Buckets: {info['buckets_evaluated']}, Cost: {info['best_cost']:.3f}")

            print(f"{prefix}  Left:")
            self.print_tree(node.left_child, prefix + "    ")
            print(f"{prefix}  Right:")
            self.print_tree(node.right_child, prefix + "    ")
        else:
            print(f"{prefix}  Leaf node")
            if hasattr(node, 'sfc_performance') and node.sfc_performance:
                print(f"{prefix}  SFC Performance: {node.sfc_performance:.3f}")

    def get_tree_statistics(self, root: TreeNode) -> Dict:
        """Get tree statistics"""
        if root is None:
            return {}

        leaf_nodes = self.collect_leaf_nodes(root)
        depths = [node.depth for node in leaf_nodes]
        point_counts = [len(node.points) for node in leaf_nodes]

        total_points = sum(point_counts)

        return {
            'total_nodes': len(leaf_nodes),
            'total_points': total_points,
            'n_dimensions': leaf_nodes[0].n_dims if leaf_nodes else 0,
            'max_depth': max(depths) if depths else 0,
            'min_depth': min(depths) if depths else 0,
            'avg_depth': np.mean(depths) if depths else 0,
            'max_points_per_node': max(point_counts) if point_counts else 0,
            'min_points_per_node': min(point_counts) if point_counts else 0,
            'avg_points_per_node': np.mean(point_counts) if point_counts else 0,
            'leaf_nodes': leaf_nodes
        }


# New testing and utility functions
def create_multidim_sample_data(n_dimensions: int, num_points: int = 1000, num_queries: int = 50,
                                region_size: int = 1000, random_seed: int = 42):
    """Create multi-dimensional sample data for testing"""
    np.random.seed(random_seed)

    print(f"Creating {n_dimensions}D sample data: {num_points} points, {num_queries} queries...")

    # Create multi-dimensional point data
    points = []
    for _ in range(num_points):
        point = np.random.randint(0, region_size, n_dimensions)
        points.append(point)

    # Create multi-dimensional queries
    queries = []
    for _ in range(num_queries):
        # Randomly select query center
        center = np.random.randint(region_size // 10, region_size * 9 // 10, n_dimensions)

        # Random query size (50-200 per dimension)
        query_sizes = np.random.randint(50, 200, n_dimensions)

        # Calculate query boundaries
        query_min = np.maximum(0, center - query_sizes // 2)
        query_max = np.minimum(region_size, center + query_sizes // 2)

        queries.append(Query(query_min.tolist(), query_max.tolist(), weight=1.0))

    print(f"Created {len(points)} {n_dimensions}D points and {len(queries)} {n_dimensions}D queries")
    return points, queries


def test_multidim_spatial_partitioning():
    """Test multi-dimensional spatial partitioning functionality"""
    print("=== Multi-dimensional Spatial Partitioning Test ===")

    # Test 2D (Backward Compatibility)
    print("\n1. Test 2D Spatial Partitioning (Backward Compatibility):")
    points_2d, queries_2d = create_multidim_sample_data(2, 100, 10)

    partitioner_2d = MultiLevelSpatialPartitioning(
        min_points=10, max_depth=3, num_buckets=5, bucket_levels=2
    )

    root_2d = partitioner_2d.build_tree(points_2d, queries_2d, debug=True)
    stats_2d = partitioner_2d.get_tree_statistics(root_2d)

    print(f"2D Tree Stats: {stats_2d['total_nodes']} leaf nodes, "
          f"max depth {stats_2d['max_depth']}, "
          f"avg {stats_2d['avg_points_per_node']:.1f} points per node")

    # Test 3D
    print("\n2. Test 3D Spatial Partitioning:")
    points_3d, queries_3d = create_multidim_sample_data(3, 100, 10)

    partitioner_3d = MultiLevelSpatialPartitioning(
        min_points=10, max_depth=3, num_buckets=5, bucket_levels=2
    )

    root_3d = partitioner_3d.build_tree(points_3d, queries_3d, debug=True)
    stats_3d = partitioner_3d.get_tree_statistics(root_3d)

    print(f"3D Tree Stats: {stats_3d['total_nodes']} leaf nodes, "
          f"max depth {stats_3d['max_depth']}, "
          f"avg {stats_3d['avg_points_per_node']:.1f} points per node")

    # Test 4D
    print("\n3. Test 4D Spatial Partitioning:")
    points_4d, queries_4d = create_multidim_sample_data(4, 100, 10)

    partitioner_4d = MultiLevelSpatialPartitioning(
        min_points=10, max_depth=3, num_buckets=5, bucket_levels=2
    )

    root_4d = partitioner_4d.build_tree(points_4d, queries_4d, debug=True)
    stats_4d = partitioner_4d.get_tree_statistics(root_4d)

    print(f"4D Tree Stats: {stats_4d['total_nodes']} leaf nodes, "
          f"max depth {stats_4d['max_depth']}, "
          f"avg {stats_4d['avg_points_per_node']:.1f} points per node")

    # Test query intersection
    print("\n4. Test Multi-dimensional Query Intersection:")
    test_query_3d = Query([100, 200, 300], [150, 250, 350])
    test_region_min = np.array([120, 180, 320])
    test_region_max = np.array([180, 280, 380])

    intersects = test_query_3d.intersects_region(test_region_min, test_region_max)
    overlap_volume = test_query_3d.get_overlap_volume(test_region_min, test_region_max)

    print(f"3D Query intersects region: {intersects}")
    print(f"Overlap Volume: {overlap_volume}")

    print("\n=== Multi-dimensional Spatial Partitioning Test Completed ===")


def validate_multidim_tree(root: TreeNode, original_points: List[np.ndarray]):
    """Validate multi-dimensional tree correctness"""
    print("Validating multi-dimensional tree structure...")

    # Collect points from all leaf nodes
    leaf_nodes = []

    def collect_points(node):
        if node.is_leaf:
            leaf_nodes.extend(node.points)
        else:
            if node.left_child:
                collect_points(node.left_child)
            if node.right_child:
                collect_points(node.right_child)

    collect_points(root)

    # Validate point count consistency
    original_count = len(original_points)
    collected_count = len(leaf_nodes)

    if original_count == collected_count:
        print(f"✓ Point count validation passed: {original_count} == {collected_count}")
    else:
        print(f"✗ Point count validation failed: {original_count} != {collected_count}")

    # Validate dimension consistency
    if original_points and leaf_nodes:
        orig_dim = len(original_points[0])
        collected_dim = len(leaf_nodes[0])

        if orig_dim == collected_dim:
            print(f"✓ Dimension validation passed: {orig_dim}D")
        else:
            print(f"✗ Dimension validation failed: {orig_dim} != {collected_dim}")


if __name__ == "__main__":
    test_multidim_spatial_partitioning()