import timeit
from typing import List

import numpy as np
from skopt import Optimizer
from skopt.space import Real
import pickle
import os
from sfc_utils import get_theta, create_sfc_multidim, merge_sort_with_custom_order, get_dimension_count, \
    generate_multidim_theta_structure, merge_sort_with_custom_order_multidim
from query_utils import objective_function_with_queries_multidim, evaluate_sfc_performance_multidim, \
    QueryWorkloadManager, convert_query_to_multidim_format
import time
from query_utils import QuerySplitter, apply_query_splitting


class SFCOptimizer:
    """Multi-dimensional SFC Optimizer"""

    def __init__(self, num_iterations: int = 100, n_initial_points: int = 20,
                 use_query_splitting: bool = True, bit_length: int = 10, train_ratio: float = 0.1,
                 data_sample_ratio: float = 1.0):
        self.num_iterations = num_iterations
        self.n_initial_points = n_initial_points
        self.bit_length = bit_length
        self.use_query_splitting = use_query_splitting
        self.train_ratio = train_ratio
        self.data_sample_ratio = data_sample_ratio
        self.query_manager = QueryWorkloadManager(train_ratio=train_ratio)

    def sample_data_points(self, data_points: List[np.ndarray],
                           sample_ratio: float = None) -> List[np.ndarray]:
        if sample_ratio is None:
            sample_ratio = self.data_sample_ratio
        if sample_ratio >= 1.0:
            return data_points

        n_total = len(data_points)
        n_sample = max(1, int(n_total * sample_ratio))
        np.random.seed(42)
        sampled_indices = np.random.choice(n_total, n_sample, replace=False)
        sampled_points = [data_points[i] for i in sorted(sampled_indices)]
        return sampled_points

    def create_search_space_multidim(self, n_dimensions: int):
        search_space = []
        for dim in range(n_dimensions):
            dim_space = [Real(0, 1, name=f'theta_{dim}_{i}') for i in range(self.bit_length)]
            search_space.extend(dim_space)
        return search_space

    def create_optimizer(self, space):
        return Optimizer(
            dimensions=space,
            base_estimator='RF',
            acq_func="LCB",
            random_state=42,
            n_initial_points=self.n_initial_points
        )

    def prepare_queries_for_subspace_multidim(self, original_queries, data_points, region_min, region_max,
                                              n_dimensions):
        relevant_queries = []
        for query in original_queries:
            try:
                if hasattr(query, 'min_point') and hasattr(query, 'max_point'):
                    q_min = query.min_point
                    q_max = query.max_point
                else:
                    q_min, q_max = convert_query_to_multidim_format(query, n_dimensions)

                if not np.any(q_max < region_min) and not np.any(q_min > region_max):
                    clipped_min = np.maximum(q_min, region_min)
                    clipped_max = np.minimum(q_max, region_max)
                    clipped_min_int = np.ceil(clipped_min).astype(int)
                    clipped_max_int = np.floor(clipped_max).astype(int)

                    if np.all(clipped_min_int <= clipped_max_int):
                        clipped_query = (tuple(clipped_min_int), tuple(clipped_max_int))
                        relevant_queries.append(clipped_query)
            except Exception:
                continue

        if len(relevant_queries) == 0:
            return [], []

        if self.use_query_splitting and len(relevant_queries) >= 4:
            train_queries, test_queries = self.query_manager.split_queries(relevant_queries)
        else:
            train_queries, test_queries = self.query_manager.split_queries(relevant_queries)

        return train_queries, test_queries

    def optimize_sfc_for_subspace_multidim(self, data_points, region_min, region_max,
                                           original_queries, subspace_id: str = "default", debug: bool = False):
        if len(data_points) == 0:
            return None, None, float('inf'), {}

        n_dimensions = get_dimension_count(data_points)
        sampled_data_points = self.sample_data_points(data_points, self.data_sample_ratio)

        train_queries, test_queries = self.prepare_queries_for_subspace_multidim(
            original_queries, data_points, region_min, region_max, n_dimensions
        )

        if len(train_queries) == 0 and len(test_queries) == 0:
            return None, None, float('inf'), {'skipped': True, 'reason': 'no_queries'}

        space = self.create_search_space_multidim(n_dimensions)
        optimizer = self.create_optimizer(space)

        def objective(theta_values):
            theta_values_sorted = merge_sort_with_custom_order_multidim(
                theta_values.copy(), n_dimensions=n_dimensions, bit_length=self.bit_length
            )
            theta = self.reconstruct_theta_from_values_multidim(theta_values_sorted, n_dimensions)
            curve = create_sfc_multidim(sampled_data_points, theta)
            return objective_function_with_queries_multidim(
                sampled_data_points, region_min, region_max, theta, theta_values_sorted, curve, train_queries
            )

        best_performance = float('inf')
        best_theta = None
        best_theta_values = None

        train_start = time.time()
        for i in range(self.num_iterations):
            next_point = optimizer.ask()
            f_val = objective(next_point)

            if f_val < best_performance:
                best_performance = f_val
                best_theta_values = merge_sort_with_custom_order_multidim(
                    next_point.copy(), n_dimensions=n_dimensions, bit_length=self.bit_length
                )
                best_theta = self.reconstruct_theta_from_values_multidim(best_theta_values, n_dimensions)

            optimizer.tell(next_point, f_val)

        subspace_training_time = time.time() - train_start

        if best_theta is not None:
            best_curve = create_sfc_multidim(data_points, best_theta)
            all_queries = train_queries + test_queries

            start_time = timeit.default_timer()
            split_queries = apply_query_splitting(all_queries, best_theta, k_maxsplit=3, debug=debug)
            end_time = timeit.default_timer()
            total_query_split_time = (end_time - start_time)

            validation_results = evaluate_sfc_performance_multidim(
                data_points, region_min, region_max, best_theta,
                best_theta_values, best_curve, split_queries
            )

            scan_time = validation_results['total_time']
            total_query_time = scan_time + total_query_split_time

            validation_results['scan_time'] = scan_time
            validation_results['query_split_time'] = total_query_split_time
            validation_results['total_time'] = total_query_time
            validation_results['avg_query_time'] = total_query_time / len(all_queries) if len(all_queries) > 0 else 0

            detailed_results = {
                'training_performance': best_performance,
                'training_time': subspace_training_time,  # [Fix] Added training time
                'validation_results': validation_results,
                'performance_metrics': {
                    'query_split_time': total_query_split_time,
                    'scan_time': scan_time,
                    'total_query_time': total_query_time
                }
            }
            validation_performance = validation_results['total_time']

        else:
            validation_performance = float('inf')
            detailed_results = {'n_dimensions': n_dimensions}

        return best_theta, best_theta_values, validation_performance, detailed_results

    def reconstruct_theta_from_values_multidim(self, theta_values_sorted, n_dimensions):
        values_per_dim = len(theta_values_sorted) // n_dimensions
        theta = []
        for dim in range(n_dimensions):
            start_idx = dim * values_per_dim
            end_idx = (dim + 1) * values_per_dim
            dim_values = np.array(theta_values_sorted[start_idx:end_idx])
            theta.append(get_theta(dim_values))
        return np.array(theta)


class SubspaceSFCManager:
    """Multi-dimensional Subspace SFC Manager"""

    def __init__(self, sfc_optimizer: SFCOptimizer):
        self.sfc_optimizer = sfc_optimizer
        self.subspace_sfcs = {}

    def optimize_all_subspaces_multidim(self, leaf_nodes, original_queries, debug: bool = False):
        if debug:
            print(f"Starting SFC optimization for {len(leaf_nodes)} subspaces")

        for i, node in enumerate(leaf_nodes):
            if node.points:
                n_dims = get_dimension_count(node.points)
                subspace_id = f"subspace_{i}_depth_{node.depth}_dims_{n_dims}"
            else:
                subspace_id = f"subspace_{i}_depth_{node.depth}"


            best_theta, best_theta_values, validation_performance, detailed_results = self.sfc_optimizer.optimize_sfc_for_subspace_multidim(
                node.points, node.region_min, node.region_max, original_queries, subspace_id, debug
            )

            if best_theta is not None:
                best_curve = create_sfc_multidim(node.points, best_theta)
                node.sfc_theta = best_theta
                node.sfc_curve = best_curve
                node.sfc_performance = validation_performance
                node.detailed_results = detailed_results

                self.subspace_sfcs[subspace_id] = {
                    'theta': best_theta,
                    'theta_values': best_theta_values,
                    'curve': best_curve,
                    'performance': validation_performance,
                    'detailed_results': detailed_results,
                    'node': node
                }

        if debug:
            print(f"\n{'=' * 60}")
            print("All subspace SFC optimization processes completed.")

    def print_optimization_summary_multidim(self):
        pass
    def print_optimization_summary(self):

        return self.print_optimization_summary_multidim()

    def get_subspace_sfc(self, subspace_id: str):
        """Get Subspace SFC"""
        return self.subspace_sfcs.get(subspace_id)

    def save_results(self, filename: str):
        """Save optimization results"""
        save_data = {}
        for subspace_id, info in self.subspace_sfcs.items():
            detailed = info.get('detailed_results', {})

            save_data[subspace_id] = {
                'theta': info['theta'].tolist() if info['theta'] is not None else None,
                'theta_values': info['theta_values'],
                'validation_performance': info['performance'],
                'training_performance': detailed.get('training_performance', None),
                'validation_results': detailed.get('validation_results', {}),
                'region_min': info['node'].region_min.tolist(),
                'region_max': info['node'].region_max.tolist(),
                'points_count': len(info['node'].points),
                'n_dimensions': detailed.get('n_dimensions', 0),
                'query_counts': {
                    'train': detailed.get('train_query_count', 0),
                    'test': detailed.get('test_query_count', 0),
                }
            }

        with open(filename, 'w') as f:
            import json
            json.dump(save_data, f, indent=2)

        print(f"Multi-dimensional optimization results saved to {filename}")



def test_multidim_sfc_optimization():
    """Test multi-dimensional SFC optimization functionality"""
    print("=== Multi-dimensional SFC Optimization Test ===")

    # Create test data
    from spatial_partitioning import create_multidim_sample_data

    # Test 2D optimization
    print("\n1. Test 2D SFC optimization:")
    points_2d, queries_2d = create_multidim_sample_data(2, 50, 5)

    optimizer_2d = SFCOptimizer(num_iterations=10, n_initial_points=5)
    region_min_2d = np.min(points_2d, axis=0)
    region_max_2d = np.max(points_2d, axis=0)

    theta_2d, _, perf_2d, results_2d = optimizer_2d.optimize_sfc_for_subspace_multidim(
        points_2d, region_min_2d, region_max_2d, queries_2d, "test_2d", debug=True
    )

    if theta_2d is not None:
        print(f"2D optimization successful, performance: {perf_2d:.3f}")

    # Test 3D optimization
    print("\n2. Test 3D SFC optimization:")
    points_3d, queries_3d = create_multidim_sample_data(3, 50, 5)

    optimizer_3d = SFCOptimizer(num_iterations=10, n_initial_points=5)
    region_min_3d = np.min(points_3d, axis=0)
    region_max_3d = np.max(points_3d, axis=0)

    theta_3d, _, perf_3d, results_3d = optimizer_3d.optimize_sfc_for_subspace_multidim(
        points_3d, region_min_3d, region_max_3d, queries_3d, "test_3d", debug=True
    )

    if theta_3d is not None:
        print(f"3D optimization successful, performance: {perf_3d:.3f}")

    print("\n=== Multi-dimensional SFC Optimization Test Completed ===")


if __name__ == "__main__":
    test_multidim_sfc_optimization()