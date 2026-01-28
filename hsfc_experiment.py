import numpy as np
import time
from typing import List, Dict
from sfc_optimization import SFCOptimizer, SubspaceSFCManager
from spatial_partitioning import MultiLevelSpatialPartitioning
from query_utils import QueryWorkloadManager, convert_query_to_multidim_format
from sfc_utils import get_dimension_count


class HSFCExperiment:
    """
    HSFC Experiment Controller
    Runs only the Hybrid Space-Filling Curve (HSFC) related experimental logic:
    1. MSP-Tree Spatial Partitioning
    2. Subspace Local SFC Learning
    3. Query Splitting and Evaluation
    """

    def __init__(self, num_iterations: int = 50, n_initial_points: int = 10,
                 train_ratio: float = 0.1, data_sample_ratio: float = 1.0):
        self.num_iterations = num_iterations
        self.n_initial_points = n_initial_points
        self.train_ratio = train_ratio
        self.data_sample_ratio = data_sample_ratio
        self.results = {}

    def run_experiment(self, points: List[np.ndarray], queries: List,
                       partitioning_config: Dict = None, debug: bool = True) -> Dict:
        """
        Run HSFC Experiment
        """
        # Get data dimensions
        n_dimensions = get_dimension_count(points)

        if debug:
            print("=" * 80)
            print(f"Running HSFC (Hybrid Space-Filling Curve) Experiment - {n_dimensions}D")
            print("=" * 80)
            print(f"Data points: {len(points)}")
            print(f"Queries: {len(queries)}")

        # --- Core Logic: Run Subspace Optimization (HSFC) ---
        start_time = time.time()
        hsfc_results = self._run_subspace_optimization(points, queries, partitioning_config, n_dimensions, debug)
        total_time = time.time() - start_time

        if 'error' in hsfc_results:
            print(f"HSFC Experiment Failed: {hsfc_results['error']}")
            return {'error': hsfc_results['error']}

        hsfc_results['total_execution_time'] = total_time

        # Save results
        self.results = {
            'hsfc_results': hsfc_results,
            'experiment_config': {
                'num_iterations': self.num_iterations,
                'n_initial_points': self.n_initial_points,
                'data_points_count': len(points),
                'queries_count': len(queries),
                'n_dimensions': n_dimensions,
                'partitioning_config': partitioning_config
            }
        }

        return self.results

    def _run_subspace_optimization(self, points: List[np.ndarray], queries: List,
                                   config: Dict = None, n_dimensions: int = None, debug: bool = True) -> Dict:
        """
        Execute the complete HSFC workflow: Build MSP-Tree -> Bayesian Optimization for local SFC -> Evaluation
        """
        try:
            if n_dimensions is None:
                n_dimensions = get_dimension_count(points)

            # Default MSP-Tree configuration
            if config is None:
                config = {
                    'min_points': min(10000, len(points) // 2),
                    'max_depth': 3,
                    'num_buckets': 20,
                    'bucket_levels': 2
                }

            # 1. Build MSP-Tree (Multi-level Spatial Partitioning)
            if debug:
                print("\n[Step 1] Building MSP-Tree...")
            partitioner = MultiLevelSpatialPartitioning(**config)
            root = partitioner.build_tree(points, queries, debug=debug)
            leaf_nodes = partitioner.collect_leaf_nodes(root)

            if debug:
                print(f"  MSP-Tree generated {len(leaf_nodes)} subspaces (leaf nodes).")

            # 2. Initialize SFC Optimizer
            sfc_optimizer = SFCOptimizer(
                num_iterations=self.num_iterations,
                n_initial_points=self.n_initial_points,
                use_query_splitting=True,  # Enable query splitting
                bit_length=20,
                train_ratio=self.train_ratio,
                data_sample_ratio=self.data_sample_ratio
            )
            sfc_manager = SubspaceSFCManager(sfc_optimizer)

            # 3. Learn Local SFCs for each subspace
            if debug:
                print(f"\n[Step 2] Learning Local SFCs for subspaces...")

            sfc_manager.optimize_all_subspaces_multidim(leaf_nodes, queries, debug=debug)

            # 4. Collect statistical results
            subspace_performances = []
            total_points = 0
            total_validation_time = 0
            total_false_positives = 0
            total_points_found = 0
            skipped_subspaces = 0

            # Performance metrics statistics
            total_query_split_time = 0.0
            total_scan_time = 0.0
            total_query_count = 0
            total_training_time = 0.0

            for node in leaf_nodes:
                if hasattr(node, 'detailed_results') and node.detailed_results.get('skipped', False):
                    skipped_subspaces += 1
                    continue

                if hasattr(node, 'sfc_performance') and node.sfc_performance != float('inf'):
                    subspace_performances.append(node.sfc_performance)
                    total_points += len(node.points)

                    if hasattr(node, 'detailed_results'):
                        dr = node.detailed_results
                        val_results = dr.get('validation_results', {})
                        pm = dr.get('performance_metrics', {})

                        total_validation_time += val_results.get('total_time', 0)
                        total_false_positives += val_results.get('false_positives', 0)
                        total_points_found += val_results.get('total_points', 0)
                        total_training_time += dr.get('training_time', 0.0)

                        # Statistics for splitting and scanning time
                        total_query_split_time += pm.get('query_split_time', val_results.get('query_split_time', 0))
                        total_scan_time += pm.get('scan_time', val_results.get('scan_time', 0))
                        total_query_count += pm.get('total_queries', val_results.get('query_count', 0))

            # Calculate average metrics
            avg_false_positive_rate = (total_false_positives /
                                       max(1, total_points_found + total_false_positives))
            avg_query_time = total_validation_time / total_query_count if total_query_count > 0 else 0

            results = {
                'method': f'HSFC (Subspace {n_dimensions}D)',
                'n_dimensions': n_dimensions,
                'total_subspaces': len(leaf_nodes),
                'optimized_subspaces': len(leaf_nodes) - skipped_subspaces,
                'total_points_indexed': total_points,
                'avg_false_positive_rate': avg_false_positive_rate,
                'total_false_positives': total_false_positives,
                'total_training_time': total_training_time,
                'performance_metrics': {
                    'query_split_time': total_query_split_time,
                    'scan_time': total_scan_time,
                    'total_query_time': total_validation_time,
                    'avg_query_time': avg_query_time,
                    'total_queries': total_query_count
                },
                # Keep tree info for debugging, be careful when serializing
                'tree_stats': partitioner.get_tree_statistics(root)
            }

            if debug:
                print(f"\n[HSFC Summary]")
                print(f"  Total Subspaces: {len(leaf_nodes)}")
                print(f"  Optimized: {len(leaf_nodes) - skipped_subspaces}")
                print(f"  Avg Query Time: {avg_query_time:.6f}s")
                print(f"  Avg False Positive Rate: {avg_false_positive_rate:.4f}")

            return results

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'error': f'HSFC optimization failed: {str(e)}'}

    def generate_report(self) -> str:
        """Generate HSFC Experiment Report"""
        if not self.results:
            return "No results available."

        res = self.results['hsfc_results']
        metrics = res['performance_metrics']
        config = self.results['experiment_config']

        report = []
        report.append("=" * 60)
        report.append(f"HSFC Experiment Report ({res['n_dimensions']}D)")
        report.append("=" * 60)

        report.append(f"\nConfiguration:")
        report.append(f"  Data Points: {config['data_points_count']}")
        report.append(f"  Queries: {config['queries_count']}")
        report.append(f"  Iterations: {config['num_iterations']}")

        report.append(f"\nPerformance:")
        report.append(f"  Total Training Time: {res['total_training_time']:.2f}s")
        report.append(f"  Total Query Processing Time: {metrics['total_query_time']:.4f}s")
        report.append(f"    - Query Splitting Time: {metrics['query_split_time']:.4f}s")
        report.append(f"    - Index Scanning Time: {metrics['scan_time']:.4f}s")
        report.append(f"  Average Query Latency: {metrics['avg_query_time']:.6f}s")
        report.append(f"  False Positive Rate: {res['avg_false_positive_rate']:.4f}")
        report.append(f"  Total False Positives: {res['total_false_positives']}")

        report.append(f"\nStructure:")
        report.append(f"  Total Subspaces: {res['total_subspaces']}")

        return "\n".join(report)