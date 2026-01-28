import json
import numpy as np
import os
import sys
import time
from datetime import datetime
from result_saver import ExperimentResultSaver
from hsfc_experiment import HSFCExperiment
from spatial_partitioning import Query
from sfc_utils import get_dimension_count, validate_multidim_compatibility


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


def create_sample_data(n_dimensions: int = 3, num_points: int = 5000, num_queries: int = 50,
                       region_size: int = 2 ** 15, distribution_type: str = "clustered",
                       random_seed: int = 42):
    """Create synthetic data for testing"""
    np.random.seed(random_seed)
    print(f"Generating {n_dimensions}D data ({distribution_type}): {num_points} points, {num_queries} queries...")

    points = []
    # Simple clustered distribution generation
    if distribution_type == "clustered":
        num_clusters = max(2, n_dimensions)
        centers = [np.random.randint(0, region_size, n_dimensions) for _ in range(num_clusters)]
        for _ in range(num_points):
            center = centers[np.random.randint(num_clusters)]
            offset = np.random.normal(0, region_size // 10, n_dimensions).astype(int)
            point = np.clip(center + offset, 0, region_size - 1)
            points.append(point)
    else:  # Uniform
        for _ in range(num_points):
            points.append(np.random.randint(0, region_size, n_dimensions))

    queries = []
    for _ in range(num_queries):
        center = np.random.randint(region_size // 10, region_size * 9 // 10, n_dimensions)
        sizes = np.random.randint(50, region_size // 5, n_dimensions)
        q_min = np.maximum(0, center - sizes // 2)
        q_max = np.minimum(region_size, center + sizes // 2)
        queries.append(Query(q_min.tolist(), q_max.tolist()))

    return points, queries


def run_hsfc_experiment():
    """HSFC Experiment Main Function"""
    print("=" * 80)
    print("HSFC (Hybrid Space-Filling Curve) Indexing Experiment")
    print("=" * 80)

    # 1. Setup or load data
    # Using generated data for demonstration, replace with load_data logic if needed
    n_dims = 2
    num_points = 100000
    num_queries = 2000
    points, queries = create_sample_data(n_dims, num_points, num_queries)

    try:
        validate_multidim_compatibility(points, queries)
    except ValueError as e:
        print(f"Data validation failed: {e}")
        return

    # 2. Configure parameters
    partitioning_config = {
        'min_points': min(5000, len(points) // 2),
        'max_depth': 4,
        'num_buckets': 20,
        'bucket_levels': 2,
        'penalty': 2.0,
        'alpha_start': 0.3,
        'alpha_end': 0.8
    }

    # 3. Run experiment
    experiment = HSFCExperiment(
        num_iterations=20,  # Bayesian optimization iterations
        n_initial_points=10,  # Initial points
        train_ratio=0.1  # Query sample ratio
    )

    print("\nStarting execution...")
    start_time = time.time()
    results = experiment.run_experiment(
        points=points,
        queries=queries,
        partitioning_config=partitioning_config,
        debug=True
    )

    if 'error' in results:
        return

    # 4. Output report
    print("\n" + experiment.generate_report())

    # 5. Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saver = ExperimentResultSaver(base_dir="hsfc_results")
    saver.save_experiment_results(results, f"hsfc_{n_dims}d_{timestamp}")


if __name__ == "__main__":
    run_hsfc_experiment()