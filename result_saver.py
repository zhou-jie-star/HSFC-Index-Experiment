import json
import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any
import numpy as np


class ExperimentResultSaver:
    """Saves HSFC experiment results only"""

    def __init__(self, base_dir: str = "hsfc_results"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save_experiment_results(self, results: Dict[str, Any], filename_prefix: str):
        """Save JSON results and CSV performance summary"""
        print(f"\nSaving results to {self.base_dir}...")

        # 1. Save CSV summary
        self._save_summary_csv(results, f"{filename_prefix}_summary.csv")

    def _save_json(self, results: Dict, filename: str):
        filepath = os.path.join(self.base_dir, filename)
        clean_data = self._clean_for_json(results)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(clean_data, f, indent=2, ensure_ascii=False)
            print(f"✓ JSON saved: {filename}")
        except Exception as e:
            print(f"✗ JSON save failed: {e}")

    def _save_summary_csv(self, results: Dict, filename: str):
        filepath = os.path.join(self.base_dir, filename)
        if 'hsfc_results' not in results:
            return

        res = results['hsfc_results']
        metrics = res['performance_metrics']

        data = [{
            'Timestamp': datetime.now().isoformat(),
            'Dimensions': res['n_dimensions'],
            'Subspaces': res['total_subspaces'],
            'Total_Points': res['total_points_indexed'],
            'Training_Time_s': res['total_training_time'],
            'Avg_Query_Time_s': metrics['avg_query_time'],
            'Split_Time_s': metrics['query_split_time'],
            'Scan_Time_s': metrics['scan_time'],
            'False_Positive_Rate': res['avg_false_positive_rate']
        }]

        try:
            pd.DataFrame(data).to_csv(filepath, index=False)
            print(f"✓ CSV saved: {filename}")
        except Exception as e:
            print(f"✗ CSV save failed: {e}")

    def _clean_for_json(self, data: Any) -> Any:
        # Simple recursive cleaning, handling numpy types
        if isinstance(data, dict):
            return {k: self._clean_for_json(v) for k, v in data.items() if k not in ['leaf_nodes']}
        elif isinstance(data, list):
            return [self._clean_for_json(i) for i in data]
        elif isinstance(data, (np.integer, int)):
            return int(data)
        elif isinstance(data, (np.floating, float)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif hasattr(data, '__dict__'):
            return str(type(data).__name__)
        return data