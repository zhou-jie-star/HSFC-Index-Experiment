
# An Index Structure based on Learned Hybrid Space-Filling Curves

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

This repository contains the official implementation of the experimental framework for the paper: **"An Index Structure based on Learned Hybrid Space-Filling Curves"**.

## 📖 Introduction

**HSFC** (Hybrid Space-Filling Curve) is a learned multidimensional indexing framework designed to address the limitations of fixed mapping strategies (like Z-curve) and single global learned mappings. HSFC combines **Multi-level Spatial Partitioning Tree (MSP-Tree)** with **Subspace-level Curve Optimization** to adapt to complex data distributions and diverse query workloads.

Key components of this framework include:
**MSP-Tree:** A query-aware spatial partitioning tree that recursively divides the global data space into localized subspaces, aligning boundaries with high-frequency query regions.
**Local Learned SFCs:** Independent monotonic SFCs learned via Bayesian Optimization for each subspace.
**Recursive Query Splitting:** A proactive strategy leveraging weight pruning to decompose window queries into subqueries with tighter projection intervals, significantly reducing false positives.

Experiments show that HSFC achieves **1.5x - 8.8x** reductions in query latency and reduces false positives by **27% - 88%** compared to state-of-the-art methods.

## 📂 Project Structure

The project is structured as follows:

```text
.
├── Data/
│   └── OSMdata_large.json   # Real-world OSM dataset for evaluation
├── hsfc_experiment.py       # Core controller for HSFC experiments
├── spatial_partitioning.py  # Implementation of the MSP-Tree (Multi-Level Spatial Partitioning)
├── sfc_optimization.py      # Bayesian Optimization logic for local SFC parameter learning
├── sfc_utils.py             # Utilities for SFC calculation (Morton codes, bit interleaving)
├── query_utils.py           # Implementation of the Recursive Query Splitting strategy
├── run_hsfc.py              # Main entry point to run experiments
├── result_saver.py          # Utilities for saving logs and CSV results
└── README.md

```

## 🛠️ Requirements

The code is implemented in **Python 3.8**. To install the necessary dependencies, run:

```bash
pip install numpy pandas scikit-optimize scikit-learn

```

**Key Dependencies:**

* `numpy`: Matrix and vector operations.
* `scikit-optimize` (`skopt`): Used for Bayesian Optimization (SMBO) of SFC parameters.
* `pandas`: For data handling and result export.

## 🚀 Quick Start

To run a standard experiment using synthetic data (clustered distribution):

```bash
python run_hsfc.py

```

By default, this script will:

1. Construct the **MSP-Tree** based on data variance and query workload.
2. Perform **Bayesian Optimization** to learn optimal local SFCs for leaf nodes.
3. Execute window queries using the **Query Splitting** strategy.
4. Output the Average Query Time and False Positive Rate.


## 📊 Core Components

### 1. Multi-level Spatial Partitioning (MSP-Tree)

Located in `spatial_partitioning.py`.
The `build_tree` function implements the maximum-variance dimension selection and the composite cost function (Equation 9 in the paper) that balances query cost and partition balance.

### 2. Learned Local SFCs

Located in `sfc_optimization.py`.
The `SFCOptimizer` class uses Gaussian Process-based Bayesian Optimization to find the optimal bit-weight matrix () for each subspace, minimizing the query cost.

### 3. Query Splitting

Located in `query_utils.py`.
The `QuerySplitter` class implements the recursive splitting algorithm described in **Section 5** of the paper. It uses MSB (Most Significant Bit) alignment and weight pruning to minimize the SFC interval gap.

## 📧 Contact

If you have any questions, please feel free to contact the authors or open an issue in this repository.

* **GitHub:** [zhou-jie-star](https://www.google.com/search?q=https://github.com/zhou-jie-star)

