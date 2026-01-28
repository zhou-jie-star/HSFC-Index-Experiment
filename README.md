# HSFC Experiment Code

This repository contains the source code for the paper **"[Insert Paper Title Here]"**.

## Overview

This project implements the Hybrid Space-Filling Curve (HSFC) indexing strategy, combining Multi-level Spatial Partitioning (MSP-Tree) with subspace-optimized Space-Filling Curves.

## File Structure

* `run_hsfc.py`: The main entry point to run the experiment.
* `hsfc_experiment.py`: Controller logic for the HSFC experiment.
* `spatial_partitioning.py`: Implementation of the MSP-Tree.
* `sfc_optimization.py`: Bayesian optimization for local SFC learning.
* `sfc_utils.py`: Utility functions for SFC calculations (Morton code, etc.).
* `query_utils.py`: Functions for query processing and splitting.
* `result_saver.py`: Helper class to save results to JSON/CSV.

## How to Run

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the main experiment:
    ```bash
    python run_hsfc.py
    ```

## Requirements
* Python 3.8+
* numpy
* scikit-optimize
* pandas
