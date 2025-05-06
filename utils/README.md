# Utility Modules

This directory contains utility modules for the ML-AC-OPF project.

## Module Overview

- **case39_utils.py** - Utilities for IEEE 39-bus (New England) system. Recommended for numerical stability.
- **data_utils.py** - General data processing utilities for power system data
- **optimization.py** - Base optimization utilities
- **optimization_improved.py** - Enhanced optimization utilities with better numerical properties
- **training.py** - Utilities for model training
- **metrics.py** - Functions for evaluating model performance
- **evaluation.py** - Comprehensive model evaluation tools
- **power_system_validation.py** - Validation utilities for power flow results
- **robust_data.py** - Utilities for handling noisy and out-of-distribution data
- **error_handling.py** - Error handling and reporting utilities
- **config.py** - Configuration utilities

## Case39 Utilities

The `case39_utils.py` module provides specialized functions for working with the IEEE 39-bus system, which offers better numerical stability compared to case30. Key functions include:

- `load_case39()` - Load the IEEE 39-bus test case
- `create_case39_graph()` - Create a NetworkX graph representation
- `pyg_graph_from_case39()` - Create a PyTorch Geometric Data object
- `generate_case39_scenarios()` - Generate synthetic scenarios for ML training
- `get_case39_bounds()` - Get variable bounds for case39
- `case39_to_tensors()` - Convert case39 data to tensors for ML models
- `denormalize_case39_outputs()` - Denormalize model outputs

For more details on transitioning from case30 to case39, see `docs/CASE39_TRANSITION.md`. 