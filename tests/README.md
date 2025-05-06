# ML-OPF Tests

This directory contains tests for the ML-OPF project.

## Test Organization

- `test_model.py`: Tests for neural network models
- `test_training.py`: Tests for training utilities
- `test_evaluation.py`: Tests for evaluation metrics
- `test_evaluation_features.py`: Tests for additional evaluation features
- `test_ml_data_loading.py`: Tests for data loading
- `test_opf_module.py`: Tests for the OPF optimization module
- `test_gurobi.py`: Tests for Gurobi solver integration

### Case Tests

The `case_tests` directory contains tests for specific power system cases:

- `test_case30.py`: Tests for the 30-bus system (case30)

## Running Tests

To run all tests:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/test_model.py
```

To run tests for a specific case:

```bash
pytest tests/case_tests/test_case30.py
``` 