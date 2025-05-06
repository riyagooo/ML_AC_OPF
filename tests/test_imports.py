#!/usr/bin/env python
"""
Test file to diagnose PyTorch Geometric import issues
"""

import sys
import os
import importlib.util

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("\nPython path:")
for p in sys.path:
    print(f"  - {p}")

# Check torch installation
print("\nTorch installation:")
try:
    import torch
    print(f"  - Found torch version: {torch.__version__}")
    print(f"  - Torch location: {torch.__file__}")
except ImportError as e:
    print(f"  - Error importing torch: {e}")

# Check torch_geometric installation
print("\nTorch Geometric installation:")
try:
    import torch_geometric
    print(f"  - Found torch_geometric version: {torch_geometric.__version__}")
    print(f"  - Torch Geometric location: {torch_geometric.__file__}")
except ImportError as e:
    print(f"  - Error importing torch_geometric: {e}")

# Check torch_geometric dependencies
print("\nTorch Geometric dependencies:")
for module in ["torch_scatter", "torch_sparse", "torch_cluster"]:
    try:
        spec = importlib.util.find_spec(module)
        if spec is not None:
            m = importlib.import_module(module)
            print(f"  - {module}: Found at {m.__file__}")
        else:
            print(f"  - {module}: Not found (but module exists)")
    except ImportError as e:
        print(f"  - {module}: Error - {e}")

# Check if models.gnn can be imported
print("\nModels GNN module:")
try:
    from models.gnn import EnhancedDirectPredictionGNN
    print(f"  - Successfully imported EnhancedDirectPredictionGNN from models.gnn")
except ImportError as e:
    print(f"  - Error importing from models.gnn: {e}")
    # Try to locate the models directory
    models_dir = os.path.join(os.getcwd(), "models")
    if os.path.exists(models_dir):
        print(f"  - Models directory exists at: {models_dir}")
        print(f"  - Files in models directory: {os.listdir(models_dir)}")
    else:
        print(f"  - Models directory not found at: {models_dir}")

print("\nConclusion:")
print("If you see 'No module named torch_geometric' but torch_geometric is installed,")
print("the issue might be with Python path or with how the modules are being imported.")
print("If models.gnn imports torch_geometric but your script doesn't use it directly,")
print("consider modifying your evaluate_domain_metrics.py to avoid importing from models.gnn.") 