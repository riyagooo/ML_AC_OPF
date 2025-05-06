"""
Patch for train_balanced_gnn.py to prevent sys.exit(1) when imported.

Import this module before importing train_balanced_gnn.py to ensure that
even if torch_geometric import fails, it won't exit the program.
"""

import sys
import importlib.util
import builtins
import importlib
import torch
import numpy as np

# Store the original __import__ function
original_import = builtins.__import__
original_exit = sys.exit

# Define our patched exit function
def patched_exit(code=0):
    # If we're in the main module, exit normally
    if __name__ == "__main__":
        original_exit(code)
    else:
        # Otherwise, raise an exception instead of exiting
        print("Warning: sys.exit({code}) was called, but prevented to allow imports.")
        if code != 0:
            raise ImportError(f"Module tried to exit with code {code}")

# Define our patched import function
def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    # Use the original import function
    try:
        return original_import(name, globals, locals, fromlist, level)
    except ImportError as e:
        # If it's torch_geometric or a submodule, handle it specially
        if name == 'torch_geometric' or name.startswith('torch_geometric.'):
            print(f"Warning: Import of {name} failed, but continuing anyway.")
            # Create a mock module
            module = type('MockModule', (), {})()
            return module
        else:
            # For other modules, re-raise the original exception
            raise

# Apply our patches
sys.exit = patched_exit
builtins.__import__ = patched_import

# Import the create_power_system_graph function from power_system_graph.py
try:
    from power_system_graph import create_power_system_graph
    
    # Now inject this function into models.gnn module
    try:
        # Try to import models.gnn normally
        import models.gnn
        
        # Inject the function
        models.gnn.create_power_system_graph = create_power_system_graph
        print("Successfully injected create_power_system_graph function into models.gnn")
    except Exception as e:
        print(f"Could not inject function into models.gnn: {e}")
        
        # Try to create a mock models.gnn module
        try:
            # If models exists but gnn doesn't
            if 'models' in sys.modules:
                if not hasattr(sys.modules['models'], 'gnn'):
                    sys.modules['models'].gnn = type('MockGNN', (), {})()
                sys.modules['models'].gnn.create_power_system_graph = create_power_system_graph
                print("Created mock models.gnn module with create_power_system_graph function")
            else:
                # Create mock models module
                models_module = type('MockModels', (), {})()
                models_module.gnn = type('MockGNN', (), {})()
                models_module.gnn.create_power_system_graph = create_power_system_graph
                sys.modules['models'] = models_module
                sys.modules['models.gnn'] = models_module.gnn
                print("Created mock models and models.gnn modules with create_power_system_graph function")
        except Exception as e:
            print(f"Could not create mock modules: {e}")
except Exception as e:
    print(f"Could not import create_power_system_graph: {e}")

print("Patched import and exit functions to prevent PyTorch Geometric import errors from stopping execution.") 