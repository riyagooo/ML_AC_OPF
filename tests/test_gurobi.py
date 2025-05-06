#!/usr/bin/env python
"""
Simple script to test Gurobi license.
"""

import sys
import os

print("Python version:", sys.version)
print("Testing Gurobi license...")

try:
    import gurobipy as gp
    from gurobipy import GRB
    
    # Create a simple model
    m = gp.Model("test")
    
    # Add a variable
    x = m.addVar(vtype=GRB.CONTINUOUS, name="x")
    y = m.addVar(vtype=GRB.CONTINUOUS, name="y")
    
    # Set objective
    m.setObjective(x + y, GRB.MAXIMIZE)
    
    # Add constraint
    m.addConstr(x + 2 * y <= 10, "c0")
    m.addConstr(x <= 5, "c1")
    
    # Optimize model
    m.optimize()
    
    print("\nGurobi license is valid and working!")
    print("Model status:", m.Status)
    if m.Status == GRB.OPTIMAL:
        print("Optimal solution found")
        print("x =", x.X)
        print("y =", y.X)
        print("Obj =", m.ObjVal)
    
    # Print license details
    env = gp.Env()
    print("\nLicense details:")
    print("License type:", env.params.LicenseType)
    print("License expiration:", env.params.LicenseExpiration)
    print("Academic license:", "Yes" if env.params.LicenseType == 4 else "No")
    
except ImportError:
    print("Error: gurobipy module not found. Gurobi is not installed.")
except Exception as e:
    print("Error testing Gurobi license:", e)

print("\nDone.")