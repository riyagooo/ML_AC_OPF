#!/usr/bin/env python
"""
Script to test Gurobi license in Google Colab.
"""

import sys
import os

def test_gurobi():
    """Test Gurobi license and solve a simple optimization problem."""
    print("Python version:", sys.version)
    print("Testing Gurobi license...")
    
    # Check if license file exists
    if os.path.exists("gurobi.lic"):
        print("Found gurobi.lic file")
        # Set license file environment variable
        os.environ["GRB_LICENSE_FILE"] = "gurobi.lic"
    else:
        print("Warning: No gurobi.lic file found in current directory")
        
    try:
        import gurobipy as gp
        from gurobipy import GRB
        
        print("\nSuccessfully imported gurobipy")
        
        # Create a simple model
        m = gp.Model("test")
        
        # Add variables
        x = m.addVar(vtype=GRB.CONTINUOUS, name="x")
        y = m.addVar(vtype=GRB.CONTINUOUS, name="y")
        
        # Set objective
        m.setObjective(x + y, GRB.MAXIMIZE)
        
        # Add constraints
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
        
        # Try to get license expiration (may not work in all environments)
        try:
            print("\nLicense information from gurobi.license():")
            license_info = gp.dispLicense()
            print(license_info)
        except:
            print("Could not retrieve detailed license information")
        
        return True
    
    except ImportError:
        print("Error: gurobipy module not found. Please install with: pip install gurobipy")
        return False
    except Exception as e:
        print("Error testing Gurobi license:", e)
        return False

if __name__ == "__main__":
    success = test_gurobi()
    print("\nGurobi test completed. Success:", success)