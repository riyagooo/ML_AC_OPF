#!/usr/bin/env python
"""
Patch script to fix Gurobi constraint handling in the OPFOptimizer.
"""

import os
import re

def patch_gurobi_constraints():
    """Patch the OPFOptimizer to properly handle Gurobi constraints."""
    file_path = os.path.join('utils', 'optimization.py')
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return False
    
    # Read the current file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the addGenConstrCos and addGenConstrSin calls
    # Problem: Gurobi expects a variable, not an expression, for angle differences
    # Solution: Create an auxiliary variable for angle differences
    
    # Find the solve_opf_gurobi function
    updated_content = []
    lines = content.split('\n')
    inside_func = False
    constraint_fixed = False
    
    for line in lines:
        if 'def solve_opf_gurobi(' in line:
            inside_func = True
        
        if inside_func and 'model.addGenConstrCos(va[i] - va[j],' in line:
            # Replace with code to create auxiliary variable
            indent = line.split('model')[0]
            updated_content.append(f"{indent}# Create auxiliary variable for angle difference")
            updated_content.append(f"{indent}angle_diff = model.addVar(lb=-np.pi, ub=np.pi, name=f\"angle_diff_{i}_{j}\")")
            updated_content.append(f"{indent}model.addConstr(angle_diff == va[i] - va[j], f\"angle_diff_constr_{i}_{j}\")")
            updated_content.append(f"{indent}# Use auxiliary variable in cosine constraint")
            updated_content.append(f"{indent}model.addGenConstrCos(angle_diff, p_term, f\"cos_{i}_{j}\")")
            constraint_fixed = True
        elif inside_func and 'model.addGenConstrSin(va[i] - va[j],' in line:
            # Skip this line as it's replaced with our code above
            indent = line.split('model')[0]
            updated_content.append(f"{indent}# Use auxiliary variable in sine constraint")
            updated_content.append(f"{indent}model.addGenConstrSin(angle_diff, q_term, f\"sin_{i}_{j}\")")
        elif inside_func and 'angle_diff = va[from_bus] - va[to_bus] - shift' in line:
            # Handle the branch angle difference similarly
            indent = line.split('angle_diff')[0]
            updated_content.append(f"{indent}# Create auxiliary variable for angle difference with phase shift")
            updated_content.append(f"{indent}angle_diff = model.addVar(lb=-np.pi, ub=np.pi, name=f\"angle_diff_br_{i}\")")
            updated_content.append(f"{indent}model.addConstr(angle_diff == va[from_bus] - va[to_bus] - shift, f\"angle_diff_br_constr_{i}\")")
            constraint_fixed = True
        elif inside_func and 'model.addGenConstrCos(angle_diff, p_ft_term,' in line:
            # Skip this line as it's replaced with our code above
            indent = line.split('model')[0]
            updated_content.append(f"{indent}model.addGenConstrCos(angle_diff, p_ft_term, f\"p_ft_cos_{i}\")")
        elif inside_func and 'model.addGenConstrSin(angle_diff, q_ft_term,' in line:
            # Skip this line as it's replaced with our code above
            indent = line.split('model')[0]
            updated_content.append(f"{indent}model.addGenConstrSin(angle_diff, q_ft_term, f\"q_ft_sin_{i}\")")
        else:
            updated_content.append(line)
    
    # Write the updated content back
    if constraint_fixed:
        with open(file_path, 'w') as f:
            f.write('\n'.join(updated_content))
        print(f"Successfully patched {file_path}")
        return True
    else:
        print(f"No changes made to {file_path}")
        return False

if __name__ == "__main__":
    patch_gurobi_constraints() 