#!/usr/bin/env python
"""
Script to create a modified version of the MATPOWER file that works with pypower.
"""

import os
import sys

def fix_matpower_file(case_name, data_dir="data"):
    """
    Create a modified version of the MATPOWER file that's compatible with pypower.
    
    Args:
        case_name: Base name of the case (e.g., 'case5')
        data_dir: Directory containing the data files
        
    Returns:
        Path to the fixed MATPOWER file
    """
    # Original file path
    original_file = os.path.join(data_dir, f"pglib_opf_{case_name}.m")
    
    if not os.path.exists(original_file):
        print(f"Error: Original file not found: {original_file}")
        return None
    
    # Fixed file path 
    fixed_file = os.path.join(data_dir, f"{case_name}.m")
    
    # Read original content
    with open(original_file, 'r') as f:
        content = f.read()
    
    # Replace function name
    if 'function mpc = pglib_opf_' in content:
        content = content.replace(f'function mpc = pglib_opf_{case_name}_pjm', f'function mpc = {case_name}')
    
    # More robust replacement in case the function name format is different
    content = content.replace(f'function mpc = pglib_opf_{case_name}', f'function mpc = {case_name}')
    
    # Make sure we have the correct 'mpc =' format for variable assignment
    content = content.replace('baseMVA = ', 'mpc.baseMVA = ')
    content = content.replace('bus = [', 'mpc.bus = [')
    content = content.replace('gen = [', 'mpc.gen = [')
    content = content.replace('branch = [', 'mpc.branch = [')
    content = content.replace('gencost = [', 'mpc.gencost = [')
    
    # Write fixed content
    with open(fixed_file, 'w') as f:
        f.write(content)
    
    print(f"Created fixed MATPOWER file: {fixed_file}")
    return fixed_file

if __name__ == "__main__":
    if len(sys.argv) > 1:
        case_name = sys.argv[1]
    else:
        case_name = "case5"
    
    fix_matpower_file(case_name)