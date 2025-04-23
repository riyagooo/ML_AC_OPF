#!/usr/bin/env python
"""
Debug script to check if case data loads correctly
"""

import os
import sys
import numpy as np
from pathlib import Path
import pprint

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pypower.api import loadcase

def main():
    case_name = "case5"
    data_dir = "data"
    file_path = os.path.join(data_dir, f"pglib_opf_{case_name}.m")
    
    print(f"Loading file: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    
    # Create a temp file with the correct function name for pypower
    temp_file = os.path.join(data_dir, f"temp_{case_name}.m")
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        content_lines = content.split('\n')
        function_lines = [line for line in content_lines if 'function mpc =' in line]
        print(f"Original function name line: {function_lines[0] if function_lines else 'Not found'}")
        
        # Replace the function name to match what pypower expects
        content = content.replace(f'function mpc = pglib_opf_{case_name}_pjm', f'function mpc = {case_name}')
        
        content_lines = content.split('\n')
        function_lines = [line for line in content_lines if 'function mpc =' in line]
        print(f"Modified function name line: {function_lines[0] if function_lines else 'Not found'}")
        
        with open(temp_file, 'w') as f:
            f.write(content)
        
        # Load the modified case data
        print(f"Loading case from temp file: {temp_file}")
        case_data = loadcase(temp_file)
        
        print(f"Type of case_data: {type(case_data)}")
        if isinstance(case_data, dict):
            print("Case data keys:", case_data.keys())
            print("baseMVA:", case_data.get('baseMVA'))
            print("Number of buses:", len(case_data.get('bus', [])))
            print("Number of branches:", len(case_data.get('branch', [])))
            print("Number of generators:", len(case_data.get('gen', [])))
        else:
            print(f"Case data content: {case_data}")
        
        # Clean up temp file
        os.remove(temp_file)
        
    except Exception as e:
        print(f"Error loading case: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()