#!/bin/bash
# This script creates a symbolic link for Python 3.10 to fix PyTorch Geometric C++ extension issues

echo "Creating symbolic link for Python 3.10..."

# Path to your conda environment's Python library
CONDA_PREFIX="/opt/homebrew/Caskroom/miniconda/base/envs/mldl"
PYTHON_LIB="$CONDA_PREFIX/lib/libpython3.10.dylib"

# Check if the Python library exists
if [ ! -f "$PYTHON_LIB" ]; then
    echo "Error: Python library not found at $PYTHON_LIB"
    exit 1
fi

# Create the directory if it doesn't exist
# Note: This requires sudo permissions
echo "Creating directory /Library/Frameworks/Python.framework/Versions/3.10/"
sudo mkdir -p /Library/Frameworks/Python.framework/Versions/3.10/

# Create the symbolic link
echo "Creating symbolic link to $PYTHON_LIB"
sudo ln -sf "$PYTHON_LIB" /Library/Frameworks/Python.framework/Versions/3.10/Python

echo "Symbolic link created successfully!"
echo ""
echo "Now try running ./run_evaluation.sh" 