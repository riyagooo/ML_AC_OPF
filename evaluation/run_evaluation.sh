#!/bin/bash
# This script runs evaluate_domain_metrics.py using the mldl environment

echo "Running evaluation with mldl environment..."

# Proper conda activation
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mldl

# Fix for PyTorch Geometric on macOS
export KMP_DUPLICATE_LIB_OK=TRUE

# Run the patched evaluation script
python run_evaluation_patched.py

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully."
else
    echo "Evaluation failed."
fi 