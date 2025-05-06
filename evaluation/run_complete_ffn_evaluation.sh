#!/bin/bash
# This script runs evaluate_ffn_complete.py which uses a reconstruction approach
# to evaluate the FFN model for domain-specific metrics

echo "Running complete FFN model evaluation with mldl environment..."

# Proper conda activation
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mldl

# Fix for PyTorch Geometric on macOS
export KMP_DUPLICATE_LIB_OK=TRUE

# Run the complete FFN evaluation script
python evaluate_ffn_complete.py

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "Complete FFN model evaluation finished successfully."
    echo "Results saved to output/domain_metrics/"
else
    echo "Complete FFN model evaluation failed."
fi 