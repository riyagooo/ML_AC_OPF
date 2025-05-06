#!/bin/bash
# This script runs evaluate_ffn_metrics.py using the mldl environment
# This focuses only on evaluating the FFN model for domain-specific metrics

echo "Running FFN model evaluation with mldl environment..."

# Proper conda activation
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mldl

# Fix for PyTorch Geometric on macOS
export KMP_DUPLICATE_LIB_OK=TRUE

# Run the FFN-only evaluation script
python evaluate_ffn_metrics.py

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "FFN model evaluation completed successfully."
    echo "Results saved to output/domain_metrics/"
else
    echo "FFN model evaluation failed."
fi 