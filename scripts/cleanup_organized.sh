#!/bin/bash
# cleanup_organized.sh - Remove the redundant 'organized' directory
# This script will remove the 'organized' directory after proper restructuring

set -e  # Exit on error

echo "==== Checking if files were properly copied ===="

# Check core modules
if [ ! -f "core/models/feedforward.py" ] || [ ! -f "core/utils/metrics.py" ] || [ ! -f "core/validation/power_system_validation.py" ]; then
    echo "ERROR: Core modules were not copied properly!"
    exit 1
fi

# Check docs
if [ ! -f "docs/DOMAIN_METRICS_EVALUATION.md" ] || [ ! -f "docs/BALANCED_MODEL_COMPARISON.md" ]; then
    echo "ERROR: Documentation files were not copied properly!"
    exit 1
fi

# Check other directories
if [ ! -f "train/train_balanced_ffn.py" ] || [ ! -f "evaluation/evaluate_domain_metrics.py" ]; then
    echo "ERROR: Training or evaluation scripts were not copied properly!"
    exit 1
fi

# If all checks pass, create list of files that will be deleted (for safety)
echo "==== Creating backup listing of files to be deleted ===="
find organized -type f | grep -v "__pycache__" > organized_files_to_remove.txt
echo "List of files to be removed saved to 'organized_files_to_remove.txt'"

echo "==== SUMMARY ===="
echo "Total files in organized: $(find organized -type f | wc -l)"
echo "Total directories in organized: $(find organized -type d | wc -l)"

echo ""
echo "Are you sure you want to remove the 'organized' directory? (y/n)"
read -p "> " confirm

if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    echo "Removing 'organized' directory..."
    rm -rf organized
    echo "Done! The 'organized' directory has been removed."
else
    echo "Operation cancelled. 'organized' directory remains intact."
fi 