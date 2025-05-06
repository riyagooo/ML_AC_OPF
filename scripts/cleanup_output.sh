#!/bin/bash
# Script to clean up unnecessary output directories while preserving essential ones

echo "Cleaning up unnecessary output directories..."

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Essential directories to keep
KEEP_DIRS=(
  "output/domain_metrics"      # Domain metrics evaluation results
  "output/balanced_ffn"        # FFN model evaluated in domain metrics
  "output/balanced_gnn"        # GNN model referenced in documentation
  "output/ieee39_data_small"   # Test data used for evaluation
  "output/balanced_comparison" # Balanced model comparison visualizations
  "output/model_comparison"    # Model comparison metrics and visualizations
  "output/data_exploration"    # Dataset exploration visualizations
)

# Get all directories in output
ALL_DIRS=$(find output -maxdepth 1 -type d | grep -v "^output$")

# Remove directories that are not in KEEP_DIRS
for dir in $ALL_DIRS; do
  if [[ ! " ${KEEP_DIRS[@]} " =~ " ${dir} " ]]; then
    echo "Removing directory: $dir"
    rm -rf "$dir"
  else
    echo "Keeping directory: $dir"
  fi
done

# Remove any stray files in output root that aren't needed
if [ -f "output/matrix_metrics.csv" ]; then
  echo "Removing file: output/matrix_metrics.csv"
  rm -f "output/matrix_metrics.csv"
fi

echo "Cleanup complete. Essential directories preserved."
echo "Preserved directories:"
for dir in "${KEEP_DIRS[@]}"; do
  echo "- $dir"
done 