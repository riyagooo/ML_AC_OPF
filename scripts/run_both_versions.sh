#!/bin/bash

# Script to run both GNN and non-GNN versions of ML-OPF approaches

echo "==============================================================="
echo "Running ML-AC-OPF with both neural network architectures"
echo "==============================================================="

# Create separate output directories for each version
mkdir -p output/direct_prediction_ff
mkdir -p output/constraint_screening_ff
mkdir -p output/warm_starting_ff
mkdir -p output/direct_prediction_gnn
mkdir -p output/constraint_screening_gnn
mkdir -p output/warm_starting_gnn

# First run with standard neural networks
echo "STEP 1: Running with standard feedforward neural networks..."
echo "---------------------------------------------------------------"
bash run_all_methods.sh false

# Move the results to the _ff directories
echo "Moving standard neural network results to separate directories..."
cp -r output/direct_prediction/* output/direct_prediction_ff/
cp -r output/constraint_screening/* output/constraint_screening_ff/
cp -r output/warm_starting/* output/warm_starting_ff/

# Wait for the first run to fully complete
echo "Waiting for all processes to complete before continuing..."
wait

# Clean up any remaining processes from the first run
for pid in $(ps -ef | grep "python" | grep "direct_prediction.py\|ieee39_constraint_screening.py\|warmstart_model.py" | grep -v grep | awk '{print $2}'); do
  echo "Stopping process $pid"
  kill -9 $pid 2>/dev/null
done

echo "First run completed. Waiting 5 seconds before starting second run..."
sleep 5

# Then run with Graph Neural Networks
echo ""
echo "STEP 2: Running with Graph Neural Networks..."
echo "---------------------------------------------------------------"
bash run_all_methods.sh true

# Move the results to the _gnn directories
echo "Moving GNN results to separate directories..."
cp -r output/direct_prediction/* output/direct_prediction_gnn/
cp -r output/constraint_screening/* output/constraint_screening_gnn/
cp -r output/warm_starting/* output/warm_starting_gnn/

echo ""
echo "==============================================================="
echo "All runs completed! Results are available in:"
echo "  - Standard Neural Networks:"
echo "      - Direct Prediction: output/direct_prediction_ff"
echo "      - Constraint Screening: output/constraint_screening_ff"
echo "      - Warm Starting: output/warm_starting_ff"
echo "  - Graph Neural Networks:"
echo "      - Direct Prediction: output/direct_prediction_gnn"
echo "      - Constraint Screening: output/constraint_screening_gnn"
echo "      - Warm Starting: output/warm_starting_gnn"
echo "==============================================================="

# No need to generate visualizations here as they are already generated
# in both run_feedforward.sh and run_gnn.sh

echo ""
echo "==============================================================="
echo "All model training completed!"
echo "Timing visualizations have been saved to output/visualizations/"
echo "===============================================================" 