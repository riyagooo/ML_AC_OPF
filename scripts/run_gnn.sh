#!/bin/bash

# Script to run all three ML-OPF approaches sequentially with Graph Neural Networks
# Copied from run_all_methods.sh with USE_GNN=true for comparison

# Define common variables for the IEEE39 dataset
IEEE39_DIR="data/realistic_case39/IEEE39"
SETPOINTS_FILE="$IEEE39_DIR/IEEE_39BUS_setpoints.csv"
LABELS_FILE="$IEEE39_DIR/IEEE_39BUS_labels.csv"
DATA_DIR="data/case39/processed/ml_data"

# Create output directories if they don't exist
mkdir -p output/direct_prediction_gnn
mkdir -p output/constraint_screening_gnn
mkdir -p output/warm_starting_gnn
mkdir -p logs/direct_prediction_gnn
mkdir -p logs/constraint_screening_gnn
mkdir -p logs/warm_starting_gnn

# Set common parameters
EPOCHS=30              # Reduced to 30 for faster completion
BATCH_SIZE=32         # Adjusted for IEEE39 dataset which has many samples
NUM_SAMPLES=100000     # Using full 50,000 samples for better training
SAVE_RESULTS=true
K_FOLDS=5              # Number of folds for cross-validation
USE_GNN=true           # Using Graph Neural Networks

# Check if torch_geometric is installed for GNN support
if $USE_GNN; then
  echo "Checking for GNN dependencies..."
  python -c "import torch_geometric" &> /dev/null
  if [ $? -ne 0 ]; then
    echo "Warning: torch_geometric not found. GNN models may not work correctly."
    echo "Consider installing with: pip install torch-geometric torch-scatter torch-sparse"
    echo "Continuing with standard models as fallback..."
  else
    echo "GNN dependencies found. Using GNN models where applicable."
  fi
fi

echo "==============================================================="
echo "ML-AC-OPF: Running All Three ML Approaches with IEEE39 Data (GNN)"
echo "==============================================================="
echo "Using data from: $IEEE39_DIR"
echo "Training with: $EPOCHS epochs, batch size $BATCH_SIZE, $NUM_SAMPLES samples"
echo "Using GNN: $USE_GNN"
echo ""

echo "1. Running Direct Prediction with GNN..."
echo "---------------------------------------------------------------"
# Run the direct prediction model with GNN settings
python direct_prediction.py \
  --input-dir "output/ieee39_data" \
  --output-dir "output/direct_prediction_gnn" \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --hidden-dim 256 \
  --num-layers 4 \
  --dropout 0.2 \
  --learning-rate 0.0005 \
  --use-scaled-data \
  --save-model \
  --k-folds $K_FOLDS \
  --use-gnn

echo ""
echo "2. Running Constraint Screening with GNN..."
echo "---------------------------------------------------------------"
# Run the constraint screening model with GNN settings
python ieee39_constraint_screening.py \
  --input-dir "output/ieee39_data" \
  --output-dir "output/constraint_screening_gnn" \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --hidden-dims "128,256,128" \
  --learning-rate 0.0005 \
  --dropout 0.3 \
  --weight-decay 1e-4 \
  --early-stopping 10 \
  --save-model \
  --k-folds $K_FOLDS \
  --use-gnn

echo ""
echo "3. Running Warm Starting with GNN..."
echo "---------------------------------------------------------------"
# Run the warm starting model with GNN settings
python warmstart_model.py \
  --input-dir "output/ieee39_data" \
  --output-dir "output/warm_starting_gnn" \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --hidden-dim 256 \
  --num-layers 4 \
  --dropout 0.2 \
  --learning-rate 0.0005 \
  --weight-decay 1e-5 \
  --early-stopping 10 \
  --use-scaled-data \
  --model-type "gnn" \
  --save-model \
  --k-folds $K_FOLDS

echo ""
echo "==============================================================="
echo "Graph Neural Network training completed!"
echo "Results are saved in:"
echo "  - Direct Prediction: output/direct_prediction_gnn"
echo "  - Constraint Screening: output/constraint_screening_gnn"
echo "  - Warm Starting: output/warm_starting_gnn"
echo "==============================================================="

# Add timing visualization generation
echo ""
echo "Generating timing visualizations..."
echo "---------------------------------------------------------------"

# Create visualization script if it doesn't already exist
if [ ! -d "visualization_scripts" ]; then
  mkdir -p visualization_scripts
fi

# Create output directory for visualizations
mkdir -p output/visualizations

# Execute the timing visualization script
python visualization_scripts/generate_timing_visualizations.py

echo ""
echo "Timing visualizations saved to output/visualizations/"
echo "These visualizations show the computational advantage of ML approaches"
echo "compared to traditional AC-OPF solvers."
echo "===============================================================" 