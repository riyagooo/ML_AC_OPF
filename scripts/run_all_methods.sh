#!/bin/bash

# Script to run all three ML-OPF approaches sequentially
# 1. Direct Prediction: Standard ML approach that directly predicts OPF solutions
# 2. Constraint Screening: Uses ML to identify which constraints will be binding
# 3. Warm Starting: Uses ML predictions to provide initialization points for optimization

# Accept command line argument for GNN usage
# Default is true (use GNN) if no argument provided
if [ "$1" == "false" ] || [ "$1" == "False" ] || [ "$1" == "0" ]; then
  USE_GNN=false
  echo "Running with standard neural networks (USE_GNN=false)"
else
  USE_GNN=true
  echo "Running with Graph Neural Networks (USE_GNN=true)"
fi

# Display a clear header to identify which architecture is being run
if [ "$USE_GNN" = true ]; then
  echo "==============================================================="
  echo "ML-AC-OPF: Running All Three ML Approaches with GRAPH NEURAL NETWORKS"
  echo "==============================================================="
else
  echo "==============================================================="
  echo "ML-AC-OPF: Running All Three ML Approaches with FEEDFORWARD NEURAL NETWORKS"
  echo "==============================================================="
fi

# Define common variables for the IEEE39 dataset
INPUT_DIR="data/realistic_case39/IEEE39"
OUTPUT_BASE_DIR="output"

# Create output directories if they don't exist
mkdir -p output/direct_prediction
mkdir -p output/constraint_screening
mkdir -p output/warm_starting
mkdir -p logs/direct_prediction
mkdir -p logs/constraint_screening
mkdir -p logs/warm_starting

# Set common parameters
EPOCHS=50             # Using 50 epochs for comprehensive training
BATCH_SIZE=32         # Adjusted for IEEE39 dataset which has many samples
NUM_SAMPLES=100000     # Using full 100,000 samples for better training
SAVE_RESULTS=true
K_FOLDS=5              # Number of folds for cross-validation

# Echo the epochs setting for debugging
echo "Using EPOCHS=$EPOCHS for training"

# Check if torch_geometric is installed for GNN support
if [ "$USE_GNN" = true ]; then
  python -c "import torch_geometric" 2>/dev/null
  if [ $? -ne 0 ]; then
    echo "Warning: torch_geometric not found. GNN models may not work correctly."
    echo "Consider installing with: pip install torch-geometric torch-scatter torch-sparse"
    echo "Continuing with standard models as fallback..."
    USE_GNN=false
  else
    echo "GNN dependencies found. Using GNN models where applicable."
  fi
fi

echo "Using data from: $INPUT_DIR"
echo "Training with: $EPOCHS epochs, batch size $BATCH_SIZE, $NUM_SAMPLES samples"
echo ""

# Function to check if a directory exists and create it if it doesn't
function ensure_dir() {
  if [ ! -d "$1" ]; then
    mkdir -p "$1"
  fi
}

# 1. Run Direct Prediction
echo "1. Running Direct Prediction..."
echo "---------------------------------------------------------------"
ensure_dir "$OUTPUT_BASE_DIR/direct_prediction"
ensure_dir "logs/direct_prediction"

# Preprocess data if needed
python preprocessing/preprocess_ieee39.py --input-dir $INPUT_DIR --output-dir $OUTPUT_BASE_DIR --num-samples $NUM_SAMPLES --mode direct_prediction

# Run direct prediction with or without GNN based on parameter
python direct_prediction.py \
  --input-dir "$OUTPUT_BASE_DIR/ieee39_data" \
  --output-dir "$OUTPUT_BASE_DIR/direct_prediction" \
  --epochs=$EPOCHS \
  --batch-size $BATCH_SIZE \
  --hidden-dim 256 \
  --num-layers 4 \
  --dropout 0.2 \
  --learning-rate 0.0005 \
  --weight-decay 1e-5 \
  --use-scaled-data \
  $([ "$USE_GNN" = true ] && echo "--use-gnn") \
  --save-model \
  --k-folds $K_FOLDS \
  2>&1 | tee "logs/direct_prediction/direct_prediction.log"

# 2. Run Constraint Screening
echo ""
echo "2. Running Constraint Screening..."
echo "---------------------------------------------------------------"
ensure_dir "$OUTPUT_BASE_DIR/constraint_screening"
ensure_dir "logs/constraint_screening"

# Run constraint screening with or without GNN based on parameter
python ieee39_constraint_screening.py \
  --input-dir "$OUTPUT_BASE_DIR/ieee39_data" \
  --output-dir "$OUTPUT_BASE_DIR/constraint_screening" \
  --epochs=$EPOCHS \
  --batch-size $BATCH_SIZE \
  --hidden-dims "128,256,128" \
  --dropout 0.3 \
  --learning-rate 0.001 \
  --weight-decay 1e-4 \
  --early-stopping 10 \
  $([ "$USE_GNN" = true ] && echo "--use-gnn") \
  --save-model \
  --k-folds $K_FOLDS \
  2>&1 | tee "logs/constraint_screening/constraint_screening.log"

# 3. Run Warm Starting
echo ""
echo "3. Running Warm Starting with IEEE39 Data..."
echo "---------------------------------------------------------------"
ensure_dir "$OUTPUT_BASE_DIR/warm_starting"
ensure_dir "logs/warm_starting"

# Preprocess data for warm starting if needed
python preprocessing/preprocess_ieee39.py --input-dir $INPUT_DIR --output-dir $OUTPUT_BASE_DIR --num-samples $NUM_SAMPLES --mode warm_starting

# Run warm starting with or without GNN based on parameter
python warmstart_model.py \
  --input-dir "$OUTPUT_BASE_DIR/ieee39_data" \
  --output-dir "$OUTPUT_BASE_DIR/warm_starting" \
  --epochs=$EPOCHS \
  --batch-size $BATCH_SIZE \
  --hidden-dim 256 \
  --num-layers 4 \
  --dropout 0.2 \
  --learning-rate 0.0005 \
  --weight-decay 1e-5 \
  --early-stopping 10 \
  --use-scaled-data \
  --model-type $([ "$USE_GNN" = true ] && echo "gnn" || echo "standard") \
  --save-model \
  --k-folds $K_FOLDS \
  2>&1 | tee "logs/warm_starting/warm_starting.log"

# Print completion message
echo ""
echo "==============================================================="
echo "All methods completed!"
echo "Results are saved in:"
echo "  - Direct Prediction: $OUTPUT_BASE_DIR/direct_prediction"
echo "  - Constraint Screening: $OUTPUT_BASE_DIR/constraint_screening"
echo "  - Warm Starting: $OUTPUT_BASE_DIR/warm_starting"
echo "===============================================================" 