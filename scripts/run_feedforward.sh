#!/bin/bash

# Script to run all three ML-OPF approaches sequentially with feedforward neural networks
# Copied from run_all_methods.sh with USE_GNN=false for comparison

# Define common variables for the IEEE39 dataset
IEEE39_DIR="data/realistic_case39/IEEE39"
SETPOINTS_FILE="$IEEE39_DIR/IEEE_39BUS_setpoints.csv"
LABELS_FILE="$IEEE39_DIR/IEEE_39BUS_labels.csv"
DATA_DIR="data/case39/processed/ml_data"

# Create output directories if they don't exist
mkdir -p output/direct_prediction_ff
mkdir -p output/constraint_screening_ff
mkdir -p output/warm_starting_ff
mkdir -p logs/direct_prediction_ff
mkdir -p logs/constraint_screening_ff
mkdir -p logs/warm_starting_ff

# Set common parameters
EPOCHS=30              # Reduced to 30 for faster completion
BATCH_SIZE=32         # Adjusted for IEEE39 dataset which has many samples
NUM_SAMPLES=100000     # Using full 50,000 samples for better training
SAVE_RESULTS=true
K_FOLDS=5              # Number of folds for cross-validation
USE_GNN=false           # Using standard feedforward neural networks

echo "==============================================================="
echo "ML-AC-OPF: Running All Three ML Approaches with IEEE39 Data (Feedforward NN)"
echo "==============================================================="
echo "Using data from: $IEEE39_DIR"
echo "Training with: $EPOCHS epochs, batch size $BATCH_SIZE, $NUM_SAMPLES samples"
echo "Using GNN: $USE_GNN"
echo ""

echo "1. Running Direct Prediction with Feedforward NN..."
echo "---------------------------------------------------------------"
# Run the direct prediction model with feedforward settings
python direct_prediction.py \
  --input-dir "output/ieee39_data" \
  --output-dir "output/direct_prediction_ff" \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --hidden-dim 256 \
  --num-layers 4 \
  --dropout 0.2 \
  --learning-rate 0.0005 \
  --use-scaled-data \
  --save-model \
  --k-folds $K_FOLDS

echo ""
echo "2. Running Constraint Screening with Feedforward NN..."
echo "---------------------------------------------------------------"
# Run the constraint screening model with feedforward settings
python ieee39_constraint_screening.py \
  --input-dir "output/ieee39_data" \
  --output-dir "output/constraint_screening_ff" \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --hidden-dims "128,256,128" \
  --learning-rate 0.0005 \
  --dropout 0.3 \
  --weight-decay 1e-4 \
  --early-stopping 10 \
  --save-model \
  --k-folds $K_FOLDS

echo ""
echo "3. Running Warm Starting with Feedforward NN..."
echo "---------------------------------------------------------------"
# Run the warm starting model with feedforward settings
python warmstart_model.py \
  --input-dir "output/ieee39_data" \
  --output-dir "output/warm_starting_ff" \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --hidden-dim 256 \
  --num-layers 4 \
  --dropout 0.2 \
  --learning-rate 0.0005 \
  --weight-decay 1e-5 \
  --early-stopping 10 \
  --use-scaled-data \
  --model-type "standard" \
  --save-model \
  --k-folds $K_FOLDS

echo ""
echo "==============================================================="
echo "Feedforward Neural Network training completed!"
echo "Results are saved in:"
echo "  - Direct Prediction: output/direct_prediction_ff"
echo "  - Constraint Screening: output/constraint_screening_ff"
echo "  - Warm Starting: output/warm_starting_ff"
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