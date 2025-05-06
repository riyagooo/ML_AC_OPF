#!/usr/bin/env python
"""
Script for directly using the IEEE 39-bus dataset from Zenodo

This script processes the IEEE 39-bus data downloaded from Zenodo
and prepares it for machine learning model training and analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ieee39_data_processor')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process IEEE 39-bus data')
    parser.add_argument('--input-dir', type=str, default='data/realistic_case39',
                      help='Input directory with IEEE data (default: data/realistic_case39)')
    parser.add_argument('--output-dir', type=str, default='data/case39/processed',
                      help='Output directory (default: data/case39/processed)')
    parser.add_argument('--sample-size', type=int, default=5000,
                      help='Number of samples to use for analysis and training (default: 5000, -1 for all)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed (default: 42)')
    parser.add_argument('--analyze-only', action='store_true',
                      help='Perform data analysis only without generating training files')
    parser.add_argument('--test-split', type=float, default=0.2,
                      help='Test split ratio (default: 0.2)')
    parser.add_argument('--val-split', type=float, default=0.1,
                      help='Validation split ratio from training data (default: 0.1)')
    return parser.parse_args()

def load_ieee_data(input_dir, sample_size=-1):
    """
    Load the complete IEEE 39-bus dataset.
    
    Args:
        input_dir: Directory containing the IEEE39 data
        sample_size: Number of samples to load (-1 for all)
    
    Returns:
        DataFrame with combined setpoints and labels
    """
    logger.info(f"Loading IEEE 39-bus data from {input_dir}")
    
    # Define file paths
    setpoints_file = os.path.join(input_dir, 'IEEE39', 'IEEE_39BUS_setpoints.csv')
    labels_file = os.path.join(input_dir, 'IEEE39', 'IEEE_39BUS_labels.csv')
    
    # Check if files exist
    if not os.path.exists(setpoints_file) or not os.path.exists(labels_file):
        logger.error(f"Dataset files not found in {input_dir}/IEEE39")
        sys.exit(1)
    
    # Determine number of rows to read
    if sample_size > 0:
        logger.info(f"Loading {sample_size} samples")
        setpoints_df = pd.read_csv(setpoints_file, nrows=sample_size)
        labels_df = pd.read_csv(labels_file, nrows=sample_size)
    else:
        logger.info("Loading all samples")
        setpoints_df = pd.read_csv(setpoints_file)
        labels_df = pd.read_csv(labels_file)
    
    # Combine data
    data = pd.concat([setpoints_df, labels_df], axis=1)
    
    logger.info(f"Loaded {len(data)} samples with {len(data.columns)} columns")
    return data

def analyze_data(data, output_dir):
    """
    Perform exploratory data analysis on the IEEE data.
    
    Args:
        data: DataFrame with the IEEE data
        output_dir: Directory to save analysis results
    """
    logger.info("Analyzing IEEE 39-bus data")
    
    # Create output directory for analysis
    analysis_dir = os.path.join(output_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Basic statistics
    logger.info(f"Dataset shape: {data.shape}")
    logger.info(f"Total samples: {len(data)}")
    logger.info(f"Feasible samples: {data['class'].sum()} ({data['class'].mean()*100:.2f}%)")
    logger.info(f"Infeasible samples: {len(data) - data['class'].sum()} ({(1-data['class'].mean())*100:.2f}%)")
    
    # Save basic statistics
    with open(os.path.join(analysis_dir, 'dataset_stats.txt'), 'w') as f:
        f.write(f"Dataset shape: {data.shape}\n")
        f.write(f"Total samples: {len(data)}\n")
        f.write(f"Feasible samples: {data['class'].sum()} ({data['class'].mean()*100:.2f}%)\n")
        f.write(f"Infeasible samples: {len(data) - data['class'].sum()} ({(1-data['class'].mean())*100:.2f}%)\n")
        f.write("\nBinding Constraints Distribution:\n")
        
        # Count samples by binding constraint type
        for col in ['PG', 'QG', 'VM', 'SF', 'SSS']:
            if col in data.columns:
                binding_count = (data[col] == 1).sum()
                binding_percent = binding_count / len(data) * 100
                f.write(f"{col}: {binding_count} samples ({binding_percent:.2f}%)\n")
    
    # Create visualizations
    
    # 1. Generator active power distributions
    plt.figure(figsize=(15, 10))
    pg_cols = [col for col in data.columns if col.startswith('PG')]
    
    # Plot only the first 10 generators to avoid overcrowding
    for i, col in enumerate(pg_cols[:10]):
        plt.subplot(3, 4, i+1)
        sns.histplot(data=data, x=col, hue='class', kde=True, 
                    palette=['red', 'green'], alpha=0.5)
        plt.title(f'{col} Distribution')
        plt.xlabel('Active Power (p.u.)')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'generator_power_distribution.png'))
    plt.close()
    
    # 2. Voltage magnitude distributions
    plt.figure(figsize=(15, 10))
    vm_cols = [col for col in data.columns if col.startswith('VM')]
    
    # Plot only the first 10 voltage magnitudes
    for i, col in enumerate(vm_cols[:10]):
        plt.subplot(3, 4, i+1)
        sns.histplot(data=data, x=col, hue='class', kde=True, 
                    palette=['red', 'green'], alpha=0.5)
        plt.title(f'{col} Distribution')
        plt.xlabel('Voltage Magnitude (p.u.)')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'voltage_distribution.png'))
    plt.close()
    
    # 3. Correlation matrix of key variables
    plt.figure(figsize=(12, 10))
    selected_cols = pg_cols[:5] + vm_cols[:5] + ['class']
    corr_matrix = data[selected_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'correlation_matrix.png'))
    plt.close()
    
    # 4. Binding constraints analysis
    constraint_cols = ['PG', 'QG', 'VM', 'SF', 'SSS']
    if all(col in data.columns for col in constraint_cols):
        # Count binding constraints
        binding_counts = {col: (data[col] == 1).sum() for col in constraint_cols}
        
        plt.figure(figsize=(10, 6))
        plt.bar(binding_counts.keys(), binding_counts.values())
        plt.title('Binding Constraint Distribution')
        plt.xlabel('Constraint Type')
        plt.ylabel('Number of Samples')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'binding_constraints.png'))
        plt.close()
    
    logger.info(f"Analysis results saved to {analysis_dir}")
    return analysis_dir

def prepare_ml_data(data, output_dir, test_split=0.2, val_split=0.1, seed=42):
    """
    Prepare the data for machine learning models.
    
    Args:
        data: DataFrame with the IEEE data
        output_dir: Directory to save processed data
        test_split: Test set proportion
        val_split: Validation set proportion (from training set)
        seed: Random seed
    """
    logger.info("Preparing data for machine learning")
    
    # Create ML data directory
    ml_dir = os.path.join(output_dir, 'ml_data')
    os.makedirs(ml_dir, exist_ok=True)
    
    # Define key variable groups
    pg_cols = [col for col in data.columns if col.startswith('PG')]
    vm_cols = [col for col in data.columns if col.startswith('VM')]
    qg_cols = [col for col in data.columns if col.startswith('QG')]
    va_cols = [col for col in data.columns if col.startswith('VA')]
    
    # Create feature and target sets for different ML tasks
    
    # 1. Feasibility classification
    # Features: Generator setpoints and voltage magnitudes
    X_feasibility = data[pg_cols + vm_cols]
    y_feasibility = data['class']
    
    # 2. Direct prediction (predict Volt/VAR optimization)
    # Features: Active power injections
    # Targets: Voltage magnitudes and reactive power
    X_direct = data[pg_cols]
    y_direct = data[vm_cols + qg_cols]
    
    # 3. Warm-starting (predict OPF solution)
    # Features: System demand 
    # Targets: Active and reactive power, voltage magnitude and angle
    if 'PD' in data.columns:
        pd_cols = [col for col in data.columns if col.startswith('PD')]
        qd_cols = [col for col in data.columns if col.startswith('QD')]
        X_warmstart = data[pd_cols + qd_cols]
        y_warmstart = data[pg_cols + qg_cols + vm_cols + va_cols]
    else:
        # If no explicit demand columns, use a subset of generators as proxy for demand
        X_warmstart = data[pg_cols[:5]]
        y_warmstart = data[pg_cols[5:] + qg_cols + vm_cols + va_cols]
    
    # Split data
    np.random.seed(seed)
    
    # Split for feasibility task
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
        X_feasibility, y_feasibility, test_size=test_split, random_state=seed, stratify=y_feasibility
    )
    X_train_f, X_val_f, y_train_f, y_val_f = train_test_split(
        X_train_f, y_train_f, test_size=val_split, random_state=seed, stratify=y_train_f
    )
    
    # Split for direct prediction task
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        X_direct, y_direct, test_size=test_split, random_state=seed
    )
    X_train_d, X_val_d, y_train_d, y_val_d = train_test_split(
        X_train_d, y_train_d, test_size=val_split, random_state=seed
    )
    
    # Split for warm-starting task
    X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
        X_warmstart, y_warmstart, test_size=test_split, random_state=seed
    )
    X_train_w, X_val_w, y_train_w, y_val_w = train_test_split(
        X_train_w, y_train_w, test_size=val_split, random_state=seed
    )
    
    # Create scalers
    scaler_X_f = StandardScaler().fit(X_train_f)
    scaler_X_d = StandardScaler().fit(X_train_d)
    scaler_X_w = StandardScaler().fit(X_train_w)
    scaler_y_d = StandardScaler().fit(y_train_d)
    scaler_y_w = StandardScaler().fit(y_train_w)
    
    # Save datasets for each task
    
    # Feasibility classification
    feasibility_dir = os.path.join(ml_dir, 'feasibility')
    os.makedirs(feasibility_dir, exist_ok=True)
    
    # Save raw data
    save_data(X_train_f, y_train_f, os.path.join(feasibility_dir, 'train'))
    save_data(X_val_f, y_val_f, os.path.join(feasibility_dir, 'val'))
    save_data(X_test_f, y_test_f, os.path.join(feasibility_dir, 'test'))
    
    # Save scaled data
    save_data(scaler_X_f.transform(X_train_f), y_train_f, os.path.join(feasibility_dir, 'train_scaled'))
    save_data(scaler_X_f.transform(X_val_f), y_val_f, os.path.join(feasibility_dir, 'val_scaled'))
    save_data(scaler_X_f.transform(X_test_f), y_test_f, os.path.join(feasibility_dir, 'test_scaled'))
    
    # Save column names
    with open(os.path.join(feasibility_dir, 'feature_cols.txt'), 'w') as f:
        f.write('\n'.join(X_feasibility.columns))
    
    # Direct prediction
    direct_dir = os.path.join(ml_dir, 'direct_prediction')
    os.makedirs(direct_dir, exist_ok=True)
    
    # Save raw data
    save_data(X_train_d, y_train_d, os.path.join(direct_dir, 'train'))
    save_data(X_val_d, y_val_d, os.path.join(direct_dir, 'val'))
    save_data(X_test_d, y_test_d, os.path.join(direct_dir, 'test'))
    
    # Save scaled data
    save_data(scaler_X_d.transform(X_train_d), scaler_y_d.transform(y_train_d), 
              os.path.join(direct_dir, 'train_scaled'))
    save_data(scaler_X_d.transform(X_val_d), scaler_y_d.transform(y_val_d), 
              os.path.join(direct_dir, 'val_scaled'))
    save_data(scaler_X_d.transform(X_test_d), scaler_y_d.transform(y_test_d), 
              os.path.join(direct_dir, 'test_scaled'))
    
    # Save column names
    with open(os.path.join(direct_dir, 'feature_cols.txt'), 'w') as f:
        f.write('\n'.join(X_direct.columns))
    with open(os.path.join(direct_dir, 'target_cols.txt'), 'w') as f:
        f.write('\n'.join(y_direct.columns))
    
    # Warm-starting
    warmstart_dir = os.path.join(ml_dir, 'warm_starting')
    os.makedirs(warmstart_dir, exist_ok=True)
    
    # Save raw data
    save_data(X_train_w, y_train_w, os.path.join(warmstart_dir, 'train'))
    save_data(X_val_w, y_val_w, os.path.join(warmstart_dir, 'val'))
    save_data(X_test_w, y_test_w, os.path.join(warmstart_dir, 'test'))
    
    # Save scaled data
    save_data(scaler_X_w.transform(X_train_w), scaler_y_w.transform(y_train_w), 
              os.path.join(warmstart_dir, 'train_scaled'))
    save_data(scaler_X_w.transform(X_val_w), scaler_y_w.transform(y_val_w), 
              os.path.join(warmstart_dir, 'val_scaled'))
    save_data(scaler_X_w.transform(X_test_w), scaler_y_w.transform(y_test_w), 
              os.path.join(warmstart_dir, 'test_scaled'))
    
    # Save column names
    with open(os.path.join(warmstart_dir, 'feature_cols.txt'), 'w') as f:
        f.write('\n'.join(X_warmstart.columns))
    with open(os.path.join(warmstart_dir, 'target_cols.txt'), 'w') as f:
        f.write('\n'.join(y_warmstart.columns))
    
    # Save scalers
    np.save(os.path.join(ml_dir, 'scaler_X_f_params.npy'), 
            [scaler_X_f.mean_, scaler_X_f.scale_])
    np.save(os.path.join(ml_dir, 'scaler_X_d_params.npy'), 
            [scaler_X_d.mean_, scaler_X_d.scale_])
    np.save(os.path.join(ml_dir, 'scaler_X_w_params.npy'), 
            [scaler_X_w.mean_, scaler_X_w.scale_])
    np.save(os.path.join(ml_dir, 'scaler_y_d_params.npy'), 
            [scaler_y_d.mean_, scaler_y_d.scale_])
    np.save(os.path.join(ml_dir, 'scaler_y_w_params.npy'), 
            [scaler_y_w.mean_, scaler_y_w.scale_])
    
    # Print summary
    logger.info(f"Prepared ML data for 3 tasks:")
    logger.info(f"  Feasibility classification: {X_train_f.shape[0]} train, {X_val_f.shape[0]} val, {X_test_f.shape[0]} test samples")
    logger.info(f"  Direct prediction: {X_train_d.shape[0]} train, {X_val_d.shape[0]} val, {X_test_d.shape[0]} test samples")
    logger.info(f"  Warm-starting: {X_train_w.shape[0]} train, {X_val_w.shape[0]} val, {X_test_w.shape[0]} test samples")
    logger.info(f"ML data saved to {ml_dir}")
    
    return ml_dir

def save_data(X, y, path_prefix):
    """Save data to CSV and numpy formats."""
    # Save as numpy arrays
    np.save(f"{path_prefix}_X.npy", X.values if isinstance(X, pd.DataFrame) else X)
    np.save(f"{path_prefix}_y.npy", y.values if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series) else y)
    
    # Save as CSV files
    if isinstance(X, pd.DataFrame):
        X.to_csv(f"{path_prefix}_X.csv", index=False)
    else:
        pd.DataFrame(X).to_csv(f"{path_prefix}_X.csv", index=False)
    
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y.to_csv(f"{path_prefix}_y.csv", index=False)
    else:
        pd.DataFrame(y).to_csv(f"{path_prefix}_y.csv", index=False)

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    data = load_ieee_data(args.input_dir, args.sample_size)
    
    # Analyze data
    analysis_dir = analyze_data(data, args.output_dir)
    
    # Prepare ML data if not in analyze-only mode
    if not args.analyze_only:
        ml_dir = prepare_ml_data(data, args.output_dir, args.test_split, args.val_split, args.seed)
    
    logger.info("Processing completed successfully!")
    
    # Print guide for next steps
    print("\n" + "="*80)
    print("IEEE 39-BUS DATA PROCESSING COMPLETED")
    print("="*80)
    print("\nNext steps you can take:")
    print("\n1. View the data analysis in:")
    print(f"   {os.path.abspath(analysis_dir)}")
    
    if not args.analyze_only:
        print("\n2. Train models using the prepared ML datasets:")
        print("   - Feasibility Classification: Predict if a power system state is feasible")
        print("   - Direct Prediction: Predict voltage and reactive power from active power")
        print("   - Warm-Starting: Predict OPF solution to initialize optimization solvers")
        print("\n   Example command to train a feasibility model:")
        print("   python scripts/train_feasibility_model.py --data-dir "+ 
              os.path.join(args.output_dir, "ml_data/feasibility"))
    
    print("\n3. Run a sample training with:")
    print("   python scripts/constraint_screening.py --case case39 --data-dir " + 
          os.path.join(args.output_dir, "ml_data") + " --epochs 10")
    
    print("\n" + "="*80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 