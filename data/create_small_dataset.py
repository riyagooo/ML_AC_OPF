import pandas as pd
import numpy as np
import os
import random
import argparse
from sklearn.preprocessing import StandardScaler

def parse_args():
    parser = argparse.ArgumentParser(description='Create small IEEE39 dataset for GNN training')
    parser.add_argument('--num-samples', type=int, default=10000, 
                        help='Number of samples to extract')
    parser.add_argument('--input-dir', type=str, default='data/realistic_case39/IEEE39', 
                        help='Input directory with IEEE39 data')
    parser.add_argument('--output-dir', type=str, default='output/ieee39_data_small', 
                        help='Output directory for processed data')
    parser.add_argument('--random-seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define file paths
    setpoints_file = os.path.join(args.input_dir, 'IEEE_39BUS_setpoints.csv')
    labels_file = os.path.join(args.input_dir, 'IEEE_39BUS_labels.csv')
    
    # Load the data
    print(f"Loading IEEE39 data from {setpoints_file} and {labels_file}")
    setpoints = pd.read_csv(setpoints_file)
    labels = pd.read_csv(labels_file)
    
    print(f"Loaded {len(setpoints)} samples from setpoints")
    print(f"Loaded {len(labels)} samples from labels")
    
    # Randomly select a subset of samples
    if args.num_samples < len(setpoints):
        print(f"Selecting {args.num_samples} random samples from the dataset")
        random_indices = random.sample(range(len(setpoints)), args.num_samples)
        setpoints = setpoints.iloc[random_indices]
        labels = labels.iloc[random_indices]
    
    # Save combined data for reference
    combined_data = pd.concat([setpoints, labels], axis=1)
    combined_data.to_csv(f"{args.output_dir}/ieee39_combined_data.csv", index=False)
    
    # Extract features for Direct Prediction task
    # For direct prediction, we'll use generator setpoints as inputs and voltage magnitudes as targets
    generator_cols = [col for col in setpoints.columns if col.startswith('PG')]
    voltage_cols = [col for col in setpoints.columns if col.startswith('VM')]
    
    # Get data for direct prediction
    X_direct = setpoints[generator_cols]
    y_direct = setpoints[voltage_cols]
    
    # Normalize the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X_direct)
    y_scaled = scaler_y.fit_transform(y_direct)
    
    # Save scalers for later use
    np.save(f"{args.output_dir}/scaler_X_mean.npy", scaler_X.mean_)
    np.save(f"{args.output_dir}/scaler_X_scale.npy", scaler_X.scale_)
    np.save(f"{args.output_dir}/scaler_y_mean.npy", scaler_y.mean_)
    np.save(f"{args.output_dir}/scaler_y_scale.npy", scaler_y.scale_)
    
    # Save raw data
    X_direct.to_csv(f"{args.output_dir}/X_direct.csv", index=False)
    y_direct.to_csv(f"{args.output_dir}/y_direct.csv", index=False)
    
    # Save scaled data
    pd.DataFrame(X_scaled, columns=X_direct.columns).to_csv(f"{args.output_dir}/X_direct_scaled.csv", index=False)
    pd.DataFrame(y_scaled, columns=y_direct.columns).to_csv(f"{args.output_dir}/y_direct_scaled.csv", index=False)
    
    # Also save in numpy format
    np.save(f"{args.output_dir}/X_direct.npy", X_direct.values)
    np.save(f"{args.output_dir}/y_direct.npy", y_direct.values)
    np.save(f"{args.output_dir}/X_direct_scaled.npy", X_scaled)
    np.save(f"{args.output_dir}/y_direct_scaled.npy", y_scaled)
    
    print(f"Saved data for Direct Prediction:")
    print(f"  - Input features: {len(generator_cols)} generator setpoints")
    print(f"  - Target variables: {len(voltage_cols)} voltage magnitudes")
    print(f"  - Total samples: {len(X_direct)}")
    print(f"  - Data normalized using StandardScaler")
    
    # Also save class labels for feasibility classification
    if 'class' in labels.columns:
        class_data = labels['class']
        class_data.to_csv(f"{args.output_dir}/y_feasibility.csv", index=False)
        np.save(f"{args.output_dir}/y_feasibility.npy", class_data.values)
        
        # Count feasible/infeasible samples
        feasible_count = class_data.sum()
        infeasible_count = len(class_data) - feasible_count
        print(f"Feasibility information:")
        print(f"  - Feasible samples: {feasible_count} ({feasible_count/len(class_data)*100:.1f}%)")
        print(f"  - Infeasible samples: {infeasible_count} ({infeasible_count/len(class_data)*100:.1f}%)")

if __name__ == '__main__':
    main() 