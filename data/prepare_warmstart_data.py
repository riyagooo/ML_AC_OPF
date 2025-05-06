import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler

# Command line arguments
setpoints_file = sys.argv[1]  # Setpoints file
labels_file = sys.argv[2]     # Labels file
output_dir = sys.argv[3]      # Output directory
num_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 1000  # Number of samples to use

# Load the data
print(f"Loading IEEE39 data from {setpoints_file} and {labels_file}")
setpoints = pd.read_csv(setpoints_file)
labels = pd.read_csv(labels_file)

# Use a subset of data if specified
if num_samples < len(setpoints):
    setpoints = setpoints.iloc[:num_samples]
    labels = labels.iloc[:num_samples]

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# For warm starting, we'll split the generator setpoints:
# - Some will be inputs (representing initial conditions or system demand)
# - Others will be part of the target (to predict good starting points)

# Get generator setpoint columns
pg_cols = [col for col in setpoints.columns if col.startswith('PG')]
qg_cols = [col for col in setpoints.columns if col.startswith('QG')]
vm_cols = [col for col in setpoints.columns if col.startswith('VM')]
va_cols = [col for col in setpoints.columns if col.startswith('VA')]

# Split PG columns into input and target
num_input_gens = len(pg_cols) // 2  # Use half for input, half for target
input_pg_cols = pg_cols[:num_input_gens]
target_pg_cols = pg_cols[num_input_gens:]

# Inputs are a subset of generator setpoints
input_cols = input_pg_cols
X = setpoints[input_cols]

# Target is the remaining gens plus voltage setpoints
target_cols = target_pg_cols + vm_cols + qg_cols
y = setpoints[target_cols]

# Normalize data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Save scalers
np.save(os.path.join(output_dir, "scaler_X_mean.npy"), scaler_X.mean_)
np.save(os.path.join(output_dir, "scaler_X_scale.npy"), scaler_X.scale_)
np.save(os.path.join(output_dir, "scaler_y_mean.npy"), scaler_y.mean_)
np.save(os.path.join(output_dir, "scaler_y_scale.npy"), scaler_y.scale_)

# Save raw data
X.to_csv(os.path.join(output_dir, 'X_warmstart.csv'), index=False)
y.to_csv(os.path.join(output_dir, 'y_warmstart.csv'), index=False)

# Save scaled data
pd.DataFrame(X_scaled, columns=X.columns).to_csv(os.path.join(output_dir, 'X_warmstart_scaled.csv'), index=False)
pd.DataFrame(y_scaled, columns=y.columns).to_csv(os.path.join(output_dir, 'y_warmstart_scaled.csv'), index=False)

# Save as numpy arrays (needed by some scripts)
np.save(os.path.join(output_dir, 'X_warmstart.npy'), X.values)
np.save(os.path.join(output_dir, 'y_warmstart.npy'), y.values)
np.save(os.path.join(output_dir, 'X_warmstart_scaled.npy'), X_scaled)
np.save(os.path.join(output_dir, 'y_warmstart_scaled.npy'), y_scaled)

# Create specific formats needed by existing scripts
warmstart_arr = X.values
np.save(os.path.join(output_dir, 'case39_realistic_load_data.npy'), warmstart_arr)

print(f"Prepared warm starting data:")
print(f"  - Inputs: {len(input_cols)} generator setpoints")
print(f"  - Targets: {len(target_cols)} variables (generators, voltages)")
print(f"  - Total samples: {len(X)}")
print(f"  - Data normalized using StandardScaler")
