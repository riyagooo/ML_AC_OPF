#!/usr/bin/env python
"""
Data exploration script for IEEE 39-bus dataset.

This script analyzes binding constraints and creates visualizations for the IEEE 39-bus dataset.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from matplotlib.ticker import PercentFormatter

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('data_exploration')

def load_data(data_dir="data/realistic_case39/IEEE39", sample_size=None, random_seed=42):
    """
    Load and process IEEE 39-bus data.
    
    Args:
        data_dir: Directory containing the data files
        sample_size: Number of samples to use (None for all data)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (setpoints, labels) DataFrames
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Load data files
    setpoints_file = os.path.join(data_dir, "IEEE_39BUS_setpoints.csv")
    labels_file = os.path.join(data_dir, "IEEE_39BUS_labels.csv")
    
    logger.info(f"Loading data from {data_dir}")
    setpoints = pd.read_csv(setpoints_file)
    labels = pd.read_csv(labels_file)
    
    # If sample_size specified, take a random subset
    if sample_size is not None and sample_size < len(setpoints):
        idx = np.random.choice(len(setpoints), sample_size, replace=False)
        setpoints = setpoints.iloc[idx].reset_index(drop=True)
        labels = labels.iloc[idx].reset_index(drop=True)
        logger.info(f"Using random sample of {sample_size} operating points")
    else:
        logger.info(f"Using full dataset with {len(setpoints)} operating points")
    
    return setpoints, labels

def analyze_binding_constraints(labels):
    """
    Analyze binding constraints from labels data.
    
    Args:
        labels: DataFrame containing constraint information
        
    Returns:
        Dictionary with analysis results
    """
    # Get overall feasibility statistics
    feasible_count = labels['class'].sum()
    total_count = len(labels)
    feasible_percent = feasible_count / total_count * 100
    
    logger.info(f"Feasibility analysis: {feasible_count} feasible samples out of {total_count} ({feasible_percent:.1f}%)")
    
    # Get binding constraint statistics (non-zero values mean binding)
    binding_stats = {}
    for column in labels.columns:
        if column != 'class':
            binding_count = labels[column].sum()
            binding_percent = binding_count / total_count * 100
            binding_stats[column] = {
                'count': binding_count,
                'total': total_count,
                'percentage': binding_percent
            }
            logger.info(f"{column} constraints: {binding_count} binding out of {total_count} ({binding_percent:.1f}%)")
    
    return {
        'feasible': {
            'count': feasible_count,
            'total': total_count,
            'percentage': feasible_percent
        },
        'binding_constraints': binding_stats
    }

def plot_binding_constraints(stats, labels, output_dir="output/data_exploration"):
    """
    Create visualizations for binding constraints.
    
    Args:
        stats: Statistics dictionary from analyze_binding_constraints
        labels: DataFrame containing constraint information
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    constraints = list(stats['binding_constraints'].keys())
    percentages = [stats['binding_constraints'][c]['percentage'] for c in constraints]
    counts = [stats['binding_constraints'][c]['count'] for c in constraints]
    total = stats['binding_constraints'][constraints[0]]['total']
    
    # 1. Create horizontal bar chart of binding percentages
    plt.figure(figsize=(10, 6))
    bars = plt.barh(constraints, percentages, color='skyblue')
    plt.xlabel('Percentage of Operating Points (%)')
    plt.title('Percentage of Operating Points with Binding Constraints')
    plt.xlim(0, 100)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add percentage labels to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'binding_constraints_percentage.png'))
    plt.close()
    
    # 2. Create pie chart for feasibility
    plt.figure(figsize=(8, 8))
    feasible = stats['feasible']['count']
    infeasible = stats['feasible']['total'] - feasible
    plt.pie([feasible, infeasible], 
            labels=['Feasible', 'Infeasible'],
            autopct='%1.1f%%',
            colors=['#4CAF50', '#F44336'],
            startangle=90,
            explode=(0.1, 0))
    plt.title('Feasible vs. Infeasible Operating Points')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feasibility_pie.png'))
    plt.close()
    
    # 3. Create stacked bar chart showing binding vs. non-binding for each constraint
    plt.figure(figsize=(12, 7))
    bar_width = 0.6
    non_binding_counts = [total - count for count in counts]
    
    # Create stacked bars
    plt.bar(constraints, counts, bar_width, label='Binding', color='#FF9800')
    plt.bar(constraints, non_binding_counts, bar_width, bottom=counts, label='Non-binding', color='#2196F3')
    
    # Add percentages inside the binding part of each bar
    for i, (constraint, count, percentage) in enumerate(zip(constraints, counts, percentages)):
        if percentage > 10:  # Only add text if there's enough space
            plt.text(i, count/2, f'{percentage:.1f}%', ha='center', va='center', color='white', fontweight='bold')
    
    plt.xlabel('Constraint Type')
    plt.ylabel('Number of Operating Points')
    plt.title('Binding vs. Non-binding Constraints')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'binding_constraints_stacked.png'))
    plt.close()
    
    # 4. Heatmap showing correlation between binding constraints
    constraint_cols = constraints + ['class']
    correlation = labels[constraint_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Binding Constraints and Feasibility')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'binding_constraints_correlation.png'))
    plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")

def analyze_setpoints(setpoints):
    """
    Analyze setpoint data.
    
    Args:
        setpoints: DataFrame containing setpoint values
        
    Returns:
        Dictionary with analysis results
    """
    # Group columns by type (PG, VM, etc.)
    column_groups = {}
    for col in setpoints.columns:
        prefix = col[:2]  # Get first 2 characters (PG, VM, etc.)
        if prefix not in column_groups:
            column_groups[prefix] = []
        column_groups[prefix].append(col)
    
    # Calculate statistics for each group
    group_stats = {}
    for prefix, cols in column_groups.items():
        group_data = setpoints[cols]
        group_stats[prefix] = {
            'mean': group_data.mean(),
            'std': group_data.std(),
            'min': group_data.min(),
            'max': group_data.max(),
            'median': group_data.median()
        }
        logger.info(f"{prefix} setpoints: {len(cols)} variables, mean range: [{group_data.mean().min():.2f}, {group_data.mean().max():.2f}]")
    
    return {
        'column_groups': column_groups,
        'group_stats': group_stats
    }

def plot_setpoint_distributions(setpoints, setpoint_stats, output_dir="output/data_exploration"):
    """
    Create visualizations for setpoint distributions.
    
    Args:
        setpoints: DataFrame containing setpoint values
        setpoint_stats: Statistics dictionary from analyze_setpoints
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Distribution of generator active power (PG)
    if 'PG' in setpoint_stats['column_groups']:
        pg_cols = setpoint_stats['column_groups']['PG']
        
        plt.figure(figsize=(12, 6))
        for col in pg_cols:
            sns.kdeplot(setpoints[col], label=col)
        
        plt.xlabel('Active Power (PG)')
        plt.ylabel('Density')
        plt.title('Distribution of Generator Active Power Setpoints')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pg_distribution.png'))
        plt.close()
    
    # 2. Distribution of voltage magnitudes (VM)
    if 'VM' in setpoint_stats['column_groups']:
        vm_cols = setpoint_stats['column_groups']['VM']
        
        plt.figure(figsize=(12, 6))
        for col in vm_cols:
            sns.kdeplot(setpoints[col], label=col)
        
        plt.xlabel('Voltage Magnitude (VM)')
        plt.ylabel('Density')
        plt.title('Distribution of Bus Voltage Magnitude Setpoints')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'vm_distribution.png'))
        plt.close()
    
    # 3. Create boxplots for each setpoint type
    for prefix, cols in setpoint_stats['column_groups'].items():
        if len(cols) > 1:  # Only create boxplot if there are multiple columns
            plt.figure(figsize=(12, 6))
            
            # Create a long-format DataFrame for seaborn
            df_long = pd.melt(setpoints[cols], var_name='Variable', value_name='Value')
            
            # Create boxplot
            sns.boxplot(x='Variable', y='Value', data=df_long)
            
            plt.title(f'Distribution of {prefix} Setpoints')
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{prefix.lower()}_boxplot.png'))
            plt.close()
    
    logger.info(f"Setpoint visualizations saved to {output_dir}")

def check_for_correlations(setpoints, labels, output_dir="output/data_exploration"):
    """
    Check for correlations between setpoints and binding constraints.
    
    Args:
        setpoints: DataFrame containing setpoint values
        labels: DataFrame containing constraint information
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Create violin plots comparing setpoint distributions for binding vs. non-binding constraints
    constraint_cols = [col for col in labels.columns if col != 'class']
    
    for constraint in constraint_cols:
        # Create a subset of key setpoints (sample 3 PG and 3 VM for clarity)
        pg_cols = [col for col in setpoints.columns if col.startswith('PG')][:3]
        vm_cols = [col for col in setpoints.columns if col.startswith('VM')][:3]
        selected_cols = pg_cols + vm_cols
        
        # Prepare data
        binding_mask = (labels[constraint] == 1)
        data_pairs = []
        
        for col in selected_cols:
            binding_values = setpoints.loc[binding_mask, col]
            non_binding_values = setpoints.loc[~binding_mask, col]
            
            data_pairs.append({
                'column': col,
                'binding': binding_values.values,
                'non_binding': non_binding_values.values
            })
        
        # Create violin plots
        plt.figure(figsize=(14, 8))
        
        for i, pair in enumerate(data_pairs):
            plt.subplot(2, len(data_pairs)//2 + len(data_pairs)%2, i+1)
            
            # Create a DataFrame for seaborn
            df = pd.DataFrame({
                'Value': np.concatenate([pair['binding'], pair['non_binding']]),
                'Status': ['Binding'] * len(pair['binding']) + ['Non-binding'] * len(pair['non_binding'])
            })
            
            # Create violin plot
            sns.violinplot(x='Status', y='Value', data=df, palette=['#FF9800', '#2196F3'])
            
            plt.title(pair['column'])
            plt.tight_layout()
        
        plt.suptitle(f'Setpoint Distributions for {constraint} Constraint', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'setpoint_distribution_{constraint}.png'))
        plt.close()
    
    # 2. Create a correlation matrix between key setpoints and constraints
    # Select a subset of setpoint columns to keep the matrix readable
    selected_setpoints = []
    for prefix in ['PG', 'VM']:
        cols = [col for col in setpoints.columns if col.startswith(prefix)]
        selected_setpoints.extend(cols[:3])  # Take first 3 of each type
    
    # Combine setpoints and constraints
    combined_data = pd.concat([setpoints[selected_setpoints], labels[constraint_cols]], axis=1)
    correlation = combined_data.corr()
    
    # Create a heatmap for the correlations between setpoints and constraints
    constraint_correlations = correlation.loc[constraint_cols, selected_setpoints]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(constraint_correlations, annot=True, cmap='coolwarm', vmin=-0.5, vmax=0.5, center=0)
    plt.title('Correlation Between Setpoints and Binding Constraints')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'setpoint_constraint_correlation.png'))
    plt.close()
    
    logger.info(f"Correlation visualizations saved to {output_dir}")

def main():
    # Define parameters
    data_dir = "data/realistic_case39/IEEE39"
    output_dir = "output/data_exploration"
    sample_size = 50000  # Set to None to use all data
    
    # Load data
    setpoints, labels = load_data(data_dir, sample_size)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze and visualize binding constraints
    logger.info("Analyzing binding constraints...")
    stats = analyze_binding_constraints(labels)
    
    logger.info("Creating binding constraint visualizations...")
    plot_binding_constraints(stats, labels, output_dir)
    
    # Analyze and visualize setpoints
    logger.info("Analyzing setpoints...")
    setpoint_stats = analyze_setpoints(setpoints)
    
    logger.info("Creating setpoint visualizations...")
    plot_setpoint_distributions(setpoints, setpoint_stats, output_dir)
    
    # Check for correlations between setpoints and constraints
    logger.info("Analyzing correlations between setpoints and binding constraints...")
    check_for_correlations(setpoints, labels, output_dir)
    
    logger.info("Data exploration completed. Visualizations saved to {}".format(output_dir))

if __name__ == "__main__":
    main() 