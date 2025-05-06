import numpy as np
import pandas as pd
import time
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Callable, Optional
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('timing_benchmark')

def benchmark_execution_time(model, data_loader, n_runs=10, warmup_runs=3, device='cpu'):
    """
    Benchmark the execution time of an ML model.
    
    Args:
        model: PyTorch model to benchmark
        data_loader: DataLoader containing test data
        n_runs: Number of runs to average over
        warmup_runs: Number of warmup runs to perform (not counted in average)
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        Dict with timing statistics
    """
    model.eval()
    model = model.to(device)
    
    # Get a batch of data for consistent timing
    X_batch, _ = next(iter(data_loader))
    X_batch = X_batch.to(device)
    
    # Warmup runs
    logger.info(f"Performing {warmup_runs} warmup runs...")
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(X_batch)
    
    # Timed runs
    logger.info(f"Performing {n_runs} timed runs...")
    execution_times = []
    
    for run in range(n_runs):
        # Time execution
        start_time = time.time()
        
        with torch.no_grad():
            _ = model(X_batch)
            
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        execution_times.append(execution_time)
        
        if (run + 1) % 5 == 0:
            logger.info(f"Completed {run + 1}/{n_runs} runs")
    
    # Calculate statistics
    avg_time = np.mean(execution_times)
    std_time = np.std(execution_times)
    min_time = np.min(execution_times)
    max_time = np.max(execution_times)
    
    logger.info(f"Execution time (ms): Avg={avg_time:.2f}, Std={std_time:.2f}, Min={min_time:.2f}, Max={max_time:.2f}")
    
    return {
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'all_times_ms': execution_times
    }

def benchmark_solver_time(solver_fn, data_loader, n_runs=10):
    """
    Benchmark the execution time of a traditional solver.
    
    Args:
        solver_fn: Function that runs the solver on a batch of data
        data_loader: DataLoader containing test data
        n_runs: Number of runs to average over
        
    Returns:
        Dict with timing statistics
    """
    # Get a batch of data for consistent timing
    X_batch, _ = next(iter(data_loader))
    X_np = X_batch.numpy()
    
    # Timed runs
    logger.info(f"Performing {n_runs} solver runs...")
    execution_times = []
    
    for run in range(n_runs):
        # Time execution
        start_time = time.time()
        
        solver_fn(X_np)
            
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        execution_times.append(execution_time)
        
        if (run + 1) % 1 == 0:
            logger.info(f"Completed {run + 1}/{n_runs} solver runs")
    
    # Calculate statistics
    avg_time = np.mean(execution_times)
    std_time = np.std(execution_times)
    min_time = np.min(execution_times)
    max_time = np.max(execution_times)
    
    logger.info(f"Solver time (ms): Avg={avg_time:.2f}, Std={std_time:.2f}, Min={min_time:.2f}, Max={max_time:.2f}")
    
    return {
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'all_times_ms': execution_times
    }

def compare_execution_times(ml_time_stats, solver_time_stats):
    """
    Compare execution times between ML model and solver.
    
    Args:
        ml_time_stats: Dict with ML model timing statistics
        solver_time_stats: Dict with solver timing statistics
        
    Returns:
        Dict with comparison metrics
    """
    ml_avg = ml_time_stats['avg_time_ms']
    solver_avg = solver_time_stats['avg_time_ms']
    
    speedup_factor = solver_avg / ml_avg
    time_reduction_percent = (1 - ml_avg / solver_avg) * 100
    
    logger.info(f"ML avg time: {ml_avg:.2f} ms")
    logger.info(f"Solver avg time: {solver_avg:.2f} ms")
    logger.info(f"Speedup factor: {speedup_factor:.2f}x")
    logger.info(f"Time reduction: {time_reduction_percent:.2f}%")
    
    return {
        'ml_avg_time_ms': ml_avg,
        'solver_avg_time_ms': solver_avg,
        'speedup_factor': speedup_factor,
        'time_reduction_percent': time_reduction_percent
    }

def visualize_execution_times(ml_time_stats, solver_time_stats, output_dir, method_name=""):
    """
    Create visualization comparing ML model and solver execution times.
    
    Args:
        ml_time_stats: Dict with ML model timing statistics
        solver_time_stats: Dict with solver timing statistics
        output_dir: Directory to save visualizations
        method_name: Name of the ML method for labeling
        
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Bar chart comparing average execution times
    plt.figure(figsize=(10, 6))
    labels = [f"{method_name} ML Model", "Traditional Solver"]
    times = [ml_time_stats['avg_time_ms'], solver_time_stats['avg_time_ms']]
    colors = ['dodgerblue', 'firebrick']
    
    bars = plt.bar(labels, times, color=colors, width=0.5)
    
    # Add error bars showing standard deviation
    plt.errorbar(
        x=labels, 
        y=times, 
        yerr=[ml_time_stats['std_time_ms'], solver_time_stats['std_time_ms']], 
        fmt='none', 
        color='black', 
        capsize=5
    )
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 5,
            f'{height:.2f} ms',
            ha='center', 
            va='bottom',
            fontweight='bold'
        )
    
    # Add speedup annotation
    speedup = solver_time_stats['avg_time_ms'] / ml_time_stats['avg_time_ms']
    time_reduction = (1 - ml_time_stats['avg_time_ms'] / solver_time_stats['avg_time_ms']) * 100
    
    plt.annotate(
        f"Speedup: {speedup:.2f}x\nTime Reduction: {time_reduction:.2f}%",
        xy=(0.5, 0.9),
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8),
        ha='center',
        fontsize=12
    )
    
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.title(f'Average Execution Time Comparison: {method_name}', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'execution_time_bar_{method_name.lower().replace(" ", "_")}.png'))
    plt.close()
    
    # 2. Violin plot showing distribution of execution times
    plt.figure(figsize=(10, 6))
    
    data = {
        f"{method_name} ML Model": ml_time_stats['all_times_ms'],
        "Traditional Solver": solver_time_stats['all_times_ms']
    }
    
    df = pd.DataFrame({
        'Execution Time (ms)': np.concatenate([ml_time_stats['all_times_ms'], solver_time_stats['all_times_ms']]),
        'Method': [f"{method_name} ML Model"] * len(ml_time_stats['all_times_ms']) + 
                 ["Traditional Solver"] * len(solver_time_stats['all_times_ms'])
    })
    
    sns.violinplot(x='Method', y='Execution Time (ms)', data=df, palette=colors)
    
    # Add individual data points for clarity
    sns.stripplot(x='Method', y='Execution Time (ms)', data=df, color='black', alpha=0.5, size=4)
    
    plt.title(f'Execution Time Distribution: {method_name}', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'execution_time_violin_{method_name.lower().replace(" ", "_")}.png'))
    plt.close()
    
    # 3. Horizontal bar chart for multiple models if available
    if hasattr(visualize_execution_times, 'comparison_data'):
        visualize_execution_times.comparison_data.append({
            'method': method_name,
            'ml_time': ml_time_stats['avg_time_ms'],
            'solver_time': solver_time_stats['avg_time_ms'],
            'speedup': speedup,
            'reduction': time_reduction
        })
    else:
        visualize_execution_times.comparison_data = [{
            'method': method_name,
            'ml_time': ml_time_stats['avg_time_ms'],
            'solver_time': solver_time_stats['avg_time_ms'],
            'speedup': speedup,
            'reduction': time_reduction
        }]
    
    # Create combined comparison chart if we have multiple methods
    if len(visualize_execution_times.comparison_data) > 1:
        create_multi_method_comparison(visualize_execution_times.comparison_data, output_dir)

def create_multi_method_comparison(comparison_data, output_dir):
    """
    Create a visualization comparing multiple ML methods to the solver.
    
    Args:
        comparison_data: List of dicts with comparison metrics for each method
        output_dir: Directory to save visualizations
        
    Returns:
        None
    """
    # Sort methods by speedup (descending)
    comparison_data = sorted(comparison_data, key=lambda x: x['speedup'], reverse=True)
    
    # Create DataFrame for plotting
    df = pd.DataFrame(comparison_data)
    
    # 1. Speedup comparison
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar chart of speedup factors
    bars = plt.barh(df['method'], df['speedup'], color='dodgerblue')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.1,
            bar.get_y() + bar.get_height()/2,
            f'{width:.2f}x',
            ha='left',
            va='center',
            fontweight='bold'
        )
    
    plt.xlabel('Speedup Factor (higher is better)', fontsize=12)
    plt.title('ML Methods Speedup Comparison', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'speedup_comparison.png'))
    plt.close()
    
    # 2. Time reduction percentage comparison
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar chart of time reduction percentages
    bars = plt.barh(df['method'], df['reduction'], color='forestgreen')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.5,
            bar.get_y() + bar.get_height()/2,
            f'{width:.2f}%',
            ha='left',
            va='center',
            fontweight='bold'
        )
    
    plt.xlabel('Time Reduction Percentage (%)', fontsize=12)
    plt.title('ML Methods Time Reduction Comparison', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'time_reduction_comparison.png'))
    plt.close()
    
    # 3. Combined absolute timing comparison
    plt.figure(figsize=(12, 8))
    
    # Prepare data
    methods = df['method']
    ml_times = df['ml_time']
    solver_times = df['solver_time']
    
    # Create positions for side-by-side bars
    x = np.arange(len(methods))
    width = 0.35
    
    # Create bars
    plt.bar(x - width/2, ml_times, width, label='ML Model', color='dodgerblue')
    plt.bar(x + width/2, solver_times, width, label='Traditional Solver', color='firebrick')
    
    # Add labels and title
    plt.xlabel('Execution Time (ms)', fontsize=12)
    plt.ylabel('Method', fontsize=12)
    plt.title('Execution Time Comparison Across Methods', fontsize=14)
    plt.yticks(x, methods)
    plt.legend()
    
    # Make it horizontal
    plt.gca().invert_yaxis()  # Invert y-axis to match previous charts
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'absolute_time_comparison.png'))
    plt.close()
    
    # 4. Save comparison data as CSV
    df.to_csv(os.path.join(output_dir, 'timing_comparison.csv'), index=False)
    
    # 5. Create a summary text file
    with open(os.path.join(output_dir, 'timing_summary.txt'), 'w') as f:
        f.write("=========================================\n")
        f.write("ML-AC-OPF TIMING PERFORMANCE SUMMARY\n")
        f.write("=========================================\n\n")
        
        f.write("Method                    | ML Time (ms) | Solver Time (ms) | Speedup | Reduction (%)\n")
        f.write("--------------------------+-------------+-----------------+---------+---------------\n")
        
        for row in comparison_data:
            f.write(f"{row['method']:<25} | {row['ml_time']:11.2f} | {row['solver_time']:15.2f} | {row['speedup']:7.2f}x | {row['reduction']:13.2f}%\n")
        
        f.write("\n")
        best_method = comparison_data[0]['method']
        best_speedup = comparison_data[0]['speedup']
        f.write(f"The best performing method is '{best_method}' with a {best_speedup:.2f}x speedup over the traditional solver.\n")
        f.write(f"This represents a {comparison_data[0]['reduction']:.2f}% reduction in execution time.\n")

# Function to run a dummy solver for testing
def dummy_solver(input_data):
    """Simulates a traditional solver with artificial delay"""
    time.sleep(0.2)  # 200ms artificial delay
    return np.random.rand(input_data.shape[0], 10)  # Random output 