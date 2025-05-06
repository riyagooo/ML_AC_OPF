#!/usr/bin/env python3
"""
Module for generating timing visualizations comparing ML-based AC-OPF approaches
with traditional solvers. This can be imported and called from other scripts.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import json

# Constants
TIMING_DATA_FILE = "output/timing_data.json"

def load_timing_data():
    """
    Load timing data from JSON file or use default values if file doesn't exist
    
    Returns:
    --------
    tuple
        (methods, inference_times)
    """
    if os.path.exists(TIMING_DATA_FILE):
        try:
            with open(TIMING_DATA_FILE, 'r') as f:
                data = json.load(f)
                print(f"Loaded actual timing measurements from {TIMING_DATA_FILE}")
                return data["methods"], data["inference_times"]
        except Exception as e:
            print(f"Error loading timing data: {e}")
    
    # Default values if file doesn't exist or there's an error
    print("Using default timing values (not actual measurements)")
    methods = [
        'Traditional AC-OPF Solver',
        'Standard Feedforward',
        'Standard GNN',
        'Advanced Feedforward',
        'Advanced GNN'
    ]
    
    inference_times = [342.0, 3.2, 4.8, 5.1, 7.3]
    return methods, inference_times

def generate_timing_visualizations(output_dir='output/visualizations'):
    """
    Generate timing visualizations comparing ML approaches with traditional AC-OPF solvers.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load actual timing data or use defaults
    methods, inference_times = load_timing_data()
    
    # Calculate speedup relative to traditional solver
    traditional_time = next((t for m, t in zip(methods, inference_times) 
                            if 'Traditional' in m), inference_times[0])
    speedups = [traditional_time/time for time in inference_times]
    
    # Colors for the bars
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
    
    # Create the figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Inference time per sample
    x = np.arange(len(methods))
    ax1.bar(x, inference_times, color=colors[:len(methods)])
    ax1.set_ylabel('Inference Time (ms)')
    ax1.set_title('Computation Time per Sample')
    ax1.set_yscale('log')  # Use log scale for better visualization
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, time in enumerate(inference_times):
        ax1.text(i, time * 1.1, f"{time:.1f} ms", ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Speedup relative to traditional solver
    ax2.bar(x, speedups, color=colors[:len(methods)])
    ax2.set_ylabel('Speedup Factor (×)')
    ax2.set_title('Speedup Relative to Traditional AC-OPF Solver')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.set_ylim(0, max(speedups)*1.1)
    
    for i, speedup in enumerate(speedups):
        ax2.text(i, speedup * 1.02, f"{speedup:.1f}×", ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'computation_time_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create scaling visualization
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Number of OPF problems to solve
    problem_counts = [100, 1000, 10000]
    
    # Calculate solution times for different numbers of problems (in seconds)
    solution_times = {
        method: [inference_times[i] * count / 1000 for count in problem_counts]
        for i, method in enumerate(methods)
    }
    
    # Set positions for the bars
    bar_width = 0.15
    positions = np.arange(len(problem_counts))
    
    # Create bars
    for i, method in enumerate(methods):
        offset = (i - len(methods)/2 + 0.5) * bar_width
        bars = ax.bar(positions + offset, solution_times[method], bar_width, 
                     label=method, color=colors[i % len(colors)])
        
        # Add time labels inside or above the bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if method == methods[0]:  # Traditional solver
                # Convert to minutes if > 60 seconds
                if height > 60:
                    time_str = f"{height/60:.1f} min"
                else:
                    time_str = f"{height:.1f} s"
                ax.text(bar.get_x() + bar.get_width()/2, height/2, time_str,
                       ha='center', va='center', fontsize=9, color='white', fontweight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width()/2, height*1.05, f"{height:.1f} s",
                       ha='center', va='bottom', fontsize=8)
    
    # Customize the plot
    ax.set_ylabel('Computation Time (seconds)')
    ax.set_title('Time to Solve Multiple AC-OPF Problems')
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{count} Problems" for count in problem_counts])
    ax.legend(title="Method", loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scaling_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a line plot showing how computation time scales with problem size
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # More detailed problem counts for smooth curve
    detailed_counts = np.logspace(1, 5, 50)  # From 10 to 100,000 problems
    
    # Calculate times for each method
    for i, method in enumerate(methods):
        times = [inference_times[i] * count / 1000 for count in detailed_counts]
        ax.plot(detailed_counts, times, marker='', linewidth=2.5, label=method, color=colors[i % len(colors)])
    
    # Add log scales for better visualization
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add reference annotations
    for count in [10000]:  # Only annotate one point to avoid clutter
        # Traditional solver
        traditional_idx = 0
        traditional_time = inference_times[traditional_idx] * count / 1000
        
        # Fastest ML method
        ml_times = [(m, t) for m, t in zip(methods, inference_times) if 'Traditional' not in m]
        if ml_times:
            fastest_ml = min(ml_times, key=lambda x: x[1])
            fastest_ml_method = fastest_ml[0]
            fastest_ml_time = fastest_ml[1] * count / 1000
        else:
            fastest_ml_method = "ML Method"
            fastest_ml_time = inference_times[1] * count / 1000
        
        # Convert to appropriate time unit for traditional
        if traditional_time > 3600:
            unit = "hours"
            traditional_time_display = traditional_time / 3600
        elif traditional_time > 60:
            unit = "minutes"
            traditional_time_display = traditional_time / 60
        else:
            unit = "seconds"
            traditional_time_display = traditional_time
            
        # Add annotations
        ax.annotate(f"Traditional: {traditional_time_display:.1f} {unit}",
                   xy=(count, inference_times[traditional_idx] * count / 1000),
                   xytext=(count*1.2, inference_times[traditional_idx] * count / 1000 * 1.2),
                   arrowprops=dict(arrowstyle="->", color='black'))
        
        ax.annotate(f"{fastest_ml_method}: {fastest_ml_time:.1f} seconds",
                   xy=(count, fastest_ml_time),
                   xytext=(count*1.2, fastest_ml_time * 0.8),
                   arrowprops=dict(arrowstyle="->", color='black'))
    
    # Customize the plot
    ax.set_xlabel('Number of AC-OPF Problems')
    ax.set_ylabel('Computation Time (seconds)')
    ax.set_title('Scaling of Computation Time with Problem Size')
    ax.legend(title="Method")
    ax.grid(True, which="both", linestyle='--', alpha=0.7)
    
    # Add horizontal lines for reference time scales
    reference_times = [
        (1, "1 second"),
        (60, "1 minute"),
        (3600, "1 hour"),
        (3600*24, "1 day")
    ]
    
    for time, label in reference_times:
        ax.axhline(y=time, linestyle='--', color='gray', alpha=0.5)
        ax.text(10, time*1.1, label, fontsize=8, ha='left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'computation_scaling.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Timing visualizations saved to {output_dir}/")

if __name__ == "__main__":
    # If the script is run directly, generate visualizations in the default directory
    generate_timing_visualizations()
    print("Done!") 