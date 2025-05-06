#!/usr/bin/env python
"""
Simplified demonstration of Warm-Starting approach for ML-OPF project.
This script shows the concept of warm-starting using simulated data.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ML-OPF Warm-Starting Demo')
    parser.add_argument('--case', type=str, default='case39', help='Case name (default: case39)')
    parser.add_argument('--num-samples', type=int, default=10, 
                        help='Number of samples to simulate (default: 10)')
    parser.add_argument('--output-dir', type=str, default='output/case39_warm_starting_demo',
                        help='Output directory for results')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Running warm-starting demonstration for {args.case} with {args.num_samples} samples...")
    
    # Create simulated results
    n_samples = args.num_samples
    no_warm_times = np.random.uniform(0.5, 2.0, size=n_samples)  # Random times without warm-start
    
    # Warm-start typically makes solving 20-50% faster
    warm_times = no_warm_times * np.random.uniform(0.5, 0.8, size=n_samples)
    
    # Calculate speedups and objective differences
    speedups = no_warm_times / warm_times
    obj_diffs = np.random.uniform(0.0, 0.1, size=n_samples)  # Very small differences in objective value
    
    # Print the results
    print("\nSimulated warm-starting results:")
    
    for i in range(n_samples):
        print(f"\nSample {i+1}/{n_samples}:")
        print(f"  Without warm-start: {no_warm_times[i]:.4f} seconds")
        print(f"  With warm-start:    {warm_times[i]:.4f} seconds")
        print(f"  Speedup:           {speedups[i]:.2f}x")
        print(f"  Objective diff:    {obj_diffs[i]:.6f}%")
    
    # Create a results dataframe
    results = {
        'sample': list(range(1, n_samples + 1)),
        'no_warm_time': no_warm_times,
        'warm_time': warm_times,
        'speedup': speedups,
        'obj_diff': obj_diffs
    }
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    csv_path = os.path.join(args.output_dir, 'warm_starting_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Create summary statistics
    avg_speedup = np.mean(speedups)
    min_speedup = np.min(speedups)
    max_speedup = np.max(speedups)
    avg_obj_diff = np.mean(obj_diffs)
    
    # Save summary to text file
    summary_path = os.path.join(args.output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Warm-Starting Simulation Summary\n")
        f.write("===============================\n\n")
        f.write(f"Case: {args.case}\n")
        f.write(f"Number of samples: {n_samples}\n\n")
        f.write("Performance Metrics:\n")
        f.write(f"  Average solving time without warm-starting: {np.mean(no_warm_times):.4f} seconds\n")
        f.write(f"  Average solving time with warm-starting:    {np.mean(warm_times):.4f} seconds\n")
        f.write(f"  Average speedup:                            {avg_speedup:.2f}x\n")
        f.write(f"  Speedup range:                              {min_speedup:.2f}x - {max_speedup:.2f}x\n")
        f.write(f"  Average objective value difference:         {avg_obj_diff:.6f}%\n\n")
        f.write("Note: These are simulated results to demonstrate the warm-starting concept.\n")
        f.write("In real OPF problems, warm-starting can provide significant performance benefits.\n")
    
    print(f"Summary saved to {summary_path}")
    
    # Create visualizations
    plt.figure(figsize=(10, 6))
    plt.bar(range(n_samples), speedups)
    plt.axhline(y=1.0, color='r', linestyle='--', label='No speedup')
    plt.xlabel('Sample')
    plt.ylabel('Speedup Factor')
    plt.title(f'Warm-Starting Speedup Simulation for {args.case}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = os.path.join(args.output_dir, 'speedup_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    
    # Create comparison plot
    plt.figure(figsize=(12, 6))
    
    width = 0.35
    indices = np.arange(n_samples)
    
    plt.bar(indices - width/2, no_warm_times, width, label='Without Warm-Starting')
    plt.bar(indices + width/2, warm_times, width, label='With Warm-Starting')
    
    plt.xlabel('Sample')
    plt.ylabel('Solving Time (seconds)')
    plt.title(f'OPF Solving Time Comparison for {args.case}')
    plt.xticks(indices)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save comparison plot
    comparison_path = os.path.join(args.output_dir, 'time_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {comparison_path}")
    
    print("\nWarm-starting demonstration completed successfully!")

if __name__ == '__main__':
    main() 