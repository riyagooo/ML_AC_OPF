import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

def load_metrics(metrics_file):
    """Load metrics from a metrics.txt file"""
    metrics = {}
    with open(metrics_file, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(': ', 1)
                metrics[key] = float(value)
    return metrics

def extract_r2_values(metrics_file):
    """Extract R² values for each output dimension"""
    r2_values = []
    with open(metrics_file, 'r') as f:
        for line in f:
            if 'R^2 for output' in line:
                r2_value = float(line.strip().split(': ')[1])
                r2_values.append(r2_value)
    return r2_values

def plot_comparison(gnn_metrics, ffn_metrics, output_dir="output/model_comparison"):
    """Generate comparison plots for GNN vs FFN models"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract common metrics
    metrics_to_compare = ['test_loss', 'mse', 'mae', 'r2']
    metrics_data = {
        'Metric': [],
        'Value': [],
        'Model': []
    }
    
    for metric in metrics_to_compare:
        if metric in gnn_metrics:
            metrics_data['Metric'].append(metric)
            metrics_data['Value'].append(gnn_metrics[metric])
            metrics_data['Model'].append('GNN')
            
            metrics_data['Metric'].append(metric)
            metrics_data['Value'].append(ffn_metrics[metric])
            metrics_data['Model'].append('FFN')
    
    # Create metrics comparison dataframe
    metrics_df = pd.DataFrame(metrics_data)
    
    # Plot overall metrics comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Metric', y='Value', hue='Model', data=metrics_df)
    plt.title('Comparison of GNN vs FFN Models')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'))
    plt.close()
    
    # Extract R² values for each output dimension
    gnn_r2_values = extract_r2_values(os.path.join('output/direct_prediction_gnn_fast', 'metrics.txt'))
    ffn_r2_values = extract_r2_values(os.path.join('output/direct_prediction_ffn_fast', 'metrics.txt'))
    
    # Create dataframe for R² values
    r2_data = {
        'Output Dimension': [],
        'R² Value': [],
        'Model': []
    }
    
    for i, (gnn_r2, ffn_r2) in enumerate(zip(gnn_r2_values, ffn_r2_values), 1):
        r2_data['Output Dimension'].append(f'Output {i}')
        r2_data['R² Value'].append(gnn_r2)
        r2_data['Model'].append('GNN')
        
        r2_data['Output Dimension'].append(f'Output {i}')
        r2_data['R² Value'].append(ffn_r2)
        r2_data['Model'].append('FFN')
    
    r2_df = pd.DataFrame(r2_data)
    
    # Plot R² comparison by output dimension
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Output Dimension', y='R² Value', hue='Model', data=r2_df)
    plt.title('R² Comparison by Output Dimension')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r2_comparison.png'))
    plt.close()
    
    # Create a comparison table
    comparison_table = pd.DataFrame({
        'Metric': ['test_loss', 'mse', 'mae', 'Average R²'],
        'GNN': [gnn_metrics.get('test_loss', '-'), 
                gnn_metrics.get('mse', '-'), 
                gnn_metrics.get('mae', '-'), 
                gnn_metrics.get('r2', '-')],
        'FFN': [ffn_metrics.get('test_loss', '-'), 
                ffn_metrics.get('mse', '-'), 
                ffn_metrics.get('mae', '-'), 
                ffn_metrics.get('r2', '-')],
        'Difference': [
            ffn_metrics.get('test_loss', 0) - gnn_metrics.get('test_loss', 0),
            ffn_metrics.get('mse', 0) - gnn_metrics.get('mse', 0),
            ffn_metrics.get('mae', 0) - gnn_metrics.get('mae', 0),
            ffn_metrics.get('r2', 0) - gnn_metrics.get('r2', 0)
        ]
    })
    
    # Save comparison table to CSV
    comparison_table.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    # Create a detailed HTML report
    html_content = f"""
    <html>
    <head>
        <title>Model Comparison: GNN vs FFN</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .container {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
            .image-container {{ width: 48%; margin-bottom: 20px; }}
            img {{ max-width: 100%; height: auto; }}
            .highlight {{ font-weight: bold; color: #27ae60; }}
            .footer {{ margin-top: 30px; color: #7f8c8d; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <h1>Model Comparison: Graph Neural Network vs Feedforward Neural Network</h1>
        <p>Comparison of model performance metrics for AC-OPF prediction task on IEEE39 dataset.</p>
        
        <h2>Summary Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>GNN</th>
                <th>FFN</th>
                <th>Difference (FFN-GNN)</th>
            </tr>
            <tr>
                <td>Test Loss</td>
                <td>{gnn_metrics.get('test_loss', '-'):.6f}</td>
                <td>{ffn_metrics.get('test_loss', '-'):.6f}</td>
                <td>{ffn_metrics.get('test_loss', 0) - gnn_metrics.get('test_loss', 0):.6f}</td>
            </tr>
            <tr>
                <td>MSE</td>
                <td>{gnn_metrics.get('mse', '-'):.6f}</td>
                <td>{ffn_metrics.get('mse', '-'):.6f}</td>
                <td>{ffn_metrics.get('mse', 0) - gnn_metrics.get('mse', 0):.6f}</td>
            </tr>
            <tr>
                <td>MAE</td>
                <td>{gnn_metrics.get('mae', '-'):.6f}</td>
                <td>{ffn_metrics.get('mae', '-'):.6f}</td>
                <td>{ffn_metrics.get('mae', 0) - gnn_metrics.get('mae', 0):.6f}</td>
            </tr>
            <tr>
                <td>Average R²</td>
                <td>{gnn_metrics.get('r2', '-'):.6f}</td>
                <td>{ffn_metrics.get('r2', '-'):.6f}</td>
                <td class="{'highlight' if ffn_metrics.get('r2', 0) > gnn_metrics.get('r2', 0) else ''}">
                    {ffn_metrics.get('r2', 0) - gnn_metrics.get('r2', 0):.6f}
                </td>
            </tr>
        </table>
        
        <h2>Visualization</h2>
        <div class="container">
            <div class="image-container">
                <h3>Overall Metrics</h3>
                <img src="metrics_comparison.png" alt="Metrics Comparison">
            </div>
            <div class="image-container">
                <h3>R² by Output Dimension</h3>
                <img src="r2_comparison.png" alt="R² Comparison">
            </div>
        </div>
        
        <h2>Training Configuration</h2>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Epochs</td>
                <td>20</td>
            </tr>
            <tr>
                <td>Batch Size</td>
                <td>128</td>
            </tr>
            <tr>
                <td>Hidden Dimension</td>
                <td>64</td>
            </tr>
            <tr>
                <td>Number of Layers</td>
                <td>2</td>
            </tr>
            <tr>
                <td>Dropout Rate</td>
                <td>0.2</td>
            </tr>
            <tr>
                <td>Learning Rate</td>
                <td>0.001</td>
            </tr>
            <tr>
                <td>Sample Size</td>
                <td>5,000</td>
            </tr>
        </table>
        
        <div class="footer">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, 'comparison_report.html'), 'w') as f:
        f.write(html_content)
    
    print(f"Comparison results saved to {output_dir}")

def main():
    # Create output directory
    output_dir = "output/model_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metrics from both models
    gnn_metrics = load_metrics(os.path.join('output/direct_prediction_gnn_fast', 'metrics.txt'))
    ffn_metrics = load_metrics(os.path.join('output/direct_prediction_ffn_fast', 'metrics.txt'))
    
    # Plot comparison
    plot_comparison(gnn_metrics, ffn_metrics, output_dir)

if __name__ == "__main__":
    main() 