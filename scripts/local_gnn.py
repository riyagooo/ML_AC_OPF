#!/usr/bin/env python
"""
Local GNN training script for ML-OPF project.
This script trains Graph Neural Networks on small power system cases locally.
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import networkx as nx

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import load_pglib_data, load_case_network, prepare_data_loaders, create_power_network_graph
from utils.optimization import OPFOptimizer
from utils.metrics import optimality_gap_metric
from models.gnn import TopologyAwareGNN, HybridGNN, prepare_pyg_data, GCNLayer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Local GNN Training for ML-OPF')
    parser.add_argument('--case', type=str, default='case5', 
                       choices=['case5', 'case14', 'case30'],
                       help='Case name (only small cases supported: case5, case14, case30)')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--model-type', type=str, default='gnn', 
                       choices=['gnn', 'hybrid_gnn'],
                       help='GNN model type (gnn or hybrid_gnn)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-channels', type=int, default=64, help='Hidden channels in GNN')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--dropout-rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--print-columns', action='store_true', help='Print column names for debugging')
    parser.add_argument('--approach', type=str, default='warm_starting',
                       choices=['warm_starting', 'constraint_screening'],
                       help='ML approach to use with GNN')
    
    return parser.parse_args()

def visualize_graph(G, log_dir):
    """Visualize power network graph and save it."""
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    
    # Color nodes by type (PQ, PV, etc.)
    node_colors = []
    for node in G.nodes():
        node_type = G.nodes[node]['type']
        if node_type == 1:  # PQ bus
            node_colors.append('blue')
        elif node_type == 2:  # PV bus 
            node_colors.append('green')
        elif node_type == 3:  # Slack bus
            node_colors.append('red')
        else:
            node_colors.append('gray')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                         node_size=500, 
                         node_color=node_colors,
                         alpha=0.8)
    
    # Draw edges with width proportional to capacity
    edge_width = [G[u][v].get('rateA', 1.0)/100 for u, v in G.edges()]
    edge_width = [max(w, 0.5) for w in edge_width]  # Ensure minimum visibility
    
    nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.5)
    
    # Add labels
    labels = {node: str(node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_color='white')
    
    plt.title(f'Power Network Graph')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'network_graph.png'))
    plt.close()

def train_epoch(model, data_loader, graph_data, optimizer, criterion, device, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader, desc='Training')):
        # Move inputs and targets to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass through the model
        if config['model_type'] == 'gnn':
            # For TopologyAwareGNN, we use the same graph for all samples
            outputs = model(graph_data.to(device))
        else:
            # For HybridGNN, we use both graph and input features
            outputs = model(graph_data.to(device), inputs)
        
        # Compute loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track loss
        total_loss += loss.item()
    
    # Return average loss
    return total_loss / len(data_loader)

def validate(model, data_loader, graph_data, criterion, device, config):
    """Validate model on validation data."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc='Validation'):
            # Move inputs and targets to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass through the model
            if config['model_type'] == 'gnn':
                outputs = model(graph_data.to(device))
            else:
                outputs = model(graph_data.to(device), inputs)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Track loss
            total_loss += loss.item()
    
    # Return average loss
    return total_loss / len(data_loader)

def test_warm_start(model, test_loader, graph_data, device, config, optimizer, log_dir):
    """Test the warm-start performance of the GNN model."""
    model.eval()
    n_samples = min(3, len(test_loader.dataset))
    
    # List to store results
    no_warm_times = []
    warm_times = []
    speedups = []
    obj_diffs = []
    
    # Generate sample indices
    sample_indices = np.random.choice(len(test_loader.dataset), n_samples, replace=False)
    
    # Extract dimensions from the optimizer
    n_gen = optimizer.n_gen
    n_bus = optimizer.n_bus
    
    # Create results directory
    results_dir = os.path.join(log_dir, 'warm_start_results')
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\nTesting warm-start performance with {n_samples} samples...")
    
    # Loop through samples
    for i, idx in enumerate(sample_indices):
        # Get sample
        sample_input, sample_target = test_loader.dataset[idx]
        sample_input = sample_input.to(device).unsqueeze(0)  # Add batch dimension
        
        # Generate prediction
        with torch.no_grad():
            if config['model_type'] == 'gnn':
                prediction = model(graph_data.to(device))
            else:
                prediction = model(graph_data.to(device), sample_input)
        
        # Convert prediction to numpy
        prediction_np = prediction.cpu().numpy()
        
        # Prepare warm-start solution
        warm_start = {
            'pg': prediction_np[:n_gen],
            'qg': prediction_np[n_gen:2*n_gen],
            'vm': prediction_np[2*n_gen:2*n_gen+n_bus],
            'va': prediction_np[2*n_gen+n_bus:2*n_gen+2*n_bus]
        }
        
        # Prepare load data
        load_data = sample_input[0].cpu().numpy().reshape(-1, 2)
        
        # Solve OPF without warm-start
        print(f"\nSample {i+1}/{n_samples}:")
        print("Solving OPF without warm-start...")
        solution_no_warm = optimizer.solve_opf_gurobi(load_data, verbose=False)
        
        if solution_no_warm['success']:
            print(f"Solution found in {solution_no_warm['runtime']:.6f} seconds.")
            no_warm_times.append(solution_no_warm['runtime'])
            
            # Solve OPF with warm-start
            print("Solving OPF with warm-start...")
            solution_warm = optimizer.solve_opf_gurobi(load_data, warm_start=warm_start, verbose=False)
            
            if solution_warm['success']:
                print(f"Solution found in {solution_warm['runtime']:.6f} seconds.")
                warm_times.append(solution_warm['runtime'])
                
                # Compute speedup
                speedup = solution_no_warm['runtime'] / solution_warm['runtime']
                speedups.append(speedup)
                print(f"Speedup: {speedup:.2f}x")
                
                # Compare objective values
                obj_diff = abs(solution_no_warm['f'] - solution_warm['f']) / solution_no_warm['f'] * 100
                obj_diffs.append(obj_diff)
                print(f"Objective value difference: {obj_diff:.6f}%")
            else:
                print("Warm-start solution failed.")
        else:
            print("No-warm-start solution failed.")
    
    # Save results
    if speedups:
        # Create summary plot
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(speedups)), speedups)
        plt.axhline(y=1.0, color='r', linestyle='--')
        plt.xlabel('Sample')
        plt.ylabel('Speedup Factor')
        plt.title('Warm-Starting Speedup')
        plt.savefig(os.path.join(results_dir, 'speedup_plot.png'))
        plt.close()
        
        # Save summary statistics
        avg_speedup = np.mean(speedups)
        avg_obj_diff = np.mean(obj_diffs)
        
        with open(os.path.join(results_dir, 'summary.txt'), 'w') as f:
            f.write(f"Number of samples: {len(speedups)}\n")
            f.write(f"Average no-warm-start time: {np.mean(no_warm_times):.6f} seconds\n")
            f.write(f"Average warm-start time: {np.mean(warm_times):.6f} seconds\n")
            f.write(f"Average speedup: {avg_speedup:.2f}x\n")
            f.write(f"Speedup range: {np.min(speedups):.2f}x - {np.max(speedups):.2f}x\n")
            f.write(f"Average objective difference: {avg_obj_diff:.6f}%\n")
        
        print(f"\nWarm-starting results:")
        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Average objective difference: {avg_obj_diff:.6f}%")
    
    return speedups, obj_diffs

def plot_training_history(history, log_dir):
    """Plot training history."""
    plt.figure(figsize=(12, 5))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    plt.grid(True)
    
    # Plot optimization gap if available
    if 'opt_gap' in history:
        plt.subplot(1, 2, 2)
        plt.plot(history['opt_gap'], label='Optimization Gap')
        plt.xlabel('Epoch')
        plt.ylabel('Gap (%)')
        plt.title('Optimization Gap vs. Epoch')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'training_history.png'))
    plt.close()

def main(args):
    """Main function."""
    # Create config dictionary
    config = {
        'case': args.case,
        'data_dir': args.data_dir,
        'model_type': args.model_type,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'hidden_channels': args.hidden_channels,
        'num_layers': args.num_layers,
        'dropout_rate': args.dropout_rate,
        'approach': args.approach
    }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create log directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f"local_gnn_{args.model_type}_{args.approach}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(log_dir, 'config.txt'), 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    try:
        # Load data
        print(f"Loading data for {args.case}...")
        data = load_pglib_data(args.case, args.data_dir)
        case_data = load_case_network(args.case, args.data_dir)
        print(f"Data loaded: {len(data)} samples")
        
        # Print all columns for debugging
        if args.print_columns:
            print("\nAll data columns:")
            for col in data.columns:
                print(f"  {col}")
            print("")
        
        # Create network graph
        G = create_power_network_graph(case_data)
        print(f"Created power network graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # If the graph is empty, create a minimal synthetic graph
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            print("Warning: Empty graph detected. Creating synthetic graph for demonstration.")
            n_bus = len(case_data['bus'])
            
            # Create a simple ring topology
            for i in range(n_bus):
                G.add_node(i, type=2, Pd=0.0, Qd=0.0, Vm=1.0, Va=0.0, baseKV=1.0, Vmax=1.05, Vmin=0.95)
                
            for i in range(n_bus):
                G.add_edge(i, (i+1) % n_bus, r=0.01, x=0.1, b=0.001, rateA=1.0)
                
            print(f"Created synthetic graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Visualize graph
        visualize_graph(G, log_dir)
        
        # Extract input and output columns
        input_cols = [col for col in data.columns if col.startswith('load_p') or col.startswith('load_q')]
        
        # If no columns found, try alternative naming pattern
        if not input_cols:
            input_cols = [col for col in data.columns if col.startswith('load') and (':pl' in col or ':ql' in col)]
        
        # If still no input columns, create synthetic ones
        if not input_cols:
            print("Warning: No load columns found. Creating synthetic load features.")
            n_buses = len(case_data['bus'])
            # Create synthetic load data
            for i in range(n_buses):
                col_p = f"load:pl_{i+1}"
                col_q = f"load:ql_{i+1}"
                data[col_p] = np.random.uniform(0.1, 1.0, size=len(data))
                data[col_q] = np.random.uniform(0.05, 0.5, size=len(data))
                input_cols.extend([col_p, col_q])
            
        # Try different output column patterns
        output_cols_try1 = [col for col in data.columns if col.startswith('gen_p') or 
                           col.startswith('gen_q') or col.startswith('bus_v')]
        
        # If no columns found, try alternative naming pattern
        if not output_cols_try1:
            output_cols_try2 = [col for col in data.columns if col.startswith('gen') and (':pg' in col or ':qg' in col)]
            bus_v_cols = [col for col in data.columns if col.startswith('bus') and ':v_' in col]
            output_cols = output_cols_try2 + bus_v_cols
        else:
            output_cols = output_cols_try1
            
        # If still no output columns, create synthetic ones
        if not output_cols:
            print("Warning: No generator or bus columns found. Creating synthetic output features.")
            n_gen = len(case_data['gen'])
            n_bus = len(case_data['bus'])
            
            # Create synthetic generator data
            for i in range(n_gen):
                col_pg = f"gen:{i+1}:pg"
                col_qg = f"gen:{i+1}:qg"
                data[col_pg] = np.random.uniform(0.1, 1.0, size=len(data))
                data[col_qg] = np.random.uniform(-0.5, 0.5, size=len(data))
                output_cols.extend([col_pg, col_qg])
            
            # Create synthetic bus voltage data
            for i in range(n_bus):
                col_vm = f"bus:{i+1}:v_m"
                data[col_vm] = np.random.uniform(0.95, 1.05, size=len(data))
                output_cols.append(col_vm)
        
        print(f"Input features: {len(input_cols)}")
        if args.print_columns and input_cols:
            print("Input columns:")
            for col in input_cols:
                print(f"  {col}")
        
        print(f"Output features: {len(output_cols)}")
        if args.print_columns and output_cols:
            print("Output columns:")
            for col in output_cols:
                print(f"  {col}")
        
        # Prepare data loaders
        train_loader, val_loader, test_loader = prepare_data_loaders(
            data, input_cols, output_cols, batch_size=args.batch_size)
        
        # Define node and edge features
        node_features = ['type', 'Pd', 'Qd', 'Vm', 'Va', 'baseKV', 'Vmax', 'Vmin']
        edge_features = ['r', 'x', 'b', 'rateA']
        
        # Create PyG data
        graph_data = prepare_pyg_data(G, node_features, edge_features)
        
        # Create bounds for outputs based on problem constraints
        pg_min = torch.tensor(case_data['gen'][:, 9] / case_data['baseMVA'], dtype=torch.float32, device=device)
        pg_max = torch.tensor(case_data['gen'][:, 8] / case_data['baseMVA'], dtype=torch.float32, device=device)
        qg_min = torch.tensor(case_data['gen'][:, 4] / case_data['baseMVA'], dtype=torch.float32, device=device)
        qg_max = torch.tensor(case_data['gen'][:, 3] / case_data['baseMVA'], dtype=torch.float32, device=device)
        vm_min = torch.tensor(case_data['bus'][:, 12], dtype=torch.float32, device=device)
        vm_max = torch.tensor(case_data['bus'][:, 11], dtype=torch.float32, device=device)
        
        # Assemble bounds
        n_gen = len(case_data['gen'])
        n_bus = len(case_data['bus'])
        
        min_vals = []
        max_vals = []
        
        # Add generator active power bounds
        min_vals.append(pg_min)
        max_vals.append(pg_max)
        
        # Add generator reactive power bounds
        min_vals.append(qg_min)
        max_vals.append(qg_max)
        
        # Add voltage magnitude bounds
        min_vals.append(vm_min)
        max_vals.append(vm_max)
        
        # Add voltage angle bounds
        min_vals.append(torch.full((n_bus,), -np.pi, device=device))
        max_vals.append(torch.full((n_bus,), np.pi, device=device))
        
        # Concatenate bounds
        try:
            min_bounds = torch.cat(min_vals)
            max_bounds = torch.cat(max_vals)
            
            # Ensure bounds match output dimension
            min_bounds = min_bounds[:len(output_cols)]
            max_bounds = max_bounds[:len(output_cols)]
            output_bounds = (min_bounds, max_bounds)
        except:
            print("Warning: Could not create output bounds. Using None instead.")
            output_bounds = None
        
        # Initialize model
        if args.model_type == 'gnn':
            model = TopologyAwareGNN(
                node_features=len(node_features),
                edge_features=len(edge_features),
                hidden_channels=args.hidden_channels,
                output_dim=len(output_cols),
                num_layers=args.num_layers,
                dropout_rate=args.dropout_rate,
                output_bounds=output_bounds
            ).to(device)
            print("Initialized TopologyAwareGNN model")
        else:
            model = HybridGNN(
                node_features=len(node_features),
                edge_features=len(edge_features),
                global_features=len(input_cols),
                hidden_channels=args.hidden_channels,
                output_dim=len(output_cols),
                num_layers=args.num_layers,
                dropout_rate=args.dropout_rate,
                output_bounds=output_bounds
            ).to(device)
            print("Initialized HybridGNN model")
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # Initialize criterion
        criterion = torch.nn.MSELoss()
        
        # Initialize history
        history = {
            'train_loss': [],
            'val_loss': [],
            'opt_gap': []
        }
        
        # Train model
        print(f"Training {args.model_type} model for {args.epochs} epochs...")
        best_val_loss = float('inf')
        
        for epoch in range(args.epochs):
            # Train for one epoch
            train_loss = train_epoch(model, train_loader, graph_data, optimizer, criterion, device, config)
            
            # Validate
            val_loss = validate(model, val_loader, graph_data, criterion, device, config)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{args.epochs}:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pt'))
                print("  Saved best model")
        
        # Plot training history
        plot_training_history(history, log_dir)
        
        # Initialize OPF optimizer for warm-start testing
        opf_optimizer = OPFOptimizer(case_data, device=device)
        
        # Load best model
        model.load_state_dict(torch.load(os.path.join(log_dir, 'best_model.pt')))
        
        # Test warm-start performance
        speedups, obj_diffs = test_warm_start(model, test_loader, graph_data, device, config, opf_optimizer, log_dir)
        
        print(f"Training and evaluation completed. Results saved to {log_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    args = parse_args()
    main(args) 