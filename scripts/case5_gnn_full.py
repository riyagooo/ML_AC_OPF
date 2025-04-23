#!/usr/bin/env python
"""
Script to train a GNN model for case5 using the provided data.
This version directly loads the MATPOWER file to use topology information
without relying on pytest fixtures.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
from torch_geometric.data import Data

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gnn import TopologyAwareGNN
from utils.data_utils import OPFDataset
from utils.training import Trainer

def create_case5_data():
    """
    Create case5 data structure directly without relying on pytest fixtures.
    
    Returns:
        Dictionary with case5 power system data
    """
    return {
        'baseMVA': 100.0,
        'bus': [
            [1, 2, 0.0, 0.0, 0.0, 0.0, 1, 1.0, 0.0, 230.0, 1, 1.1, 0.9],  # Bus 1
            [2, 1, 300.0, 98.61, 0.0, 0.0, 1, 1.0, 0.0, 230.0, 1, 1.1, 0.9],  # Bus 2
            [3, 2, 300.0, 98.61, 0.0, 0.0, 1, 1.0, 0.0, 230.0, 1, 1.1, 0.9],  # Bus 3
            [4, 3, 400.0, 131.47, 0.0, 0.0, 1, 1.0, 0.0, 230.0, 1, 1.1, 0.9],  # Bus 4
            [5, 2, 0.0, 0.0, 0.0, 0.0, 1, 1.0, 0.0, 230.0, 1, 1.1, 0.9]   # Bus 5
        ],
        'gen': [
            [1, 20.0, 0.0, 30.0, -30.0, 1.0, 100.0, 1, 40.0, 0.0],  # Gen 1
            [1, 85.0, 0.0, 127.5, -127.5, 1.0, 100.0, 1, 170.0, 0.0],  # Gen 2
            [3, 260.0, 0.0, 390.0, -390.0, 1.0, 100.0, 1, 520.0, 0.0],  # Gen 3
            [4, 100.0, 0.0, 150.0, -150.0, 1.0, 100.0, 1, 200.0, 0.0],  # Gen 4
            [5, 300.0, 0.0, 450.0, -450.0, 1.0, 100.0, 1, 600.0, 0.0]   # Gen 5
        ],
        'branch': [
            [1, 2, 0.00281, 0.0281, 0.00712, 400.0, 400.0, 400.0, 0.0, 0.0, 1, -30.0, 30.0],
            [1, 4, 0.00304, 0.0304, 0.00658, 426.0, 426.0, 426.0, 0.0, 0.0, 1, -30.0, 30.0],
            [1, 5, 0.00064, 0.0064, 0.03126, 426.0, 426.0, 426.0, 0.0, 0.0, 1, -30.0, 30.0],
            [2, 3, 0.00108, 0.0108, 0.01852, 426.0, 426.0, 426.0, 0.0, 0.0, 1, -30.0, 30.0],
            [3, 4, 0.00297, 0.0297, 0.00674, 426.0, 426.0, 426.0, 0.0, 0.0, 1, -30.0, 30.0],
            [4, 5, 0.00297, 0.0297, 0.00674, 240.0, 240.0, 240.0, 0.0, 0.0, 1, -30.0, 30.0]
        ],
        'gencost': [
            [2, 0.0, 0.0, 3, 0.0, 14.0, 0.0],
            [2, 0.0, 0.0, 3, 0.0, 15.0, 0.0],
            [2, 0.0, 0.0, 3, 0.0, 30.0, 0.0],
            [2, 0.0, 0.0, 3, 0.0, 40.0, 0.0],
            [2, 0.0, 0.0, 3, 0.0, 10.0, 0.0]
        ]
    }

def create_power_network_graph(case_data):
    """
    Create NetworkX graph from power system case data.
    
    Args:
        case_data: Power system case data structure
        
    Returns:
        NetworkX graph
    """
    # Create an empty graph
    G = nx.DiGraph()
    
    # Add buses as nodes
    for i, bus in enumerate(case_data['bus']):
        bus_id = int(bus[0])
        bus_type = int(bus[1])
        pd = float(bus[2])  # Active load
        qd = float(bus[3])  # Reactive load
        G.add_node(bus_id, 
                   type=bus_type, 
                   Pd=pd, 
                   Qd=qd, 
                   Vm=float(bus[7]),
                   Va=float(bus[8]),
                   baseKV=float(bus[9]), 
                   Vmax=float(bus[11]), 
                   Vmin=float(bus[12]),
                   index=i,
                   is_gen=0)  # Default to not a generator
    
    # Add branches as edges
    for branch in case_data['branch']:
        from_bus = int(branch[0])
        to_bus = int(branch[1])
        r = float(branch[2])  # resistance
        x = float(branch[3])  # reactance
        b = float(branch[4])  # susceptance
        rate_a = float(branch[5])  # MVA rating
        G.add_edge(from_bus, to_bus, r=r, x=x, b=b, rateA=rate_a)
        G.add_edge(to_bus, from_bus, r=r, x=x, b=b, rateA=rate_a)  # Add reverse direction too
    
    # Add generator information to nodes
    for gen in case_data['gen']:
        bus_id = int(gen[0])
        pg = float(gen[1])  # Active power generation
        qg = float(gen[2])  # Reactive power generation
        qmax = float(gen[3])  # Max reactive power
        qmin = float(gen[4])  # Min reactive power
        pmax = float(gen[8])  # Max active power
        pmin = float(gen[9])  # Min active power
        
        # Add generator info to the corresponding bus
        if G.has_node(bus_id):
            G.nodes[bus_id]['pg'] = pg
            G.nodes[bus_id]['qg'] = qg
            G.nodes[bus_id]['qmax'] = qmax
            G.nodes[bus_id]['qmin'] = qmin
            G.nodes[bus_id]['pmax'] = pmax
            G.nodes[bus_id]['pmin'] = pmin
            G.nodes[bus_id]['is_gen'] = 1
    
    return G

def visualize_network(G, save_path=None):
    """
    Visualize power network graph.
    
    Args:
        G: NetworkX graph
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 8))
    
    # Set node positions using spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes with different colors based on type
    bus_types = {1: 'blue', 2: 'green', 3: 'red'}
    node_colors = [bus_types.get(G.nodes[n]['type'], 'gray') for n in G.nodes]
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.8, node_size=500)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.title("Power Network Graph")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Network visualization saved to {save_path}")
    else:
        plt.show()

def prepare_pyg_data(G, node_features, edge_features=None):
    """
    Convert NetworkX graph to PyTorch Geometric data.
    
    Args:
        G: NetworkX graph
        node_features: List of node features to extract
        edge_features: List of edge features to extract
        
    Returns:
        PyTorch Geometric Data object
    """
    # Get node indices
    node_indices = {node: i for i, node in enumerate(G.nodes())}
    
    # Prepare edge index
    edge_index = []
    for u, v in G.edges():
        edge_index.append([node_indices[u], node_indices[v]])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Prepare node features
    x = []
    for node in G.nodes():
        # Get feature values, default to 0 if not present
        node_feat = []
        for feat in node_features:
            if feat in G.nodes[node]:
                node_feat.append(float(G.nodes[node][feat]))
            else:
                node_feat.append(0.0)
        x.append(node_feat)
    
    x = torch.tensor(x, dtype=torch.float)
    
    # Prepare edge features if provided
    edge_attr = None
    if edge_features is not None:
        edge_attr = []
        for u, v in G.edges():
            # Get feature values, default to 0 if not present
            edge_feat = []
            for feat in edge_features:
                if feat in G[u][v]:
                    edge_feat.append(float(G[u][v][feat]))
                else:
                    edge_feat.append(0.0)
            edge_attr.append(edge_feat)
        
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Create PyTorch Geometric data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data

def update_node_features_with_load(graph_data, load_data):
    """
    Update node features with load data for a specific sample.
    
    Args:
        graph_data: PyTorch Geometric Data object
        load_data: Tensor with load values
        
    Returns:
        Updated PyTorch Geometric Data object
    """
    # Create a new copy of the graph data
    new_data = Data(
        x=graph_data.x.clone(),
        edge_index=graph_data.edge_index.clone(),
        edge_attr=graph_data.edge_attr.clone() if graph_data.edge_attr is not None else None
    )
    
    # Assuming the first features in load_data correspond to active loads (Pd)
    # and the next set corresponds to reactive loads (Qd)
    n_buses = new_data.x.shape[0]
    n_load_features = load_data.shape[0]
    
    # Update Pd and Qd values in the node features
    # This assumes that Pd and Qd are the 2nd and 3rd features in the node feature vector
    for i in range(min(n_buses, n_load_features // 2)):
        # Update active load (Pd)
        new_data.x[i, 1] = load_data[i].item()
        
        # Update reactive load (Qd)
        if i + n_buses < n_load_features:
            new_data.x[i, 2] = load_data[i + n_buses].item()
    
    return new_data

def main():
    # Configuration
    case_name = "case5"
    data_dir = "data"
    epochs = 20  # Full training
    batch_size = 32  
    learning_rate = 0.001
    hidden_channels = 64  # Larger network
    num_layers = 3  # Deeper network
    dropout_rate = 0.1
    
    # Create log directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join("logs", f"case5_gnn_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get the case data directly from our function (not using pytest fixtures)
    print(f"Getting case5 network data...")
    case_data = create_case5_data()
    print(f"Case data loaded: {len(case_data['bus'])} buses, {len(case_data['branch'])} branches, {len(case_data['gen'])} generators")
    
    # Load OPF solutions from CSV
    csv_file = os.path.join(data_dir, f"pglib_opf_{case_name}.csv")
    print(f"Loading OPF solutions from: {csv_file}")
    solutions_data = pd.read_csv(csv_file)
    # Use all available data
    print(f"OPF solutions loaded: {len(solutions_data)} samples")
    
    # Create and visualize network graph
    print("Creating network graph...")
    G = create_power_network_graph(case_data)
    print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Visualize the network
    network_viz_path = os.path.join(log_dir, "network_graph.png")
    visualize_network(G, save_path=network_viz_path)
    
    # Prepare PyTorch Geometric data
    node_features = ['type', 'Pd', 'Qd', 'Vm', 'Va', 'baseKV', 'Vmax', 'Vmin', 'is_gen']
    edge_features = ['r', 'x', 'b', 'rateA']
    
    print("Converting graph to PyTorch Geometric format...")
    graph_data = prepare_pyg_data(G, node_features, edge_features)
    print(f"Graph data: {graph_data}")
    
    # Determine the input and output columns from the CSV file
    input_cols = [col for col in solutions_data.columns if col.startswith('load')]
    output_cols = [col for col in solutions_data.columns if col.startswith('gen') and ':pg' in col]
    
    # If no pg columns are found, try another pattern
    if not output_cols:
        output_cols = [col for col in solutions_data.columns if ':pg' in col]
    
    # If still no columns found, just use any gen columns
    if not output_cols:
        output_cols = [col for col in solutions_data.columns if 'gen' in col][:5]  # Limit to first 5
    
    print(f"Input features: {len(input_cols)}")
    print(f"Output features: {len(output_cols)}")
    print(f"First few input columns: {input_cols[:5]}")
    print(f"First few output columns: {output_cols[:5]}")
    
    # Split the data into train, validation, and test sets
    n_samples = len(solutions_data)
    train_size = int(0.8 * n_samples)
    val_size = int(0.1 * n_samples)
    
    train_data = solutions_data.iloc[:train_size]
    val_data = solutions_data.iloc[train_size:train_size+val_size]
    test_data = solutions_data.iloc[train_size+val_size:]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Create datasets
    train_dataset = OPFDataset(train_data, input_cols, output_cols)
    val_dataset = OPFDataset(val_data, input_cols, output_cols)
    test_dataset = OPFDataset(test_data, input_cols, output_cols)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size
    )
    
    # Initialize GNN model
    model = TopologyAwareGNN(
        node_features=len(node_features),
        edge_features=len(edge_features) if edge_features else 0,
        hidden_channels=hidden_channels,
        output_dim=len(output_cols),
        num_layers=num_layers,
        dropout_rate=dropout_rate
    ).to(device)
    
    print(f"Model initialized: {model}")
    
    # Define a custom forward function to handle graph data with load inputs
    def custom_forward(model, inputs, graph_data):
        """Custom forward function that combines graph data with load inputs."""
        # Move inputs to device
        inputs = inputs.to(device)
        
        # Create a batch of graph data with updated node features
        batch_size = inputs.size(0)
        batch_outputs = []
        
        for i in range(batch_size):
            # Update graph with current load data
            sample_graph = update_node_features_with_load(graph_data, inputs[i])
            
            # Move to device
            sample_graph = sample_graph.to(device)
            
            # Get outputs for this sample (don't use torch.no_grad during training)
            sample_output = model(sample_graph)
            batch_outputs.append(sample_output.unsqueeze(0))
        
        # Combine all outputs into a batch
        return torch.cat(batch_outputs, dim=0)
    
    # Create trainer
    class GNNTrainer(Trainer):
        def __init__(self, model, graph_data, *args, **kwargs):
            super().__init__(model, *args, **kwargs)
            self.graph_data = graph_data
        
        def train_epoch(self, train_loader, metrics=None):
            """Train for one epoch with custom forward pass."""
            self.model.train()
            total_loss = 0
            total_metrics = {name: 0 for name in metrics.keys()} if metrics else {}
            
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass with custom function
                outputs = custom_forward(self.model, inputs, self.graph_data)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Track loss
                total_loss += loss.item()
                
                # Compute metrics if provided
                if metrics:
                    with torch.no_grad():
                        for name, metric_fn in metrics.items():
                            metric_value = metric_fn(outputs, targets).item()
                            total_metrics[name] += metric_value
            
            # Calculate averages
            avg_loss = total_loss / len(train_loader)
            avg_metrics = {name: value / len(train_loader) 
                         for name, value in total_metrics.items()}
            
            return avg_loss, avg_metrics
        
        def validate(self, val_loader, metrics=None):
            """Validate model with custom forward pass."""
            self.model.eval()
            total_loss = 0
            total_metrics = {name: 0 for name in metrics.keys()} if metrics else {}
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Forward pass with custom function
                    outputs = custom_forward(self.model, inputs, self.graph_data)
                    
                    # Compute loss
                    loss = self.criterion(outputs, targets)
                    
                    # Track loss
                    total_loss += loss.item()
                    
                    # Compute metrics if provided
                    if metrics:
                        for name, metric_fn in metrics.items():
                            metric_value = metric_fn(outputs, targets).item()
                            total_metrics[name] += metric_value
            
            # Calculate averages
            avg_loss = total_loss / len(val_loader)
            avg_metrics = {name: value / len(val_loader) 
                         for name, value in total_metrics.items()}
            
            return avg_loss, avg_metrics
    
    # Initialize trainer with graph data
    trainer = GNNTrainer(
        model=model,
        graph_data=graph_data,
        optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate),
        criterion=torch.nn.MSELoss(),
        device=device
    )
    
    # Train the model
    print(f"Training model for {epochs} epochs...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        save_best=True,
        verbose=True
    )
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    history_plot_path = os.path.join(log_dir, "training_history.png")
    plt.savefig(history_plot_path)
    print(f"Training history plot saved to {history_plot_path}")
    
    # Test the model
    test_loss, _ = trainer.validate(test_loader)
    print(f"Test loss: {test_loss:.6f}")
    
    # Save test results and metrics
    with open(os.path.join(log_dir, "results.txt"), 'w') as f:
        f.write(f"Test Loss: {test_loss:.6f}\n\n")
        f.write("Network Structure:\n")
        f.write(f"Nodes: {G.number_of_nodes()}\n")
        f.write(f"Edges: {G.number_of_edges()}\n\n")
        f.write("Model Configuration:\n")
        f.write(f"Hidden Channels: {hidden_channels}\n")
        f.write(f"Num Layers: {num_layers}\n")
        f.write(f"Dropout Rate: {dropout_rate}\n")
    
    # Visualize some predictions vs targets
    model.eval()
    with torch.no_grad():
        # Get first batch from test loader
        inputs, targets = next(iter(test_loader))
        
        # Make predictions with grad disabled (for evaluation)
        with torch.no_grad():
            outputs = custom_forward(model, inputs, graph_data)
        
        # Convert to numpy for plotting
        outputs = outputs.cpu().numpy()
        targets = targets.cpu().numpy()
        
        # Plot first 5 samples for the first output
        plt.figure(figsize=(12, 6))
        
        # Plot for first output dimension
        plt.subplot(1, 2, 1)
        x = np.arange(min(5, len(outputs)))
        width = 0.35
        
        plt.bar(x - width/2, targets[:5, 0], width, label='Target')
        plt.bar(x + width/2, outputs[:5, 0], width, label='Prediction')
        
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.title(f'Predictions vs Targets for {output_cols[0]}')
        plt.xticks(x)
        plt.legend()
        plt.grid(True)
        
        # Plot for second output dimension if available
        if outputs.shape[1] > 1:
            plt.subplot(1, 2, 2)
            plt.bar(x - width/2, targets[:5, 1], width, label='Target')
            plt.bar(x + width/2, outputs[:5, 1], width, label='Prediction')
            
            plt.xlabel('Sample')
            plt.ylabel('Value')
            plt.title(f'Predictions vs Targets for {output_cols[1]}')
            plt.xticks(x)
            plt.legend()
            plt.grid(True)
        
        predictions_plot_path = os.path.join(log_dir, "predictions_vs_targets.png")
        plt.tight_layout()
        plt.savefig(predictions_plot_path)
        print(f"Predictions plot saved to {predictions_plot_path}")
    
    print(f"All results saved to {log_dir}")

if __name__ == "__main__":
    main()