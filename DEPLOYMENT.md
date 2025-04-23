# Deployment Guide: Local and Google Colab

This guide provides instructions for deploying the ML-OPF project in both local and Google Colab environments.

## Local Deployment

### Prerequisites

- Python 3.8 or higher
- Git
- Gurobi license (academic licenses are available for free)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ML-OPF-Project.git
   cd ML-OPF-Project
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset:
   ```bash
   mkdir -p data
   wget https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/pglib_opf_case118_ieee.m -O data/pglib_opf_case118.m
   wget https://data.nrel.gov/system/files/177/pglib_opf_case118_ieee.csv -O data/pglib_opf_case118.csv
   ```

### Running Tests

1. Run a simple local test with the feedforward neural network:
   ```bash
   python scripts/local_test.py --case case118 --model-type feedforward --epochs 10
   ```

2. Try the constraint screening approach:
   ```bash
   python scripts/constraint_screening.py --case case118 --epochs 10
   ```

3. Try the warm-starting approach:
   ```bash
   python scripts/warm_starting.py --case case118 --epochs 10 --num-samples 5
   ```

### Local Development

For local development, you can use any IDE or editor that supports Python. The project is structured as follows:

- `models/`: Neural network models (feedforward, GNN)
- `utils/`: Utility functions for data loading, optimization, etc.
- `scripts/`: Scripts for running experiments
- `notebooks/`: Jupyter notebooks for exploration and visualization

## Google Colab Deployment

### Setup

1. Upload the notebook files from `notebooks/` to Google Colab.

2. Run the setup cell in the notebook to install dependencies and clone the repository:
   ```python
   !pip install torch-geometric torch-sparse torch-scatter
   !pip install pypower networkx gurobipy
   !pip install pandas numpy matplotlib wandb tqdm
   
   # Clone the repository
   !git clone https://github.com/your-username/ML-OPF-Project.git
   %cd ML-OPF-Project
   
   # Download data
   !mkdir -p data
   !wget https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master/pglib_opf_case118_ieee.m -O data/pglib_opf_case118.m
   !wget https://data.nrel.gov/system/files/177/pglib_opf_case118_ieee.csv -O data/pglib_opf_case118.csv
   ```

3. Connect to the GPU runtime by selecting Runtime > Change runtime type > GPU.

### Running Experiments

1. In the notebook, you can run experiments with the GNN model:
   ```python
   # Set configuration
   config = {
       'case': 'case118',
       'data_dir': 'data',
       'log_dir': 'logs',
       'model_type': 'gnn',  # 'gnn' or 'hybrid_gnn'
       'epochs': 50,
       'batch_size': 32,
       'learning_rate': 0.001,
       'hidden_channels': 64,
       'num_layers': 3,
       'dropout_rate': 0.1,
       'use_wandb': False
   }
   
   # Run training
   # ... (Follow the notebook instructions)
   ```

2. You can also run other experiments by uploading and executing the scripts:
   ```python
   !python scripts/constraint_screening.py --case case118 --epochs 10 --gpu
   ```

### Saving Results

To save results from Google Colab to your local machine:

1. Use the Files panel on the left to download logs and model files.

2. Alternatively, save them to Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Copy results to Google Drive
   !cp -r logs /content/drive/MyDrive/ML-OPF-Results/
   ```

## Hybrid Deployment Strategy

For an effective hybrid deployment strategy:

1. **Development and Testing**: Use the local environment for initial development, debugging, and quick testing.

2. **Data Preparation**: Use either environment, but ensure the data processing steps are consistent.

3. **Training**:
   - For small models and datasets: Use the local environment
   - For larger models and datasets: Use Google Colab with GPU acceleration

4. **Model Evaluation**: Use both environments to ensure consistency.

5. **Sharing and Collaboration**: Use Google Colab for collaboration and sharing results.

## File Synchronization

To maintain code consistency between local and Colab environments:

1. Use a GitHub repository to sync code.

2. Before running on Colab, commit and push changes from local environment.

3. In Colab, pull the latest changes:
   ```python
   %cd ML-OPF-Project
   !git pull
   ```

4. After making changes in Colab, you can push them back (if you have git credentials set up):
   ```python
   !git add .
   !git commit -m "Updates from Colab"
   !git push
   ```

## Model Serving

For deploying trained models in production:

1. Save trained models in a standard format (PyTorch's `.pt`):
   ```python
   torch.save({
       'model_type': config['model_type'],
       'state_dict': model.state_dict(),
       'config': config
   }, os.path.join('deployment', f"{config['model_type']}_model.pt"))
   ```

2. Create a deployment script that loads the model and provides inference:
   ```python
   def load_model(model_path, device='cpu'):
       checkpoint = torch.load(model_path, map_location=device)
       # Create model instance based on checkpoint info
       # ...
       return model
   
   def predict(model, inputs):
       model.eval()
       with torch.no_grad():
           outputs = model(inputs)
       return outputs
   ```

3. Use the model for warm-starting or constraint screening in your OPF solver. 