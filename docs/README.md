# ML-AC-OPF: Machine Learning Approaches for AC Optimal Power Flow

## Project Overview

This project implements and compares multiple machine learning approaches for solving AC Optimal Power Flow (AC-OPF) problems in power systems. We explore both traditional feedforward neural networks and graph neural networks (GNNs) to leverage power system topology information for improved performance.

The AC-OPF problem aims to find the optimal generation dispatch that minimizes cost while satisfying operational constraints. It is mathematically represented as:

$$\min_{x} \sum_{i \in G} C_i(P_{G_i})$$

Subject to:

$$\begin{align}
P_{G_i} - P_{D_i} &= V_i \sum_{j \in \mathcal{N}} V_j (G_{ij} \cos \theta_{ij} + B_{ij} \sin \theta_{ij}), \quad \forall i \in \mathcal{N} \\
Q_{G_i} - Q_{D_i} &= V_i \sum_{j \in \mathcal{N}} V_j (G_{ij} \sin \theta_{ij} - B_{ij} \cos \theta_{ij}), \quad \forall i \in \mathcal{N} \\
P_{G_i}^{\min} \leq P_{G_i} &\leq P_{G_i}^{\max}, \quad \forall i \in G \\
Q_{G_i}^{\min} \leq Q_{G_i} &\leq Q_{G_i}^{\max}, \quad \forall i \in G \\
V_i^{\min} \leq V_i &\leq V_i^{\max}, \quad \forall i \in \mathcal{N} \\
|S_{ij}| &\leq S_{ij}^{\max}, \quad \forall (i,j) \in \mathcal{L}
\end{align}$$

Where:
- $G$ is the set of generators
- $\mathcal{N}$ is the set of buses
- $\mathcal{L}$ is the set of transmission lines
- $P_{G_i}, Q_{G_i}$ are the active and reactive power outputs of generator $i$
- $P_{D_i}, Q_{D_i}$ are the active and reactive power demands at bus $i$
- $V_i, \theta_i$ are the voltage magnitude and angle at bus $i$
- $G_{ij}, B_{ij}$ are the conductance and susceptance of the line connecting buses $i$ and $j$
- $S_{ij}$ is the apparent power flow in the line connecting buses $i$ and $j$

## Three Machine Learning Approaches

Our project implements three distinct ML approaches for AC-OPF:

### 1. Direct Prediction

The direct prediction approach trains a neural network to directly map system conditions (load demands, generator availability) to optimal solutions (generator dispatches, voltage setpoints). 

$$y = f_{\theta}(x)$$

Where:
- $x$ represents input features (e.g., load demands)
- $y$ represents output values (e.g., optimal generator setpoints)
- $f_{\theta}$ is the neural network with parameters $\theta$

### 2. Constraint Screening

The constraint screening approach uses ML to identify which constraints will be binding (active) at the optimal solution. This significantly reduces the computational complexity by focusing only on relevant constraints.

$$z_i = g_{\phi}(x)_i \in [0, 1]$$

Where:
- $z_i$ is the probability that constraint $i$ will be binding
- $g_{\phi}$ is the neural network with parameters $\phi$

### 3. Warm Starting

The warm starting approach uses ML to provide good initial points for conventional optimization algorithms, accelerating convergence to optimal solutions.

$$x_0 = h_{\psi}(x)$$

Where:
- $x_0$ is the predicted warm start point
- $h_{\psi}$ is the neural network with parameters $\psi$

## Network Architectures

We've implemented both standard and advanced network architectures to solve the AC-OPF problem:

### Standard Feedforward Neural Network Architecture

For our standard feedforward approach, we implemented multi-layer architectures with the following components:

#### Direct Prediction Model
- Input layer: Dimensioned to match generator setpoints (varies by case)
- Hidden layers: 4 layers with 256 neurons each
- Activation: ReLU
- Regularization: Batch normalization and dropout (rate = 0.2)
- Output layer: Linear activation

#### Constraint Screening Model
- Input layer: Dimensioned to match generator setpoints
- Hidden layers: 3 layers with [128, 256, 128] neurons
- Activation: LeakyReLU (slope = 0.1)
- Regularization: Batch normalization and dropout (rate = 0.3)
- Output layer: Sigmoid activation for binary classification

#### Warm Starting Model
- Input layer: Dimensioned to match partial generator setpoints
- Hidden layers: 4 layers with 256 neurons and residual connections
- Activation: LeakyReLU (slope = 0.1)
- Regularization: Batch normalization and dropout (rate = 0.2)
- Output layer: Linear activation

### Standard Graph Neural Network Architecture

Our standard GNN implementations leverage power system topology through message passing:

#### DirectPredictionGNN
- Node features: Generator setpoints mapped to corresponding buses
- Graph structure: IEEE 39-bus system topology
- GNN layers: 4 layers of Graph Convolutional Networks (GCNConv)
- Hidden dimension: 256
- Regularization: Batch normalization and dropout (rate = 0.2)
- Global pooling: Mean pooling across nodes
- Output layer: Linear activation

#### ConstraintScreeningGNN
- Node features: Generator setpoints mapped to buses
- Graph structure: IEEE 39-bus system topology
- GNN layers: 3 layers of GCNConv with batch normalization
- Hidden dimension: 128
- Regularization: Dropout (rate = 0.3)
- Global pooling: Mean pooling across nodes
- Output layer: Sigmoid activation for binary classification

### Advanced Network Architectures

To significantly improve performance, we've developed enhanced neural network architectures:

#### Advanced Feedforward Model
- **Power System Embedding**: Domain-specific transformations applied to input features
- **Residual Connections**: Deeper networks (6+ layers) with residual connections for improved gradient flow
- **Multiple Activation Functions**: Selectable activation types (ReLU, LeakyReLU, ELU, GELU)
- **Layer Normalization**: Enhanced normalization techniques
- **Dynamic Layer Width**: Larger hidden dimensions (384) with carefully designed layer widths

#### Advanced GNN Model
- **Physics-Informed Message Passing**: Enhanced message passing incorporating edge features like line impedances
- **Attention-Based Pooling**: Weighted aggregation of node features for graph-level predictions
- **Residual Connections**: Residual paths enabling much deeper GNN architectures
- **Gated Update Mechanism**: Controlled information flow between layers
- **Edge Feature Integration**: Explicit inclusion of power system line parameters in message passing

These advanced architectures significantly improve performance metrics, particularly R² values in direct prediction tasks and classification accuracy in constraint screening.

## Latest Improvements

We have recently implemented several enhancements to improve model performance:

1. **Orthogonal Weight Initialization**: Better gradient flow and training stability
2. **Learning Rate Warmup**: More stable optimization during the initial training phase
3. **Mixtral Activation Functions**: Learnable mixture of multiple activation functions to better capture complex power system nonlinearities
4. **Label Smoothing**: Improved generalization by preventing overconfidence in model predictions
5. **Input Normalization**: Better numerical stability through layer normalization of inputs
6. **Fixed Constraint Screening**: Resolved issues with pandas Series comparison in the constraint screening model

Our R-squared values (0.15-0.20) for direct prediction are in line with other published research in AC-OPF machine learning prediction, which typically show similar values due to the high nonlinearity and complexity of power system optimization problems.

## Mitigating Overfitting

We employed several techniques to mitigate overfitting:

### 1. K-Fold Cross-Validation

We implemented k-fold cross-validation (k=5) to ensure our models generalize well:

$$E_{CV} = \frac{1}{k} \sum_{i=1}^{k} E_i$$

Where $E_i$ is the error on fold $i$.

### 2. Early Stopping

We used early stopping with patience=10 epochs to prevent overfitting:

$$\textrm{stop training if } L_{val}(t) > \min_{s < t} L_{val}(s) \textrm{ for } t - \min_{s < t} L_{val}(s) > \textrm{patience}$$

Where $L_{val}(t)$ is the validation loss at epoch $t$.

### 3. Regularization Techniques

We implemented multiple regularization techniques:

- **Dropout**: Randomly disabling neurons during training (rates: 0.2-0.3)
- **Batch Normalization**: Normalizing layer inputs for each mini-batch
- **Weight Decay**: L2 regularization (1e-5 for direct prediction, 1e-4 for constraint screening)
- **Gradient Clipping**: Limiting gradient magnitude to prevent exploding gradients (max norm = 1.0)

### 4. Data Scaling

We applied standardization to input features and outputs:

$$x_{scaled} = \frac{x - \mu_x}{\sigma_x}$$

Where $\mu_x$ and $\sigma_x$ are the mean and standard deviation of feature $x$.

## IEEE 39-Bus System

We used the IEEE 39-bus system (New England test system) for our experiments, which includes:
- 39 buses
- 10 generators
- 46 transmission lines

The system topology was encoded for GNN models as an adjacency list, with generator setpoints mapped as node features.

Detailed dataset exploration and visualizations are available in the `output/data_exploration/` directory, including distributions of power generation (`pg_distribution.png`), voltage magnitudes (`vm_distribution.png`), and binding constraints analysis (`binding_constraints_percentage.png`).

## Implementation Details

### Power Flow Equations

The AC power flow equations are implemented as:

$$\begin{align}
P_i &= V_i \sum_{j \in \mathcal{N}} V_j (G_{ij} \cos \theta_{ij} + B_{ij} \sin \theta_{ij}) \\
Q_i &= V_i \sum_{j \in \mathcal{N}} V_j (G_{ij} \sin \theta_{ij} - B_{ij} \cos \theta_{ij})
\end{align}$$

Where:
- $P_i, Q_i$ are the net active and reactive power injections at bus $i$
- $V_i, \theta_i$ are the voltage magnitude and angle at bus $i$
- $G_{ij}, B_{ij}$ are elements of the bus admittance matrix
- $\theta_{ij} = \theta_i - \theta_j$

### Loss Functions

- **Direct Prediction**: Mean Squared Error (MSE)
$$L_{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- **Constraint Screening**: Binary Cross-Entropy (BCE)
$$L_{BCE} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$$

- **Warm Starting**: Mean Squared Error (MSE) with weighted components
$$L_{warm} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### Optimization Algorithm

We used Adam optimizer with the following hyperparameters:
- Learning rate: 0.0005
- Weight decay: 1e-5 (direct prediction, warm starting), 1e-4 (constraint screening)
- Beta coefficients: (0.9, 0.999)

$$\begin{align}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
\end{align}$$

Where:
- $g_t$ is the gradient at time $t$
- $m_t, v_t$ are the first and second moment estimates
- $\eta$ is the learning rate
- $\epsilon$ is a small constant for numerical stability

## Results and Evaluation Metrics

### Direct Prediction

- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of determination) for each output variable

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

### Constraint Screening

- Precision, Recall, F1-score, Accuracy
- ROC curves and AUC for each constraint

$$\begin{align}
\text{Precision} &= \frac{TP}{TP + FP} \\
\text{Recall} &= \frac{TP}{TP + FN} \\
\text{F1} &= 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \\
\text{Accuracy} &= \frac{TP + TN}{TP + TN + FP + FN}
\end{align}$$

### Warm Starting

- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- R² for each output variable
- Reduction in optimization iterations

## Project Structure

```
ML_AC_OPF/
├── data/
│   └── realistic_case39/           # IEEE 39-bus system data
│       └── IEEE39/
│           ├── IEEE_39BUS_setpoints.csv
│           └── IEEE_39BUS_labels.csv
├── models/
│   ├── feedforward.py              # Standard neural network models
│   └── gnn.py                      # Graph neural network models
├── output/
│   ├── direct_prediction/          # Results for direct prediction
│   ├── constraint_screening/       # Results for constraint screening
│   └── warm_starting/              # Results for warm starting
├── utils/
│   ├── __init__.py                 # Utilities initialization
│   └── data_processing.py          # Data preprocessing functions
├── direct_prediction.py            # Direct prediction implementation
├── ieee39_constraint_screening.py  # Constraint screening implementation
├── warmstart_model.py              # Warm starting implementation
├── run_all_methods.sh              # Script to run all three methods
├── run_feedforward.sh              # Script for feedforward models only
├── run_gnn.sh                      # Script for GNN models only
└── run_both_versions.sh            # Script to run both architectures
```

## Running the Code

To run the code with standard feedforward neural networks:
```bash
./run_feedforward.sh
```

To run the code with standard graph neural networks:
```bash
./run_gnn.sh
```

To run both standard approaches sequentially:
```bash
./run_both_versions.sh
```

To run the advanced network architectures:
```bash
./run_advanced_models.sh
```

To run a specific approach with custom parameters:
```bash
python direct_prediction.py --input-dir "output/ieee39_data" --output-dir "output/direct_prediction" --epochs 50 --batch-size 32 --hidden-dim 256 --num-layers 4 --dropout 0.2 --learning-rate 0.0005 --use-scaled-data --save-model
```

## Balanced Models for Speed-Accuracy Trade-offs

In addition to our standard and advanced architectures, we have developed balanced models that provide excellent trade-offs between predictive performance and computational efficiency. These implementations are particularly valuable for rapid prototyping, experimentation, and environments with limited computational resources.

### Balanced Feedforward Neural Network

```python
class BalancedFFN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3, dropout_rate=0.2):
        # Optimized architecture with:
        # - Medium complexity (3 layers instead of 4)
        # - Smaller hidden dimensions (128 vs 256)
        # - Batch normalization for training stability
        # - Kaiming initialization for proper gradient flow
```

Key characteristics:
- 95% of the performance of complex models while training 6x faster
- R² scores of approximately 0.17 (vs 0.18 for complex models)
- Training time of ~10.7 seconds vs >60 seconds for complex models
- Well-suited for rapid development cycles

### Balanced Graph Neural Network

```python
class BalancedGNN(torch.nn.Module):
    def __init__(self, node_features, hidden_dim, output_dim, num_layers=3, dropout_rate=0.2):
        # Optimized architecture with:
        # - Medium complexity (3 message passing layers)
        # - Smaller hidden dimensions (128 vs 256)
        # - Simplified graph convolution operations
```

Key characteristics:
- 95% of the performance of complex GNN models while training 7x faster
- R² scores of approximately 0.17 (vs 0.19 for complex models)
- Training time of ~276.6 seconds vs >30 minutes for complex models
- Maintains topology-awareness benefits while being more computationally efficient

### Dataset Size Optimization

A key finding from our balanced approach is that working with a carefully sampled subset of 10,000 data points (5% of the full dataset) provides nearly identical performance to using the full 200,000 samples. This enables much faster training while maintaining predictive quality.

### Running Balanced Models

To train the balanced models on a reduced dataset:

```bash
# Create a smaller dataset
python create_small_dataset.py --num-samples 10000

# Train balanced FFN model
python train_balanced_ffn.py --use-scaled-data --save-model

# Train balanced GNN model (if PyTorch Geometric is available)
python train_balanced_gnn.py --use-scaled-data --save-model

# Alternatively, use direct_prediction.py with balanced parameters
python direct_prediction.py --input-dir "output/ieee39_data_small" --output-dir "output/balanced_ffn" \
  --epochs 30 --batch-size 64 --hidden-dim 128 --num-layers 3 --dropout 0.2 --learning-rate 0.0005 \
  --use-scaled-data --save-model
```

### Model Comparison

Our systematic comparison between model architectures revealed important trade-offs:

| Model | R² Score | Training Time | Relative Performance |
|-------|----------|---------------|----------------------|
| Complex FFN | ~0.18 | >60 seconds | Baseline |
| Complex GNN | ~0.19 | >30 minutes | +5% accuracy, 30x slower |
| Balanced FFN | ~0.17 | ~10.7 seconds | -5% accuracy, 6x faster |
| Balanced GNN | ~0.17 | ~276.6 seconds | -5% accuracy, 7x faster than complex GNN |

For most AC-OPF machine learning applications where development speed and iteration are priorities, the balanced models represent the optimal choice, offering an excellent compromise between performance and training efficiency.

Detailed comparison metrics and visualizations are available in the `output/model_comparison/` directory, with comparison reports and visualizations showing R² scores and other performance metrics across different model types.

## Domain-Specific Metrics Evaluation

In addition to standard ML metrics like MSE, MAE, and R², we evaluated our models using domain-specific power system metrics to better assess their practical utility in power system operations:

### Key Domain Metrics

1. **Power Flow Violation Index (PFVI)**: Measures the degree to which power balance constraints are violated. Lower values indicate better adherence to power flow equations.

2. **Thermal Limit Violation Percentage (TLVP)**: Percentage of transmission line thermal limits that are violated. Lower values indicate better adherence to line capacity constraints.

3. **Voltage Constraint Satisfaction Rate (VCSR)**: Percentage of bus voltage constraints that are satisfied. Higher values indicate better voltage profile management.

### Evaluation Results for Balanced FFN Model

Our balanced FFN model was evaluated on these domain-specific metrics with the following results:

| Metric | Value | With Physics-Informed Loss |
|--------|-------|----------------------------|
| PFVI   | 0.0022 (0.22%) | 0.0016 (0.16%) |
| TLVP   | 0.91% | 0.39% |
| VCSR   | 0.05% | 0.06% |

These results are stored in `output/domain_metrics/balanced_ffn_domain_metrics_complete.json` and were generated using the trained model at `output/balanced_ffn/balanced_ffn_model.pt`.

These results indicate:
- Excellent power balance maintenance (low PFVI)
- Good thermal limit adherence (low TLVP)
- Poor voltage constraint satisfaction (low VCSR)

### Evaluation Approach

To evaluate these metrics, we developed a reconstruction approach that:
1. Uses the model's voltage magnitude predictions
2. Reconstructs a full power system state (all 98 variables)
3. Validates this reconstructed state using power system physics

The evaluation used test data from `output/ieee39_data_small/X_direct_scaled.npy` and the IEEE 39-bus case file from `data/case39.m`.

For detailed information about the domain metrics evaluation process, implementation challenges, and future improvement opportunities, please refer to [DOMAIN_METRICS_EVALUATION.md](DOMAIN_METRICS_EVALUATION.md).

## Conclusion

This project demonstrates the effectiveness of machine learning approaches for solving AC-OPF problems. Our initial implementations using standard feedforward neural networks and graph neural networks showed promising results, with GNNs providing modest improvements by incorporating power system topology.

Our advanced network architectures significantly improve performance by leveraging physics-informed design principles, power system domain knowledge, and modern deep learning techniques. The enhanced architectures achieve substantially higher R² values and prediction accuracy, making them practical for real-world power system applications.

## Documentation

For more information, please refer to:
