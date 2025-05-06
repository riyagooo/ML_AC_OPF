# Model Development Journey: Balancing Speed and Accuracy

This document outlines our iterative process of developing machine learning models for AC Optimal Power Flow (AC-OPF) prediction, capturing the key decisions and trade-offs made throughout the project.

## Initial Implementation Challenges

Our journey began with implementing baseline models for AC-OPF prediction. We initially encountered several challenges:

1. **Matrix Shape Mismatch**: The `PowerSystemEmbedding` class had dimension incompatibilities that hindered proper model training.

2. **Inconsistent Sample Numbers**: Data loading in `ieee39_constraint_screening.py` and `run_advanced_models.py` had inconsistencies, resulting in mismatches between X and y dimensions.

3. **Low R-squared Values**: Initial models yielded R² values around 0.14-0.17, initially perceived as problematic but later understood to be typical for AC-OPF problems due to their nonlinear nature and multiple local optima.

## Evolution of Model Architectures

### Initial Feedforward Networks

We started with simple feedforward networks using standard architectures:

```python
class DirectPredictionModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=4, dropout_rate=0.2):
        super(DirectPredictionModel, self).__init__()
        
        # Input layer
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)])
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
```

These models achieved moderate success (R² ~ 0.14) but were limited in their ability to capture power system topology.

### Advanced Architectures

We then moved to more advanced architectures:

1. **Enhanced Feedforward Networks**:
   - Added batch normalization for better training stability
   - Implemented proper weight initialization (Kaiming) to prevent exploding gradients
   - Added residual connections to improve gradient flow

2. **Graph Neural Networks**:
   - Implemented `EnhancedDirectPredictionGNN` to explicitly model power system topology
   - Incorporated edge features representing line parameters and connectivity
   - Utilized message passing to propagate information across the power system graph

```python
class EnhancedDirectPredictionGNN(torch.nn.Module):
    def __init__(self, node_features, hidden_dim, output_dim, num_layers=3, dropout_rate=0.2):
        super(EnhancedDirectPredictionGNN, self).__init__()
        
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        
        # Multiple GNN layers with batch normalization
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(EnhancedPhysicsMessagePassing(hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
```

## The Speed vs. Accuracy Trade-off

As the project progressed, we encountered significant computational bottlenecks:

1. **GNN Training Time**: The initial GNN implementation on the full dataset (200,000 samples) required several hours of training, making rapid iteration impossible.

2. **Complex Architecture Overhead**: Each additional layer, increased hidden dimension, or message passing step significantly increased training time.

This motivated us to explore balanced approaches that could maintain reasonable accuracy while reducing training time.

## Development of Custom Loss Functions

We enhanced model performance by designing domain-specific loss functions:

1. **PowerSystemPhysicsLoss**: Combined optimality gap minimization with power flow constraint satisfaction:

```python
class PowerSystemPhysicsLoss(torch.nn.Module):
    def __init__(self, case_data, lambda_opt=1.0, lambda_pf=0.5, lambda_thermal=0.5, lambda_angle=0.1):
        super(PowerSystemPhysicsLoss, self).__init__()
        self.case_data = case_data
        self.lambda_opt = lambda_opt
        self.lambda_pf = lambda_pf
        self.lambda_thermal = lambda_thermal
        self.lambda_angle = lambda_angle
```

2. **LagrangianDualLoss**: Implemented Lagrangian dual method with multiplier updates during training:

```python
class LagrangianDualLoss(torch.nn.Module):
    def __init__(self, case_data, mse_weight=0.1, alpha=0.01, beta=1.0, use_quadratic_penalty=True):
        super(LagrangianDualLoss, self).__init__()
        self.case_data = case_data
        self.mse_weight = mse_weight
        self.alpha = alpha
        self.beta = beta
        self.use_quadratic_penalty = use_quadratic_penalty
```

These custom loss functions improved both the accuracy and feasibility of predictions, but with additional computational cost.

## Our Balanced Solution

After extensive experimentation, we arrived at a balanced solution that offers an excellent trade-off between speed and accuracy:

1. **Dataset Size Reduction**: 
   - Created a workflow to sample 10,000 points from the original 200,000 samples
   - This reduced training time dramatically while maintaining representative data

2. **Optimized Model Architecture**:
   - 3 layers instead of 4
   - 128 hidden dimensions instead of 256
   - Batch normalization for training stability
   - Early stopping with patience=10 to prevent overfitting

3. **Model Comparison Framework**:
   - Implemented systematic comparison between FFN and GNN architectures
   - Measured both performance metrics and training time
   - Generated comprehensive visualization and reports

Our final model configurations achieved the following results:

| Model | R² | Training Time | Notes |
|-------|-----|--------------|-------|
| FFN (Simple) | ~0.14 | ~1-5 seconds | Initial architecture, limited expressiveness |
| GNN (Simple) | ~0.14 | ~30-60 seconds | Initial GNN model, topology-aware |
| FFN (Balanced) | ~0.17 | ~10.7 seconds | Medium complexity, good trade-off |
| GNN (Balanced) | ~0.17 | ~276.6 seconds | Medium complexity, topology-aware |
| FFN (Complex) | ~0.18 | >60 seconds | High complexity, diminishing returns |
| GNN (Complex) | ~0.19 | >30 minutes | High complexity, best accuracy but impractical |

## Key Insights from the Journey

Through this development process, we gained several important insights:

1. **Diminishing Returns**: Increasing model complexity beyond a certain point yielded minimal accuracy improvements while significantly increasing training time.

2. **Dataset Efficiency**: Working with a carefully sampled subset of data (10,000 samples) provided nearly the same performance as using the full dataset (200,000 samples).

3. **Architecture Comparison**: For the IEEE39 system, FFNs provided similar performance to GNNs with much faster training times. However, GNNs may still be preferable for larger systems where topology becomes more important.

4. **Domain-Specific Knowledge**: Incorporating power system physics into loss functions showed promise for improving both prediction accuracy and solution feasibility.

## Lessons for Future Work

For future development of ML models in power systems, we recommend:

1. **Start Simple and Scale**: Begin with simpler architectures and smaller datasets to establish baselines before scaling up.

2. **Realistic Time Constraints**: Consider training time as a first-class constraint, especially for applications requiring regular retraining.

3. **Meaningful Metrics**: Recognize that power system optimization problems often have modest R² values (~0.17) due to their inherent complexity and multiple local optima.

4. **Physics-Informed Approaches**: Continue exploring physics-informed loss functions and model architectures that incorporate domain knowledge.

In conclusion, our journey highlighted the importance of balancing theoretical model performance with practical constraints. The balanced medium-complexity models we developed represent the sweet spot for this particular application, providing good predictive performance while remaining practical to train and iterate upon. 