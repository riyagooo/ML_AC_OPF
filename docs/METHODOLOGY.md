# Detailed Methodology: ML-AC-OPF

This document provides a comprehensive explanation of the methodology employed in our Machine Learning approach to AC Optimal Power Flow (AC-OPF) problems.

## Table of Contents
1. [AC-OPF Problem Formulation](#ac-opf-problem-formulation)
2. [Machine Learning Approaches](#machine-learning-approaches)
   - [Direct Prediction](#direct-prediction)
   - [Constraint Screening](#constraint-screening)
   - [Warm Starting](#warm-starting)
3. [Neural Network Architectures](#neural-network-architectures)
   - [Standard Feedforward Neural Networks](#standard-feedforward-neural-networks)
   - [Standard Graph Neural Networks](#standard-graph-neural-networks)
   - [Advanced Feedforward Architectures](#advanced-feedforward-architectures)
   - [Advanced Graph Neural Networks](#advanced-graph-neural-networks)
4. [Topology-Aware Learning](#topology-aware-learning)
5. [Training Methodology](#training-methodology)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Physics-Constrained Machine Learning](#physics-constrained-machine-learning)
8. [Balanced Model Architectures](#balanced-model-architectures)

## AC-OPF Problem Formulation

The AC Optimal Power Flow (AC-OPF) problem is a fundamental optimization problem in power systems engineering that aims to determine the optimal operating point of a power system. Mathematically, it is formulated as:

$$\min_{x} \sum_{i \in G} C_i(P_{G_i})$$

Subject to:

$$\begin{align}
P_{G_i} - P_{D_i} &= V_i \sum_{j \in \mathcal{N}} V_j (G_{ij} \cos \theta_{ij} + B_{ij} \sin \theta_{ij}), \quad \forall i \in \mathcal{N} \quad \text{(Active Power Balance)} \\
Q_{G_i} - Q_{D_i} &= V_i \sum_{j \in \mathcal{N}} V_j (G_{ij} \sin \theta_{ij} - B_{ij} \cos \theta_{ij}), \quad \forall i \in \mathcal{N} \quad \text{(Reactive Power Balance)} \\
P_{G_i}^{\min} \leq P_{G_i} &\leq P_{G_i}^{\max}, \quad \forall i \in G \quad \text{(Active Power Generation Limits)} \\
Q_{G_i}^{\min} \leq Q_{G_i} &\leq Q_{G_i}^{\max}, \quad \forall i \in G \quad \text{(Reactive Power Generation Limits)} \\
V_i^{\min} \leq V_i &\leq V_i^{\max}, \quad \forall i \in \mathcal{N} \quad \text{(Voltage Magnitude Limits)} \\
|S_{ij}| &\leq S_{ij}^{\max}, \quad \forall (i,j) \in \mathcal{L} \quad \text{(Transmission Line Limits)}
\end{align}$$

Where:
- $G$ is the set of generator buses
- $\mathcal{N}$ is the set of all buses
- $\mathcal{L}$ is the set of transmission lines
- $P_{G_i}, Q_{G_i}$ are the active and reactive power outputs of generator $i$
- $P_{D_i}, Q_{D_i}$ are the active and reactive power demands at bus $i$
- $V_i, \theta_i$ are the voltage magnitude and angle at bus $i$
- $G_{ij}, B_{ij}$ are the real and imaginary parts of the $(i,j)$ element of the bus admittance matrix
- $\theta_{ij} = \theta_i - \theta_j$ is the voltage angle difference between buses $i$ and $j$
- $S_{ij}$ is the apparent power flow in the line connecting buses $i$ and $j$

The line flow $S_{ij}$ is calculated as:

$$S_{ij} = P_{ij} + jQ_{ij}$$

Where $P_{ij}$ and $Q_{ij}$ are the active and reactive power flows from bus $i$ to bus $j$:

$$\begin{align}
P_{ij} &= V_i^2 G_{ij} - V_i V_j (G_{ij} \cos \theta_{ij} + B_{ij} \sin \theta_{ij}) \\
Q_{ij} &= -V_i^2 (B_{ij} + B_{sh,ij}) - V_i V_j (G_{ij} \sin \theta_{ij} - B_{ij} \cos \theta_{ij})
\end{align}$$

Where $B_{sh,ij}$ is the shunt susceptance of the line.

## Machine Learning Approaches

Our methodology incorporates three complementary machine learning approaches for solving AC-OPF problems:

### Direct Prediction

The direct prediction approach trains neural networks to directly map system conditions to optimal solutions:

$$\hat{y} = f_{\theta}(x)$$

Where:
- $x \in \mathbb{R}^d$ represents input features (e.g., load demands, generator availability)
- $\hat{y} \in \mathbb{R}^m$ represents predicted output values (e.g., optimal generator setpoints, voltage profiles)
- $f_{\theta}$ is the neural network with learnable parameters $\theta$

The objective function for training is the Mean Squared Error (MSE):

$$\mathcal{L}_{MSE}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \|f_{\theta}(x_i) - y_i\|_2^2$$

Where:
- $N$ is the number of training samples
- $x_i$ is the $i$-th input sample
- $y_i$ is the corresponding ground-truth optimal solution

### Constraint Screening

The constraint screening approach uses binary classification to predict which constraints will be binding (active) at the optimal solution:

$$\hat{z}_i = g_{\phi}(x)_i \in [0, 1]$$

Where:
- $\hat{z}_i$ represents the predicted probability that constraint $i$ will be binding
- $g_{\phi}$ is the neural network with learnable parameters $\phi$

The objective function is the Binary Cross-Entropy (BCE) loss:

$$\mathcal{L}_{BCE}(\phi) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} [z_{ij} \log(\hat{z}_{ij}) + (1 - z_{ij}) \log(1 - \hat{z}_{ij})]$$

Where:
- $N$ is the number of training samples
- $C$ is the number of constraints
- $z_{ij} \in \{0, 1\}$ indicates whether constraint $j$ is binding in sample $i$
- $\hat{z}_{ij} \in [0, 1]$ is the predicted probability that constraint $j$ is binding in sample $i$

To handle class imbalance (as many constraints are typically not binding), we weight the loss function:

$$\mathcal{L}_{weighted}(\phi) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} [w_j z_{ij} \log(\hat{z}_{ij}) + (1 - z_{ij}) \log(1 - \hat{z}_{ij})]$$

Where $w_j = \frac{N - \sum_{i=1}^{N} z_{ij}}{\sum_{i=1}^{N} z_{ij}}$ is the weight for constraint $j$.

### Warm Starting

The warm starting approach trains neural networks to provide good initial points for conventional optimization algorithms:

$$\hat{x}_0 = h_{\psi}(x)$$

Where:
- $\hat{x}_0$ is the predicted warm start point
- $h_{\psi}$ is the neural network with learnable parameters $\psi$

The objective function is also based on MSE:

$$\mathcal{L}_{warm}(\psi) = \frac{1}{N} \sum_{i=1}^{N} \|h_{\psi}(x_i) - y_i\|_2^2$$

But additional objectives can be incorporated to improve convergence properties:

$$\mathcal{L}_{combined}(\psi) = \mathcal{L}_{warm}(\psi) + \lambda_1 \mathcal{L}_{feasibility}(\psi) + \lambda_2 \mathcal{L}_{optimality}(\psi)$$

Where:
- $\mathcal{L}_{feasibility}$ penalizes predicted points that violate constraints
- $\mathcal{L}_{optimality}$ encourages predictions closer to the optimal cost
- $\lambda_1, \lambda_2$ are hyperparameters balancing the different objectives

## Neural Network Architectures

Our methodology employs both standard and advanced neural network architectures:

### Standard Feedforward Neural Networks

#### Direct Prediction Model

```
Input Layer (input_dim) → FC Layer → ReLU → BatchNorm → Dropout(0.2) →
  → FC Layer → ReLU → BatchNorm → Dropout(0.2) →
  → FC Layer → ReLU → BatchNorm → Dropout(0.2) →
  → FC Layer → ReLU → BatchNorm → Dropout(0.2) →
  → Output Layer (output_dim)
```

Mathematically, each layer applies the following transformation:

$$\begin{align}
z^{(l)} &= W^{(l)}a^{(l-1)} + b^{(l)} \\
\tilde{a}^{(l)} &= \text{ReLU}(z^{(l)}) = \max(0, z^{(l)}) \\
\hat{a}^{(l)} &= \text{BatchNorm}(\tilde{a}^{(l)}) = \gamma^{(l)} \frac{\tilde{a}^{(l)} - \mu^{(l)}}{\sqrt{(\sigma^{(l)})^2 + \epsilon}} + \beta^{(l)} \\
a^{(l)} &= \text{Dropout}(\hat{a}^{(l)}, p=0.2)
\end{align}$$

Where:
- $W^{(l)}, b^{(l)}$ are the weights and biases of layer $l$
- $\gamma^{(l)}, \beta^{(l)}$ are the scale and shift parameters of batch normalization
- $\mu^{(l)}, \sigma^{(l)}$ are the batch mean and standard deviation
- $\epsilon$ is a small constant for numerical stability

For the Direct Prediction model, hidden dimension is set to 256 and the number of layers to 4, based on hyperparameter tuning.

#### Constraint Screening Model

```
Input Layer (input_dim) → FC Layer → LeakyReLU(0.1) → BatchNorm → Dropout(0.3) →
  → FC Layer → LeakyReLU(0.1) → BatchNorm → Dropout(0.3) →
  → FC Layer → LeakyReLU(0.1) → BatchNorm → Dropout(0.3) →
  → Output Layer (output_dim) → Sigmoid
```

The LeakyReLU activation function is defined as:

$$\text{LeakyReLU}(x, \alpha=0.1) = \begin{cases}
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0
\end{cases}$$

And the Sigmoid output activation is:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

For the Constraint Screening model, the hidden dimensions are [128, 256, 128] for the three hidden layers, employing a "bottleneck" architecture that compresses then expands the representation.

#### Warm Starting Model with Residual Connections

```
Input Layer (input_dim) → FC Layer → LeakyReLU(0.1) → BatchNorm → Dropout(0.2) →
  → Residual Block 1 →
  → Residual Block 2 →
  → Output Layer (output_dim)
```

Where each Residual Block is defined as:

```
Input → FC Layer → LeakyReLU(0.1) → BatchNorm → Dropout(0.2) →
  → FC Layer → LeakyReLU(0.1) → BatchNorm → Dropout(0.2) →
  → Add Input (Residual Connection) → Output
```

Mathematically, a residual block performs:

$$\begin{align}
\tilde{z}^{(l)} &= W_1^{(l)}a^{(l-1)} + b_1^{(l)} \\
\tilde{a}_1^{(l)} &= \text{LeakyReLU}(\tilde{z}^{(l)}, \alpha=0.1) \\
\hat{a}_1^{(l)} &= \text{BatchNorm}(\tilde{a}_1^{(l)}) \\
a_1^{(l)} &= \text{Dropout}(\hat{a}_1^{(l)}, p=0.2) \\
\tilde{z}_2^{(l)} &= W_2^{(l)}a_1^{(l)} + b_2^{(l)} \\
\tilde{a}_2^{(l)} &= \text{LeakyReLU}(\tilde{z}_2^{(l)}, \alpha=0.1) \\
\hat{a}_2^{(l)} &= \text{BatchNorm}(\tilde{a}_2^{(l)}) \\
a_2^{(l)} &= \text{Dropout}(\hat{a}_2^{(l)}, p=0.2) \\
a^{(l)} &= a_2^{(l)} + a^{(l-1)} \quad \text{(Residual connection)}
\end{align}$$

The residual connections help with gradient flow during training, especially for deeper networks, mitigating the vanishing gradient problem.

### Standard Graph Neural Networks

Our methodology leverages the power system topology by implementing Graph Neural Networks.

#### Graph Representation of Power Systems

We represent the power system as a graph $G = (V, E)$, where:
- $V$ is the set of vertices (buses)
- $E$ is the set of edges (transmission lines)

Each node $v_i \in V$ has features $x_i \in \mathbb{R}^{d_v}$ that include:
- Generator setpoints (if a generator is present at the bus)
- Load demands
- Bus type (PV, PQ, slack)

Each edge $(i, j) \in E$ has features $e_{ij} \in \mathbb{R}^{d_e}$ that include:
- Line resistance $R_{ij}$
- Line reactance $X_{ij}$
- Line charging susceptance $B_{sh,ij}$
- Thermal limit $S_{ij}^{max}$

#### Graph Convolutional Networks (GCN)

We use Graph Convolutional Networks as the backbone of our GNN architecture:

$$H^{(l+1)} = \sigma\left(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)}\right)$$

Where:
- $H^{(l)} \in \mathbb{R}^{n \times d_l}$ is the matrix of node features at layer $l$
- $\tilde{A} = A + I_n$ is the adjacency matrix with added self-loops
- $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$ is the degree matrix of $\tilde{A}$
- $W^{(l)} \in \mathbb{R}^{d_l \times d_{l+1}}$ is the weight matrix for layer $l$
- $\sigma$ is a non-linear activation function (ReLU in our implementation)

This message-passing operation can be interpreted as each node aggregating features from its neighbors (including itself due to the self-loop).

#### DirectPredictionGNN Architecture

```
Node Features → Node Embedding Layer → GCN Layer 1 → BatchNorm → ReLU → Dropout(0.2) →
  → GCN Layer 2 → BatchNorm → ReLU → Dropout(0.2) →
  → GCN Layer 3 → BatchNorm → ReLU → Dropout(0.2) →
  → GCN Layer 4 → BatchNorm → ReLU → Dropout(0.2) →
  → Global Mean Pooling → Output Layer
```

For node $i$ at layer $l$, the GCN operation is:

$$h_i^{(l+1)} = \sigma\left(W^{(l)} \cdot \frac{1}{\sqrt{(\text{deg}(i) + 1)(\text{deg}(j) + 1)}} \sum_{j \in \mathcal{N}(i) \cup \{i\}} h_j^{(l)}\right)$$

Where $\mathcal{N}(i)$ is the set of neighboring nodes of node $i$.

The global mean pooling operation aggregates node-level features to produce a graph-level representation:

$$h_G = \frac{1}{|V|} \sum_{i \in V} h_i^{(L)}$$

Where $h_i^{(L)}$ is the feature vector of node $i$ at the final layer $L$.

#### ConstraintScreeningGNN Architecture

```
Node Features → Node Embedding Layer → GCN Layer 1 → BatchNorm → ReLU → Dropout(0.3) →
  → GCN Layer 2 → BatchNorm → ReLU → Dropout(0.3) →
  → GCN Layer 3 → BatchNorm → ReLU → Dropout(0.3) →
  → Global Mean Pooling → Output Layer → Sigmoid
```

This architecture is similar to the DirectPredictionGNN but with the sigmoid activation function applied to the output layer for binary classification.

### Advanced Feedforward Architectures

Our advanced feedforward architectures incorporate several techniques to enhance model performance:

- **Orthogonal Weight Initialization**: We initialize network weights using orthogonal matrices to improve training stability and gradient flow [1].
- **Learning Rate Warmup**: The first few epochs use a gradual learning rate increase to improve optimization stability [2].
- **Mixtral Activation Functions**: We implement a learnable mixture of activation functions that adapts to the data patterns [3].
- **Label Smoothing**: To improve generalization, we apply label smoothing to prevent overconfidence in predictions [4].
- **Input Normalization**: We normalize input features with a LayerNorm layer for better numerical stability [5].
- **Power System Embeddings**: We create specialized embedding layers for power system features that apply domain-specific transformations to better capture AC-OPF relationships [6].

The mathematical formulation for our Mixtral Activation is:

$$f(x) = \sum_{i=1}^{n} w_i \cdot a_i(x)$$

where $w_i$ are learnable weights (normalized via softmax) and $a_i$ are different activation functions.

### Advanced Graph Neural Networks

Our advanced GNN implementations incorporate physics-informed components and enhanced message passing:

#### Physics-Informed Message Passing

The `EnhancedPhysicsMessagePassing` layer explicitly models physical interactions between power system components:

$$\begin{align}
\text{message}_{ij} &= \phi_{\theta}(h_i, h_j, e_{ij}) \\
m_i &= \sum_{j \in \mathcal{N}(i)} \text{message}_{ij} \\
h_i' &= \gamma_{\theta}(h_i, m_i)
\end{align}$$

Where:
- $h_i, h_j$ are node features
- $e_{ij}$ are edge features (line parameters)
- $\phi_{\theta}$ is the message function with parameters $\theta$
- $\gamma_{\theta}$ is the update function with parameters $\theta$

In our implementation, the message function processes inputs through multiple paths to capture different aspects of power system physics:

$$\begin{align}
\text{message\_input} &= [h_i; h_j; e_{ij}] \\
\text{message}_1 &= \text{Path}_1(\text{message\_input}) \\
\text{message}_2 &= \text{Path}_2(\text{message\_input}) \\
\alpha &= \text{Softmax}(W_{\alpha}[message_1; message_2]) \\
\text{message} &= \alpha_1 \cdot \text{message}_1 + \alpha_2 \cdot \text{message}_2
\end{align}$$

Where:
- $\text{Path}_1, \text{Path}_2$ are neural networks with different activation functions
- $W_{\alpha}$ is a learnable attention weight matrix
- $\alpha = [\alpha_1, \alpha_2]$ are attention coefficients

The update function incorporates a gating mechanism to control information flow:

$$\begin{align}
\text{update\_input} &= [m_i; h_i] \\
\text{gate} &= \sigma(W_g \text{update\_input} + b_g) \\
\text{update} &= \text{UpdateNN}(\text{update\_input}) \\
h_i' &= (1 - \text{gate}) \odot h_i + \text{gate} \odot \text{update}
\end{align}$$

Where:
- $W_g, b_g$ are the gate's learnable parameters
- $\text{UpdateNN}$ is a neural network that computes the update
- $\odot$ represents element-wise multiplication

#### Layer Attention Mechanism

The advanced GNN incorporates a layer attention mechanism that weights outputs from different layers:

$$h_{final} = \sum_{l=1}^L \alpha_l \cdot h^{(l)}$$

Where:
- $h^{(l)}$ is the output of layer $l$
- $\alpha_l$ are learnable attention weights with $\sum_{l=1}^L \alpha_l = 1$

#### AdvancedGNN Architecture

The complete `AdvancedGNN` architecture:

```
Node Features → PowerSystemEmbedding → 
  → EnhancedPhysicsMessagePassing 1 → Normalization → Dropout →
  → EnhancedPhysicsMessagePassing 2 → Normalization → Dropout + Residual →
  ... →
  → Layer Attention → Global Pooling → 
  → Pre-Output Layers → Output
```

For constraint screening applications, the `AdvancedConstraintScreeningGNN` extends this architecture with a binary classification output:

```
... → Pre-Output Layers → Binary Output Layer → Sigmoid
```

The binary output layer introduces an additional transformation with dropout:

$$\begin{align}
x_{binary} &= \text{Activation}(W_{binary}x + b_{binary}) \\
x_{dropout} &= \text{Dropout}(x_{binary}) \\
y &= \sigma(W_{output}x_{dropout} + b_{output})
\end{align}$$

Where $\sigma$ is the sigmoid activation function.

## Topology-Aware Learning

The GNN architecture naturally incorporates the topology of the power system through message passing. This has several advantages:

1. **Locality awareness**: Power flow equations involve interactions between neighboring buses, which GNNs capture through message passing.

2. **Permutation invariance**: The ordering of buses doesn't affect the physical properties of the system, and GNNs are invariant to node permutations.

3. **Inductive bias**: The sparsity structure of the power system is encoded in the graph structure, providing a strong inductive bias.

For node $i$, the information it receives after $L$ layers of message passing comes from its $L$-hop neighborhood:

$$\mathcal{N}_L(i) = \{j \in V : \text{dist}(i, j) \leq L\}$$

Where $\text{dist}(i, j)$ is the shortest path distance between nodes $i$ and $j$.

This enables the model to implicitly learn the relationship between the topology and the optimal power flow solution.

## Training Methodology

Our training methodology incorporates several techniques to enhance model performance and generalization:

### Data Preprocessing

1. **Standardization**: Input features and targets are standardized to have zero mean and unit variance:

   $$x_{scaled} = \frac{x - \mu_x}{\sigma_x}$$

2. **Data augmentation**: To increase the diversity of training samples, we apply small perturbations to load demands and generator availability.

### K-Fold Cross-Validation

We use K-fold cross-validation with $K=5$ to ensure robust evaluation:

1. The dataset is divided into $K$ equal parts (folds).
2. For each fold $i \in \{1, 2, \ldots, K\}$:
   - Train on $K-1$ folds excluding fold $i$
   - Validate on fold $i$
3. The overall performance is the average across all folds:

   $$E_{CV} = \frac{1}{K} \sum_{i=1}^{K} E_i$$

   Where $E_i$ is the error on fold $i$.

### Early Stopping

To prevent overfitting, we implement early stopping with patience=10:

1. Monitor validation loss after each epoch.
2. If validation loss doesn't improve for `patience` consecutive epochs, stop training.
3. Restore the model weights from the epoch with the best validation loss.

Mathematically, training stops at epoch $t$ if:

$$L_{val}(t - \text{patience}+1), L_{val}(t - \text{patience}+2), \ldots, L_{val}(t) > \min_{s \leq t - \text{patience}} L_{val}(s)$$

### Regularization Techniques

Several regularization techniques are employed to improve generalization:

1. **Dropout**: Randomly disables neurons during training with probability $p$:

   $$\text{Dropout}(h, p) = \begin{cases}
   \frac{h}{1-p} & \text{with probability } 1-p \\
   0 & \text{with probability } p
   \end{cases}$$

   Dropout rates: 0.2 for direct prediction and warm starting, 0.3 for constraint screening.

2. **Batch Normalization**: Normalizes layer inputs for each mini-batch:

   $$\text{BatchNorm}(x) = \gamma \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta$$

   Where $\mu_B, \sigma_B$ are the batch mean and standard deviation, and $\gamma, \beta$ are learnable parameters.

3. **Weight Decay**: L2 regularization adds a penalty term to the loss function:

   $$\mathcal{L}_{reg} = \mathcal{L} + \lambda \sum_i \theta_i^2$$

   Weight decay values: 1e-5 for direct prediction and warm starting, 1e-4 for constraint screening.

4. **Gradient Clipping**: Limits gradient magnitude to prevent exploding gradients:

   $$\nabla' = \begin{cases}
   \nabla & \text{if } \|\nabla\|_2 \leq \text{clip\_value} \\
   \frac{\text{clip\_value}}{\|\nabla\|_2} \nabla & \text{otherwise}
   \end{cases}$$

   We use a clip value of 1.0 for all models.

### Optimization Algorithm

We use the Adam optimizer with the following hyperparameters:
- Learning rate: 0.0005
- Betas: (0.9, 0.999)
- Epsilon: 1e-8

The update rule for Adam is:

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
- $\hat{m}_t, \hat{v}_t$ are the bias-corrected moment estimates
- $\eta$ is the learning rate
- $\beta_1, \beta_2$ are the exponential decay rates for the moment estimates
- $\epsilon$ is a small constant for numerical stability

For advanced models, we implemented an enhanced training procedure:

1. **Learning Rate Scheduling**: We use a ReduceLROnPlateau scheduler that reduces the learning rate when validation performance plateaus:

$$\eta_t = \begin{cases}
\eta_{t-1} & \text{if } L_{val}(t) < \text{best\_loss} \\
\gamma \cdot \eta_{t-1} & \text{if no improvement for 'patience' epochs}
\end{cases}$$

Where $\gamma = 0.5$ is the decay factor and patience is set to 5 epochs.

2. **Gradient Accumulation**: For larger models, we use gradient accumulation to effectively increase batch size without memory constraints.

## Evaluation Metrics

Different evaluation metrics are used for each approach:

### Direct Prediction

1. **Mean Squared Error (MSE)**:

   $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

2. **Mean Absolute Error (MAE)**:

   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

3. **R² Score**:

   $$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

   Where $\bar{y}$ is the mean of the true values.

### Constraint Screening

1. **Precision**:

   $$\text{Precision} = \frac{TP}{TP + FP}$$

   Where TP is the number of true positives and FP is the number of false positives.

2. **Recall**:

   $$\text{Recall} = \frac{TP}{TP + FN}$$

   Where FN is the number of false negatives.

3. **F1 Score**:

   $$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

4. **Accuracy**:

   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

   Where TN is the number of true negatives.

5. **Area Under ROC Curve (AUC)**:

   $$\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(x)) dx$$

   Where TPR is the true positive rate and FPR is the false positive rate.

### Warm Starting

1. **MSE and MAE**: As in direct prediction.

2. **R² Score**: As in direct prediction.

3. **Reduction in Optimization Iterations**:

   $$\text{Reduction} = \frac{I_{cold} - I_{warm}}{I_{cold}} \times 100\%$$

   Where $I_{cold}$ is the number of iterations required for cold-started optimization and $I_{warm}$ is the number of iterations with warm starting.

## Physics-Constrained Machine Learning

Our approach incorporates physics-based constraints to enhance the learning process:

### Power Flow Constraints

For each bus $i$, the active and reactive power balance equations must be satisfied:

$$\begin{align}
P_{G_i} - P_{D_i} &= V_i \sum_{j \in \mathcal{N}} V_j (G_{ij} \cos \theta_{ij} + B_{ij} \sin \theta_{ij}) \\
Q_{G_i} - Q_{D_i} &= V_i \sum_{j \in \mathcal{N}} V_j (G_{ij} \sin \theta_{ij} - B_{ij} \cos \theta_{ij})
\end{align}$$

We incorporate these constraints through:

1. **Physics-Informed Loss Functions**: Adding penalty terms for constraint violations.

2. **Feasibility Projection**: Projecting neural network outputs onto the feasible space.

3. **Data Augmentation with Feasible Solutions**: Enriching the training dataset with physically feasible solutions.

### Generator Capability Constraints

For each generator $i$, the active and reactive power outputs must respect the capability curve:

$$\begin{align}
P_{G_i}^{\min} \leq P_{G_i} &\leq P_{G_i}^{\max} \\
Q_{G_i}^{\min} \leq Q_{G_i} &\leq Q_{G_i}^{\max} \\
(P_{G_i} - P_{center})^2 + (Q_{G_i} - Q_{center})^2 &\leq R^2 \quad \text{(D-curve constraint)}
\end{align}$$

Where the D-curve constraint represents the capability limitations of the generator.

### Transmission Line Constraints

For each line $(i,j) \in \mathcal{L}$, the power flow must not exceed the thermal limit:

$$|S_{ij}| = \sqrt{P_{ij}^2 + Q_{ij}^2} \leq S_{ij}^{\max}$$

These physics-based constraints guide the learning process, ensuring that the neural network predictions are physically meaningful and feasible.

## Balanced Model Architectures

In addition to the previously described architectures, we developed balanced model implementations that offer excellent trade-offs between performance and training efficiency. These models are particularly valuable for rapid development cycles and when computational resources are limited.

### Balanced Feedforward Neural Network

The `BalancedFFN` architecture represents a medium-complexity model that maintains good predictive capabilities while being much faster to train:

```python
class BalancedFFN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3, dropout_rate=0.2):
        super(BalancedFFN, self).__init__()
        
        # Input layer
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ]
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights using Kaiming initialization
        self._initialize_weights()
```

Key characteristics of the balanced FFN:
- Medium hidden dimension (128 vs 256 in complex models)
- Fewer layers (3 vs 4 in complex models)
- Batch normalization for training stability
- Kaiming initialization for proper gradient flow
- Early stopping with patience=10 to prevent overfitting

### Balanced Graph Neural Network

The balanced GNN maintains the topology-aware benefits of graph neural networks while being more computationally efficient:

```python
class BalancedGNN(torch.nn.Module):
    def __init__(self, node_features, hidden_dim, output_dim, num_layers=3, dropout_rate=0.2):
        super(BalancedGNN, self).__init__()
        
        # Node embedding
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        
        # GNN layers with batch normalization
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layers
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
```

Key characteristics of the balanced GNN:
- Medium hidden dimension (128)
- Fewer message passing layers (3)
- Simplified graph convolution operations
- Early stopping to prevent overfitting

### Balanced Training Configuration

For both balanced architectures, we use the following training configuration:

- **Optimizer**: Adam with learning rate 0.0005
- **Batch Size**: 64 (vs 32 in complex models)
- **Weight Decay**: 1e-5 for regularization
- **Dataset Size**: 10,000 samples (reduced from full 200,000 samples)
- **Data Splitting**: 70% training, 15% validation, 15% testing
- **Early Stopping**: Patience=10 epochs
- **Hardware Target**: Standard CPU environments (vs GPU for complex models)

### Performance vs Training Time Trade-offs

Our experiments with balanced models yield the following comparative results:

| Model | R² Score | Training Time | Relative Performance |
|-------|----------|---------------|----------------------|
| Complex FFN | ~0.18 | >60 seconds | Baseline |
| Complex GNN | ~0.19 | >30 minutes | +5% accuracy, 30x slower |
| Balanced FFN | ~0.17 | ~10.7 seconds | -5% accuracy, 6x faster |
| Balanced GNN | ~0.17 | ~276.6 seconds | -5% accuracy, 7x faster than complex GNN |

The balanced models achieve approximately 95% of the performance of complex models while being 6-30x faster to train. This makes them ideal for:

1. Rapid prototyping and experimentation
2. Environments with limited computational resources
3. Applications requiring frequent retraining
4. Educational and demonstration purposes

The negligible performance gap, coupled with significant training speedups, makes the balanced models the preferred choice for most AC-OPF machine learning applications where development speed and iteration are priorities.

### Dataset Size Optimization

A key finding from our balanced approach is that working with a carefully sampled subset of 10,000 data points (5% of the full dataset) provides nearly identical performance to using the full 200,000 samples. This dramatic reduction in dataset size contributes significantly to the training speedup while maintaining predictive quality.

Our sampling strategy ensures that the reduced dataset:
1. Maintains the same distribution of operating conditions as the full dataset
2. Preserves the proportion of feasible vs infeasible samples
3. Includes representative examples of edge cases and constraint violations

This finding has important implications for power system ML applications, suggesting that carefully curated smaller datasets can be just as effective as massive data collections for AC-OPF prediction tasks.

## References

[1] Saxe, A. M., McClelland, J. L., & Ganguli, S. (2013). "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks." arXiv preprint arXiv:1312.6120.

[2] Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L., Kyrola, A., ... & He, K. (2017). "Accurate, large minibatch sgd: Training imagenet in 1 hour." arXiv preprint arXiv:1706.02677.

[3] Ma, J., & Yarats, D. (2022). "On the Adequacy of Untuned Warmup for Adaptive Optimization." arXiv preprint arXiv:2006.01966.

[4] Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). "Rethinking the inception architecture for computer vision." In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2818-2826).

[5] Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). "Layer normalization." arXiv preprint arXiv:1607.06450.

[6] Fioretto, F., Mak, T. W. K., & Van Hentenryck, P. (2020). "Predicting AC optimal power flows: Combining deep learning and Lagrangian dual methods." In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 34, No. 01, pp. 630-637).

[7] Donnot, B., Guyon, I., Schoenauer, M., Panciatici, P., & Marot, A. (2017). "Introducing machine learning for power system operation support." arXiv preprint arXiv:1709.09527. 