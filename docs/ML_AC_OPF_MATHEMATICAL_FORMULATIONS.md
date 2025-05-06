# Mathematical Formulations for ML-AC-OPF Project

This document provides a comprehensive collection of all mathematical formulations used in the ML-AC-OPF project.

## AC Optimal Power Flow (AC-OPF) Problem

### Objective Function

The AC-OPF problem aims to find the optimal generation dispatch that minimizes cost while satisfying operational constraints. The objective function is:

$$\min_{x} \sum_{i \in G} C_i(P_{G_i})$$

Where:
- $G$ is the set of generators
- $C_i(P_{G_i})$ is the cost function for generator $i$
- $P_{G_i}$ is the active power output of generator $i$

### Constraints

The optimization is subject to the following constraints:

#### Power Flow Equations (Kirchhoff's Laws)

$$\begin{align}
P_{G_i} - P_{D_i} &= V_i \sum_{j \in \mathcal{N}} V_j (G_{ij} \cos \theta_{ij} + B_{ij} \sin \theta_{ij}), \quad \forall i \in \mathcal{N} \\
Q_{G_i} - Q_{D_i} &= V_i \sum_{j \in \mathcal{N}} V_j (G_{ij} \sin \theta_{ij} - B_{ij} \cos \theta_{ij}), \quad \forall i \in \mathcal{N}
\end{align}$$

Where:
- $P_{G_i}, Q_{G_i}$ are the active and reactive power outputs of generator $i$
- $P_{D_i}, Q_{D_i}$ are the active and reactive power demands at bus $i$
- $V_i, \theta_i$ are the voltage magnitude and angle at bus $i$
- $G_{ij}, B_{ij}$ are the conductance and susceptance of the line connecting buses $i$ and $j$
- $\theta_{ij} = \theta_i - \theta_j$ is the angle difference
- $\mathcal{N}$ is the set of all buses

#### Generator Capacity Constraints

$$\begin{align}
P_{G_i}^{\min} \leq P_{G_i} &\leq P_{G_i}^{\max}, \quad \forall i \in G \\
Q_{G_i}^{\min} \leq Q_{G_i} &\leq Q_{G_i}^{\max}, \quad \forall i \in G
\end{align}$$

#### Voltage Constraints

$$V_i^{\min} \leq V_i \leq V_i^{\max}, \quad \forall i \in \mathcal{N}$$

#### Thermal Limit Constraints

$$|S_{ij}| \leq S_{ij}^{\max}, \quad \forall (i,j) \in \mathcal{L}$$

Where:
- $S_{ij}$ is the apparent power flow in the line connecting buses $i$ and $j$
- $\mathcal{L}$ is the set of transmission lines

## Machine Learning Approaches

### 1. Direct Prediction

The direct prediction approach trains a neural network to directly map system conditions to optimal solutions:

$$y = f_{\theta}(x)$$

Where:
- $x$ represents input features (e.g., load demands)
- $y$ represents output values (e.g., optimal generator setpoints)
- $f_{\theta}$ is the neural network with parameters $\theta$

### 2. Constraint Screening

The constraint screening approach identifies which constraints will be binding at the optimal solution:

$$z_i = g_{\phi}(x)_i \in [0, 1]$$

Where:
- $z_i$ is the probability that constraint $i$ will be binding
- $g_{\phi}$ is the neural network with parameters $\phi$

### 3. Warm Starting

The warm starting approach provides good initial points for conventional optimization algorithms:

$$x_0 = h_{\psi}(x)$$

Where:
- $x_0$ is the predicted warm start point
- $h_{\psi}$ is the neural network with parameters $\psi$

## Evaluation Metrics

### Standard Machine Learning Metrics

#### Mean Squared Error (MSE)

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Where:
- $y_i$ is the true value
- $\hat{y}_i$ is the predicted value
- $n$ is the number of samples

#### Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

#### Coefficient of Determination (R²)

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

Where:
- $\bar{y}$ is the mean of the true values

### Domain-Specific Power System Metrics

#### Power Flow Violation Index (PFVI)

$$\text{PFVI} = \frac{1}{N} \sum_{i=1}^{N} \sqrt{(\Delta P_i)^2 + (\Delta Q_i)^2}$$

Where:
- $\Delta P_i$ and $\Delta Q_i$ are the active and reactive power mismatches at bus $i$
- $N$ is the number of buses

#### Thermal Limit Violation Percentage (TLVP)

$$\text{TLVP} = \frac{|\{\ell \in \mathcal{L} : |S_\ell| > S_\ell^{\max}\}|}{|\mathcal{L}|} \times 100\%$$

Where:
- $\mathcal{L}$ is the set of all transmission lines
- $S_\ell$ is the apparent power flow in line $\ell$
- $S_\ell^{\max}$ is the thermal limit of line $\ell$

#### Voltage Constraint Satisfaction Rate (VCSR)

$$\text{VCSR} = \frac{|\{i \in \mathcal{N} : V_i^{\min} \leq V_i \leq V_i^{\max}\}|}{|\mathcal{N}|} \times 100\%$$

Where:
- $\mathcal{N}$ is the set of all buses
- $V_i$ is the voltage magnitude at bus $i$
- $V_i^{\min}$ and $V_i^{\max}$ are the minimum and maximum voltage limits at bus $i$

## Physics-Informed Loss Functions

### General Form

The physics-informed loss function combines standard machine learning loss with physics-based constraints:

$$L_{\text{total}} = \lambda_{\text{opt}} \cdot L_{\text{MSE}} + \lambda_{\text{pf}} \cdot L_{\text{pf}} + \lambda_{\text{thermal}} \cdot L_{\text{thermal}} + \lambda_{\text{angle}} \cdot L_{\text{angle}}$$

Where:
- $L_{\text{MSE}}$ is the standard Mean Squared Error loss
- $L_{\text{pf}}$ is the power flow violation loss
- $L_{\text{thermal}}$ is the thermal limit violation loss
- $L_{\text{angle}}$ is the angle difference violation loss
- $\lambda$ values are weighting coefficients

### Component Losses

#### Power Flow Violation Loss

$$L_{\text{pf}} = \frac{1}{N} \sum_{i=1}^{N} \left[ \left( P_{G_i} - P_{D_i} - \sum_{j \in \mathcal{N}} P_{ij} \right)^2 + \left( Q_{G_i} - Q_{D_i} - \sum_{j \in \mathcal{N}} Q_{ij} \right)^2 \right]$$

Where:
- $P_{ij}$ and $Q_{ij}$ are the active and reactive power flows from bus $i$ to bus $j$

#### Thermal Limit Violation Loss

$$L_{\text{thermal}} = \frac{1}{|\mathcal{L}|} \sum_{\ell \in \mathcal{L}} \max(0, |S_\ell| - S_\ell^{\max})^2$$

#### Angle Difference Violation Loss

$$L_{\text{angle}} = \frac{1}{|\mathcal{L}|} \sum_{(i,j) \in \mathcal{L}} \max(0, |\theta_i - \theta_j| - \theta_{ij}^{\max})^2$$

Where:
- $\theta_{ij}^{\max}$ is the maximum allowable angle difference between buses $i$ and $j$

## Optimizer (Adam)

The Adam optimizer used for training the neural networks has the following update rules:

$$\begin{align}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
\end{align}$$

Where:
- $g_t$ is the gradient at time $t$
- $m_t, v_t$ are the first and second moment estimates
- $\beta_1, \beta_2$ are hyperparameters controlling moment decay rates
- $\alpha$ is the learning rate
- $\epsilon$ is a small constant for numerical stability

## Power System State Reconstruction

To evaluate domain-specific metrics, we need to reconstruct the full power system state from model predictions:

$$\begin{align}
\text{pg}_i &= \begin{cases}
0, & \text{if } i = 0 \text{ (slack bus)} \\
\text{input\_data}_i, & \text{otherwise}
\end{cases} \\

\text{vm}_i &= \begin{cases}
\text{predicted\_voltages}_j, & \text{if bus } i \text{ has generator } j \\
\text{predicted\_voltages}_{\text{nearest\_gen}}, & \text{otherwise}
\end{cases} \\

\text{va}_0 &= 0 \text{ (reference bus)}
\end{align}$$

Where:
- $\text{pg}_i$ is the active power of generator $i$
- $\text{vm}_i$ is the voltage magnitude at bus $i$
- $\text{va}_i$ is the voltage angle at bus $i$

## Early Stopping Criterion

The early stopping criterion used to prevent overfitting:

$$\text{stop if } L_{\text{val}}(t) > \min_{s < t} L_{\text{val}}(s) \text{ for } t - \arg\min_{s < t} L_{\text{val}}(s) > \text{patience}$$

Where:
- $L_{\text{val}}(t)$ is the validation loss at epoch $t$
- $\text{patience}$ is the number of epochs to wait for improvement

## Data Scaling

Standardization applied to input features and outputs:

$$x_{\text{scaled}} = \frac{x - \mu_x}{\sigma_x}$$

Where:
- $\mu_x$ is the mean of feature $x$
- $\sigma_x$ is the standard deviation of feature $x$

## Verified Domain Metrics Results

The following domain-specific metrics were experimentally verified with the balanced FFN model:

### Power Flow Violation Index (PFVI)

$$\text{PFVI} = \frac{1}{N} \sum_{i=1}^{N} \sqrt{(\Delta P_i)^2 + (\Delta Q_i)^2}$$

**Measured Results**:
- Balanced FFN: 0.0022 (0.22%)
- Balanced FFN with Physics-Informed Loss: 0.0016 (0.16%)

### Thermal Limit Violation Percentage (TLVP)

$$\text{TLVP} = \frac{|\{\ell \in \mathcal{L} : |S_\ell| > S_\ell^{\max}\}|}{|\mathcal{L}|} \times 100\%$$

**Measured Results**:
- Balanced FFN: 0.91%
- Balanced FFN with Physics-Informed Loss: 0.39%

### Voltage Constraint Satisfaction Rate (VCSR)

$$\text{VCSR} = \frac{|\{i \in \mathcal{N} : V_i^{\min} \leq V_i \leq V_i^{\max}\}|}{|\mathcal{N}|} \times 100\%$$

**Measured Results**:
- Balanced FFN: 0.05%
- Balanced FFN with Physics-Informed Loss: 0.06%

**Note on VCSR**: The extremely low VCSR values likely reflect implementation errors in our voltage reconstruction and evaluation methodology rather than fundamental limitations of the models. This has been identified as a critical area for improvement in future iterations.

The domain metrics were calculated using the trained balanced FFN model (`output/balanced_ffn/balanced_ffn_model.pt`) and stored in `output/domain_metrics/balanced_ffn_domain_metrics_complete.json`. The GNN model could not be evaluated due to implementation and compatibility issues with PyTorch Geometric libraries. Specifically, our environment experienced problems importing PyTorch Geometric libraries, with issues including library loading errors, dependency conflicts, and version incompatibilities. 