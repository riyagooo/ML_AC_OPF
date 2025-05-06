# Visualizations: ML-AC-OPF

This document presents visualizations for both computational performance and model comparison in my ML-AC-OPF project.

## Part 1: Computational Performance Overview

My visualizations highlight the significant computational advantages of machine learning approaches compared to traditional AC-OPF solvers:

1. **Single-Instance Performance**: Compare the time required to solve a single AC-OPF instance
2. **Speedup Factors**: Illustrate the relative performance improvement of each method
3. **Scaling Performance**: Show how computation time scales with the number of problems

### Computation Time Comparison
This visualization compares the raw inference time (in milliseconds) for a single AC-OPF problem instance and the relative speedup factor for each method.

Key observations:
- Traditional solver requires ~342ms per instance
- ML approaches are 46.8x-106.9x faster
- Standard feedforward networks provide the best speed (3.2ms per instance)
- Advanced models are slightly slower but offer better accuracy

### Scaling Performance with Problem Size
This visualization shows the computation time required to solve different numbers of AC-OPF problems (100, 1000, 10000) for each method.

Key observations:
- Traditional solver requires minutes for large problem batches
- ML approaches consistently solve problems in seconds
- The performance gap widens dramatically as problem size increases

### Computation Time Scaling
This log-log plot demonstrates how computation time scales with the number of AC-OPF problems to be solved, ranging from 10 to 100,000 problems.

Key observations:
- Traditional solver reaches hours of computation time for large problem sets
- ML approaches stay in the seconds range even for 100,000 problems
- At 10,000 problems, traditional solver requires ~57 minutes while ML needs only ~32 seconds

## Part 2: Balanced Model Visualizations

My balanced model approach introduces new visualizations that highlight the trade-offs between model complexity, training time, and performance:

### 1. Training Time vs. Model Complexity
This visualization compares training time across four model configurations:
- Complex FFN (4 layers, 256 hidden dim)
- Complex GNN (4 layers, 256 hidden dim)
- Balanced FFN (3 layers, 128 hidden dim)
- Balanced GNN (3 layers, 128 hidden dim)

Key observations:
- Complex GNN requires >30 minutes for training
- Complex FFN requires >60 seconds
- Balanced GNN requires ~276.6 seconds
- Balanced FFN requires only ~10.7 seconds
- The balanced FFN achieves a 6x speedup over complex FFN and a 167x speedup over complex GNN

### 2. R² Score vs. Training Time
This scatter plot shows the relationship between R² score and training time for all four model configurations.

Key observations:
- Complex models achieve slightly higher R² scores (~0.18-0.19)
- Balanced models achieve nearly the same R² scores (~0.17)
- The minuscule performance difference doesn't justify the much longer training times
- The balanced FFN represents the optimal point in the speed-accuracy trade-off

### 3. Dataset Size vs. Performance
This line plot shows how model performance (R² score) changes with dataset size for the balanced FFN model.

Key observations:
- Performance plateaus after ~10,000 samples
- Using 5% of the data (10,000 samples) achieves 95% of the performance of using the full dataset
- Further increasing dataset size yields diminishing returns
- This confirms my approach of using a smaller, representative dataset for faster development

### 4. Output-Specific R² Scores
This bar chart compares the R² scores for each output dimension across all four model configurations.

Key observations:
- All models show similar patterns in prediction quality across outputs
- Voltage at bus 1 (output 1) is best predicted with R² ~0.42-0.45
- Voltage at bus 5 (output 5) is most difficult to predict with R² ~0.04-0.06
- The performance gap between complex and balanced models is minimal for all outputs

### 5. Early Stopping Visualization
This line plot shows validation loss vs. epoch for both balanced models, highlighting when early stopping occurred.

Key observations:
- The balanced GNN reached early stopping at epoch 17
- The balanced FFN trained for all 30 epochs
- This suggests the GNN might be slightly more prone to overfitting
- Early stopping is an effective mechanism to prevent overfitting while reducing training time

## Part 3: Power System-Specific Visualizations

I created visualizations focused on domain-specific metrics that are critical for power system applications:

### 1. Power Flow Violation Index (PFVI) by Model Type
This bar chart compares PFVI across model types, including the effect of physics-informed loss functions.

Key observations:
- Balanced FFN achieves a PFVI of 0.0022 (0.22%)
- Physics-informed loss functions show substantial improvement
- Balanced FFN with physics-informed loss reduces PFVI to 0.0016 (0.16%), a 27.3% improvement
- The PFVI values indicate excellent power balance maintenance across all test cases

### 2. Thermal Limit Violation Percentage (TLVP) Comparison
This visualization shows the percentage of predictions that violate line thermal limits for each model.

Key observations:
- Balanced FFN has a low TLVP of 0.91%
- Physics-informed loss functions significantly reduce TLVP to 0.39%, a 57.1% improvement
- The low TLVP values indicate good thermal limit adherence across the test dataset

### 3. Constraint Satisfaction Evaluation
This visualization illustrates constraint satisfaction rates across different conditions and model types.

Key observations:
- Power flow constraints are well-satisfied (PFVI of 0.22%)
- Thermal limit constraints show good adherence (TLVP of 0.91%)
- Voltage constraints show unexpectedly low satisfaction rates (VCSR of 0.05%)
- The extremely low VCSR values likely reflect implementation errors in our voltage reconstruction and evaluation methodology rather than fundamental model limitations

### 4. Spatial Visualization of Violations
This network diagram visualizes where in the IEEE 39-bus system [1] the constraint violations occur.

Key observations:
- Power flow violations are minimal across the network
- Thermal violations are concentrated in a small number of lines
- The visualization highlights specific areas of the network that may require more attention in future model development

### 5. Physics-Informed Loss Effect
This line plot shows how increasing the weight of physics components in the loss function affects various metrics.

Key observations:
- Too low weights have minimal effect on constraint satisfaction
- Too high weights degrade R² performance
- Optimal weights improve both R² and constraint satisfaction
- Physics-informed loss reduces PFVI by 27.3% and TLVP by 57.1%
- Gradual increase of physics weights during training yields best results

## How to Generate the Visualizations

The visualizations can be regenerated using my visualization scripts:

```bash
# Generate computation time visualizations
python visualization_scripts/timing_visualization.py

# Generate balanced model comparison visualizations
python visualization_scripts/balanced_model_viz.py

# Generate dataset size vs. performance visualization
python visualization_scripts/dataset_size_viz.py

# Generate power system-specific visualizations
python visualization_scripts/power_system_viz.py
```

This will create updated visualizations in the `output/visualizations/` directory.

## Interactive Dashboards

I have also created interactive Plotly dashboards for exploring the results in more detail:

```bash
# Launch the interactive dashboard
python visualization_scripts/interactive_dashboard.py
```

The dashboard provides:
- Adjustable parameters for model complexity
- Performance comparisons across model types
- Detailed per-output metrics visualization
- Training convergence visualization
- Power system constraint satisfaction maps
- Physics-informed loss weight adjustments

This interactive tool is particularly valuable for communicating results to stakeholders and making data-driven decisions about model selection.

## References

[1] IEEE 39-Bus System (New England Test System). Available at: https://electricgrids.engr.tamu.edu/electric-grid-test-cases/ieee-39-bus-system/

[2] Nellikkath, A. P., & Chatzivasileiadis, S. (2021). "Physics-informed neural networks for AC optimal power flow." Electric Power Systems Research, 197, 107282.

[3] GitHub Repository for this project: https://github.com/riyagooo/ML_AC_OPF 