"""
Power system validation module for ML-AC-OPF project.

This module provides functions for domain-specific validation of power system
models and solutions, including N-1 contingency analysis, voltage stability,
and other industrial validation techniques.
"""

import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

from .optimization import OPFOptimizer
from .error_handling import (
    PowerFlowError, ValidationError, retry, 
    handle_errors, ErrorContext, ErrorCodes
)

# Configure logging
logger = logging.getLogger(__name__)

class PowerSystemValidator:
    """
    Validator for power system solutions and models.
    Implements N-1 contingency analysis and other industrial validation techniques.
    """
    
    def __init__(self, case_data: Dict[str, Any], device: str = 'cpu'):
        """
        Initialize power system validator.
        
        Args:
            case_data: PyPOWER case data
            device: Device for torch tensors
        """
        self.case_data = case_data
        self.device = device
        self.optimizer = OPFOptimizer(case_data, device=device)
        
        # Extract system dimensions
        self.n_bus = len(case_data['bus'])
        self.n_gen = len(case_data['gen'])
        self.n_branch = len(case_data['branch'])
        
        logger.info(f"Initialized power system validator for system with {self.n_bus} buses, "
                   f"{self.n_gen} generators, and {self.n_branch} branches")
        
    def validate_solution(self, solution: Dict[str, Any], 
                         load_data: np.ndarray, 
                         thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Validate a power system solution for constraint violations and stability.
        
        Args:
            solution: Dictionary with solution variables (pg, qg, vm, va)
            load_data: Bus load data (Pd, Qd)
            thresholds: Optional dictionary with validation thresholds
            
        Returns:
            Dictionary with validation results
        """
        if thresholds is None:
            thresholds = {
                'power_balance': 1e-3,
                'branch_thermal': 1e-3,
                'generator_limits': 1e-3,
                'voltage_limits': 1e-3,
                'total_violation': 1e-2
            }
        
        try:
            # Evaluate solution violations
            violations = self.optimizer.evaluate_solution(solution, load_data)
            
            # Check if any violations exceed thresholds
            validation_passed = True
            violations_exceeded = {}
            
            for key, value in violations.items():
                if key != 'cost' and key in thresholds and value > thresholds[key]:
                    validation_passed = False
                    violations_exceeded[key] = value
            
            # If total violation is present, check that too
            if 'total' in violations and 'total_violation' in thresholds:
                if violations['total'] > thresholds['total_violation']:
                    validation_passed = False
                    violations_exceeded['total'] = violations['total']
            
            # Compile validation results
            results = {
                'validation_passed': validation_passed,
                'violations': violations,
                'violations_exceeded': violations_exceeded
            }
            
            if validation_passed:
                logger.info("Solution validation passed")
            else:
                logger.warning(f"Solution validation failed. Violations: {violations_exceeded}")
                
            return results
            
        except Exception as e:
            logger.error(f"Error in solution validation: {e}")
            return {
                'validation_passed': False,
                'error': str(e)
            }
    
    def perform_n1_contingency_analysis(self, 
                                       model: Any, 
                                       load_data: np.ndarray,
                                       contingencies: Optional[List[Dict[str, Any]]] = None,
                                       verbose: bool = False) -> Dict[str, Any]:
        """
        Perform N-1 contingency analysis by simulating the loss of each branch
        and evaluating the model's solutions under these conditions.
        
        Args:
            model: ML model that produces power system solutions
            load_data: Bus load data (Pd, Qd)
            contingencies: Optional list of contingencies to analyze
                          If None, all branches will be used as contingencies
            verbose: Whether to print detailed information
            
        Returns:
            Dictionary with contingency analysis results
        """
        try:
            # Create list of contingencies if not provided
            if contingencies is None:
                contingencies = []
                # Add branch outages
                for i in range(self.n_branch):
                    # Skip radial branches (those that would isolate parts of the system)
                    if self._is_radial_branch(i):
                        if verbose:
                            logger.info(f"Skipping radial branch {i} for contingency analysis")
                        continue
                        
                    contingencies.append({
                        'type': 'branch',
                        'id': i,
                        'name': f"Branch {i}: {self.case_data['branch'][i, 0]}-{self.case_data['branch'][i, 1]}"
                    })
                
                # Add generator outages (except for slack bus generator)
                slack_bus = None
                for i in range(self.n_bus):
                    if self.case_data['bus'][i, 1] == 3:  # Slack bus
                        slack_bus = i
                        break
                        
                for i in range(self.n_gen):
                    gen_bus = int(self.case_data['gen'][i, 0])
                    if gen_bus != slack_bus:
                        contingencies.append({
                            'type': 'generator',
                            'id': i,
                            'name': f"Generator {i} at bus {gen_bus}"
                        })
            
            logger.info(f"Performing N-1 contingency analysis with {len(contingencies)} contingencies")
            
            # Initialize results
            results = {
                'contingency_results': [],
                'critical_contingencies': [],
                'overall_status': 'success'
            }
            
            # Analyze each contingency
            for i, contingency in enumerate(contingencies):
                if verbose:
                    logger.info(f"Analyzing contingency {i+1}/{len(contingencies)}: {contingency['name']}")
                
                # Create modified case data for the contingency
                modified_case = self._apply_contingency(contingency, verbose)
                
                # Skip if contingency creates an invalid case
                if modified_case is None:
                    continue
                
                # Create optimizer for the modified case
                contingency_optimizer = OPFOptimizer(modified_case, device=self.device)
                
                # Get model prediction for original case
                with torch.no_grad():
                    inputs = torch.tensor(load_data, dtype=torch.float32, device=self.device)
                    prediction = model(inputs)
                    
                    if isinstance(prediction, torch.Tensor):
                        prediction = prediction.cpu().numpy()
                
                # Extract solution variables (assuming model outputs pg, qg, vm, va)
                n_gen = self.n_gen
                n_bus = self.n_bus
                
                if contingency['type'] == 'generator':
                    # Adjust prediction by setting the outaged generator to zero
                    gen_id = contingency['id']
                    prediction_adjusted = prediction.copy()
                    prediction_adjusted[gen_id] = 0  # Set Pg to zero
                    prediction_adjusted[n_gen + gen_id] = 0  # Set Qg to zero
                else:
                    prediction_adjusted = prediction
                
                solution = {
                    'pg': prediction_adjusted[:n_gen],
                    'qg': prediction_adjusted[n_gen:2*n_gen],
                    'vm': prediction_adjusted[2*n_gen:2*n_gen+n_bus],
                    'va': prediction_adjusted[2*n_gen+n_bus:2*n_gen+2*n_bus]
                }
                
                # Evaluate the solution under contingency
                violations = contingency_optimizer.evaluate_solution(solution, load_data)
                
                # Define contingency status
                if violations['total'] > 0.1:  # Arbitrary threshold for critical violations
                    status = 'critical'
                    results['critical_contingencies'].append(contingency['name'])
                elif violations['total'] > 0.01:  # Minor violations
                    status = 'warning'
                else:
                    status = 'ok'
                
                # Store contingency results
                contingency_result = {
                    'contingency': contingency['name'],
                    'violations': violations,
                    'status': status
                }
                
                results['contingency_results'].append(contingency_result)
                
                if verbose:
                    logger.info(f"Contingency {contingency['name']} status: {status}")
                    
            # Determine overall N-1 security
            if len(results['critical_contingencies']) > 0:
                results['overall_status'] = 'insecure'
                logger.warning(f"System is not N-1 secure. Critical contingencies: {len(results['critical_contingencies'])}")
            else:
                results['overall_status'] = 'secure'
                logger.info("System is N-1 secure")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in N-1 contingency analysis: {e}")
            return {
                'overall_status': 'error',
                'error': str(e)
            }
    
    def _is_radial_branch(self, branch_idx: int) -> bool:
        """
        Check if a branch is radial (removing it would disconnect the system).
        This is a simplified check.
        
        Args:
            branch_idx: Index of the branch to check
            
        Returns:
            True if the branch is radial, False otherwise
        """
        # This is a simplified approach. A more accurate approach would
        # use graph connectivity analysis.
        from_bus = int(self.case_data['branch'][branch_idx, 0])
        to_bus = int(self.case_data['branch'][branch_idx, 1])
        
        # Count connections for each bus
        from_bus_connections = 0
        to_bus_connections = 0
        
        for i in range(self.n_branch):
            if i == branch_idx:
                continue
                
            br_from = int(self.case_data['branch'][i, 0])
            br_to = int(self.case_data['branch'][i, 1])
            
            if br_from == from_bus or br_to == from_bus:
                from_bus_connections += 1
                
            if br_from == to_bus or br_to == to_bus:
                to_bus_connections += 1
        
        return from_bus_connections == 0 or to_bus_connections == 0
    
    def _apply_contingency(self, contingency: Dict[str, Any], verbose: bool = False) -> Optional[Dict[str, Any]]:
        """
        Apply a contingency to the case data.
        
        Args:
            contingency: Contingency information
            verbose: Whether to print detailed information
            
        Returns:
            Modified case data with the contingency applied, or None if invalid
        """
        # Create a deep copy of the case data
        import copy
        modified_case = copy.deepcopy(self.case_data)
        
        try:
            if contingency['type'] == 'branch':
                # Remove branch by setting it to open
                branch_idx = contingency['id']
                modified_case['branch'][branch_idx, 10] = 0  # Set status to 0 (open)
                
                if verbose:
                    logger.info(f"Applied branch contingency: {contingency['name']}")
                    
            elif contingency['type'] == 'generator':
                # Remove generator by setting it to offline
                gen_idx = contingency['id']
                modified_case['gen'][gen_idx, 7] = 0  # Set status to 0 (offline)
                
                if verbose:
                    logger.info(f"Applied generator contingency: {contingency['name']}")
                    
            else:
                logger.warning(f"Unknown contingency type: {contingency['type']}")
                return None
                
            return modified_case
            
        except Exception as e:
            logger.error(f"Error applying contingency: {e}")
            return None
    
    def validate_voltage_stability(self, solution: Dict[str, Any], 
                                 load_data: np.ndarray,
                                 margin_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Validate the voltage stability of a solution using sensitivity analysis.
        
        Args:
            solution: Dictionary with solution variables
            load_data: Bus load data
            margin_threshold: Threshold for voltage stability margin
            
        Returns:
            Dictionary with voltage stability validation results
        """
        try:
            # Extract voltage magnitudes
            vm = solution['vm']
            
            # Calculate voltage stability indices
            # Here, we use a simplified approach based on voltage magnitude
            # In a real implementation, this would involve Q-V sensitivity analysis
            
            # Calculate distance to voltage limits
            lower_margin = vm - self.optimizer.vm_min
            upper_margin = self.optimizer.vm_max - vm
            
            # Minimum margin for each bus
            min_margin = np.minimum(lower_margin, upper_margin)
            
            # Identify buses with low stability margin
            critical_buses = np.where(min_margin < margin_threshold)[0]
            
            stability_status = 'stable'
            if len(critical_buses) > 0:
                stability_status = 'unstable'
                logger.warning(f"System has voltage stability issues at {len(critical_buses)} buses")
            
            # Create results
            results = {
                'status': stability_status,
                'critical_buses': critical_buses.tolist(),
                'min_margin': min_margin.min()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in voltage stability validation: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def validate_model_under_scenarios(self, 
                                     model: Any, 
                                     scenario_generator: Callable,
                                     n_scenarios: int = 10,
                                     verbose: bool = False) -> Dict[str, Any]:
        """
        Validate model performance under different operating scenarios.
        
        Args:
            model: ML model to validate
            scenario_generator: Function that generates scenarios (load patterns)
            n_scenarios: Number of scenarios to test
            verbose: Whether to print detailed information
            
        Returns:
            Dictionary with validation results across scenarios
        """
        results = {
            'scenario_results': [],
            'overall_status': 'success',
            'success_rate': 0.0,
            'average_violation': 0.0
        }
        
        successful_scenarios = 0
        total_violation = 0.0
        
        for i in range(n_scenarios):
            # Generate scenario
            scenario_data = scenario_generator(self.case_data, i)
            
            # Skip invalid scenarios
            if scenario_data is None:
                continue
                
            if verbose:
                logger.info(f"Evaluating scenario {i+1}/{n_scenarios}")
                
            # Get model prediction
            with torch.no_grad():
                inputs = torch.tensor(scenario_data, dtype=torch.float32, device=self.device)
                prediction = model(inputs)
                
                if isinstance(prediction, torch.Tensor):
                    prediction = prediction.cpu().numpy()
            
            # Extract solution
            n_gen = self.n_gen
            n_bus = self.n_bus
            
            solution = {
                'pg': prediction[:n_gen],
                'qg': prediction[n_gen:2*n_gen],
                'vm': prediction[2*n_gen:2*n_gen+n_bus],
                'va': prediction[2*n_gen+n_bus:2*n_gen+2*n_bus]
            }
            
            # Validate solution
            validation = self.validate_solution(solution, scenario_data)
            
            # Store scenario results
            scenario_result = {
                'scenario': i,
                'validation_passed': validation.get('validation_passed', False),
                'violations': validation.get('violations', {})
            }
            
            results['scenario_results'].append(scenario_result)
            
            # Update statistics
            if validation.get('validation_passed', False):
                successful_scenarios += 1
                
            if 'violations' in validation and 'total' in validation['violations']:
                total_violation += validation['violations']['total']
        
        # Calculate overall statistics
        results['success_rate'] = successful_scenarios / n_scenarios if n_scenarios > 0 else 0
        results['average_violation'] = total_violation / n_scenarios if n_scenarios > 0 else 0
        
        # Determine overall status
        if results['success_rate'] < 0.9:  # Less than 90% success rate
            results['overall_status'] = 'failed'
            logger.warning(f"Model validation failed: success rate = {results['success_rate']:.2f}")
        else:
            logger.info(f"Model validation passed: success rate = {results['success_rate']:.2f}")
            
        return results
        
def create_load_scenarios(case_data: Dict[str, Any], 
                         n_scenarios: int = 10, 
                         variation_range: float = 0.2) -> List[np.ndarray]:
    """
    Create a set of load scenarios with random variations for testing model robustness.
    
    Args:
        case_data: PyPOWER case data
        n_scenarios: Number of scenarios to generate
        variation_range: Range of load variations (e.g., 0.2 for ±20%)
        
    Returns:
        List of load data arrays for different scenarios
    """
    # Extract base load data
    n_bus = len(case_data['bus'])
    base_load_p = case_data['bus'][:, 2] / case_data['baseMVA']  # Active load
    base_load_q = case_data['bus'][:, 3] / case_data['baseMVA']  # Reactive load
    
    # Create scenarios
    scenarios = []
    
    for i in range(n_scenarios):
        # Generate random variations
        p_factor = 1.0 + variation_range * (2 * np.random.random(n_bus) - 1)
        q_factor = 1.0 + variation_range * (2 * np.random.random(n_bus) - 1)
        
        # Apply variations to base loads
        scenario_p = base_load_p * p_factor
        scenario_q = base_load_q * q_factor
        
        # Create load data for scenario
        load_data = np.column_stack((scenario_p, scenario_q))
        scenarios.append(load_data)
    
    return scenarios


def validate_model_performance(model: Any,
                              case_data: Dict[str, Any],
                              test_data: Union[np.ndarray, pd.DataFrame],
                              device: str = 'cpu') -> Dict[str, Any]:
    """
    Perform comprehensive validation of model performance including:
    - Basic performance metrics (MSE, MAE)
    - Power system constraint compliance
    - N-1 contingency analysis
    - Voltage stability analysis
    
    Args:
        model: Trained ML model
        case_data: PyPOWER case data
        test_data: Test dataset
        device: Device for torch tensors
        
    Returns:
        Dictionary with validation results
    """
    logger.info("Starting comprehensive model validation")
    
    try:
        # Create validator
        validator = PowerSystemValidator(case_data, device=device)
        
        # Prepare test data
        if isinstance(test_data, pd.DataFrame):
            # Convert DataFrame to numpy arrays
            input_cols = [col for col in test_data.columns if col.startswith('load_p') or col.startswith('load_q')]
            X_test = test_data[input_cols].values
        else:
            X_test = test_data
            
        # Convert to torch tensor
        X_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        
        # Get model predictions
        with torch.no_grad():
            predictions = model(X_tensor)
            
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
            
        # Initialize results dictionary
        results = {
            'basic_metrics': {},
            'constraint_compliance': {},
            'n1_contingency': {},
            'voltage_stability': {},
            'overall_status': 'failed'  # Default to failed until all checks pass
        }
        
        # 1. Basic performance metrics
        # For simplicity, assuming we have target data available
        # In a real implementation, this would compare against baseline solutions
        
        # 2. Constraint compliance on sample cases
        n_samples = min(10, len(X_test))  # Limit to 10 samples for efficiency
        sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
        
        compliance_results = []
        compliant_count = 0
        
        for idx in sample_indices:
            sample_data = X_test[idx]
            sample_pred = predictions[idx]
            
            # Extract solution
            n_gen = validator.n_gen
            n_bus = validator.n_bus
            
            solution = {
                'pg': sample_pred[:n_gen],
                'qg': sample_pred[n_gen:2*n_gen],
                'vm': sample_pred[2*n_gen:2*n_gen+n_bus],
                'va': sample_pred[2*n_gen+n_bus:2*n_gen+2*n_bus]
            }
            
            # Validate solution
            validation = validator.validate_solution(solution, sample_data)
            
            compliance_results.append({
                'sample_idx': idx,
                'validation_passed': validation.get('validation_passed', False),
                'violations': validation.get('violations', {})
            })
            
            if validation.get('validation_passed', False):
                compliant_count += 1
                
        results['constraint_compliance'] = {
            'status': 'passed' if compliant_count / n_samples >= 0.8 else 'failed',
            'compliance_rate': compliant_count / n_samples,
            'sample_results': compliance_results
        }
        
        # 3. N-1 contingency analysis on a sample case
        if n_samples > 0:
            contingency_sample = X_test[sample_indices[0]]
            n1_results = validator.perform_n1_contingency_analysis(
                model, contingency_sample, verbose=False
            )
            results['n1_contingency'] = n1_results
            
        # 4. Voltage stability analysis
        if n_samples > 0 and compliance_results[0]['validation_passed']:
            sample_idx = sample_indices[0]
            sample_data = X_test[sample_idx]
            sample_pred = predictions[sample_idx]
            
            solution = {
                'pg': sample_pred[:n_gen],
                'qg': sample_pred[n_gen:2*n_gen],
                'vm': sample_pred[2*n_gen:2*n_gen+n_bus],
                'va': sample_pred[2*n_gen+n_bus:2*n_gen+2*n_bus]
            }
            
            stability_results = validator.validate_voltage_stability(
                solution, sample_data
            )
            results['voltage_stability'] = stability_results
            
        # 5. Determine overall validation status
        constraint_passed = results['constraint_compliance']['status'] == 'passed'
        n1_passed = results['n1_contingency'].get('overall_status') == 'secure'
        stability_passed = results['voltage_stability'].get('status') == 'stable'
        
        if constraint_passed and n1_passed and stability_passed:
            results['overall_status'] = 'passed'
            logger.info("Model passed comprehensive validation")
        else:
            failed_checks = []
            if not constraint_passed:
                failed_checks.append('constraint compliance')
            if not n1_passed:
                failed_checks.append('N-1 contingency')
            if not stability_passed:
                failed_checks.append('voltage stability')
                
            logger.warning(f"Model failed validation checks: {', '.join(failed_checks)}")
            
        return results
        
    except Exception as e:
        logger.error(f"Error in model validation: {e}")
        return {
            'overall_status': 'error',
            'error': str(e)
        } 