"""
Optimization utility module for AC-OPF problems.

This module provides utilities for solving AC-OPF problems and evaluating solutions.
"""

import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

class OPFOptimizer:
    """
    Optimizer for AC-OPF problems.
    
    This is a simplified version created to support the PowerSystemValidator class.
    """
    
    def __init__(self, case_data, device='cpu'):
        """
        Initialize the optimizer.
        
        Args:
            case_data: Power system case data
            device: Device to use for computations
        """
        self.case_data = case_data
        self.device = device
        
        # Extract dimensions
        self.n_bus = len(case_data.get('bus', []))
        self.n_gen = len(case_data.get('gen', []))
        self.n_branch = len(case_data.get('branch', []))
        
        # Extract system base MVA
        self.baseMVA = case_data.get('baseMVA', 100.0)
        
        # Set default voltage limits if not available
        self.vm_min = np.ones(self.n_bus) * 0.94
        self.vm_max = np.ones(self.n_bus) * 1.06
        
        # Set thermal limits
        self.thermal_limits = np.ones(self.n_branch)
        if 'branch' in case_data and case_data['branch'].shape[1] > 5:
            # Real thermal limits from case data
            self.thermal_limits = case_data['branch'][:, 5] / self.baseMVA
        
        logger.info(f"Initialized OPF optimizer for system with {self.n_bus} buses, "
                   f"{self.n_gen} generators, and {self.n_branch} branches")
    
    def evaluate_solution(self, solution, load_data, thresholds=None):
        """
        Evaluate a solution for constraint violations.
        
        Args:
            solution: Dictionary with solution variables
            load_data: Load data
            thresholds: Violation thresholds
            
        Returns:
            Dictionary with violation metrics
        """
        if thresholds is None:
            thresholds = {
                'power_balance': 1e-3,
                'branch_thermal': 1e-3,
                'generator_limits': 1e-3,
                'voltage_limits': 1e-3
            }
        
        # This is a simplified evaluation that returns mock violations
        # In a real implementation, this would compute actual power flow violations
        
        # Extract solution components
        pg = solution.get('pg', np.zeros(self.n_gen))
        qg = solution.get('qg', np.zeros(self.n_gen))
        vm = solution.get('vm', np.ones(self.n_bus))
        va = solution.get('va', np.zeros(self.n_bus))
        
        # Initialize violations
        violations = {
            'power_balance': 0.0,
            'branch_thermal': 0.0,
            'generator_limits': 0.0,
            'voltage_limits': 0.0,
            'total': 0.0,
            'cost': 0.0
        }
        
        # Simulate power balance violations
        # In a real implementation, this would calculate actual power mismatches
        # using the power flow equations
        pv_indices = np.random.uniform(0, 0.1, self.n_bus)
        qv_indices = np.random.uniform(0, 0.1, self.n_bus)
        
        # Scale by a factor based on voltage deviations from nominal
        vm_deviation = np.abs(vm - 1.0)
        scale_factor = 1.0 + 2.0 * np.mean(vm_deviation)
        
        violations['power_balance'] = 0.05 * scale_factor * np.random.random()
        
        # Simulate branch thermal violations
        # Random violation level higher for solutions with large angle differences
        angle_diff_severity = np.mean(np.abs(np.diff(va)))
        branch_violation_level = 0.03 * (1.0 + 3.0 * angle_diff_severity) * np.random.random()
        violations['branch_thermal'] = branch_violation_level
        
        # Simulate generator limit violations
        gen_limit_violation = 0.02 * np.random.random()
        violations['generator_limits'] = gen_limit_violation
        
        # Calculate voltage limit violations
        v_min_violations = np.sum(np.maximum(0, self.vm_min - vm))
        v_max_violations = np.sum(np.maximum(0, vm - self.vm_max))
        violations['voltage_limits'] = (v_min_violations + v_max_violations) / self.n_bus
        
        # Calculate total violation
        violations['total'] = (
            violations['power_balance'] + 
            violations['branch_thermal'] + 
            violations['generator_limits'] + 
            violations['voltage_limits']
        )
        
        # Simulate generation cost
        cost_coeffs = np.linspace(10, 100, self.n_gen)
        violations['cost'] = np.sum(cost_coeffs * pg)
        
        return violations 