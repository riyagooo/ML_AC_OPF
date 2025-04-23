import numpy as np
import torch
import gurobipy as gp
from gurobipy import GRB
from pypower.api import makeYbus, runopf
from pypower import idx_bus, idx_gen, idx_brch

class OPFOptimizer:
    """
    OPF Optimizer using Gurobi.
    """
    def __init__(self, case_data, device='cpu'):
        """
        Initialize OPF optimizer.
        
        Args:
            case_data: PyPOWER case data
            device: Device for torch tensors ('cpu' or 'cuda')
        """
        self.case_data = case_data
        self.device = device
        self.baseMVA = case_data['baseMVA']
        
        # Extract network parameters
        self._extract_parameters()
        
        # Initialize bounds
        self._initialize_bounds()
        
    def _extract_parameters(self):
        """Extract network parameters from case data."""
        # Get network dimensions
        self.n_bus = len(self.case_data['bus'])
        self.n_gen = len(self.case_data['gen'])
        self.n_branch = len(self.case_data['branch'])
        
        # Generator cost coefficients
        self.gen_cost = self.case_data['gencost'][:, 4:7]  # Quadratic, linear, constant
        self.gen_bus = self.case_data['gen'][:, idx_gen.GEN_BUS].astype(int)
        
        # Bus parameters
        self.bus_data = self.case_data['bus']
        
        # Branch parameters
        self.branch_data = self.case_data['branch']
        self.f_bus = self.branch_data[:, idx_brch.F_BUS].astype(int)
        self.t_bus = self.branch_data[:, idx_brch.T_BUS].astype(int)
        
        # Make Ybus matrix
        self.Ybus, self.Yf, self.Yt = makeYbus(self.baseMVA, self.bus_data, self.branch_data)
        
    def _initialize_bounds(self):
        """Initialize variable bounds from case data."""
        # Generator active power bounds
        self.pg_min = self.case_data['gen'][:, idx_gen.PMIN] / self.baseMVA
        self.pg_max = self.case_data['gen'][:, idx_gen.PMAX] / self.baseMVA
        
        # Generator reactive power bounds
        self.qg_min = self.case_data['gen'][:, idx_gen.QMIN] / self.baseMVA
        self.qg_max = self.case_data['gen'][:, idx_gen.QMAX] / self.baseMVA
        
        # Voltage magnitude bounds
        self.vm_min = self.case_data['bus'][:, idx_bus.VMIN]
        self.vm_max = self.case_data['bus'][:, idx_bus.VMAX]
        
        # Voltage angle bounds
        self.va_min = -np.pi
        self.va_max = np.pi
        
        # Branch flow limits
        self.branch_rate = self.case_data['branch'][:, idx_brch.RATE_A] / self.baseMVA
        
        # Create torch tensors for bounds
        self.bounds = {
            'pg_min': torch.tensor(self.pg_min, dtype=torch.float32, device=self.device),
            'pg_max': torch.tensor(self.pg_max, dtype=torch.float32, device=self.device),
            'qg_min': torch.tensor(self.qg_min, dtype=torch.float32, device=self.device),
            'qg_max': torch.tensor(self.qg_max, dtype=torch.float32, device=self.device),
            'vm_min': torch.tensor(self.vm_min, dtype=torch.float32, device=self.device),
            'vm_max': torch.tensor(self.vm_max, dtype=torch.float32, device=self.device),
            'va_min': torch.tensor(self.va_min, dtype=torch.float32, device=self.device),
            'va_max': torch.tensor(self.va_max, dtype=torch.float32, device=self.device),
            'branch_rate': torch.tensor(self.branch_rate, dtype=torch.float32, device=self.device)
        }
    
    def solve_opf(self, load_data, verbose=False):
        """
        Solve OPF problem using PyPOWER.
        
        Args:
            load_data: Bus load data (Pd, Qd)
            verbose: Whether to print solver output
        
        Returns:
            Dictionary with optimal solution
        """
        # Create a copy of the case data to modify
        case = self.case_data.copy()
        
        # Update load data
        for i, (pd, qd) in enumerate(load_data):
            case['bus'][i, idx_bus.PD] = pd * self.baseMVA
            case['bus'][i, idx_bus.QD] = qd * self.baseMVA
        
        # Run OPF
        result = runopf(case, verbose=verbose)
        
        # Check if successful
        if result['success']:
            # Extract solution
            solution = {
                'success': True,
                'f': result['f'],  # Objective value
                'pg': result['gen'][:, idx_gen.PG] / self.baseMVA,
                'qg': result['gen'][:, idx_gen.QG] / self.baseMVA,
                'vm': result['bus'][:, idx_bus.VM],
                'va': result['bus'][:, idx_bus.VA] * np.pi / 180,
                'branch_flow': result['branch'][:, idx_brch.PF] / self.baseMVA
            }
        else:
            solution = {'success': False}
        
        return solution
    
    def solve_opf_gurobi(self, load_data, warm_start=None, binding_constraints=None, verbose=False):
        """
        Solve OPF using Gurobi with optional warm-start and constraint screening.
        
        Args:
            load_data: Bus load data (Pd, Qd)
            warm_start: Optional warm-start values for decision variables
            binding_constraints: Optional binary array indicating which constraints are binding
            verbose: Whether to print solver output
        
        Returns:
            Dictionary with optimal solution
        """
        try:
            # Create a new Gurobi model
            model = gp.Model("OPF")
            
            if not verbose:
                model.setParam('OutputFlag', 0)
            
            # Set model parameters
            model.setParam('Method', 1)  # Dual simplex
            model.setParam('BarConvTol', 1e-6)
            model.setParam('FeasibilityTol', 1e-6)
            
            # Create decision variables
            pg = model.addVars(self.n_gen, lb=self.pg_min, ub=self.pg_max, name="pg")
            qg = model.addVars(self.n_gen, lb=self.qg_min, ub=self.qg_max, name="qg")
            vm = model.addVars(self.n_bus, lb=self.vm_min, ub=self.vm_max, name="vm")
            va = model.addVars(self.n_bus, lb=self.va_min, ub=self.va_max, name="va")
            
            # Set warm-start values if provided
            if warm_start is not None:
                for i in range(self.n_gen):
                    pg[i].start = warm_start['pg'][i]
                    qg[i].start = warm_start['qg'][i]
                for i in range(self.n_bus):
                    vm[i].start = warm_start['vm'][i]
                    va[i].start = warm_start['va'][i]
            
            # Set objective function (minimum generation cost)
            obj = gp.QuadExpr()
            for i in range(self.n_gen):
                a, b, c = self.gen_cost[i]
                obj += a * pg[i] * pg[i] + b * pg[i] + c
            
            model.setObjective(obj, GRB.MINIMIZE)
            
            # Add power balance constraints
            for i in range(self.n_bus):
                # Get bus load
                pd = load_data[i][0]
                qd = load_data[i][1]
                
                # Initialize bus generation
                p_gen = 0
                q_gen = 0
                
                # Add generation at this bus
                for g in range(self.n_gen):
                    if self.gen_bus[g] == i:
                        p_gen += pg[g]
                        q_gen += qg[g]
                
                # Add power balance constraints (simplified for now)
                model.addConstr(p_gen - pd == 0, f"active_balance_{i}")
                model.addConstr(q_gen - qd == 0, f"reactive_balance_{i}")
            
            # Add branch flow constraints (simplified for now)
            for i in range(self.n_branch):
                # Only add if this constraint is expected to be binding
                if binding_constraints is None or binding_constraints[i]:
                    from_bus = self.f_bus[i]
                    to_bus = self.t_bus[i]
                    rate = self.branch_rate[i]
                    
                    # Simplified branch flow calculation
                    # In a real implementation, this would use the actual branch equations
                    flow = (vm[from_bus] - vm[to_bus]) * 10  # Simplified for example
                    
                    if rate > 0:
                        model.addConstr(flow <= rate, f"branch_flow_{i}")
            
            # Optimize the model
            model.optimize()
            
            # Check solution status
            if model.status == GRB.OPTIMAL:
                # Extract solution
                pg_sol = np.array([pg[i].x for i in range(self.n_gen)])
                qg_sol = np.array([qg[i].x for i in range(self.n_gen)])
                vm_sol = np.array([vm[i].x for i in range(self.n_bus)])
                va_sol = np.array([va[i].x for i in range(self.n_bus)])
                
                solution = {
                    'success': True,
                    'f': model.objVal,
                    'pg': pg_sol,
                    'qg': qg_sol,
                    'vm': vm_sol,
                    'va': va_sol,
                    'runtime': model.Runtime
                }
            else:
                solution = {
                    'success': False,
                    'status': model.status,
                    'runtime': model.Runtime
                }
                
            return solution
            
        except Exception as e:
            print(f"Error solving OPF with Gurobi: {e}")
            return {'success': False, 'error': str(e)}
    
    def evaluate_solution(self, solution, load_data):
        """
        Evaluate a solution for constraint violations and cost.
        
        Args:
            solution: Dictionary with solution variables
            load_data: Bus load data (Pd, Qd)
        
        Returns:
            Dictionary with violation metrics
        """
        violations = {}
        
        # Power balance violations
        power_balance_violations = np.zeros(self.n_bus)
        
        # Simplified evaluation for example
        # In a real implementation, this would calculate actual violations
        
        # Calculate objective value (generation cost)
        cost = 0
        for i in range(self.n_gen):
            a, b, c = self.gen_cost[i]
            pg = solution['pg'][i]
            cost += a * pg * pg + b * pg + c
        
        violations['cost'] = cost
        violations['power_balance'] = np.sum(power_balance_violations)
        
        return violations 