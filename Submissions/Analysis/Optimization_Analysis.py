import numpy as np
from scipy.optimize import differential_evolution
from pyH2A.Utilities.input_modification import convert_input_to_dictionary, parse_parameter, set_by_path
from pyH2A.Discounted_Cash_Flow import Discounted_Cash_Flow
import copy

class Optimization_Analysis:
    '''
    General-purpose optimization module for PyH2A.

    This class allows optimizing any parameter defined in the input file
    under specified bounds using a global optimization algorithm (differential evolution). 
    The objective is to minimize the Levelized Cost of Hydrogen (LCOH) 
    calculated through the Discounted Cash Flow (DCF) analysis.

    Attributes
    ----------
    inp_dict : dict
        Fully processed input dictionary extracted from the input file.
    base_case : Discounted_Cash_Flow
        Baseline DCF calculation used for reference.
    param_paths : list of list
        Paths to parameters in the input dictionary that will be optimized.
    param_types : list of str
        Type of each parameter ('value' or 'factor').
    bounds : list of tuple
        Lower and upper bounds for each parameter.
    optimal_values : np.ndarray
        Array storing the optimized parameter values after running optimization.
    optimal_h2_cost : float
        LCOH corresponding to the optimized parameters.
    '''

    def __init__(self, input_file):
        '''
        Initialize Optimization_Analysis object.

        Parameters
        ----------
        input_file : str
            Path to the pyH2A input file containing the plant model and Optimization_Analysis table.
        '''
        # Convert input file to a full dictionary for manipulation
        self.inp_dict = convert_input_to_dictionary(input_file)
        # Generate base case DCF for reference
        self.base_case = Discounted_Cash_Flow(input_file, print_info=False)

        # Lists to store optimization parameter information
        self.param_paths = []   # e.g., ['Photovoltaic', 'Nominal Power (kW)', 'Value']
        self.param_types = []   # 'value' or 'factor'
        self.bounds = []        # tuples (lower, upper)

        # Extract parameters from the 'Optimization_Analysis' section of the input
        for key, param_info in self.inp_dict['Optimization_Analysis'].items():
            self.param_paths.append(parse_parameter(key))
            self.param_types.append(param_info['Type'])
            self.bounds.append((param_info['Lower'], param_info['Upper']))

    def objective_function(self, x):
        '''
        Objective function to be minimized by the optimizer.

        Parameters
        ----------
        x : np.ndarray
            Array of trial parameter values suggested by the optimizer.

        Returns
        -------
        float
            LCOH corresponding to the current set of parameter values.
        '''
        # Create a deep copy of the input dictionary to avoid modifying the base case
        input_copy = copy.deepcopy(self.inp_dict)

        # Update parameters in the copied input dictionary
        for i, value in enumerate(x):
            set_by_path(input_copy, self.param_paths[i], value, value_type=self.param_types[i])

        # Run DCF analysis with the updated parameters
        dcf = Discounted_Cash_Flow(input_copy, print_info=False)

        # Return hydrogen cost as the objective to minimize
        return dcf.h2_cost

    def run_optimization(self, maxiter, popsize, seed=None):
        '''
        Execute the differential evolution optimization.

        Parameters
        ----------
        maxiter : int
            Maximum number of generations for the differential evolution algorithm.
        popsize : int
            Population size per generation.
        seed : int or None, optional
            Random seed for reproducibility.

        Returns
        -------
        result : OptimizeResult
            Output object from SciPy differential_evolution containing optimization results.
        '''
        print("Starting differential evolution optimization...")

        # Run SciPy's differential evolution algorithm
        result = differential_evolution(
            self.objective_function,
            self.bounds,
            maxiter=maxiter,
            popsize=popsize,
            seed=seed,
            polish=True,   # final local optimization after convergence
        )

        # Store optimized results
        self.optimal_values = result.x
        self.optimal_h2_cost = result.fun

        # Print summary of results
        print("\nOptimization complete!")
        print("Optimal H2 cost: {:.4f} $/kg".format(self.optimal_h2_cost))
        for i, val in enumerate(self.optimal_values):
            param_name = " > ".join(self.param_paths[i])
            print(f"Parameter '{param_name}' optimized to: {val}")

        return result
