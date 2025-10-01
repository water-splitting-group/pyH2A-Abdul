import numpy as np
from scipy.optimize import differential_evolution
from pyH2A.Utilities.input_modification import (
    convert_input_to_dictionary,
    parse_parameter,
    set_by_path
)
from pyH2A.Discounted_Cash_Flow import Discounted_Cash_Flow
import copy


class Optimization_Analysis:
    """
    General-purpose optimization module for PyH2A.

    Optimizes parameters defined in the input file using differential evolution
    to minimize the Levelized Cost of Hydrogen (LCOH).
    """

    def __init__(self, input_file, cache_tolerance=6):
        """
        Initialize OptimizationAnalysis.

        Parameters
        ----------
        input_file : str
            Path to the pyH2A input file containing plant model and Optimization_Analysis table.
        cache_tolerance : int, optional
            Decimal rounding for caching parameter sets (default: 6).
        """
        # Convert input file to dictionary
        self.inp_dict = convert_input_to_dictionary(input_file)

        # Validate optimization section
        if 'Optimization_Analysis' not in self.inp_dict:
            raise ValueError("Input file missing 'Optimization_Analysis' section.")

        # Generate base case DCF for reference
        self.base_case = Discounted_Cash_Flow(input_file, print_info=False)

        # Extract parameters
        self.param_key_paths = []
        self.param_types = []
        self.bounds = []

        for key, param_info in self.inp_dict['Optimization_Analysis'].items():
            self.param_key_paths.append(parse_parameter(key))
            self.param_types.append(param_info['Type'])
            lower, upper = param_info['Lower'], param_info['Upper']
            if lower >= upper:
                raise ValueError(f"Invalid bounds for parameter '{key}': lower >= upper")
            self.bounds.append((lower, upper))

        # Results
        self.optimal_values = None
        self.optimal_h2_cost = None

        # Cache
        self._cache = {}
        self._cache_tolerance = cache_tolerance

    def _make_cache_key(self, x):
        """Create a rounded tuple key for caching."""
        return tuple(np.round(x, self._cache_tolerance))

    def objective_function(self, x, verbose=False):
        """
        Objective function to minimize.

        Parameters
        ----------
        x : np.ndarray
            Trial parameter values.
        verbose : bool
            If True, prints trial results.

        Returns
        -------
        float
            LCOH for current parameter set.
        """
        # Check cache
        key = self._make_cache_key(x)
        if key in self._cache:
            return self._cache[key]

        # Copy base input
        input_copy = copy.deepcopy(self.inp_dict)

        # Apply parameter updates
        for i, value in enumerate(x):
            set_by_path(input_copy, self.param_key_paths[i], value,
                        value_type=self.param_types[i])

        # Run DCF and catch failures
        try:
            dcf = Discounted_Cash_Flow(input_copy, print_info=False)
            cost = dcf.h2_cost
        except Exception as e:
            # Penalize invalid solutions
            cost = float("inf")
            if verbose:
                print(f"DCF failed with error: {e}")

        # Store in cache
        self._cache[key] = cost

        # Optional verbose logging
        if verbose:
            param_names = [" > ".join(p) for p in self.param_key_paths]
            trial_info = ", ".join(f"{name}: {val:.4f}"
                                   for name, val in zip(param_names, x))
            print(f"Trial params: {trial_info} => LCOH: {cost:.4f}")

        return cost

    def run_optimization(self, maxiter, popsize, seed,
                         verbose, workers):
        """
        Run differential evolution optimization.

        Parameters
        ----------
        maxiter : int
            Maximum number of generations (default: 100).
        popsize : int
            Population size per generation (default: 15).
        seed : int or None
            Random seed for reproducibility (default: None).
        verbose : bool
            If True, prints optimization progress and results.
        workers : int
            Number of parallel workers (default: -1 = use all cores).

        Returns
        -------
        dict
            Results dictionary containing:
            - optimal_values : np.ndarray
            - optimal_h2_cost : float
            - scipy_result : OptimizeResult
        """
        if verbose:
            print("Starting differential evolution optimization...")

        # Run SciPy optimizer with parallel execution
        result = differential_evolution(
            self.objective_function,
            self.bounds,
            maxiter=maxiter,
            popsize=popsize,
            seed=seed,
            polish=True,
            workers=workers,
            updating="deferred"
        )

        # Store results
        self.optimal_values = result.x
        self.optimal_h2_cost = result.fun

        if verbose:
            print("\nOptimization complete!")
            print(f"Optimal H2 cost: {self.optimal_h2_cost:.4f} $/kg")
            for i, val in enumerate(self.optimal_values):
                param_name = " > ".join(self.param_key_paths[i])
                print(f"Parameter '{param_name}' optimized to: {val:.4f}")

        return {
            "optimal_values": self.optimal_values,
            "optimal_h2_cost": self.optimal_h2_cost,
            "scipy_result": result
        }
