"""
Utilities: Includes utility and helper functions to support the DAESIM model code structure and implementation
"""

import inspect
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Callable, List, Optional, Tuple, Dict
from scipy.integrate import solve_ivp, OdeSolution
import numpy as np

def array_like_wrapper(func, array_like_args):
    """
    Wrap a function to handle array-like inputs for specified arguments that could be array-like.

    This wrapper allows a function designed for scalar inputs to handle array-like inputs
    by automatically iterating over these inputs and calling the original function for each element.
    The results are aggregated and returned as a numpy array.

    Parameters
    ----------
    func (callable): The original function to wrap. This function should accept scalar inputs.
    array_like_args (list of str): List of argument names that could be array-like. The wrapper will check
                                   these arguments and iterate over them if they are array-like.

    Returns
    -------
    callable: A wrapped function that can handle both scalar and array-like inputs for the specified arguments.

    Raises
    ------
    ValueError: If the array-like arguments have different lengths.

    Example
    -------
    >>> def canopy_absorption(LAI, SW, nlayers=4):
    >>>     dlai = LAI * np.ones(nlayers) / nlayers
    >>>     sunlit_frac = np.linspace(1, 0, nlayers)
    >>>     swlayer = np.zeros(nlayers)
    >>>     for ic in range(nlayers):
    >>>         swlayer[ic] = dlai[ic] * sunlit_frac[ic] * SW
    >>>     return swlayer
    >>>
    >>> wrapped_func = array_like_wrapper(canopy_absorption, ['LAI'])
    >>> LAI = [2.0, 3.0]
    >>> SW = 100
    >>> wrapped_func(LAI, SW)
    array([[ 50.,  37.5,  25.,  12.5],
           [ 75.,  56.25, 37.5,  18.75]])

    """
    def wrapper(*args, **kwargs):
        # Get the original function's argument names and default values
        sig = inspect.signature(func)
        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()

        # Determine which arguments are actually array-like
        actual_array_like_args = []
        for arg_name in array_like_args:
            if arg_name in bound_args.arguments:
                if isinstance(bound_args.arguments[arg_name], (list, np.ndarray)):
                    actual_array_like_args.append(arg_name)
                    bound_args.arguments[arg_name] = np.atleast_1d(bound_args.arguments[arg_name])

        # Determine the length of the array-like arguments
        length = None
        for arg_name in actual_array_like_args:
            if length is None:
                length = len(bound_args.arguments[arg_name])
            else:
                if length != len(bound_args.arguments[arg_name]):
                    raise ValueError(f"Array-like arguments must have the same length, but '{arg_name}' has length {len(bound_args.arguments[arg_name])} while others have length {length}")

        # Handle the case when no array-like arguments are provided
        if length is None:
            length = 1

        # Call the original function for each element in the array-like arguments
        results = []
        for i in range(length):
            single_call_args = {
                arg_name: bound_args.arguments[arg_name][i] if arg_name in actual_array_like_args else bound_args.arguments[arg_name]
                for arg_name in bound_args.arguments
            }
            result = func(**single_call_args)
            results.append(result)
        
        return np.array(results)
    
    return wrapper


@dataclass
class ODEModelSolver:
    """
    Numerical solver for ordinary differential equations.
    """

    calculator: Callable        # Generic callable (e.g. GenericClass.calculator method)
    states_init: List[float]    # List of initial state values
    time_start: float
    log_diagnostics: bool = False  # New flag to control diagnostics logging
    diagnostics: Dict[str, List[Any]] = field(default_factory=lambda: defaultdict(list))  # To store diagnostic data

    def run(
        self,
        time_axis: List[float],
        forcing_inputs: List[Callable[[float], float]],  # Forcing inputs passed here (must be in correct order for calculator Callable
        solver: str = "ivp",
        zero_crossing_indices: Optional[List[int]] = None,
        reset_days: Optional[List[int]] = None,
        rtol: float = 1e-3,
        atol: float = 1e-3,
    ) -> OdeSolution:
        """
        Runs the solver to solve the differential equations with zero-crossing event handling.

        Parameters
        ----------
        - time_axis: List[float], sequence of time points at which to solve the equations.
        - forcing_inputs: List[Callable[[float], float]], list of Callable forcing data to provide forcing values at any time step
        - solver: str, the solver method to use ("ivp" or "euler").
        - zero_crossing_indices: List[int], indices of state variables to monitor for zero-crossing events.
        - reset_days: List[int], list of days-of-year at which specified state should be reset to zero.
        - rtol: float, relative tolerance for the solver (only for "ivp").
        - atol: float, absolute tolerance for the solver (only for "ivp").

        Returns
        -------
        - dict, the result of the integration.
        """
        self._event_triggered = {day: False for day in reset_days}  # Track if an event has been triggered for each day

        func_to_solve = self._construct_ode_function(forcing_inputs)

        if solver == "ivp":
            return self._run_with_events_ivp(func_to_solve, time_axis, zero_crossing_indices, reset_days, rtol, atol)
        elif solver == "euler":
            return self._run_with_events_euler(func_to_solve, time_axis, zero_crossing_indices, reset_days)
        else:
            raise ValueError(f"Unknown solver: {solver}")

    def _construct_ode_function(
        self,
        forcing_inputs: List[Callable[[float], float]],
    ) -> Callable[[float, np.ndarray], np.ndarray]:
        """
        Constructs the differential equation function for the solver.

        Parameters
        ----------
        - forcing_inputs: List[Callable[[float], float]], list of Callable forcing data to provide forcing values at any time step

        Returns
        -------
        - Callable[[float, np.ndarray], np.ndarray], the differential equation function.

        Notes
        -----
        The input argument list (inputargs) must be in the correct order as expected by the Callable calculator 
        """
        def func_to_solve(t: float, y: np.ndarray) -> np.ndarray:
            forcing_values = [forcing_input(t) for forcing_input in forcing_inputs]
            inputargs = y.tolist() + forcing_values
            # Separate dydt and diagnostics, assuming that diagnostics are always the last element returned
            *dydt, diagnostics = self.calculator(*inputargs, return_diagnostics=self.log_diagnostics)  # Unpack all but the last element into dydt, and the last into diagnostics

            # Log diagnostic outputs if they are available
            if diagnostics is not None:
                # Add time to the diagnostic outputs dictionary for context
                diagnostics_entry = {'t': t, **diagnostics}

                # Append time to the 'time' key
                self.diagnostics['t'].append(t)

                # Append values for each key in the diagnostics
                for key, value in diagnostics.items():
                    self.diagnostics[key].append(value)

            return np.array(dydt)

        return func_to_solve

    def _bio_time_reset_event(self, reset_days: List[int]) -> List[Callable]:
        """
        Event functions to reset state variables when the day-of-year matches any in reset_days.
        """
        events = []
        for reset_day in reset_days:
            def event(t, y, reset_day=reset_day):
                _doy = int(t % 365)  # Assuming t is in days and you want day-of-year. TODO: Need to double check definitions of t, _doy and _doy forcing.
                if self._event_triggered[reset_day]:
                    return 1    # Ensure event does not trigger again for this day
                return _doy - reset_day

            event.terminal = True
            event.direction = 0
            events.append(event)
        return events

    def _run_with_events_ivp(
        self, func_to_solve: Callable[[float, np.ndarray], np.ndarray], 
        time_axis: List[float], zero_crossing_indices: Optional[List[int]], reset_days: Optional[List[int]], rtol: float, atol: float
    ) -> dict:
        """
        Runs the IVP solver with zero-crossing event handling.

        Parameters:
        - func_to_solve: Callable[[float, np.ndarray], np.ndarray], the differential equation function.
        - time_axis: List[float], sequence of time points at which to solve the equations.
        - zero_crossing_indices: List[int], indices of state variables to monitor for zero-crossing events.
        - reset_days: List[int], list of days-of-year at which specified state should be reset to zero.
        - rtol: float, relative tolerance for the solver.
        - atol: float, absolute tolerance for the solver.

        Returns:
        - dict, the result of the integration.
        """
        t_eval = time_axis
        t_span = (self.time_start, t_eval[-1])
        start_state = self.states_init

        events = None
        if zero_crossing_indices and reset_days:
            events = self._bio_time_reset_event(reset_days)

        solve_kwargs = {
            "t_span": t_span,
            "t_eval": t_eval,
            "y0": start_state,
            "events": events,
            "rtol": rtol,
            "atol": atol,
            "dense_output": True,
        }

        res_raw = solve_ivp(func_to_solve, **solve_kwargs)

        # Handle zero-crossing event
        while res_raw.status == 1:
            # Find the most recent event time and handle it
            t_events = res_raw.t_events
            event_times = [t_event[0] for t_event in t_events if len(t_event) > 0]
            if not event_times:
                break
            t_restart = max(event_times)
            _doy_event = int(t_restart % 365)
            if _doy_event in self._event_triggered:
                self._event_triggered[_doy_event] = True
            t_eval = time_axis[time_axis >= t_restart]
            t_span = (t_restart, t_eval[-1])
            start_state_restart = res_raw.sol(t_restart)
            if zero_crossing_indices:
                for idx in zero_crossing_indices:
                    start_state_restart[idx] = 0  # Reset the specified state variables to zero

            solve_kwargs = {
                "t_span": t_span,
                "t_eval": t_eval,
                "y0": tuple(start_state_restart),
                "events": events,
                "rtol": rtol,
                "atol": atol,
                "dense_output": True,
            }

            res_next = solve_ivp(func_to_solve, **solve_kwargs)
            res_raw.t = np.append(res_raw.t, res_next.t)
            res_raw.y = np.append(res_raw.y, res_next.y, axis=1)
            if res_next.status == 1:
                res_raw.t_events = res_raw.t_events + res_next.t_events
                res_raw.y_events = res_raw.y_events + res_next.y_events
            res_raw.nfev += res_next.nfev
            res_raw.njev += res_next.njev
            res_raw.nlu += res_next.nlu
            res_raw.sol = res_next.sol
            res_raw.message = res_next.message
            res_raw.success = res_next.success
            res_raw.status = res_next.status

        return dict(res_raw)

    def _run_with_events_euler(
        self, func_to_solve: Callable[[float, np.ndarray], np.ndarray], time_axis: List[float],
        zero_crossing_indices: Optional[List[int]], reset_days: Optional[List[int]]
    ) -> dict:
        """
        Runs the Euler solver with zero-crossing event handling.

        Parameters:
        - func_to_solve: Callable[[float, np.ndarray], np.ndarray], the differential equation function.
        - time_axis: List[float], sequence of time points at which to solve the equations.
        - zero_crossing_indices: List[int], indices of state variables to monitor for zero-crossing events.
        - reset_days: List[int], list of days-of-year at which specified state should be reset to zero.

        Returns:
        - dict, the result of the integration.
        """
        dt = time_axis[1] - time_axis[0]    # assume that the time step is defined by the increments in time_axis
        y = np.zeros((len(time_axis), len(self.states_init)))
        y[0] = self.states_init

        t_events = []
        y_events = []

        for i in range(1, len(time_axis)):
            t = time_axis[i - 1]
            y[i] = y[i - 1] + dt * func_to_solve(t, y[i - 1])

            # Check for zero-crossing events 
            if zero_crossing_indices and reset_days:
                for idx in zero_crossing_indices:
                    _doy = int(t % 365)
                    if _doy in reset_days:
                        if not self._event_triggered[_doy]:
                            t_events.append(t)
                            y_events.append(y[i].copy())
                        y[i][idx] = 0  # Reset the specified state variable to zero
        # Set the event triggered flag after processing all indices
        if _doy in reset_days:
            self._event_triggered[_doy] = True

        result = {
            "t": time_axis,
            "y": y.T,
            "t_events": [np.array(t_events)],
            "y_events": [np.array(y_events)],
            "nfev": len(time_axis),
            "status": 0,
            "success": True,
            "message": "Integration successful." if not t_events else "Zero-crossing event occurred.",
        }

        return result

    def reset_diagnostics(self):
        """Resets the diagnostics dictionary to its default empty state."""
        self.diagnostics = defaultdict(list)


class SolveError(Exception):
    """Custom exception for solver errors."""
    pass
