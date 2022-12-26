import copy
import inspect
import warnings
import numbers

from sklearn.utils import check_random_state
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import numpy as np
from optimizer import Optimizer
from skopt.callbacks import check_callback, VerboseCallback
from skopt.utils import eval_callbacks, normalize_dimensions, cook_estimator

def custom_minimize(func, dimensions,
                  n_calls=100, n_random_starts=None,
                  n_initial_points=10,
                  initial_point_generator="random",
                  acq_func="EI", acq_optimizer="lbfgs",
                  x0=None, y0=None, random_state=None, verbose=False,
                  callback=None, n_points=10000, n_restarts_optimizer=5,
                  xi=0.01, kappa=1.96, noise="gaussian", n_jobs=1, model_queue_size=None):

    # Check params
    rng = check_random_state(random_state)
    space = normalize_dimensions(dimensions)

    base_estimator = cook_estimator(
        "GP", space=space, random_state=rng.randint(0, np.iinfo(np.int32).max),
        noise=noise)

    specs = {"args": copy.copy(inspect.currentframe().f_locals),
             "function": inspect.currentframe().f_code.co_name}

    acq_optimizer_kwargs = {
        "n_points": n_points, "n_restarts_optimizer": n_restarts_optimizer,
        "n_jobs": n_jobs}
    acq_func_kwargs = {"xi": xi, "kappa": kappa}

    # Initialize optimization
    # Suppose there are points provided (x0 and y0), record them

    # check x0: list-like, requirement of minimal points
    if x0 is None:
        x0 = []
    elif not isinstance(x0[0], (list, tuple)):
        x0 = [x0]
    if not isinstance(x0, list):
        raise ValueError("`x0` should be a list, but got %s" % type(x0))

    # Check `n_random_starts` deprecation first
    if n_random_starts is not None:
        warnings.warn(("n_random_starts will be removed in favour of "
                       "n_initial_points. It overwrites n_initial_points."),
                      DeprecationWarning)
        n_initial_points = n_random_starts

    if n_initial_points <= 0 and not x0:
        raise ValueError("Either set `n_initial_points` > 0,"
                         " or provide `x0`")
    # check y0: list-like, requirement of maximal calls
    if isinstance(y0, Iterable):
        y0 = list(y0)
    elif isinstance(y0, numbers.Number):
        y0 = [y0]
    # required_calls = n_initial_points + (len(x0) if not y0 else 0)
    required_calls = n_initial_points
    if n_calls < required_calls:
        raise ValueError(
            "Expected `n_calls` >= %d, got %d" % (required_calls, n_calls))
    # calculate the total number of initial points
    n_initial_points = n_initial_points + len(x0)

    # Build optimizer

    # create optimizer class
    optimizer = Optimizer(dimensions, base_estimator,
                          n_initial_points=n_initial_points,
                          initial_point_generator=initial_point_generator,
                          n_jobs=n_jobs,
                          acq_func=acq_func, acq_optimizer=acq_optimizer,
                          random_state=random_state,
                          model_queue_size=model_queue_size,
                          acq_optimizer_kwargs=acq_optimizer_kwargs,
                          acq_func_kwargs=acq_func_kwargs)
    # check x0: element-wise data type, dimensionality
    assert all(isinstance(p, Iterable) for p in x0)
    if not all(len(p) == optimizer.space.n_dims for p in x0):
        raise RuntimeError("Optimization space (%s) and initial points in x0 "
                           "use inconsistent dimensions." % optimizer.space)
    # check callback
    callbacks = check_callback(callback)
    if verbose:
        callbacks.append(VerboseCallback(
            n_init=len(x0) if not y0 else 0,
            n_random=n_initial_points,
            n_total=n_calls))

    # Record provided points

    # create return object
    result = None
    # evaluate y0 if only x0 is provided
    if x0 and y0 is None:
        y0 = list(map(func, x0))
        n_calls -= len(y0)
    # record through tell function
    if x0:
        if not (isinstance(y0, Iterable) or isinstance(y0, numbers.Number)):
            raise ValueError(
                "`y0` should be an iterable or a scalar, got %s" % type(y0))
        if len(x0) != len(y0):
            raise ValueError("`x0` and `y0` should have the same length")
        result = optimizer.tell(x0, y0)
        result.specs = specs
        if eval_callbacks(callbacks, result):
            return result

    # Optimize
    for n in range(n_calls):
        next_x = optimizer.ask()
        next_y = func(next_x)
        result = optimizer.tell(next_x, next_y)
        result.specs = specs
        if eval_callbacks(callbacks, result):
            break

    return result, optimizer

def resume_optimize(n_calls, func, optimizer, specs, callback=None, fixed_dim=None):
    xs = []
    ys = []
    callbacks = check_callback(callback)
    for n in range(n_calls):
        next_x = optimizer.ask()
        if fixed_dim is not None:
            next_x[fixed_dim[0]] = fixed_dim[1]
        next_y = func(next_x)

        xs.append(next_x)
        ys.append(next_y)

        result = optimizer.tell(next_x, next_y, fixed_dim=fixed_dim)
        result.specs = specs
        if eval_callbacks(callbacks, result):
            break
        

    return result, optimizer, xs, ys