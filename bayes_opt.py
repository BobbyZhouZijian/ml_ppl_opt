import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

class BayesOpt:
    def __init__(self, func, bounds, x_init, model, noise = 0.2):
        self.func = func
        self.bounds = bounds
        self.x_init = x_init
        self.gpr = model
        self.noise = noise
    
    def run(self, n_iters, fixed_dim=None):
        X_sample = self.x_init
        Y_sample = [self.func(x) for x in X_sample]
        for _ in range(n_iters):
            # Update Gaussian process with existing samples
            self.gpr.fit(X_sample, Y_sample)

            # Obtain next sampling point from the acquisition function (expected_improvement)
            X_next = propose_location(expected_improvement, X_sample, Y_sample, self.gpr, self.bounds, fixed_dim)
            
            # Obtain next noisy sample from the objective function
            Y_next = self.func(X_next, self.noise)
            
            # Add sample to previous samples
            X_sample = np.vstack((X_sample, X_next))
            Y_sample = np.vstack((Y_sample, Y_next))
        return X_sample, Y_sample

def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, 1)
    
    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [1]
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei

def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25, fixed_dim=None):
    '''
    Proposes the next sampling point by optimizing the acquisition function.
    
    Args:
        acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.

    Returns:
        Location of the acquisition function maximum.
    '''
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None

    def transform_input(x):
        if fixed_dim is None:
            return x
        elif fixed_dim[0] == 0:
            return np.hstack((np.ones((x.shape[0],1)) * fixed_dim[1], x))
        elif fixed_dim[0] == 1:
            return np.hstack((x, np.ones((x.shape[0],1)) * fixed_dim[1]))
        else:
            raise NotImplementedError("fixed dim more than 2 not implemented")
    
    if fixed_dim is None:
        pass
    elif fixed_dim[0] == 0:
        bounds = bounds[1:, :]
        dim -= 1
    elif fixed_dim[0] == 1:
        bounds = bounds[:1, :]
        dim -= 1
    else:
        raise NotImplementedError("fixed dim more than 2 not implemented")

    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(transform_input(X.reshape(-1, dim)), X_sample, Y_sample, gpr)
    
    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')       
        if res.fun < min_val:
            min_val = res.fun
            min_x = res.x           
            
    x_out = min_x.reshape(-1, 1)
    return transform_input(x_out)