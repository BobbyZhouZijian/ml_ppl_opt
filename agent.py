# collaborative optimization
from bayes_opt import custom_minimize, resume_optimize

class AgentBase:
    def __init__(self, random_seed=2023):
        self.punish = 0
        self.random_seed = random_seed
    
    def initialize_model(self, 
                    func,
                    space,
                    x0,
                    initial_point_generator="grid", 
                    acq_func="EI", 
                    n_calls=5, 
                    n_random_starts=5
        ):

        res, optimizer = custom_minimize(func,        # the function to minimize
                    space,      # the bounds on each dimension of x
                    initial_point_generator=initial_point_generator,
                    x0=x0,
                    acq_func=acq_func,      # the acquisition function
                    n_calls=n_calls,         # the number of evaluations of f
                    n_random_starts=n_random_starts,  # the number of random initialization points
                    noise="gaussian",       # the noise level (optional)
                    n_jobs=-1,
                    random_state=self.random_seed
                )   # the random seed
        return res, optimizer
    
    def update(self, func, n_points, res_resume, optimizer, fixed_dim=None):
        res_resume, optimizer = resume_optimize(n_points, func, optimizer, res_resume.specs, fixed_dim=fixed_dim)
        return res_resume, optimizer