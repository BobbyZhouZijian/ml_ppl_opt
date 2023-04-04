# collaborative optimization
import math

from bayes_opt import custom_minimize, resume_optimize


class AgentBase:
    def __init__(self, dim, delta=1.0, random_seed=2023):
        self.punish_count = 0
        self.dim = dim
        self.delta = delta
        self.random_seed = random_seed

    @property
    def need_punish(self):
        return self.punish_count > 0

    def punish(self, delta, r_f, r_star):
        if r_star <= 0:
            return 999999
        log = 1 - (1 - delta) * r_f / r_star
        if log <= 1e-9:
            return 9999999
        if log >= delta:
            return 0
        return math.ceil(math.log(log, delta) + 1e-9) - 1

    def initialize_model(self,
                         func,
                         space,
                         x0,
                         initial_point_generator="grid",
                         acq_func="EI",
                         n_random_starts=5
                         ):
        res, optimizer = custom_minimize(func,  # the function to minimize
                                         space,  # the bounds on each dimension of x
                                         initial_point_generator=initial_point_generator,
                                         x0=x0,
                                         acq_func=acq_func,  # the acquisition function
                                         n_calls=n_random_starts,  # the number of evaluations of f
                                         n_random_starts=n_random_starts,  # the number of random initialization points
                                         noise="gaussian",  # the noise level (optional)
                                         n_jobs=-1,
                                         random_state=self.random_seed
                                         )  # the random seed
        return res, optimizer

    def update(self, func, n_points, res_resume, optimizer, other_dim, other_val):
        fixed_dim = (other_dim, other_val)
        res_resume, optimizer, xs, ys = resume_optimize(n_points, func, optimizer, res_resume.specs,
                                                        fixed_dim=fixed_dim)
        return res_resume, optimizer, xs, ys

    def has_improvement(self, next_res=None, prev_res=None):
        if next_res is None or prev_res is None:
            return True
        # assuming minimization
        return prev_res.fun - next_res.fun > self.delta
