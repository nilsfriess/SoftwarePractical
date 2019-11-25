import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import norm
from scipy.stats import multivariate_normal

from ParticleFilter import ParticleFilter, Model

class LotkaVolterra_Bootstrap(Model):
    def __init__(cls, alpha, beta, gamma, delta, noise_sys = 0.1, noise_obs = 0.5):
        cls.alpha = alpha
        cls.beta  = beta
        cls.gamma = gamma
        cls.delta = delta
        
        cls.noise_sys = noise_sys
        cls.noise_obs = noise_obs

        cls.mu = [3,3]
        cls.cov = 0.5 * np.eye(2)

    def is_bootstrap(cls):
        return True

    def dxdt(cls, t, x):
        """ Returns the RHS of the ODE modelling the preys (x[0]) and the predators x[1] """
        return np.array([ 
            cls.alpha * x[0] - cls.beta * x[0] * x[1],
            cls.delta * x[0] * x[1] - cls.gamma * x[1]
        ])

    def sample_prior(cls, n):
        """ Returns n samples of a 2D N(5,0.5) distribution, the prior on the predator and prey values """
        return np.random.multivariate_normal(cls.mu, cls.cov, n)

    def pdf_prior(cls, x, y):
        rv = multivariate_normal(cls.mu, cls.cov)
        return rv.pdf(x,y)

    def evolve(cls, x, t):
        """ Solve the lotka volterra equations for one time step """
        sol = solve_ivp(cls.dxdt, t_span = t, y0 = x)
        return np.array([sol.y[0][-1], sol.y[1][-1]])
        
    def sample_transition(cls, x, t, dt = 1):
        """  Samples from a 2D Gaussian N(M(x), noise_sys) """
        mu = cls.evolve(x,[t, t + dt])
        cov = cls.noise_sys * np.eye(2)
        return np.random.multivariate_normal(mu, cov)
        
    def observation_density(cls, x, y, t):
        """ Compute Gaussian pdf N(y | x[1], noise_obs) """
        return norm.pdf(y, loc = x[1], scale = cls.noise_obs)

    def create_observations(cls, n, t_start, t_end):
        """ Create n observations between t_start and t_end """

        # Simulate at 8*n timesteps to get smoother results
        times_obs = 8
        
        y = np.empty((n*times_obs,2))
        y[0] = cls.sample_prior(1)

        # create time steps by dividing the interval into times_obs*n parts
        timesteps = np.linspace(t_start, t_end, n*times_obs)
        for i in range(1, len(timesteps)):
            sol = solve_ivp(cls.dxdt, t_span = [timesteps[i-1], timesteps[i]], y0 = y[i-1])
            y[i] = np.array([sol.y[0][-1], sol.y[1][-1]])

        # extract observations from solution by taking one in times_obs values of the predator and perturbing it by additive gaussian noise
        obs = np.random.normal(y[:,1][::times_obs], cls.noise_obs)

        observations = dict()
        observations["exact"] = y                      
        observations["timeexact"] = timesteps
        observations["data"] = obs
        observations["timeobs"] = timesteps[::times_obs]

        return observations
