import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import norm
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from scipy.stats import multivariate_normal

from ParticleFilter import ParticleFilter, Model

class LotkaVolterra(Model):
    def __init__(cls, alpha, beta, gamma, delta, noise_sys = 0.1, noise_obs = 0.5):
        cls.alpha = alpha
        cls.beta  = beta
        cls.gamma = gamma
        cls.delta = delta
        
        cls.noise_sys = noise_sys
        cls.noise_obs = noise_obs

        cls.mu = [3,3]
        cls.cov = 0.5 * np.eye(2)

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

def plot(obs, history):

    # Plot simulated data and observations
    plt.plot(obs["timeexact"], obs["exact"][:,0], label="Prey")
    plt.plot(obs["timeexact"], obs["exact"][:,1], label="Predator")
    plt.plot(obs["timeobs"],   obs["data"], 'ro', label="Observations")

    # Compute sample mean, variance and 95%-confidence interval
    particles_means = dict()
    particles_means["preys"] = np.array([])

    for particles,weights in zip(history["particles"], history["weights"]):
        mean = np.average(particles[:,0], weights=weights)
        particles_means["preys"] = np.append(particles_means["preys"], mean)

    conf = np.array([0,0])
    for i in range(len(history["particles"])):
        # Compute confidence interval
        conf = np.vstack([conf, np.percentile(history["particles"][i][:,0], [2.5, 97.5])])

    conf = np.delete(conf, (0), axis=0)

    plt.plot(obs["timeobs"], particles_means["preys"], '-yo', label="Simulated preys")
    plt.fill_between(obs["timeobs"], conf[:,0], conf[:,1], color='y', alpha=0.3)

    plt.title("Evolution of predator and prey")
    plt.xlabel("time")
    plt.legend()
    
    plt.show()

model = LotkaVolterra(alpha = 7/3, beta = 1/3, gamma = 5/3, delta = 1)
n_obs = 20
artificial_observations = model.create_observations(n_obs, 0, 10)

N = 200
pf = ParticleFilter(N = N, 
                    model = model,
                    save_history = True,
                    resampling = "systematic"
                    )

# timesteps are equidistant, it's sufficient to compute dt once
dt = artificial_observations["timeobs"][1] - artificial_observations["timeobs"][0]

start = timer()
# First observation is not used
for i, (obsv, t) in enumerate(zip(artificial_observations["data"][1:], artificial_observations["timeobs"][1:])):
    pf.evolve(obsv, t, dt = dt)
end = timer()
print("Simulated " + str(N) + " particles using " + str(n_obs) + " observations in %.2f seconds." % (end - start))

plot(artificial_observations, pf.history)

