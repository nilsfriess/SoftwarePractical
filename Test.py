from ParticleFilter import ParticleFilter, Model
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import norm
import matplotlib.pyplot as plt
from timeit import default_timer as timer

class LotkaVolterra(Model):
    def __init__(cls, alpha, beta, gamma, delta, noise_sys = 0.1, noise_obs = 0.5):
        cls.alpha = alpha
        cls.beta  = beta
        cls.gamma = gamma
        cls.delta = delta
        
        cls.noise_sys = noise_sys
        cls.noise_obs = noise_obs

    def dxdt(cls, t, x):
        """ Returns the RHS of the ODE modelling the preys (x[0]) and the predators x[1] """
        return np.array([ 
            cls.alpha * x[0] - cls.beta * x[0] * x[1],
            cls.delta * x[0] * x[1] - cls.gamma * x[1]
        ])

    def sample_prior(cls, n):
        """ Returns n samples of a 2D N(5,0.5) distribution, the prior on the predator and prey values """
        mu = [3,3]
        cov = 0.5 * np.eye(2)
        return np.random.multivariate_normal(mu, cov, n)

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
    particles_means["predators"] = np.array([])

    preys_std = np.array([])

    for particles,weights in zip(history["particles"], history["weights"]):
        mean = np.average(particles[:,0], weights=weights)
        particles_means["preys"] = np.append(particles_means["preys"], mean)

        preys_std = np.sqrt(np.append(preys_std,  np.average((particles[:,0] - mean)**2)))
        # particles_means["predators"] = np.append(particles_means["predators"], np.average(particles[:,1], weights=weights))
    
    conf_upper = particles_means["preys"] + 1.96*preys_std/np.sqrt(len(history["particles"][0]))
    conf_lower = particles_means["preys"] - 1.96*preys_std/np.sqrt(len(history["particles"][0]))

    plt.plot(obs["timeobs"], particles_means["preys"], '-yo', label="Simulated preys")
    plt.fill_between(obs["timeobs"], conf_upper, conf_lower, color='y', alpha=0.3)

    # plt.plot(obs["timeobs"],   particles_means["predators"], '-go', label="Simulated predators")

    plt.title("Evolution of predator and prey")
    plt.xlabel("time")
    plt.legend()

    # Phase plot
    # plt.subplot(212)
    # plt.plot(observations["exact"][:,0], observations["exact"][:,1])
    # plt.title("Phase plot")
    # plt.xlabel("Prey")
    # plt.ylabel("Predator")

    # # plot particles inside phase plot
    # for particles in history["particles"]:
    #     plt.scatter(particles[:,0], particles[:,1], c = np.random.random((1,3)))
    
    plt.show()

         
# setup model and create synthetic observations
model = LotkaVolterra(alpha = 5/3, beta = 1/3, gamma = 5/3, delta = 1)
n_obs = 20
observations = model.create_observations(n_obs, 0, 10)

N = 400
pf = ParticleFilter(N = N, 
                    model = model,
                    save_history = True,
                    resampling = "stratified"
                    )

# timesteps are all equal, it's sufficient to compute dt once
dt = observations["timeobs"][1] - observations["timeobs"][0]
# First observation is not used
start = timer()
for i, (obsv, t) in enumerate(zip(observations["data"][1:], observations["timeobs"][1:])):
    pf.evolve(obsv, t, dt = dt)
end = timer()
print("Simulated " + str(N) + " particles using " + str(n_obs) + " observations in %.2f seconds." % (end - start))
    
plot(observations, pf.history)






