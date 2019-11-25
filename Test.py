import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import norm
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from scipy.stats import multivariate_normal

from ParticleFilter import ParticleFilter, Model

from LotkaVolterra_Bootstrap import LotkaVolterra_Bootstrap
from LotkaVolterra_Gaussian import LotkaVolterra_Gaussian 

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

model = LotkaVolterra_Gaussian(alpha = 7/3, beta = 1/3, gamma = 5/3, delta = 1)
n_obs = 20
artificial_observations = model.create_observations(n_obs, 0, 10)

N = 200
pf = ParticleFilter(N = N, 
                    model = model,
                    save_history = True,
                    resampling = "stratified"
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

