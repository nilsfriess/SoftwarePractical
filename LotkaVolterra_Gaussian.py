import numpy as np
from scipy.stats import multivariate_normal

from LotkaVolterra_Bootstrap import LotkaVolterra_Bootstrap

class LotkaVolterra_Gaussian(LotkaVolterra_Bootstrap):

    def is_bootstrap(cls):
        return False

    def sample_proposal(cls, x, t, dt):
        return np.random.multivariate_normal(x, 0.8*np.eye(2))

    def proposal_density(cls, x_before, x, t):
        gaussian = multivariate_normal(x_before, cls.noise_sys * np.eye(2))
        return gaussian.pdf(x)
                

    def transition_density(cls, x_before, x, t, dt):
        """ x = particle, y = observation """
        # Evolve particle
        evolved = cls.evolve(x_before,[t, t + dt])
        gaussian = multivariate_normal(evolved, cls.noise_obs * np.eye(2))
        return gaussian.pdf(x)
