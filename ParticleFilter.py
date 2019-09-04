import numpy as np
from abc import ABC, abstractmethod

class ParticleFilter(object):
    def __init__(
        cls,            
        model,          
        N,              # Number of particles
        resampling = "systematic",  # Possible options: "systematic", "stratified"
        save_history = False        # Preserve particles and weights?
    ):
        cls.model           = model
        cls.N               = N
        cls.resampling      = resampling
        cls.save_history    = save_history

        if not (cls.resampling == "systematic" or cls.resampling == "stratified"):
            raise Exception("Resampling strategy not implemented")

        cls.create_particles()

    def create_particles(cls):
        """ Creates the initial set of particles by drawing from the prior defined by the model """
        cls.particles = cls.model.sample_prior(cls.N)
        if type(cls.particles) is not np.ndarray:
            # make sure not to create np.array of np.array
            cls.particles = np.array(cls.particles)
        cls.weights   = np.full(cls.N, 1 / cls.N)

        if cls.save_history:
            cls.history = dict()
            cls.history["particles"] = np.array([cls.particles])
            cls.history["weights"] = np.array([cls.weights])

    def evolve(cls, obs, t, dt = 1):
        """ Evolve all particles using the observation obs and time t """

        # evolve particles
        cls.particles = np.array([
            cls.model.sample_transition(p, t, dt) for p in cls.particles
        ])

        # update weights
        cls.weights = np.array([
            weight * cls.model.observation_density(p, obs, t) 
            for weight, p in zip(cls.weights, cls.particles)
        ])

        # normalise weights
        cls.weights /= cls.weights.sum()
        

        if cls.resampling_necessary():
            cls.resample()

        if cls.save_history:
            cls.history["particles"] = np.vstack([
                cls.history["particles"],
                [cls.particles]
            ])
            cls.history["weights"] = np.vstack([
                cls.history["weights"],
                [cls.weights]
            ])

    def resampling_necessary(cls):
        effective_sampling_size = 1 / np.square(cls.weights).sum()
        return effective_sampling_size < (0.5 * cls.N )

    def resample(cls):
        if cls.resampling == "systematic":
            # choose positions with constant random offset
            positions = (np.random.random() + np.arange(cls.N)) / cls.N
        elif cls.resampling == "stratified":
            # make N random subdivisions and choose a random position within
            positions = (np.random.random(cls.N) + range(cls.N)) / cls.N        
        
        indices = np.zeros(cls.N, 'i')
        cumulative_sum = np.cumsum(cls.weights)
        i,j = 0,0
        while i < cls.N:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1

        cls.particles = np.take(cls.particles, indices, axis = 0)
        cls.weights = np.full(cls.N, 1 / cls.N)

class Model(ABC):
    """ Abstract interface for the model used in the ParticleFilter class """
    @abstractmethod
    def sample_prior(cls,n):
        """ Returns array-like of n samples from the prior """
        pass

    @abstractmethod
    def sample_transition(cls, x, t, dt):
        """ Returns one sample from the transition density f(. | x) from time t to t + dt """
        pass

    @abstractmethod
    def observation_density(cls, x, y, t):
        """ Returns the probability that at time t the value of y is observed """
        pass