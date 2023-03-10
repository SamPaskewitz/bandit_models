import numpy as np
import pandas as pd
from numpy.random import default_rng

class mixture_bandit:
    '''
    Class for "one-armed bandit" tasks with mixture distributions.
    
    Attributes
    ----------
    n_a: int
        Number of actions (is always equal to 1 in this case).
    n_t: int
        Number of time steps in the task.
    n_dist: int
        Number of mixture components.
    dist_list: list
        List of sampling functions for each mixture component.
    dist_probs: array of floats
        Probabilities of each mixture component.
    rng: object
        Random number generator.
    
    Notes
    -----
    The "bandit" is simply a probability distribution of rewards.
    For flexibility it is defined as a mixture distribution, but the
    "mixture" may have only one component if desired.
    '''
    def __init__(self, dist_list, dist_probs = None, n_t = 1000):
        '''
        Arguments
        ---------
        dist_list: list
            List of sampling functions for each probability distribution.
        dist_probs: list/array of floats or None, optional
            If a list/array, gives the probabilities of each mixture component.
            If None (default), then all mixture components will be given equal probabilities
            (which in particular means that if dist_list has only one component then the
            probability of that lone component will be automatically set to 1.0).
        n_t: int, optional
            Number of time steps in the task.  Defaults to 1000.
        '''
        self.n_a = 1
        self.n_t = n_t
        self.n_dist = len(dist_list)
        self.dist_list = dist_list
        if dist_probs is None:
            self.dist_probs = np.array(self.n_dist*[1/self.n_dist], dtype = 'float')
        else:
            self.dist_probs = dist_probs
        self.rng = np.random.default_rng()
            
    def get_gain_loss(self, a, t):
        component = self.rng.choice(self.n_dist, p = self.dist_probs) # choose mixture component
        rwd = self.dist_list[component]() # sample reward
        # return gain and loss
        if rwd > 0:
            return (rwd, 0)
        else:
            return (0, rwd)