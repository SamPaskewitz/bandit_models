import numpy as np

def true(gain, loss, sim_pars):
    '''
    Subjective reward is true reward, i.e. gain - loss.
    '''
    return gain - loss
true.par_names = []

def weighted(gain, loss, sim_pars):
    '''
    Subjective reward is the weighted difference of gain and loss,
    as in the expectancy-value model.
    '''
    return sim_pars['gain_weight']*gain - (1 - sim_pars['gain_weight'])*loss
weighted.par_names = ['gain_weight']