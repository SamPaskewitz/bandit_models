import numpy as np
from scipy.special import softmax as sfmx

def epsilon_greedy(sim_data, a_psb, sim_pars):
    if sim_data['n_a'] > 1:
        n_a_psb = np.sum(a_psb) # number of actions possible on current trial
        a_prob = a_psb*sim_pars['epsilon']/(n_a_psb - 1) # default prob for possible actions 
        a_prob[sim_data['q'].argmax()] = 1 - sim_pars['epsilon'] # prob for action with greatest estimated value
    else:
        a_prob = [1.0]
    return a_prob
epsilon_greedy.par_names = ['epsilon']

def softmax(sim_data, a_psb, sim_pars):
    foo = sfmx(sim_pars['inv_temp']*sim_data['q'])
    bar = a_psb*foo
    return bar/bar.sum()
softmax.par_names = ['inv_temp']

def fixed_softmax(sim_data, a_psb, sim_pars):
    foo = sfmx(5*sim_data['q'])
    bar = a_psb*foo
    return bar/bar.sum()
fixed_softmax.par_names = []

def time_varying_softmax(sim_data, a_psb, sim_pars):
    '''
    The inverse temperature varies over time according to 
    a power law function that can either increase or decrease.
    See e.g. Wetzels et al 2010.
    '''
    inv_temp = ((sim_data['t'] + 1)/10)**(sim_pars['inv_temp_power'])
    foo = sfmx(inv_temp*sim_data['q'])
    bar = a_psb*foo
    return bar/bar.sum()
time_varying_softmax.par_names = ['inv_temp_power']