import numpy as np
import pandas as pd

class simple_bandit:
    '''
    Class for simple multi-armed bandit tasks.
    
    Attributes
    ----------
    n_a: int
        Number of actions.
    n_t: int
        Number of time steps in the task.
    dist_list: list
        List of sampling functions for each action.
    
    Notes
    -----
    Each action has an associated reward distribution (sampling function).
    '''
    def __init__(self, dist_list, n_t = 1000):
        '''
        Arguments
        ---------
        dist_list: list
            List of sampling functions for each probability distribution.
        n_t: int, optional
            Number of time steps in the task.  Defaults to 1000.
        '''
        self.n_a = len(dist_list)
        self.n_t = n_t
        self.dist_list = dist_list
            
    def get_gain_loss(self, a, t):
        rwd = self.dist_list[a]() # sample reward
        # return gain and loss
        if rwd > 0:
            return (rwd, 0)
        else:
            return (0, rwd)
    
class changepoint_bandit:
    '''
    Class for multi-armed bandit tasks with distributions that change at pre-specified time steps.
    
    Attributes
    ----------
    n_a: int
        Number of actions.
    n_t: int
        Number of time steps in the task.
    n_stage: int
        Number of stages (equal to 1 + number of change points).
    dist_lists: list of lists
        Each element is a list of sampling functions for each probability distribution.
        The element (of the outer list) used depends the stage of the experiment.
        The number of elements (inner lists) should be 1 + the number of change points.
    changepoints: list or array of ints
        Time step of each change point (switches to distribution list 1 at
        the 0th change point).
    '''
    def __init__(self, dist_lists, changepoints, n_t = 1000):
        '''
        Arguments
        ---------
        dist_lists: list of lists
            Each element is a list of sampling functions for each probability distribution.
            The element (of the outer list) used depends the stage of the experiment.
            The number of elements (inner lists) should be 1 + the number of change points.
        changepoints: list or array of ints
            Time step of each change point (switches to distribution list 1 at
            the 0th change point).
        n_t: int
            Number of time steps in the task.
        '''
        self.n_a = len(dist_lists[0])
        self.n_t = n_t
        self.n_stage = len(dist_lists)
        self.dist_lists = dist_lists
        self.changepoints = changepoints      
            
    def get_gain_loss(self, a, t):
        # loop through changepoints to see what stage the task is in
        rwd = None
        for i in range(self.n_stage - 1):
            if t < self.changepoints[i]:
                rwd = self.dist_lists[i][a]() # sample reward
                break
        # if not in any previous stage must be in final stage
        if rwd is None:
            rwd = self.dist_lists[self.n_stage - 1][a]() # sample reward
        # return gain and loss
        if rwd > 0:
            return (rwd, 0)
        else:
            return (0, rwd)
    
class schedule_bandit:
    '''
    Class for bandit tasks in which rewards follow a pre-specified schedule,
    i.e. a table specifying the payout for each action on each trial.
    
    Attributes
    ----------
    n_a: int
        Number of actions (is always equal to 1 in this case).
    n_t: int
        Number of time steps in the task.
    reward_schedule: array
        Matrix specifying the reward for each action on each trial.
        Rows correspond to trials and columns to actions.
    
    Notes
    -----
    The Iowa Gambling Task is an example of this type of bandit task.
    '''
    def __init__(self, gain_schedule, loss_schedule):
        '''
        Arguments
        ---------
        reward_schedule: array
            Matrix specifying the reward for each action on each trial.
            Rows correspond to trials and columns to actions.
        '''
        self.n_a = gain_schedule.shape[1]
        self.n_t = loss_schedule.shape[0]
        self.gain_schedule = gain_schedule
        self.loss_schedule = loss_schedule
        
    def get_gain_loss(self, a, t):
        return (self.gain_schedule[t, a], self.loss_schedule[t, a])