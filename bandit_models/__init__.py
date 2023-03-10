import numpy as np
import pandas as pd
import xarray as xr
import nlopt
from numpy.random import default_rng

class model:
    def __init__(self, name, a_prob, sim_data_setup, subj_rwd, learn):
        # add attributes to object ('self')
        self.name = name
        self.sim_data_setup = sim_data_setup
        self.a_prob = a_prob
        self.subj_rwd = subj_rwd
        self.learn = learn
        self.rng = np.random.default_rng()
        # determine model's parameter space
        self.par_names = list(np.unique(sim_data_setup.par_names + a_prob.par_names + subj_rwd.par_names + learn.par_names))
        self.pars = pars.loc[self.par_names]
        self.n_p = len(self.par_names) # number of parameters
        
    def train(self, bandit, par_val = None):
        '''
        Simulate learning using a bandit task object with given parameter values.

        Notes
        -----
        Currently this assumes that all actions are possible on each trial and
        that feedback is given on each trial.
        ''' 
        # use default parameters unless others are given
        if par_val is None:
            sim_pars = self.pars['default']
        else:
            # check that parameter values are within acceptable limits; if so assemble into a pandas series
            # for some reason, the optimization functions go slightly outside the specified bounds
            abv_min = par_val >= self.pars['min'] - 0.0001
            blw_max = par_val <= self.pars['max'] + 0.0001
            all_ok = np.prod(abv_min & blw_max)
            assert all_ok, 'par_val outside acceptable limits'
            sim_pars = pd.Series(par_val, self.pars.index)
        
        # setup
        n_a = bandit.n_a # number of actions
        gain = np.zeros(bandit.n_t, dtype = 'float')
        loss = np.zeros(bandit.n_t, dtype = 'float')
        rwd = np.zeros(bandit.n_t, dtype = 'float')
        a_prob = np.zeros((bandit.n_t, n_a), dtype = 'float')
        a = np.zeros(bandit.n_t, dtype = 'int')
        sim_data = self.sim_data_setup(n_a, sim_pars)
        
        # loop through trials
        for t in range(bandit.n_t):
            # action selection
            a_prob[t, :] = self.a_prob(sim_data, np.ones(n_a), sim_pars)
            a[t] = self.rng.choice(n_a, p = a_prob[t, :])
            # observe gain and loss and calculate (subjective) reward
            (gain[t], loss[t]) = bandit.get_gain_loss(a[t], t)
            subj_rwd = self.subj_rwd(gain[t], loss[t], sim_pars) # subjective reward
            rwd[t] = gain[t] - loss[t] # true reward
            # learning
            sim_data = self.learn(sim_data, subj_rwd, a[t], sim_pars)
            sim_data['n_times_chosen'][a[t]] += 1 # update number of times the current action has been chosen
            sim_data['t'] += 1
            
        ds = xr.Dataset(data_vars = {'a': (['t'], a),
                                     'a_psb': (['t', 'a_name'], np.ones((bandit.n_t, n_a))),
                                     'fb_given': (['t'], np.ones(bandit.n_t)),
                                     'gain': (['t'], gain),
                                     'loss': (['t'], loss),
                                     'rwd': (['t'], rwd)},
                            coords = {'t': range(bandit.n_t)})
        return {'gain': gain, 'loss': loss, 'rwd': rwd, 'a': a, 'a_prob': a_prob, 'ds': ds, 'sim_data': sim_data}
        
    def log_lik(self, data, par_val):
        '''
        Obtain log-likelihood of existing data (single participant/task) with given
        parameter values.
        ''' 
        # use default parameters unless others are given
        if par_val is None:
            sim_pars = self.pars['default']
        else:
            # check that parameter values are within acceptable limits; if so assemble into a pandas series
            # for some reason, the optimization functions go slightly outside the specified bounds
            abv_min = par_val >= self.pars['min'] - 0.0001
            blw_max = par_val <= self.pars['max'] + 0.0001
            all_ok = np.prod(abv_min & blw_max)
            assert all_ok, 'par_val outside acceptable limits'
            sim_pars = pd.Series(par_val, self.pars.index)
        
        # setup
        n_a = data['a_psb'].shape[1] # number of actions
        t0 = int(data['t'].min()) # initial trial
        t_final = int(data['t'].max()) # final trial
        sim_data = self.sim_data_setup(n_a, sim_pars)
        log_lik = 0.0 # total log-likelihood of observed data (add up trial by trial)
        
        # loop through trials
        for t in range(t0, t_final):
            # compute action probabilities
            a_prob = self.a_prob(sim_data, data['a_psb'].loc[{'t': t}], sim_pars)
            a_prob[a_prob == 0.0] = 0.00000001
            # add current trial's log-likelihood to the total
            a_t = data['a'].loc[{'t': t}]
            log_lik += np.log(a_prob[a_t]).values
            # learning (only if feedback is given)
            if data['fb_given'].loc[{'t': t}] == 1:
                sim_data['n_times_chosen'][a_t] += 1 # update number of times the current action has been chosen
                subj_rwd = self.subj_rwd(data['gain'].loc[{'t': t}], data['loss'].loc[{'t': t}], sim_pars)
                sim_data = self.learn(sim_data, subj_rwd, a_t, sim_pars)
        return log_lik
    
    def fit_ml(self, data, global_time = 15, local_time = 15, algorithm = nlopt.GN_DIRECT_L, x0 = None):
        '''
        Obtain maximum likelihood fit to data from a single participant and session.
        '''
        par_min = self.pars['min']
        par_max = self.pars['max']
        
        def objective(par_val, grad = None):
            return self.log_lik(data, par_val)

        # global optimization (to find approximate optimum)
        if x0 is None:
            x0 = np.array((par_max + par_min)/2) # midpoint of each parameter's allowed interval
        gopt = nlopt.opt(algorithm, self.n_p)
        gopt.set_max_objective(objective)
        gopt.set_lower_bounds(np.array(par_min + 0.001))
        gopt.set_upper_bounds(np.array(par_max - 0.001))
        gopt.set_maxtime(global_time)
        gxopt = gopt.optimize(x0)
        if local_time > 0:
            # local optimization (to refine answer)
            lopt = nlopt.opt(nlopt.LN_SBPLX, self.n_p)
            lopt.set_max_objective(objective)
            lopt.set_lower_bounds(np.array(par_min + 0.001))
            lopt.set_upper_bounds(np.array(par_max - 0.001))
            lopt.set_maxtime(local_time)
            lxopt = lopt.optimize(gxopt)
            best_pars = lxopt
            log_lik = lopt.last_optimum_value()
        else:
            best_pars = gxopt
            log_lik = gopt.last_optimum_value()

        return {'par_val': pd.Series(index = self.par_names, data = best_pars), 'log_lik': log_lik}
    
    def prediction_fit(self, data_dict, global_time = 15, local_time = 15, algorithm = nlopt.GN_DIRECT_L):
        '''
        Fit to the first half of each individual's data and compute log-likelihood of the remaining half.
        This is a measure of generalization.
        
        Parameters
        ----------
        data_dict: dict
            Keys are individual ID codes and values are xarray datasets that each represent
            data from the corresponding individual.
        global_time: int, optional
            Number of seconds to run the global optimization algorithm for each individual.
            Defaults to 15.
        local_time: int, optional
            Number of seconds to run the local optimization algorithm for each individual.
            Defaults to 15.
        algorithm: object, optional
            Specifies the algorithm used for global optimization.  Defaults to nlopt.GD_STOGO.
            
        Output
        ------
        A data frame with the following columns (index/row is individual participant)
        
        log_lik_fit: Log-likelihood of the first half of the data, using the maximum
        likelihood parameter estimates for the first half of the data.
        log_lik_pred: Log-likelihood of the second half of the data, using the maximum
        likelihood parameter estimates for the first half of the data.
        model: Model name.
        global_time: Global optimization time (in seconds).
        local_time: Local optimization time (in seconds).
        algorithm: Name of the global optimization algorithm.
        
        Notes
        -----
        To obtain the log-likelihood of the second half of the data using the best fitting
        parameters from the first half, the function performs the following computation:
        
        log-likelihood (second half) = log-likelihood (all trials) - log-likelihood (first half)
        
        This works because log-likelihood is additive across trials.  It would be incorrect
        to compute the log-likelihood of the second half of the trials in the simple and obvious
        way, because this would ignore the learning done by the participant during the first half
        of the trials.
        '''
        ident = list(data_dict.keys()) # list of participant IDs
        n_i = len(ident) # number of participants
        t0 = int(data_dict[ident[0]]['t'].min())
        t_final = int(data_dict[ident[0]]['t'].max())
        t_half = int((t_final - t0)/2)
        data_dict_half0 = {}
        for i in ident:
            data_dict_half0[i] = data_dict[i].loc[{'t': range(t0, t_half)}]
        df = pd.DataFrame({'log_lik_fit': n_i*[0.0],
                           'log_lik_pred': n_i*[0.0]},
                          index = ident)
        for i in ident:
            fit_half0 = self.fit_ml(data_dict_half0[i], global_time, local_time, algorithm)
            df.loc[i, 'log_lik_fit'] = fit_half0['log_lik']
            df.loc[i, 'log_lik_pred'] = self.log_lik(data_dict[i], fit_half0['par_val'].values) - fit_half0['log_lik']
        
        df['model'] = self.name
        df['global_time'] = global_time
        df['local_time'] = local_time
        df['algorithm'] = nlopt.algorithm_name(algorithm)
        return df
    
    def multi_fit(self, data_dict, global_time = 15, local_time = 15, algorithm = nlopt.GN_DIRECT_L):
        '''
        Obtain maximum likelihood fits to data from multiple individuals.
        
        Parameters
        ----------
        data_dict: dict
            Keys are individual ID codes and values are xarray datasets that each represent
            data from the corresponding individual.
        global_time: int, optional
            Number of seconds to run the global optimization algorithm for each individual.
            Defaults to 15.
        local_time: int, optional
            Number of seconds to run the local optimization algorithm for each individual.
            Defaults to 15.
        algorithm: object, optional
            Specifies the algorithm used for global optimization.  Defaults to nlopt.GD_STOGO.
        '''
        ident = list(data_dict.keys()) # list of participant IDs
        n_i = len(ident) # number of participants
        n_par = len(self.par_names) # number of model free parameters
        df = pd.DataFrame({'log_lik': n_i*[0.0],
                           'aic': n_i*[0.0]},
                          index = ident)
        df.index.name = 'ident'
        for par_name in self.par_names:
            df[par_name] = 0.0

        # loop through participants ('ident'/'subject')
        for i in ident:
            try:
                fit_result = self.fit_ml(data_dict[i], global_time, local_time, algorithm)
                df.loc[i, 'log_lik'] = fit_result['log_lik']
                df.loc[i, 'aic'] = 2*(n_par - fit_result['log_lik'])
                for par_name in self.par_names:
                    df.loc[i, par_name] = fit_result['par_val'][par_name]                
            except:
                df.loc[i] = np.nan
        
        df['model'] = self.name
        df['global_time'] = global_time
        df['local_time'] = local_time
        df['algorithm'] = nlopt.algorithm_name(algorithm)
        return df
    
########## PARAMETERS ##########
par_names = []; par_list = [] 
par_names += ['lrate']; par_list += [{'min': 0.0, 'max': 1.0, 'default': 0.2}]
par_names += ['lrate_pos']; par_list += [{'min': 0.0, 'max': 1.0, 'default': 0.2}]
par_names += ['lrate_neg']; par_list += [{'min': 0.0, 'max': 1.0, 'default': 0.2}]
par_names += ['rwd_var']; par_list += [{'min' : 0.0, 'max' : 20.0, 'default' : 1.0}]
par_names += ['drift_var']; par_list += [{'min' : 0.0, 'max' : 20.0, 'default' : 1.0}]
par_names += ['prior_var']; par_list += [{'min' : 0.0, 'max' : 20.0, 'default' : 1.0}]
par_names += ['prior_mean']; par_list += [{'min' : -5.0, 'max' : 5.0, 'default' : 0.0}]
par_names += ['hazard_rate']; par_list += [{'min' : 0.0, 'max' : 1.0, 'default' : 0.1}] # constant hazard
par_names += ['exp_hazard_par']; par_list += [{'min' : 0.0, 'max' : 2.0, 'default' : 0.1}] # for increasing hazard
par_names += ['initial_hazard']; par_list += [{'min' : 0.0, 'max' : 0.5, 'default' : 0.1}] # for increasing hazard
par_names += ['epsilon']; par_list += [{'min' : 0.0, 'max' : 0.5, 'default' : 0.1}]
par_names += ['inv_temp']; par_list += [{'min' : 0.01, 'max' : 10.0, 'default' : 5.0}]
par_names += ['prior_mean_of_beta_dist']; par_list += [{'min' : 0.01, 'max' : 0.99, 'default' : 0.5}]
par_names += ['prior_nu']; par_list += [{'min' : 0.0, 'max' : 20.0, 'default' : 2.0}] # a natural hyperparameter for exponential family distributions (prior "sample size")
par_names += ['alpha']; par_list += [{'min' : 0.0, 'max' : 10.0, 'default' : 1.0}] # concentration parameter for cluster models
par_names += ['r']; par_list += [{'min' : 1.0, 'max' : 10.0, 'default' : 1.5}] # power parameter for cluster models
par_names += ['inv_temp_power']; par_list += [{'min' : -5.0, 'max' : 5.0, 'default' : 1.0}] # parameter for time-varying softmax decision
par_names += ['gain_weight']; par_list += [{'min' : 0.0, 'max' : 1.0, 'default' : 0.5}] # subjective weighting for gains vs. losses
pars = pd.DataFrame(par_list, index = par_names)
del par_names; del par_list
