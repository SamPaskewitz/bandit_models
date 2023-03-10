import numpy as np

def q_only(n_a, sim_pars):
    '''
    Only keep track of q (estimated average reward per action).
    '''
    sim_data = {'n_a': n_a,
                't': 0,
                'n_times_chosen': np.zeros(n_a, dtype = 'int'), # number of times each action has been chosen
                'q': np.zeros(n_a)}
    return sim_data
q_only.par_names = []

def normal(n_a, sim_pars):
    '''
    Conjugate prior updating; keep track of posterior mean and variance of mu (mean reward per action).
    q (estimated average reward) is simply a copy of mu_mean (point estimate).
    The prior mean is fixed at 0.
    '''
    sim_data = {'n_a': n_a,
                't': 0,
                'n_times_chosen': np.zeros(n_a, dtype = 'int'), # number of times each action has been chosen
                'mu_mean': np.zeros(n_a, dtype = 'float'),
                'mu_var': np.array(n_a*[sim_pars['prior_var']], dtype = 'float'),
                'q': np.zeros(n_a, dtype = 'float')}
    return sim_data
normal.par_names = ['prior_var']

def normal_free_prior_mean(n_a, sim_pars):
    '''
    Conjugate prior updating; keep track of posterior mean and variance of mu (mean reward per action).
    q (estimated average reward) is simply a copy of mu_mean (point estimate).
    The prior mean is a free parameter.
    
    Notes
    -----
    The influence of the prior mean on the posterior mean of any response option decreases with
    the number of times that response option has been chosen.  Thus, a positive prior mean acts
    as an exploration bonus (the fewer times an option has been chosen, the better it looks), while
    a negative prior mean discourages exploration (the more times an option has been chosen, the
    better it looks).
    '''
    sim_data = {'n_a': n_a,
                't': 0,
                'n_times_chosen': np.zeros(n_a, dtype = 'int'), # number of times each action has been chosen
                'mu_mean': np.array(n_a*[sim_pars['prior_mean']], dtype = 'float'),
                'mu_var': np.array(n_a*[sim_pars['prior_var']], dtype = 'float'),
                'q': np.zeros(n_a, dtype = 'float')}
    return sim_data
normal_free_prior_mean.par_names = ['prior_var', 'prior_mean']

def bernoulli(n_a, sim_pars):
    '''
    Conjugate prior updating for Bernoulli likelihood (likelihood of reward = p).
    q (estimated average reward) should be the posterior predictive mean of p.
    '''
    prior_alpha = sim_pars['prior_mean_of_beta_dist']*sim_pars['prior_nu']
    prior_beta = sim_pars['prior_nu'] - prior_alpha
    sim_data = {'n_a': n_a,
                't': 0,
                'n_times_chosen': np.zeros(n_a, dtype = 'int'), # number of times each action has been chosen
                'p_alpha': np.array(n_a*[prior_alpha], dtype = 'float'), # hyperparameter
                'p_beta': np.array(n_a*[prior_beta], dtype = 'float'), # other hyperparameter
                'q': np.zeros(n_a, dtype = 'float')}
    return sim_data
bernoulli.par_names = ['prior_mean_of_beta_dist', 'prior_nu']

def cluster_bernoulli(n_a, sim_pars):
    '''
    Bernoulli likelihood (likelihood of reward = p) in which trials belong to
    different clusters, i.e. latent causes.
    q (estimated average reward) should be the posterior predictive mean of p.
    
    Notes
    -----
    We set an upper limit of 10 on the number of inferred clusters (i.e. latent causes).
    
    For ease of interpretability, we specify the prior mean as a psychological model parameter
    instead of the prior value of tau (the first natural parameter of the conjugate prior).
    '''
    prior_tau = (sim_pars['prior_nu'] + 2)*sim_pars['prior_mean_of_beta_dist'] - 1
    sim_data = {'n_a': n_a,
                't': 0,
                'n_times_chosen': np.zeros(n_a, dtype = 'int'), # number of times each action has been chosen
                'tau': np.array(10*n_a*[prior_tau], dtype = 'float').reshape((10, n_a)), # natural hyperparameter
                'nu': np.array(10*n_a*[sim_pars['prior_nu']], dtype = 'float').reshape((10, n_a)), # other natural hyperparameter ('sample size')
                'q': np.zeros(n_a, dtype = 'float'),
                'sum_r': 0,
                'Eq_r': np.zeros(10),
                'Vq_r': np.zeros(10),
                'Eq_log_prior': np.array([0] + 9*[-np.inf]),
                'N': 1} # estimated number of latent causes (tends to increase with learning)
    return sim_data
cluster_bernoulli.par_names = ['prior_mean_of_beta_dist', 'prior_nu']

def adams_mackay_normal_constant_hazard(n_a, sim_pars):
    '''
    Implements the changepoint detection algorithm of Adams and MacKay (2007).
    We assume a normal distribution for rewards.
    We also assume a constant hazard function.
    This implementation keeps track of sufficient statistics for each possible run length up to 100.
    To adapt this for bandit tasks, we model each response option with an independent changepoint process.
    '''
    sim_data = {'n_a': n_a,
                't': 0,
                'n_times_chosen': np.zeros(n_a, dtype = 'int'), # number of times each action has been chosen
                'mu_mean': np.zeros((100, n_a), dtype = 'float'),
                'mu_var': np.array(100*n_a*[sim_pars['prior_var']], dtype = 'float').reshape((100, n_a)),
                'joint_prob': np.zeros((100, n_a), dtype = 'float'),
                'run_length_dist': np.zeros((100, n_a), dtype = 'float'),
                'hazard': np.array(100*[sim_pars['hazard_rate']]),
                'q': np.zeros(n_a, dtype = 'float')
               }
    sim_data['joint_prob'][0, :] = 1.0
    sim_data['run_length_dist'][0, :] = 1.0
    return sim_data
adams_mackay_normal_constant_hazard.par_names = ['hazard_rate', 'prior_var']

def adams_mackay_normal_free_prior_mean_constant_hazard(n_a, sim_pars):
    '''
    Implements the changepoint detection algorithm of Adams and MacKay (2007).
    We assume a normal distribution for rewards.
    We also assume a constant hazard function.
    This implementation keeps track of sufficient statistics for each possible run length up to 100.
    To adapt this for bandit tasks, we model each response option with an independent changepoint process.
    '''
    sim_data = {'n_a': n_a,
                't': 0,
                'n_times_chosen': np.zeros(n_a, dtype = 'int'), # number of times each action has been chosen
                'mu_mean': np.array(100*n_a*[sim_pars['prior_mean']], dtype = 'float').reshape((100, n_a)),
                'mu_var': np.array(100*n_a*[sim_pars['prior_var']], dtype = 'float').reshape((100, n_a)),
                'joint_prob': np.zeros((100, n_a), dtype = 'float'),
                'run_length_dist': np.zeros((100, n_a), dtype = 'float'),
                'hazard': np.array(100*[sim_pars['hazard_rate']]),
                'q': np.zeros(n_a, dtype = 'float')
               }
    sim_data['joint_prob'][0, :] = 1.0
    sim_data['run_length_dist'][0, :] = 1.0
    return sim_data
adams_mackay_normal_free_prior_mean_constant_hazard.par_names = ['hazard_rate', 'prior_mean', 'prior_var']

def adams_mackay_normal_increasing_hazard(n_a, sim_pars):
    '''
    Implements the changepoint detection algorithm of Adams and MacKay (2007).
    We assume a normal distribution for rewards.
    We also assume an increasing hazard function of the following form:
        hazard = initial_hazard - (1 - initial_hazard)*exp(-exp_hazard_par*r_t).
    This implementation keeps track of sufficient statistics for each possible run length up to 100.
    To adapt this for bandit tasks, we model each response option with an independent changepoint process.
    '''
    sim_data = {'n_a': n_a,
                't': 0,
                'n_times_chosen': np.zeros(n_a, dtype = 'int'), # number of times each action has been chosen
                'mu_mean': np.zeros((100, n_a), dtype = 'float'),
                'mu_var': np.array(100*n_a*[sim_pars['prior_var']], dtype = 'float').reshape((100, n_a)),
                'joint_prob': np.zeros((100, n_a), dtype = 'float'),
                'run_length_dist': np.zeros((100, n_a), dtype = 'float'),
                'hazard': sim_pars['initial_hazard'] + (1 - sim_pars['initial_hazard'])*(1 - np.exp(-sim_pars['exp_hazard_par']*np.arange(100))), 
                'q': np.zeros(n_a, dtype = 'float')
               }
    sim_data['joint_prob'][0, :] = 1.0
    sim_data['run_length_dist'][0, :] = 1.0
    return sim_data
adams_mackay_normal_increasing_hazard.par_names = ['initial_hazard', 'exp_hazard_par', 'prior_var']

def adams_mackay_normal_free_prior_mean_increasing_hazard(n_a, sim_pars):
    '''
    Implements the changepoint detection algorithm of Adams and MacKay (2007).
    We assume a normal distribution for rewards.
    We also assume an increasing hazard function of the following form:
        hazard = initial_hazard - (1 - initial_hazard)*exp(-exp_hazard_par*r_t).
    This implementation keeps track of sufficient statistics for each possible run length up to 100.
    To adapt this for bandit tasks, we model each response option with an independent changepoint process.
    The prior mean is a free parameter.
    '''
    sim_data = {'n_a': n_a,
                't': 0,
                'n_times_chosen': np.zeros(n_a, dtype = 'int'), # number of times each action has been chosen
                'mu_mean': np.array(100*n_a*[sim_pars['prior_mean']], dtype = 'float').reshape((100, n_a)),
                'mu_var': np.array(100*n_a*[sim_pars['prior_var']], dtype = 'float').reshape((100, n_a)),
                'joint_prob': np.zeros((100, n_a), dtype = 'float'),
                'run_length_dist': np.zeros((100, n_a), dtype = 'float'),
                'hazard': sim_pars['initial_hazard'] + (1 - sim_pars['initial_hazard'])*(1 - np.exp(-sim_pars['exp_hazard_par']*np.arange(100))), 
                'q': np.zeros(n_a, dtype = 'float')
               }
    sim_data['joint_prob'][0, :] = 1.0
    sim_data['run_length_dist'][0, :] = 1.0
    return sim_data
adams_mackay_normal_free_prior_mean_increasing_hazard.par_names = ['initial_hazard', 'exp_hazard_par', 'prior_mean', 'prior_var']

def adams_mackay_normal_increasing_hazard_start0(n_a, sim_pars):
    '''
    Implements the changepoint detection algorithm of Adams and MacKay (2007).
    We assume a normal distribution for rewards.
    We also assume an increasing hazard function of the following form:
        hazard = 1 - exp(-exp_hazard_par*r_t).
    This ensures that hazard starts at 0.001 (approximately 0) for a run length of 0.
    This implementation keeps track of sufficient statistics for each possible run length up to 100.
    To adapt this for bandit tasks, we model each response option with an independent changepoint process.
    '''
    sim_data = {'n_a': n_a,
                't': 0,
                'n_times_chosen': np.zeros(n_a, dtype = 'int'), # number of times each action has been chosen
                'mu_mean': np.zeros((100, n_a), dtype = 'float'),
                'mu_var': np.array(100*n_a*[sim_pars['prior_var']], dtype = 'float').reshape((100, n_a)),
                'joint_prob': np.zeros((100, n_a), dtype = 'float'),
                'run_length_dist': np.zeros((100, n_a), dtype = 'float'),
                'hazard': 0.001 + (1 - 0.001)*(1 - np.exp(-sim_pars['exp_hazard_par']*np.arange(100))), 
                'q': np.zeros(n_a, dtype = 'float')
               }
    sim_data['joint_prob'][0, :] = 1.0
    sim_data['run_length_dist'][0, :] = 1.0
    return sim_data
adams_mackay_normal_increasing_hazard_start0.par_names = ['exp_hazard_par', 'prior_var']

def adams_mackay_bernoulli_constant_hazard(n_a, sim_pars):
    '''
    Implements the changepoint detection algorithm of Adams and MacKay (2007).
    We assume a Bernoulli distribution for rewards.
    We also assume a constant hazard function.
    This implementation keeps track of sufficient statistics for each possible run length up to 100.
    To adapt this for bandit tasks, we model each response option with an independent changepoint process.
    '''
    prior_alpha = sim_pars['prior_mean_of_beta_dist']*sim_pars['prior_nu']
    prior_beta = sim_pars['prior_nu'] - prior_alpha
    sim_data = {'n_a': n_a,
                't': 0,
                'n_times_chosen': np.zeros(n_a, dtype = 'int'), # number of times each action has been chosen
                'p_alpha': np.array(100*n_a*[prior_alpha], dtype = 'float').reshape((100, n_a)), # hyperparameter
                'p_beta': np.array(100*n_a*[prior_beta], dtype = 'float').reshape((100, n_a)), # other hyperparameter
                'joint_prob': np.zeros((100, n_a), dtype = 'float'),
                'run_length_dist': np.zeros((100, n_a), dtype = 'float'),
                'hazard': np.array(100*[sim_pars['hazard_rate']]), 
                'q': np.zeros(n_a, dtype = 'float')
               }
    sim_data['joint_prob'][0, :] = 1.0
    sim_data['run_length_dist'][0, :] = 1.0
    return sim_data
adams_mackay_bernoulli_constant_hazard.par_names = ['hazard_rate', 'prior_mean_of_beta_dist', 'prior_nu']

def adams_mackay_bernoulli_increasing_hazard(n_a, sim_pars):
    '''
    Implements the changepoint detection algorithm of Adams and MacKay (2007).
    We assume a Bernoulli distribution for rewards.
    We also assume an increasing hazard function of the following form:
        hazard = initial_hazard - (1 - initial_hazard)*exp(-exp_hazard_par*r_t).
    This implementation keeps track of sufficient statistics for each possible run length up to 100.
    To adapt this for bandit tasks, we model each response option with an independent changepoint process.
    '''
    prior_alpha = sim_pars['prior_mean_of_beta_dist']*sim_pars['prior_nu']
    prior_beta = sim_pars['prior_nu'] - prior_alpha
    sim_data = {'n_a': n_a,
                't': 0,
                'n_times_chosen': np.zeros(n_a, dtype = 'int'), # number of times each action has been chosen
                'p_alpha': np.array(100*n_a*[prior_alpha], dtype = 'float').reshape((100, n_a)), # hyperparameter
                'p_beta': np.array(100*n_a*[prior_beta], dtype = 'float').reshape((100, n_a)), # other hyperparameter
                'joint_prob': np.zeros((100, n_a), dtype = 'float'),
                'run_length_dist': np.zeros((100, n_a), dtype = 'float'),
                'hazard': sim_pars['initial_hazard'] + (1 - sim_pars['initial_hazard'])*(1 - np.exp(-sim_pars['exp_hazard_par']*np.arange(100))), 
                'q': np.zeros(n_a, dtype = 'float')
               }
    sim_data['joint_prob'][0, :] = 1.0
    sim_data['run_length_dist'][0, :] = 1.0
    return sim_data
adams_mackay_bernoulli_increasing_hazard.par_names = ['initial_hazard', 'exp_hazard_par', 'prior_mean_of_beta_dist', 'prior_nu']