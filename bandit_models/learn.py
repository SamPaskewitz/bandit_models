import numpy as np
from scipy import stats
from scipy.special import digamma

def delta_rule(sim_data, rwd, a, sim_pars):
    new_sim_data = sim_data
    delta = rwd - sim_data['q'][a]
    new_sim_data['q'][a] += sim_pars['lrate']*delta
    return new_sim_data
delta_rule.par_names = ['lrate']

def pos_neg_lrates(sim_data, rwd, a, sim_pars):
    new_sim_data = sim_data
    delta = rwd - sim_data['q'][a]
    if delta > 0:
        new_sim_data['q'][a] += sim_pars['lrate_pos']*delta
    else:
        new_sim_data['q'][a] += sim_pars['lrate_neg']*delta
    return new_sim_data
pos_neg_lrates.par_names = ['lrate_pos', 'lrate_neg']

def Kalman(sim_data, rwd, a, sim_pars):
    '''
    Without loss of generality, the reward variance is fixed at 1.
    '''
    new_sim_data = sim_data
    new_mu_var = 1/(1/(sim_data['mu_var'][a] + sim_pars['drift_var']) + 1/1)
    new_sim_data['mu_mean'][a] = new_mu_var*(sim_data['mu_mean'][a]/(sim_data['mu_var'][a] + sim_pars['drift_var']) + rwd/1)
    new_sim_data['mu_var'][a] = new_mu_var
    new_sim_data['q'] = new_sim_data['mu_mean']
    return new_sim_data
Kalman.par_names = ['drift_var']

def normal(sim_data, rwd, a, sim_pars):
    '''
    Conjugate prior updating for a normal likelihood.
    
    Notes
    -----
    'mu' denotes the mean reward for each action.
    
    The reward variance is assumed to be known, so the only thing to be estimated is mu.
    It follows that the predictive/marginal likelihood is a normal distribution, rather
    than a Student's t distribution.
    
    Without loss of generality, the reward variance is fixed at 1.  The updates only rely on
    the ratio between the variance of mu and the reward variance.  Thus, the relevant free
    parameter of the model is this variance ratio, which is equivalent to the prior variance of
    mu when the reward variance is fixed at 1.
    '''
    # learning
    new_mu_var = 1/(1/sim_data['mu_var'][a] + 1/1)
    new_sim_data = sim_data
    new_sim_data['mu_mean'][a] = new_mu_var*(sim_data['mu_mean'][a]/sim_data['mu_var'][a] + rwd/1)
    new_sim_data['mu_var'][a] = new_mu_var
    new_sim_data['q'][a] = new_sim_data['mu_mean'][a]
    return new_sim_data
normal.par_names = []

def changepoint_normal(sim_data, rwd, a, sim_pars):
    '''
    This uses variational Bayesian changepoint learning (normal likelihood) for each action.
    
    Notes
    -----
    'mu' denotes the mean reward for each action.
    
    The reward variance is assumed to be known, so the only thing to be estimated is mu.
    It follows that the predictive/marginal likelihood is a normal distribution, rather
    than a Student's t distribution.
    
    Without loss of generality, the reward variance is fixed at 1.
    '''
    # estimate probability that a change point has occured
    likelihood0 = stats.norm.pdf(rwd, loc = sim_data['mu_mean'][a], scale = np.sqrt(sim_data['mu_var'][a] + 1)) # (posterior predictive) likelihood
    numerator0 = likelihood0*(1 - sim_pars['hazard_rate']) # proportional to posterior for no change point
    likelihood1 = stats.norm.pdf(rwd, loc = 0, scale = np.sqrt(sim_pars['prior_var'] + 1)) # (prior predictive) likelihood
    numerator1 = likelihood1*sim_pars['hazard_rate'] # proportional to posterior for no change point
    phi = numerator1/(numerator0 + numerator1) # estimated posterior probability for change point
    # learning
    new_mu_var = 1/((1 - phi)/sim_data['mu_var'][a] + phi/sim_pars['prior_var'] + 1/1)
    new_sim_data = sim_data
    new_sim_data['mu_mean'][a] = new_mu_var*((1 - phi)*sim_data['mu_mean'][a]/sim_data['mu_var'][a] + rwd/1)
    new_sim_data['mu_var'][a] = new_mu_var
    new_sim_data['q'][a] = (1 - sim_pars['hazard_rate'])*new_sim_data['mu_mean'][a]
    return new_sim_data
changepoint_normal.par_names = ['hazard_rate']

def bernoulli(sim_data, rwd, a, sim_pars):
    '''
    Conjugate prior updating for a Bernoulli likelihood (probability of reward being positive = p).
    
    Notes
    -----
    This is designed for tasks in which reward is either present (rwd = 1) or absent (rwd = 0).
    To make it more robust (applicable to other tasks), it implements a data transformation such
    that p is defined as the probability that reward is positive, while 1 - p is the probability
    that reward is zero or negative.  Of course this is equivalent to p = probability of reward when
    reward actually only takes on the values 1 and 0.
    '''
    new_sim_data = sim_data
    rwd_pos = rwd > 0 # indicator variable for whether reward is positive (1) or not (0)
    new_sim_data['p_alpha'][a] += rwd_pos # first hyperparameter (number of positive rewards observed)
    new_sim_data['p_beta'][a] += 1 - rwd_pos # second hyperparameter (number of non-positive rewards observed)
    new_sim_data['q'][a] = new_sim_data['p_alpha'][a]/(new_sim_data['p_alpha'][a] + new_sim_data['p_beta'][a]) # posterior predictive mean
    return new_sim_data
bernoulli.par_names = []

def changepoint_bernoulli(sim_data, rwd, a, sim_pars):
    '''
    This uses variational Bayesian changepoint learning (Bernoulli likelihood) for each action.
    
    Notes
    -----
    This is designed for tasks in which reward is either present (rwd = 1) or absent (rwd = 0).
    To make it more robust (applicable to other tasks), it implements a data transformation such
    that p is defined as the probability that reward is positive, while 1 - p is the probability
    that reward is zero or negative.  Of course this is equivalent to p = probability of reward when
    reward actually only takes on the values 1 and 0.
    '''
    new_sim_data = sim_data
    rwd_pos = rwd > 0 # indicator variable for whether reward is positive (1) or not (0)
    # estimate probability that a change point has occurred
    prior_alpha = sim_pars['prior_mean_of_beta_dist']*sim_pars['prior_nu']
    prior_beta = sim_pars['prior_nu'] - prior_alpha
    likelihood0 = stats.bernoulli.pmf(rwd, p = sim_data['q'][a]) # (posterior predictive) likelihood
    numerator0 = likelihood0*(1 - sim_pars['hazard_rate']) # proportional to posterior for no change point
    prior_pred_p = prior_alpha/(prior_alpha + prior_beta)
    likelihood1 = stats.bernoulli.pmf(rwd, p = prior_pred_p) # (prior predictive) likelihood
    numerator1 = likelihood1*sim_pars['hazard_rate'] # proportional to posterior for no change point
    phi = numerator1/(numerator0 + numerator1) # estimated posterior probability for change point
    # learning
    new_sim_data['p_alpha'][a] += (1 - phi)*rwd_pos + phi*prior_alpha # first hyperparameter (number of positive rewards observed)
    new_sim_data['p_beta'][a] += (1 - phi)*(1 - rwd_pos) + phi*prior_beta # second hyperparameter (number of non-positive rewards observed)
    new_sim_data['q'][a] = new_sim_data['p_alpha'][a]/(new_sim_data['p_alpha'][a] + new_sim_data['p_beta'][a]) # posterior predictive mean
    return new_sim_data
changepoint_bernoulli.par_names = ['hazard_rate']

def cluster_bernoulli(sim_data, rwd, a, sim_pars):
    '''
    This uses a streaming version of variational Bayes with a cluster (latent cause)
    model.  Each action's reward has a Bernoulli likelihood, which differs between clusters.
    
    Notes
    -----
    The streaming variational Bayes code is similar to the statsrat package's method for
    latent cause models.  Also see Sam Paskewitz's doctoral dissertation and his "notes on
    variational Bayes for latent cause models" for further details
    In the future I should create better documentation for this package that includes a
    detailed description.
    
    The generative model uses a simple Chinese Restaurant Process prior.
    
    This is designed for tasks in which reward is either present (rwd = 1) or absent (rwd = 0).
    To make it more robust (applicable to other tasks), it implements a data transformation such
    that p is defined as the probability that reward is positive, while 1 - p is the probability
    that reward is zero or negative.  Of course this is equivalent to p = probability of reward when
    reward actually only takes on the values 1 and 0.
    '''
    new_sim_data = sim_data
    
    # indexing variables (for convenience)
    N1 = np.min([sim_data['N'] + 1, 10]) # either the number of old clusters plus 1, or 10 (the maximum number of clusters allowed)
    ind_n = range(sim_data['N']) # index for old clusters
    ind_n1 = range(N1) # index for old clusters and potential new cluster
    
    # compute expected log-likelihood of y (indicator of whether reward was positive)
    Eq_log_lik = -np.inf*np.ones(10)
    y = rwd > 0 # indicator variable for whether reward is positive (1) or not (0)
    Eq_eta = digamma(sim_data['tau'][ind_n1, a] + 1) - digamma(sim_data['nu'][ind_n1, a] - sim_data['tau'][ind_n1, a] + 1) # Eq[eta]
    Eq_a_eta = digamma(sim_data['nu'][ind_n1, a] - sim_data['tau'][ind_n1, a] + 1) - digamma(sim_data['nu'][ind_n1, a] + 2) # Eq[a(eta)]
    b_y = 0 # b(y) This is soley here for the sake of a pedantic level of clarity when comparing to written equations elsewhere.
    T_y = y # T(y) Ditto.
    Eq_log_lik[ind_n1] = Eq_eta*T_y - Eq_a_eta - b_y
    
    # the expected log-prior on clusters was computed on the previous trial
    
    # compute phi (approximate posterior on cluster membership)
    s = np.exp(Eq_log_lik + sim_data['Eq_log_prior']) # un-normalized approximate posterior
    phi = s/s.sum()
        
    # decide whether to add a new cluster
    most_prob = np.argmax(phi) # winning (most probable) cluster
    if (most_prob == sim_data['N']) and (sim_data['N'] < 10): # same heuristic as my statsrat code
        phi_learn = phi
        new_sim_data['N'] += 1 # increase number of clusters
    else:
        phi_learn = np.zeros(10)
        phi_learn[ind_n] = phi[ind_n]/phi[ind_n].sum() # drop new cluster and re-normalize over old clusters    
    
    # learning (update hyperparameters)
    new_sim_data['tau'][:, a] += phi_learn*T_y
    new_sim_data['nu'][:, a] += phi_learn
    
    # update other variables in sim_data
    new_sim_data['sum_r'] += 1
    new_sim_data['Eq_r'] += phi_learn # This is equal to sum_r in this model (constant temporal kernel), but not in general.
    new_sim_data['Vq_r'] += phi_learn*(1 - phi_learn)   
    
    # approximate next trial's expected log-prior on clusters (using a Taylor series method)
    ind_n = range(new_sim_data['N']) # update indexing variable
    # this is based on Matthew Scherreik's dissertation (page 21)
    new_sim_data['Eq_log_prior'] = -np.inf*np.ones(10)
    D = sim_pars['alpha'] + new_sim_data['Eq_r'][ind_n].sum()
    term1 = np.log((new_sim_data['Eq_r'][ind_n]**sim_pars['r'])/D)
    term2 = ((sim_pars['r']*new_sim_data['Eq_r'][ind_n]**(sim_pars['r'] - 1))**2)/(D**2)
    term3 = ((sim_pars['r'] - sim_pars['r']**2)*new_sim_data['Eq_r'][ind_n]**(sim_pars['r'] - 2))/D
    term4 = -sim_pars['r']/(new_sim_data['Eq_r'][ind_n]**2)
    new_sim_data['Eq_log_prior'][ind_n] = term1 + 0.5*new_sim_data['Vq_r'][ind_n]*(term2 + term3 + term4)
    if new_sim_data['N'] < 10:
        new_sim_data['Eq_log_prior'][new_sim_data['N']] = np.log(sim_pars['alpha']/D)
    #Eq_log_r = np.log(new_sim_data['Eq_r'][ind_n]) - 0.5*new_sim_data['Vq_r'][ind_n]/(new_sim_data['Eq_r'][ind_n]**2)  # expected log of r (in this case the total number of times each cluster has been observed)
    #new_sim_data['Eq_log_prior'][ind_n] = Eq_log_r - np.log(new_sim_data['sum_r'] + sim_pars['alpha'])
    #if new_sim_data['N'] < 10:
        #new_sim_data['Eq_log_prior'][new_sim_data['N']] = sim_pars['alpha'] - np.log(new_sim_data['sum_r'] + sim_pars['alpha'])
    
    # predict y (indicator of whether reward was positive) for the next trial based on expected log-prior
    next_s = np.exp(new_sim_data['Eq_log_prior']) # un-normalized approximate prior on clusters
    next_phi = next_s/next_s.sum()
    post_pred_y = (new_sim_data['tau'] + 1)/(new_sim_data['nu'] + 2) # posterior predictive mean of y for each latent cause
    sim_data['q'] = (next_phi.reshape((1, 10))@post_pred_y).squeeze()
    
    return new_sim_data
cluster_bernoulli.par_names = ['alpha', 'r']

def adams_mackay_normal(sim_data, rwd, a, sim_pars):
    '''
    Implements the changepoint detection algorithm of Adams and MacKay (2007).
    We assume a normal distribution with known variance for rewards.
    The hazard function is defined by sim_data_setup.
    This implementation keeps track of sufficient statistics for each possible run length up to 50.
    To adapt this for bandit tasks, we model each response option with an independent changepoint process.
    
    Without loss of generality, the reward variance is fixed at 1.
    
    Relating Variable Names to Adams and MacKay's Notation
    ------------------------------------------------------
    rwd = x_t
    sim_data['joint_prob'][:, a] = r_{t-1}
    new_sim_data['joint_prob'][:, a] = r_t
    sim_data['hazard'] = H(r_{t-1})
    evidence = P(x_{1:t})
    run_length_dist = P(r_t | x_{1:t})
    '''
    old_joint_prob = sim_data['joint_prob'].copy()
    max_r = np.min([sim_data['n_times_chosen'][a], 100])
    
    # evaluate predictive probability
    pi = stats.norm.pdf(rwd, loc = sim_data['mu_mean'][:, a], scale = np.sqrt(sim_data['mu_var'][:, a] + 1))
    
    if sim_data['n_times_chosen'][a] > 0:
        # calculate growth probability
        for r in range(1, max_r + 1):
            sim_data['joint_prob'][r, a] = pi[r]*old_joint_prob[r - 1, a]*(1 - sim_data['hazard'][r])

        # calculate changepoint probability
        sim_data['joint_prob'][0, a] = pi[0]*np.sum(old_joint_prob[:, a]*sim_data['hazard'])
    
    # calculate evidence
    evidence = np.sum(sim_data['joint_prob'][:, a])
    
    # determine run length distribution
    sim_data['run_length_dist'][:, a] = sim_data['joint_prob'][:, a]/evidence
    
    # update sufficient statistics
    old_mu_mean = sim_data['mu_mean'].copy()
    old_mu_var = sim_data['mu_var'].copy()
    for r in range(0, np.min([max_r + 1, 99])):
        new_mu_var = 1/(1/old_mu_var[r, a] + 1/1)
        sim_data['mu_mean'][r + 1, a] = new_mu_var*(old_mu_mean[r, a]/old_mu_var[r, a] + rwd/1)
        sim_data['mu_var'][r + 1, a] = new_mu_var
       
    # perform prediction
    sim_data['q'] = np.sum(sim_data['mu_mean']*sim_data['run_length_dist'], axis = 0) # posterior predictive mean, averaged across run lengths
    
    return sim_data
adams_mackay_normal.par_names = []

def adams_mackay_bernoulli(sim_data, rwd, a, sim_pars):
    '''
    Implements the changepoint detection algorithm of Adams and MacKay (2007).
    We assume a Bernoulli distribution for rewards.
    The hazard function is defined by sim_data_setup.
    This implementation keeps track of sufficient statistics for each possible run length up to 50.
    To adapt this for bandit tasks, we model each response option with an independent changepoint process.
    
    Notes
    -----
    This is designed for tasks in which reward is either present (rwd = 1) or absent (rwd = 0).
    To make it more robust (applicable to other tasks), it implements a data transformation such
    that p is defined as the probability that reward is positive, while 1 - p is the probability
    that reward is zero or negative.  Of course this is equivalent to p = probability of reward when
    reward actually only takes on the values 1 and 0.
    
    Relating Variable Names to Adams and MacKay's Notation
    ------------------------------------------------------
    rwd = x_t
    sim_data['joint_prob'][:, a] = r_{t-1}
    new_sim_data['joint_prob'][:, a] = r_t
    sim_data['hazard'] = H(r_{t-1})
    evidence = P(x_{1:t})
    run_length_dist = P(r_t | x_{1:t})
    '''
    old_joint_prob = sim_data['joint_prob'].copy()
    max_r = np.min([sim_data['n_times_chosen'][a], 100])
    rwd_pos = np.float(rwd > 0) # indicator variable for whether reward is positive (1) or not (0)
    
    # evaluate predictive probability
    pred_mean = sim_data['p_alpha']/(sim_data['p_alpha'] + sim_data['p_beta']) # predictive mean for each run length
    pi = stats.bernoulli.pmf(rwd_pos, p = pred_mean[:, a])
    
    if sim_data['n_times_chosen'][a] > 0:
        # calculate growth probability
        for r in range(1, max_r + 1):
            sim_data['joint_prob'][r, a] = pi[r]*old_joint_prob[r - 1, a]*(1 - sim_data['hazard'][r])

        # calculate changepoint probability
        sim_data['joint_prob'][0, a] = pi[0]*np.sum(old_joint_prob[:, a]*sim_data['hazard'])
    
    # calculate evidence
    evidence = np.sum(sim_data['joint_prob'][:, a])
    
    # determine run length distribution
    sim_data['run_length_dist'][:, a] = sim_data['joint_prob'][:, a]/evidence
    
    # update sufficient statistics
    old_p_alpha = sim_data['p_alpha'].copy()
    old_p_beta = sim_data['p_beta'].copy()
    for r in range(0, np.min([max_r + 1, 99])):      
        sim_data['p_alpha'][r + 1, a] = old_p_alpha[r, a] + rwd_pos # first hyperparameter (number of positive rewards observed)
        sim_data['p_beta'][r + 1, a] = old_p_beta[r, a] + 1 - rwd_pos # second hyperparameter (number of non-positive rewards observed)
       
    # perform prediction
    post_pred_mean = sim_data['p_alpha']/(sim_data['p_alpha'] + sim_data['p_beta']) # posterior predictive mean for each run length
    sim_data['q'] = np.sum(post_pred_mean*sim_data['run_length_dist'], axis = 0) # posterior predictive mean, averaged across run lengths
    
    return sim_data
adams_mackay_bernoulli.par_names = []

def adams_mackay_normal_common(sim_data, rwd, a, sim_pars):
    '''
    Implements the changepoint detection algorithm of Adams and MacKay (2007).
    We assume a normal distribution with known variance for rewards.
    The hazard function is defined by sim_data_setup.
    This implementation keeps track of sufficient statistics for each possible run length up to 50.
    To adapt this for bandit tasks, we model each response option with an independent changepoint process.
    
    In this version, change points are assumed to be common across response options rather
    than separate for each response option.
    
    Without loss of generality, the reward variance is fixed at 1.
    
    Relating Variable Names to Adams and MacKay's Notation
    ------------------------------------------------------
    rwd = x_t
    sim_data['joint_prob'][:, a] = r_{t-1}
    new_sim_data['joint_prob'][:, a] = r_t
    sim_data['hazard'] = H(r_{t-1})
    evidence = P(x_{1:t})
    run_length_dist = P(r_t | x_{1:t})
    '''
    old_joint_prob = sim_data['joint_prob'].copy()
    max_r = np.min([sim_data['n_times_chosen'][a], 100])
    
    # FINISH UPDATING.
    
    # evaluate predictive probability
    pi = stats.norm.pdf(rwd, loc = sim_data['mu_mean'][:, a], scale = np.sqrt(sim_data['mu_var'][:, a] + 1))
    
    if sim_data['n_times_chosen'][a] > 0:
        # calculate growth probability
        for r in range(1, max_r + 1):
            sim_data['joint_prob'][r, :] = old_joint_prob[r - 1, a]*pi[r]*(1 - sim_data['hazard'][r])

        # calculate changepoint probability
        sim_data['joint_prob'][0, :] = np.sum(old_joint_prob[:, a]*pi*sim_data['hazard'])
    
    # calculate evidence
    evidence = np.sum(sim_data['joint_prob'][:, a])
    
    # determine run length distribution (all response options have the same run length distribution)
    sim_data['run_length_dist'][:, :] = sim_data['joint_prob'][:, a]/evidence
    
    # update sufficient statistics
    old_mu_mean = sim_data['mu_mean'].copy()
    old_mu_var = sim_data['mu_var'].copy()
    for r in range(0, np.min([max_r + 1, 99])):
        new_mu_var = 1/(1/old_mu_var[r, a] + 1/1)
        sim_data['mu_mean'][r + 1, a] = new_mu_var*(old_mu_mean[r, a]/old_mu_var[r, a] + rwd/1)
        sim_data['mu_var'][r + 1, a] = new_mu_var
       
    # perform prediction
    sim_data['q'] = np.sum(sim_data['mu_mean']*sim_data['run_length_dist'], axis = 0) # posterior predictive mean, averaged across run lengths
    
    return sim_data
adams_mackay_normal_common.par_names = []