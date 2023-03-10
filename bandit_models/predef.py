from bandit_models import model, sim_data_setup, a_prob, subj_rwd, learn

delta_rule_eg = model(name = 'delta_rule_eg', 
                      sim_data_setup = sim_data_setup.q_only, 
                      a_prob = a_prob.epsilon_greedy,
                      subj_rwd = subj_rwd.true,
                      learn = learn.delta_rule)

delta_rule_sfmx = model(name = 'delta_rule_sfmx', 
                        sim_data_setup = sim_data_setup.q_only, 
                        a_prob = a_prob.softmax,
                        subj_rwd = subj_rwd.true,
                        learn = learn.delta_rule)

delta_rule_tvsfmx = model(name = 'delta_rule_tvsfmx', 
                          sim_data_setup = sim_data_setup.q_only, 
                          a_prob = a_prob.time_varying_softmax,
                          subj_rwd = subj_rwd.true,
                          learn = learn.delta_rule)

Kalman_eg = model(name = 'Kalman_eg', 
                  sim_data_setup = sim_data_setup.normal, 
                  a_prob = a_prob.epsilon_greedy,
                  subj_rwd = subj_rwd.true,
                  learn = learn.Kalman)

Kalman_sfmx = model(name = 'Kalman_sfmx', 
                    sim_data_setup = sim_data_setup.normal, 
                    a_prob = a_prob.softmax,
                    subj_rwd = subj_rwd.true,
                    learn = learn.Kalman)

changepoint_normal_eg = model(name = 'changepoint_normal_eg', 
                              sim_data_setup = sim_data_setup.normal, 
                              a_prob = a_prob.epsilon_greedy,
                              subj_rwd = subj_rwd.true,
                              learn = learn.changepoint_normal)

changepoint_normal_sfmx = model(name = 'changepoint_normal_sfmx', 
                                sim_data_setup = sim_data_setup.normal, 
                                a_prob = a_prob.softmax,
                                subj_rwd = subj_rwd.true,
                                learn = learn.changepoint_normal)

changepoint_normal_tvsfmx = model(name = 'changepoint_normal_tvsfmx', 
                                  sim_data_setup = sim_data_setup.normal, 
                                  a_prob = a_prob.time_varying_softmax,
                                  subj_rwd = subj_rwd.true,
                                  learn = learn.changepoint_normal)

pos_neg_lrates_sfmx = model(name = 'pos_neg_lrates_sfmx', 
                            sim_data_setup = sim_data_setup.q_only, 
                            a_prob = a_prob.softmax,
                            subj_rwd = subj_rwd.true,
                            learn = learn.pos_neg_lrates)

normal_sfmx = model(name = 'normal_sfmx',
                    sim_data_setup = sim_data_setup.normal, 
                    a_prob = a_prob.softmax,
                    subj_rwd = subj_rwd.true,
                    learn = learn.normal)

normal_free_prior_mean_sfmx = model(name = 'normal_free_prior_mean_sfmx',
                                    sim_data_setup = sim_data_setup.normal_free_prior_mean, 
                                    a_prob = a_prob.softmax,
                                    subj_rwd = subj_rwd.true,
                                    learn = learn.normal)

normal_free_prior_mean_fixed_sfmx = model(name = 'normal_free_prior_mean_fixed_sfmx',
                                          sim_data_setup = sim_data_setup.normal_free_prior_mean, 
                                          a_prob = a_prob.fixed_softmax,
                                          subj_rwd = subj_rwd.true,
                                          learn = learn.normal)

bernoulli_sfmx = model(name = 'bernoulli_sfmx',
                       sim_data_setup = sim_data_setup.bernoulli,
                       a_prob = a_prob.softmax,
                       subj_rwd = subj_rwd.true,
                       learn = learn.bernoulli)

changepoint_bernoulli_sfmx = model(name = 'changepoint_bernoulli_sfmx',
                                   sim_data_setup = sim_data_setup.bernoulli,
                                   a_prob = a_prob.softmax,
                                   subj_rwd = subj_rwd.true,
                                   learn = learn.changepoint_bernoulli)

cluster_bernoulli_sfmx = model(name = 'cluster_bernoulli_sfmx',
                               sim_data_setup = sim_data_setup.cluster_bernoulli,
                               a_prob = a_prob.softmax,
                               subj_rwd = subj_rwd.true,
                               learn = learn.cluster_bernoulli)

adams_mackay_nch_sfmx = model(name = 'adams_mackay_nch_sfmx',
                              sim_data_setup = sim_data_setup.adams_mackay_normal_constant_hazard,
                              a_prob = a_prob.softmax,
                              subj_rwd = subj_rwd.true,
                              learn = learn.adams_mackay_normal)

adams_mackay_nfpmch_sfmx = model(name = 'adams_mackay_nfpmch_sfmx',
                                 sim_data_setup = sim_data_setup.adams_mackay_normal_free_prior_mean_constant_hazard,
                                 a_prob = a_prob.softmax,
                                 subj_rwd = subj_rwd.true,
                                 learn = learn.adams_mackay_normal)

adams_mackay_nih_sfmx = model(name = 'adams_mackay_nih_sfmx',
                              sim_data_setup = sim_data_setup.adams_mackay_normal_increasing_hazard,
                              a_prob = a_prob.softmax,
                              subj_rwd = subj_rwd.true,
                              learn = learn.adams_mackay_normal)

adams_mackay_nihs0_sfmx = model(name = 'adams_mackay_nihs0_sfmx',
                              sim_data_setup = sim_data_setup.adams_mackay_normal_increasing_hazard_start0,
                              a_prob = a_prob.softmax,
                              subj_rwd = subj_rwd.true,
                              learn = learn.adams_mackay_normal)

adams_mackay_nih_eg = model(name = 'adams_mackay_nih_eg',
                            sim_data_setup = sim_data_setup.adams_mackay_normal_increasing_hazard,
                            a_prob = a_prob.epsilon_greedy,
                            subj_rwd = subj_rwd.true,
                            learn = learn.adams_mackay_normal)

adams_mackay_nfpmih_sfmx = model(name = 'adams_mackay_nfpmih_sfmx',
                                 sim_data_setup = sim_data_setup.adams_mackay_normal_free_prior_mean_increasing_hazard,
                                 a_prob = a_prob.softmax,
                                 subj_rwd = subj_rwd.true,
                                 learn = learn.adams_mackay_normal)

adams_mackay_bch_sfmx = model(name = 'adams_mackay_bch_sfmx',
                              sim_data_setup = sim_data_setup.adams_mackay_bernoulli_constant_hazard,
                              a_prob = a_prob.softmax,
                              subj_rwd = subj_rwd.true,
                              learn = learn.adams_mackay_bernoulli)

adams_mackay_bih_sfmx = model(name = 'adams_mackay_bih_sfmx',
                              sim_data_setup = sim_data_setup.adams_mackay_bernoulli_increasing_hazard,
                              a_prob = a_prob.softmax,
                              subj_rwd = subj_rwd.true,
                              learn = learn.adams_mackay_bernoulli)

expectancy_valence_sfmx = model(name = 'expectancy_valence_sfmx', 
                                sim_data_setup = sim_data_setup.q_only, 
                                a_prob = a_prob.softmax,
                                subj_rwd = subj_rwd.weighted,
                                learn = learn.delta_rule)