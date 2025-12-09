import numpy as np
import matplotlib.pyplot as plt
from pfilter import ParticleFilter, gaussian_noise, squared_error, independent_sample
from scipy.stats import norm, gamma, uniform, expon
import random
from Parameters_pf import *
from DDE_pf import *
from scipy.ndimage import gaussian_filter

def right_skewed_exponential(n):
    scale = 3
    value = 14 - expon(scale=scale).rvs(size=n)  
    return np.clip(value, 2, 14) 


# Prior function (inf_rate, phi_rate, g_rate, lambda_q)
prior_fn = independent_sample([uniform(loc=0, scale=1).rvs, 
            uniform(loc=0, scale=0.5).rvs, 
            uniform(loc=0, scale=0.05).rvs,
            right_skewed_exponential
            # uniform(loc=0, scale=14).rvs
            ])

def mean_squared_error_similarity(x, y):
    parts = x.shape[0]
    mse = np.zeros((parts))

    for p in range(parts):
        mse[p] = 0.5 * (np.sqrt((x[p, 0] - y[0, 0]) ** 2 )) / y[0, 0] + 0.5 * (np.sqrt((x[p, 1] - y[0, 1]) ** 2 )) / y[0, 1]
        

    return 1 / mse

# Observation function
def observe_fn(states):
    global initial_condition
    duration = 2
    obs = []
    st_idx = 0
    for state in states:
        print("Particle: ", st_idx+1)
        inf_rate, phi_s, g_rate = state[:1], state[1:2], state[2:3]
        T_delay["lambda_q"] = state[3:]
        
        yy = runModel(initial_condition[st_idx], T_delay, params, duration, inf_rate, phi_s, g_rate)

        next_init_cond = []
        for s in M_states.keys():
            next_init_cond.append(yy[1][M_states[s]])
        
        initial_condition[st_idx] = next_init_cond
        st_idx += 1

        # =============== Calculate the daily pos and daily negative from the SIRTEM model sim ====================

        # # Positive_N_PS(t) = (F_ST_1(t))*0.7;                Negative_N_PS(t) = 0.3*(F_ST_1(t)); 
        # # Positive_N_FT1_1(t) = 0.05*(t)*SF(t)*phi_s_1;      Negative_N_FT1_1(t) = 0.95*SF(t)*phi_s_1; 
        # # Positive_N_GT1(t) =  (F_GT1(t))*0.05;              Negative_N_GT1(t) =  0.95*F_GT1(t);    
        # est_daily_pos = [ yy[t][states['F_ST_1']]*0.7 + 0.05*yy[t][states['SF']]*phi_s[int(t/7)] + yy[t][states['F_GT1']]*0.05 for t in range(self.duration)]
        # est_daily_neg = [ yy[t][states['F_ST_1']]*0.3 + 0.95*yy[t][states['SF']]*phi_s[int(t/7)] + yy[t][states['F_GT1']]*0.95 for t in range(self.duration)]

        est_daily_pos = [ (yy[t][M_states['F_AT_1']] + yy[t][M_states['F_ST_1']])*(params['True_P_1']) +
                            (yy[t][M_states['F_AT_2']] + yy[t][M_states['F_ST_2']])*(params['True_P_2']) +
                            (1-params["True_N_1"]) * (yy[t][M_states['F_NT_1']] + yy[t][M_states['F_AT_3']] + 
                                                    yy[t][M_states['F_FT_1']] + yy[t][M_states['F_GT1']] + 
                                                    yy[t][M_states['F_ST_4']] + yy[t][M_states['F_ST_3']] + 
                                                    yy[t][M_states['F_FT_3']]) +
                            (1 - params["True_N_2"]) * (yy[t][M_states['F_NT_2']] + yy[t][M_states['F_AT_4']] + 
                                                    yy[t][M_states['F_FT_2']] + yy[t][M_states['F_GT2']] ) for t in range(duration)]

        
        est_daily_neg = [ (yy[t][M_states['F_AT_1']] + yy[t][M_states['F_ST_1']])*(1 - params['True_P_1']) +
                            (yy[t][M_states['F_AT_2']] + yy[t][M_states['F_ST_2']])*(1 - params['True_P_2']) +
                            params["True_N_1"] * (yy[t][M_states['F_NT_1']] + yy[t][M_states['F_AT_3']] + 
                                                    yy[t][M_states['F_FT_1']] + yy[t][M_states['F_GT1']] + 
                                                    yy[t][M_states['F_ST_4']] + yy[t][M_states['F_ST_3']] + 
                                                    yy[t][M_states['F_FT_3']]) +
                            params["True_N_2"] * (yy[t][M_states['F_NT_2']] + yy[t][M_states['F_AT_4']] + 
                                                    yy[t][M_states['F_FT_2']] + yy[t][M_states['F_GT2']] ) for t in range(duration)]
    
        obs_daily = [est_daily_pos[1], est_daily_neg[1]]

        obs.append(obs_daily)
    return np.array(obs)

# Resampling function
def resample_fn(weights):
    # This function should take an array of weights and return an array of indices
    # For example, you might use np.random.choice to perform weighted sampling:
    return np.random.choice(np.arange(len(weights)), size=len(weights), p=weights)


# Dynamics function
def dynamics_fn_autoreg_ord1(states):
    xp = states
    for col in range(xp.shape[1]):
        for row in range(xp.shape[0]):
            
            # for quarantine days
            if col == 3:
                new = right_skewed_exponential(1)[0]
                err = np.random.normal(loc=0, scale=0.001, size=1)[0]
                lower_bnd, upper_bnd = 0, 14
                if new + err < lower_bnd:
                    while new + err < lower_bnd:
                        new = right_skewed_exponential(1)
                        err = np.random.normal(loc=0, scale=scale, size=1)[0]
                    
                elif new + err > upper_bnd:
                    while new + err > upper_bnd:
                        new = right_skewed_exponential(1)
                        err = np.random.normal(loc=0, scale=scale, size=1)[0]
            else:
                new = random.uniform(0.25, 1) * states[row, col] 
                scale = 0.001 if col == 2 else 0.001
                upper_bnd = 0.1 if col == 2 else (min(0.5, xp[row, col] + 0.15)  if col == 1 else 1)
                lower_bnd = 0 if col == 2 else (max(0, xp[row, col] - 0.1) if col == 1 else 0)
                err = np.random.normal(loc=0, scale=scale, size=1)[0]

                if new + err < lower_bnd:
                    while new + err < lower_bnd:
                        new = random.uniform(0.25, 1) * states[row, col]
                        err = np.random.normal(loc=0, scale=scale, size=1)[0]
                    
                elif new + err > upper_bnd:
                    while new + err > upper_bnd:
                        new = random.uniform(0.25, 1) * states[row, col]
                        err = np.random.normal(loc=0, scale=scale, size=1)[0]
        
        
            xp[row, col] = new + err
    return xp

# Dynamics function
def dynamics_fn_autoreg_ord2(states):
    global time_counter, prev2_states

    if time_counter == 1:
        time_counter += 1
        prev2_states = states
        return dynamics_fn_autoreg_ord1(states)
    else:
        time_counter += 1
        xp = states

        # piecewise, only change rate weekly
        if time_counter % 7 == 0:
            for col in range(xp.shape[1]):
                for row in range(xp.shape[0]):
                    # for quarantine days
                    if col == 3:
                        new = right_skewed_exponential(1)[0]
                        err = np.random.normal(loc=0, scale=0.001, size=1)[0]
                        lower_bnd, upper_bnd = 0, 14
                        if new + err < lower_bnd:
                            while new + err < lower_bnd:
                                new = right_skewed_exponential(1)
                                err = np.random.normal(loc=0, scale=scale, size=1)[0]
                            
                        elif new + err > upper_bnd:
                            while new + err > upper_bnd:
                                new = right_skewed_exponential(1)
                                err = np.random.normal(loc=0, scale=scale, size=1)[0]
                    # for inf_rate, test_rate, g_rate
                    else:
                        new = random.uniform(0.1, 0.8) * states[row, col] + random.uniform(0.1, 0.8) * prev2_states[row, col]
                        scale = 0.001 if col == 2 else 0.001
                        upper_bnd = 0.1 if col == 2 else (min(0.5, xp[row, col] + 0.05)  if col == 1 else min(1, xp[row, col] + 0.2))
                        lower_bnd = 0 if col == 2 else (max(0, xp[row, col] - 0.05) if col == 1 else max(0, xp[row, col] - 0.2))
                        err = np.random.normal(loc=0, scale=scale, size=1)[0]
                        if new + err < lower_bnd:
                            while new + err < lower_bnd:
                                new = random.uniform(0.1, 0.8) * states[row, col] + random.uniform(0.1, 0.8) * prev2_states[row, col]
                                err = np.random.normal(loc=0, scale=scale, size=1)[0]
                            
                        elif new + err > upper_bnd:
                            while new + err > upper_bnd:
                                new = random.uniform(0.1, 0.8) * states[row, col] + random.uniform(0.1, 0.8) * prev2_states[row, col]
                                err = np.random.normal(loc=0, scale=scale, size=1)[0]
                    
                
                    xp[row, col] = new + err

                prev2_states = states
        else:
            xp = prev2_states
    return xp

# Noise function
def noise_fn(states):
    # This function should take an array of states and return a new array with noise added
    return states 

prev1_states = []
prev2_states = []
time_counter = 1

iterations = 1
all_results = []
# all_results = pickle.load(open("est_obs_pf_autoreg_all.p", "rb"))
for iter in range(iterations):
    num_particles = 50
    initial_condition = []
    for n in range(num_particles):
        initial_condition.append(init_cond);

    print("init cond done...")
    # Create the particle filter
    pf = ParticleFilter(
        prior_fn=prior_fn,
        observe_fn=observe_fn,
        # resample_fn=resample_fn,
        n_particles=num_particles,
        dynamics_fn=dynamics_fn_autoreg_ord2,
        # noise_fn=noise_fn,
        weight_fn=mean_squared_error_similarity,
        resample_proportion=0.1,
        n_eff_threshold=0.5
    )
    
    print("Filter initialization done...")

    all_obs = [pickle.load(open("training_data_AZ.p", "rb"))]
    all_obs = np.array(all_obs[iter]).T

    timepoints = len(all_obs)
    observations = all_obs[1:timepoints+1]
    
    # Prepare to store estimates for plotting
    rate_estimates = []
    case_estimates = []

    map_rate_estimates = []
    map_case_estimates = []
    t0 = time.time()
    # Now you can update the filter with your observations
    for observation in observations:
        
        print("Filter update starting...")
        pf.update(observation)
        rate_estimates.append(pf.mean_state)  # Store the estimated state
        case_estimates.append(pf.mean_hypothesis)
        map_rate_estimates.append(pf.map_state)  # Store the estimated state
        map_case_estimates.append(pf.map_hypothesis)
        print("est state:", pf.mean_state)
        print("map state:", pf.map_state)
        print("est obs:", pf.mean_hypothesis)
        print("map obs:", pf.map_hypothesis)
        print("act obs:", observation)
        print("Time:", time_counter)
        print("------------------------")
        # print("w:", pf.original_weights)
        # input('')

        pickle.dump([rate_estimates, map_rate_estimates, case_estimates, map_case_estimates], open("temp_est_results_pf_AZ.p", "wb"))

    # Convert list of estimates to numpy array for easier slicing
    rate_estimates = np.array(rate_estimates)
    case_estimates = np.array(case_estimates)
    map_rate_estimates = np.array(map_rate_estimates)
    map_case_estimates = np.array(map_case_estimates)

    # [inf_rates, phi_rates, g_rates] = pickle.load(open("gen_params_pf.p", "rb"))
    # [coeff_a, coeff_b, inf_rates, phi_rates, g_rates, error_a, error_b] = pickle.load(open("../SOAR/gen_params_daily_ord1_pf2.p", "rb"))
    
    
    rate_generated = np.array([[0 for _ in range(len(observations))], [0 for _ in range(len(observations))], [0 for _ in range(len(observations))]]).T

    all_results.append([rate_estimates, rate_generated, map_rate_estimates, case_estimates, map_case_estimates, observations])

    print("Time Elapsed:", time.time() - t0)
    
pickle.dump(all_results, open("est_results_pf_AZ.p", "wb"))
