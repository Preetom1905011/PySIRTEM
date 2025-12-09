import matplotlib.pyplot as plt
import pickle
import numpy as np


# --- Plot Results ---
def plot_results(baseline_phi_s1, baseline_phi_a1, baseline_infected, counterfactuals, t_span):
    n_weeks = len(baseline_phi_s1)
    weeks = np.arange(n_weeks)
    times = np.arange(t_span[0], t_span[1])

    plt.figure(figsize=(16, 12))
    
    # phi_s1
    plt.subplot(3, 1, 1)
    plt.plot(weeks, baseline_phi_s1, label='Baseline phi_s1', color='black', linewidth=2)
    for i, cf in enumerate(counterfactuals):
        plt.plot(weeks, cf['phi_s1'], label=f'CF {i+1} phi_s1', alpha=0.7)
    plt.title("phi_s1 (weekly rates)")
    plt.xlabel("Week")
    plt.ylabel("phi_s1")
    plt.legend()
    
    # phi_a1
    plt.subplot(3, 1, 2)
    plt.plot(weeks, baseline_phi_a1, label='Baseline phi_a1', color='black', linewidth=2)
    for i, cf in enumerate(counterfactuals):
        plt.plot(weeks, cf['phi_a1'], label=f'CF {i+1} phi_a1', alpha=0.7)
    plt.title("phi_a1 (weekly rates)")
    plt.xlabel("Week")
    plt.ylabel("phi_a1")
    plt.legend()
    
    # Infected curves
    plt.subplot(3, 1, 3)
    plt.plot(times, baseline_infected, label='Baseline infected', color='black', linewidth=2)
    for i, cf in enumerate(counterfactuals):
        plt.plot(times, cf['infected'], label=f'CF {i+1} infected', alpha=0.7)
    plt.title("Infected over time")
    plt.xlabel("Days")
    plt.ylabel("Infected individuals")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("optuna_CFE_pysirtem2.png")


# --- Extract N best counterfactuals ---
[counterfactuals, study_trials] = pickle.load(open("CFE_sols2.p", "rb"))

N = 4
top_trials = sorted(study_trials, key=lambda x: x.value)[:N]

counterfactuals = []
for trial in top_trials:
    phi_s1_list = [trial.params[f"phi_s1_{i}"] for i in range(N_WEEKS)]
    phi_a1_list = [trial.params[f"phi_a1_{i}"] for i in range(N_WEEKS)]
    sol = runModel(init_cond, T_delay, params, t_span[1], def_inf_rate, phi_s1_list, def_phi_s2, phi_a1_list, def_phi_a2, def_g_rate) 
    I_sim = list(params['r'] * (np.array(sol)[:, M_states['PS']] + np.array(sol)[:, M_states['PA']] + np.array(sol)[:, M_states['IA']] + np.array(sol)[:, M_states['ATN']]) + np.array(sol)[:, M_states['IS']] + np.array(sol)[:, M_states['STN']])
    
    counterfactuals.append({
        "phi_s1": phi_s1_list,
        "phi_a1": phi_a1_list,
        "infected": I_sim
    })



plot_results(baseline_phi_s1, baseline_phi_a1, baseline_I, counterfactuals, t_span)
pickle.dump([counterfactuals, study.trials], open("CFE_sols2.p", "wb"))
exit()
# all_results = pickle.load(open("est_obs_pf_autoreg_all.p", "rb"))
state = "AZ"
all_results = pickle.load(open("est_results_pf_"+state+"_10.p", "rb"))
print([t[-1] for t in all_results])
exit() 

# pw_FL, pw_AZ, pw_MN, pw_WI

all_error_inf = []
all_error_phi = []
all_error_g = []
all_error_pos = []
all_error_neg = []

# for iter in range(1):
#     [rate_estimates, rate_generated,map_rate_estimates, case_estimates, map_case_estimates, observations] = all_results[iter]
    
#     timepoints = len(observations)-1
#     [rate_estimates, rate_generated,map_rate_estimates, case_estimates, map_case_estimates, observations] = [rate_estimates[:timepoints,], rate_generated[:timepoints,],map_rate_estimates[:timepoints,], case_estimates[:timepoints,], map_case_estimates[:timepoints,], observations[:timepoints,]]
    
#     print(case_estimates.shape, rate_estimates.shape)
#     print(rate_estimates[:,0])
#     print(rate_estimates[:,1])
#     print(rate_estimates[:,2])
#     print(case_estimates[:,0])
#     print(case_estimates[:,1])

#     mean_inf_est = [sum(list(map_rate_estimates[:, 0])[i:i+7]) / len(list(map_rate_estimates[:, 0])[i:i+7]) for i in range(0, len(list(map_rate_estimates[:, 0])), 7) if list(map_rate_estimates[:, 0])[i:i+7]]
#     mean_phi_est = [sum(list(map_rate_estimates[:, 1])[i:i+7]) / len(list(map_rate_estimates[:, 1])[i:i+7]) for i in range(0, len(list(map_rate_estimates[:, 1])), 7) if list(map_rate_estimates[:, 1])[i:i+7]]
#     mean_g_est = [sum(list(map_rate_estimates[:, 2])[i:i+7]) / len(list(map_rate_estimates[:, 2])[i:i+7]) for i in range(0, len(list(map_rate_estimates[:, 2])), 7) if list(map_rate_estimates[:, 2])[i:i+7]]
#     # exit()
#     # Plot the true vs estimated positions over time
#     x_pts = np.arange(1, len(mean_inf_est)+1, 1)
#     plt.plot(x_pts, mean_inf_est, label="Mean Weekly Inf Rate", color='red')
#     plt.plot(x_pts, mean_phi_est, label="Mean Weekly phi Rate", color='blue')
#     plt.plot(x_pts, mean_g_est, label="Mean Weekly g Rate", color='orange')
#     plt.xlabel('Weeks')
#     plt.ylabel('Rates')
#     plt.legend()
#     plt.savefig("pf_rates_est_pw_"+state+"_weekly.png", dpi=300)
#     plt.show()
#     plt.clf()
#     # continue

#     # Plot the true vs estimated positions over time
#     x_pts = np.arange(1, timepoints+1, 1)
#     plt.plot(x_pts, map_case_estimates[:, 0], label='Estimated')
#     plt.plot(x_pts, observations[:, 0], label='Observed')
#     plt.title('True vs Estimated Positive Over Time')
#     plt.xlabel('Days')
#     plt.ylabel('Numbers')
#     plt.legend()
#     plt.savefig("pf_cmp_case_pos_pw_"+state+".png", dpi=300)
#     plt.show()
#     plt.clf()


#     plt.plot(x_pts, map_case_estimates[:, 1], label='Estimated')
#     plt.plot(x_pts, observations[:, 1], label='Observed')
#     plt.title('True vs Estimated negative Over Time')
#     plt.xlabel('Days')
#     plt.ylabel('Numbers')
#     plt.legend()
#     plt.savefig("pf_cmp_case_neg_pw_"+state+".png", dpi=300)
#     plt.show()
#     plt.clf()


#     # # Plot the true vs estimated velocities over time
#     # plt.plot(x_pts, map_rate_estimates[:, 0], label='Estimated')
#     # plt.title('Estimated Inf rates Over Time')
#     # plt.xlabel('Days')
#     # plt.ylabel('Rates')
#     # plt.legend()
#     # # plt.savefig("pf_cmp_rate_inf_"+state+".png", dpi=300)
#     # plt.show()
#     # # plt.clf()

#     # plt.plot(x_pts, map_rate_estimates[:, 1], label='Estimated')
#     # plt.title('Estimated Phi rates Over Time')
#     # plt.xlabel('Days')
#     # plt.ylabel('Rates')
#     # plt.legend()
#     # # plt.savefig("pf_cmp_rate_phi_"+state+".png", dpi=300)
#     # plt.show()
#     # # plt.clf()

#     # plt.plot(x_pts, map_rate_estimates[:, 2], label='Estimated')
#     # plt.title('Estimated g rates Over Time')
#     # plt.xlabel('Days')
#     # plt.ylabel('Rates')
#     # plt.legend()
#     # # plt.savefig("pf_cmp_rate_g_"+state+".png", dpi=300)
#     # plt.show()
#     # # plt.clf()


#     # error plots ==> (est - obs) / obs
#     start = 50
#     x_pts = [i for i in range(start, start + observations[start:, 0].shape[0], 1)]
#     plt.plot(x_pts, (map_case_estimates[start:, 0] - observations[start:, 0] ) / observations[start:, 0], label='Positive rel. error', color='green')
#     plt.plot(x_pts, (map_case_estimates[start:, 1] - observations[start:, 1] ) / observations[start:, 1], label='Negative rel. error', color='red')
#     plt.title('relative error (Cases) Over Time')
#     plt.xlabel('Days')
#     plt.ylabel('Error')
#     plt.legend()
#     plt.savefig("pf_rel_error_cases_"+state+".png", dpi=300)
#     plt.show()
#     plt.clf()

    