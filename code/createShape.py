import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools
import csv
from Parameters_pf import *
from DDE_pf import *


np.random.seed(42)

def generate_values(start, end, num_of_vals, shape):
    x = np.linspace(0, 1, num_of_vals)
    
    if shape == "increasing":
        y = np.sort(np.random.rand(num_of_vals))
    elif shape == "decreasing":
        y = np.sort(np.random.rand(num_of_vals))[::-1]
    elif shape == "one peak":
        y = np.exp(-((x - 0.5) * 5) ** 2)  # Gaussian-like shape
    elif shape == "two peak":
        y = np.exp(-((x - 0.3) * 5) ** 2) + np.exp(-((x - 0.7) * 5) ** 2) - 0.5 * np.exp(-((x - 0.5) * 8) ** 2)  # Deeper dip between peaks
    else:
        raise ValueError("Invalid shape. Choose from 'increasing', 'decreasing', 'one peak', or 'two peak'.")
    
    y = start + (end - start) * (y - np.min(y)) / (np.max(y) - np.min(y))  # Scale to [start, end]
    return list(y)

# plotting the shapes to verify
def plot_values(values, filename="shape", title="Plot of Generated Values"):
    plt.figure(figsize=(8, 4))
    plt.plot(values, marker='o', linestyle='-', color='b')
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(title)
    plt.grid()
    plt.savefig(filename+".png", dpi=300)
    plt.clf()

# # Example usage:
# values = generate_values(0.0001, 0.004, 40, "increasing")
# plot_values(values, "One Peak Shape")
# exit()

def calculate_daily_counts(data):
    len_x = len(data)
    
    sim_daily_pos = [ (data[t][M_states['F_AT_1']] + data[t][M_states['F_ST_1']])*(params['True_P_1']) + (data[t][M_states['F_AT_2']] + data[t][M_states['F_ST_2']])*(params['True_P_2']) + (1-params["True_N_1"]) * (data[t][M_states['F_NT_1']] + data[t][M_states['F_AT_3']] + data[t][M_states['F_FT_1']] + data[t][M_states['F_GT1']] + data[t][M_states['F_ST_4']] + data[t][M_states['F_ST_3']] + data[t][M_states['F_FT_3']]) +(1 - params["True_N_2"]) * (data[t][M_states['F_NT_2']] + data[t][M_states['F_AT_4']] + data[t][M_states['F_FT_2']] + data[t][M_states['F_GT2']] ) for t in range(len_x)]

    
    sim_daily_neg = [ (data[t][M_states['F_AT_1']] + data[t][M_states['F_ST_1']])*(1 - params['True_P_1']) + (data[t][M_states['F_AT_2']] + data[t][M_states['F_ST_2']])*(1 - params['True_P_2']) + params["True_N_1"] * (data[t][M_states['F_NT_1']] + data[t][M_states['F_AT_3']] + data[t][M_states['F_FT_1']] + data[t][M_states['F_GT1']] + data[t][M_states['F_ST_4']] + data[t][M_states['F_ST_3']] + data[t][M_states['F_FT_3']]) + params["True_N_2"] * (data[t][M_states['F_NT_2']] + data[t][M_states['F_AT_4']] + data[t][M_states['F_FT_2']] + data[t][M_states['F_GT2']] ) for t in range(len_x)]

    return sim_daily_pos, sim_daily_neg


def function_to_evaluate(rates, horizon):
    
    initial_condition = init_cond
    res = [init_cond]
    for t in range(horizon):
        inf_rate, phi_s, g_rate = [rates[0][t]], [rates[1][t]], [rates[2][t]]
        # T_delay["lambda_q"] = q_days[t]
        
        yy = runModel(initial_condition, T_delay, params, 2, inf_rate, phi_s, g_rate)

        next_init_cond = []
        for s in M_states.keys():
            next_init_cond.append(yy[1][M_states[s]])
        
        initial_condition = next_init_cond
        res.append(next_init_cond)

    
    positives, negatives = calculate_daily_counts(res)
    
    return np.max(positives), np.max(negatives)


def create_doe():
    
    shapes = ["increasing", "decreasing", "one peak"]
    factors = ["infRate", "testRate", "gRate"]
    num_of_vals = 100
    
    # Define unique start-end pairs for each factor (low, medium, high)
    start_end_dict = {
        "infRate": [(0.05, 0.99), (0.05, 0.7), (0.05, 0.5)],  # High, Medium, Low
        "testRate": [(0.05, 0.4), (0.05, 0.3), (0.05, 0.2)], # High, Medium, Low
        "gRate": [(0.001, 0.05), (0.001, 0.03), (0.001, 0.02)] # High, Medium, Low
    }
    
    peak_levels = ["high", "medium", "low"]
    factor_levels = {"infRate": [], "testRate": [], "gRate": []}
    results = {}
    
    # Generate values and store in results
    for factor in factors:
        for shape in shapes:
            for idx, (start, end) in enumerate(start_end_dict[factor]):
                key = f"{shape} ({factor}, {peak_levels[idx]})"
                values = generate_values(start, end, num_of_vals, shape)
                results[key] = [(start, end), values]
                factor_levels[factor].append(key)
    
    # Generate all possible DoE combinations where:
    # - Index 0 is always from infRate
    # - Index 1 is always from testRate
    # - Index 2 is always from gRate
    all_combinations = list(itertools.product(factor_levels["infRate"], 
                                              factor_levels["testRate"], 
                                              factor_levels["gRate"]))
    
    # print(all_combinations)

    # Filter combinations to ensure no two rates have the same shape
    filtered_combinations = []
    for combination in all_combinations:
        # Extract shapes from keys
        shape_1 = combination[0].split()[0]  # infRate shape
        shape_2 = combination[1].split()[0]  # testRate shape
        shape_3 = combination[2].split()[0]  # gRate shape
        
        # Ensure all three shapes are unique
        if len(set([shape_1, shape_2, shape_3])) == 3:
            filtered_combinations.append(combination)

    
    doe_matrix = []
    output_values_max = []
    
    idx = 0
    horizon = num_of_vals
    # results = pickle.load(open("doe_results_4shape_100.p", "rb"))
    # Evaluate function for each combination
    for combination in all_combinations:
        if idx <= 562:
            idx += 1
            continue
        x1_values = results[combination[0]][1]  # infRate values
        x2_values = results[combination[1]][1]  # testRate values
        x3_values = results[combination[2]][1]  # gRate values
        pos, neg = function_to_evaluate([x1_values, x2_values, x3_values], horizon)
        doe_matrix.append((combination[0], combination[1], combination[2], pos, neg))
        output_values_max.append((pos, neg))
    
        with open("doe_results_3shape_100_non_unique.p", "wb") as f:
            pickle.dump(results, f)
    
        
        with open("doe_matrix_3shape_100_non_unique.csv", "a", newline="") as f:
            writer = csv.writer(f)
            # writer.writerow(factors + ["Positive Max", "Negative Max"])
            # writer.writerows(doe_matrix)
            writer.writerow((combination[0], combination[1], combination[2], pos, neg))

        idx += 1

        print(idx, len(all_combinations))
        # exit()
    
    return doe_matrix, output_values_max

# Example usage:
doe_matrix, outputs = create_doe()
# print("Design of Experiment Matrix:", doe_matrix)
# print("Function Outputs:", outputs)


# results = pickle.load(open("doe_results_4shape_100.p", "rb"))

# print(results.keys())

# setup = ['decreasing (infRate, high)', 'increasing (testRate, medium)', 'one peak (gRate, high)']

# sim_daily_pos, sim_daily_neg = function_to_evaluate([results[setup[0]][1], results[setup[1]][1], results[setup[2]][1]])

# plot_values(sim_daily_pos, "positive", "positive")
# plot_values(sim_daily_neg, "negative", "negative")

# pickle.dump([sim_daily_pos, sim_daily_neg], open("doe_test_indv.p", "wb"))

# print(results[setup[0]][1])
# plt.plot(np.arange(len(results[setup[0]][1])), results[setup[0]][1], label="inf rate")
# plt.plot(np.arange(len(results[setup[0]][1])), results[setup[1]][1], label="test rate")
# plt.plot(np.arange(len(results[setup[0]][1])), results[setup[2]][1], label="g rate")

# plt.legend()
# plt.savefig("rates.png", dpi=300)



