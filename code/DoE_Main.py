import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import math
import itertools
import pickle
import seaborn as sns


def create_doe_matrix(csv_file, output_file):
    # Load the original DOE matrix
    df = pd.read_csv(csv_file)
    
    # Rename columns for clarity
    df.columns = ["infRate", "testRate", "gRate", "Positive_Max", "Negative_Max"]
    
    # Define the factor levels for each column
    infRate_shapes = ["increasing", "decreasing", "one_peak"]
    infRate_sizes = ["high", "medium", "low"]
    testRate_shapes = ["increasing", "decreasing", "one_peak"]
    testRate_sizes = ["high", "medium", "low"]
    gRate_shapes = ["increasing", "decreasing", "one_peak"]
    gRate_sizes = ["high", "medium", "low"]

    # Create the columns for the new DataFrame
    columns = []
    
    # Order infRate columns first
    for shape in infRate_shapes:
        for size in infRate_sizes:
            columns.append(f"infRate_{shape}_{size}")
    
    # Order testRate columns next
    for shape in testRate_shapes:
        for size in testRate_sizes:
            columns.append(f"testRate_{shape}_{size}")
    
    # Order gRate columns next
    for shape in gRate_shapes:
        for size in gRate_sizes:
            columns.append(f"gRate_{shape}_{size}")
    
    # Add Positive_Max and Negative_Max columns at the end
    columns.append("Positive_Max")
    columns.append("Negative_Max")

    
    # Create the new DataFrame with all -1 initially
    new_df = pd.DataFrame(-1, index=df.index, columns=columns)
    new_df["Positive_Max"] = new_df["Positive_Max"].astype(float)
    new_df["Negative_Max"] = new_df["Negative_Max"].astype(float)

    print(new_df.columns)
    
    # Fill the new DataFrame
    for idx, row in df.iterrows():
        # Parse each factor and update corresponding columns
        def parse_factor(factor_str):
            parts = factor_str.split(" ")
            if parts[0] == "one" or parts[0] == "two":
                shape = parts[0] + "_" + parts[1]  # Handling cases like "one peak", "two peak"
            else:
                shape = parts[0]  # First word is shape (increasing, decreasing, etc.)
            size = parts[-1][:-1]  # Last word before closing parenthesis is size (high, medium, low)
            return shape, size  # Return both shape and size as a tuple

        
        infRate_shape, infRate_size = parse_factor(row["infRate"])
        testRate_shape, testRate_size = parse_factor(row["testRate"])
        gRate_shape, gRate_size = parse_factor(row["gRate"])

        if (infRate_shape not in infRate_shapes) or (infRate_size not in infRate_sizes) or (testRate_shape not in testRate_shapes) or (testRate_size not in testRate_sizes) or (gRate_shape not in gRate_shapes) or (gRate_size not in gRate_sizes):
            continue

        # Set the values for infRate
        new_df.loc[idx, f"infRate_{infRate_shape}_{infRate_size}"] = 1
        for size in infRate_sizes:
            if size != infRate_size:
                new_df.loc[idx, f"infRate_{infRate_shape}_{size}"] = -1

        # Set the values for testRate
        new_df.loc[idx, f"testRate_{testRate_shape}_{testRate_size}"] = 1
        for size in testRate_sizes:
            if size != testRate_size:
                new_df.loc[idx, f"testRate_{testRate_shape}_{size}"] = -1

        # Set the values for gRate
        new_df.loc[idx, f"gRate_{gRate_shape}_{gRate_size}"] = 1
        for size in gRate_sizes:
            if size != gRate_size:
                new_df.loc[idx, f"gRate_{gRate_shape}_{size}"] = -1

        # Add the Positive_Max and Negative_Max values
        new_df.loc[idx, "Positive_Max"] = row["Positive_Max"]
        new_df.loc[idx, "Negative_Max"] = row["Negative_Max"]

    # Save the new DataFrame to a CSV file
    new_df.to_csv(output_file, index=False)
# Example usage
create_doe_matrix("doe_matrix_3shape_100_non_unique.csv", "doe_matrix_3shape_100_non_unique_transformed.csv")

exit()
# =============== ================ ====================
# =============== Interaction Plot ====================
# =============== ================ ====================

def interaction_point_plot(csv_file, result_column, factor_of_interest):
    df = pd.read_csv(csv_file)
    
    # Extract the factors and their levels
    factors = [col for col in df.columns if col != result_column]
    
    # Identify the base part of the factor_of_interest (e.g., "infRate_increasing_high")
    base_factor = factor_of_interest.split('_high')[0]  # For example, "infRate_increasing"
    rate = base_factor.split("_")[0]
    
    # Find all factors that match the size (e.g., *_high)
    matching_factors = [factor for factor in factors if rate not in factor and '_high' in factor]
    
    # Calculate number of rows and columns for subplots
    cols = 4
    rows = math.ceil(len(matching_factors) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(16, 16*rows/4))
    
    # plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Adjust these values to control spacing
    
    # Create interaction plots
    for idx, factor in enumerate(matching_factors):
        row = idx // cols
        col = idx % cols

        ax = axs[row, col] if rows > 1 else axs[col]

        for level in [-1, 1]:
            subset = df[df[factor_of_interest] == level]
            ax.plot(subset[factor].unique(), subset.groupby(factor)[result_column].mean(), 'o-', label=f'{factor_of_interest} = {level}')
        
        ax.set_title(f'{factor_of_interest} x {factor}')
        ax.legend()
        ax.grid(True)

    # Handle any remaining axes (if there's no data to plot in them)
    for idx in range(len(matching_factors), rows*cols):
        row = idx // cols
        col = idx % cols
        axs[row, col].axis('off')

    plt.tight_layout()
    # plt.show()
    plt.savefig("DoE_figures/interaction_plot_"+factor_of_interest+".png", dpi=300)

# Example usage:
rate_var = "infRate"
shape = "increasing"
for level in ["high", "medium", "low"]:
    interaction_point_plot("doe_matrix_4shape_100_non_unique_transformed.csv", "Positive_Max", rate_var+"_"+"_"+level)

