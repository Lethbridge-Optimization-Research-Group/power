import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Input files
#case_file = "Cases/test/data/AC/case5.csv"
#results_file = "Cases/test/data/branch_specific/scenario_results_case5.csv"

import os
import glob

# Paths to folders
case_folder = "Cases/test/data/AC/"
results_folder = "Cases/test/data/branch_specific/"

# Get all case files
case_files = [f for f in glob.glob(os.path.join(case_folder, "case*.csv")) if "-pg" not in f]

# For each case file, find the corresponding results file
for case_file in case_files:
    print(case_file)
    # Extract the number X from 'caseX.csv'
    base_name = os.path.basename(case_file)       # e.g., 'case5.csv'
    case_number = ''.join(filter(str.isdigit, base_name))  # e.g., '5'

    # Construct the expected results filename
    results_file = os.path.join(results_folder, f"scenario_results_case{case_number}.csv")

    # Check if the results file exists
    if os.path.exists(results_file):
        # Load data
        df_case = pd.read_csv(case_file)
        df_results = pd.read_csv(results_file)

        # Add line_id to case file
        df_case["line_id"] = df_case["Bus_from"].astype(str) + "_" + df_case["Bus_to"].astype(str)

        feature_cols = ['vi2', 'vj2', 'theta_from', 'theta_to']
        target_cols  = ['y_cos_f','y_sin_f']

        # Map features to actual result column suffixes
        feature_col_map = {
            'vi2': 'w1_vm_from',
            'vj2': 'w2_vm_to',
            'theta_from': 'w3_theta_i',
            'theta_to': 'w4_theta_j'
        }

        # Thresholds for detecting extreme coefficients
        COEF_TOO_LARGE = 200
        COEF_TOO_SMALL = -200

        def is_extreme(row):
            """Check if any coefficient in this row is extreme."""
            for target in target_cols:
                for feature, col_suffix in feature_col_map.items():
                    coef = row.get(f'{target}_{col_suffix}', 0)
                    if coef > COEF_TOO_LARGE or (coef < COEF_TOO_SMALL and coef != 0):
                        return True
            return False

        def refit_line(line_id, df_case, n_samples=100000):
            df_line = df_case[df_case["line_id"] == line_id].copy()
            if df_line.empty:
                return None
            
            # Compute min and max for each feature
            ranges = {
                'volatge_magnitude_from': (df_line['volatge_magnitude_from'].min(), df_line['volatge_magnitude_from'].max()),
                'volatge_magnitude_to': (df_line['volatge_magnitude_to'].min(), df_line['volatge_magnitude_to'].max()),
                'theta_from': (df_line['theta_from'].min(), df_line['theta_from'].max()),
                'theta_to': (df_line['theta_to'].min(), df_line['theta_to'].max())
            }

            # Generate synthetic samples
            df_synth = pd.DataFrame({
                col: np.random.uniform(low, high, n_samples)
                for col, (low, high) in ranges.items()
            })

            # Compute squared voltages
            df_synth['vi2'] = df_synth['volatge_magnitude_from'] ** 2
            df_synth['vj2'] = df_synth['volatge_magnitude_to'] ** 2

            # Physics-based targets
            df_synth['cos_theta_diff_from'] = np.cos(df_synth['theta_from'] - df_synth['theta_to'])
            df_synth['sin_theta_diff_from'] = np.sin(df_synth['theta_from'] - df_synth['theta_to'])
            df_synth['y_cos_f'] = np.sqrt(df_synth['vi2'] * df_synth['vj2']) * df_synth['cos_theta_diff_from']
            df_synth['y_sin_f'] = np.sqrt(df_synth['vi2'] * df_synth['vj2']) * df_synth['sin_theta_diff_from']

            results = {"line_id": line_id}

            # Features for regression
            feature_cols = ['vi2', 'vj2', 'theta_from', 'theta_to']
            X = df_synth[feature_cols].values

            # Fit linear regression for each target
            for target in ['y_cos_f','y_sin_f']:
                y = df_synth[target].values
                model = LinearRegression()
                model.fit(X, y)

                # Map to result column names
                results[f'{target}_w1_vm_from'] = model.coef_[0]
                results[f'{target}_w2_vm_to'] = model.coef_[1]
                results[f'{target}_w3_theta_i'] = model.coef_[2]
                results[f'{target}_w4_theta_j'] = model.coef_[3]
                results[f'{target}_bias'] = model.intercept_

            return results


        # Refit only extreme rows
        updated_rows = []
        for i, row in df_results.iterrows():
            if is_extreme(row):
                line_id = row['line_id']
                new_res = refit_line(line_id, df_case)
                if new_res:
                    for k, v in new_res.items():
                        df_results.at[i, k] = v
                    updated_rows.append(line_id)

        # Save updated file
        df_results.to_csv(results_file, index=False)

        print(f"Updated {len(updated_rows)} line_ids with resampled coefficients.")

    else:
        print(f"Warning: Results file not found for {case_file}")