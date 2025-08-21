import pandas as pd
import numpy as np
from pathlib import Path
import re
import os
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from pathlib import Path

folder_path = Path("Cases/test/data/AC")

for file_path in folder_path.iterdir():
    if (
        file_path.is_file()
        and file_path.suffix == ".csv"
        and file_path.name.startswith("case")
        and not file_path.name.endswith("-pg.csv")
    ):
        # Extract full case name (e.g., "30Q" from "case30Q.csv")
        match = re.match(r"case(.+)", file_path.stem)
        if match:
            case = match.group(1)
            print(f"Processing case {case} from file {file_path.name}")
        else:
            print(f"No case name found in {file_path.name}")
            continue

        # Read the CSV
        df = pd.read_csv(file_path)
        # process df here

        # create id
        df["line_id"] = df["Bus_from"].astype(str) + "_" + df["Bus_to"].astype(str)
        df_sorted = df.sort_values(by=["Bus_from", "Bus_to"])
        grouped = df_sorted.groupby(["Bus_from", "Bus_to"], sort=False)
        df_final = pd.concat([group for _, group in grouped], axis=0).reset_index(drop=True)
        # df_final


        #  features and targets
        df_final['vi2'] = df_final['volatge_magnitude_from'] ** 2
        df_final['vj2'] = df_final['volatge_magnitude_to'] ** 2

        df_final['cos_theta_diff_from'] = np.cos(df_final['theta_from'] - df_final['theta_to'])
        df_final['cos_theta_diff_to']   = np.cos(df_final['theta_to'] - df_final['theta_from'])
        df_final['sin_theta_diff_from'] = np.sin(df_final['theta_from'] - df_final['theta_to'])
        df_final['sin_theta_diff_to']   = np.sin(df_final['theta_to'] - df_final['theta_from'])

        df_final['y_cos_f'] = df_final['volatge_magnitude_from'] * df_final['volatge_magnitude_to'] * df_final['cos_theta_diff_from']
        df_final['y_cos_t'] = df_final['volatge_magnitude_from'] * df_final['volatge_magnitude_to'] * df_final['cos_theta_diff_to']
        df_final['y_sin_f'] = df_final['volatge_magnitude_from'] * df_final['volatge_magnitude_to'] * df_final['sin_theta_diff_from']
        df_final['y_sin_t'] = df_final['volatge_magnitude_from'] * df_final['volatge_magnitude_to'] * df_final['sin_theta_diff_to']

        feature_cols = ['vi2', 'vj2', 'theta_from', 'theta_to']
        target_cols = ['y_cos_f','y_sin_f']

        results = []
        for line_id in df_final["line_id"].unique():
            df_line = df_final[df_final["line_id"] == line_id]
            row = {"line_id": line_id}

            for target in target_cols:
                X = df_line[feature_cols].values
                y = df_line[target].values

                model = LinearRegression()
                #model = Ridge(alpha=1.0)
                model.fit(X, y)

                row[f'{target}_w1_vm_from'] = model.coef_[0]
                row[f'{target}_w2_vm_to']   = model.coef_[1]
                row[f'{target}_w3_theta_i'] = model.coef_[2]
                row[f'{target}_w4_theta_j'] = model.coef_[3]
                row[f'{target}_bias']       = model.intercept_

            results.append(row)

        df_results = pd.DataFrame(results)
        output_dir = 'Cases/test/data/branch_specific'
        os.makedirs(output_dir, exist_ok=True)  # create directory if it doesn't exist

        df_results.to_csv(f'{output_dir}/scenario_results_case{case}.csv', index=False)

        #df_results.to_csv(f'Cases/test/data/branch_specific/scenario_results_case{case}.csv', index=False)