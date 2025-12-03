import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import re
import os
from pathlib import Path

folder_path = Path("Cases/test/data/AC")
output_path = Path("Cases/test/data/trigcoef")
os.makedirs(output_path)

for file_path in folder_path.iterdir():
    if (
        file_path.is_file()
        and file_path.suffix == ".csv"
        and file_path.name.startswith("case")
        and not file_path.name.endswith("-pg.csv")
    ):
        match = re.match(r"case(.+)", file_path.stem)
        if match:
            case = match.group(1)
            print(f"Processing case {case} from file {file_path.name}")
        else:
            print(f"No case name found in {file_path.name}")
            continue

        filename = os.path.basename(file_path)
        file = output_path.joinpath(filename)
        df_out = pd.DataFrame(columns=["part","theta_from","theta_to","intercept"])

        df = pd.read_csv(file_path)
        df = df[df["Status"] != "Infeasible"]
        # Example data (replace with your own)
        # a = np.array([1, 2, 3, 4, 5])
        # b = np.array([2, 1, 0, 1, 2])
        # f = np.array([5, 7, 6, 10, 13])  # output values

        a = df["theta_from"]
        b = df["theta_to"]
        f = np.cos(df["theta_from"] - df["theta_to"])
        f2 = np.sin(df["theta_from"] - df["theta_to"])
        # Stack input variables as columns
        X = np.column_stack((a, b))

        # Fit model
        model = LinearRegression().fit(X, f)
        model2 = LinearRegression().fit(X, f2)

        # Extract coefficients
        A = model.coef_[0]   # coefficient for a
        B = model.coef_[1]   # coefficient for b
        C = model.intercept_ # constant term
        df_out.loc[len(df_out)] = ["cos",A,B,C]
        # print("A =", A)
        # print("B =", B)
        # print("C =", C)

        A2 = model2.coef_[0]   # coefficient for a
        B2 = model2.coef_[1]   # coefficient for b
        C2 = model2.intercept_ # constant term
        df_out.loc[len(df_out)] = ["sin",A2,B2,C2]

        # print("A2 =", A2)
        # print("B2 =", B2)
        # print("C2 =", C2)
        df_out.to_csv(file)