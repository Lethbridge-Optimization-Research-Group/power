import pandas as pd
import re
import os
from pathlib import Path

ac_folder_path = Path("./Cases/test/data/AC/")
approx_folder_path = Path("./Cases/test/data/Approx/")
output_dir = Path("./Cases/test/data/Analysis")
os.makedirs(output_dir, exist_ok=True)

for file_name in ac_folder_path.iterdir():
    if (
        file_name.is_file()
        and file_name.suffix == ".csv"
        and file_name.name.startswith("case")
        and not file_name.name.endswith("-pg.csv")
    ):
        # Extract full case name (e.g., "30Q" from "case30Q.csv")
        match = re.match(r"case(.+)", file_name.stem)
        if match:
            case = match.group(1)
            print(f"Processing case {case} from file {file_name.name}")
        else:
            print(f"No case name found in {file_name.name}")
            continue
        
        #filename, extension = os.path.splitext(file_name)
        ac_file_path = ac_folder_path.joinpath(file_name.name)
        approx_file_path = approx_folder_path.joinpath(file_name.name)
        acdf = pd.read_csv(ac_file_path, header = 0)
        approxdf = pd.read_csv(approx_file_path, header = 0)

        #print(acdf.head(5))
        #print("\n")
        #print(approxdf.head(5))

        subdf = approxdf.iloc[:, 3:]
        numeric_cols = subdf.select_dtypes(include='number').columns
        diff_numeric = acdf[numeric_cols] - approxdf[numeric_cols]

        diff_numeric.insert(0,approxdf.columns[2], approxdf.iloc[:, 2])
        diff_numeric.insert(0,approxdf.columns[1], approxdf.iloc[:, 1])

        #print(diff_numeric)
        diff_numeric.to_csv(f'{output_dir}/difference_case{case}.csv', index=False)

