import os
import glob

# Paths to folders
case_folder = "Cases/test/data/AC/"
results_folder = "Cases/test/data/branch_specific/"

# Get all case files, ignoring any with '-pg' in the name
case_files = [f for f in glob.glob(os.path.join(case_folder, "case*.csv")) if "-pg" not in f]

# Pair each case file with its corresponding results file
file_pairs = []
for case_file in case_files:
    # Extract the number X from 'caseX.csv'
    base_name = os.path.basename(case_file)  # e.g., 'case5.csv'
    case_number = ''.join(filter(str.isdigit, base_name))  # e.g., '5'

    # Construct the expected results filename
    results_file = os.path.join(results_folder, f"scenario_results_case{case_number}.csv")

    # Check if the results file exists
    if os.path.exists(results_file):
        file_pairs.append((case_file, results_file))
    else:
        print(f"Warning: Results file not found for {case_file}")

# file_pairs now contains only the valid pairs
print(file_pairs)
