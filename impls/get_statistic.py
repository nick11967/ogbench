import json
import os
import argparse

import tqdm
import pandas as pd

# --- config ---
CSV_FILE = 'test_results.csv'
# --------------

def print_remove_targets():
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        return
    
    if 'Name' not in df.columns:
        return
    
    for index, row in df.iterrows():
        state = str(row['State']).strip().lower()
        if state != 'finished':
            name = str(row['Name']).strip()
            if not name or name.lower() == 'nan':
                continue
            command = f'rm -rf ./test/SSHIQL/{name}'
            print(command)

def get_statistic():
    idx_map_robot = {'Poi': 0, 'Ant': 1, 'Hum': 2}
    idx_map_size = {'Med': 0, 'Lar': 1, 'Gia': 2}
    idx_map_method = {'HIQL': 0, 'SS+Temp': 1, 'SS+Simi': 2}

    full_stat = [[[[] for _ in range(3)] for _ in range(3)] for _ in range(3)]  # robot, size, method

    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        return
    
    # For each run.
    for index, row in df.iterrows():
        proc_name = str(row['proc_name']).strip()
        parts = proc_name.split('_')
        if len(parts) != 3:
            print("Unexpected proc_name format: ", proc_name)
            continue
        robot_short, size_short, method_short = parts
        if (robot_short not in idx_map_robot) or (size_short not in idx_map_size) or (method_short not in idx_map_method):
            print("Unknown category in proc_name: ", proc_name)
            continue
        robot_idx = idx_map_robot[robot_short]
        size_idx = idx_map_size[size_short]
        method_idx = idx_map_method[method_short]

        try:
            test_csv_path = os.path.join('./test/SSHIQL_TEST', row['Name'], 'test.csv')
            test_df = pd.read_csv(test_csv_path)
            # for each test with different seeds.
            for _, test_row in test_df.iterrows():
                overall_success = float(test_row['overall_success'])
                if not pd.isna(overall_success):
                    full_stat[robot_idx][size_idx][method_idx].append(overall_success * 100)  # convert to percentage
        except:
            print("Failed to process test.csv for: ", row['Name'])
            continue
    
    print(f'        {"HIQL":^12} {"SS+Temp":^12}{"SS+Sim":^10}')
    for robot_key, robot_idx in idx_map_robot.items():
        for size_key, size_idx in idx_map_size.items():
            avg_list = []
            std_list = []
            for method_key, method_idx in idx_map_method.items():
                results = full_stat[robot_idx][size_idx][method_idx]
                if results:
                    count = len(results)
                    avg_success = sum(results) / count
                    avg_list.append(avg_success)
                    std = (sum((x - avg_success) ** 2 for x in results) / count) ** 0.5
                    std_list.append(std)
                    # print(f"{robot_key}_{size_key}_{method_key:7}: {avg_success:.1f} ± {std:.1f} over {count} runs")
                # else:
                    # print(f"{robot_key}_{size_key}_{method_key}: No data available.")
            print(f"{robot_key}-{size_key}: ", end="")
            max_avg = max(avg_list)
            for i, avg in enumerate(avg_list):
                if avg >= max_avg * 0.95:
                    print(f"\033[1m{avg:.1f}\033[0m ± {std_list[i]:<4.1f} ", end="")
                else:
                    print(f"{avg:.1f} ± {std_list[i]:<4.1f} ", end="")
            print()


if __name__ == "__main__":
    get_statistic()
