import pandas as pd
import os
import subprocess

# --- config ---
CSV_FILE = 'agent_list.csv'
BASE_DIR = './exp/point/'
EVAL_SCRIPT = 'python test_agents.py'
AGENT_FILE = 'agents/sshiql.py'
# ------------------

def run_evaluation_scripts():
    """
    Read a CSV file, check for directory existence, and run evaluation scripts accordingly.
    """
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        return

    # Check if 'Name' column exists in the DataFrame
    if 'Name' not in df.columns:
        return

    # Iterate over 'Name' entries, check conditions, and run scripts accordingly
    for index, row in df.iterrows():
        name = str(row['Name']).strip()
        if not name or name.lower() == 'nan':
            continue

        restore_path = os.path.join(BASE_DIR, name)

        PROC_NAME = 'Poi_'
        stack_max_size = 64
        similarity_beta = 4

        # Env Size
        if index < 3:
            PROC_NAME += 'Med'
        elif index < 6:
            PROC_NAME += 'Lar'
        else:
            PROC_NAME += 'Gia'

        # Print Cmd
        command_base = (
            f'{EVAL_SCRIPT} '
            f'--restore_path="{restore_path}" '
            f'--proc_name="{PROC_NAME}_HIQL" ; \\'
        )
        command_temp = (
            f'{EVAL_SCRIPT} '
            f'--restore_path="{restore_path}" '
            f'--agent="{AGENT_FILE}" '
            f'--agent.stack_max_size=16 '
            f'--agent.ensemble_mode="temporal" '
            f'--agent.temporal_decay_rate=0.95 '
            f'--proc_name="{PROC_NAME}_SS+Temp" ; \\'
        )
        command_simi = (
            f'{EVAL_SCRIPT} '
            f'--restore_path="{restore_path}" '
            f'--agent="{AGENT_FILE}" '
            f'--agent.stack_max_size={stack_max_size} '
            f'--agent.ensemble_mode="similarity" '
            f'--agent.similarity_beta={similarity_beta} '
            f'--proc_name="{PROC_NAME}_SS+Simi" ; \\'
        )
        print(command_base)
        print(command_temp)
        print(command_simi)

if __name__ == "__main__":
    run_evaluation_scripts()
