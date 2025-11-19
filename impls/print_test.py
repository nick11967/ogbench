import pandas as pd
import os
import subprocess

# --- config ---
CSV_FILE = 'train_result_large.csv'
BASE_DIR = './exp/train/'
EVAL_SCRIPT = 'python eval_sim.py'
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

        if os.path.isdir(restore_path):
            command = (
                f"{EVAL_SCRIPT} "
                f"--restore_path=\"{restore_path}\" "
                f"--agent={AGENT_FILE} "
                f"--eval_on_cpu=0 "
                f'--ensemble_mode="similarity" '
                f"--agent.similarity_beta=5 "
                f"--video_to_wandb=1 ; \\"
            )

        print(command)

if __name__ == "__main__":
    run_evaluation_scripts()
