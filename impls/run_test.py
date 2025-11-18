import pandas as pd
import os
import subprocess

# --- config ---
CSV_FILE = 'train_result.csv'
BASE_DIR = './exp/train/'
EVAL_SCRIPT = 'python eval_agents.py'
AGENT_FILE = 'agents/sshiql.py'
# ------------------

def run_evaluation_scripts():
    """
    Read a CSV file, check for directory existence, and run evaluation scripts accordingly.
    """
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"Read '{CSV_FILE}'")

    except FileNotFoundError:
        print(f"Error: '{CSV_FILE}' file not found. Please check the path.")
        return
    except Exception as e:
        print(f"Error: Unexpected error while processing the file: {e}")
        return

    # Check if 'Name' column exists in the DataFrame
    if 'Name' not in df.columns:
        print("Error: 'Name' column is not present in the CSV file.")
        return

    print("-" * 30)

    # 2. Iterate over 'Name' entries, check conditions, and run scripts accordingly
    for index, row in df.iterrows():
        # Skip NaN or invalid values.
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
                f"--video_to_wandb=1"
            )

            print(f"Directory exists: '{restore_path}'")
            print(f"Running command: {command}")

            try:
                # Run subprocess (execute shell command)
                subprocess.run(command, shell=True, check=True)
                print("✨ Script execution completed.")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Error: Command execution failed (exit code {e.returncode}). Error output:\n{e.stderr.decode()}")
            except Exception as e:
                print(f"❌ Error: Unexpected error occurred during command execution: {e}")

        else:
            print(f"Directory does not exist: '{restore_path}'")

        print("-" * 30)

if __name__ == "__main__":
    run_evaluation_scripts()
