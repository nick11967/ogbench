import os

# --- config ---
BASE_DIR = './exp/test/'
# CKPT_NAME = 'sd005_20251116_112018' # Hum-Med-hiql
# CKPT_NAME = 'sd005_20251115_232840' # Hum-Lar-hiql
# CKPT_NAME = 'sd004_20251116_092630' # Hum-Gia-hiql
EVAL_SCRIPT = 'python eval_agents.py'
AGENT_FILE = 'agents/sshiql.py'
# --------------


# test_name = "mean"
test_name = "temporal"
# test_name = "similarity"
# test_name = "stack-size"
# test_name = "ss-sim"

ckpts = [
    'sd003_20251120_074529', # Point-Med
    'sd003_20251120_110302', # Point-Large
    'sd003_20251120_144050', # Point-Giant
    'sd003_20251120_181126', # Ant-Med
    'sd003_20251120_215253', # Ant-Large
    'sd003_20251121_020939', # Ant-Giant
    ]
temp_decay_rate = [0.95, 0.9, 0.8]
sim_beta = [7, 5, 4]
stack_size = [4, 8, 16, 32, 64, 128]


for ckpt in ckpts:
    restore_path = os.path.join(BASE_DIR, ckpt) 
    base_command = (
        f'{EVAL_SCRIPT} '
        f'--run_group="Hp-Tune" '
        f'--restore_path="{restore_path}" '
        f'--agent={AGENT_FILE} '
        f'--eval_on_cpu=0 '
        f'--video_to_wandb=1 '
    )
    if test_name == "temporal":
        for tdr in temp_decay_rate:
            command = (
                f'--agent.stack_max_size=16 '
                f'--agent.ensemble_mode="temporal" '
                f'--agent.temporal_decay_rate={tdr} '
                f'--proc_name="ryujm-temp-{tdr}" ; \\'
            )
            print(base_command + command)
    elif test_name == "ss-sim":
        for ss in stack_size:
            for sb in sim_beta:
                command = (
                    f'--agent.stack_max_size={ss} '
                    f'--agent.ensemble_mode="similarity" '
                    f'--agent.similarity_beta={sb} '
                    f'--proc_name="ryujm-ss+sim-{ss}+{sb}" ; \\'
                )
                print(base_command + command)
    elif test_name == "similarity":
        stack_max_size = 16
        for sb in sim_beta:
            command = (
                f'--agent.stack_max_size={stack_max_size} '
                f'--agent.ensemble_mode="similarity" '
                f'--agent.similarity_beta={sb} '
                f'--proc_name="ryujm-temp-{sb}" ; \\'
            )
            print(base_command + command)
    elif test_name == "stack-size":
        for ss in stack_size:
            command = (
                f'--run_group="Stack_Size" '
                f'--agent.stack_max_size={ss} '
                f'--agent.ensemble_mode="temporal" '
                f'--agent.temporal_decay_rate=0.95 '
                f'--proc_name="ryujm-stack-{ss}" ; \\'
            )
            print(base_command + command)
