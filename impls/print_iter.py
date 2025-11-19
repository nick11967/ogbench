import os

# --- config ---
BASE_DIR = './exp/train/'
# CKPT_NAME = 'sd005_20251116_112018' # Hum-Med-hiql
CKPT_NAME = 'sd005_20251115_232840' # Hum-Lar-hiql
# CKPT_NAME = 'sd004_20251116_092630' # Hum-Gia-hiql
EVAL_SCRIPT = 'python eval_agents.py'
AGENT_FILE = 'agents/sshiql.py'
# --------------


# test_name = "mean"
# test_name = "temporal"
# test_name = "similarity"
# test_name = "stack-size"
test_name = "ss-sim"


temp_decay_rate = [0.995, 0.99, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2]
sim_beta = [8, 7, 6, 5, 4]
stack_size = [3, 5, 10, 25, 50]

restore_path = os.path.join(BASE_DIR, CKPT_NAME) 

if test_name == "temporal":
    for tdr in temp_decay_rate:
        command = (
            f'{EVAL_SCRIPT} '
            f'--restore_path="{restore_path}" '
            f'--agent={AGENT_FILE} '
            f'--agent.ensemble_mode="temporal" '
            f'--agent.temporal_decay_rate={tdr} '
            f'--proc_name="ryujm-temp-{tdr}" '
            f'--eval_on_cpu=0 '
            f'--video_to_wandb=1 ; \\'
        )
        print(command)
elif test_name == "similarity":
    for sb in sim_beta:
        command = (
            f'{EVAL_SCRIPT} '
            f'--restore_path="{restore_path}" '
            f'--agent={AGENT_FILE} '
            f'--agent.ensemble_mode="similarity" '
            f'--agent.similarity_beta={sb} '
            f'--proc_name="ryujm-temp-{sb} '
            f'--eval_on_cpu=0 '
            f'--video_to_wandb=1 ; \\'
        )
        print(command)
elif test_name == "stack-size":
    for ss in stack_size:
        command = (
            f'{EVAL_SCRIPT} '
            f'--run_group="Stack_Size" '
            f'--restore_path="{restore_path}" '
            f'--agent={AGENT_FILE} '
            f'--agent.stack_max_size={ss} '
            f'--agent.ensemble_mode="temporal" '
            f'--agent.temporal_decay_rate=0.95 '
            f'--proc_name="ryujm-stack-{ss}" '
            f'--eval_on_cpu=0 '
            f'--video_to_wandb=1 ; \\'
        )
        print(command)
elif test_name == "ss-sim":
    for ss in stack_size:
        for sb in sim_beta:
            command = (
                f'{EVAL_SCRIPT} '
                f'--run_group="Stack_Size" '
                f'--restore_path="{restore_path}" '
                f'--agent={AGENT_FILE} '
                f'--agent.stack_max_size={ss} '
                f'--agent.ensemble_mode="similarity" '
                f'--agent.similarity_beta={sb} '
                f'--proc_name="ryujm-ss+sim-{ss}+{sb}" '
                f'--eval_on_cpu=0 '
                f'--video_to_wandb=1 ; \\'
            )
            print(command)
