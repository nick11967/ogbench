import os

# --- config ---
BASE_DIR = './exp/train/'
CKPT_NAME = 'sd005_20251115_232840' # Hum-Lar-hiql
EVAL_SCRIPT = 'python eval_agents.py'
AGENT_FILE = 'agents/sshiql.py'
# --------------


# esm_mode = "mean"
esm_mode = "temporal"
# esm_mode = "similarity"

temp_decay_rate = [0.995, 0.99, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2]
sim_beta = [8, 7, 5, 3, 1]

restore_path = os.path.join(BASE_DIR, CKPT_NAME) 

if esm_mode == "temporal":
    for tdr in temp_decay_rate:
        command = (
            f'{EVAL_SCRIPT} '
            f'--restore_path="{restore_path}" '
            f'--agent={AGENT_FILE} '
            f'--eval_on_cpu=0 '
            f'--ensemble_mode="temporal" '
            f'--agent.temporal_decay_rate={tdr} '
            f'--proc_name="ryujm-temp-{tdr}" '
            f'--video_to_wandb=1 ; \\'
        )
        print(command)

if esm_mode == "similarity":
    for sb in sim_beta:
        command = (
            f'{EVAL_SCRIPT} '
            f'--restore_path="{restore_path}" '
            f'--agent={AGENT_FILE} '
            f'--eval_on_cpu=0 '
            f'--ensemble_mode="temporal" '
            f'--agent.similarity_beta={sb} '
            f'--proc_name="ryujm-temp-{sb} '
            f'--video_to_wandb=1 ; \\'
        )
        print(command)
