# --- config ---
EVAL_SCRIPT = 'python main.py'
AGENT_FILE = 'agents/hiql.py'
ENV_POSTPIX = 'navigate-v0'
# --------------

envs = ['pointmaze', 'antmaze']
sizes = ['-medium-', '-large-', '-giant-']
seeds = [1, 2, 3]

for env in envs:
    for size in sizes:
        for seed in seeds:
            env_name = env + size + ENV_POSTPIX
            if size == '-giant-':
                command = (
                    f'{EVAL_SCRIPT} '
                    f'--env_name={env_name} '
                    f'--eval_episodes=50 '
                    f'--agent={AGENT_FILE} '
                    f'--seed={seed} '
                    f'--agent.high_alpha=3.0 '
                    f'--agent.low_alpha=3.0 ; \\'
                )
            else:
                command = (
                    f'{EVAL_SCRIPT} '
                    f'--env_name={env_name} '
                    f'--eval_episodes=50 '
                    f'--agent={AGENT_FILE} '
                    f'--seed={seed} '
                    f'--agent.discount=0.995 '
                    f'--agent.high_alpha=3.0 '
                    f'--agent.low_alpha=3.0 ; \\'
                )
            print(command)
