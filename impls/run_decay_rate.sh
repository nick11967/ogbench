python eval_agents.py --restore_path="./exp/train/sd005_20251116_112018" --agent=agents/sshiql.py --eval_on_cpu=0 --agent.temporal_decay_rate=0.4 --video_to_wandb=1 ; \
python eval_agents.py --restore_path="./exp/train/sd005_20251116_112018" --agent=agents/sshiql.py --eval_on_cpu=0 --agent.temporal_decay_rate=0.3 --video_to_wandb=1 ; \
python eval_agents.py --restore_path="./exp/train/sd005_20251116_112018" --agent=agents/sshiql.py --eval_on_cpu=0 --agent.temporal_decay_rate=0.2 --video_to_wandb=1 ; \
python eval_agents.py --restore_path="./exp/train/sd005_20251116_112018" --agent=agents/sshiql.py --eval_on_cpu=0 --agent.temporal_decay_rate=0.1 --video_to_wandb=1 ; \
python eval_agents.py --restore_path="./exp/train/sd005_20251116_112018" --agent=agents/sshiql.py --eval_on_cpu=0 --agent.temporal_decay_rate=0.95 --video_to_wandb=1 ; \
python eval_agents.py --restore_path="./exp/train/sd005_20251116_112018" --agent=agents/sshiql.py --eval_on_cpu=0 --agent.temporal_decay_rate=0.99 --video_to_wandb=1 ; \
