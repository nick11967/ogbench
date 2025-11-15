python main.py --env_name=humanoidmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100 --seed=1 ; \
python main.py --env_name=humanoidmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.1 --agent.discount=0.995 --seed=1 ; \
python main.py --env_name=humanoidmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100 --seed=2 ; \
python main.py --env_name=humanoidmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.1 --agent.discount=0.995 --seed=2 ; \
python main.py --env_name=humanoidmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/hiql.py --agent.discount=0.995 --agent.high_alpha=3.0 --agent.low_alpha=3.0 --agent.subgoal_steps=100 --seed=3 ; \
python main.py --env_name=humanoidmaze-giant-navigate-v0 --eval_episodes=50 --agent=agents/crl.py --agent.alpha=0.1 --agent.discount=0.995 --seed=3
