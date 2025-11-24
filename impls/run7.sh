python test_agents.py --restore_path="./exp/test/sd005_20251115_232840" --agent="agents/sshiql.py" --agent.stack_max_size=8 --agent.ensemble_mode="similarity" --agent.similarity_beta=4 --proc_name="Hum_Lar_SS+Simi" ; \
python test_agents.py --restore_path="./exp/test/sd003_20251116_053157" --proc_name="Hum_Gia_HIQL" ; \
python test_agents.py --restore_path="./exp/test/sd001_20251114_223022" --agent="agents/sshiql.py" --agent.stack_max_size=16 --agent.ensemble_mode="temporal" --agent.temporal_decay_rate=0.95 --proc_name="Hum_Gia_SS+Temp" ; \
