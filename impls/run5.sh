python test_agents.py --restore_path="./exp/OGBench/Debug/sd004_20251126_093813" --agent="agents/sshiql.py" --agent.stack_max_size=100 --agent.ensemble_mode="temporal" --agent.temporal_decay_rate=0.95 --proc_name="Hum_Med_SS+Temp" ; \
python test_agents.py --restore_path="./exp/OGBench/Debug/sd005_20251126_093839" --agent="agents/sshiql.py" --agent.stack_max_size=8 --agent.ensemble_mode="similarity" --agent.similarity_beta=4 --proc_name="Hum_Med_SS+Simi" ; \
python test_agents.py --restore_path="./exp/OGBench/Debug/sd003_20251126_093943" --proc_name="Hum_Lar_HIQL" ; \
python test_agents.py --restore_path="./exp/OGBench/Debug/sd004_20251126_094052" --agent="agents/sshiql.py" --agent.stack_max_size=100 --agent.ensemble_mode="temporal" --agent.temporal_decay_rate=0.95 --proc_name="Hum_Lar_SS+Temp" ; \
python test_agents.py --restore_path="./exp/OGBench/Debug/sd005_20251124_210427" --agent="agents/sshiql.py" --agent.stack_max_size=8 --agent.ensemble_mode="similarity" --agent.similarity_beta=4 --proc_name="Hum_Lar_SS+Simi" ; \
python test_agents.py --restore_path="./exp/OGBench/Debug/sd003_20251125_020038" --proc_name="Hum_Gia_HIQL" ; \
python test_agents.py --restore_path="./exp/OGBench/Debug/sd004_20251125_040515" --agent="agents/sshiql.py" --agent.stack_max_size=100 --agent.ensemble_mode="temporal" --agent.temporal_decay_rate=0.95 --proc_name="Hum_Gia_SS+Temp" ; \
