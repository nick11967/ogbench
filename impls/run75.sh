python test_agents.py --restore_path="./exp/point/sd008_20251127_002125" --proc_name="Poi_Med_HIQL" ; \
python test_agents.py --restore_path="./exp/point/sd009_20251127_030727" --agent="agents/sshiql.py" --agent.stack_max_size=16 --agent.ensemble_mode="temporal" --agent.temporal_decay_rate=0.95 --proc_name="Poi_Med_SS+Temp" ; \
python test_agents.py --restore_path="./exp/point/sd006_20251126_194104" --agent="agents/sshiql.py" --agent.stack_max_size=64 --agent.ensemble_mode="similarity" --agent.similarity_beta=4 --proc_name="Poi_Lar_SS+Simi" ; \
python test_agents.py --restore_path="./exp/point/sd010_20251127_064718" --proc_name="Poi_Lar_HIQL" ; \
python test_agents.py --restore_path="./exp/point/sd005_20251126_174943" --agent="agents/sshiql.py" --agent.stack_max_size=16 --agent.ensemble_mode="temporal" --agent.temporal_decay_rate=0.95 --proc_name="Poi_Gia_SS+Temp" ; \
python test_agents.py --restore_path="./exp/point/sd009_20251127_045610" --agent="agents/sshiql.py" --agent.stack_max_size=64 --agent.ensemble_mode="similarity" --agent.similarity_beta=4 --proc_name="Poi_Gia_SS+Simi" ; \
