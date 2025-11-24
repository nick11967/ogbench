python test_agents.py --restore_path="./exp/test/sd003_20251120_074529" --agent="agents/sshiql.py" --agent.stack_max_size=64 --agent.ensemble_mode="similarity" --agent.similarity_beta=4 --proc_name="Poi_Med_SS+Simi" ; \
python test_agents.py --restore_path="./exp/test/sd001_20251120_055024" --proc_name="Poi_Med_HIQL" ; \
python test_agents.py --restore_path="./exp/test/sd003_20251120_110302" --agent="agents/sshiql.py" --agent.stack_max_size=16 --agent.ensemble_mode="temporal" --agent.temporal_decay_rate=0.95 --proc_name="Poi_Lar_SS+Temp" ; \
python test_agents.py --restore_path="./exp/test/sd002_20251120_095339" --agent="agents/sshiql.py" --agent.stack_max_size=64 --agent.ensemble_mode="similarity" --agent.similarity_beta=4 --proc_name="Poi_Lar_SS+Simi" ; \
python test_agents.py --restore_path="./exp/test/sd003_20251120_144050" --proc_name="Poi_Gia_HIQL" ; \
python test_agents.py --restore_path="./exp/test/sd002_20251120_132654" --agent="agents/sshiql.py" --agent.stack_max_size=16 --agent.ensemble_mode="temporal" --agent.temporal_decay_rate=0.95 --proc_name="Poi_Gia_SS+Temp" ; \
python test_agents.py --restore_path="./exp/test/sd001_20251120_120934" --agent="agents/sshiql.py" --agent.stack_max_size=64 --agent.ensemble_mode="similarity" --agent.similarity_beta=4 --proc_name="Poi_Gia_SS+Simi" ; \
python test_agents.py --restore_path="./exp/test/sd002_20251120_170441" --proc_name="Ant_Med_HIQL" ; \
python test_agents.py --restore_path="./exp/test/sd001_20251120_155709" --agent="agents/sshiql.py" --agent.stack_max_size=16 --agent.ensemble_mode="temporal" --agent.temporal_decay_rate=0.95 --proc_name="Ant_Med_SS+Temp" ; \
python test_agents.py --restore_path="./exp/test/sd003_20251120_215253" --agent="agents/sshiql.py" --agent.stack_max_size=8 --agent.ensemble_mode="similarity" --agent.similarity_beta=7 --proc_name="Ant_Lar_SS+Simi" ; \
python test_agents.py --restore_path="./exp/test/sd001_20251120_191634" --proc_name="Ant_Lar_HIQL" ; \
python test_agents.py --restore_path="./exp/test/sd003_20251121_020939" --agent="agents/sshiql.py" --agent.stack_max_size=16 --agent.ensemble_mode="temporal" --agent.temporal_decay_rate=0.95 --proc_name="Ant_Gia_SS+Temp" ; \
python test_agents.py --restore_path="./exp/test/sd002_20251121_004031" --agent="agents/sshiql.py" --agent.stack_max_size=8 --agent.ensemble_mode="similarity" --agent.similarity_beta=7 --proc_name="Ant_Gia_SS+Simi" ; \
python test_agents.py --restore_path="./exp/test/sd004_20251115_212409" --proc_name="Hum_Med_HIQL" ; \
python test_agents.py --restore_path="./exp/test/sd005_20251116_112018" --agent="agents/sshiql.py" --agent.stack_max_size=16 --agent.ensemble_mode="temporal" --agent.temporal_decay_rate=0.95 --proc_name="Hum_Med_SS+Temp" ; \
python test_agents.py --restore_path="./exp/test/sd006_20251116_015801" --agent="agents/sshiql.py" --agent.stack_max_size=8 --agent.ensemble_mode="similarity" --agent.similarity_beta=4 --proc_name="Hum_Med_SS+Simi" ; \
python test_agents.py --restore_path="./exp/test/sd004_20251116_031642" --proc_name="Hum_Lar_HIQL" ; \
python test_agents.py --restore_path="./exp/test/sd005_20251115_232840" --agent="agents/sshiql.py" --agent.stack_max_size=16 --agent.ensemble_mode="temporal" --agent.temporal_decay_rate=0.95 --proc_name="Hum_Lar_SS+Temp" ; \
python test_agents.py --restore_path="./exp/test/sd004_20251116_092630" --agent="agents/sshiql.py" --agent.stack_max_size=8 --agent.ensemble_mode="similarity" --agent.similarity_beta=4 --proc_name="Hum_Gia_SS+Simi" ; \
python test_agents.py --restore_path="./exp/test/sd001_20251114_223022" --proc_name="Hum_Gia_HIQL" ; \
