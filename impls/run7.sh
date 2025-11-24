python test_agents.py --restore_path="./exp/test/sd002_20251120_064654" --proc_name="Poi_Med_HIQL" ; \
python test_agents.py --restore_path="./exp/test/sd001_20251120_055024" --agent="agents/sshiql.py" --agent.stack_max_size=16 --agent.ensemble_mode="temporal" --agent.temporal_decay_rate=0.95 --proc_name="Poi_Med_SS+Temp" ; \
python test_agents.py --restore_path="./exp/test/sd003_20251120_110302" --agent="agents/sshiql.py" --agent.stack_max_size=64 --agent.ensemble_mode="similarity" --agent.similarity_beta=4 --proc_name="Poi_Lar_SS+Simi" ; \
python test_agents.py --restore_path="./exp/test/sd001_20251120_084232" --proc_name="Poi_Lar_HIQL" ; \
python test_agents.py --restore_path="./exp/test/sd003_20251120_144050" --agent="agents/sshiql.py" --agent.stack_max_size=16 --agent.ensemble_mode="temporal" --agent.temporal_decay_rate=0.95 --proc_name="Poi_Gia_SS+Temp" ; \
python test_agents.py --restore_path="./exp/test/sd002_20251120_132654" --agent="agents/sshiql.py" --agent.stack_max_size=64 --agent.ensemble_mode="similarity" --agent.similarity_beta=4 --proc_name="Poi_Gia_SS+Simi" ; \
python test_agents.py --restore_path="./exp/test/sd003_20251120_181126" --proc_name="Ant_Med_HIQL" ; \
python test_agents.py --restore_path="./exp/test/sd002_20251120_170441" --agent="agents/sshiql.py" --agent.stack_max_size=16 --agent.ensemble_mode="temporal" --agent.temporal_decay_rate=0.95 --proc_name="Ant_Med_SS+Temp" ; \
python test_agents.py --restore_path="./exp/test/sd001_20251120_155709" --agent="agents/sshiql.py" --agent.stack_max_size=8 --agent.ensemble_mode="similarity" --agent.similarity_beta=7 --proc_name="Ant_Med_SS+Simi" ; \
python test_agents.py --restore_path="./exp/test/sd002_20251120_203514" --proc_name="Ant_Lar_HIQL" ; \
python test_agents.py --restore_path="./exp/test/sd001_20251120_191634" --agent="agents/sshiql.py" --agent.stack_max_size=16 --agent.ensemble_mode="temporal" --agent.temporal_decay_rate=0.95 --proc_name="Ant_Lar_SS+Temp" ; \
python test_agents.py --restore_path="./exp/test/sd003_20251121_020939" --agent="agents/sshiql.py" --agent.stack_max_size=8 --agent.ensemble_mode="similarity" --agent.similarity_beta=7 --proc_name="Ant_Gia_SS+Simi" ; \
python test_agents.py --restore_path="./exp/test/sd001_20251120_230934" --proc_name="Ant_Gia_HIQL" ; \
python test_agents.py --restore_path="./exp/test/sd004_20251115_212409" --agent="agents/sshiql.py" --agent.stack_max_size=16 --agent.ensemble_mode="temporal" --agent.temporal_decay_rate=0.95 --proc_name="Hum_Med_SS+Temp" ; \
python test_agents.py --restore_path="./exp/test/sd005_20251116_112018" --agent="agents/sshiql.py" --agent.stack_max_size=8 --agent.ensemble_mode="similarity" --agent.similarity_beta=4 --proc_name="Hum_Med_SS+Simi" ; \
python test_agents.py --restore_path="./exp/test/sd003_20251115_184813" --proc_name="Hum_Lar_HIQL" ; \
python test_agents.py --restore_path="./exp/test/sd004_20251116_031642" --agent="agents/sshiql.py" --agent.stack_max_size=16 --agent.ensemble_mode="temporal" --agent.temporal_decay_rate=0.95 --proc_name="Hum_Lar_SS+Temp" ; \
python test_agents.py --restore_path="./exp/test/sd005_20251115_232840" --agent="agents/sshiql.py" --agent.stack_max_size=8 --agent.ensemble_mode="similarity" --agent.similarity_beta=4 --proc_name="Hum_Lar_SS+Simi" ; \
python test_agents.py --restore_path="./exp/test/sd003_20251116_053157" --proc_name="Hum_Gia_HIQL" ; \
python test_agents.py --restore_path="./exp/test/sd001_20251114_223022" --agent="agents/sshiql.py" --agent.stack_max_size=16 --agent.ensemble_mode="temporal" --agent.temporal_decay_rate=0.95 --proc_name="Hum_Gia_SS+Temp" ; \
