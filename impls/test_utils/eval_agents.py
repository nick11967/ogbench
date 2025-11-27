import json
import os
import random
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

import jax
import numpy as np
import tqdm
import wandb
import setproctitle
from absl import app, flags
from agents import agents
from ml_collections import config_flags, ConfigDict
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, restore_agent_network_only
from utils.log_utils import (
    CsvLogger,
    get_flag_dict,
    get_wandb_video,
    setup_wandb,
    save_video,
)


FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Eval', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'eval/', 'Save directory.')
# Need to specify restore_path and restore_epoch for evaluation.
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer("restore_epoch", 1000000, "Restore epoch.")

flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer("video_to_wandb", 1, "Whether to save videos to Weights & Biases.")
flags.DEFINE_integer('eval_on_cpu', 0, 'Whether to evaluate on CPU.')
flags.DEFINE_string("proc_name", "ryujm-og-eval", "Process names.")

# Evaluation over multiple seeds.
flags.DEFINE_integer(
    "num_eval_seeds", 3, "Number of different seeds to run evaluation."
)
flags.DEFINE_integer("debug_level", 0, "Debug level.")

config_flags.DEFINE_config_file('agent', 'agents/hiql.py', lock_config=False)


def main(_):
    setproctitle.setproctitle(FLAGS.proc_name)
    if FLAGS.restore_path is None:
        raise ValueError('restore_path must be specified for evaluation.')

    # Load flags.
    train_flag_path = os.path.join(FLAGS.restore_path, "flags.json")
    config = FLAGS.agent  # Current configDict copy.
    if os.path.exists(train_flag_path):
        with open(train_flag_path, "r") as f:
            train_flag_dict = json.load(f)
        # Environment name from training flags.
        if "env_name" in train_flag_dict:
            FLAGS.env_name = train_flag_dict["env_name"]
        # Agent Config overriding.
        if "agent" in train_flag_dict and isinstance(train_flag_dict["agent"], dict):
            current_config_dict = config.to_dict()
            config_from_train = train_flag_dict["agent"]
            current_config_dict.update(config_from_train)
            if FLAGS.agent.agent_name == "sshiql":
                current_config_dict["agent_name"] = "sshiql"
            FLAGS.agent = ConfigDict(current_config_dict)

    # Set run name for wandb logging.
    env_name_mappings = {
        "antmaze": "Ant",
        "humanoidmaze": "Hum",
        "pointmaze": "Poi",
    }
    env_name = FLAGS.env_name.split("-")[0]
    env_name_short = env_name_mappings.get(env_name, env_name)
    size_name_mapping = {"medium": "Med", "large": "Lar", "giant": "Gia"}
    size_tag = FLAGS.env_name.split("-")[1]
    size_name_short = size_name_mapping.get(size_tag, size_tag)
    train_seed = FLAGS.restore_path.split("/")[-1].split("_")[0]

    run_name = (
        f"{env_name_short}_{size_name_short}_{FLAGS.agent.agent_name}"
    )

    if config["agent_name"] == "sshiql":
        ensemble_mapping = {"mean": "Mean", "temporal": "Temp", "similarity": "Simi"}
        ensemble_mode = FLAGS.agent.ensemble_mode
        ensemble_name_short = ensemble_mapping.get(ensemble_mode, ensemble_mode)
        run_name = run_name + f"_{ensemble_name_short}"
        if ensemble_name_short == 'Temp':
            run_name = run_name + f'_{FLAGS.agent.temporal_decay_rate}'
        elif ensemble_name_short == 'Simi':
            run_name = run_name + f'_{FLAGS.agent.similarity_beta}'

    run_name = run_name + f"_{train_seed}"

    # Set up logger.
    if FLAGS.debug_level == 0:
        setup_wandb(
            entity="nick11967-seoul-national-university",
            project="Eval_Agents",
            group=FLAGS.run_group,
            name=run_name,
        )

    # Save flags.
    if FLAGS.debug_level == 0:
        FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, run_name)
    else:
        FLAGS.save_dir = os.path.join(FLAGS.save_dir, "Debug", run_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set up environment and dataset.
    config = FLAGS.agent
    env = make_env_and_datasets(
        FLAGS.env_name, frame_stack=config["frame_stack"], env_only=True
    )
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape

    ex_observation = np.zeros((1, *obs_shape), dtype=np.float32)
    ex_action = np.zeros((1, *action_shape), dtype=np.float32)

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    agent_class = agents[config['agent_name']]
    if FLAGS.debug_level >= 1:
        print(f"Ensemble Mode: {config['ensemble_mode']}")
    agent = agent_class.create(
        FLAGS.seed,
        ex_observation,
        ex_action,
        config,
    )

    # Set target device for evaluation.
    if FLAGS.eval_on_cpu:
        target_device = jax.devices("cpu")[0]
        print("Evaluation will run on CPU.")
    elif jax.devices("gpu"):
        target_device = jax.devices("gpu")[0]
        print("Evaluation will run on GPU.")
    else:
        target_device = jax.devices("cpu")[0]
        print("GPU not found. Evaluation will run on CPU.")

    # Restore agent.
    if config["agent_name"] == "sshiql":
        agent = restore_agent_network_only(
            agent, FLAGS.restore_path, FLAGS.restore_epoch
        )
    else:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
    agent = jax.device_put(agent, target_device)

    # Evaluation loop.
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))

    total_success = 0
    total_episodes = 0

    # Evaluate multiple seeds.
    for seed_idx in tqdm.trange(FLAGS.num_eval_seeds, desc="Running Multiple Seeds..."):
        current_seed = FLAGS.seed + seed_idx

        eval_agent = agent.replace(rng=jax.random.PRNGKey(current_seed))

        # Evaluate agent.
        renders = []
        eval_metrics = {}
        overall_metrics = defaultdict(list)
        task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
        num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)

        # Task loop.
        for task_id in tqdm.trange(
            1, num_tasks + 1, desc=f"Seed {current_seed} Evaluating Tasks..."
        ):
            task_name = task_infos[task_id - 1]["task_name"]
            eval_info, trajs, cur_renders = evaluate(
                agent=eval_agent,
                env=env,
                task_id=task_id,
                config=config,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
                eval_temperature=FLAGS.eval_temperature,
                eval_gaussian=FLAGS.eval_gaussian,
                debug_level=FLAGS.debug_level,
            )

            renders.extend(cur_renders)
            metric_names = ['success']
            eval_metrics.update(
                {
                    f"{task_name}_{k}": v
                    for k, v in eval_info.items()
                    if k in metric_names
                }
            )

            for k, v in eval_info.items():
                if k in metric_names:
                    overall_metrics[k].append(v)
                    total_success += v * FLAGS.eval_episodes
                    total_episodes += FLAGS.eval_episodes

        for k, v in overall_metrics.items():
            eval_metrics[f"overall_{k}"] = np.mean(v)

        if FLAGS.video_episodes > 0:
            if FLAGS.video_to_wandb and FLAGS.debug_level == 0:
                # Log video to wandb.
                video = get_wandb_video(renders=renders, n_cols=num_tasks)
                eval_metrics["video"] = video
            else:
                # Save video locally.
                video_dir = os.path.join(FLAGS.save_dir, "videos")
                os.makedirs(video_dir, exist_ok=True)
                video_filename = f"s{seed_idx:02d}.mp4"
                video_path = os.path.join(video_dir, video_filename)
                save_video(renders=renders, path=video_path, n_cols=num_tasks)
        if FLAGS.debug_level == 0:
            wandb.log(eval_metrics, step=seed_idx)
        eval_logger.log(eval_metrics, step=seed_idx)

    total_success_rate = total_success / total_episodes if total_episodes > 0 else 0.0

    if FLAGS.debug_level == 0:
        wandb.log({"total_success_rate": total_success_rate})
        wandb.summary["total_success_rate"] = total_success_rate
    eval_logger.log(
        {"total_success_rate": total_success_rate},
        step=FLAGS.seed + FLAGS.num_eval_seeds,
    )

    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
