import json
import os
import random
import time
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from agents import agents
from ml_collections import config_flags
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Eval', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'eval/', 'Save directory.')
# Need to specify restore_path and restore_epoch for evaluation.
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 20, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.')

flags.DEFINE_string(
    'ensemble_mode',
    'temporal',
    'Action ensemble mode: mean, temporal, similarity'
)
flags.DEFINE_integer('num_eval_seeds', 5, 'Number of different seeds to run evaluation.')

config_flags.DEFINE_config_file('agent', 'agents/hiql.py', lock_config=False)


def main(_):
    if FLAGS.restore_path is None:
        raise ValueError('restore_path must be specified for evaluation.')

    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(
        entity="nick11967-seoul-national-university",
        project='Eval_Agents', 
        group=FLAGS.run_group, 
        name=f"Eval_{os.path.basename(FLAGS.restore_path)}_{FLAGS.ensemble_mode}",
        reinit=True
    )

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set up environment and dataset.
    config = FLAGS.agent
    env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=config['frame_stack'])

    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
    
    # Evaluate for multiple seeds.
    all_seed_results = defaultdict(list)
    video_renders_for_wandb = []

    # Evaluation loop.
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))

    for seed_idx in tqdm.trange(FLAGS.num_eval_seeds, desc="Running Multiple Seeds..."):
        current_seed = FLAGS.seed + seed_idx

        current_agent = agent.replace(rng=jax.random.PRNGKey(current_seed))

        # Evaluate agent.
        if FLAGS.eval_on_cpu:
            eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
        else:
            eval_agent = agent

        renders = []
        task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
        num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)

        # Task loop.
        for task_id in tqdm.trange(1, num_tasks + 1):
            task_name = task_infos[task_id - 1]['task_name']
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
            )
            renders.extend(cur_renders)
            metric_names = ['success']
            for k, v in eval_info.items():
                if k in metric_names:
                    all_seed_results[f'seed_{current_seed}/{task_name}_{k}'].append(v)
                    all_seed_results[f'seed_{current_seed}/overall_{k}'].append(v)

    final_metrics = {}
    overall_success_rates = []

    for k in all_seed_results.keys():
        if 'overall_success' in k:
            overall_success_rates.extend(all_seed_results[k])
        
        final_metrics[f'evaluation/{k}_mean'] = np.mean(all_seed_results[k])

    final_metrics['evaluation/mean_overall_success'] = np.mean(overall_success_rates)
    final_metrics['evaluation/std_overall_success'] = np.std(overall_success_rates)
    final_metrics['ensemble_mode'] = FLAGS.ensemble_mode

    if FLAGS.video_episodes > 0 and len(video_renders_for_wandb) > 0:
        video = get_wandb_video(renders=video_renders_for_wandb, n_cols=num_tasks)
        final_metrics['video'] = video
    
    wandb.log(final_metrics, step=0)
    eval_logger.log(final_metrics, step=0)
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
