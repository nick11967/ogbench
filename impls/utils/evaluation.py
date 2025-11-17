from collections import defaultdict

import jax
import numpy as np
from tqdm import trange


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def actor_function_with_rng(agent, observations, goals, temperature, seed):
    """Actor function that takes in a RNG key."""
    return agent.sample_actions(
        observations=observations, goals=goals, temperature=temperature, seed=seed
    )


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    task_id=None,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
    debug_level=0,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn_with_rng = supply_rng(
        actor_function_with_rng, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))
    )
    trajs = []
    stats = defaultdict(list)

    IS_SSHIQL = hasattr(agent, "subgoal_stack")

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        if IS_SSHIQL:
            agent = agent.replace(
                subgoal_stack=agent.subgoal_stack.__class__.create(
                    max_size=agent.subgoal_stack.max_size,
                    subgoal_dim=agent.subgoal_stack.subgoal_dim,
                )
            )
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
        goal = info.get('goal')
        goal_frame = info.get('goal_rendered')
        done = False
        step = 0
        render = []
        while not done:
            if debug_level >= 1:
                print(f"\n----- Step {step} -----")
            action = actor_fn_with_rng(
                agent=agent,
                observations=observation,
                goals=goal,
                temperature=eval_temperature,
            )
            if IS_SSHIQL:
                new_agent, action = action
                agent = new_agent
                if debug_level >= 2:
                    print(
                        f"Step: {step}, Subgoal Stack Size: {agent.subgoal_stack.size}"
                    )
            action = np.array(action)
            if not config.get('discrete'):
                if eval_gaussian is not None:
                    action = np.random.normal(action, eval_gaussian)
                action = np.clip(action, -1, 1)

            if debug_level >= 1:
                print(f"Step: {step}, Action: {action}")

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders
