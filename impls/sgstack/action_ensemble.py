import jax.numpy as jnp

def average_actions(actions: jnp.ndarray) -> jnp.ndarray:
    """Averages all sampled actions.
    Args:
        actions: Sampled actions of shape (num_samples, action_dim)
    Returns:
        final_action: Averaged action of shape (action_dim,)
    """
    final_action = jnp.mean(actions, axis=0)
    return final_action

def temporal_ensemble(actions: jnp.ndarray, decay_rate: float = 0.5) -> jnp.ndarray:
    """Performs temporal ensembling of sampled actions.
    Args:
        actions: Sampled actions of shape (num_samples, action_dim)
        decay_rate: Decay rate for exponential moving average.
    Returns:
        final_action: Temporally ensembled action of shape (action_dim,)
    """
    num_subgoals = actions.shape[0]

    indices = jnp.arange(num_subgoals, dtype=jnp.float32)

    # Exponential decay weights
    weights = jnp.exp(indices * decay_rate)
    weights = weights / jnp.sum(weights) # Normalize weights

    # actions (N, D) * weights (N, 1) -> (N, D)
    weighted_actions = actions * weights[:, jnp.newaxis]
    final_action = jnp.sum(weighted_actions, axis=0)

    return final_action

def weighted_average_similarity(
        actions: jnp.ndarray,
        subgoals_in_stack: jnp.ndarray,
        new_subgoal: jnp.ndarray,
        beta: float = 5.0
) -> jnp.ndarray:
    """
    Weighted average based on similarity with newest subgoal.
    Args:
        actions: Sampled actions of shape (num_subgoals, action_dim)
        subgoals_in_stack: Subgoals in the stack of shape (num_subgoals, rep_dim)
        new_subgoal: The newest subgoal of shape (rep_dim,)
        beta: Scaling factor for similarity weights.
    Returns:
        final_action: Weighted average action of shape (action_dim,)
    """    

    # Cosine similarity
    # subgoals_in_stack (N, D), new_subgoal (D,) -> similarities (N,)
    similarities = jnp.dot(subgoals_in_stack, new_subgoal)

    weights = jnp.exp(similarities * beta)
    weights = weights / jnp.sum(weights) # Normalize weights

    weighted_actions = actions * weights[:, jnp.newaxis]
    final_action = jnp.sum(weighted_actions, axis=0)
    return final_action
