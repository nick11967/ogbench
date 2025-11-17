import jax.numpy as jnp


def average_actions(actions: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Averages all sampled actions.
    Args:
        actions: Sampled actions of shape (num_samples, action_dim)
        mask: Mask indicating valid actions of shape (num_samples,)
    Returns:
        final_action: Averaged action of shape (action_dim,)
    """
    masked_actions = actions * mask[:, jnp.newaxis]
    num_valid = jnp.sum(mask)
    final_action = jnp.sum(masked_actions, axis=0) / jnp.maximum(num_valid, 1e-6)
    return final_action


def temporal_ensemble(
    actions: jnp.ndarray, mask: jnp.ndarray, decay_rate: float = 0.5
) -> jnp.ndarray:
    """Performs temporal ensembling of sampled actions.
    Args:
        actions: Sampled actions of shape (num_samples, action_dim)
        mask: Mask indicating valid actions of shape (num_samples,)
        decay_rate: Decay rate for exponential moving average.
    Returns:
        final_action: Temporally ensembled action of shape (action_dim,)
    """
    max_size = mask.shape[0]

    indices_full = jnp.arange(max_size, dtype=jnp.float32)
    indices_reverse = jnp.flip(indices_full, axis=0)

    # Exponential decay weights
    weights = jnp.exp(-indices_reverse * decay_rate)
    weights = weights * mask  # Apply mask
    weights_sum = jnp.sum(weights)
    weights = weights / jnp.maximum(weights_sum, 1e-6)  # Normalize weights

    # actions (N, D) * weights (N, 1) -> (N, D)
    weighted_actions = actions * weights[:, jnp.newaxis]
    final_action = jnp.sum(weighted_actions, axis=0)

    return final_action


def weighted_average_similarity(
    actions: jnp.ndarray,
    mask: jnp.ndarray,
    subgoals_in_stack: jnp.ndarray,
    new_subgoal: jnp.ndarray,
    beta: float = 5.0,
) -> jnp.ndarray:
    """
    Weighted average based on similarity with newest subgoal.
    Args:
        actions: Sampled actions of shape (num_subgoals, action_dim)
        subgoals_in_stack: Subgoals in the stack of shape (num_subgoals, rep_dim)
        new_subgoal: The newest subgoal of shape (rep_dim,)
        mask: Mask indicating valid subgoals of shape (num_subgoals,)
        beta: Scaling factor for similarity weights.
    Returns:
        final_action: Weighted average action of shape (action_dim,)
    """

    # Cosine similarity
    # subgoals_in_stack (N, D), new_subgoal (D,) -> similarities (N,)
    similarities = jnp.dot(subgoals_in_stack, new_subgoal) * mask

    weights = jnp.exp(similarities * beta)
    weights = weights * mask  # Apply mask
    weights_sum = jnp.sum(weights)
    weights = weights / jnp.maximum(weights_sum, 1e-6)  # Normalize weights

    weighted_actions = actions * weights[:, jnp.newaxis]
    final_action = jnp.sum(weighted_actions, axis=0)

    return final_action
