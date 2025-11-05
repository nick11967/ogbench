import jax
import jax.numpy as jnp
import flax.struct


class SubgoalStack(flax.struct.PyTreeNode):
    """A stack to hold subgoals for hierarchical RL agents."""

    stack: jnp.ndarray
    size: int

    max_size: int = flax.struct.field(pytree_node=False)
    subgoal_dim: int = flax.struct.field(pytree_node=False)

    @classmethod
    def create(cls, max_size, subgoal_dim):
        """Creates an empty SubgoalStack."""
        return cls(
            stack=jnp.zeros((max_size, subgoal_dim), dtype=jnp.float32),
            size=0,
            max_size=max_size,
            subgoal_dim=subgoal_dim,
        )

    def push(self, subgoal):
        """Push a new subgoal and return a new SubgoalStack."""
        is_full = self.size >= self.max_size

        # Roll the stack if it's full
        rolled_stack = jax.lax.cond(
            is_full,
            lambda s: jnp.roll(s, shift=-1, axis=0),
            lambda s: s,
            self.stack,
        )

        # Determine the index to insert the new subgoal
        push_index = jax.lax.cond(
            is_full,
            lambda: self.max_size - 1,
            lambda: self.size,
        )

        # Push the new subgoal
        new_stack = rolled_stack.at[push_index].set(subgoal)

        new_size = jax.lax.cond(
            is_full,
            lambda: self.max_size,
            lambda: self.size + 1,
        )

        return self.replace(stack=new_stack, size=new_size)

    def get_current_stack(self):
        """Get the current subgoal stack up to the current size."""
        return self.stack[: self.size]
