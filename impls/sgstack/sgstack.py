import numpy as np


class SubgoalStack:
    """A stack to hold subgoals for hierarchical RL agents."""

    def __init__(self, max_size, subgoal_dim):
        self.max_size = max_size
        self.subgoal_dim = subgoal_dim
        self.stack = np.zeros((max_size, subgoal_dim), dtype=np.float32)
        self.size = 0

    def pop(self):
        """Pop the first subgoal in the stack."""
        if self.size == 0:
            raise IndexError("Subgoal stack is empty")
        self.stack = np.roll(self.stack, -1, axis=0)
        self.size -= 1

    def push(self, subgoal):
        """Push a new subgoal at the last position."""
        if self.size < self.max_size:
            self.stack[self.size] = subgoal
            self.size += 1
        else:
            while self.size >= self.max_size:
                self.pop()
            self.stack[self.size] = subgoal
            self.size += 1

    def get_stack(self):
        """Get the current subgoal stack up to the current size."""
        return self.stack[: self.size]
