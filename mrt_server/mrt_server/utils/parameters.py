"""Server parameters for use by the worker.

Written by: Zahi Kakish (zmk5)
"""
from typing import Optional

from dataclasses import dataclass


@dataclass
class ServerParameters():
    """Mean Field experiment parameters stored in a dataclass."""

    __slots__ = [
        'n_states',
        'n_actions',
        'n_episodes',
        'n_workers',
        'alpha',
        'epsilon',
        'training_delay',
    ]

    n_states: int        # default: 5
    n_actions: int       # default: 5
    n_episodes: int      # default: 5000
    n_workers: int       # default: 2
    alpha: float         # default: 0.3
    epsilon: float       # default: 1.0
    training_delay: int  # default: 100

    def decay_epsilon(self, episode: int, decay_rate: int) -> None:
        """Decay value of epsilon for use during training."""
        if (episode + 1) % decay_rate == 0:
            if self.epsilon > 0.1:
                self.epsilon -= 0.1

            else:
                self.epsilon = 0.1

    def reset_epsilon(self, new_epsilon: Optional[float] = None) -> None:
        """Reset the epsilon value."""
        if new_epsilon is None:
            self.epsilon = 1.0

        else:
            self.epsilon = new_epsilon


@dataclass
class Flags():
    """Control flags used by the worker node."""

    __slots__ = [
        'pull',     # Pull neural net parameters from global ROS node.
        'collect',  # Collect a set of trajectories and store.
        'compute',  # Compute the gradient of the local network.
        'push',     # Push neural net parameters to global ROS node.
        'action',   # Intermediate action between each action.
    ]

    pull: bool
    collect: bool
    compute: bool
    push: bool
    action: bool

    def reset(self):
        """Reset flags back to original."""
        self.pull = True
        self.collect = False
        self.compute = False
        self.push = False
        self.action = False

    def shift_to_collect_cycle(self):
        """Shift flags for `collect`  cycle."""
        self.pull = False
        self.collect = True
        self.action = False

    def shift_to_compute_cycle(self):
        """Shift flags for `compute` cycle."""
        self.collect = False
        self.compute = True
        self.action = False

    def shift_to_pull_cycle(self):
        """Shift flags for `pull` cycle."""
        self.compute = False
        self.pull = True
        self.action = False
