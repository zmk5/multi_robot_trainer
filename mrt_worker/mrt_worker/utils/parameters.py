"""Paramters for use by the worker.

Written by: Zahi Kakish (zmk5)
"""
from typing import Optional
from typing import Union

from dataclasses import dataclass

import numpy as np


@dataclass
class WorkerParameters():
    """Mean Field experiement parameters stored in a dataclass."""

    __slots__ = [
        'n_iterations',
        'n_states',
        'n_actions',
        'alpha',
        'beta',
        'gamma',
        'epsilon',
        'decay_rate',
        'training_delay',
        'batch_size'
    ]

    n_iterations: int    # default: 1000
    n_states: int        # default: 5
    n_actions: int       # default: 5
    alpha: float         # default: 0.3
    beta: float          # default: 0.1
    gamma: float         # default: 0.99
    epsilon: float       # default: 1.0
    decay_rate: int      # default: 500
    training_delay: int  # default: 100
    batch_size: int      # default: 16

    def decay_epsilon(self, episode: int, rate_value: float = 0.1) -> None:
        """Decay value of epsilon for use during training."""
        if (episode + 1) % self.decay_rate == 0:
            # The decay rate_value is same as final epsilon value.
            if self.epsilon > rate_value:
                self.epsilon -= rate_value

            else:
                self.epsilon = rate_value

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
        """Reset flags back to original `pull` cycle."""
        self.pull = True
        self.collect = False
        self.compute = False
        self.push = False
        self.action = True

    def shift_to_collect_cycle(self):
        """Shift flags for `collect`  cycle."""
        self.pull = False
        self.collect = True
        self.action = True

    def shift_to_compute_cycle(self):
        """Shift flags for `compute` cycle."""
        self.collect = False
        self.compute = True
        self.action = True

    def shift_to_push_cycle(self):
        """Shift flags for `push` cycle."""
        self.compute = False
        self.push = True
        self.action = True


@dataclass
class Experience():
    """Experience undergone by the agent for an experiment."""

    __slots__ = [
        'step',
        'state',
        'action',
        'reward',
        'next_state',
        'next_action',
        'done',
    ]

    step: int
    state: np.ndarray
    action: Union[int, float]
    reward: Union[int, float]
    next_state: np.ndarray
    next_action: Union[int, float]
    done: int

    def __init__(self, n_states: int):
        """Instantiate Experience class."""
        self.step = 0
        self.state = np.empty((1, n_states))
        self.action = 0
        self.reward = 0
        self.next_state = np.empty((1, n_states))
        self.next_action = 0
        self.done = 0
