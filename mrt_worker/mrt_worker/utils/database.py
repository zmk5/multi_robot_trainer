"""Local Database for Herding Experiment.

Written by: Zahi Kakish (zmk5)
"""
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import TypeVar

import numpy as np

from mrt_worker.utils.parameters import Experience


# Used for documentation purposes only
DataType = TypeVar('DataType', np.float32, np.float64)


class LocalDatabase():
    """Local Database class for storing leader experiences.

    Parameters
    ----------
    k : int
        Size of the database.
    n_v : int
        Length of one side of grid graph. For example, a 2x2 grid graph has an
        n_v value of 2.
    n_states : int
        Length of one side of grid graph. For example, a 2x2 grid graph has an
        n_v value of 2.
    n_iterations: int,
        Number of iterations per episode.
    dtype : {32, 64}, optional
        Type of float used to store the data for the states. Options are either
        64 for float64 and 32 for float 32. Default is 32.

    """

    __slots__ = [
        '_k',
        '_n_states',
        '_n_iterations',
        '_max',
        '_offset',
        '_experience',
        '_dtype',
    ]

    def __init__(
            self,
            k: int,
            n_states: int,
            n_iterations: int,
            dtype: DataType) -> None:
        """Initialize the Database class."""
        self._k: int = k
        self._n_states = n_states
        self._n_iterations = n_iterations
        self._max = 0
        self._offset = 0

        # Set data type for database and ROS messages
        self._dtype = dtype

        # Initialize local database.
        self._initialize_experience(self._dtype)

    def _initialize_experience(self, data_type: np.dtype) -> None:
        """Initialize the experience data structure."""
        self._experience: Dict[str, np.ndarray] = {
            'state': np.empty((self._k, self._n_states), data_type),
            'action': np.zeros((self._k, 1), np.int32),
            'reward': np.zeros((self._k, 1), np.float32),
            'next_state': np.empty((self._k, self._n_states), data_type),
            'next_action': np.empty((self._k, 1), np.int32),
            'done': np.zeros((self._k, 1), np.int32),
        }

    def save_experience(
            self,
            exp: Experience,
            done: bool) -> None:
        """Save experience to database from callback msg."""
        step: int = self._offset + exp.step

        # Set the max threshold value for batches.
        if step > self._max:
            self._max = step

        # Set data from msg.
        self._experience['state'][step, :] = exp.state
        self._experience['action'][step, :] = exp.action
        self._experience['reward'][step, :] = exp.reward
        self._experience['next_state'][step, :] = exp.next_state
        self._experience['next_action'][step, :] = exp.next_action
        self._experience['done'][step, :] = exp.done

        # Reset or increment offset if it hits terminal state.
        if done:
            if step > self._k - self._n_iterations:
                self._offset = 0

            else:
                self._offset += exp.step

    def sample_batch(
            self,
            batch_size: int,
            batch_type: str = 'random') -> Tuple[np.ndarray]:
        """Send batch of information when requested by client."""
        if batch_type == 'random':  # Create completely random samples batch.
            idx = np.random.randint(0, self._max, size=batch_size).tolist()

        # Create random initial value but sequential afterwards.
        elif batch_type == 'random_sequential':
            i = np.random.randint(0, self._max - batch_size)
            idx = np.arange(i, i + batch_size).tolist()

        else:  # Create batch of values from the beginning.
            idx = np.arange(0, batch_size + 1).tolist()

        return (
            self._experience['state'][idx, :],
            self._experience['action'][idx, :],
            self._experience['reward'][idx, :],
            self._experience['next_state'][idx, :],
            self._experience['next_action'][idx, :],
            self._experience['done'][idx, :],
        )

    def reset(self, new_k: Optional[int] = None) -> None:
        """Reset all variables to initial state."""
        if new_k is not None:
            self._k = new_k

        # Rest values and local database.
        self._max = 0
        self._offset = 0
        self._initialize_experience(self._dtype)
