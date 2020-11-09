"""Base class for worker containing large initialization.

Written by: Zahi Kakish (zmk5)
"""
from typing import Tuple

import numpy as np

from rclpy.node import Node

from mrt_worker.policy.actor_critic import WorkerPolicyActorCriticDual
from mrt_worker.policy.actor_critic import WorkerPolicyActorCriticShared
from mrt_worker.policy.dqn import WorkerPolicyDQN
from mrt_worker.policy.ddqn import WorkerPolicyDDQN
from mrt_worker.policy.reinforce import WorkerPolicyREINFORCE
from mrt_worker.utils.database import LocalDatabase
from mrt_worker.utils.parameters import Experience
from mrt_worker.utils.parameters import WorkerParameters


class WorkerBase(Node):
    """Base class for worker containing large initialization."""

    def __init__(
            self,
            worker_id: int,
            name: str,
            policy_type: str = 'DQN') -> None:
        """Initialize the WorkerBase Node class."""
        super().__init__(name)
        self._worker_id = worker_id
        self.declare_parameters(
            namespace='',
            parameters=[
                ('use_gpu', False),
                ('number.states', 5),
                ('number.actions', 5),
                ('number.iterations', 1000),
                ('hyperparameter.alpha', 0.3),
                ('hyperparameter.beta', 0.1),
                ('hyperparameter.gamma', 0.99),
                ('hyperparameter.epsilon', 1.0),
                ('database.local_size', 10000),
                ('training_delay', 100),
                ('decay_rate', 500),
                ('batch_size', 16),
                ('hidden_layers', [16, 16]),
            ]
        )
        self._wp = WorkerParameters(
            self.get_parameter('number.iterations').value,
            self.get_parameter('number.states').value,
            self.get_parameter('number.actions').value,
            self.get_parameter('hyperparameter.alpha').value,
            self.get_parameter('hyperparameter.beta').value,
            self.get_parameter('hyperparameter.gamma').value,
            self.get_parameter('hyperparameter.epsilon').value,
            self.get_parameter('decay_rate').value,
            self.get_parameter('training_delay').value,
            self.get_parameter('batch_size').value
        )

        # Initialize local database for experience storage.
        self._db = LocalDatabase(
            self.get_parameter('database.local_size').value,
            self._wp.n_states,
            self._wp.n_iterations,
            np.float32
        )

        # Initialize policy for inference and training.
        if policy_type == 'DQN':
            self._policy = WorkerPolicyDQN(
                self._wp.n_states, self._wp.n_actions,
                self._wp.alpha, self._wp.gamma,
                self.get_parameter('hidden_layers').value,
                self.get_parameter('use_gpu').value
            )

        elif policy_type == 'DDQN':
            self._policy = WorkerPolicyDDQN(
                self._wp.n_states, self._wp.n_actions,
                self._wp.alpha, self._wp.gamma,
                self.get_parameter('hidden_layers').value,
                self.get_parameter('use_gpu').value
            )

        elif policy_type == 'REINFORCE':
            self._policy = WorkerPolicyREINFORCE(
                self._wp.n_states, self._wp.n_actions,
                self._wp.alpha, self._wp.gamma,
                self.get_parameter('hidden_layers').value,
                self.get_parameter('use_gpu').value
            )

        elif policy_type == 'A2C':
            self._policy = WorkerPolicyActorCriticShared(
                self._wp.n_states, self._wp.n_actions,
                self._wp.alpha, self._wp.gamma,
                self.get_parameter('hidden_layers').value,
                self.get_parameter('use_gpu').value
            )

        elif policy_type == 'A2CD':
            self._policy = WorkerPolicyActorCriticDual(
                self._wp.n_states, self._wp.n_actions,
                self._wp.alpha, self._wp.gamma,
                self.get_parameter('hidden_layers').value,
                self.get_parameter('use_gpu').value
            )

    @property
    def atype(self):
        """Return type of RL algorithm as string."""
        return self._policy.atype

    def step(self, collect: bool = True) -> Tuple[int, float]:
        """Generate experiences using the mean-field model for a policy."""
        _ = collect
        raise NotImplementedError
