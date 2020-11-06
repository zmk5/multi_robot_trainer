"""Double DQN model class for RL experiments with neural net function approx.

Written by: Zahi Kakish (zmk5)

"""
from typing import Dict
from typing import Union

import numpy as np

import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

from mrt_worker.policy.dqn import WorkerPolicyDQN


STATE = 0
ACTION = 1
REWARD = 2
NEXT_STATE = 3
NEXT_ACTION = 4
DONE = 5


class WorkerPolicyDDQN(WorkerPolicyDQN):
    """Double DQN Class containing all relatvent RL information."""

    @property
    def atype(self):
        """Return type of RL algorithm as string."""
        return 'DDQN'

    def calculate_q_loss(
            self,
            batch: Dict[str, np.ndarray],
            batch_size: int) -> Union[int, float]:
        """Calculate the state-value function loss for Double DQN."""
        prediction: EagerTensor = self._neural_net(batch[STATE])
        nn_next_prediction = tf.stop_gradient(
            self._neural_net(batch[NEXT_STATE]))
        target_next_prediction: EagerTensor = tf.stop_gradient(
            self._target_net(batch[NEXT_STATE]))

        # Action Prediction from Neural Net
        idx = tf.Variable(
            np.append(np.arange(batch_size).reshape(batch_size, 1),
                      batch[ACTION], axis=1),
            dtype=tf.int32
        )
        act_prediction = tf.reshape(
            tf.gather_nd(prediction, idx), (batch_size, 1))

        # Target Prediction from Target Neural Net
        nn_actions = tf.Variable(
            np.append(
                np.arange(batch_size).reshape(batch_size, 1),
                tf.argmax(
                    nn_next_prediction, axis=1).numpy().reshape(batch_size, 1),
                axis=1),
            dtype=tf.int32
        )
        max_next_prediction = tf.reshape(
            tf.gather_nd(target_next_prediction, nn_actions), (batch_size, 1))
        q_target = batch[REWARD] + self._gamma * (1 - batch[DONE]) * max_next_prediction

        return self._loss_function(act_prediction, q_target)
