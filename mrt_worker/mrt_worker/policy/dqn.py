"""DQN model class for RL experiments with neural net function approx.

Written by: Zahi Kakish (zmk5)

"""
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.framework.ops import EagerTensor

from mean_field_msgs.srv import Gradients
from mean_field_msgs.srv import Weights


STATE = 0
ACTION = 1
REWARD = 2
NEXT_STATE = 3
NEXT_ACTION = 4
DONE = 5


class WorkerPolicyDQN():
    """DQN Class containing all relatvent RL information."""

    __slots__ = [
        '_n_states',
        '_n_actions',
        '_alpha',
        '_gamma',
        '_neural_net',
        '_target_net',
        '_loss_function',
        '_gradients',
        '_hidden_layer_sizes',
    ]

    def __init__(
            self,
            n_states: int,
            n_actions: int,
            alpha: float,
            gamma: float,
            hidden_layer_sizes: List[int],
            use_gpu: bool = False) -> None:
        """Initialize the PolicyDQN class."""
        self._n_states = n_states
        self._n_actions = n_actions
        self._alpha = alpha
        self._gamma = gamma
        self._hidden_layer_sizes = hidden_layer_sizes

        # Turn off GPU, if needed.
        if not use_gpu:
            tf.config.set_visible_devices([], 'GPU')

        # Set Q-function Neural Net Approximator and Target Network
        self._neural_net = keras.Sequential([
            keras.layers.Dense(
                hidden_layer_sizes[0], activation='relu',
                input_shape=(n_states,)),
            keras.layers.Dense(
                hidden_layer_sizes[1], activation='relu'),
            keras.layers.Dense(n_actions, activation='softmax')
        ])
        self._target_net = keras.Sequential([
            keras.layers.Dense(
                hidden_layer_sizes[0], activation='relu',
                input_shape=(n_states,)),
            keras.layers.Dense(
                hidden_layer_sizes[1], activation='relu'),
            keras.layers.Dense(n_actions, activation='softmax')
        ])

        # Set tensorflow loss function and optimizer
        self._loss_function: MeanSquaredError = MeanSquaredError()
        self._gradients: List[np.ndarray] = []

    @property
    def atype(self):
        """Return type of RL algorithm as string."""
        return 'DQN'

    def train(
            self,
            batch: Tuple[np.ndarray],
            batch_size: int = 16) -> None:
        """Train the policy based on a sample batch."""
        with tf.GradientTape() as tape:
            loss = self.calculate_q_loss(batch, batch_size)

        # Calculate and apply graidents.
        self._gradients = tape.gradient(
            loss, self._neural_net.trainable_variables)

    def calculate_q_loss(
            self,
            batch: Dict[str, np.ndarray],
            batch_size: int) -> Union[int, float]:
        """Calculate the state-value function loss."""
        prediction: EagerTensor = self._neural_net(batch[STATE])
        next_prediction: EagerTensor = tf.stop_gradient(
            self._target_net(batch[NEXT_STATE]))

        idx = tf.Variable(
            np.append(np.arange(batch_size).reshape(batch_size, 1),
                      batch[ACTION], axis=1), dtype=tf.int32)
        act_prediction = tf.reshape(
            tf.gather_nd(prediction, idx), (batch_size, 1))

        max_next_prediction = tf.reshape(
            tf.reduce_max(next_prediction, axis=1), (batch_size, 1))
        q_target = batch[REWARD] + self._gamma * (1 - batch[DONE]) * max_next_prediction

        return self._loss_function(act_prediction, q_target)

    def act(
            self,
            state: np.ndarray,
            epsilon: float = 0.3) -> Union[int, np.integer]:
        """Apply the target policy (π)."""
        if np.random.uniform() > epsilon:
            # print(self._neural_net(state).numpy())
            return np.argmax(self._neural_net(state).numpy())

        # Random action if below epsilon thershold
        return self.act_random()

    def act_b(
            self,
            state: np.ndarray,
            epsilon: float = 0.3) -> Union[int, np.integer]:
        """Apply the behavior policy (π)."""
        if np.random.uniform() > epsilon:
            return np.argmax(self._target_net(state).numpy())

        # Random action if below epsilon thershold
        return self.act_random()

    def act_random(self) -> np.int:
        """Random Policy Action based on Leader Location."""
        return np.random.randint(0, self._n_actions)

    def transfer_parameters(self) -> None:
        """Transfers parameters from main network to target network."""
        self._target_net.set_weights(self._neural_net.get_weights())

    def transfer_gradients(
            self,
            request: Gradients.Request) -> Gradients.Request:
        """Transfer calculated gradients to Gradients srv file."""
        request.layer.input_layer = (self._gradients[0].numpy()).flatten().tolist()
        request.layer.hidden_0 = (self._gradients[1].numpy()).flatten().tolist()
        request.layer.middle_0 = (self._gradients[2].numpy()).flatten().tolist()
        request.layer.hidden_1 = (self._gradients[3].numpy()).flatten().tolist()
        request.layer.output_layer = (self._gradients[4].numpy()).flatten().tolist()
        request.layer.output = (self._gradients[5].numpy()).flatten().tolist()
        return request

    def set_policy_weights(
            self,
            network_weights: List[np.ndarray]) -> None:
        """Set neural network weights for policy from list."""
        self._neural_net.set_weights(network_weights)

    def set_target_weights(
            self,
            network_weights: List[np.ndarray]) -> None:
        """Set target network weights for training from list."""
        self._target_net.set_weights(network_weights)

    def parse_and_set_policy_weights(
            self,
            response: Weights.Response()) -> None:
        """Parse and set neural network weights from srv response."""
        weights = []
        weights.append(
            np.array(response.layer.input_layer).reshape(
                self._n_states,
                self._hidden_layer_sizes[0]))
        weights.append(np.array(response.layer.hidden_0))
        weights.append(np.array(response.layer.middle_0).reshape(
            self._hidden_layer_sizes[0],
            self._hidden_layer_sizes[1]))
        weights.append(np.array(response.layer.hidden_1))
        weights.append(np.array(response.layer.output_layer).reshape(
            self._hidden_layer_sizes[1],
            self._n_actions))
        weights.append(np.array(response.layer.output))
        self.set_policy_weights(weights)

    def get_policy_weights(self) -> List[np.ndarray]:
        """Get weights for policy."""
        return self._neural_net.get_weights()

    def load_model(self, path_to_model: str) -> None:
        """Load model for inference or training use."""
        self._neural_net = keras.models.load_model(path_to_model)
