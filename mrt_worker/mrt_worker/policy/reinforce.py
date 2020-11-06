"""REINFORCE model class for RL experiments with neural net function approx.

Written by: Zahi Kakish (zmk5)

"""
from typing import List
from typing import Optional
from typing import Union
from typing import Tuple

import numpy as np

import tensorflow as tf
from tensorflow import keras

from mean_field_msgs.srv import Gradients
from mean_field_msgs.srv import Weights


STATE = 0
ACTION = 1
REWARD = 2
NEXT_STATE = 3
NEXT_ACTION = 4
DONE = 5


class WorkerPolicyREINFORCE():
    """REINFORCE Class containing all relatvent RL information."""

    __slots__ = [
        '_n_states',
        '_n_actions',
        '_alpha',
        '_gamma',
        '_neural_net',
        '_target_net',
        '_loss_function',
        '_gradients',
        '_average_gradients',
    ]

    def __init__(
            self,
            n_states: int,
            n_actions: int,
            alpha: float,
            gamma: float,
            use_gpu: bool = False) -> None:
        """Initialize the WorkerPolicyEINFORCE class."""
        self._n_states = n_states
        self._n_actions = n_actions
        self._alpha: float = alpha
        self._gamma: float = gamma

        # Turn off GPU, if needed.
        if not use_gpu:
            tf.config.set_visible_devices([], 'GPU')

        # Set the neural net approximator
        self._neural_net = keras.Sequential([
            keras.layers.Dense(
                self._n_vertices * self._n_vertices,
                activation='relu',
                input_shape=(self._n_states,)),
            keras.layers.Dense(
                self._n_vertices * self._n_vertices,
                activation='relu'),
            keras.layers.Dense(self._n_actions)  # , activation='softmax')
        ])

        # Set tensorflow loss function and optimizer
        self._loss_function = keras.losses.MeanSquaredError()
        self._gradients: List[np.ndarray] = []

    @property
    def atype(self):
        """Return type of RL algorithm as string."""
        return 'REINFORCE'

    def train(
            self,
            batch: Tuple[np.ndarray],
            batch_size: int = 16) -> None:
        """Train the policy based on a sample batch."""
        ret_values = self.calculate_returns(batch, batch_size)
        with tf.GradientTape() as tape:
            pd_params = self.calculate_distribution_param(batch[STATE])
            # softmax_pd_params = tf.nn.softmax(pd_params)
            # ret_values = self.calculate_returns(batch, batch_size)
            loss = self.calculate_policy_loss(
                batch, pd_params, ret_values, batch_size)

        # Calculate and apply graidents.
        self._gradients = tape.gradient(
            loss, self._neural_net.trainable_variables)

    def calculate_distribution_param(self, state) -> np.ndarray:
        """Calculate the probability distrbution parameters from neural net."""
        return self._neural_net(state, training=True)

    def calculate_returns(
            self,
            batch: Tuple[np.ndarray],
            batch_size: int,
            previous_future_ret: float = 0.0) -> np.ndarray:
        """Calculate returns for each element in the sample batch."""
        # done = np.where(batch[REWARD] == 1, 1, 0)  # 0, 1)
        ret_value = np.zeros_like(batch[REWARD])
        future_ret = previous_future_ret
        for t in reversed(range(batch_size + 1)):
            ret_value[t] = future_ret = batch[REWARD][t] + \
                self._gamma * future_ret * (1 - batch[DONE][t])

        return ret_value

    def calculate_policy_loss(
            self,
            batch: Tuple[np.ndarray],
            dist_params: Union[np.ndarray, tf.Tensor],
            ret_values: np.ndarray,
            batch_size: int) -> Union[int, float]:
        """Calculate the policy loss."""
        log_probs = tf.math.log(dist_params)
        idx = tf.Variable(
            np.append(np.arange(batch_size + 1).reshape(batch_size + 1, 1),
                      batch[ACTION], axis=1),
            dtype=tf.int32
        )
        act_log_probs = tf.reshape(
            tf.gather_nd(log_probs, idx), (batch_size + 1, 1))
        return -1 * self._alpha * tf.math.reduce_sum(act_log_probs * ret_values)

    def act(
            self,
            state: np.ndarray,
            epsilon: Optional[float] = None) -> Union[int, np.integer]:
        """Apply the policy for a ROS inference service request."""
        _ = epsilon  # Unused by REINFORCE
        dist_parameters = self._neural_net(state)
        return tf.random.categorical(dist_parameters, 1)[0, 0].numpy()

    def transfer_parameters(self) -> None:
        """Transfers parameters from main network to target network."""

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

    def parse_and_set_policy_weights(
            self,
            response: Weights.Response()) -> None:
        """Parse and set neural network weights from srv response."""
        weights = []
        weights.append(
            np.array(response.layer.input_layer).reshape(
                self._n_states,
                self._n_vertices * self._n_vertices))
        weights.append(np.array(response.layer.hidden_0))
        weights.append(np.array(response.layer.middle_0).reshape(
            self._n_vertices * self._n_vertices,
            self._n_vertices * self._n_vertices))
        weights.append(np.array(response.layer.hidden_1))
        weights.append(np.array(response.layer.output_layer).reshape(
            self._n_vertices * self._n_vertices,
            self._n_actions))
        weights.append(np.array(response.layer.output))
        self.set_policy_weights(weights)

    def get_policy_weights(self) -> List[np.ndarray]:
        """Get weights for policy."""
        return self._neural_net.get_weights()

    def load_model(self, path_to_model: str) -> None:
        """Load model for inference or training use."""
        self._neural_net = keras.models.load_model(path_to_model)
