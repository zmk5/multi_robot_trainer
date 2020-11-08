"""DQN model class for RL experiments with neural net function approx.

Written by: Zahi Kakish (zmk5)
"""
from typing import List

import numpy as np

from tensorflow import keras
from tensorflow.keras.optimizers import Adam

from mrt_msgs.srv import Gradients
from mrt_msgs.srv import Weights


class ServerPolicyDQN():
    """DQN Class containing all relevant RL information."""

    __slots__ = [
        '_n_states',
        '_n_actions',
        '_alpha',
        '_neural_net',
        '_optimizer',
        '_weights',
        '_hidden_layer_sizes',
    ]

    def __init__(
            self,
            n_states: int,
            n_actions: int,
            alpha: float,
            hidden_layer_sizes: List[int]) -> None:
        """Initialize the PolicyDQN class."""
        self._n_states = n_states
        self._n_actions = n_actions
        self._alpha = alpha
        self._hidden_layer_sizes = hidden_layer_sizes

        # Check to make sure hidden layer sizes are correct.
        if len(hidden_layer_sizes) != 2:
            raise ValueError('Hidden layers must be a list of size 2!')

        # Set Q-function Neural Net Approximator and Target Network
        self._neural_net = keras.Sequential([
            keras.layers.Dense(
                hidden_layer_sizes[0], activation='relu',
                input_shape=(n_states,)),
            keras.layers.Dense(
                hidden_layer_sizes[1], activation='relu'),
            keras.layers.Dense(n_actions, activation='softmax')
        ])

        # Set tensorflow loss function and optimizer
        self._optimizer = Adam(learning_rate=self._alpha)
        self._weights: List[np.ndarray] = None

    @property
    def atype(self):
        """Return type of RL algorithm as string."""
        return 'DQN'

    def optimize(self, gradients: List[np.ndarray]) -> None:
        """Optimize global network policy."""
        self._optimizer.apply_gradients(
            zip(gradients, self._neural_net.trainable_variables))

    def optimize_from_request(
            self,
            n_states: int,
            n_actions: int,
            request: Gradients.Request) -> None:
        """Optimize the policy from a gradient request."""
        self.optimize([
            np.array(request.layer.input_layer).reshape(
                n_states,
                self._hidden_layer_sizes[0]),
            np.array(request.layer.hidden_0),
            np.array(request.layer.middle_0).reshape(
                self._hidden_layer_sizes[0],
                self._hidden_layer_sizes[1]),
            np.array(request.layer.hidden_1),
            np.array(request.layer.output_layer).reshape(
                self._hidden_layer_sizes[1],
                n_actions),
            np.array(request.layer.output)
        ])

    def set_policy_weights(
            self,
            network_weights: List[np.ndarray]) -> None:
        """Set neural network weights for policy from list."""
        self._neural_net.set_weights(network_weights)

    def get_policy_weights(self) -> List[np.ndarray]:
        """Get weights for policy."""
        return self._neural_net.get_weights()

    def save_model(self, path_to_model: str) -> None:
        """Load model for inference or training use."""
        self._neural_net.save(path_to_model)

    def set_response_weights(
            self,
            response: Weights.Response) -> Weights.Response:
        """Set the weights response to return to a worker."""
        weights = self.get_policy_weights()
        response.layer.input_layer = weights[0].flatten().tolist()
        response.layer.hidden_0 = weights[1].flatten().tolist()
        response.layer.middle_0 = weights[2].flatten().tolist()
        response.layer.hidden_1 = weights[3].flatten().tolist()
        response.layer.output_layer = weights[4].flatten().tolist()
        response.layer.output = weights[5].flatten().tolist()

        return response
