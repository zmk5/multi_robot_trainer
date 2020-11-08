"""Actor-Critic model class for RL experiments with neural net function approx.

Written by: Zahi Kakish (zmk5)
"""
from typing import List

import numpy as np

from tensorflow.keras.optimizers import Adam

from mrt_msgs.srv import Gradients
from mrt_msgs.srv import Weights

from mrt_server.policy.models import ActorCriticModel
from mrt_server.policy.models import ActorModel
from mrt_server.policy.models import CriticModel


class ServerPolicyActorCriticShared():
    """Actor-Critic Class containing all relevant RL information."""

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
        """Initialize the ServerPolicyActorCritic class."""
        self._n_states = n_states
        self._n_actions = n_actions
        self._alpha = alpha
        self._hidden_layer_sizes = hidden_layer_sizes

        # Set the neural net approximator
        self._neural_net = ActorCriticModel(
            n_states, n_actions, hidden_layer_sizes)
        self._neural_net.build((1, n_states))

        # Set tensorflow loss function and optimizer
        self._optimizer = Adam(learning_rate=self._alpha, clipvalue=1.0)
        self._weights: List[np.ndarray] = None

    @property
    def atype(self):
        """Return type of RL algorithm as string."""
        return 'A2C'

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
            np.array(request.layer.output),
            np.array(request.layer.critic_output_layer).reshape(
                self._hidden_layer_sizes[1],
                1),
            np.array(request.layer.critic_output),
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
        # Predict required for saving due to odd error found here:
        # https://github.com/tensorflow/tensorflow/issues/31057
        self._neural_net.predict(np.array([[0.1, 0.4, 0.4, 0.1, 1.0]]))
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
        response.layer.critic_output_layer = weights[6].flatten().tolist()
        response.layer.critic_output = weights[7].flatten().tolist()

        return response


class ServerPolicyActorCriticDual():
    """Soft Actor-Critic Class containing all relevant RL information."""

    __slots__ = [
        '_n_states',
        '_n_actions',
        '_alpha',
        '_neural_net',
        '_critic_net',
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
        """Initialize the ServerPolicyActorCritic class."""
        self._n_states = n_states
        self._n_actions = n_actions
        self._alpha = alpha
        self._hidden_layer_sizes = hidden_layer_sizes

        # Set the neural net approximator
        self._neural_net = ActorModel(n_states, n_actions, hidden_layer_sizes)
        self._critic_net = CriticModel(n_states, hidden_layer_sizes)
        self._neural_net.build((1, n_states))
        self._critic_net.build((1, n_states))

        # Set tensorflow loss function and optimizer
        self._optimizer = Adam(learning_rate=self._alpha, clipvalue=1.0)
        self._weights: List[np.ndarray] = None

    @property
    def atype(self):
        """Return type of RL algorithm as string."""
        return 'A2C'

    def optimize(self, gradients: List[np.ndarray], network_type: str) -> None:
        """Optimize global network policy."""
        if network_type == 'actor':
            self._optimizer.apply_gradients(
                zip(gradients, self._neural_net.trainable_variables))

        else:
            self._optimizer.apply_gradients(
                zip(gradients, self._critic_net.trainable_variables))

    def optimize_from_request(
            self,
            n_states: int,
            n_actions: int,
            request: Gradients.Request) -> None:
        """Optimize the policy from a gradient request."""
        n_outputs = n_actions if request.name == 'actor' else 1
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
                n_outputs),
            np.array(request.layer.output),
        ], request.name)

    def set_policy_weights(
            self,
            network_type: str,
            network_weights: List[np.ndarray]) -> None:
        """Set neural network weights for policy from list."""
        if network_type == 'actor':
            self._neural_net.set_weights(network_weights)

        else:
            self._critic_net.set_weights(network_weights)

    def get_policy_weights(self, network_type: str) -> List[np.ndarray]:
        """Get weights for policy."""
        if network_type == 'actor':
            return self._neural_net.get_weights()

        return self._critic_net.get_weights()

    def save_model(self, path_to_model: str, network_type: str) -> None:
        """Load model for inference or training use."""
        # Predict required for saving due to odd error found here:
        # https://github.com/tensorflow/tensorflow/issues/31057
        if network_type == 'actor':
            # self._neural_net.predict(np.arange(self._n_states).reshape(1, self._n_states))
            self._neural_net.predict(np.array([[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.92, 0.0]]))
            self._neural_net.save(path_to_model + '_actor')

        else:
            # self._critic_net.predict(np.arange(self._n_states).reshape(1, self._n_states))
            self._critic_net.predict(np.array([[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.92, 0.0]]))
            self._critic_net.save(path_to_model + '_critic')

    def set_response_weights(
            self,
            network_type: str,
            response: Weights.Response) -> Weights.Response:
        """Set the weights response to return to a worker."""
        weights = self.get_policy_weights(network_type)

        response.layer.input_layer = weights[0].flatten().tolist()
        response.layer.hidden_0 = weights[1].flatten().tolist()
        response.layer.middle_0 = weights[2].flatten().tolist()
        response.layer.hidden_1 = weights[3].flatten().tolist()
        response.layer.output_layer = weights[4].flatten().tolist()
        response.layer.output = weights[5].flatten().tolist()

        return response
