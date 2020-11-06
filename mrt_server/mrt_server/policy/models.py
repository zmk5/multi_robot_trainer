"""Unique class-based models not formed using the keras Sequential class.

Written by: Zahi Kakish (zmk5)
"""
from typing import Optional
from typing import Tuple

from tensorflow import keras
from tensorflow import Tensor


#pylint: disable=W0223
class ActorCriticModel(keras.Model):
    """Class containining joint network for Actor-Critic NN model."""

    def __init__(self, n_states: int, n_vertices: int, n_actions: int) -> None:
        """Initialize ActorCriticModel class."""
        super().__init__()
        self._layer_hidden_0 = keras.layers.Dense(
            n_vertices * n_vertices, input_shape=(n_states,), activation="relu")
        self._layer_hidden_1 = keras.layers.Dense(
            n_vertices * n_vertices, activation="relu")
        self._layer_output_actor = keras.layers.Dense(n_actions)
        self._layer_output_critic = keras.layers.Dense(1)

    def call(
            self,
            inputs: Tensor,
            training: Optional[bool] = None,
            mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Forward pass of the Actor-Critic neural network."""
        x = self._layer_hidden_0(inputs)
        x = self._layer_hidden_1(x)
        return self._layer_output_actor(x), self._layer_output_critic(x)


class ActorModel(keras.Model):
    """Class containining separate network for Actor NN model."""

    def __init__(self, n_states: int, n_vertices: int, n_actions: int) -> None:
        """Initialize ActorModel class."""
        super().__init__()
        self._layer_hidden_0 = keras.layers.Dense(
            n_vertices * n_vertices, input_shape=(n_states,), activation="relu")
        self._layer_hidden_1 = keras.layers.Dense(
            n_vertices * n_vertices, activation="relu")
        # self._layer_hidden_2 = keras.layers.Dense(
        #     n_vertices * n_vertices, activation="relu")
        self._layer_output = keras.layers.Dense(n_actions)

    def call(
            self,
            inputs: Tensor,
            training: Optional[bool] = None,
            mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Forward pass of the Actor neural network."""
        x = self._layer_hidden_0(inputs)
        x = self._layer_hidden_1(x)
        # x = self._layer_hidden_2(x)
        return self._layer_output(x)


class CriticModel(keras.Model):
    """Class containining separate network for Critic NN model."""

    def __init__(self, n_states: int, n_vertices: int) -> None:
        """Initialize CriticModel class."""
        super().__init__()
        self._layer_hidden_0 = keras.layers.Dense(
            n_vertices * n_vertices, input_shape=(n_states,), activation="relu")
        self._layer_hidden_1 = keras.layers.Dense(
            n_vertices * n_vertices, activation="relu")
        # self._layer_hidden_2 = keras.layers.Dense(
        #     n_vertices * n_vertices, activation="relu")
        self._layer_output = keras.layers.Dense(1)

    def call(
            self,
            inputs: Tensor,
            training: Optional[bool] = None,
            mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Forward pass of the Critic neural network."""
        x = self._layer_hidden_0(inputs)
        x = self._layer_hidden_1(x)
        # x = self._layer_hidden_2(x)
        return self._layer_output(x)
