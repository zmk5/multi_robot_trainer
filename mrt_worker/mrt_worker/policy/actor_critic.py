"""Actor-Critic policy class for RL experiments with neural net function approx.

This network uses a shared network architecture, i.e. a singular network that
has two ouputs: one for the actor and one for the critic.

Written by: Zahi Kakish (zmk5)

"""
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

import tensorflow as tf
from tensorflow import keras

from mean_field_msgs.srv import Gradients
from mean_field_msgs.srv import Weights

from mrt_worker.policy.models import ActorCriticModel
from mrt_worker.policy.models import ActorModel
from mrt_worker.policy.models import CriticModel
from mrt_worker.policy.reinforce import WorkerPolicyREINFORCE


STATE = 0
ACTION = 1
REWARD = 2
NEXT_STATE = 3
NEXT_ACTION = 4
DONE = 5


class WorkerPolicyActorCriticShared(WorkerPolicyREINFORCE):
    """Actor-Critic Shared Network Class containing relatvent RL information."""

    def __init__(
            self,
            n_states: int,
            n_actions: int,
            alpha: float,
            gamma: float,
            hidden_layer_sizes: List[int],
            use_gpu: bool = False) -> None:
        """Initialize the ModelActorCritic class."""
        super().__init__(
            n_states, n_actions, alpha, gamma, hidden_layer_sizes, use_gpu)

        # Use Actor-Critic model instead of that within WorkerPolicyREINFORCE.
        self._neural_net = ActorCriticModel(
            n_states, n_actions, hidden_layer_sizes)

        # Huber loss
        self._loss_function = keras.losses.Huber(
            reduction=keras.losses.Reduction.SUM)

        # Build and compile Actor-Critic model
        self._neural_net.build((1, n_states))

    @property
    def atype(self):
        """Return type of RL algorithm as string."""
        return 'A2C'

    def train(
            self,
            batch: Tuple[np.ndarray],
            batch_size: int = 16) -> None:
        """Train the policy based on a sample batch."""
       #  _, values_pred = self._neural_net(batch[STATE])
        _, next_values_pred = self._neural_net(batch[NEXT_STATE])
        returns = self.calculate_nstep_returns(
            batch, batch_size, next_values_pred)
        # returns = self.calculate_gae_returns(
        #     batch, batch_size, values_pred, next_values_pred)

        with tf.GradientTape() as tape:
            # Compute the action probs and value for current and next state.
            action_logits, values = self._neural_net(batch[STATE])
            action_probs = tf.nn.softmax(action_logits)
            # print(f'Grads: {[var.name for var in tape.watched_variables()]}')

            # Compute the returns and loss.
            loss = self.calculate_actor_critic_loss(
                batch[ACTION], action_probs, returns, values, batch_size)

        # Calculate and apply graidents.
        self._gradients = tape.gradient(
            loss, self._neural_net.trainable_variables)

    def calculate_nstep_returns(
            self,
            batch: Tuple[np.ndarray],
            batch_size: int,
            next_v_pred: tf.Tensor) -> np.ndarray:
        """Calculate n-step advantage returns."""
        ret_value = np.zeros_like(batch[REWARD])
        # future_ret = next_v_pred.numpy()[-1]
        # print(f'Future Return: {future_ret}')
        future_ret = 0.0

        for t in reversed(range(batch_size + 1)):
            ret_value[t] = future_ret = batch[REWARD][t] + \
                self._gamma * future_ret * (1 - batch[DONE][t])

        return ret_value

    def calculate_gae_returns(
            self,
            batch: Tuple[np.ndarray],
            batch_size: int,
            v_preds: tf.Tensor,
            next_v_pred: tf.Tensor) -> np.ndarray:
        """Calculate Generalaized Advantage Estimation (GAE) returns."""
        gaes = np.zeros_like(batch[REWARD])
        future_gae = 0.0

        for t in reversed(range(batch_size + 1)):
            delta = batch[REWARD][t] + self._gamma * next_v_pred[t] * (1 - batch[DONE][t]) - v_preds[t]
            gaes[t] = future_gae = delta + self._gamma * 0.95 * (1 - batch[DONE][t]) * future_gae  # lambda = 0.95

        return gaes

    def calculate_actor_critic_loss(
            self,
            action_batch: np.ndarray,
            action_probs: Union[np.ndarray, tf.Tensor],
            returns: Union[np.ndarray, tf.Tensor],
            values: Union[np.ndarray, tf.Tensor],
            batch_size: int) -> tf.Tensor:
        """Calculate the Actor-Critic network loss."""
        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        idx = tf.Variable(
            np.append(np.arange(batch_size + 1).reshape(batch_size + 1, 1),
                      action_batch, axis=1),
            dtype=tf.int32
        )
        act_log_probs = tf.reshape(
            tf.gather_nd(action_log_probs, idx), (batch_size + 1, 1))
        actor_loss = -1 * tf.math.reduce_sum(act_log_probs * advantage)
        print(f'Actor Loss: {actor_loss}')

        critic_loss = self._loss_function(values, returns)
        print(f'Critic Loss: {critic_loss}')
        print(f'Loss: {actor_loss + critic_loss}')

        return actor_loss + critic_loss

    def act(
            self,
            state: np.ndarray,
            epsilon: Optional[float] = None) -> Union[int, np.integer]:
        """Apply the policy for a ROS inference service request."""
        _ = epsilon  # Unused by REINFORCE
        dist_parameters, _ = self._neural_net(state)
        return tf.random.categorical(dist_parameters, 1)[0, 0].numpy()

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
        request.layer.critic_output_layer = (self._gradients[6].numpy()).flatten().tolist()
        request.layer.critic_output = (self._gradients[7].numpy()).flatten().tolist()

        return request

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
        weights.append(np.array(response.layer.critic_output_layer).reshape(
            self._hidden_layer_sizes[1],
            1))
        weights.append(np.array(response.layer.critic_output))
        self.set_policy_weights(weights)


class WorkerPolicyActorCriticDual(WorkerPolicyREINFORCE):
    """Actor-Critic Class containing all relatvent RL information."""

    def __init__(
            self,
            n_states: int,
            n_actions: int,
            alpha: float,
            gamma: float,
            hidden_layer_sizes: List[int],
            use_gpu: bool = False) -> None:
        """Initialize the ModelActorCritic class."""
        super().__init__(n_states, n_actions, alpha, gamma, hidden_layer_sizes, use_gpu)

        # Use Actor-Critic model instead of that within WorkerPolicyREINFORCE.
        self._neural_net = ActorModel(n_states, n_actions, hidden_layer_sizes)
        self._critic_net = CriticModel(n_states, hidden_layer_sizes)

        # Huber loss
        self._loss_function = keras.losses.Huber(
            reduction=keras.losses.Reduction.SUM)

        # Build and compile Actor-Critic model
        self._neural_net.build((1, n_states))
        self._critic_net.build((1, n_states))

        # Create additional variable for critic gradient.
        self._critic_gradients: List[np.ndarray] = []


    @property
    def atype(self) -> str:
        """Return type of RL algorithm as string."""
        return 'A2C'

    def train(
            self,
            batch: Tuple[np.ndarray],
            batch_size: int = 16) -> None:
        """Train the soft actor-critic policy based on a sample batch."""
        # values_pred = self._critic_net(batch[STATE])
        next_values_pred = self._critic_net(batch[NEXT_STATE])
        returns = self.calculate_nstep_returns(
            batch, batch_size, next_values_pred)
        # returns = self.calculate_gae_returns(
        #     batch, batch_size, values_pred, next_values_pred)

        self.train_actor(returns, batch, batch_size)
        self.train_critic(returns, batch, batch_size)

    def train_actor(
            self,
            returns: np.ndarray,
            batch: Tuple[np.ndarray],
            batch_size: int = 16) -> None:
        """Train the actor policy based on a sample batch."""
        values = self._critic_net(batch[STATE])
        with tf.GradientTape() as tape:
            # Compute the action probs and value for current and next state.
            action_logits = self._neural_net(batch[STATE])
            action_probs = tf.nn.softmax(action_logits)
            # print(f'Grads: {[var.name for var in tape.watched_variables()]}')

            # Compute the returns and loss.
            loss = self.calculate_actor_loss(
                batch[ACTION], action_probs, returns, values, batch_size)

        # Calculate and apply graidents.
        self._gradients = tape.gradient(
            loss, self._neural_net.trainable_variables)

    def train_critic(
            self,
            returns: np.ndarray,
            batch: Tuple[np.ndarray],
            batch_size: int = 16) -> None:
        """Train the actor policy based on a sample batch."""
        _ = batch_size  # TODO: Batch size is not used.
        with tf.GradientTape() as tape:
            # Compute the value for current and next state.
            values = self._critic_net(batch[STATE])

            # Compute the returns and loss.
            # print(f'Grads: {[var.name for var in tape.watched_variables()]}')
            loss = self.calculate_critic_loss(returns, values)

        # Calculate and apply graidents.
        self._critic_gradients = tape.gradient(
            loss, self._critic_net.trainable_variables)

    def calculate_nstep_returns(
            self,
            batch: Tuple[np.ndarray],
            batch_size: int,
            next_v_pred: tf.Tensor) -> np.ndarray:
        """Calculate n-step advantage returns."""
        ret_value = np.zeros_like(batch[REWARD])
        # try:
        #     future_ret = next_v_pred.numpy()[-1]
        #     print(f'Future Return: {future_ret}')

        # except IndexError:
        #     future_ret = next_v_pred.numpy()
        future_ret = 0.0

        for t in reversed(range(batch_size + 1)):
            ret_value[t] = future_ret = batch[REWARD][t] + \
                self._gamma * future_ret * (1 - batch[DONE][t])

        return ret_value

    def calculate_gae_returns(
            self,
            batch: Tuple[np.ndarray],
            batch_size: int,
            v_preds: tf.Tensor,
            next_v_pred: tf.Tensor) -> np.ndarray:
        """Calculate Generalaized Advantage Estimation (GAE) returns."""
        gaes = np.zeros_like(batch[REWARD])
        future_gae = 0.0

        for t in reversed(range(batch_size + 1)):
            delta = batch[REWARD][t] + self._gamma * next_v_pred[t] * (1 - batch[DONE][t]) - v_preds[t]
            gaes[t] = future_gae = delta + self._gamma * 0.95 * (1 - batch[DONE][t]) * future_gae  # lambda = 0.95

        return gaes

    def calculate_actor_loss(
            self,
            action_batch: np.ndarray,
            action_probs: Union[np.ndarray, tf.Tensor],
            returns: Union[np.ndarray, tf.Tensor],
            values: Union[np.ndarray, tf.Tensor],
            batch_size: int) -> tf.Tensor:
        """Calculate the Actor network loss."""
        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)
        idx = tf.Variable(
            np.append(np.arange(batch_size + 1).reshape(batch_size + 1, 1),
                      action_batch, axis=1),
            dtype=tf.int32
        )
        act_log_probs = tf.reshape(
            tf.gather_nd(action_log_probs, idx), (batch_size + 1, 1))

        # Actor Loss with Entropy
        # entropy = np.sum(
        #     -1 * action_probs.numpy() * np.log(action_probs.numpy()), axis=1)
        # print(f'Entropy: {entropy.mean()}')
        # actor_loss = -1 * tf.math.reduce_sum(act_log_probs * advantage) - (0.0 * entropy.mean())

        # entropy = np.sum(
        #     -1 * action_probs * np.log(action_probs), axis=1
        # ).reshape(batch_size + 1, 1)
        # actor_loss = -1 * tf.math.reduce_sum((act_log_probs * advantage) - (0.01 * entropy))

        actor_loss = -1 * tf.math.reduce_sum(act_log_probs * advantage)
        # print(f'Actor Loss: {actor_loss}')

        return actor_loss

    def calculate_critic_loss(
            self,
            returns: Union[np.ndarray, tf.Tensor],
            values: Union[np.ndarray, tf.Tensor]) -> tf.Tensor:
        """Calculate the Critic network loss."""
        critic_loss = self._loss_function(values, returns)
        # print(f'Critic Loss: {critic_loss}')

        return critic_loss

    def transfer_gradients(
            self,
            request: Gradients.Request,
            gradient_type: str = 'actor') -> Gradients.Request:
        """Transfer calculated gradients to Gradients srv file."""
        if gradient_type == 'actor':
            request.layer.input_layer = (self._gradients[0].numpy()).flatten().tolist()
            request.layer.hidden_0 = (self._gradients[1].numpy()).flatten().tolist()
            request.layer.middle_0 = (self._gradients[2].numpy()).flatten().tolist()
            request.layer.hidden_1 = (self._gradients[3].numpy()).flatten().tolist()
            request.layer.output_layer = (self._gradients[4].numpy()).flatten().tolist()
            request.layer.output = (self._gradients[5].numpy()).flatten().tolist()

        else:
            request.layer.input_layer = (self._critic_gradients[0].numpy()).flatten().tolist()
            request.layer.hidden_0 = (self._critic_gradients[1].numpy()).flatten().tolist()
            request.layer.middle_0 = (self._critic_gradients[2].numpy()).flatten().tolist()
            request.layer.hidden_1 = (self._critic_gradients[3].numpy()).flatten().tolist()
            request.layer.output_layer = (self._critic_gradients[4].numpy()).flatten().tolist()
            request.layer.output = (self._critic_gradients[5].numpy()).flatten().tolist()

        return request

    def parse_and_set_policy_weights(
            self,
            network_type: str,
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

        if network_type == 'actor':
            weights.append(np.array(response.layer.output_layer).reshape(
                self._hidden_layer_sizes[1],
                self._n_actions))
            weights.append(np.array(response.layer.output))

        else:
            weights.append(np.array(response.layer.output_layer).reshape(
                self._hidden_layer_sizes[1],
                1))
            weights.append(np.array(response.layer.output))

        self.set_policy_weights(network_type, weights)

    def set_policy_weights(
            self,
            network_type: str,
            network_weights: List[np.ndarray]) -> None:
        """Set neural network weights for policy from list."""
        if network_type == 'actor':
            self._neural_net.set_weights(network_weights)

        else:
            self._critic_net.set_weights(network_weights)

    def load_model(self, path_to_model: str) -> None:
        """Load model for inference or training use."""
        self._neural_net = keras.models.load_model(path_to_model + '_actor')
        self._critic_net = keras.models.load_model(path_to_model + '_critic')
