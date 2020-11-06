"""Synchronous Server node for training.

Written by: Zahi Kakish
"""
import sys
from typing import Dict
from typing import Optional

import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.service import Service

from mrt_msgs.srv import Gradients
from mrt_msgs.srv import Weights

from mrt_server.policy.actor_critic import ServerPolicyActorCriticShared
from mrt_server.policy.dqn import ServerPolicyDQN
from mrt_server.policy.ddqn import ServerPolicyDDQN
from mrt_server.policy.reinforce import ServerPolicyREINFORCE
from mrt_server.utils.parameters import ServerParameters


class ServerSync(Node):
    """Server node as an intermediary generating and training experiences."""

    def __init__(
            self,
            name: str,
            n_workers: Optional[int] = None,
            output_file: str = 'test_model',
            policy_type: str = 'DQN') -> None:
        """Initialize Server node class."""
        super().__init__(name)
        self.declare_parameters(
            namespace='',
            parameters=[
                ('number.states', 5),
                ('number.actions', 5),
                ('number.vertices', 4),
                ('number.episodes', 5000),
                ('number.workers', 3),
                ('hyperparameter.alpha', 0.3),
                ('hyperparameter.epsilon', 0.99),
                ('training_delay', 100),
                ('decay_rate', 500),
            ]
        )
        self._sp = ServerParameters(
            self.get_parameter('number.states').value,
            self.get_parameter('number.actions').value,
            self.get_parameter('number.episodes').value,
            self.get_parameter('number.workers').value,
            self.get_parameter('hyperparameter.alpha').value,
            self.get_parameter('hyperparameter.epsilon').value,
            self.get_parameter('training_delay').value
        )

        # Set number of workers if value is different from yaml.
        if n_workers is not None:
            self._sp.n_workers = n_workers
            self.get_logger().info(
                f'Will connect to {self._sp.n_workers} worker nodes.')

        # Set the output file for final model.
        if output_file is not None:
            self.output_file = output_file

        else:
            self.output_file = 'test_model'

        # Set the number of episodes to zero.
        # self._episodes = [0] * self._sp.n_workers
        self._episodes = np.zeros(self._sp.n_workers, np.int32)
        self._current_episode = 1
        self._timer = self.create_timer(0.1, self._synchronization_callback)

        # Initialize policy for inference and training.
        if policy_type == 'DQN':
            self._policy = ServerPolicyDQN(
                self._sp.n_vertices, self._sp.n_actions, self._sp.n_states,
                self._sp.alpha)

        elif policy_type == 'DDQN':
            self._policy = ServerPolicyDDQN(
                self._sp.n_vertices, self._sp.n_actions, self._sp.n_states,
                self._sp.alpha)

        elif policy_type == 'REINFORCE':
            self._policy = ServerPolicyREINFORCE(
                self._sp.n_vertices, self._sp.n_actions, self._sp.n_states,
                self._sp.alpha)

        elif policy_type == 'A2C':
            self._policy = ServerPolicyActorCriticShared(
                self._sp.n_vertices, self._sp.n_actions, self._sp.n_states,
                self._sp.alpha)

        # Create ROS service servers.
        self._srv: Dict[str, Service] = {
            'weights': self.create_service(
                Weights, '/weights', self._weights_callback),
            'gradients': self.create_service(
                Gradients, '/gradient', self._gradients_callback),
        }

    def _weights_callback(
            self,
            request: Weights.Request,
            response: Weights.Response) -> Weights.Response:
        """Send weights to client upon request."""
        if self._episodes[request.id] < self._current_episode:
            self.get_logger().info(
                f'Weights request from worker_{request.id} ' +
                f'for episode {self._episodes[request.id]}')
            response.synchronized = True
            return self._policy.set_response_weights(response)

        response.synchronized = False
        return response

    def _gradients_callback(
            self,
            request: Gradients.Request,
            response: Gradients.Response) -> Gradients.Response:
        """Receive gradients and optimize from client."""
        self.get_logger().info(
            f'Recieved gradients from worker node {request.id} ' +
            f'for episode {self._episodes[request.id]}.')

        if self._episodes[request.id] >= self._sp.training_delay:
            self._policy.optimize_from_request(
                self._sp.n_states,
                self._sp.n_vertices,
                self._sp.n_actions,
                request
            )

        else:
            self.get_logger().info(
                f'Skipping optimization till episode for wn_{request.id}!')

        # Send response.
        if self._episodes[request.id] == self._sp.n_episodes:
            self.get_logger().warn(
                f'Training ended for worker wn_{request.id}!')
            response.done = True
            self._policy.save_model(self.output_file)

        else:
            response.done = False
            self._episodes[request.id] += 1

        return response

    def _synchronization_callback(self) -> None:
        """Synchronize the current episodes of workers."""
        if np.all(self._episodes == self._current_episode):
            self._current_episode += 1


def main():
    """Start the Server Node."""
    rclpy.init()

    n_workers = None if sys.argv[1] == '' else int(sys.argv[1])
    output_file = None if sys.argv[2] == '' else sys.argv[2]
    policy_type = sys.argv[3]
    node = ServerSync('server_node', n_workers, output_file, policy_type)

    try:
        executor = MultiThreadedExecutor()
        rclpy.spin(node, executor)

    except KeyboardInterrupt:
        pass

    # Destroy the node explicitly
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
