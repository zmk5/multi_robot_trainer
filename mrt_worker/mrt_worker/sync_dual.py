"""Synchronous Worker node for training.

Written by: Zahi Kakish (zmk5)
"""
import sys
from typing import Dict

import numpy as np

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.client import Client
from rclpy.executors import Executor
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Float32

from mrt_msgs.srv import Gradients
from mrt_msgs.srv import Weights

from mrt_worker.base import WorkerBase
from mrt_worker.utils.parameters import Flags


class WorkerSyncDual(WorkerBase):
    """Worker node for generating and training experiences."""

    def __init__(
            self,
            worker_id: int,
            name: str,
            policy_type: str = 'DQN') -> None:
        """Initialize Worker Node class."""
        super().__init__(worker_id, name, policy_type)
        self.episode = 0

        # Set timer flags, futures, and callback groups.
        self.flag = Flags(True, False, False, False, True)
        self._future_weights = {
            'actor': rclpy.Future(),
            'critic': rclpy.Future(),
        }
        self._future_gradients = {
            'actor': rclpy.Future(),
            'critic': rclpy.Future(),
        }
        self._cb_group = ReentrantCallbackGroup()
        self._total_reward = 0
        self._update = 10

        # Create ROS service clients and publisher.
        self._cli: Dict[str, Client] = {
            'weights': self.create_client(
                Weights, '/weights', callback_group=self._cb_group),
            'gradients': self.create_client(
                Gradients, '/gradient', callback_group=self._cb_group),
        }
        self._pub = self.create_publisher(
            Float32, 'total_reward', 10, callback_group=self._cb_group)

        # Check if ROS service servers are available.
        while not self._cli['weights'].wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                '`Weights` service not available, waiting again...')

        while not self._cli['gradients'].wait_for_service(timeout_sec=1.0):
            self.get_logger().info(
                '`Gradients` service not available, waiting again...')

    def pull(self, executor: Executor) -> None:
        """Pull neural net parameters from global ROS node."""
        if self.flag.action:
            self.get_logger().info('Sending for weights!')
            self.get_logger().info('Waiting for worker synchronization...')
            self.flag.action = False

        # Attempt to pull from server.
        self.pull_server_parameters(executor, 'actor')
        actor_result = self._future_weights['actor'].result()
        self.pull_server_parameters(executor, 'critic')
        critic_result = self._future_weights['critic'].result()

        # if result.synchronized:  # Only get weights once workers synchronize.
        if actor_result.synchronized and critic_result.synchronized:
            self.get_logger().info('New weights recieved!')
            # Once srv response received, set weights for policy
            self._policy.parse_and_set_policy_weights('actor', actor_result)
            self._policy.parse_and_set_policy_weights('critic', critic_result)

            # Move to `collect` section
            self.flag.shift_to_collect_cycle()

    def collect(self) -> int:
        """Collect a set of trajectories and store."""
        self.get_logger().info('Generating experiences...')
        # Decay epsilon if requred
        self._wp.decay_epsilon(self.episode, 0.001)

        # Generate experiences
        trajectory_length, total_reward = self.step()
        self.publish(total_reward)
        self.flag.shift_to_compute_cycle()

        # Total reward section
        self._total_reward += total_reward
        if (self.episode + 1) % self._update == 0:
            self.get_logger().error(
                f'Expected Reward: {self._total_reward / self._update}')
            self._total_reward = 0

        return trajectory_length

    def compute(self, step: int) -> None:
        """Compute the gradient of the local network."""
        if self.episode >= self._wp.training_delay:
            # Transfer network parameters if episode 0 or 100 * n.
            if self.episode % 100 == 0:
                self._policy.transfer_parameters()

            self.get_logger().info('Computing gradients...')
            if self.atype in ['REINFORCE', 'A2C']:
                batch = self._db.sample_batch(step, 'all')
                self._policy.train(batch, step)

            else:
                batch = self._db.sample_batch(self._wp.batch_size)
                self._policy.train(batch, self._wp.batch_size)

        else:
            self.get_logger().warn(
                'Skipping computing gradients till episode ' +
                f'{self._wp.training_delay}!'
            )

        # Move to `push` section
        self.flag.shift_to_push_cycle()

    def push(self, executor: Executor) -> bool:
        """Push neural net parameters to global ROS node."""
        if self.episode >= self._wp.training_delay:
            self.get_logger().info('Pushing gradients...')
            self.push_gradients(executor, 'actor')
            self.push_gradients(executor, 'critic')

        else:
            self.get_logger().info('Pushing empty gradients...')
            self.push_empty_gradients(executor, 'actor')
            self.push_empty_gradients(executor, 'critic')

        experiment_done = self._future_gradients['actor'].result().done

        if experiment_done:
            self.get_logger().warn('Experiment complete!')

        else:
            self.get_logger().info(f'Episode {self.episode} complete!')

            # Move to top `pull` section
            self.flag.reset()

        # Increment episodes
        self.episode += 1

        return experiment_done

    def test(self, n_test_runs: int = 10) -> None:
        """Test the current network to check how well the networks trained."""
        steps: np.ndarray = np.zeros(n_test_runs)
        rewards: np.ndarray = np.zeros(n_test_runs)
        for t in range(n_test_runs):
            steps[t], rewards[t] = self.step(collect=False)

        self.get_logger().warn('---------- TEST RUN RESULTS ----------')
        self.get_logger().warn(f'Average: {steps.mean()}')
        self.get_logger().warn(f'STD: {steps.std()}')
        self.get_logger().warn(f'Median: {np.median(steps)}')
        self.get_logger().warn(f'Average Reward: {rewards.mean()}')

    def pull_server_parameters(self, executor: Executor, network: str) -> None:
        """Pull neural net parameters FROM server ROS node."""
        request = Weights.Request()
        request.id = self._worker_id
        request.name = network
        self._future_weights[network] = self._cli['weights'].call_async(request)
        rclpy.spin_until_future_complete(
            self, self._future_weights[network], executor)

    def push_gradients(self, executor: Executor, network: str) -> None:
        """Push neural net gradients TO server ROS node."""
        request = self._policy.transfer_gradients(Gradients.Request(), network)
        request.id = self._worker_id
        request.name = network
        self._future_gradients[network] = self._cli['gradients'].call_async(request)
        rclpy.spin_until_future_complete(
            self, self._future_gradients[network], executor)

    def push_empty_gradients(self, executor: Executor, network: str) -> None:
        """Push empty neural net gradients TO server ROS node."""
        request = Gradients.Request()
        request.id = self._worker_id
        request.name = network
        self._future_gradients[network] = self._cli['gradients'].call_async(request)
        rclpy.spin_until_future_complete(
            self, self._future_gradients[network], executor)

    def publish(self, total_reward: float) -> None:
        """Publish the total reward for a experience trajectory."""
        msg = Float32()
        msg.data = total_reward
        self._pub.publish(msg)

    def upkeep(self) -> None:
        """Run policy dependent end-of-experiment upkeep on database, etc."""
        if self.atype in ['REINFORCE', 'A2C', 'A2CD']:
            self._db.reset()


def main():
    """Start the Worker Node."""
    rclpy.init()

    worker_id = int(sys.argv[1])
    policy_type = sys.argv[2]
    node = WorkerSyncDual(worker_id, 'worker_node', policy_type)

    try:
        executor = MultiThreadedExecutor()
        steps = 0

        while rclpy.ok():
            if node.flag.pull:
                node.pull(executor)

            elif node.flag.collect:
                steps = node.collect()

            elif node.flag.compute:
                node.compute(steps)

            elif node.flag.push:
                experiment_complete = node.push(executor)
                node.upkeep()

                # End experiment if passed number of max episodes.
                if experiment_complete:
                    node.test(100)
                    break

    except KeyboardInterrupt:
        pass

    # Destroy the node explicitly
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
