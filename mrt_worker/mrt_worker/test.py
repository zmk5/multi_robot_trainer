"""Worker node for testing.

Written by: Zahi Kakish (zmk5)
"""
import sys

import numpy as np

import rclpy

from mrt_worker.base import WorkerBase


class WorkerTestNode(WorkerBase):
    """Worker node for generating and training experiences."""

    def __init__(self, worker_id: int, name: str, policy_type: str = 'DQN'):
        """Initialize Tester Node class."""
        super().__init__(worker_id, name, policy_type)

        # Create ROS timer.
        self._timer = self.create_timer(0.1, self.timer_callback)

        # Load model for testing.
        self._policy.load_model('test_model')

        # Initialize testing parameters and data storage.
        self._episode = 0
        self._total_episodes = 100
        self._total_reward = 0
        self._n_steps_to_equilibrium = np.zeros(self._total_episodes)
        self._ebc_count = 0  # Early break count

    def timer_callback(self):
        """Repeat testing operation."""
        self.get_logger().info(
            f'Episode: {self._episode} / {self._total_episodes}')

        if self._episode < self._total_episodes:
            # Generate experiences using RL.
            self._n_steps_to_equilibrium[self._episode], reward = self.step()
            self._total_reward += reward

            # Update EBC if applicable.
            if self._n_steps_to_equilibrium[self._episode] < self._wp.n_iterations - 1:
                self._ebc_count += 1

            # Increment episdoes number.
            self._episode += 1

        else:
            # Calculate and log final results.
            self.get_logger().warn('---------- FINAL ----------')
            self.get_logger().warn(
                f'Average: {np.mean(self._n_steps_to_equilibrium)}')
            self.get_logger().warn(
                f'STD: {np.std(self._n_steps_to_equilibrium)}')
            self.get_logger().warn(
                f'Median: {np.median(self._n_steps_to_equilibrium)}')
            self.get_logger().warn(
                f'Average Reward: {self._total_reward / self._total_episodes}')
            self.get_logger().warn(f'EBC: {self._ebc_count}')

            # Destroy node
            self._timer.destroy()
            raise KeyboardInterrupt  # Forces out of rclpy.spin() loop.


def main():
    """Start the Test Node."""
    rclpy.init()

    worker_id = int(sys.argv[1])
    policy_type = sys.argv[2]
    node = WorkerTestNode(worker_id, 'tester_node', policy_type)

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    # Destroy the node explicitly
    node.destroy_node()
    rclpy.shutdown()
