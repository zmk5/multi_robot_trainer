"""Launch file for synchronous `Worker` (Training/Dual Network)."""
import os

import rclpy

from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def get_default_file_path(file_name: str):
    """Get file path for default paramater file."""
    rclpy.logging.get_logger('Launch File').info(file_name)
    return os.path.join(
        get_package_share_directory('mrt_worker'),
        'config',
        file_name)


def generate_launch_description():
    """Launch file for worker node to generate experiences."""
    return LaunchDescription([
        DeclareLaunchArgument(
            'yaml_file',
            default_value=[get_default_file_path('template.yaml')],
            description='Parameter file for experiment.'
        ),
        DeclareLaunchArgument(
            'worker_id',
            default_value=['0'],
            description='The Worker node ID number.'
        ),
        DeclareLaunchArgument(
            'worker_ns',
            default_value=['wn_0'],
            description='The Worker node namespace.'
        ),
        DeclareLaunchArgument(
            'policy_type',
            default_value=['DQN'],
            description='Policy worker will use for training.'
        ),
        Node(
            package='mrt_worker',
            executable='worker_sync_dual_node',
            name='worker_node',
            namespace=LaunchConfiguration('worker_ns'),
            output='screen',
            parameters=[LaunchConfiguration('yaml_file')],
            arguments=[
                LaunchConfiguration('worker_id'),
                LaunchConfiguration('policy_type'),
            ]
        ),
    ])
