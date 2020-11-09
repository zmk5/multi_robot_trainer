"""Launch file for `Server` and `Worker` (Training)."""
import os

import rclpy

from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def get_default_file_path_server(file_name: str):
    """Get file path for default paramater file."""
    rclpy.logging.get_logger('Launch File').info(file_name)
    return os.path.join(
        get_package_share_directory('mrt_server'),
        'config',
        file_name)

def get_default_file_path_worker(file_name: str):
    """Get file path for default paramater file."""
    rclpy.logging.get_logger('Launch File').info(file_name)
    return os.path.join(
        get_package_share_directory('mrt_worker'),
        'config',
        file_name)


def generate_launch_description():
    """Launch file for training with both `Server` and one `Worker`."""
    return LaunchDescription([
        DeclareLaunchArgument(
            'yaml_file_server',
            default_value=[get_default_file_path_server('template.yaml')],
            description='Parameter file for experiment.'
        ),
        DeclareLaunchArgument(
            'yaml_file_worker',
            default_value=[get_default_file_path_worker('template.yaml')],
            description='Parameter file for experiment.'
        ),
        DeclareLaunchArgument(
            'n_workers',
            default_value=['1'],
            description='Number of workers that the Server node will oversee.'
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
            'output_file',
            default_value=[''],
            description='Name of output file for neural network model'
        ),
        DeclareLaunchArgument(
            'policy_type',
            default_value=['DQN'],
            description='Policy worker will use for training.'
        ),
        Node(
            package='mrt_server',
            executable='server_sync_node',  # server_sync_dual_node or server_async_node
            name='server_node',
            output='screen',
            parameters=[LaunchConfiguration('yaml_file_server')],
            arguments=[
                LaunchConfiguration('n_workers'),
                LaunchConfiguration('output_file'),
                LaunchConfiguration('policy_type'),
            ]
        ),
        Node(
            package='mrt_worker',
            executable='worker_sync_dual_node',
            name='worker_node',
            namespace=LaunchConfiguration('worker_ns'),
            output='screen',
            parameters=[LaunchConfiguration('yaml_file_worker')],
            arguments=[
                LaunchConfiguration('worker_id'),
                LaunchConfiguration('policy_type'),
            ]
        ),
    ])
