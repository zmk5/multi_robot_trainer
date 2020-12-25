"""Launch file for asynchronous `Server` (Training/Dual Network)."""
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
        get_package_share_directory('mrt_server'),
        'config',
        file_name)


def generate_launch_description():
    """Launch file for training node to training network."""
    return LaunchDescription([
        DeclareLaunchArgument(
            'yaml_file',
            default_value=[get_default_file_path('template.yaml')],
            description='Parameter file for experiment.'
        ),
        DeclareLaunchArgument(
            'n_workers',
            default_value=[''],
            description='Number of workers that the Server node will oversee.'
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
            executable='server_async_dual_node',
            name='server_node',
            output='screen',
            parameters=[LaunchConfiguration('yaml_file')],
            arguments=[
                LaunchConfiguration('n_workers'),
                LaunchConfiguration('output_file'),
                LaunchConfiguration('policy_type'),
            ]
        ),
    ])
