"""Launch file for testing network."""
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
    """Launch file for testing network over a set of experiences."""
    return LaunchDescription([
        DeclareLaunchArgument(
            'yaml_file',
            default_value=[get_default_file_path('template.yaml')],
            description='Parameter file for experiment.'
        ),
        DeclareLaunchArgument(
            'policy_type',
            default_value=['DQN'],
            description='Policy worker will use for training.'
        ),
        Node(
            package='mrt_worker',
            executable='tester_node',
            name='tester_node',
            output='screen',
            parameters=[LaunchConfiguration('yaml_file')],
            arguments=[['0'], LaunchConfiguration('policy_type')],
        ),
    ])
