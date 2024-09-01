from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # List to hold the launch descriptions for spawning robots and starting nodes
    launch_descriptions = []

    # Add TurtleBot3 model spawns
    for i in range(1, 6):  # Loop to spawn 5 TurtleBots
        launch_descriptions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    get_package_share_directory('turtlebot3_gazebo'), '/launch', '/turtlebot3_world.launch.py']),
                launch_arguments={
                    'use_sim_time': 'true',
                    'robot_name': f'turtlebot{i}',
                    'x_pose': str(i),  # Adjust the x position for each TurtleBot
                    'y_pose': '0',
                    'z_pose': '0.01'
                }.items()
            )
        )

    # Add custom SNN Formation Control Nodes for each TurtleBot
    for i in range(5):
        launch_descriptions.append(
            Node(
                package='snn_formation_control',
                executable='snn_formation_control_node',
                name=f'snn_formation_control_node_{i}',
                output='screen',
                parameters=[{'agent_id': i, 'robot_name': f'turtlebot{i+1}'}]
            )
        )

    return LaunchDescription(launch_descriptions)