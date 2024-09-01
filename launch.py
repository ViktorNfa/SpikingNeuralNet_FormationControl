from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        # Spawn TurtleBot1
        ExecuteProcess(
            cmd=['ros2', 'run', 'gazebo_ros', 'spawn_entity.py', '-entity', 'turtlebot1', 
                 '-topic', 'robot_description', '-x', '0', '-y', '0', '-z', '0'],
            output='screen'
        ),
        # Spawn TurtleBot2
        ExecuteProcess(
            cmd=['ros2', 'run', 'gazebo_ros', 'spawn_entity.py', '-entity', 'turtlebot2', 
                 '-topic', 'robot_description', '-x', '1', '-y', '0', '-z', '0'],
            output='screen'
        ),
        # Spawn TurtleBot3
        ExecuteProcess(
            cmd=['ros2', 'run', 'gazebo_ros', 'spawn_entity.py', '-entity', 'turtlebot3', 
                 '-topic', 'robot_description', '-x', '2', '-y', '0', '-z', '0'],
            output='screen'
        ),
        # Spawn TurtleBot4
        ExecuteProcess(
            cmd=['ros2', 'run', 'gazebo_ros', 'spawn_entity.py', '-entity', 'turtlebot4', 
                 '-topic', 'robot_description', '-x', '3', '-y', '0', '-z', '0'],
            output='screen'
        ),
        # Spawn TurtleBot5
        ExecuteProcess(
            cmd=['ros2', 'run', 'gazebo_ros', 'spawn_entity.py', '-entity', 'turtlebot5', 
                 '-topic', 'robot_description', '-x', '4', '-y', '0', '-z', '0'],
            output='screen'
        ),

        # Start the SNN Formation Control Node for TurtleBot1
        Node(
            package='snn_formation_control',
            executable='snn_formation_control_node',
            name='snn_formation_control_node_0',
            output='screen',
            parameters=[{'agent_id': 0, 'robot_name': 'turtlebot1'}]
        ),
        # Start the SNN Formation Control Node for TurtleBot2
        Node(
            package='snn_formation_control',
            executable='snn_formation_control_node',
            name='snn_formation_control_node_1',
            output='screen',
            parameters=[{'agent_id': 1, 'robot_name': 'turtlebot2'}]
        ),
        # Start the SNN Formation Control Node for TurtleBot3
        Node(
            package='snn_formation_control',
            executable='snn_formation_control_node',
            name='snn_formation_control_node_2',
            output='screen',
            parameters=[{'agent_id': 2, 'robot_name': 'turtlebot3'}]
        ),
        # Start the SNN Formation Control Node for TurtleBot4
        Node(
            package='snn_formation_control',
            executable='snn_formation_control_node',
            name='snn_formation_control_node_3',
            output='screen',
            parameters=[{'agent_id': 3, 'robot_name': 'turtlebot4'}]
        ),
        # Start the SNN Formation Control Node for TurtleBot5
        Node(
            package='snn_formation_control',
            executable='snn_formation_control_node',
            name='snn_formation_control_node_4',
            output='screen',
            parameters=[{'agent_id': 4, 'robot_name': 'turtlebot5'}]
        ),
    ])
