import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time
import pickle

def load_simulation_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

class SNNFormationControlNode(Node):
    def __init__(self, simulation_data, agent_id, robot_name):
        super().__init__('snn_formation_control_node_' + str(agent_id))
        self.publisher_ = self.create_publisher(Twist, '/' + robot_name + '/cmd_vel', 10)
        self.simulation_data = simulation_data
        self.agent_id = agent_id
        self.control_data = simulation_data['control'][self.agent_id]
        self.state_data = simulation_data['state'][self.agent_id]
        self.time_step = 0.1  # Assuming time step is 0.1 seconds
        self.timer_ = self.create_timer(self.time_step, self.timer_callback)
        self.start_time = time.time()
        self.current_step = 0

    def timer_callback(self):
        # Check if 30 seconds have passed
        if time.time() - self.start_time > 30:
            self.get_logger().info(f"Agent {self.agent_id}: Simulation complete.")
            self.destroy_node()
            return
        
        # Get control inputs for the current time step
        if self.current_step < len(self.control_data):
            u = self.control_data[self.current_step]
            twist = Twist()
            twist.linear.x = u[0]  # Linear velocity
            twist.angular.z = u[1]  # Angular velocity
            self.publisher_.publish(twist)
            self.current_step += 1
        else:
            self.get_logger().info(f"Agent {self.agent_id}: No more control data.")

def main(args=None):
    rclpy.init(args=args)

    # Load simulation data
    file_path = '~/ros2_ws/src/snn_formation_control/dataset/simulation.pkl'
    simulation_data = load_simulation_data(file_path)

    # Create nodes for each agent (assuming 5 agents)
    agent_nodes = []
    for agent_id in range(5):
        robot_name = f'turtlebot{agent_id+1}'
        node = SNNFormationControlNode(simulation_data, agent_id, robot_name)
        agent_nodes.append(node)

    rclpy.spin_multi(agent_nodes)

    rclpy.shutdown()

if __name__ == '__main__':
    main()
