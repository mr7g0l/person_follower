import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class PersonFollower(Node):

    def __init__(self):
        super().__init__('person_follower')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            10)
        self.max_linear_speed = 0.5   # m/s, ajustable
        self.max_angular_speed = 1.0  # rad/s, ajustable

    def listener_callback(self, scan: LaserScan):
        ranges = scan.ranges
        if not ranges:
            return

        # Tomamos un sector frontal (~60°)
        sector_size = len(ranges) // 3
        front_sector = ranges[len(ranges)//3 : 2*len(ranges)//3]

        # Buscamos la distancia mínima en el sector frontal
        min_distance = min(front_sector)
        min_index = front_sector.index(min_distance)
        center_index = len(front_sector) // 2

        # Control proporcional simple
        linear_speed = 0.0
        angular_speed = 0.0

        if min_distance < 3.0:  # detecta persona en rango
            # Velocidad lineal proporcional a la distancia
            linear_speed = min(self.max_linear_speed, 0.5 * min_distance)

            # Gira hacia la persona
            error = min_index - center_index
            angular_speed = max(-self.max_angular_speed,
                                min(self.max_angular_speed, 0.01 * error))
        
        # Publicar comando
        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = angular_speed
        self.publisher_.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = PersonFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
