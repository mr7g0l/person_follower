import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math


class PersonFollower(Node):

    def __init__(self):
        super().__init__('person_follower')

        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            10)

        # Parámetros ajustables
        self.max_linear_speed = 0.5    # m/s
        self.max_angular_speed = 1.0   # rad/s
        self.desired_distance = 1.0    # metros
        self.max_detection_distance = 3.0
        self.min_detection_distance = 0.2

        self.get_logger().info("Person Follower iniciado")

    def listener_callback(self, scan: LaserScan):

        ranges = scan.ranges
        if not ranges:
            return

        front_ranges = []
        front_angles = []

        angle_min = scan.angle_min
        angle_increment = scan.angle_increment

        # Analizar sector frontal ±30° (0.52 rad)
        for i, distance in enumerate(ranges):
            angle = angle_min + i * angle_increment

            if -0.52 < angle < 0.52:  # sector frontal

                # Filtrar valores inválidos
                if math.isfinite(distance):

                    if self.min_detection_distance < distance < self.max_detection_distance:
                        front_ranges.append(distance)
                        front_angles.append(angle)

        # Si no detecta nada válido → detener robot
        if not front_ranges:
            self.publisher_.publish(Twist())
            return

        # Encontrar objeto más cercano en el sector válido
        min_distance = min(front_ranges)
        min_index = front_ranges.index(min_distance)
        target_angle = front_angles[min_index]

        # ------------------------
        # CONTROL PROPORCIONAL
        # ------------------------

        distance_error = min_distance - self.desired_distance

        k_linear = 0.8
        k_angular = 1.5

        linear_speed = k_linear * distance_error
        angular_speed = k_angular * target_angle

        # Limitar velocidades
        linear_speed = max(-self.max_linear_speed,
                           min(self.max_linear_speed, linear_speed))

        angular_speed = max(-self.max_angular_speed,
                            min(self.max_angular_speed, angular_speed))

        # Si está demasiado cerca → retrocede suave
        if min_distance < self.desired_distance:
            linear_speed = -0.2

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
