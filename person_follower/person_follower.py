# person_follower.py
# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        self.subscription  # prevent unused variable warning

    def listener_callback(self, input_msg):
        ranges = input_msg.ranges
        angle_min = input_msg.angle_min
        angle_increment = input_msg.angle_increment

        # Filtrar valores inválidos
        valid_ranges = [(i, r) for i, r in enumerate(ranges) if r > 0.0 and r < 10.0]  # LIDAR máximo 10 m

        if not valid_ranges:
            vx = 0.0
            wz = 0.0
        else:
            # Encontrar el punto más cercano
            min_index, min_range = min(valid_ranges, key=lambda x: x[1])
            angle_to_person = angle_min + min_index * angle_increment

            # Control proporcional simple
            vx = 0.5 * (min_range - 1.0)   # Mantener ~1 metro de distancia
            vx = max(0.0, min(vx, 0.5))    # Limitar velocidad lineal

            wz = 2.0 * angle_to_person      # Girar hacia la persona
            wz = max(-1.0, min(wz, 1.0))   # Limitar velocidad angular

        # Publicar comando
        output_msg = Twist()
        output_msg.linear.x = vx
        output_msg.angular.z = wz
        self.publisher_.publish(output_msg)

def main(args=None):
    rclpy.init(args=args)
    person_follower = PersonFollower()
    rclpy.spin(person_follower)
    person_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
