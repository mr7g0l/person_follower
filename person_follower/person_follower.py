import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from ultralytics import YOLO
import math


class PersonFollower(Node):
    def __init__(self):
        super().__init__('person_follower')

        # Publisher de velocidad
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        # Suscripción al LiDAR
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)

        # Suscripción a la cámara RGB
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)

        # Parámetros ajustables
        self.max_linear_speed = 0.5     # m/s
        self.max_angular_speed = 1.0    # rad/s
        self.desired_distance = 1.0     # metros
        self.max_detection_distance = 3.0
        self.min_detection_distance = 0.2

        # FOV horizontal de la cámara del TurtleBot3 en Webots (64° por defecto)
        self.camera_fov_rad = math.radians(64.0)

        # Resultado de YOLO: ángulo normalizado [-1 izquierda, 0 centro, +1 derecha]
        # None = no se detectó ninguna persona
        self.yolo_person_angle_norm = None

        # Modelo YOLOv8 nano (ligero, suficiente para seguimiento en tiempo real)
        # Se descarga automáticamente la primera vez (~6 MB)
        self.model = YOLO('yolov8n.pt')
        self.bridge = CvBridge()

        self.get_logger().info("Person Follower con YOLOv8 iniciado")
        self.get_logger().info("Esperando detección de persona...")

    # ------------------------------------------------------------------
    # CALLBACK CÁMARA: detección de persona con YOLOv8
    # ------------------------------------------------------------------
    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"Error convirtiendo imagen: {e}")
            return

        # Inferencia YOLOv8 (verbose=False para no saturar la consola)
        results = self.model(cv_image, verbose=False)

        img_width = cv_image.shape[1]
        best_box = None
        best_area = 0

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                # Clase 0 = 'person' en el dataset COCO
                if cls_id != 0:
                    continue
                conf = float(box.conf[0])
                if conf < 0.5:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                # Nos quedamos con la persona más grande (más cercana)
                if area > best_area:
                    best_area = area
                    best_box = (x1, y1, x2, y2)

        if best_box is not None:
            x1, _, x2, _ = best_box
            cx = (x1 + x2) / 2.0
            # Ángulo normalizado: -1 = extremo izq, 0 = centro, +1 = extremo der
            self.yolo_person_angle_norm = (cx / img_width) * 2.0 - 1.0
            self.get_logger().debug(
                f"Persona detectada | centro_x={cx:.0f} | norm={self.yolo_person_angle_norm:.2f}"
            )
        else:
            self.yolo_person_angle_norm = None

    # ------------------------------------------------------------------
    # CALLBACK LIDAR: control de movimiento fusionado con YOLO
    # ------------------------------------------------------------------
    def scan_callback(self, scan: LaserScan):
        ranges = scan.ranges
        if not ranges:
            return

        # Si YOLO no detectó persona → detener el robot
        if self.yolo_person_angle_norm is None:
            self.publisher_.publish(Twist())
            return

        angle_min = scan.angle_min
        angle_increment = scan.angle_increment

        # Convertir ángulo normalizado YOLO → ángulo real en radianes
        # usando el FOV de la cámara del TurtleBot3 (64°)
        yolo_angle_rad = self.yolo_person_angle_norm * (self.camera_fov_rad / 2.0)

        # Buscar en el sector LiDAR alineado con la detección YOLO (±15°)
        search_half = math.radians(15.0)
        angle_low  = yolo_angle_rad - search_half
        angle_high = yolo_angle_rad + search_half

        front_ranges = []
        front_angles = []

        for i, distance in enumerate(ranges):
            angle = angle_min + i * angle_increment
            if angle_low < angle < angle_high:
                if math.isfinite(distance):
                    if self.min_detection_distance < distance < self.max_detection_distance:
                        front_ranges.append(distance)
                        front_angles.append(angle)

        # Si el LiDAR no confirma nada en esa zona → detener
        if not front_ranges:
            self.publisher_.publish(Twist())
            return

        # Objeto más cercano en la zona confirmada por YOLO
        min_distance = min(front_ranges)
        min_index    = front_ranges.index(min_distance)
        target_angle = front_angles[min_index]

        # ----------------------------------------------------------------
        # CONTROL PROPORCIONAL
        # ----------------------------------------------------------------
        distance_error = min_distance - self.desired_distance
        k_linear  = 0.8
        k_angular = 1.5

        linear_speed  = k_linear * distance_error
        angular_speed = -k_angular * target_angle  # negativo: girar hacia el objetivo

        # Limitar velocidades
        linear_speed  = max(-self.max_linear_speed,
                            min(self.max_linear_speed, linear_speed))
        angular_speed = max(-self.max_angular_speed,
                            min(self.max_angular_speed, angular_speed))

        # Si está demasiado cerca → retrocede suave
        if min_distance < self.desired_distance:
            linear_speed = -0.2

        twist = Twist()
        twist.linear.x  = linear_speed
        twist.angular.z = angular_speed
        self.publisher_.publish(twist)

        self.get_logger().info(
            f"Dist: {min_distance:.2f}m | "
            f"Ángulo: {math.degrees(target_angle):.1f}° | "
            f"v={linear_speed:.2f} w={angular_speed:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = PersonFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
