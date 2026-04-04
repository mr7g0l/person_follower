"""
Person Follower — Fusión completa
  RGB + Depth + LiDAR + Odometría + Filtro de Kalman
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import cv2.aruco as aruco
import numpy as np
import math
import time


# ── Parámetros de la cámara ────────────────────────────────────────────
_FOV_H = 1.117
_W, _H = 640, 480
_fx    = (_W / 2.0) / math.tan(_FOV_H / 2.0)

CAMERA_MATRIX = np.array([
    [_fx, 0.0, _W / 2.0],
    [0.0, _fx, _H / 2.0],
    [0.0, 0.0,       1.0],
], dtype=np.float64)
DIST_COEFFS = np.zeros((5, 1), dtype=np.float64)

# ── Marcadores Aruco (backup de localización absoluta) ─────────────────
ARUCO_MARKERS = {
    0: np.array([7.5,   5.0, 1.2]),
    1: np.array([8.0,  -5.0, 1.2]),
    2: np.array([10.0,  2.0, 1.2]),
    3: np.array([10.0, -2.0, 1.2]),
}
ARUCO_MARKER_SIZE = 0.3


# ══════════════════════════════════════════════════════════════════════
# Filtro de Kalman 2D para la posición de la persona
# Estado: [x, y, vx, vy]
# ══════════════════════════════════════════════════════════════════════
class KalmanTracker:
    def __init__(self):
        self.initialized = False
        self.x = np.zeros(4)
        self.P = np.eye(4) * 5.0
        self.Q = np.diag([0.05, 0.05, 0.2, 0.2])
        self.R = np.eye(2) * 0.3
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.last_time = None

    def _predict(self, dt):
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0,  1,  0],
                      [0, 0,  0,  1]], dtype=float)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, world_pos):
        now = time.time()
        z   = np.array(world_pos[:2], dtype=float)
        if not self.initialized:
            self.x[:2]    = z
            self.last_time = now
            self.initialized = True
            return
        dt = max(0.01, now - self.last_time)
        self.last_time = now
        self._predict(dt)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P  = (np.eye(4) - K @ self.H) @ self.P

    def get_position(self):
        return (self.x[0], self.x[1]) if self.initialized else None

    def get_predicted(self, t_ahead=0.8):
        """Posición predicha t_ahead segundos en el futuro."""
        if not self.initialized:
            return None
        return (self.x[0] + self.x[2] * t_ahead,
                self.x[1] + self.x[3] * t_ahead)


# ══════════════════════════════════════════════════════════════════════
# Nodo principal
# ══════════════════════════════════════════════════════════════════════
class PersonFollower(Node):

    def __init__(self):
        super().__init__('person_follower')

        # ── Publishers ────────────────────────────────────────────────
        self.cmd_pub   = self.create_publisher(Twist, '/cmd_vel', 10)
        self.debug_pub = self.create_publisher(Image, '/person_follower/debug_image', 10)

        # ── Suscripciones ─────────────────────────────────────────────
        self.create_subscription(
            Image, '/TurtleBot3Burger/camera/image_color',
            self.rgb_callback, 10)
        self.create_subscription(
            Image, '/TurtleBot3Burger/range_finder/range_image/image',
            self.depth_callback, 10)
        self.create_subscription(
            LaserScan, '/scan',
            self.scan_callback, 10)
        self.create_subscription(
            Odometry, '/odom',
            self.odom_callback, 10)

        # ── Parámetros de control ──────────────────────────────────────
        self.desired_distance  = 0.8    # m
        self.max_linear_speed  = 0.15   # m/s
        self.max_angular_speed = 0.6    # rad/s
        self.OBSTACLE_STOP     = 0.40   # m — freno de emergencia
        self.OBSTACLE_SLOW     = 0.65   # m — reducir velocidad

        # ── Estado del robot (odometría) ───────────────────────────────
        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0

        # ── LiDAR ─────────────────────────────────────────────────────
        self.obstacle_front = float('inf')
        self.leg_clusters   = []   # [(x,y), ...] en frame robot

        # ── Profundidad ───────────────────────────────────────────────
        self.latest_depth = None

        # ── Aruco (backup localización) ───────────────────────────────
        aruco_dict   = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        aruco_params = aruco.DetectorParameters()
        self.aruco_detector = aruco.ArucoDetector(aruco_dict, aruco_params)

        # ── YOLO + tracking ───────────────────────────────────────────
        self.model  = YOLO('yolov8n.pt')
        self.bridge = CvBridge()

        self.locked_track_id  = None
        self.last_seen_time   = None
        self.last_angular_dir = 1.0

        # Re-ID
        self.target_histogram     = None
        self.HIST_MATCH_THRESHOLD = 0.75
        self.REID_MIN_LOST_SECS   = 2.0

        # Estado del objetivo
        self.target_angle_rad  = None
        self.target_distance_m = None

        # ── Filtro de Kalman ───────────────────────────────────────────
        self.kalman = KalmanTracker()
        self.kalman_mode_start = None   # para timeout en modo Kalman
        self.KALMAN_TIMEOUT    = 8.0    # segundos máximo en modo Kalman

        # ── Timer de control 10 Hz ─────────────────────────────────────
        self.create_timer(0.1, self.control_callback)

        self.get_logger().info("Person Follower Fusionado iniciado")
        self.get_logger().info("  Sensores: RGB + Depth + LiDAR + Odometría + Kalman")

    # ──────────────────────────────────────────────────────────────────
    # ODOMETRÍA
    # ──────────────────────────────────────────────────────────────────
    def odom_callback(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny, cosy)

    # ──────────────────────────────────────────────────────────────────
    # LIDAR — obstáculos + detección de piernas
    # ──────────────────────────────────────────────────────────────────
    def scan_callback(self, msg: LaserScan):
        ranges    = msg.ranges
        angle_min = msg.angle_min
        angle_inc = msg.angle_increment

        # Obstáculo frontal (±30°)
        front_min = float('inf')
        for i, r in enumerate(ranges):
            a = angle_min + i * angle_inc
            if abs(a) < math.radians(30) and math.isfinite(r) and r > 0.05:
                front_min = min(front_min, r)
        self.obstacle_front = front_min

        # Clusters de piernas
        self.leg_clusters = self._detect_leg_clusters(
            ranges, angle_min, angle_inc)

    def _detect_leg_clusters(self, ranges, angle_min, angle_inc,
                              max_range=2.5, gap=0.15,
                              min_pts=3, max_pts=12, max_width=0.35):
        """Agrupa puntos LiDAR consecutivos en clusters pequeños (≈ piernas)."""
        points = []
        for i, r in enumerate(ranges):
            if math.isfinite(r) and 0.1 < r < max_range:
                a = angle_min + i * angle_inc
                points.append((r * math.cos(a), r * math.sin(a)))

        clusters = []
        if not points:
            return clusters

        cluster = [points[0]]
        for p, q in zip(points, points[1:]):
            if math.hypot(p[0]-q[0], p[1]-q[1]) < gap:
                cluster.append(q)
            else:
                if min_pts <= len(cluster) <= max_pts:
                    xs = [pt[0] for pt in cluster]
                    ys = [pt[1] for pt in cluster]
                    if math.hypot(max(xs)-min(xs), max(ys)-min(ys)) < max_width:
                        clusters.append((float(np.mean(xs)), float(np.mean(ys))))
                cluster = [q]

        if min_pts <= len(cluster) <= max_pts:
            xs = [pt[0] for pt in cluster]
            ys = [pt[1] for pt in cluster]
            if math.hypot(max(xs)-min(xs), max(ys)-min(ys)) < max_width:
                clusters.append((float(np.mean(xs)), float(np.mean(ys))))

        return clusters

    def _lidar_dist_at_angle(self, angle_rad, tolerance_deg=15.0):
        """Distancia LiDAR al cluster más cercano al ángulo dado."""
        tol = math.radians(tolerance_deg)
        best = None
        for lx, ly in self.leg_clusters:
            a = math.atan2(ly, lx)
            if abs(a - angle_rad) < tol:
                d = math.hypot(lx, ly)
                if best is None or d < best:
                    best = d
        return best

    def _validate_with_lidar(self, candidate_angle_rad):
        """True si hay un cluster LiDAR cerca del ángulo del candidato."""
        if not self.leg_clusters:
            return True
        tol = math.radians(25.0)
        for lx, ly in self.leg_clusters:
            if abs(math.atan2(ly, lx) - candidate_angle_rad) < tol:
                return True
        return False

    # ──────────────────────────────────────────────────────────────────
    # DEPTH
    # ──────────────────────────────────────────────────────────────────
    def depth_callback(self, msg: Image):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
        except Exception as e:
            self.get_logger().warn(f"Depth: {e}")

    # ──────────────────────────────────────────────────────────────────
    # RGB: Aruco + YOLO
    # ──────────────────────────────────────────────────────────────────
    def rgb_callback(self, msg: Image):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f"RGB: {e}")
            return
        self._process_aruco(image)
        self._process_yolo(image)

    # ──────────────────────────────────────────────────────────────────
    # ARUCO (backup localización absoluta)
    # ──────────────────────────────────────────────────────────────────
    def _process_aruco(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        if ids is None:
            return
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id not in ARUCO_MARKERS:
                continue
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                [corners[i]], ARUCO_MARKER_SIZE, CAMERA_MATRIX, DIST_COEFFS)
            tvec = tvecs[0].flatten()
            R_cm, _ = cv2.Rodrigues(rvecs[0])
            dist  = np.linalg.norm(tvec)
            angle = math.atan2(tvec[0], tvec[2])
            normal = R_cm[:, 2]
            yaw   = math.atan2(-normal[0], normal[2])
            wa    = yaw + angle - math.pi
            mw    = ARUCO_MARKERS[marker_id]
            # Corrección del odom offset si el Aruco da posición distinta
            aruco_rx = mw[0] + dist * math.cos(wa)
            aruco_ry = mw[1] + dist * math.sin(wa)
            # Actualizar Kalman si tenemos posición de persona
            # (Aruco da la pose del robot, útil cuando odom ha derivado)
            self.get_logger().debug(
                f"Aruco {marker_id}: robot≈({aruco_rx:.1f},{aruco_ry:.1f})")
            break

    # ──────────────────────────────────────────────────────────────────
    # RE-ID por histograma HSV
    # ──────────────────────────────────────────────────────────────────
    def _compute_histogram(self, image, x1, y1, x2, y2):
        h, w = image.shape[:2]
        my = int((y2 - y1) * 0.25)
        mx = int((x2 - x1) * 0.10)
        crop = image[max(0, int(y1)+my):min(h, int(y2)-my),
                     max(0, int(x1)+mx):min(w, int(x2)-mx)]
        if crop.size == 0:
            return None
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [18, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def _hist_sim(self, h1, h2):
        if h1 is None or h2 is None:
            return 0.0
        return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)

    # ──────────────────────────────────────────────────────────────────
    # YOLO + BOTSORT + FUSIÓN
    # ──────────────────────────────────────────────────────────────────
    def _process_yolo(self, image):
        img_width = image.shape[1]
        now       = time.time()
        debug_img = image.copy()

        results = self.model.track(
            image, persist=True, tracker='botsort.yaml', verbose=False)

        # Recopilar personas detectadas
        persons = []
        for result in results:
            if result.boxes.id is None:
                continue
            for box, tid in zip(result.boxes, result.boxes.id.int().tolist()):
                if int(box.cls[0]) != 0 or float(box.conf[0]) < 0.30:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                cx   = int((x1 + x2) / 2)
                cy   = int((y1 + y2) / 2)
                persons.append((tid, area, cx, cy, x1, y1, x2, y2))

        # ── Sin detecciones ───────────────────────────────────────────
        if not persons:
            lost_s = now - self.last_seen_time if self.last_seen_time else 0
            cv2.putText(debug_img,
                        f"RECUPERANDO ({lost_s:.0f}s)",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 165, 255), 2)
            self._draw_overlay(debug_img)
            self.debug_pub.publish(
                self.bridge.cv2_to_imgmsg(debug_img, 'bgr8'))
            self.target_angle_rad  = None
            self.target_distance_m = None
            return

        # ── Enganchar al más cercano si no hay objetivo ───────────────
        if self.locked_track_id is None:
            best = max(persons, key=lambda p: p[1])
            self.locked_track_id = best[0]
            self.last_seen_time  = now
            _, _, _, _, bx1, by1, bx2, by2 = best
            self.target_histogram = self._compute_histogram(
                image, bx1, by1, bx2, by2)
            self.get_logger().info(
                f"Enganchado a ID={self.locked_track_id} "
                f"(area={best[1]:.0f}px²)")

        target = next(
            (p for p in persons if p[0] == self.locked_track_id), None)

        # ── Re-ID si el objetivo no está visible ──────────────────────
        if target is None:
            lost_s = now - self.last_seen_time if self.last_seen_time else 999
            if lost_s >= self.REID_MIN_LOST_SECS:
                best_sim, best_match = -1.0, None
                for p in persons:
                    tid, _, pcx, pcy, px1, py1, px2, py2 = p
                    hist = self._compute_histogram(image, px1, py1, px2, py2)
                    sim  = self._hist_sim(self.target_histogram, hist)
                    if sim > best_sim:
                        best_sim, best_match = sim, p

                if (best_sim >= self.HIST_MATCH_THRESHOLD
                        and best_match is not None):
                    # Validar con LiDAR
                    _, _, bcx, _, bx1, by1, bx2, by2 = best_match
                    norm  = (bcx / img_width) * 2.0 - 1.0
                    b_ang = norm * (_FOV_H / 2.0)
                    if self._validate_with_lidar(b_ang):
                        old = self.locked_track_id
                        self.locked_track_id = best_match[0]
                        self.last_seen_time  = now
                        target = best_match
                        self.get_logger().info(
                            f"Re-ID: {old}→{self.locked_track_id} "
                            f"sim={best_sim:.2f}")

            if target is None:
                lost_s = now - self.last_seen_time if self.last_seen_time else 0
                cv2.putText(debug_img,
                            f"RECUPERANDO ID={self.locked_track_id} ({lost_s:.0f}s)",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 165, 255), 2)
                self._draw_overlay(debug_img)
                self.debug_pub.publish(
                    self.bridge.cv2_to_imgmsg(debug_img, 'bgr8'))
                self.target_angle_rad  = None
                self.target_distance_m = None
                return

        # ── Dibujar bounding boxes ────────────────────────────────────
        for p in persons:
            tid, _, pcx, pcy, px1, py1, px2, py2 = p
            is_tgt = (tid == self.locked_track_id)
            color  = (0, 255, 0) if is_tgt else (0, 0, 255)
            thick  = 3 if is_tgt else 1
            cv2.rectangle(debug_img,
                          (int(px1), int(py1)), (int(px2), int(py2)),
                          color, thick)
            cv2.putText(debug_img,
                        f"{'SIGUIENDO' if is_tgt else 'ID'}={tid}",
                        (int(px1), int(py1)-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.putText(debug_img,
                    f"SIGUIENDO ID={self.locked_track_id}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        self._draw_overlay(debug_img)
        self.debug_pub.publish(
            self.bridge.cv2_to_imgmsg(debug_img, 'bgr8'))

        # ── Actualizar estado del objetivo ────────────────────────────
        self.last_seen_time = now
        _, _, cx, cy, x1, y1, x2, y2 = target

        norm = (cx / img_width) * 2.0 - 1.0
        self.target_angle_rad = norm * (_FOV_H / 2.0)

        # Distancia: fusión depth + LiDAR
        depth_dist = None
        if self.latest_depth is not None:
            mg = 0.25
            xi1 = max(0, int(x1 + (x2-x1)*mg))
            xi2 = min(self.latest_depth.shape[1]-1, int(x2 - (x2-x1)*mg))
            yi1 = max(0, int(y1 + (y2-y1)*mg))
            yi2 = min(self.latest_depth.shape[0]-1, int(y2 - (y2-y1)*mg))
            roi   = self.latest_depth[yi1:yi2, xi1:xi2]
            valid = roi[np.isfinite(roi) & (roi > 0.1) & (roi < 10.0)]
            if valid.size > 0:
                depth_dist = float(np.median(valid))

        lidar_dist = self._lidar_dist_at_angle(self.target_angle_rad)

        if depth_dist is not None and lidar_dist is not None:
            self.target_distance_m = 0.7 * depth_dist + 0.3 * lidar_dist
        elif depth_dist is not None:
            self.target_distance_m = depth_dist
        elif lidar_dist is not None:
            self.target_distance_m = lidar_dist
        else:
            self.target_distance_m = None

        # Actualizar Kalman
        if self.target_distance_m is not None:
            wa = self.robot_yaw + self.target_angle_rad
            tx = self.robot_x + self.target_distance_m * math.cos(wa)
            ty = self.robot_y + self.target_distance_m * math.sin(wa)
            self.kalman.update((tx, ty))

        # Actualizar histograma (solo a buena distancia)
        if self.target_distance_m is not None and 1.0 < self.target_distance_m < 3.0:
            new_hist = self._compute_histogram(image, x1, y1, x2, y2)
            if new_hist is not None:
                if self.target_histogram is None:
                    self.target_histogram = new_hist
                else:
                    self.target_histogram = cv2.addWeighted(
                        self.target_histogram, 0.92, new_hist, 0.08, 0)

    # ──────────────────────────────────────────────────────────────────
    # DEBUG: clusters LiDAR + obstáculo en imagen
    # ──────────────────────────────────────────────────────────────────
    def _draw_overlay(self, img):
        # Clusters de piernas (puntos naranjas abajo)
        for lx, ly in self.leg_clusters:
            a    = math.atan2(ly, lx)
            norm = a / (_FOV_H / 2.0)
            px   = int((norm + 1.0) / 2.0 * img.shape[1])
            if 0 <= px < img.shape[1]:
                cv2.circle(img, (px, img.shape[0] - 25), 8, (0, 140, 255), -1)

        # Obstáculo frontal
        if self.obstacle_front < self.OBSTACLE_SLOW:
            color = (0, 0, 255) if self.obstacle_front < self.OBSTACLE_STOP \
                    else (0, 165, 255)
            cv2.putText(img,
                        f"OBST {self.obstacle_front:.2f}m",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

        # Velocidad Kalman
        vel = self.kalman.get_predicted(0.0)
        if vel and self.kalman.initialized:
            vx, vy = self.kalman.x[2], self.kalman.x[3]
            speed  = math.hypot(vx, vy)
            cv2.putText(img,
                        f"Kalman v={speed:.2f}m/s",
                        (10, img.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ──────────────────────────────────────────────────────────────────
    # CONTROL CALLBACK (10 Hz)
    # ──────────────────────────────────────────────────────────────────
    def control_callback(self):
        twist = Twist()

        # Factor de velocidad según obstáculos
        if self.obstacle_front < self.OBSTACLE_STOP:
            speed_factor = 0.0   # freno total
        elif self.obstacle_front < self.OBSTACLE_SLOW:
            t = ((self.obstacle_front - self.OBSTACLE_STOP) /
                 (self.OBSTACLE_SLOW  - self.OBSTACLE_STOP))
            speed_factor = 0.3 + 0.7 * t   # rampa 30→100%
        else:
            speed_factor = 1.0

        # ── MODO 1: persona visible con distancia ─────────────────────
        if self.target_angle_rad is not None:
            self.kalman_mode_start = None   # resetear timer Kalman
        if self.target_angle_rad is not None and self.target_distance_m is not None:
            dist_err = self.target_distance_m - self.desired_distance
            angular  = max(-self.max_angular_speed,
                           min(self.max_angular_speed,
                               -1.4 * self.target_angle_rad))
            ang_factor = max(0.0, 1.0 - abs(self.target_angle_rad) / (math.pi / 4))
            linear = max(-self.max_linear_speed,
                         min(self.max_linear_speed,
                             0.5 * dist_err * ang_factor))
            if self.target_distance_m < self.desired_distance:
                linear = -0.10
            linear *= speed_factor
            if abs(angular) > 0.05:
                self.last_angular_dir = 1.0 if angular > 0 else -1.0
            twist.linear.x  = linear
            twist.angular.z = angular
            self.get_logger().info(
                f"[SIGUIENDO ID={self.locked_track_id}] "
                f"dist={self.target_distance_m:.2f}m "
                f"ang={math.degrees(self.target_angle_rad):.1f}° "
                f"v={linear:.2f} w={angular:.2f}"
                + (" [FRENANDO]" if speed_factor < 1.0 else ""))

        # ── MODO 1b: persona visible sin distancia ────────────────────
        elif self.target_angle_rad is not None:
            angular = max(-self.max_angular_speed,
                          min(self.max_angular_speed,
                              -1.4 * self.target_angle_rad))
            ang_factor = max(0.0, 1.0 - abs(self.target_angle_rad) / (math.pi / 4))
            linear = self.max_linear_speed * ang_factor * speed_factor
            if abs(angular) > 0.05:
                self.last_angular_dir = 1.0 if angular > 0 else -1.0
            twist.linear.x  = linear
            twist.angular.z = angular
            self.get_logger().info(
                f"[ACERCANDO ID={self.locked_track_id}] "
                f"ang={math.degrees(self.target_angle_rad):.1f}° "
                f"v={linear:.2f} w={angular:.2f}")

        # ── MODO 2: Kalman — navegar a posición predicha ───────────────
        elif self.kalman.initialized:
            now = time.time()
            if self.kalman_mode_start is None:
                self.kalman_mode_start = now
            elapsed = now - self.kalman_mode_start

            # Timeout: si llevamos demasiado tiempo en Kalman sin ver a la
            # persona, resetear para buscar de nuevo
            if elapsed > self.KALMAN_TIMEOUT:
                self.get_logger().info(
                    f"[KALMAN] Timeout ({elapsed:.0f}s) — reiniciando búsqueda")
                self.locked_track_id   = None
                self.target_histogram  = None
                self.kalman.initialized = False
                self.kalman_mode_start  = None
                twist.angular.z = 0.3 * self.last_angular_dir
            else:
                predicted = self.kalman.get_predicted(t_ahead=0.8)
                if predicted:
                    tx, ty = predicted
                    dx, dy   = tx - self.robot_x, ty - self.robot_y
                    distance = math.hypot(dx, dy)
                    angle_to = math.atan2(dy, dx) - self.robot_yaw
                    angle_to = math.atan2(math.sin(angle_to), math.cos(angle_to))
                    if distance > 0.5 and speed_factor > 0:
                        twist.linear.x  = min(
                            self.max_linear_speed * speed_factor,
                            0.3 * distance)
                        twist.angular.z = max(-self.max_angular_speed,
                                              min(self.max_angular_speed,
                                                  0.8 * angle_to))
                        self.get_logger().info(
                            f"[KALMAN] ({tx:.1f},{ty:.1f}) "
                            f"d={distance:.1f}m ang={math.degrees(angle_to):.0f}°")
                    else:
                        # Llegó al punto o bloqueado — girar buscando
                        twist.angular.z = 0.3 * self.last_angular_dir

        # ── MODO 3: girar buscando ─────────────────────────────────────
        else:
            twist.angular.z = 0.3 * self.last_angular_dir

        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = PersonFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
