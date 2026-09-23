import time

import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge


class FastBotAutonomousNode(Node):

    def __init__(self):
        super().__init__('fastbot_autonomous_node')

        # Topic parameters for FastBot 1
        self.declare_parameter(
            'image_topic',
            '/fastbot_1/camera/image_raw'
        )

        self.declare_parameter(
            'scan_topic',
            '/fastbot_1/scan'
        )

        self.declare_parameter(
            'cmd_vel_topic',
            '/fastbot_1/cmd_vel'
        )

        image_topic = self.get_parameter(
            'image_topic'
        ).value

        scan_topic = self.get_parameter(
            'scan_topic'
        ).value

        cmd_vel_topic = self.get_parameter(
            'cmd_vel_topic'
        ).value

        # Sensor-data QoS profile
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.bridge = CvBridge()

        self.latest_scan = None
        self.latest_frame = None

        self.target_detected = False
        self.target_position = 'unknown'

        self.last_log_time = time.time()
        self.frame_count = 0

        # Camera subscriber
        self.image_subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            sensor_qos
        )

        # Laser subscriber
        self.scan_subscription = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            sensor_qos
        )

        # Velocity publisher
        self.velocity_publisher = self.create_publisher(
            Twist,
            cmd_vel_topic,
            10
        )

        # Navigation timer
        self.control_timer = self.create_timer(
            0.1,
            self.navigation_callback
        )

        self.get_logger().info(
            'FastBot autonomous navigation node started'
        )

        self.get_logger().info(
            f'Camera topic: {image_topic}'
        )

        self.get_logger().info(
            f'Laser topic: {scan_topic}'
        )

        self.get_logger().info(
            f'Velocity topic: {cmd_vel_topic}'
        )

    def image_callback(self, msg):
        """
        Convert ROS image messages to OpenCV images
        and perform simple green-region detection.
        """

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

            self.latest_frame = frame.copy()
            self.frame_count += 1

            height, width, _ = frame.shape

            hsv = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2HSV
            )

            # Green colour range for demonstrative
            # visual feature detection.
            lower_green = np.array(
                [35, 50, 40],
                dtype=np.uint8
            )

            upper_green = np.array(
                [90, 255, 255],
                dtype=np.uint8
            )

            mask = cv2.inRange(
                hsv,
                lower_green,
                upper_green
            )

            # Remove small isolated regions.
            kernel = np.ones(
                (5, 5),
                np.uint8
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                kernel
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel
            )

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            self.target_detected = False
            self.target_position = 'unknown'

            largest_area = 0
            largest_contour = None

            for contour in contours:
                area = cv2.contourArea(contour)

                if area > largest_area:
                    largest_area = area
                    largest_contour = contour

            # Minimum area avoids reacting to tiny regions.
            minimum_area = max(
                300,
                int(width * height * 0.01)
            )

            if (
                largest_contour is not None
                and largest_area >= minimum_area
            ):
                moments = cv2.moments(largest_contour)

                if moments['m00'] != 0:
                    center_x = int(
                        moments['m10'] / moments['m00']
                    )

                    self.target_detected = True

                    if center_x < width * 0.40:
                        self.target_position = 'left'

                    elif center_x > width * 0.60:
                        self.target_position = 'right'

                    else:
                        self.target_position = 'center'

                    x, y, w, h = cv2.boundingRect(
                        largest_contour
                    )

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        self.target_position,
                        (x, max(25, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

            # Save a periodic processed image.
            if self.frame_count % 30 == 0:
                cv2.imwrite(
                    '/tmp/fastbot_processed_image.jpg',
                    frame
                )

        except Exception as error:
            self.get_logger().error(
                f'Camera processing error: {error}'
            )

    def scan_callback(self, msg):
        """
        Store the latest laser scan.
        """
        self.latest_scan = msg

    def get_front_distance(self):
        """
        Estimate the minimum valid distance in the
        forward-facing portion of the laser scan.
        """

        if self.latest_scan is None:
            return None

        ranges = np.array(
            self.latest_scan.ranges,
            dtype=np.float32
        )

        if len(ranges) == 0:
            return None

        angle_min = self.latest_scan.angle_min
        angle_increment = self.latest_scan.angle_increment

        angles = (
            angle_min
            + np.arange(len(ranges)) * angle_increment
        )

        # Select approximately +/- 30 degrees ahead.
        front_mask = np.abs(angles) <= np.deg2rad(30)

        front_ranges = ranges[front_mask]

        valid_ranges = front_ranges[
            np.isfinite(front_ranges)
        ]

        if len(valid_ranges) == 0:
            return None

        valid_ranges = valid_ranges[
            valid_ranges > 0.05
        ]

        if len(valid_ranges) == 0:
            return None

        return float(np.min(valid_ranges))

    def create_velocity_command(
        self,
        linear_x=0.0,
        angular_z=0.0
    ):
        command = Twist()

        command.linear.x = float(linear_x)
        command.linear.y = 0.0
        command.linear.z = 0.0

        command.angular.x = 0.0
        command.angular.y = 0.0
        command.angular.z = float(angular_z)

        return command

    def publish_stop(self):
        command = self.create_velocity_command(
            linear_x=0.0,
            angular_z=0.0
        )

        self.velocity_publisher.publish(command)

    def navigation_callback(self):
        """
        Safety-first rule-based navigation.
        """

        front_distance = self.get_front_distance()

        obstacle_threshold = 0.45

        # Stop if laser data is unavailable.
        if front_distance is None:
            self.publish_stop()

            self.log_status(
                'Waiting for valid laser data'
            )

            return

        # Stop or turn when an obstacle is close.
        if front_distance < obstacle_threshold:

            command = self.create_velocity_command(
                linear_x=0.0,
                angular_z=0.35
            )

            self.velocity_publisher.publish(command)

            self.log_status(
                f'Obstacle detected: '
                f'{front_distance:.2f} m. Turning.'
            )

            return

        # Follow the detected visual region.
        if self.target_detected:

            if self.target_position == 'left':
                command = self.create_velocity_command(
                    linear_x=0.05,
                    angular_z=0.25
                )

            elif self.target_position == 'right':
                command = self.create_velocity_command(
                    linear_x=0.05,
                    angular_z=-0.25
                )

            elif self.target_position == 'center':
                command = self.create_velocity_command(
                    linear_x=0.10,
                    angular_z=0.0
                )

            else:
                command = self.create_velocity_command()

        else:
            # Move slowly when the path is clear and
            # no target has been detected.
            command = self.create_velocity_command(
                linear_x=0.05,
                angular_z=0.0
            )

        self.velocity_publisher.publish(command)

        self.log_status(
            f'Front distance: {front_distance:.2f} m | '
            f'Target: {self.target_position}'
        )

    def log_status(self, message):
        current_time = time.time()

        if current_time - self.last_log_time >= 2.0:
            self.get_logger().info(message)
            self.last_log_time = current_time

    def destroy_node(self):
        """
        Send a stop command before shutdown.
        """

        self.publish_stop()

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = FastBotAutonomousNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()