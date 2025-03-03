#!/usr/bin/env python3

import cv2
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped
from cv_bridge import CvBridge
import rospy
import numpy as np
from std_srvs.srv import Empty, EmptyResponse
from std_srvs.srv import SetBool, SetBoolResponse

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.error_sum = 0
        self.last_error = 0
        self.last_time = rospy.get_time()

    def control(self, error):
        current_time = rospy.get_time()
        delta_time = current_time - self.last_time if self.last_time else 1.0
        delta_error = error - self.last_error
        self.error_sum += error * delta_time

        output = (self.Kp * error) + (self.Ki * self.error_sum) + (self.Kd * (delta_error / delta_time))

        self.last_error = error
        self.last_time = current_time
        return output

class LaneFollowingNode(DTROS):
    def __init__(self, node_name):
        super(LaneFollowingNode, self).__init__(node_name=node_name, node_type=NodeType.CONTROL)

        # Subscribers & Publishers
        self.sub_image = rospy.Subscriber("csc22938/camera_node/image/compressed", CompressedImage, self.process_image)
        self.pub_cmd = rospy.Publisher("csc22938/joy_mapper_node/car_cmd", Twist2DStamped, queue_size=1)

        self.bridge = CvBridge()

        # Lane Following Parameters
        self.yellow_lower_bound = np.array([20, 45, 25])
        self.yellow_upper_bound = np.array([35, 255, 255])

        self.red_lower_bound1 = np.array([0, 100, 100])
        self.red_upper_bound1 = np.array([10, 255, 255])
        self.red_lower_bound2 = np.array([160, 100, 100])
        self.red_upper_bound2 = np.array([180, 255, 255])

        # PID Controller
        self.omega_controller = PIDController(Kp=3.0, Ki=0.0, Kd=0.5)
        self.drive = True
        self.base_speed = 0.1

        # Service to disable/enable lane following (EXTERNAL CONTROL)
        self.control_service = rospy.Service("lane_following_control", SetBool, self.control_callback)
        rospy.loginfo("Service lane_following_control started")

        # State Machine
        self.STATE_FOLLOWING = 0
        self.STATE_STOPPING = 1
        self.STATE_WAITING = 2  # New state: Waiting for AprilTag detection
        self.state = self.STATE_FOLLOWING

        self.stop_start_time = None
        self.stop_duration = 3.0
        self.log("Lane Following Node Initialized.")

    def control_callback(self, req):
        """Service callback to enable/disable lane following from another node."""
        if self.state != self.STATE_WAITING:  # Only allow external control when in WAITING state
            response = SetBoolResponse()
            response.success = False
            response.message = "Cannot control lane following. Not in WAITING state."
            return response

        self.drive = req.data  # Update drive based on the service request
        if self.drive:
            rospy.loginfo("Resuming lane following")
            self.state = self.STATE_FOLLOWING
        else:
            rospy.logwarn("Tried to disable lane following from outside while not detecting Red Line. Ignoring")

        response = SetBoolResponse()
        response.success = True
        response.message = "Lane following external state updated"
        return response

    def control_wheels(self, v, omega):
        cmd_msg = Twist2DStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.v = v
        cmd_msg.omega = omega
        self.pub_cmd.publish(cmd_msg)

    def calculate_error(self, img):
        """Process image and compute lane deviation error"""
        img = cv2.resize(img, (160, 120))  # Resize for speed
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.yellow_lower_bound, self.yellow_upper_bound)  # Yellow lane detection
        moments = cv2.moments(mask)

        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
        else:
            cx = 80  # Default center if lane lost

        error = (cx - 80) / 80.0  # Normalize error (-1 to 1)
        return error

    def detect_red_line(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        red_mask1 = cv2.inRange(hsv, self.red_lower_bound1, self.red_upper_bound1)
        red_mask2 = cv2.inRange(hsv, self.red_lower_bound2, self.red_upper_bound2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Check for red line
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return len(red_contours) > 0

    def process_image(self, msg):
        img = self.bridge.compressed_imgmsg_to_cv2(msg)

        if self.state == self.STATE_FOLLOWING:
            if self.detect_red_line(img):
                rospy.loginfo("Red line detected! Stopping the bot.")
                self.control_wheels(0.0, 0.0)
                self.state = self.STATE_STOPPING
                self.stop_start_time = rospy.Time.now()
                return

            error = self.calculate_error(img)
            omega = self.omega_controller.control(error)
            v = self.base_speed
            self.control_wheels(v, omega)

        elif self.state == self.STATE_STOPPING:
            time_elapsed = rospy.Time.now() - self.stop_start_time
            if time_elapsed.to_sec() >= self.stop_duration:
                rospy.loginfo("Stopping duration complete. Entering WAITING state.")
                self.state = self.STATE_WAITING
                # Now, AprilTag detection should start
                self.drive = False
                return  # Bot remains stopped and waits for an external command

        elif self.state == self.STATE_WAITING:
            self.control_wheels(0.0, 0.0)  # Ensure the bot is stopped
            rospy.loginfo("Stopped at red line")

            # Bot remains stopped in this state; AprilTag node must call the service
            return

if __name__ == '__main__':
    node = LaneFollowingNode(node_name='lane_following_node')
    rospy.spin()