#!/usr/bin/env python3
# This project is adapted from https://github.com/ekhumbata/Adventures-in-Duckietown
import dt_apriltags
import argparse
import cv2
import tf

import os
import rospy
from duckietown.dtros import DTROS, NodeType
import numpy as np

from sensor_msgs.msg import CameraInfo
from duckietown_msgs.srv import ChangePattern
from duckietown_msgs.msg import LEDPattern
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Empty, EmptyRequest
from std_srvs.srv import SetBool, SetBoolRequest
from std_srvs.srv import EmptyResponse
from duckietown_msgs.msg import Twist2DStamped



class apriltag_node(DTROS):

    def __init__(self, node_name):
        super(apriltag_node, self).__init__(node_name=node_name, node_type=NodeType.LOCALIZATION)
        self.node_name = node_name
        self.camera_calibration = None
        self.camera_parameters = None
        self.safeToRunProgram = False
        self._tf_bcaster = tf.TransformBroadcaster()

        self.grey_img = np.array([])
        self.col_img = None
        self.curr_msg = None
        self.detector = dt_apriltags.Detector()

        self.run = True
        self.prev_img = None
        #no detection
        self.curr_col = "WHITE"
        self.sign_col_map = {
            # Blue = T-intersection
            153: "BLUE", 58: "BLUE", 11: "BLUE", 62: "BLUE",
            # Red = Stop sign
            24: "RED", 26: "RED",
            # Green = UofA Tag
            57: "GREEN", 200: "GREEN", 94: "GREEN", 93: "GREEN",
            # Yellow = Turn left
            30: "LEFT",
            # Purple = Turn right
            31: "RIGHT"

        }

        self.p = 0
        self.q = 0

        # subscribers
        img_topic = f"""/{os.environ['VEHICLE_NAME']}/camera_node/image/compressed"""
        info_topic = f"""/{os.environ['VEHICLE_NAME']}/camera_node/camera_info"""
        self.img_sub = rospy.Subscriber(img_topic, CompressedImage, self.cb_img, queue_size = 1)
        self.subscriberCameraInfo = rospy.Subscriber(info_topic, CameraInfo, self.camera_info_callback,  queue_size=1)

        # publishers
        self.pub = rospy.Publisher("/" + os.environ['VEHICLE_NAME'] + '/grey_img/compressed', CompressedImage, queue_size=1)
        self.pub_led = rospy.Publisher("/" + os.environ['VEHICLE_NAME'] + "/led_emitter_node/led_pattern", LEDPattern, queue_size=1)
        self.pub_april = rospy.Publisher("/april_topic", String, queue_size=1)
        self.publishLEDs(1.0, 0.0, 0.0)
        self.twist_publisher = rospy.Publisher("csc22938/joy_mapper_node/car_cmd", Twist2DStamped, queue_size=1)

        # create service proxy
        self.lane_following_control = rospy.ServiceProxy('lane_following_control', SetBool)

        # Initial state: Don't detect AprilTags until told to
        self.detect_apriltags_enabled = False
        self.no_tag_count = 0
        self.no_tag_threshold = 3 #3 cycles of 1 rate
        self.turn_velocity = 0 #the velocity at which it will turn, it should not turn only follow.
        self.turn_omega = 0 #the omega at which it turns, it shoud not trun only follow

    def send_drive_command(self, velocity, omega):
        """Publishes a Twist2DStamped message to control the Duckiebot's motion."""
        msg = Twist2DStamped()
        msg.v = velocity
        msg.omega = omega
        self.twist_publisher.publish(msg)

    def sign_to_col(self, id):
        if(id in self.sign_col_map):
            return self.sign_col_map[id]
        else:
            print("[INFO] Recognized AprilTag - but the ID is not valid.")
            return "WHITE"


    def camera_info_callback(self, msg):
        self.camera_calibration = msg

        print("== Calibrating Camera ==")

        # currRawImage_height = img.shape[0]
        # currRawImage_width = img.shape[0]
        currRawImage_height = 640
        currRawImage_width = 480

        scale_matrix = np.ones(9)
        if self.camera_calibration.height != currRawImage_height or self.camera_calibration.width != currRawImage_width:
            scale_width = float(currRawImage_width) / self.camera_calibration.width
            scale_height = float(currRawImage_height) / self.camera_calibration.height
            scale_matrix[0] *= scale_width
            scale_matrix[2] *= scale_width
            scale_matrix[4] *= scale_height
            scale_matrix[5] *= scale_height

        self.tag_size = 0.065 #rospy.get_param("~tag_size", 0.065)
        rect_K, _ = cv2.getOptimalNewCameraMatrix(
            (np.array(self.camera_calibration.K)*scale_matrix).reshape((3, 3)),
            self.camera_calibration.D,
            (640,480),
            1.0
        )
        self.camera_parameters = (rect_K[0, 0], rect_K[1, 1], rect_K[0, 2], rect_K[1, 2])


        try:
            self.subscriberCameraInfo.shutdown()
            self.safeToRunProgram = True
            print("== Camera Info Subscriber successfully killed ==")
        except BaseException:
            pass


    def cb_img(self, msg):
        data_arr = np.frombuffer(msg.data, np.uint8)
        col_img = cv2.imdecode(data_arr, cv2.IMREAD_COLOR)
        grey_img = cv2.cvtColor(col_img, cv2.COLOR_BGR2GRAY)
        self.grey_img = grey_img[1* len(col_img) // 4 : 2 * len(col_img) // 3]
        self.col_img = col_img[1 * len(col_img) // 4 : 2 * len(col_img) // 3]
        self.curr_msg = msg

    def img_pub(self):
        if self.grey_img.any():
            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', self.grey_img)[1]).tostring()

            self.pub.publish(msg)


    def publishLEDs(self, red, green, blue):
        set_led_cmd = LEDPattern()

        for i in range(5):
            rgba = ColorRGBA()
            rgba.r = red
            rgba.g = green
            rgba.b = blue
            rgba.a = 1.0
            set_led_cmd.rgb_vals.append(rgba)

        self.pub_led.publish(set_led_cmd)

    def april_pub(self):
        msg = String()
        msg.data = f"{self.p}:{self.q}"
        print("april", msg.data)

        self.pub_april.publish(msg)


    def change_led_to(self, new_col):
        if(new_col == "RED"):
            self.publishLEDs(1.0, 0.0, 0.0)

        elif(new_col == "GREEN"):
            self.publishLEDs(0.0, 1.0, 0.0)

        elif(new_col == "BLUE"):
            self.publishLEDs(0.0, 0.0, 1.0)

        else:
            self.publishLEDs(1.0, 1.0, 1.0)


    def convert_pixel_x_to_rad_from_middle(self, pixel_x):
        # Camera FOV = 160deg => right edge is 80deg from midpoint (and left edge is -80deg)
        # 80 deg in rad
        frame_edge_rad = 1.39626
        camera_res_x = 640 # so here x=640 => +1.3626 rad from centre, x=0 => -1.3626

        percent_offset_from_midpoint = (pixel_x - (camera_res_x/2)) / (camera_res_x/2) # gives value in [-1, 1] with 0 being perfect middle

        rad_from_middle = frame_edge_rad * percent_offset_from_midpoint # gives value in [-1.3626, 1.3626] rad = [-80, 80] deg

        return rad_from_middle

    
    def _matrix_to_quaternion(self, r):
        T = np.array(((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1)), dtype=np.float64)
        T[0:3, 0:3] = r
        return tf.transformations.quaternion_from_matrix(T)


    def detect_tag(self):
        
        if(not self.safeToRunProgram):
            return

        # convert the img to greyscale
        img =  self.col_img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Only detect tags if enabled
        if not self.detect_apriltags_enabled:
            return

        tags = self.detector.detect(gray, True, self.camera_parameters, self.tag_size)

        print("[INFO] {} total AprilTags detected".format(len(tags)))


        if len(tags) == 0:
            self.change_led_to("WHITE")
            self.curr_col = "WHITE"

            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', img)[1]).tostring()
            self.pub.publish(msg)
            #count for no tags
            self.no_tag_count += 1

            if self.no_tag_count > self.no_tag_threshold:
                 self.no_tag_count = 0
                 rospy.wait_for_service('lane_following_control')
                 try:
                      control_lane = rospy.ServiceProxy('lane_following_control', SetBool)
                      resp = control_lane(True) #Turn on line following

                      return #so it doesn't go to else cases
                 except rospy.ServiceException as e:
                      print("Service call failed: %s"%e)

            return

        closest_col = "WHITE"
        closest = 0
        self.no_tag_count = 0 #if we found a tag

        for tag in tags:
            tag_id = tag.tag_id
            tag_color = self.sign_to_col(tag_id)

            if tag_color == "LEFT":
                print("Turning left!")
                self.send_drive_command(0.2, 2.0)  # Adjust speed and omega for turning left
                rospy.sleep(2)  # Sleep to allow turn
                self.send_drive_command(0.2, 0.0)  # Move forward after turn

            elif tag_color == "RIGHT":
                print("Turning right!")
                self.send_drive_command(0.2, -2.0)  # Adjust speed and omega for turning right
                rospy.sleep(1)  # Sleep to allow turn
                self.send_drive_command(0.2, 0.0)  # Move forward after turn

            elif tag_color in ["RED", "GREEN", "BLUE"]:
                self.change_led_to(tag_color)
    