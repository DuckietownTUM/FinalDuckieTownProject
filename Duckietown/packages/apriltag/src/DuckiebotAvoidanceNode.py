#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
import os
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import BoolStamped, VehicleCorners, Twist2DStamped
from std_msgs.msg import String, Float32
from sensor_msgs.msg import CompressedImage, CameraInfo
from image_geometry import PinholeCameraModel
from std_srvs.srv import SetBool, SetBoolRequest

class DuckiebotAvoidanceNode(DTROS):
    """
    Combines Duckiebot detection, distance estimation, and collision avoidance.
    Includes a basic lane-switching obstacle avoidance maneuver.
    """

    def __init__(self, node_name):
        super(DuckiebotAvoidanceNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.host = str(os.environ['VEHICLE_NAME'])

        # Parameters (tunable via launch file or dynamic reconfigure)
        self.distance_between_centers = 0.0125
        self.max_reproj_pixelerror_pose_estimation = 2.5
        self.process_frequency = 5  # Increased for responsiveness
        self.min_distance_to_react = 0.5  # Distance (meters) to start slowing down
        self.stop_distance = 0.25        # Distance (meters) to stop completely
        self.slowdown_factor = 0.5       # Reduce speed by this factor when approaching
        self.side_distance_timer_period = 4 #seconds

        self.bridge = CvBridge()
        self.pcm = PinholeCameraModel()
        self.last_calc_circle_pattern = None
        self.circlepattern_dist = None
        self.circlepattern = None

        # Subscribers
        self.sub_image = rospy.Subscriber(
            "/{}/camera_node/image/compressed".format(self.host), CompressedImage, self.cb_image, queue_size=1
        )
        self.sub_info = rospy.Subscriber(
            "/{}/camera_node/camera_info".format(self.host), CameraInfo, self.cb_camera_info, queue_size=1
        )

        # Publishers
        self.pub_cmd = rospy.Publisher("csc22938/joy_mapper_node/car_cmd", Twist2DStamped, queue_size=1) #added

        self.pub_centers = rospy.Publisher(
            "/{}/duckiebot_detection_node/centers".format(self.host), VehicleCorners, queue_size=1
        )
        self.pub_detection_image = rospy.Publisher(
            "/{}/duckiebot_detection_node/detection_image/compressed".format(self.host), CompressedImage, queue_size=1
        )
        self.pub_detection = rospy.Publisher(
            "/{}/duckiebot_detection_node/detection".format(self.host), BoolStamped, queue_size=1
        )
        self.pub_distance = rospy.Publisher(
            "/{}/duckiebot_distance_node/distance".format(self.host), Float32, queue_size=1
        )

        #Service proxy (calling line following to disable - commented out for now as unclear if needed in this context)
        #self.lane_following_control = rospy.ServiceProxy('lane_following_control', SetBool)

        #Timer variables (added)
        self.side_distance_timer = None
        self.turning_right = False
        self.obstacle_avoidance_active = False # Flag to prevent re-triggering avoidance

        self.log("Duckiebot Avoidance Node Initialized.")


    def control_wheels(self, v, omega):
        """Publishes a Twist2DStamped message to control the Duckiebot's motion."""
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        msg.v = v
        msg.omega = omega
        self.pub_cmd.publish(msg)

    def cb_camera_info(self, msg):
        """Store camera info."""
        self.pcm.fromCameraInfo(msg)

    def cb_image(self, image_msg):
        """Main processing loop - detect Duckiebots, estimate distance, and avoid."""
        image_cv = self.bridge.compressed_imgmsg_to_cv2(image_msg, "bgr8")

        # 1. Detect Duckiebot pattern
        detection, centers = self.detect_duckiebot_pattern(image_cv)

        # 2. Estimate distance (if detected)
        distance = None
        if detection:
            distance = self.estimate_distance(image_msg, image_cv, centers)

        # 3. Collision Avoidance (adjust speed based on distance)
        self.avoid_collision(distance)

        # Publish detection image (for visualization)
        if self.pub_detection_image.get_num_connections() > 0:
            cv2.drawChessboardCorners(image_cv, (7,3), centers, detection)
            img_msg_out = self.bridge.cv2_to_compressed_imgmsg(image_cv, dst_format='jpg')
            self.pub_detection_image.publish(img_msg_out)


    def detect_duckiebot_pattern(self, image_cv):
        """Detects the Duckiebot circle pattern using OpenCV."""

        simple_blob_detector = cv2.SimpleBlobDetector_create()

        detection, centers = cv2.findCirclesGrid(
            image_cv,
            patternSize=(7, 3),
            flags=cv2.CALIB_CB_SYMMETRIC_GRID,
        )

        vehicle_centers_msg_out = VehicleCorners()
        detection_flag_msg_out = BoolStamped()

        #Populating data for debugging/monitoring
        points_list = []
        vehicle_centers_msg_out.header = self.sub_image.header
        vehicle_centers_msg_out.detection.data = detection
        for point in centers:
            center = Point32()
            center.x = point[0][0][0] #point[0][0][0]
            center.y = point[0][0][1] #point[0][0][1]
            center.z = 0
            points_list.append(center)
        vehicle_centers_msg_out.corners = points_list
        vehicle_centers_msg_out.H = 3
        vehicle_centers_msg_out.W = 7

        detection_flag_msg_out.header = self.sub_image.header
        detection_flag_msg_out.data = detection
        self.pub_centers.publish(vehicle_centers_msg_out) #only publishing detected coordinates
        self.pub_detection.publish(detection_flag_msg_out)

        return detection, centers

    def estimate_distance(self, image_msg, image_cv, centers):
        """Estimates the distance to the Duckiebot using solvePnP."""
        try:
            H = 3 # hardcoded height
            W = 7 # hardcoded width
            self.calc_circle_pattern(H, W)

            #check for types, some were crashing due to points variable being a type that is not a type.
            points = np.zeros((H * W, 2), dtype=np.float32) # H, W
            for i in range(len(centers)):
                points[i] = np.array([centers[i][0][0], centers[i][0][1]]) #centers[i][0][0], centers[i][0][1]

            success, rotation_vector, translation_vector = cv2.solvePnP(
                objectPoints=self.circlepattern,
                imagePoints=points,
                cameraMatrix=self.pcm.intrinsicMatrix(),
                distCoeffs=self.pcm.distortionCoeffs(),
                flags=cv2.SOLVEPNP_ITERATIVE #added this flag
            )

            if not success:
                return None  # Pose estimation failed

            points_reproj, _ = cv2.projectPoints(
                objectPoints=self.circlepattern,
                rvec=rotation_vector,
                tvec=translation_vector,
                cameraMatrix=self.pcm.intrinsicMatrix(),
                distCoeffs=self.pcm.distortionCoeffs(),
            )

            mean_reproj_error = np.mean(
                np.sqrt(np.sum((np.squeeze(points_reproj) - points) ** 2, axis=1))
            )

            if mean_reproj_error > self.max_reproj_pixelerror_pose_estimation:
                rospy.logwarn(f"High reprojection error: {mean_reproj_error:.2f}")
                return None #Reprojection error is high, means distance is incorrect

            R, _ = cv2.Rodrigues(rotation_vector)
            R_inv = np.transpose(R)
            translation_vector = -np.dot(R_inv, translation_vector)
            distance_to_vehicle = -translation_vector[2]
            self.pub_distance.publish(Float32(distance_to_vehicle))
            return distance_to_vehicle


        except Exception as e:
            rospy.logerr(f"Error estimating distance: {e}")
            return None

    def avoid_collision(self, distance):
        """Adjusts speed and perform side step to avoid collisions
        1) The robot slows down as it approaches a vehicle

        Args:
            distance (`float`): The distance detected by the Duckiebot


        """

        if distance is None:
            #No robot in sight, full speed!
            self.control_wheels(0.2, 0)

        elif distance <= self.stop_distance:
            #Too close, stop and perform obstacle avoidance maneuver.
            rospy.logwarn("Too close! Stopping and initiating obstacle avoidance.")
            self.control_wheels(0, 0)

            if not self.obstacle_avoidance_active: #Only start avoidance once
                self.obstacle_avoidance_active = True #Set flag
                self.avoid_obstacle() # Call the lane switching maneuver

        elif distance <= self.min_distance_to_react:
            #Approaching, slow down.
            new_speed = 0.05#self.base_speed * self.slowdown_factor
            rospy.logwarn(f"Slowing down: Distance = {distance:.2f}, New Speed = {new_speed:.2f}")
            self.control_wheels(new_speed, 0) #slow

        else:
            #Robot far away, proceed as planned.
            self.control_wheels(0.2, 0) #full speed!
            rospy.logwarn("No bot close by")

    def avoid_obstacle(self):
        """
        Performs a predefined lane-switching maneuver to avoid an obstacle.
        This is a simplified example using fixed durations and speeds.
        """
        rospy.loginfo("Obstacle avoidance maneuver started...")

        # Step 1: Move diagonally left to enter the parallel lane
        rospy.loginfo("Step 1: Moving diagonally left...")
        self.control_wheels(0, 0.7)  # turning left
        rospy.sleep(0.7)
        self.control_wheels(0.2, 0)  # Drive straight briefly
        rospy.sleep(1)

        # Step 2: Adjust to the lane by slightly turning right
        rospy.loginfo("Step 2: Adjusting to the lane...")
        self.control_wheels(0, -0.7)
        rospy.sleep(0.5)

        rospy.loginfo("Lane switch complete. Following new lane for a bit...")
        rospy.sleep(3)  # Follow lane for a while (in the 'other' lane)

        # Step 3: Perform a U-turn (right turn → left turn) to return
        rospy.loginfo("Step 3: Initiating U-turn...")
        self.control_wheels(0, -0.7)  # Turn right
        rospy.sleep(0.3)
        self.control_wheels(0.2, 0)  # Drive straight briefly
        rospy.sleep(0.7)
        self.control_wheels(0, 0.7)  # Turn left
        rospy.sleep(0.3)


        rospy.loginfo("Returned to original lane. Resuming normal operation.")
        self.obstacle_avoidance_active = False # Reset flag to allow avoidance again

    def turn_right_callback(self, event): #called - not used in the obstacle avoidance maneuver anymore
        rospy.logwarn("Turning Right") #turn right
        self.control_wheels(0, -0.7) #call to turn right
        #call turn around again
        self.side_distance_timer = rospy.Timer(rospy.Duration(self.side_distance_timer_period), self.start_lane_following_callback, oneshot=True) #oneshot called again

    def start_lane_following_callback(self, event): #called - not used in the obstacle avoidance maneuver anymore
        """
        Function to start lane following again. - not used in the obstacle avoidance maneuver anymore
        Note: This is never actually called as part of the final product.
        """
        rospy.wait_for_service('lane_following_control')

        try:
            control_lane = rospy.ServiceProxy('lane_following_control', SetBool)
            control_lane(True) #Turn on line following

        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

    def calc_circle_pattern(self, height, width):
        """Calculates the physical locations of each dot in the pattern (hardcoded)."""
        self.circlepattern_dist = self.distance_between_centers
        self.circlepattern = np.zeros([height * width, 3], np.float32)  # Specify float32 data type

        for i in range(0, 7):  # width is 7
            for j in range(0, 3):  # height is 3
                self.circlepattern[i + j * 7, 0] = (
                    self.circlepattern_dist * i - self.circlepattern_dist * (7 - 1) / 2
                )
                self.circlepattern[i + j * 7, 1] = (
                    self.circlepattern_dist * j - self.circlepattern_dist * (3 - 1) / 2
                )

if __name__ == "__main__":
    duckiebot_avoidance_node = DuckiebotAvoidanceNode(node_name="duckiebot_avoidance_node")
    rospy.spin()