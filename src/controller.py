#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
import sympy as sp
from sympy import Matrix, symbols, diff, Eq, linsolve
from sympy.matrices.dense import matrix_multiply_elementwise
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from helpers import *


class controller:

    def __init__(self):
        # initialize the node named controller
        rospy.init_node('controller', anonymous=True)

        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

        self.y_sub1 = message_filters.Subscriber("/image1/joint_centers/yellow", Float64MultiArray)
        self.b_sub1 = message_filters.Subscriber("/image1/joint_centers/blue", Float64MultiArray)
        self.g_sub1 = message_filters.Subscriber("/image1/joint_centers/green", Float64MultiArray)
        self.r_sub1 = message_filters.Subscriber("/image1/joint_centers/red", Float64MultiArray)
        self.target_sub1 = message_filters.Subscriber("/image1/target_center", Float64MultiArray)
        self.box_sub1 = message_filters.Subscriber("/image1/box_center", Float64MultiArray)

        self.y_sub2 = message_filters.Subscriber("/image2/joint_centers/yellow", Float64MultiArray)
        self.b_sub2 = message_filters.Subscriber("/image2/joint_centers/blue", Float64MultiArray)
        self.g_sub2 = message_filters.Subscriber("/image2/joint_centers/green", Float64MultiArray)
        self.r_sub2 = message_filters.Subscriber("/image2/joint_centers/red", Float64MultiArray)
        self.target_sub2 = message_filters.Subscriber("/image2/target_center", Float64MultiArray)
        self.box_sub2 = message_filters.Subscriber("/image2/box_center", Float64MultiArray)


        #initialize a publisher for joint angle
        self.joint2_t2 = rospy.Publisher("/robot/theta2", Float64, queue_size=10)
        self.joint3_t3 = rospy.Publisher("/robot/theta3", Float64, queue_size=10)
        self.joint4_t4 = rospy.Publisher("/robot/theta4", Float64, queue_size=10)
        self.green_joint3dx = rospy.Publisher("/robot/greenx", Float64, queue_size=10)
        self.green_joint3dy = rospy.Publisher("/robot/greeny", Float64, queue_size=10)
        self.green_joint3dz = rospy.Publisher("/robot/greenz", Float64, queue_size=10)
        self.green_joint3dax = rospy.Publisher("/robot/greenax", Float64, queue_size=10)
        self.green_joint3day = rospy.Publisher("/robot/greenay", Float64, queue_size=10)
        self.green_joint3daz = rospy.Publisher("/robot/greenaz", Float64, queue_size=10)

        #Initialise publisher to be used for plotting against actual values of target
        self.target_3d_pub = rospy.Publisher("task2_2/target_3d", Float64MultiArray, queue_size=10)
        #Initialise publishers to be used for plotting to visualise
        #the comparison of end effector positions of FK vs vision
        self.end_effector_FK_pub = rospy.Publisher("task3_1/end_effector_position/FK", Float64MultiArray, queue_size=10)
        self.end_effector_vision_pub = rospy.Publisher("task3_2/end_effector_position/vision", Float64MultiArray, queue_size=10)

        self.prev_theta2 = 0.0
        self.prev_theta3 = 0.0
        self.prev_theta4 = 0.0

        # initialize a publisher to move the joint1
        self.joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
        # initialize a publisher to move the joint2
        self.joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
        # initialize a publisher to move the joint3
        self.joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
        # initialize a publisher to move the joint4
        self.joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)

        self.q = np.array([1.0,1.0,1.0,1.0])

        # synchronise incoming channels using the timestamps contained in their headers
        # slop defines the delay in seconds with which messages are synchronized
        ts = message_filters.ApproximateTimeSynchronizer([
            self.y_sub1, self.b_sub1, self.g_sub1, self.r_sub1,
            self.y_sub2, self.b_sub2, self.g_sub2, self.r_sub2,
            self.target_sub1, self.target_sub2,
            self.box_sub1, self.box_sub2],
            queue_size=10, slop=0.1, allow_headerless=True)
        ts.registerCallback(self.callback)

        #task 3_2
        # set error margin for readings
        self.error_margin = 0.0
        self.prev_time = rospy.get_time()
        self.error = np.array([0.0, 0.0, 0.0], dtype='float64')
        self.error_d = np.array([0.0, 0.0, 0.0], dtype='float64')
        self.Kp = np.array([[0.4, 0, 0], [0, 0.4, 0], [0, 0, 0.4]])
        self.Kd = np.array([[0.4, 0, 0], [0, 0.4, 0], [0, 0, 0.4]])


        # task4_2
        self.previous_q = np.array([0.0, 0.0, 0.0, 0.0])
        self.previous_end_effector_position = np.array([0.0, 0.0, 0.0])
        self.previous_box_obstacle_position = np.array([0.0, 0.0, 0.0])
        self.Kp_4_2 = np.array([[0.6, 0, 0], [0, 0.6, 0], [0, 0, 0.6]])
        self.Kd_4_2 = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        self.previous_jacobian = np.ones((3,4)) # Initialise a jacobian the same dimension as the jacobian we calculated



    def get_x(self, image_1_coordinates):

        self.x_blue = image_1_coordinates[1].data[0]
        self.x_yellow = image_1_coordinates[0].data[0]

        self.x_green = 0.0
        self.x_red = 0.0

        self.x_green == image_1_coordinates[2].data[0]

        
        self.x_red == image_1_coordinates[1].data[0]

        # print(self.x_yellow, self.x_blue, self.x_green, self.x_red)

    def get_y(self, image_2_coordinates):

        self.y_blue = image_2_coordinates[1].data[0]
        self.y_yellow = image_2_coordinates[0].data[0]

        self.y_green = 0.0
        self.y_red = 0.0


        self.y_green == image_2_coordinates[2].data[0]

        self.y_red == image_2_coordinates[1].data[0]
        #print(self.y_yellow, self.y_blue, self.y_green, self.y_red)

    def get_z(self, image_1_coordinates, image_2_coordinates):
        # Regard z-value of blue joint as "baseline" for green joint and z-value of yellow
        # joint as "baseline" for red joint
        # i.e. the z-value of the green and red joint cannot be lower than
        # z_blue-error margin or z_yellow-error_margin, respectively
        # Take average of the two readings to find z-value for blue joint
        self.z_blue = (image_1_coordinates[1].data[1] + image_2_coordinates[1].data[1]) / 2
        self.z_yellow = (image_1_coordinates[0].data[1] + image_2_coordinates[0].data[1]) / 2

        g1_z = 0.0
        g2_z = 0.0
        r1_z = 0.0
        r2_z = 0.0

        # if the green joint is obstructed, get the z-value from other camera for green joint and
        # use x or y value from obstructing joint
        """
        if (image_1_coordinates[2].data[1] == 1234.0):
            g1_z = image_2_coordinates[2].data[1]
            g2_z = image_2_coordinates[2].data[1]
        elif (image_2_coordinates[2].data[1] == 0.0):
            g2_z = image_1_coordinates[2].data[1]
            g1_z = image_1_coordinates[2].data[1]

        if (image_1_coordinates[3].data[1] == 0.0):
            r1_z = image_2_coordinates[3].data[1]
            r2_z = image_2_coordinates[3].data[1]
        elif (image_2_coordinates[3].data[1] == 0.0):
            r2_z = image_1_coordinates[3].data[1]
            r1_z = image_1_coordinates[3].data[1]
<<<<<<< HEAD
<<<<<<< HEAD
        # if z_green or z_red is below z_blue in both pictures, set their z-value to that
        # of z_blue
        # First, check green
        if (g1_z > self.z_blue + self.error_margin and g2_z > self.z_blue + self.error_margin):
=======
=======
>>>>>>> 0f5475ac09ca37acabe79fdd017075fbfc38c649
        """

        if (g1_z > self.z_blue+self.error_margin and g2_z > self.z_blue+self.error_margin):
            self.z_green = self.z_blue
        else:
            self.z_green = (g1_z + g2_z) / 2

        # then check red
        if (r1_z > self.z_yellow + self.error_margin and r2_z > self.z_yellow + self.error_margin):
            self.z_red = self.z_blue
        else:
            self.z_red = (r1_z + r2_z) / 2

        print(self.z_yellow, self.z_blue, self.z_green, self.z_red)

     # From the coordinates recieved from the topics for both images, construct 3D coordinates (measured in pixels)
     # Note: Image 2 - xz plane; Image 1 - yz plane.
    def create_new_3d_coordinates_from_data(self, y1, b1, g1, r1, y2, b2, g2, r2, target1, target2, box1, box2):
        #Collect the coordinates from each topic.
        self.yellow_center1 = (np.asarray(y1.data))
        self.blue_center1 = np.asarray(b1.data)
        self.green_center1 = np.asarray(g1.data)
        self.red_center1 = (np.asarray(r1.data))
        self.target_center1 = np.asarray(target1.data)
        self.box_center1 = np.asarray(box1.data)

        self.yellow_center2 = (np.asarray(y2.data))
        self.blue_center2 = np.asarray(b2.data)
        self.green_center2 = np.asarray(g2.data)
        self.red_center2 = (np.asarray(r2.data))
        self.target_center2 = np.asarray(target2.data)
        self.box_center2 = np.asarray(box2.data)

        # blue and yellow should always on the same x and y-axis:
        self.blue_center1[0] = self.yellow_center1[0]
        self.blue_center2[0] = self.yellow_center2[0]

        #Take the averages of the z from both images.
        z_center = (self.yellow_center1[1] + self.yellow_center2[1]) / 2
        z_blue = (self.blue_center1[1] + self.blue_center2[1]) / 2
        z_green = (self.green_center1[1] + self.green_center2[1]) / 2
        z_red = (self.red_center1[1] + self.red_center2[1]) / 2
        z_target = (self.target_center1[1] + self.target_center2[1]) / 2
        z_box = (self.box_center1[1] + self.box_center2[1]) / 2

        # z of green must not be below z of blue (due to the configuration space of the joint, being -pi/2, pi/2)
        # These are measured in pixel coordinates, so our z positive is downwards here
        if z_green > z_blue:
            z_green = z_blue

        self.yellow_3d = np.array([self.yellow_center2[0], self.yellow_center1[0], z_center])
        self.blue_3d = np.array([self.blue_center2[0], self.blue_center1[0], z_blue])
        self.green_3d = np.array([self.green_center2[0], self.green_center1[0], z_green])
        self.red_3d = np.array([self.red_center2[0], self.red_center1[0], z_red])
        self.target_3d = np.array([self.target_center2[0], self.target_center1[0], z_target])
        self.box_3d = np.array([self.box_center2[0], self.box_center1[0], z_box])

    # Change the coordinates such that yellow is our new origin (base frame), so all coordinates are with respect to it.
    # In addition, the function changes the coordinates to be in meters after finding a fixed estimated ratio.
    def changeAxis(self):
        new_yellow_3d = np.array([0, 0, 0])
        new_blue_3d = self.yellow_3d - self.blue_3d
        new_green_3d = self.yellow_3d - self.green_3d
        new_red_3d = self.yellow_3d - self.red_3d
        new_target_3d = self.yellow_3d - self.target_3d
        new_box_3d = self.yellow_3d - self.box_3d
        # Using a fixed estimated ratio to change our coordinates from pixel to meters.
        # We used a fixed ratio over dynamic one to minimise the margin of error because objects closer seem bigger
        # and gave us higher ratios than when the robot was stationary.
        ratio = 0.0389
        self.yellow_3d = new_yellow_3d
        self.blue_3d = new_blue_3d * ratio
        self.green_3d = new_green_3d * ratio
        self.red_3d = new_red_3d * ratio
        self.target_3d = new_target_3d * ratio
        self.box_3d = new_box_3d * ratio

        # Negate y - this was done to match up with the direction of positive direction
        # of the three-dimensional axes from the image axis.
        self.yellow_3d[0] = - self.yellow_3d[0]
        self.blue_3d[0] = - self.blue_3d[0]
        self.green_3d[0] = - self.green_3d[0]
        self.red_3d[0] = - self.red_3d[0]
        self.target_3d[0] = - self.target_3d[0]
        self.box_3d[0] = - self.box_3d[0]

        # Negate y - this was done to match up with the direction of positive direction
        # of the three-dimensional axes from the image axis.
        self.yellow_3d[1] = - self.yellow_3d[1]
        self.blue_3d[1] = - self.blue_3d[1]
        self.green_3d[1] = - self.green_3d[1]
        self.red_3d[1] = - self.red_3d[1]
        self.target_3d[1] = - self.target_3d[1]
        self.box_3d[1] = -self.box_3d[1]

        # print("Values changed to meters:")
        # print("Yellow " + str(self.yellow_3d))
        # print("Blue:" + str(self.blue_3d))
        # print("Green" + str(self.green_3d))
        # print("Red" + str(self.red_3d))
        #




    def get_jacobian(self, theta1, theta2, theta3, theta4):
        a, b, c, d = symbols('a b c d')

        fk = fk_matrix(a,b,c,d)
        xx = fk[0]
        yy = fk[1]
        zz = fk[2]



        # Using sympy diff, we can now compute the jacobian matrix:
        # Row 1
        j_11 = diff(xx, a)
        j_12 = diff(xx, b)
        j_13 = diff(xx, c)
        j_14 = diff(xx, d)

        # Row 2
        j_21 = diff(yy, a)
        j_22 = diff(yy, b)
        j_23 = diff(yy, c)
        j_24 = diff(yy, d)

        # Row 3
        j_31 = diff(zz, a)
        j_32 = diff(zz, b)
        j_33 = diff(zz, c)
        j_34 = diff(zz, d)

        # Now make it a proper matrix substituting the actual angles
        jacobian = Matrix([
            [j_11.subs(a, theta1).subs(b, theta2).subs(c, theta3).subs(d, theta4),
             j_12.subs(a, theta1).subs(b, theta2).subs(c, theta3).subs(d, theta4),
             j_13.subs(a, theta1).subs(b, theta2).subs(c, theta3).subs(d, theta4),
             j_14.subs(a, theta1).subs(b, theta2).subs(c, theta3).subs(d, theta4),
             ],

            [j_21.subs(a, theta1).subs(b, theta2).subs(c, theta3).subs(d, theta4),
             j_22.subs(a, theta1).subs(b, theta2).subs(c, theta3).subs(d, theta4),
             j_23.subs(a, theta1).subs(b, theta2).subs(c, theta3).subs(d, theta4),
             j_24.subs(a, theta1).subs(b, theta2).subs(c, theta3).subs(d, theta4),
            ],

            [j_31.subs(a, theta1).subs(b, theta2).subs(c, theta3).subs(d, theta4),
             j_32.subs(a, theta1).subs(b, theta2).subs(c, theta3).subs(d, theta4),
             j_33.subs(a, theta1).subs(b, theta2).subs(c, theta3).subs(d, theta4),
             j_34.subs(a, theta1).subs(b, theta2).subs(c, theta3).subs(d, theta4)
            ]

        ])
        jacobian = np.array(jacobian).astype(np.float64)
        # jacobian = np.empty((3,4))
        return jacobian

    def get_inverse_jacobian(self, j):
        return np.linalg.pinv(j)

    # Get the change in time
    def calculate_delta_t(self):
        # estimate delta T
        current_time = rospy.get_time()
        dt = current_time - self.prev_time  # delta T
        self.prev_time = current_time
        return dt

    #Task 3_2:
    #Returns joint angles that will be published to control the robot and make it follow the target,
    #by using the image feedback of robot end-effector position.
    def closed_loop_control(self, theta1, theta2, theta3, theta4, target):
        q = np.array([theta1, theta2, theta3, theta4])
        # Get change in time
        dt = self.calculate_delta_t()

        #Get the end effector positions from the fk_matrix function that returns them.
        fk = fk_matrix(0.0, theta2, theta3, theta4)
        current_position = fk

        #Denote the desired position as target
        pos_d = target

        # estimate derivative of error
        self.error_d = ((pos_d - current_position) - self.error) / dt

        # estimate error
        self.error = pos_d - current_position


        # Get pseudo-jacobian, from the jacobian derived from the end effector positions from the FK equation
        jacobian = self.get_jacobian(theta1, theta2, theta3, theta4)
        j_inv = self.get_inverse_jacobian(jacobian)


        # calculate velocity of the joints
        dq_d = np.dot(j_inv, (np.dot(self.Kd, self.error_d.transpose()) + np.dot(self.Kp, self.error.T)))
        q_d = q + (dt * dq_d)
        # Our configuration space for theta2,theta3, and theta4 is between -pi/2 and pi/2
        for i in range(1, 4):
            angle = q_d[i]
            if angle > np.pi / 2:
                q_d[i] = np.pi / 2
            if angle < -np.pi / 2:
                q_d[i] = -np.pi / 2

        return q_d



    def blue_joint_rotation(self, green_joint):

        # Assume we are at a joint rotation of 0 for both joints
        theta2 = 0.0
        theta3 = 0.0

        green_curr = Matrix([[0.0], [0.0], [6.0]])

        a, b, c = symbols('a b c')

        # Treat green joint as end effector, act as if blue had been rotated by 0 degrees and we wanted to see how to get to the desired position of the green joint
        # First, get the forward kinematics function up until green (treating green as the end effector)
        a01 = a_0_1(0.0)
        a02 = a01 * a_1_2(b)
        a03 = a02 * a_2_3(c)

        # get the jacobian of this function
        xx = a03.col(3).row(0)
        yy = a03.col(3).row(1)
        zz = a03.col(3).row(2)

        jacobian = Matrix([
            [diff(xx, a), diff(xx, b), diff(xx, c)],
            [diff(yy, a), diff(yy, b), diff(yy, c)],
            [diff(zz, a), diff(zz, b), diff(zz, c)]
        ])

        # get pseudo-inverse jacobian
        j_inv = jacobian.pinv()

        coordinate_difference = green_curr - green_joint
        current_error = coordinate_difference
        delta_t = 1

        q_dot = j_inv * (current_error)
        q_dot = q_dot.subs(b, theta2)
        q_dot = q_dot.subs(c, theta3)

        print(q_dot)

    def last_attempt(self, bg):
        a, b = symbols('a b')
        rot_x = Matrix([[sp.cos(a), -sp.sin(a), 0, 0], [sp.sin(a), sp.cos(a), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        rot_y = Matrix([[sp.sin(b), -sp.cos(b), 0, 0], [sp.cos(b), sp.sin(b), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        arr = Matrix([[0.0], [0.0], [1.0], [1.0]])

        transform = rot_x * rot_y * arr

        x = Eq(sp.asin(a) * sp.asin(b), bg[0, 0])
        y = Eq(-sp.acos(a) * sp.asin(b), bg[1, 0])
        z = Eq(sp.acos(b), bg[2, 0])

        print(sp.solve((x, y, z), (b)))

    def rotation_matrix_from_vectors(self, vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        a = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        b = np.arctan2(-rotation_matrix[2, 0], np.sqrt((rotation_matrix[2, 1] ** 2) + rotation_matrix[2, 2] ** 2))
        c = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        print(a, b, c)
        return rotation_matrix

    def task_2_1(self):
        # First we need the forwards kinematics equations, giving the thetas as variables
        a = 0.0
        b, c, d = symbols('b c d')

        x_rot = Matrix([[sp.cos(b), -sp.sin(b), 0, 0], [sp.sin(b), sp.cos(b), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        y_rot = Matrix([[sp.cos(c), -sp.sin(c), 0, 0], [sp.sin(c), sp.cos(c), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        init = Matrix([[0.0], [0.0], [1.0], [1.0]])
        diff_1_3_y = self.blue_3d[1] - self.green_3d[1]
        diff_1_3_x = self.blue_3d[0] - self.green_3d[0]
        diff_1_3_z = self.blue_3d[2] - self.green_3d[2]
        diff_1_3_x = self.blue_3d[0] - self.green_3d[0]
        diff_1_3_y = self.blue_3d[1] - self.green_3d[1]
        diff_1_3_z = self.blue_3d[2] - self.green_3d[2]
        v = Matrix([[diff_1_3_x], [diff_1_3_y], [diff_1_3_z], [1]])
        transform = (x_rot * y_rot * v)
        print(transform)

        # Get the frame transformations
        a01 = a_0_1(a)

        # We do not need to compute a, since joint 1 is fixed and hence we will always have a=0.0
        # To compute the angle of joint 2, we need the coordinates of the center of joint 2 and joint 4 and subtract them

        # v = Matrix([[diff_1_3_x], [diff_1_3_y], [diff_1_3_z], [1]])
        # print(a01.shape)
        # mtx = (a01.inv())*v
        # m = sp.atan(mtx[1,0]/mtx[0,0])
        # a02 = a_1_2(m)*a01
        # mtx2 = a02.inv()*v
        # m2 = sp.atan(mtx2[2,0]/mtx2[1,0])
        # a03 = (a_2_3(m2)*a02)
        # theta2 = np.arctan2(mtx.row(1).col(0), mtx.row(2).col(0))
        # transform = mtx.inv()*v
        # mtx3 = a03.inv()*mtx2
        # m3 = sp.atan(mtx3[1,0]/mtx3[0,0])
        # a03 = a_3_4(m3)*a03
        # mtx4 = a03.inv()*mtx3
        # m3 = np.arctan2(mtx4[2,0], mtx[1,0])

        # theta3 = np.arctan2(transform.row(1), transform.row(2))
        # print(m, m2)

        # a03 = a_2_3(c)*a02
        # a04 = a_3_4(d)*a03

    """
    #Generates angles for the robot in accordance with task 3.1
    def task_3_1(self):

        angles_file = open("../angles.txt", "w")
        readings_file = open("../readings.txt", "w")

        for i in range(10):
            theta1 = 0
            theta2 = 0
            theta3 = 0
            theta4 = 0

            #please don't put this on r/badcode
            while (theta1 == 0 or abs(theta1) == 1  or theta2 == 0 or abs(theta2) == 1 or theta3 == 0 or abs(theta3) == 1 or theta4 == 0 or abs(theta4) == 1):
                theta1 = random.randint(-10, 10)
                theta2 = random.randint(-10, 10)
                theta3 = random.randint(-10, 10)
                theta4 = random.randint(-10, 10)

            theta1 = np.pi/(theta1)
            theta2 = np.pi/(theta2)
            theta3 = np.pi/(theta3)
            theta4 = np.pi/(theta4)

            angles_file.write("%.4f %.4f %.4f %.4f\n" %(theta1, theta2, theta3, theta4))

            readings = self.get_jacobian(theta1, theta2, theta3, theta4)
            readings_file.write("%.4f %.4f %.4f\n" %(readings[0], readings[1], readings[2]))

        readings_file.close()
        angles_file.close()
        print("Successfully written to files.")
    """


    #Task 4_2:
    #Returns joint angles that will enable the controller to avoid hitting the box using the redundacy of robot
    #while following the target. It is also closed-loop as it uses image feedback of end effector position.
    def null_space_control(self, theta1, theta2, theta3, theta4, end_effector, target, box_obstacle):
        old_q = np.array([theta1, float(theta2), float(theta3), float(theta4)]).T

        
        # k here is a constant that will be squared and added to the element jacobian.dot(jacobian.T)
        # Using damped Jacobian avoids singularity by adding a small term so the columns are linearly independant.
        k = 1
        identity_matrix = np.eye(3)
        jacobian = self.get_jacobian(theta1, theta2, theta3, theta4)
        damped_J = jacobian.T.dot(np.linalg.inv(jacobian.dot(jacobian.T) + (k ** 2) * identity_matrix))

        # Get the change in time
        dt = self.calculate_delta_t()
        #compute secondary task
        secondary_task = self.compute_secondary_task_4_2(end_effector, box_obstacle, old_q)


        # derivative of error
        xy_d = target
        xy_e = end_effector
        error = xy_d - xy_e
        error_derivative = (error - self.error) / dt
        self.error = error  # Update error


        # Calculate Null-space projection - has no effect on end effector, used for achieving our secondary task which is to stay away from box.
        identity_matrix = np.eye(4)
        null_space_projection = (identity_matrix - damped_J.dot(jacobian)).dot(secondary_task)

        # qdot is joint velocity/joint angles derivative, calculate it using the null_space_projection
        q_dot = damped_J.dot((self.Kp_4_2.dot(error.transpose()) + self.Kd_4_2.dot(error_derivative.transpose()))) \
                + \
                null_space_projection

        # Find the new joint angles using the q_dot we found
        new_q = old_q + dt * q_dot

        # Our configuration space for theta1 is between -pi and pi
        # For theta2,theta3, and theta4, they are between -pi/2 and pi/2
        for i in range(0, 4):
            angle = new_q[i]
            if i == 0 and angle > np.pi:
                angle = np.pi
            if i ==0 and angle < -np.pi:
                angle = -np.pi
            if angle > np.pi / 2:
                new_q[i] = np.pi / 2
            if angle < -np.pi / 2:
                new_q[i] = -np.pi / 2



        return new_q

    # Differentiate cost with respect to q, given cost function,
    # which will be the distance between end effector and box.
    # The secondary task is to maximise this distance to avoid the box.
    def compute_secondary_task_4_2(self, end_effector, box_obstacle, q):
        # define k0 constant , which signifies how fast we want to move in the derivative direction
        k0_constant = 0.001

        # Get change in q, to be used for taking the derivative numerically
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
        dq0 = q0 - self.previous_q[0]
        dq1 = q1 - self.previous_q[1]
        dq2 = q2 - self.previous_q[2]
        dq3 = q3 - self.previous_q[3]
        # Define the cost function to be distance between end effector and box. Calculate the previous distance too.
        cost = np.linalg.norm(end_effector - box_obstacle)
        previous_end_eff = self.previous_end_effector_position
        previous_obstacle = self.previous_box_obstacle_position
        previous_cost = np.linalg.norm(previous_end_eff - previous_obstacle)

        # Compute partial differentiation for cost function with respect to each q
        # Offset is included in the denominator in case there's no change in q to prevent division by zero.
        offset = 0.1
        derivative_wrt_q0 = (cost - previous_cost) / (dq0 + offset)
        derivative_wrt_q1 = (cost - previous_cost) / (dq1 + offset)
        derivative_wrt_q2 = (cost - previous_cost) / (dq2 + offset)
        derivative_wrt_q3 = (cost - previous_cost) / (dq3 + offset)

        # Group our calculations into one matrix.
        cost_derivative = np.array([derivative_wrt_q0,
                                    derivative_wrt_q1,
                                    derivative_wrt_q2,
                                    derivative_wrt_q3])

        # Update by storing the current values for next iteration
        self.previous_end_effector_position = end_effector
        self.previous_box_obstacle_position = box_obstacle
        self.previous_q = q

        # Finally, calculate the secondary task. k0_constant tells us how fast or how slow we want to do t
        qdot_zero = k0_constant * cost_derivative.T

        return qdot_zero
    ###Functions to move joints 2-4 ###
    def move_joint2(self, t):
        return (np.pi / 2) * np.sin((np.pi / 15.0) * t)

    def move_joint3(self, t):
        return (np.pi / 2) * np.sin((np.pi / 18.0) * t)

    def move_joint4(self, t):
        return (np.pi / 2) * np.sin((np.pi / 20.0) * t)

    def compute_joint_angles(self):
        time = rospy.get_time()
        joint2_angle = self.move_joint2(time)
        joint3_angle = self.move_joint3(time)
        joint4_angle = self.move_joint4(time)
        return joint2_angle, joint3_angle, joint4_angle

    def callback(self, y1, b1, g1, r1, y2, b2, g2, r2, target1, target2, box1, box2):
        self.image_1_coordinates = np.array([y1, b1, g1, r1])
        self.image_2_coordinates = np.array([y2, b2, g2, r2])

        # Get coordinates from the two images and change the values to make them with respect to yellow center in meters
        self.create_new_3d_coordinates_from_data(y1, b1, g1, r1, y2, b2, g2, r2, target1, target2, box1, box2)
        self.changeAxis()

        #####Task 3_2 + 4_2 ###
        a,b,c = self.compute_joint_angles()
        self.joint2_angle = Float64()
        self.joint3_angle = Float64()
        self.joint4_angle = Float64()
        self.joint2_angle,self.joint3_angle,self.joint4_angle = self.compute_joint_angles()

        actual_coordinate = fk_green(0.0, self.joint2_angle, self.joint3_angle)
        fk = fk_matrix(np.pi/8, np.pi/2, np.pi/8, np.pi/2)

        blue_joint = np.array([[self.blue_3d[0]], [self.blue_3d[1]], [self.blue_3d[2]]])
        green_joint = np.array([[self.green_3d[0]], [self.green_3d[1]], [self.green_3d[2]]])
        actual_green = fk_green(0.0, self.joint2_angle, self.joint3_angle)
        red_joint = np.array([[self.red_3d[0]], [self.red_3d[1]], [self.red_3d[2]]])
        # print("Coordinate error: \n", green_joint-actual_green[0:3])
        
        theta2, theta3 = get_joint2_3_angles(green_joint, self.prev_theta2, self.prev_theta3)
        # print("Theta 3 error: \n", self.joint3_angle-theta3)
        self.prev_theta2 = theta2
        self.prev_theta3 = theta3
        # theta4 = get_joint4_angles(theta2, theta3, blue_joint, green_joint, red_joint)

        # ---- Task 3_2 ---- #
        new_q = self.closed_loop_control(theta1=0,
                                         theta2 = self.q[1],
                                         theta3 = self.q[2],
                                         theta4 = self.q[3],
                                         target= self.target_3d)
        self.q = new_q


        # # ---- Task 4_2 ---- #
        # new_q = self.null_space_control(theta1=self.q[0], theta2=self.q[1], theta3=self.q[2], theta4=self.q[3],
        #                      end_effector=self.red_3d,
        #                      target=self.target_3d,
        #                      box_obstacle=self.box_3d
        #                      )
        # print("NEW Q: ")
        # print(new_q)
        # self.q = new_q




        #task 2 angle 4 attempt
        # theta4 = - np.arctan2(self.red_3d[1],self.red_3d[2])  + theta2 - 2
        theta4 = - np.arctan2(self.red_3d[0] - self.green_3d[0], self.red_3d[2]- self.green_3d[2] ) - theta2
        print("ANGLES " + str(theta4))
        # theta4 = np.pi/2 + theta4 - 0.4



        # ---------For publishing-------#
        ###task2_2
        target = Float64MultiArray()
        target.data = self.target_3d
        ###task3_1

        end_effector_FK = Float64MultiArray()
        end_effector_FK.data = fk
        print( "FK NEW"+ str(end_effector_FK.data))
        end_effector_vision = Float64MultiArray()
        print("VISION" + str(self.red_3d))
        end_effector_vision.data = self.red_3d

        ### Task 2 ###
        # self.compute_joint_angles()
        # green_joint = np.array([[self.green_3d[0]], [self.green_3d[1]], [self.green_3d[2]]])
        # red_joint = np.array([[self.red_3d[0]], [self.red_3d[1]], self.red_3d[2]])
        # print("Joint 4 error: ")
        #theta2, theta3, theta4, target = self.trajectory()
        # theta2, theta3 = get_joint2_3_angles(green_joint)
        # theta4 = get_joint4_angles(theta2, theta3, red_joint)

        try:
            self.joint2_t2.publish(theta2)
            self.joint3_t3.publish(theta3)
            self.joint4_t4.publish(theta4)
            self.green_joint3dx.publish(self.green_3d[0])
            self.green_joint3dy.publish(self.green_3d[1])
            self.green_joint3dz.publish(self.green_3d[2])
            self.green_joint3dax.publish(actual_coordinate[0])
            self.green_joint3day.publish(actual_coordinate[1])
            self.green_joint3daz.publish(actual_coordinate[2])
            # self.joint2_pub.publish(self.joint2_angle)
            # self.joint3_pub.publish(self.joint3_angle)
            # self.joint4_pub.publish(self.joint4_angle)

            # Task 2_2  - Publishes the coordinates of the sphere target's center to the topic defined.
            self.target_3d_pub.publish(target)
            # Task 3_1 - Publishes the coordinates of the end effector found through vision and through the FK equation.
            self.end_effector_FK_pub.publish(end_effector_FK)
            self.end_effector_vision_pub.publish(end_effector_vision)
            #Task 3_2 / Task 4_2 - Publishes the new angles computed to the robot to follow the target
            self.joint1_pub.publish(new_q[0])
            self.joint2_pub.publish(new_q[1])
            self.joint3_pub.publish(new_q[2])
            self.joint4_pub.publish(new_q[3])

        except CvBridgeError as e:
            print(e)


# call the class
def main(args):
    cntr = controller()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)