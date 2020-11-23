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
import math

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

        
        self.yz = rospy.Publisher("/robot/yellow_z", Float64, queue_size=10)
        self.bz = rospy.Publisher("/robot/blue_z", Float64, queue_size=10)
        self.gz = rospy.Publisher("/robot/green_z", Float64, queue_size=10)
        self.rz = rospy.Publisher("/robot/red_z", Float64, queue_size=10)

        self.target_3d_pub = rospy.Publisher("task2_2/target_3d", Float64MultiArray, queue_size=10)
        self.end_effector_FK_pub = rospy.Publisher("task3_1/end_effector_position/FK", Float64MultiArray, queue_size=10)
        self.end_effector_vision_pub = rospy.Publisher("task3_1/end_effector_position/vision", Float64MultiArray, queue_size=10)



        # synchronise incoming channels using the timestamps contained in their headers
        # slop defines the delay in seconds with which messages are synchronized
        ts = message_filters.ApproximateTimeSynchronizer([
                                                        self.y_sub1, self.b_sub1, self.g_sub1, self.r_sub1,
                                                        self.y_sub2, self.b_sub2, self.g_sub2, self.r_sub2,
                                                        self.target_sub1, self.target_sub2,
                                                        ],
                                                        queue_size=10,slop= 0.1, allow_headerless=True)
        ts.registerCallback(self.callback)

        #set error margin for readings
        self.error_margin = 0.0
        self.prev_time = rospy.get_time()
        self.error = np.array([0.0,0.0,0.0], dtype='float64')
        self.error_d = np.array([0.0,0.0,0.0], dtype='float64')
        self.Kp = np.array([[10, 0, 0], [0, 10 ,0], [0, 0, 10]])
        self.Kd = np.array([[0.1, 0, 0], [0, 0.1, 0],[0, 0 , 0.1]])

    def get_x(self, image_1_coordinates):
        
        self.x_blue = image_1_coordinates[1].data[0]
        self.x_yellow = image_1_coordinates[0].data[0]

        self.x_green = 0.0
        self.x_red = 0.0
        
        if (image_1_coordinates[2].data[0] == 0.0):
            if (self.z_green == self.z_blue):
                self.x_green == self.x_blue
            elif (self.z_green == self.z_yellow):
                self.x_green == self.x_yellow
        else:
            self.x_green == image_1_coordinates[2].data[0]

        if (image_1_coordinates[3].data[0] == 0.0):
            if (self.z_red == self.z_green):
                self.x_red = self.x_green
            elif (self.z_red == self.z_blue):
                self.x_red = self.x_blue
            elif (self.z_red == self.z_yellow):
                self.x_red = self.x_yellow
        else:
            self.x_red == image_1_coordinates[1].data[0]

        #print(self.x_yellow, self.x_blue, self.x_green, self.x_red)

    def get_y(self, image_2_coordinates):
        
        self.y_blue = image_2_coordinates[1].data[0]
        self.y_yellow = image_2_coordinates[0].data[0]

        self.y_green = 0.0
        self.y_red = 0.0
        
        if (image_2_coordinates[2].data[0] == 0.0):
            if (self.z_green == self.z_blue):
                self.y_green == self.y_blue
            elif (self.z_green == self.z_yellow):
                self.y_green == self.y_yellow
        else:
            self.y_green == image_2_coordinates[2].data[0]

        if (image_2_coordinates[3].data[0] == 0.0):
            if (self.z_red == self.z_green):
                self.y_red = self.y_green
            elif (self.z_red == self.z_blue):
                self.y_red = self.y_blue
            elif (self.z_red == self.z_yellow):
                self.y_red = self.y_yellow
        else:
            self.y_red == image_2_coordinates[1].data[0]

        #print(self.y_yellow, self.y_blue, self.y_green, self.y_red)

    def get_z(self, image_1_coordinates, image_2_coordinates):
        #Regard z-value of blue joint as "baseline" for green joint and z-value of yellow
        #joint as "baseline" for red joint
        #i.e. the z-value of the green and red joint cannot be lower than 
        #z_blue-error margin or z_yellow-error_margin, respectively
        #Take average of the two readings to find z-value for blue joint
        self.z_blue = (image_1_coordinates[1].data[1]+image_2_coordinates[1].data[1])/2
        self.z_yellow = (image_1_coordinates[0].data[1]+image_2_coordinates[0].data[1])/2

        g1_z = 0.0
        g2_z = 0.0
        r1_z = 0.0
        r2_z = 0.0

        #if the green joint is obstructed, get the z-value from other camera for green joint and
        # use x or y value from obstructing joint
        if (image_1_coordinates[2].data[1] == 0.0):
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
        #if z_green or z_red is below z_blue in both pictures, set their z-value to that
        #of z_blue
        #First, check green
        if (g1_z > self.z_blue+self.error_margin and g2_z > self.z_blue+self.error_margin):
            self.z_green = self.z_blue
        else:
            self.z_green = (g1_z+g2_z)/2

        #then check red
        if (r1_z > self.z_yellow+self.error_margin and r2_z > self.z_yellow+self.error_margin):
            self.z_red = self.z_blue
        else:
            self.z_red = (r1_z+r2_z)/2

        print(self.z_yellow, self.z_blue, self.z_green, self.z_red)

        # Note: Image 2 - xz plane; Image 1 - yz plane

    def create_new_3d_coordinates_from_data(self, y1, b1, g1, r1, y2, b2, g2, r2, target1, target2):
        self.yellow_center1 = (np.asarray(y1.data))
        self.blue_center1 = np.asarray(b1.data)
        self.green_center1 = np.asarray(g1.data)
        self.red_center1 = (np.asarray(r1.data))
        self.target_center1 = np.asarray(target1.data)
        # self.box_center1 = np.asarray(box1.data)


        self.yellow_center2 = (np.asarray(y2.data))
        self.blue_center2 = np.asarray(b2.data)
        self.green_center2 = np.asarray(g2.data)
        self.red_center2 = (np.asarray(r2.data))
        self.target_center2 = np.asarray(target2.data)
        # self.box_center2 = np.asarray(box2.data)

        # blue and yellow should always on the same x and y-axis:
        self.blue_center1[0] = self.yellow_center1[0]
        self.blue_center2[0] = self.yellow_center2[0]


        z_center = (self.yellow_center1[1] + self.yellow_center2[1])/2
        z_blue = (self.blue_center1[1] + self.blue_center2[1])/2
        z_green = (self.green_center1[1] + self.green_center2[1])/2
        z_red = (self.red_center1[1] + self.red_center2[1])/2
        z_target = (self.target_center1[1] + self.target_center2[1])/2
        # z_box = (self.box_center1[1]+ self.box_center2[1])/2


        # z of green must not be below z of blue (due to the configuration space of the joint, being -pi/2, pi/2)
        # These are measured in pixel coordinates, so our z positive is downwards here
        if z_green > z_blue:
            z_green = z_blue


        self.yellow_3d = np.array([self.yellow_center2[0], self.yellow_center1[0], z_center])
        self.blue_3d = np.array([self.blue_center2[0], self.blue_center1[0], z_blue])
        self.green_3d = np.array([self.green_center2[0], self.green_center1[0], z_green])
        self.red_3d = np.array([self.red_center2[0], self.red_center1[0], z_red])
        self.target_3d = np.array([self.target_center2[0], self.target_center1[0], z_target])
        # self.box_3d = np.array([self.box_center2[0], self.box_center1[0], z_box])

    def changeAxis(self):
        new_yellow_3d = np.array([0, 0, 0])
        new_blue_3d = self.yellow_3d - self.blue_3d
        new_green_3d = self.yellow_3d - self.green_3d
        new_red_3d = self.yellow_3d - self.red_3d
        new_target_3d = self.yellow_3d - self.target_3d
        # new_box_3d = self.yellow_3d - self.box_3d
        ratio = 0.0389
        print("Ratio:" + str(ratio))
        self.yellow_3d = new_yellow_3d
        self.blue_3d = new_blue_3d * 0.0389
        self.green_3d = new_green_3d * 0.0389
        self.red_3d = new_red_3d * 0.0389
        self.target_3d = new_target_3d * 0.0389
        # self.box_3d = new_box_3d * 0.0389
        
        #negate x 
        self.yellow_3d[0] = - self.yellow_3d[0]
        self.blue_3d[0] = - self.blue_3d[0]
        self.green_3d[0] = - self.green_3d[0]
        self.red_3d[0] = - self.red_3d[0]
        self.target_3d[0] = - self.target_3d[0]
        # self.box_3d[0] = - self.box_3d[0]

        #negate y
        self.yellow_3d[1] = - self.yellow_3d[1]
        self.blue_3d[1] = - self.blue_3d[1]
        self.green_3d[1] = - self.green_3d[1]
        self.red_3d[1] = - self.red_3d[1]
        self.target_3d[1] = - self.target_3d[1]
        # self.box_3d[1] = -self.box_3d[1]

        print("Values changed to meters:")
        print("Yellow " + str(self.yellow_3d))
        print("Blue:" + str(self.blue_3d))
        print("Green" + str(self.green_3d))
        print("Red" + str(self.red_3d))

    def closed_loop_control(self, theta1, theta2, theta3, theta4, target):

        #Note: the robot is redundant, so we can get away with just fixing theta1=0.0 
        theta1 = float(0.0)

        q = np.array([theta1, float(theta2), float(theta3), float(theta4)]).T

        #Compute pseudoinverse of jacobian
        j_inv = np.linalg.pinv(self.get_jacobian(theta1, theta2, theta3, theta4))

        #Compute joint coordinates
        self.get_z(self.image_1_coordinates, self.image_2_coordinates)
        self.get_x(self.image_2_coordinates)
        self.get_y(self.image_1_coordinates)

        #Get end effector position
        end_effector = self.forward_kinematics(theta1, theta2, theta3, theta4)

        #Get delta t
        current_time = rospy.get_time()
        dt = current_time - self.prev_time
        self.prev_time = current_time

        #TODO: Get target
        target = self.target_3d

        current_error = pos-end_effector
        self.error_d = (current_error-self.error)/dt
        self.error = current_error

        #now compute q dot
        kp_d_err = np.dot(self.Kp, self.error.transpose())
        kd_d_err = np.dot(self.Kd, self.error_d.transpose())
        q_dot = np.dot(j_inv, (kp_d_err+kd_d_err))

        q = q + (dt * q_dot)
        return q

    def get_trajectory_error(self, cv_image):
        angles = self.get_angles(cv_image)

        current_time = rospy.get_time()
        current_trajectory = self.trajectory()
        dt = current_time - self.first_time
        self.first_time = current_time

        trajectory_difference = current_trajectory-self.error
        self.error_d = trajectory_difference/dt
        self.error = current_trajectory
        
        j_inv = np.linalg.pinv(self.get_jacobian(angles))
        q_dot = np.dot(j_inv, self.error_d.transpose())
        new_q = angles + dt*q_dot
        return new_q

    def forward_kinematics(self, a, b, c, d):
        xx = (-7*sp.cos(a+b+c)+7*sp.cos(a-b+c)-7*sp.cos(a+b-c)+7*sp.cos(a-b-c)-6*sp.cos(a+b+d)-6*sp.cos(a-b+d)-3*sp.cos(a+b+c+d)+3*sp.cos(a-b+c+d)-3*sp.cos(a+b-c+d)+3*sp.cos(a-b-c+d)+6*sp.cos(a+b-d)+6*sp.cos(a-b-d)-3*sp.cos(a+b+c-d)+3*sp.cos(a-b+c-d)-3*sp.cos(a+b-c-d)+3*sp.cos(a-b-c-d)+14*sp.sin(a+c)-14*sp.sin(a-c)+6*sp.sin(a+c+d)-6*sp.sin(a-c+d)+6*sp.sin(a+c-d)-6*sp.sin(a-c-d))/8
        yy = (-14*sp.cos(a+c)+14*sp.cos(a-c)-6*sp.cos(a+c+d)+6*sp.cos(a-c+d)-6*sp.cos(a+c-d)+6*sp.cos(a-c-d)-7*sp.sin(a+b+c)+7*sp.sin(a-b+c)-7*sp.sin(a+b-c)+7*sp.sin(a-b-c)-6*sp.sin(a+b+d)-6*sp.sin(a-b+d)-3*sp.sin(a+b+c+d)+3*sp.sin(a-b+c+d)-3*sp.sin(a+b-c+d)+3*sp.sin(a-b-c+d)+6*sp.sin(a+b-d)+6*sp.sin(a-b-d)-3*sp.sin(a+b+c-d)+3*sp.sin(a-b+c-d)-3*sp.sin(a+b-c-d)+3*sp.sin(a-b-c-d))/8
        zz = (7*sp.cos(b+c)+7*sp.cos(b-c)+6*sp.cos(b+d)+3*sp.cos(b+c+d)+3*sp.cos(b-c+d)-6*sp.cos(b-d)+3*sp.cos(b+c-d)+3*sp.cos(b-c-d)+10)/4
        return [xx, yy, zz]

    def get_jacobian(self, theta1, theta2, theta3, theta4):
        a, b, c, d = symbols('a b c d')

        fk = self.forward_kinematics(a,b,c,d)
        xx = fk[0]
        yy = fk[1]        
        zz = fk[2]

        #Using sympy diff, we can now compute the jacobian matrix:
        #Row 1
        j_11 = diff(xx, a)
        j_12 = diff(xx, b)
        j_13 = diff(xx, c)
        j_14 = diff(xx, d)

        #Row 2
        j_21 = diff(yy, a)
        j_22 = diff(yy, b)
        j_23 = diff(yy, c)
        j_24 = diff(yy, d)

        #Row 3
        j_31 = diff(zz, a)
        j_32 = diff(zz, b)
        j_33 = diff(zz, c)
        j_34 = diff(zz, d)

        #Now make it a proper matrix substituting the actual angles
        #jacobian = np.array([[j_11.subs(a,theta1, b, theta2, c, theta3, d, theta4), j_12.subs(a,theta1, b, theta2, c, theta3, d, theta4), j_13.subs(a,theta1, b, theta2, c, theta3, d, theta4), j_14.subs(a,theta1, b, theta2, c, theta3, d, theta4)], [j_21.subs(a,theta1, b, theta2, c, theta3, d, theta4), j_22.subs(a,theta1, b, theta2, c, theta3, d, theta4), j_23.subs(a,theta1, b, theta2, c, theta3, d, theta4), j_24.subs(a,theta1, b, theta2, c, theta3, d, theta4)], [j_31.subs(a,theta1, b, theta2, c, theta3, d, theta4), j_32.subs(a,theta1, b, theta2, c, theta3, d, theta4), j_33.subs(a,theta1, b, theta2, c, theta3, d, theta4), j_34.subs(a,theta1, b, theta2, c, theta3, d, theta4)]])
        jacobian = np.empty((3,4))
        return jacobian


    def get_inverse_jacobian(self, j):
        return np.linalg.pinv(j)

    def blue_joint_rotation(self, green_joint):

        #Assume we are at a joint rotation of 0 for both joints
        theta2 = 0.0
        theta3 = 0.0

        green_curr = Matrix([[0.0],[0.0],[6.0]])

        a, b, c = symbols('a b c')

        #Treat green joint as end effector, act as if blue had been rotated by 0 degrees and we wanted to see how to get to the desired position of the green joint
        #First, get the forward kinematics function up until green (treating green as the end effector)
        a01 = a_0_1(0.0)
        a02 = a01*a_1_2(b)
        a03 = a02*a_2_3(c)

        #get the jacobian of this function
        xx = a03.col(3).row(0)
        yy = a03.col(3).row(1)
        zz = a03.col(3).row(2)

        jacobian = Matrix([
            [diff(xx, a), diff(xx,b), diff(xx, c)],
            [diff(yy, a), diff(yy,b), diff(yy, c)],
            [diff(zz, a), diff(zz,b), diff(zz, c)]
        ])

        #get pseudo-inverse jacobian
        j_inv = jacobian.pinv()

        coordinate_difference = green_curr-green_joint
        current_error = coordinate_difference
        delta_t = 1

        q_dot = j_inv*(current_error)
        q_dot = q_dot.subs(b, theta2)
        q_dot = q_dot.subs(c, theta3)

        print(q_dot) 

    def last_attempt(self, bg):
        a,b = symbols('a b')
        rot_x = Matrix([[sp.cos(a), -sp.sin(a), 0, 0], [sp.sin(a), sp.cos(a), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        rot_y = Matrix([[sp.sin(b), -sp.cos(b), 0, 0], [sp.cos(b), sp.sin(b), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        arr = Matrix([[0.0], [0.0], [1.0], [1.0]])

        transform = rot_x*rot_y*arr

        x = Eq(sp.asin(a)*sp.asin(b), bg[0,0])
        y = Eq(-sp.acos(a)*sp.asin(b), bg[1,0])
        z = Eq(sp.acos(b), bg[2,0])

        print(sp.solve((x,y,z), (b)))

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
        a = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
        b = np.arctan2(-rotation_matrix[2,0], np.sqrt((rotation_matrix[2,1]**2)+rotation_matrix[2,2]**2))
        c = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
        print(a,b,c)
        return rotation_matrix



    def task_2_1(self):
        #First we need the forwards kinematics equations, giving the thetas as variables
        a = 0.0
        b, c, d = symbols('b c d')

        x_rot=Matrix([[sp.cos(b), -sp.sin(b), 0, 0], [sp.sin(b), sp.cos(b), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        y_rot=Matrix([[sp.cos(c), -sp.sin(c), 0, 0], [sp.sin(c), sp.cos(c), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        init = Matrix([[0.0], [0.0], [1.0], [1.0]])
        diff_1_3_y = self.blue_3d[1]-self.green_3d[1]
        diff_1_3_x = self.blue_3d[0]-self.green_3d[0]
        diff_1_3_z = self.blue_3d[2]-self.green_3d[2]
        diff_1_3_x = self.blue_3d[0]-self.green_3d[0]
        diff_1_3_y = self.blue_3d[1]-self.green_3d[1]
        diff_1_3_z = self.blue_3d[2]-self.green_3d[2]
        v = Matrix([[diff_1_3_x], [diff_1_3_y], [diff_1_3_z], [1]])
        transform= (x_rot*y_rot*v)
        print(transform)

        #Get the frame transformations
        a01 = a_0_1(a)
        
        #We do not need to compute a, since joint 1 is fixed and hence we will always have a=0.0
        #To compute the angle of joint 2, we need the coordinates of the center of joint 2 and joint 4 and subtract them
        
        #v = Matrix([[diff_1_3_x], [diff_1_3_y], [diff_1_3_z], [1]])
        #print(a01.shape)
        #mtx = (a01.inv())*v
        #m = sp.atan(mtx[1,0]/mtx[0,0])
        #a02 = a_1_2(m)*a01
        #mtx2 = a02.inv()*v
        #m2 = sp.atan(mtx2[2,0]/mtx2[1,0])
        #a03 = (a_2_3(m2)*a02)
        #theta2 = np.arctan2(mtx.row(1).col(0), mtx.row(2).col(0))
        #transform = mtx.inv()*v
        #mtx3 = a03.inv()*mtx2
        #m3 = sp.atan(mtx3[1,0]/mtx3[0,0])
        #a03 = a_3_4(m3)*a03
        #mtx4 = a03.inv()*mtx3
        #m3 = np.arctan2(mtx4[2,0], mtx[1,0])
        
        #theta3 = np.arctan2(transform.row(1), transform.row(2))
        #print(m, m2)
        
        #a03 = a_2_3(c)*a02
        #a04 = a_3_4(d)*a03


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
    def task4_2(self, end_effector, box_obstacle):
        constant_k = 0.3 #Should be positive constant
        #Cost function is distance between  end effector and box squared.
        # Maximise this secondary task , ie) maximise the distance, to avoid  the box
        cost = np.transpose(end_effector - box_obstacle).dot(end_effector - box_obstacle)
        secondary_task = constant_k.dot(sp.diff(cost,q)) #differntiate with respect to q : w for q - w for next q / delta q
        identity = np.ones(4,4)
        #qdot is joint velocity
        qdot = psuedo_jacobian.dot(end_effector_velocity) + (identity - pseudo_jacobian.dot(jacobian))*secondary_task
        new_q = angles + dt * q_dot
        return new_q

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

    def callback(self,y1,b1,g1,r1,y2,b2,g2,r2, target1, target2):
        self.image_1_coordinates = np.array([y1, b1, g1, r1])
        self.image_2_coordinates = np.array([y2, b2, g2, r2])

        self.get_z(self.image_1_coordinates, self.image_2_coordinates)
        self.get_x(self.image_2_coordinates)
        self.get_y(self.image_1_coordinates)
        print(self.x_red,self.y_red,self.z_red)

        # Get coordinates from the two images and change the values to make them with respect to yellow center in meters
        self.create_new_3d_coordinates_from_data(y1, b1, g1, r1, y2, b2, g2, r2,target1, target2)
        self.changeAxis()
        
        # print(self.box_3d)


        #---------For publishing-------#
        ###task2_2
        target = Float64MultiArray()
        target.data = self.target_3d
        ###task3_1
        
        a,b,c = self.compute_joint_angles()
        end_effector_FK = Float64MultiArray()
        end_effector_FK.data = self.forward_kinematics(0,a,b,c)
        end_effector_vision = Float64MultiArray()
        end_effector_vision.data = self.red_3d
        try:
            #publish new joint angles
            # self.joint2_pub.publish(joint2)
            # self.joint3_pub.publish(joint3)
            # self.joint4_pub.publish(joint4)
            #Task 2_2
            self.target_3d_pub.publish(target)
            #Task 3_1
            self.end_effector_FK_pub.publish(end_effector_FK)
            self.end_effector_vision_pub.publish(end_effector_vision)



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