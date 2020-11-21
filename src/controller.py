#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
import sympy as sp
from sympy import symbols, diff, sin, cos, Matrix
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import random
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

        self.y_sub2 = message_filters.Subscriber("/image2/joint_centers/yellow", Float64MultiArray)
        self.b_sub2 = message_filters.Subscriber("/image2/joint_centers/blue", Float64MultiArray)
        self.g_sub2 = message_filters.Subscriber("/image2/joint_centers/green", Float64MultiArray)
        self.r_sub2 = message_filters.Subscriber("/image2/joint_centers/red", Float64MultiArray)

        self.target_sub1 = message_filters.Subscriber("/image1/target_center", Float64MultiArray)
        self.target_sub2 = message_filters.Subscriber("/image2/target_center", Float64MultiArray)

        #self.joint_angle_1 = rospy.Subscriber("/robot/joint1_position_controller/command", Float64)
        #self.joint_angle_2 = rospy.Subscriber("/robot/joint2_position_controller/command", Float64)
        #self.joint_angle_3 = rospy.Subscriber("/robot/joint3_position_controller/command", Float64)
        #self.joint_angle_4 = rospy.Subscriber("/robot/joint1_position_controller/command", Float64)


        #Initialize publisher to move joint 1 
        self.joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
        #initialize a publisher to move the joint2
        self.joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
        #initialize a publisher to move the joint3
        self.joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
        #initialize a publisher to move the joint4
        self.joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)

        # synchronise incoming channels using the timestamps contained in their headers
        # slop defines the delay in seconds with which messages are synchronized
        ts = message_filters.ApproximateTimeSynchronizer([
                                                        self.y_sub1, self.b_sub1, self.g_sub1, self.r_sub1,
                                                        self.y_sub2, self.b_sub2, self.g_sub2, self.r_sub2,
                                                        self.target_sub1, self.target_sub2
                                                        ],
                                                        queue_size=10,slop= 0.1, allow_headerless=True)
        ts.registerCallback(self.callback)

        #set error margin for readings
        self.error_margin = 0.0
        self.prev_time = rospy.get_time()
        self.error = np.array([0.0,0.0], dtype='float64')  
        self.error_d = np.array([0.0,0.0], dtype='float64')
        self.Kp = np.array([[10, 0], [0, 10]])
        self.Kd = np.array([[0.1, 0], [0, 0.1]])

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

        print(self.x_yellow, self.x_blue, self.x_green, self.x_red)

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

        print(self.y_yellow, self.y_blue, self.y_green, self.y_red)

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
        self.yellow_center1 = np.asarray(y1.data)
        self.blue_center1 = np.asarray(b1.data)
        self.green_center1 = np.asarray(g1.data)
        self.red_center1 = np.asarray(r1.data)
        self.target_center1 = np.asarray(target1.data)

        self.yellow_center2 = np.asarray(y2.data)
        self.blue_center2 = np.asarray(b2.data)
        self.green_center2 = np.asarray(g2.data)
        self.red_center2 = np.asarray(r2.data)
        self.target_center2 = np.asarray(target2.data)

        # blue and yellow should always on the same x and y-axis:
        self.blue_center1[0] = self.yellow_center1[0]
        self.blue_center2[0] = self.yellow_center2[0]


        self.z_center = (self.yellow_center1[1] + self.yellow_center2[1])/2
        self.z_blue = (self.blue_center1[1] + self.blue_center2[1])/2
        self.z_green = (self.green_center1[1] + self.green_center2[1])/2
        self.z_red = (self.red_center1[1] + self.red_center2[1])/2
        self.z_target = (self.target_center1[1] + self.target_center2[1])/2

        # z of green must not be below z of blue (due to the configuration space of the joint, being -pi/2, pi/2)
        # These are measured in pixel coordinates, so our z positive is downwards here
        if self.z_green > self.z_blue:
            self.z_green = self.z_blue


        self.yellow_3d = np.array([self.yellow_center2[0], self.yellow_center1[0], self.z_center])
        self.blue_3d = np.array([self.blue_center2[0], self.blue_center1[0], self.z_blue])
        self.green_3d = np.array([self.green_center2[0], self.green_center1[0], self.z_green])
        self.red_3d = np.array([self.red_center2[0], self.red_center1[0], self.z_red])
        self.target_3d = np.array([self.target_center2[0], self.target_center1[0], self.z_target])


    def changeAxis(self):
        new_yellow_3d = np.array([0, 0, 0])
        new_blue_3d = self.yellow_3d - self.blue_3d
        new_green_3d = self.yellow_3d - self.green_3d
        new_red_3d = self.yellow_3d - self.red_3d
        ratio = 0.0389
        print("Ratio:" + str(ratio))
        self.yellow_3d = new_yellow_3d
        self.blue_3d = new_blue_3d * 0.0389
        self.green_3d = new_green_3d * 0.0389
        self.red_3d = new_red_3d * 0.0389
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
        self.get_x(self.image_1_coordinates, self.image_2_coordinates)
        self.get_y(self.image_1_coordinates, self.image_2_coordinates)

        #Get end effector position
        end_effector = self.forward_kinematics(theta1, theta2, theta3, theta4)

        #Get delta t
        current_time = rospy.get_time()
        dt = current_time - self.prev_time
        self.prev_time = current_time

        #TODO: Get target
        target = self.trajectory()

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

    def task_2_1(self):
        #First we need the forwards kinematics equations, giving the thetas as variables
        a = 0.0
        b, c, d = symbols('b c d')

        #Get the frame transformations
        a01 = a_0_1(a)
        a02 = a01*a_1_2(b)
        a03 = a02*a_2_3(c)
        a04 = a03*a_3_4(d)

        #We do not need to compute a, since joint 1 is fixed and hence we will always have a=0.0
        #To compute the angle of joint 2, we need the coordinates of the center of joint 2 and joint 4 and subtract them
        diff_2_4_z = self.z_blue-self.z_green
        diff_2_4_y = self.y_blue-self.y_green
        
        #Compute the arctan
        theta2 = np.arctan2(diff_2_4_z, diff_2_4_y)

        print(theta2)


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
            

    def callback(self,y1,b1,g1,r1,y2,b2,g2,r2, target1, target2):
        self.image_1_coordinates = np.array([y1, b1, g1, r1])
        self.image_2_coordinates = np.array([y2, b2, g2, r2])

        self.get_z()
        self.get_x()
        self.get_y()

        self.task_2_1()

        # Get coordinates from the two images and change the values to make them with respect to yellow center in meters
        #self.create_new_3d_coordinates_from_data(y1, b1, g1, r1, y2, b2, g2, r2,target1, target2)
        #self.changeAxis()
        #q = self.closed_loop_control(0.0, 0.0, 0.0, 0.0, self.red_3d)
        try:
            #publish new joint angles
            #self.joint1_pub.publish(q[0])
            #self.joint2_pub.publish(q[1])
            #self.joint3_pub.publish(q[2])
            #self.joint4_pub.publish(q[3])
            print(q[0])

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