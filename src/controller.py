#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from numpy import sin, cos
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
import message_filters

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
        # synchronise incoming channels using the timestamps contained in their headers
        # slop defines the delay in seconds with which messages are synchronized
        ts = message_filters.ApproximateTimeSynchronizer([
                                                        self.y_sub1, self.b_sub1, self.g_sub1, self.r_sub1,
                                                        self.y_sub2, self.b_sub2, self.g_sub2, self.r_sub2
                                                        ],
                                                        queue_size=10,slop= 0.1, allow_headerless=True)
        ts.registerCallback(self.callback)

        #set error margin for readings
        self.error_margin = 0.0

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

    def get_jacobian(self, a, b, c, d):
        x = (26*cos(a+c+d)-26*cos(a-c+d)+26*cos(a+c-d)-26*cos(a-c-d)-20*sin(a+b)+20*sin(a-b)-26*sin(a+b+d)-26*sin(a-b+d)-13*sin(a+b+c+d)+13*sin(a-b+c+d)-13*sin(a+b-c+d)+13*sin(a-b-c+d)+26*sin(a+b-d)+26*sin(a-b-d)-13*sin(a+b+c-d)+13*sin(a-b+c-d)-13*sin(a+b-c-d)+13*sin(a-b-c-d))/16
        y = (20*cos(a+b)-20*cos(a-b)+26*cos(a+b+d)+26*cos(a-b+d)+13*cos(a+b+c+d)-13*cos(a-b+c+d)+13*cos(a+b-c+d)-13*cos(a-b-c+d)-26*cos(a+b-d)-26*cos(a-b-d)+13*cos(a+b+c-d)-13*cos(a-b+c-d)+13*cos(a+b-c-d)-13*cos(a-b-c-d)+26*sin(a+c+d)-26*sin(a-c+d)+26*sin(a+c-d)-26*sin(a-c-d))/16
        z = (20*cos(b)+26*cos(b+d)+13*cos(b+c+d)+13*cos(b-c+d)-26*cos(b-d)+13*cos(b+c-d)+13*cos(b-c-d))/8
        #x = (-20*sin(a+b)-20*sin(a-b)+26*sin(a+b+d)-26*sin(a-b+d)+26*sin(a+c+d)-13*sin(a+b+c+d)-13*sin(a-b+c+d)-26*sin(a-c+d)-13*sin(a+b-c+d)-13*sin(a-b-c+d)-26*sin(a+b-d)+26*sin(a-b-d)+26*sin(a+c-d)-13*sin(a+b+c-d)-13*sin(a-b+c-d)-26*sin(a-c-d)-13*sin(a+b-c-d)-13*sin(a-b-c-d))/16
        #y = (20*cos(a+b)+20*cos(a-b)-26*cos(a+b+d)+26*cos(a-b+d)-26*cos(a+c+d)+13*cos(a+b+c+d)+13*cos(a-b+c+d)+26*cos(a-c+d)+13*cos(a+b-c+d)+13*cos(a-b-c+d)+26*cos(a+b-d)-26*cos(a-b-d)-26*cos(a+c-d)+13*cos(a+b+c-d)+13*cos(a-b+c-d)+26*cos(a-c-d)+13*cos(a+b-c-d)+13*cos(a-b-c-d))/16
        #z = (20*sin(b)-26*sin(b+d)+13*sin(b+c+d)+13*sin(b-c+d)+26*sin(b-d)+13*sin(b+c-d)+13*sin(b-c-d))/8
        return np.array([[x], [y], [z]])
    def callback(self,y1,b1,g1,r1,y2,b2,g2,r2):

        image_1_coordinates = np.array([y1, b1, g1, r1])
        image_2_coordinates = np.array([y2, b2, g2, r2])
        print(self.get_jacobian(0.0, 0.0, 0.0, 0.0))

        self.get_z(image_1_coordinates, image_2_coordinates)
        

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