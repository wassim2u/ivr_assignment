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


class image_converter:

  # Defines publisher and subscriber
  def __init__(self):
    # initialize the node named image_processing
    rospy.init_node('image_processing', anonymous=True)
    # initialize a publisher to send images from camera1 to a topic named image_topic1
    self.image_pub1 = rospy.Publisher("image_topic1",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw",Image,self.callback1)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()
    #initialize a publisher to move the joint2
    self.joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
    #initialize a publisher to move the joint3
    self.joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
    #initialize a publisher to move the joint4
    self.joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
    #get start time
    self.init_time = rospy.get_time()

  ###Functions to move joints 2-3###
  def move_joint2(self, t):
    return float((np.pi/2)*np.sin((np.pi/15.0)*t))

  def move_joint3(self, t):
    return float((np.pi/2)*np.sin((np.pi/18.0)*t))

  def move_joint4(self, t):
    return float((np.pi/2)*np.sin((np.pi/20.0)*t))

  def compute_joint_angles(self):
    time = np.array([rospy.get_time() - self.init_time])
    joint2_angle = self.move_joint2(time)
    joint3_angle = self.move_joint3(time)
    joint4_angle = self.move_joint4(time)
    return joint2_angle, joint3_angle, joint4_angle

  ###Functions to compute forward kinematics###
  def forward_kinematics(self, x, y, z, w):
    xv = (float) (7*sin(y+z)+6*sin(w+y+z)-6*sin(w-y+z)+7*sin(y-z))/4
    zv = (float) (20*cos(x)+14*cos(x+z)+12*cos(w+x+z)-12*cos(w-x+z)+7*cos(x+y+z)+6*cos(w+x+y+z)+6*cos(w-x+y+z)+7*cos(x-y+z)+6*cos(w+x-y+z)+6*cos(w-x-y+z)-14*cos(x-z)+7*cos(x+y-z)+7*cos(x-y-z))/8
    yv = (float) (20*sin(x)+14*sin(x+z)+12*sin(w+x+z)+12*sin(w-x+z)+7*sin(x+y+z)+6*sin(w+x+y+z)-6*sin(w-x+y+z)+7*sin(x-y+z)+6*sin(w+x-y+z)-6*sin(w-x-y+z)-14*sin(x-z)+7*sin(x+y-z)+7*sin(x-y-z))/8
    return xv,yv,zv

  ###These functions are merely there to verify the forward_kinematics function########################
  def get_end_effector(self, cv_image):
    lower_red = np.array([0,0,100])
    upper_red = np.array([50,50,255])
    cx_r, cy_r = self.get_joint_coordinates(cv_image, lower_red, upper_red)

    return cx_r, cy_r

  def convert_pixel_to_metres(self, cv_image):
    lower_blue = np.array([100,0,0])
    upper_blue = np.array([255,50,50])
    cx_b, cy_b = self.get_joint_coordinates(cv_image, lower_blue, upper_blue)
        
    lower_green = np.array([0,100,0])
    upper_green = np.array([50,255,50])
    cx_g, cy_g = self.get_joint_coordinates(cv_image, lower_green, upper_green)

    #get pixel to metres ratio
    b = np.array([cx_b, cy_b])
    g = np.array([cx_g, cy_g])

    s = np.sum((b-g)**2)
    dist = np.sqrt(s)
    return 3.5/dist

  def get_center(self, cv_image):
    lower_yellow = np.array([0,100,100])
    upper_yellow = np.array([50,255,255])
    cx_y, cy_y = self.get_joint_coordinates(cv_image, lower_yellow, upper_yellow)

    return cx_y, cy_y

  def get_joint_coordinates(self,cv_image, lower, upper):
    im = self.threshold(cv_image, lower, upper)
    M = cv2.moments(im)

    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    return cx, cy
  ##########################################################################################

  def threshold(self, cv_image, lower_range, upper_range):
    #create mask
    mask = cv2.inRange(cv_image, lower_range, upper_range)
    return mask

  # Recieve data from camera 1, process it, and publish
  def callback1(self,data):

    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    ratio = self.convert_pixel_to_metres(cv_image)

    e1_x, e1_y = self.get_end_effector(cv_image)
    c1_x, c1_y = self.get_center(cv_image)
    e1_x *= ratio
    e1_y *= ratio
    c1_x *= ratio
    c1_y *= ratio
    print(c1_x-e1_x, c1_y-e1_y)

    #update to current time
    self.time = rospy.get_time()
    joint2_angle, joint3_angle, joint4_angle = self.compute_joint_angles()

    print(self.forward_kinematics(0.0,0.0,0.0,1.0))

    # Receive the image
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
      #publish new joint angles
      #self.joint2_pub.publish(joint2_angle)
      #self.joint3_pub.publish(joint3_angle)
      #self.joint4_pub.publish(joint4_angle)
    except CvBridgeError as e:
      print(e)
    
    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)

    im1=cv2.imshow('window1', self.cv_image1)
    cv2.waitKey(1)
    # Publish the results
    try: 
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
    except CvBridgeError as e:
      print(e)

# call the class
def main(args):
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)

