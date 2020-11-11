#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
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
  def forward_kinematics(self, theta1, theta2, theta3, theta4):
    x = (1/4)*(7*np.sin(theta2+theta3)+6*np.sin(theta4+theta2+theta3)-6*np.sin(theta4-theta2+theta3)+7*np.sin(theta2-theta3))
    y = (1/8)*(-20*np.sin(theta1)-14*np.sin(theta1+theta3)-12*np.sin(theta4+theta1+theta3)-12*np.sin(theta4-theta1+theta3)-7*np.sin(theta1+theta2+theta3)-6*np.sin(theta4+theta1+theta2+theta3)+6*np.sin(theta4-theta1+theta2+theta3)+6*np.sin(theta4-theta1-theta2+theta3)+14*np.sin(theta1-theta3)-7*np.sin(theta1+theta2-theta3)-7*np.sin(theta1-theta2-theta3))
    z = (1/8)*(-20*np.cos(theta1)-14*np.cos(theta1+theta3)-12*np.cos(theta4+theta1+theta3)-12*np.cos(theta4-theta1+theta3)-7*np.cos(theta1+theta2+theta3)-6*np.cos(theta4+theta1+theta2+theta3)+6*np.cos(theta4-theta1+theta2+theta3)+6*np.cos(theta4-theta1-theta2+theta3)+14*np.cos(theta1-theta3)-7*np.cos(theta1+theta2-theta3)-7*np.cos(theta1-theta2-theta3))

    return x,y,z

  # Recieve data from camera 1, process it, and publish
  def callback1(self,data):

    print(self.forward_kinematics(0,0,0,0))

    #update to current time
    self.time = rospy.get_time()
    joint2_angle, joint3_angle, joint4_angle = self.compute_joint_angles()

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


