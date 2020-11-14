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


class image_converter_2:

  # Defines publisher and subscriber
  def __init__(self):
    # initialize the node named image_processing
    rospy.init_node('image_processing', anonymous=True)
    # initialize a publisher to send images from camera2 to a topic named image_topic2
    self.image_pub2 = rospy.Publisher("image_topic2",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
    self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw",Image,self.callback2)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()


    self.joint_centers_yellow_pub2 = rospy.Publisher("/image2/joint_centers/yellow", Float64MultiArray, queue_size=10)
    self.joint_centers_blue_pub2 = rospy.Publisher("/image2/joint_centers/blue", Float64MultiArray, queue_size=10)
    self.joint_centers_green_pub2 = rospy.Publisher("/image2/joint_centers/green", Float64MultiArray, queue_size=10)
    self.joint_centers_red_pub2 = rospy.Publisher("/image2/joint_centers/red", Float64MultiArray, queue_size=10)


  # Return a dictionary that contains binary images for each circle
  # Retrieve the image of a specific circle from the dictionary using their colour as key (eg. dictionary_name['Blue'])
  def detect_circles(self, img):
    # Turn RGB Image into HSV colour space
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Detect Red Circle
    red_mask = cv2.inRange(hsv_image, (0, 10, 20), (10, 255, 255))
    # Detect Blue Circle
    blue_mask = cv2.inRange(hsv_image, (90, 5, 10), (130, 255, 255))
    # Detect Green Circle
    green_mask = cv2.inRange(hsv_image, (45, 10, 20), (70, 255, 255))
    # Detect Yellow Circle
    yellow_mask = cv2.inRange(hsv_image, (28, 10, 20), (35, 255, 255))

    # cv2.imshow('Red Circle - Binary Image2', red_mask)
    # cv2.imshow('Blue Circle - Binary Image2', blue_mask)
    # cv2.imshow('Green Circle - Binary Image2', green_mask)
    # cv2.imshow('Yellow Circle - Binary Image2', yellow_mask)
    # cv2.waitKey(1)

    binary_images = {"Blue": blue_mask, "Green": green_mask, "Red": red_mask, "Yellow": yellow_mask}
    return binary_images

  # Find center of a specific circle. The image returned from camera2 is of plane xz
  def find_color_center2(self, mask_color):
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(mask_color, kernel, iterations=3)
    M = cv2.moments(dilated_mask)

    cx = int(M['m10'] / M['m00'])
    cz = int(M['m01'] / M['m00'])
    return np.array([cx, cz])

  # Recieve data, process it, and publish
  def callback2(self,data):
    # Recieve the image
    try:
      self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)
    im2=cv2.imshow('window2', self.cv_image2)
    cv2.waitKey(1)

    ##Task 2##
    masked_circles = self.detect_circles(self.cv_image2)
    yellow_center = self.find_color_center2(masked_circles['Yellow'])
    blue_center = self.find_color_center2(masked_circles['Blue'])
    green_center = self.find_color_center2(masked_circles['Green'])
    red_center = self.find_color_center2(masked_circles['Red'])


    self.y_center = Float64MultiArray()
    self.y_center.data = yellow_center
    self.b_center = Float64MultiArray()
    self.b_center.data = blue_center
    self.g_center = Float64MultiArray()
    self.g_center.data = green_center
    self.r_center = Float64MultiArray()
    self.r_center.data = red_center



    # Publish the results
    try: 
      self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
      #publish joint centers with coordinates (x,z) taken from image 2
      self.joint_centers_yellow_pub2.publish(self.y_center)
      self.joint_centers_blue_pub2.publish(self.b_center)
      self.joint_centers_green_pub2.publish(self.g_center)
      self.joint_centers_red_pub2.publish(self.r_center)


    except CvBridgeError as e:
      print(e)

# call the class
def main(args):
  ic = image_converter_2()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)


