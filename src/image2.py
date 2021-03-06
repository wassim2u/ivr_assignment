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
from helpers import *


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
    self.target_center_pub2 = rospy.Publisher("/image2/target_center", Float64MultiArray, queue_size=10)
    self.box_center_pub2 = rospy.Publisher("/image2/box_center", Float64MultiArray, queue_size=10)

    # These variables are used to keep track if the joint positions are visible
    self.circle_colorNames = ["Yellow", "Blue", "Green", "Red"]
    self.previous_circle_positions = {"Yellow": [0.0, 0.0], "Blue": [0.0, 0.0], "Green": [0.0, 0.0], "Red": [0.0, 0.0]}
    self.is_circle_visible = {"Yellow": True, "Blue": True, "Green": True, "Red": True}


    #These variables are used to keep track of target  to be used when approximating the next position of
    #target when it is not visible
    self.is_target_visible = True
    self.previous_target_positions = np.array([0.0,0.0])

    #These variables are used to keep track of the box. Used for task 4.2 to maximise distance from box to end effector.
    self.is_box_visible = True
    self.previous_box_positions = np.array([0.0, 0.0])

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

    binary_images = {"Blue": blue_mask, "Green": green_mask, "Red": red_mask, "Yellow": yellow_mask}
    return binary_images

  # Find the outline of a binary image of a specific circle, and use minEnclosingCircle to predict the center of circle
  # that is partly hidden behind an object.
  # @color: to represent the color that was thresholded returned in the current mask. Used to determine if current
  # circle of specific color is visible or not. For non-visible objects, we will simply return centers found previously.
  # @returns: the center of the circle.
  # NOTE: These do not detect the orange target or box coordinates.
  def predict_joint_center2(self, color, mask):
    morph_kernel = np.ones((5, 5), np.uint8)
    opening_mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN ,morph_kernel)
    #check whether circle is visible by checking its area:
    M = cv2.moments(opening_mask)
    area = M['m00']
    # If the circle is completely hidden, return the previous value
    if (area < 0.01):
      self.is_circle_visible[color] = False
      return self.previous_circle_positions[color]
    else:
      self.is_circle_visible[color] = True

    #Find outline of the shape of the masked circle
    contours, hierarchy = cv2.findContours(opening_mask, 1, 2)
    contour_poly = cv2.approxPolyDP(curve=contours[0], epsilon=0.1, closed=True)
    #Using the outline, draw a circle that encloses the shape of the contour found
    center, radius = cv2.minEnclosingCircle(contour_poly)
    return np.array([int(center[0]), int(center[1])])

  def threshold_orange(self, img):
    # Turn RGB Image into HSV colour space
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Detect Orange Targets
    masks = cv2.inRange(hsv_image, (10, 0, 0), (24, 255, 255))
    kernel = np.ones((3, 3), np.uint8)
    opening_mask = cv2.morphologyEx(masks,cv2.MORPH_OPEN ,kernel)
    return opening_mask

  # Returns the centers of the matched shapes, sphere and box, with the help of classifer that distinguishes the two.
  def predict_orange_centers2(self, img):
    opening_mask = self.threshold_orange(img)
    # Find outlines of our shapes in our binary images
    contours, hierarchy = cv2.findContours(opening_mask, 1, 1)
    # Initialisation
    target_center = self.previous_target_positions
    box_center = self.previous_box_positions
    self.is_target_visible = False
    self.is_box_visible = False
    # if no shape has been found, assume both become hidden and return their previous positions
    if len(contours) == 0:
      return target_center, box_center

    sphere_contour = contours[0]
    box_contour = contours[0]
    sphere_index = 0
    box_index = 1
    #Predict which shape is the sphere
    for cnt in contours:
        #Find center of mass of our current contour.
        M = cv2.moments(cnt)
        cy = int(M["m10"] / M["m00"])
        cz = int(M["m01"] / M["m00"])
        #Take the current region of interest after finding its center.
        IMG_SIZE = 36
        current_shape = opening_mask[int(cz - IMG_SIZE / 2): int(cz + IMG_SIZE / 2),
                                     int(cy - IMG_SIZE / 2): int(cy + IMG_SIZE / 2)]
        # Invert our region of interest to pass to classifer which is built on inverted images
        current_shape = cv2.bitwise_not(current_shape)
        # Increase the number of channels of our array in order to be able to process it in our classifier
        current_shape = cv2.cvtColor(current_shape,cv2.COLOR_GRAY2BGR)
        predictions = get_predictions(current_shape)
        #If the predictions for the first index (which is the result that it is a sphere) is
        #greater than the predictions for the second index (result that it is a box), then we have identified our target.
        if predictions[sphere_index] > predictions[box_index]:
          sphere_contour = cnt
          # Target shape has been detected
          self.is_target_visible = True
        else:
          # Box shape has been detected
          box_contour = cnt
          self.is_box_visible = True

  #If the target is not visible, return the center positions calculated previously
    if (not self.is_target_visible):
      #print("TARGET NOT VISIBLE")
      target_center= self.previous_target_positions
    else:
      contour_poly = cv2.approxPolyDP(curve=sphere_contour, epsilon=0.1, closed=True)
      # Using the outline, draw a circle that encloses the shape of the contour found
      target_center, radius = cv2.minEnclosingCircle(contour_poly)
      # Draw outline of shape predicted to be a sphere to validate result
      self.draw_circle_prediction(img, target_center, radius)

  #If the box is not visible, return the center positions calculated previously
    if (not self.is_box_visible):
      box_center = self.previous_box_positions
    else:
      contour_poly = cv2.approxPolyDP(curve=box_contour, epsilon=1.0, closed=True)
      # Using the outline, draw a rectangle that encloses the shape of the contour found
      topleft_x, topleft_y , width, height = cv2.boundingRect(contour_poly)
      # Draw outline of shape predicted to be a box to validate result  (Uncomment function below)
      # self.draw_box_prediction(img, topleft_x,topleft_y , width, height)

      #Box center would be located half the width and height
      box_center = [topleft_x + (width/2), topleft_y + (height/2)]


    return target_center, box_center

  def update_target_positions(self, current_position):
    self.previous_target_positions = current_position

  def update_box_positions(self, current_position):
    self.previous_box_positions = current_position

  def update_circle_positions(self, color, positions):
    self.previous_circle_positions[color] = positions

  # Draws a circle on the image. Call when needed for visualisation and to check result.
  def draw_circle_prediction(self, img, center, radius):
    new_img = img.copy()
    color = [255, 23, 0]
    line_thickness = 2
    cv2.circle(new_img, (int(center[0]), int(center[1])), int(radius), color, line_thickness)
    #cv2.imshow('Image with predicted shape of circle', new_img)
    # cv2.waitKey(1)

    # Draws a rectangle on the image. Call when needed for visualisation and to check result
  def draw_box_prediction(self, img, topleft_x, topleft_y, width, height):
    new_img = img.copy()
    color = [60, 80, 100]
    line_thickness = 2
    cv2.rectangle(new_img, (topleft_x, topleft_y), (topleft_x + width, topleft_y + height), color, line_thickness)
    cv2.imshow('Image2 with predicted shape of rectangle -', new_img)
    # cv2.waitKey(1)


  # Recieve data, process it, and publish
  def callback2(self,data):
    # Receive the image
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
    # Get Centers of each joint and end effector(red).
    yellow_center = self.predict_joint_center2("Yellow",masked_circles['Yellow'])
    blue_center = self.predict_joint_center2("Blue",masked_circles['Blue'])
    green_center = self.predict_joint_center2("Green", masked_circles['Green'])
    red_center = self.predict_joint_center2("Red",masked_circles['Red'])
    # When the joint or end effector can be detected from this camera, update  positions of our target
    for color in self.circle_colorNames:
      if self.is_circle_visible[color] and color == "Yellow":
        self.update_circle_positions(color, yellow_center)

      if self.is_circle_visible[color] and color == "Blue":
        self.update_circle_positions(color, blue_center)

      if self.is_circle_visible[color] and color == "Green":
        self.update_circle_positions(color, green_center)

      if self.is_circle_visible[color] and color == "Red":
        self.update_circle_positions(color, red_center)


    target_center, box_center = self.predict_orange_centers2(self.cv_image2)
    # When the target can be detected from this camera, update  positions of our target
    if self.is_target_visible:
      #print("target visible")
      self.update_target_positions(target_center)

    if self.is_box_visible:
      self.update_box_positions(box_center)


    # --- for publishing --- #
    self.y_center = Float64MultiArray()
    self.y_center.data = yellow_center
    self.b_center = Float64MultiArray()
    self.b_center.data = blue_center
    self.g_center = Float64MultiArray()
    self.g_center.data = green_center
    self.r_center = Float64MultiArray()
    self.r_center.data = red_center
    self.target_sphere_center = Float64MultiArray()
    self.target_sphere_center.data = target_center
    self.orange_box_center = Float64MultiArray()
    self.orange_box_center.data = box_center
    self.box_center_pub2.publish(self.orange_box_center)

    # Publish the results
    try: 
      self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
      #publish joint centers with coordinates (x,z) taken from image 2
      self.joint_centers_yellow_pub2.publish(self.y_center)
      self.joint_centers_blue_pub2.publish(self.b_center)
      self.joint_centers_green_pub2.publish(self.g_center)
      self.joint_centers_red_pub2.publish(self.r_center)
      self.target_center_pub2.publish(self.target_sphere_center)
      self.box_center_pub2.publish(self.orange_box_center)

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
