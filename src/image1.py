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
from helpers import *



class image_converter_1:

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

    self.joint_centers_yellow_pub1 = rospy.Publisher("/image1/joint_centers/yellow", Float64MultiArray, queue_size=10)
    self.joint_centers_blue_pub1 = rospy.Publisher("/image1/joint_centers/blue", Float64MultiArray, queue_size=10)
    self.joint_centers_green_pub1 = rospy.Publisher("/image1/joint_centers/green", Float64MultiArray, queue_size=10)
    self.joint_centers_red_pub1 = rospy.Publisher("/image1/joint_centers/red", Float64MultiArray, queue_size=10)
    self.target_center_pub1 = rospy.Publisher("/image1/target_center", Float64MultiArray, queue_size=10)

    #These variables are used to keep track of target velocity to be used when approximating the next position of
    #target when it is not visible
    self.is_target_visible = True
    self.prev_time = np.array([rospy.get_time()], dtype='float64')
    self.target_velocity_y = 0.0
    self.previous_target_ypos = np.array([0.0, 0.0], dtype='float64')

  ##Code for task 4.1##
  def is_visible(self, m):
    return not(m==0)

    ###Functions to move joints 2-4 ###
  def move_joint2(self, t):
    return (np.pi/2)*np.sin((np.pi/15.0)*t)

  def move_joint3(self, t):
    return (np.pi/2)*np.sin((np.pi/18.0)*t)

  def move_joint4(self, t):
    return (np.pi/2)*np.sin((np.pi/20.0)*t)

  #Return a dictionary that contains binary images for each circle
  #Retrieve the image of a specific circle from the dictionary using their colour as key (eg. dictionary_name['Blue'])
  def detect_circles(self, img):
    #Turn RGB Image into HSV colour space
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Detect Red Circle
    red_mask = cv2.inRange(hsv_image, (0, 10, 20), (10, 255, 255))
    # Detect Blue Circle
    blue_mask = cv2.inRange(hsv_image, (90, 5, 10), (130, 255, 255))
    # Detect Green Circle
    green_mask = cv2.inRange(hsv_image, (35, 0, 0), (75, 255, 255))
    # Detect Yellow Circle
    yellow_mask = cv2.inRange(hsv_image, (28, 10, 20), (35, 255, 255))

    #cv2.imshow('Red Circle - Binary Image', red_mask)
    #cv2.imshow('Blue Circle - Binary Image', blue_mask)
    #cv2.imshow('Green Circle - Binary Image', green_mask)
    #cv2.imshow('Yellow Circle - Binary Image', yellow_mask)
    # cv2.waitKey(1)

    binary_images = {"Blue": blue_mask, "Green": green_mask, "Red": red_mask, "Yellow": yellow_mask}
    return binary_images

  # Find center of a specific circle. The image returned from camera1 is of plane yz.
  # TODO: Tackle cases of 0 area where circle is completely hidden
  def find_color_center(self ,mask_color):
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(mask_color, kernel, iterations=3)
    M = cv2.moments(dilated_mask)

    if (self.is_visible(M['m00'])):
      cy = int(M['m10'] / M['m00'])
      cz = int(M['m01'] / M['m00'])
      return np.array([cy, cz])
    
    return np.array([0.0, 0.0])

  #TODO: Solve edge case for thiss well when its completely hidden

  # Find the outline of a binary image of a specific circle, and use minEnclosingCircle to predict the center of circle
  # that is partly hidden behind an object.
  def predict_circle_center(self, mask):
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=4)
    #check whether circle is visible by checking its area:
    M = cv2.moments(dilated_mask)
    area = M['m00']
    if (area<0.0001):
      #TODO: Tackle issue when its completely hidden
      pass
    #Find outline of the shape of the masked circle
    contours, hierarchy = cv2.findContours(dilated_mask, 1, 2)
    contour_poly = cv2.approxPolyDP(curve=contours[0], epsilon=0.1, closed=True)
    #Using the outline, draw a circle that encloses the partial segment of the circle that is hidden

    center, radius = cv2.minEnclosingCircle(contour_poly)
    return np.array([int(center[0]), int(center[1])]) ,radius

    #TODO: Deal with occlusion case
  def detect_sphere_target(self, img):
    # Turn RGB Image into HSV colour space
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Detect Orange Targets
    masks = cv2.inRange(hsv_image, (10, 0, 0), (24, 255, 255))
    kernel = np.ones((3, 3), np.uint8)
    opening_mask = cv2.morphologyEx(masks,cv2.MORPH_OPEN ,kernel)
    # cv2.imshow('no',opening_mask)

    # check whether circle is visible by checking its area
    M = cv2.moments(opening_mask)
    # Match template
    center = self.match_sphere_template(img, opening_mask)
    return center

  # Matches binary image with sphere template. Returns the center of the matched shape (which should be sphere).
  def match_sphere_template(self, img, opening_mask):
    # Match template
    contours, hierarchy = cv2.findContours(opening_mask, 1, 1)
    sphere_contour = contours[0]
    circularities = []
    areas=[]
    for c in contours:
        area = cv2.contourArea(c)
        areas.append(area)
        perimeter = cv2.arcLength(c, closed=True)
        circularity = 4 * np.pi * area / (perimeter ** 2)
        print(circularity)
        circularities.append(circularity)


    #If the object circularity sis less than a certain threshold( meaning we identified it as a box) and there's only
    #one shape present, then the sphere is completely hidden.
    if len(circularities) == 1 and circularities[0]<0.79:
        #Predict target position using previous information
        self.is_target_detected = False
        return self.approximate_target_y_position()
    #Compare circularities abd areas to identify which shape we are looking at.
    if circularities[0] > circularities[1]:
        if circularities[0] > 0.77:
            sphere_contour = contours[0]
        elif abs(self.previous_box_circularity - circularities[0]) <0.06 and areas[0]>areas[1]:
            self.previous_box_circularity = circularities[0]
            sphere_contour = contours[1]
        else:
            self.previous_box_circularity = circularities[1]
            sphere_contour = contours[0]
    elif circularities[1] > circularities[0]:
        if circularities[1] > 0.77:
            sphere_contour = contours[1]
        elif abs(self.previous_box_circularity - circularities[1]) <0.06 and (areas[1]>areas[0]):
            self.previous_box_circularity = circularities[1]
            sphere_contour = contours[0]
        else:
            self.previous_box_circularity = circularities[0]
            sphere_contour = contours[1]

    #Target shape has been detected
    self.is_target_detected = True

    contour_poly = cv2.approxPolyDP(curve=sphere_contour, epsilon=0.1, closed=True)
    # Using the outline, draw a circle that encloses the partial segment of the circle that is hidden
    center, radius = cv2.minEnclosingCircle(contour_poly)
    return np.array([int(center[0]), int(center[1])]) ,radius

    #TODO: Deal with occlusion case
  def detect_sphere_target(self, img):
    # Turn RGB Image into HSV colour space
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Detect Orange Targets
    masks = cv2.inRange(hsv_image, (10, 0, 0), (24, 255, 255))
    kernel = np.ones((3, 3), np.uint8)
    opening_mask = cv2.morphologyEx(masks,cv2.MORPH_OPEN ,kernel)
    #
    center = self.predict_sphere_center(img, opening_mask)
    return center

  # Returns the center of the matched shape with the help of classifer (which should be sphere).
  def predict_sphere_center(self, img, opening_mask):
    #Find outlines of our shapes inour binary images
    contours, hierarchy = cv2.findContours(opening_mask, 1, 1)
    sphere_contour = contours[0]
    sphere_index =0
    box_index = 1
    self.is_target_visible= False
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

    if (not self.is_target_visible):
      #TODO: PRedict trajectory?
      pass

    contour_poly = cv2.approxPolyDP(curve=sphere_contour, epsilon=0.1, closed=True)
    # Using the outline, draw a circle that encloses the partial segment of the circle that is hidden
    center, radius = cv2.minEnclosingCircle(contour_poly)
    #Draw outline of shape predicted to be a sphere to validate result
    self.draw_circle_prediction(img,center,radius)
    return center


  # Draws the shape on the image. Call when needed for visualisation.
  def draw_boundary(self, img, mask):
    contours, hierarchy = cv2.findContours(mask, 1, 2)
    # Draw the outline on the binary image
    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
    cv2.imshow('draw contours', img)
    cv2.waitKey(1)

  # Draws a circle on the image. Call when needed for visualisation and to check result.
  def draw_circle_prediction(self, img, center, radius):
    new_img = img.copy()
    color = [255, 23, 0]
    line_thickness = 2
    cv2.circle(new_img, (int(center[0]), int(center[1])), int(radius), color, line_thickness)
    cv2.imshow('Image with predicted shape of circle', new_img)
    cv2.waitKey(1)



  def update_target_position_and_velocity(self,current_target_ypos):
    #Get the change in time
    current_time = rospy.get_time()
    dt = current_time - self.init_time
    self.init_time = current_time
    #Get displacement in y-direction and store the new target position
    displacement=  current_target_ypos - self.previous_target_ypos
    self.previous_target_ypos = current_target_ypos
    #Calculate the velocity in the y-direction and store it
    current_velocity = displacement/ dt
    self.target_velocity_y = current_velocity

  #When the target is not visible or cant be detected, calculate the predicted position of y using information
  #on its previous movement.
  #Note: When target is hidden in one camera, we can get the z position from the other one.
  def approximate_target_y_position(self):
    # Get the change in time
    current_time = rospy.get_time()
    dt = current_time - self.init_time
    predicted_y = self.target_velocity_y * dt + self.previous_target_ypos
    return predicted_y



  # Recieve data from camera 1, process it, and publish
  def callback1(self,data):
    # Receive the image
    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    # Uncomment if you want to save the image
    # cv2.imwrite('image1_copy.png', self.cv_image1)
    #self.get_joint_positions()

    cv2.imshow('window1', self.cv_image1)
    cv2.waitKey(1)

    ##Task 2##

    masked_circles_image1 = self.detect_circles(self.cv_image1)

    yellow_center, y_radius = self.predict_circle_center(masked_circles_image1['Yellow'])
    blue_center, bl_radius= self.predict_circle_center(masked_circles_image1['Blue'])
    green_center, gr_radius= self.predict_circle_center(masked_circles_image1['Green'])
    red_center, r_radius = self.predict_circle_center(masked_circles_image1['Red'])

    target_center= self.detect_sphere_target(self.cv_image1)
    #When the target can be detected from this camera, update the velocity and y_position of our target
    if self.is_target_visible:
      self.update_target_position_and_velocity(target_center[0])



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



    # update to current time
    self.time = rospy.get_time()

    self.joint2_angle = Float64()
    self.joint3_angle = Float64()
    self.joint4_angle = Float64()
    # Publish the results
    try:
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
      #publish new joint angles
      # self.joint2_pub.publish(self.joint2_angle)
      # self.joint3_pub.publish(self.joint3_angle)
      # self.joint4_pub.publish(self.joint4_angle)

      #publish joint centers with coordinates (y,z) taken from image 1
      self.joint_centers_yellow_pub1.publish(self.y_center)
      self.joint_centers_blue_pub1.publish(self.b_center)
      self.joint_centers_green_pub1.publish(self.g_center)
      self.joint_centers_red_pub1.publish(self.r_center)
      self.target_center_pub1.publish(self.target_sphere_center)


    except CvBridgeError as e:
      print(e)

# call the class
def main(args):
  ic = image_converter_1()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)


