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
import message_filters



class images_sync:

  # Defines publisher and subscriber
  def __init__(self):
    # initialize the node named images_sync
    rospy.init_node('images_sync', anonymous=True)

    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()

    # self.joint2_sub = message_filters.Subscriber("/robot/joint2_position_controller/command", Float64)
    # self.joint3_sub = message_filters.Subscriber("/robot/joint3_position_controller/command", Float64)
    # self.joint4_sub = message_filters.Subscriber("/robot/joint4_position_controller/command", Float64)

    self.y_sub1 = message_filters.Subscriber("/image1/joint_centers/yellow", Float64MultiArray)
    self.b_sub1 = message_filters.Subscriber("/image1/joint_centers/blue", Float64MultiArray)
    self.g_sub1 = message_filters.Subscriber("/image1/joint_centers/green", Float64MultiArray)
    self.r_sub1 = message_filters.Subscriber("/image1/joint_centers/red", Float64MultiArray)
    self.target_sub1 = message_filters.Subscriber("/image1/target_center", Float64MultiArray)

    self.y_sub2 = message_filters.Subscriber("/image2/joint_centers/yellow", Float64MultiArray)
    self.b_sub2 = message_filters.Subscriber("/image2/joint_centers/blue", Float64MultiArray)
    self.g_sub2 = message_filters.Subscriber("/image2/joint_centers/green", Float64MultiArray)
    self.r_sub2 = message_filters.Subscriber("/image2/joint_centers/red", Float64MultiArray)
    self.target_sub2 = message_filters.Subscriber("/image2/target_center", Float64MultiArray)

    # synchronise incoming channels using the timestamps contained in their headers
    # slop defines the delay in seconds with which messages are synchronized
    ts = message_filters.ApproximateTimeSynchronizer([
                                                      self.y_sub1, self.b_sub1, self.g_sub1, self.r_sub1, self.target_sub1,
                                                      self.y_sub2, self.b_sub2, self.g_sub2, self.r_sub2, self.target_sub2
                                                    ],
                                                     queue_size=10,slop= 0.1, allow_headerless=True)
    ts.registerCallback(self.callback)

  #Note: Image 2 - xz plane; Image 1 - yz plane
  def create_new_3d_coordinates_from_data(self,y1,b1,g1,r1,target1,y2,b2,g2,r2,target2):
    self.yellow_center1 = np.asarray(y1.data)
    self.blue_center1 = np.asarray(b1.data)
    self.green_center1 = np.asarray(g1.data)
    self.red_center1 = np.asarray(r1.data)
    self.target_center1 = np.asarray(target1.data)
    print(self.target_center1)

    self.yellow_center2 = np.asarray(y2.data)
    self.blue_center2 = np.asarray(b2.data)
    self.green_center2 = np.asarray(g2.data)
    self.red_center2 = np.asarray(r2.data)
    self.target_center2 = np.asarray(target2.data)
    print(self.target_center2)




    #blue and yellow should always on the same x and y-axis:
    self.blue_center1[0] = self.yellow_center1[0]
    self.blue_center2[0] = self.yellow_center2[0]

    distance_blue_green_link1 = np.sqrt(np.sum((self.green_center1 - self.blue_center1) ** 2))
    distance_blue_green_link2 = np.sqrt(np.sum((self.green_center2 - self.blue_center2) ** 2))
    if distance_blue_green_link1 >= distance_blue_green_link2:
      self.z_center =self.yellow_center1[1]
      self.z_blue = self.blue_center1[1]
      self.z_green = self.green_center1[1]
      self.z_red = self.red_center1[1]
    else:
      self.z_center = self.yellow_center2[1]
      self.z_blue = self.blue_center2[1]
      self.z_green = self.green_center2[1]
      self.z_red = self.red_center2[1]
      # if distance_blue_green_link2 > 92:
      #   self.z_green = self.z_green - self.z_green*0.015
      #   self.green_center2[0] = self.green_center2[0] - self.green_center2[0]*0.015

    print(np.sqrt(np.sum((np.array([self.green_center1]) - np.array([self.blue_center1])) ** 2)))
    print(np.sqrt(np.sum((np.array([self.green_center2]) - np.array([self.blue_center2])) ** 2)))



    self.yellow_3d = np.array([self.yellow_center2[0], self.yellow_center1[0], self.z_center])
    self.blue_3d = np.array([self.blue_center2[0], self.blue_center1[0], self.z_blue])
    self.green_3d = np.array([self.green_center2[0], self.green_center1[0], self.z_green])
    self.red_3d = np.array([self.red_center2[0], self.red_center1[0], self.z_red])


  #Calculate the conversion from pixel to meter,
  # using the joints green and red and the length of the link (3 m) between them.
  def pixel2meter(self):
    # find the euclidean distance between two circles

    yellow_blue_link_length = np.sqrt(np.sum((self.yellow_3d - self.blue_3d) ** 2))

    return 2.5/yellow_blue_link_length

  # TODO: Change axis for other centers as well.
  def changeAxis(self):
    new_blue_3d = np.array([0, 0, 0])
    new_green_3d = self.blue_3d - self.green_3d
    ratio = self.pixel2meter()
    print("Ratio:" + str(ratio))
    self.blue_3d = new_blue_3d * 0.0389
    self.green_3d = new_green_3d * 0.0389
    print("Values changed to meters:")
    print("Blue:"+ str(self.blue_3d))
    print("Green" + str(self.green_3d))

  #TODO: Deal with prespective issue. Angles returned may be severely inaccurate because of the camera's perspective
  def estimate_joint_angles_brute_force_green_blue(self):
    xs = np.linspace(-np.pi / 2, np.pi / 2, 100)
    ys = np.linspace(-np.pi / 2, np.pi / 2, 100)
    print("GOAL: " + str(self.green_3d))
    original_green_3d = self.blue_3d.copy()
    original_green_3d[2] = original_green_3d[2] + 3.5
    print("ORIGNAL: " + str(original_green_3d))


    # self.green_3d[2] = self.green_center1[1]
    for x in xs:
      angle_x = x
      x=-x
      for y in ys:
        y= -y

        R_y_then_x = np.array([
          [np.cos(y), 0, np.sin(y)],
          [np.sin(x) * np.sin(y), np.cos(x), -np.sin(x) * np.cos(y)],
          [-np.cos(x) * np.sin(y), np.sin(x), np.cos(x) * np.cos(y)]
        ])

        #TODO: SEE IF X THEN Y OR OPPOSITE
        R_x_then_y = np.array([
          [np.cos(y), np.sin(y) * np.sin(x), np.sin(y) * np.cos(x)],
          [0, np.cos(x), -np.sin(x)],
          [-np.sin(y), np.cos(y) * np.sin(x), np.cos(y) * np.cos(x)]
        ])

        expected_green_yx = R_y_then_x.dot(original_green_3d)
        expected_green_xy = R_x_then_y.dot(original_green_3d)

        # if y > -0.1 and y < 0.1 and x >= 1.5 and x < 1.6:
        #   print(x)
        #   print(expected_green_yx)
        #   print(expected_green_xy)
        #   print(self.green_3d)

        if np.all(abs(expected_green_yx - self.green_3d) < 0.1):
          return R_y_then_x, -x, -y
        if np.all(abs(expected_green_xy - self.green_3d) < 0.1):

          return R_x_then_y, -x, -y

    return None, None, None


  # Find angles that the vector makes with the x,y,and z axes
  def find_direction_angles_green_blue(self):
    x_distance = (self.blue_center2[0] - self.green_center2[0])
    y_distance = (self.blue_center1[0] - self.green_center1[0])
    z_distance = (self.z_blue - self.z_green)
    print("DISTANCE:" + str(np.array([x_distance, y_distance, z_distance])))

    ax = np.arccos(x_distance / (np.sqrt(pow(x_distance, 2) + pow(y_distance, 2) + pow(z_distance, 2))))
    ay = np.arccos(y_distance / (np.sqrt(pow(x_distance, 2) + pow(z_distance, 2) + pow(y_distance, 2))))
    az = np.arccos(z_distance / (np.sqrt(pow(x_distance, 2) + pow(y_distance, 2) + pow(z_distance, 2))))
    result = (self.blue_3d.dot(self.green_3d)) / (
            np.sqrt(self.blue_3d.dot(self.blue_3d.T)) * np.sqrt(self.green_3d.dot(self.green_3d.T)))
    angle_between_vectors = np.arccos(result)
    print(angle_between_vectors)
    print(np.arctan2(x_distance, z_distance))
    # print(np.arctan2(np.linalg.norm(np.cross(center_coords,blue_coords)),np.dot(center_coords,blue_coords)))
    print(np.array([ax - np.pi / 2, ay - np.pi / 2, az]))
    # print(angle_between_vectors)

  #Testing purposes
  def print_rotation_results_green_blue(self):
    angle = -1.0

    Rx = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    Ry = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    # print(Rx.dot(green_3d_unit) * np.linalg.norm(green_3d) )
    # print(Rx)
    # print(green_3d)
    original_green_3d = self.blue_3d.copy()
    original_green_3d[2] = original_green_3d[2] + 3.5
    print("Inital Green Point Coordinates: " + str(original_green_3d))
    print("Rotation around x-axis")
    print(Rx @ original_green_3d)
    print("Rotation around y-axis")
    print(Ry @ original_green_3d)
    print("Rotation around x-axis and y-axis")
    print(Rx @ Ry @ original_green_3d)

  # Recieve data from both image_processing nodes corresponding to both cameras, process it, and publish
  # TODO: Alot needs to be changed. The processing part is incomplete. Dont forget to publish at the end too.
  def callback(self,y1,b1,g1,r1,target1,y2,b2,g2,r2,target2):
    print("--------------------------------------------")

    self.create_new_3d_coordinates_from_data(y1,b1,g1,r1,target1,y2,b2,g2,r2,target2)
    print("Green Center: Img Coordinates" + str(self.green_3d) )
    print("Blue Center: Img Coordinates" + str(self.blue_3d) )

    self.changeAxis()

    # print("Brute Force- ")
    # R, x, y = self.estimate_joint_angles_brute_force_green_blue()
    # print("RESULTS:")
    # print("Joint2: predicted x-rotation angle: " + str(x))
    # print("Joint3: predicted y-rotation angle:" + str(y))
    # print("Rotation Matrix" + str(R))
    # # print("Multiplying to Rotation Matrix: " + str(R@[0,0,3.5]))
    #
    # print("Rotation")
    # self.print_rotation_results_green_blue()
    # self.find_direction_angles_green_blue()





# call the class
def main(args):
  ic = images_sync()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)