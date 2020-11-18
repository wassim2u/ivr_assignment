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


from numpy import cos
from numpy import sin

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

    #initialize trajectory error
    self.error = np.array([0.0,0.0], dtype='float64') 
    self.init_time = rospy.get_time()
    self.target_sub2 = message_filters.Subscriber("/image2/target_center", Float64MultiArray)

    # synchronise incoming channels using the timestamps contained in their headers
    # slop defines the delay in seconds with which messages are synchronized
    ts = message_filters.ApproximateTimeSynchronizer([
                                                      self.y_sub1, self.b_sub1, self.g_sub1, self.r_sub1, self.target_sub1,
                                                      self.y_sub2, self.b_sub2, self.g_sub2, self.r_sub2, self.target_sub2
                                                    ],
                                                     queue_size=10,slop= 0.1, allow_headerless=True)
    ts.registerCallback(self.callback)


    self.previous_green = np.array([0,0,0])
    self.initial = True


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
    print(self.red_3d)

  #Calculate the conversion from pixel to meter,
  # using the joints green and red and the length of the link (3 m) between them.
  def pixel2meter(self):
    # find the euclidean distance between two circles

    yellow_blue_link_length = np.sqrt(np.sum((self.yellow_3d - self.blue_3d) ** 2))

    return 2.5/yellow_blue_link_length

  # TODO: Change axis for other centers as well.
  def changeAxis(self):
    new_yellow_3d = np.array([0,0,0])
    new_blue_3d = self.yellow_3d - self.blue_3d
    new_green_3d = self.yellow_3d - self.green_3d
    ratio = self.pixel2meter()
    print("Ratio:" + str(ratio))
    self.yellow_3d = new_green_3d
    self.blue_3d = new_blue_3d * 0.0389
    self.green_3d = new_green_3d * 0.0389
    print("Values changed to meters:")
    print("Blue:"+ str(self.blue_3d))
    print("Green" + str(self.green_3d))
    # r =np.array([(cos(a+b+c)+cos(a-b-c))/2,	-sin(b+c),	 (sin(a+b+c)+sin(a-b-c))/,	 (5*sin(a+b+c)+5*sin(a-b-c))/4
    #     (sin(a+b+c)-sin(a-b-c))/2	 cos(b+c)	(-cos(a+b+c)+cos(a-b-c))/2	(-5*cos(a+b+c)+5*cos(a-b-c))/4
    #               -sin(a)	        0	                    cos(a)	                (5*cos(a)+6)/2
    #                     0	        0	                         0	                             1
    #
    # ]
  def forward_kinematics(self):
    pass



<<<<<<< HEAD
  def forward_kinematics(self):
    # a03 =np.array([(cos(a+b+c)+cos(a-b-c))/2,	-sin(b+c),	 (sin(a+b+c)+sin(a-b-c))/,	 (5*sin(a+b+c)+5*sin(a-b-c))/4
    #     (sin(a+b+c)-sin(a-b-c))/2	 cos(b+c)	(-cos(a+b+c)+cos(a-b-c))/2	(-5*cos(a+b+c)+5*cos(a-b-c))/4
    #               -sin(a)	        0	                    cos(a)	                (5*cos(a)+6)/2
    #                     0	        0	                         0	                             1
    #
    # ]
    pass





  def get_jacobian(self, x, y, z, w):
    xv = (float) (7*sin(y+z)+6*sin(w+y+z)-6*sin(w-y+z)+7*sin(y-z))/4
    zv = (float) (20*cos(x)+14*cos(x+z)+12*cos(w+x+z)-12*cos(w-x+z)+7*cos(x+y+z)+6*cos(w+x+y+z)+6*cos(w-x+y+z)+7*cos(x-y+z)+6*cos(w+x-y+z)+6*cos(w-x-y+z)-14*cos(x-z)+7*cos(x+y-z)+7*cos(x-y-z))/8
    yv = (float) (20*sin(x)+14*sin(x+z)+12*sin(w+x+z)+12*sin(w-x+z)+7*sin(x+y+z)+6*sin(w+x+y+z)-6*sin(w-x+y+z)+7*sin(x-y+z)+6*sin(w+x-y+z)-6*sin(w-x-y+z)-14*sin(x-z)+7*sin(x+y-z)+7*sin(x-y-z))/8
    return xv,yv,zv 

  def get_inverse_jacobian(self, jacobian):
    return np.linalg.pinv(jacobian)

  #Note that the arguments here must be numpy arrays.
  def compute_tracking_error(self, desired, actual):
    err = desired-actual
    current_time = rospy.get_time()
    # current_trajectory = np.array([image1])

  def closed_loop_control(self, theta1, theta2, theta3, theta4):
    pass

=======
>>>>>>> b140d5e608708f198cb0d734e7f520ce1889009d

  # Recieve data from both image_processing nodes corresponding to both cameras, process it, and publish
  # TODO: Alot needs to be changed. The processing part is incomplete. Dont forget to publish at the end too.
  def callback(self,y1,b1,g1,r1,target1,y2,b2,g2,r2,target2):
    print("--------------------------------------------")

    self.create_new_3d_coordinates_from_data(y1,b1,g1,r1,target1,y2,b2,g2,r2,target2)
    print("Green Center: Img Coordinates" + str(self.green_3d) )
    print("Blue Center: Img Coordinates" + str(self.blue_3d) )

    self.changeAxis()


   #a01:
    # np.array([
    #   [0,0,1,0],
    #   [1,0,0,0],
    #   [0,1,0,5/2],
    #   [0,0,0,1]
    # ])

    # #a12:
    #
    # np.array([
    #   [-sin(a),0,cos(a),0],
    #   [cos(a),0,sin(a),0],
    #   [0,1,0,0],
    #   [0,0,0,1]
    # ])

    #a02:
    # 0	cos(a)	-sin(a)	5*cos(a)/2
    # 0	sin(a)	 cos(a)	5*sin(a)/2
    # 1	     0	      0	         0
    # 0	     0	      0	         1



    print("Joint2")
    print(np.arcsin((2/5)*(self.green_3d[0])))
    print(np.arccos((-2/5)*(self.green_3d[2]-0.32-(2/5*2/7))) - np.pi/2)
    print("Joint 3")


# (cos(a+b)+cos(a-b))/2	-sin(b)	 (sin(a+b)+sin(a-b))/2	 (5*sin(a+b)+5*sin(a-b))/4
# (sin(a+b)-sin(a-b))/2	 cos(b)	(-cos(a+b)+cos(a-b))/2	(-5*cos(a+b)+5*cos(a-b))/4
#                sin(a)	      0	               -cos(a)	           (-5*cos(a)+7)/2
#                     0	      0	                     0	                         1
#



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