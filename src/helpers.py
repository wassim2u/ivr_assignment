#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
import tensorflow as tf
import os
from numpy import sin, cos
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError


def get_end_effector(cv_image):
    lower_red = np.array([0,0,100])
    upper_red = np.array([50,50,255])
    cx_r, cy_r = get_joint_coordinates(cv_image, lower_red, upper_red)

    return cx_r, cy_r

def convert_pixel_to_metres(cv_image):
  lower_blue = np.array([100,0,0])
  upper_blue = np.array([255,50,50])
  cx_b, cy_b = get_joint_coordinates(cv_image, lower_blue, upper_blue)
      
  lower_yellow = np.array([0,100,100])
  upper_yellow = np.array([50,255,255])
  cx_y, cy_y = get_joint_coordinates(cv_image, lower_yellow, upper_yellow)
  #get pixel to metres ratio
  b = np.array([cx_b, cy_b])
  y = np.array([cx_y, cy_y])
  s = np.sum((b-y)**2)
  dist = np.sqrt(s)
  return 2.5/dist

def get_center(cv_image):
  lower_yellow = np.array([0,100,100])
  upper_yellow = np.array([50,255,255])
  cx_y, cy_y = get_joint_coordinates(cv_image, lower_yellow, upper_yellow)
  return cx_y, cy_y

def get_joint_coordinates(cv_image, lower, upper):
  im = threshold(cv_image, lower, upper)
  M = cv2.moments(im)
  cx = int(M['m10']/M['m00'])
  cy = int(M['m01']/M['m00'])
  return cx, cy

def threshold(cv_image, lower_range, upper_range):
    #create mask
    mask = cv2.inRange(cv_image, lower_range, upper_range)
    return mask

def get_predictions(img):
    IMG_SIZE = 32 #Size changed to match Tensorflow model shape requirements
    img = cv2.resize(img, dsize=(IMG_SIZE, IMG_SIZE))
    interpreter = tf.lite.Interpreter(model_path=os.path.expanduser("~/catkin_ws/src/ivr_assignment/target_model.tflite"))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # add N dim and change to float
    img = np.asarray(img, dtype=np.float32)
    input_data = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = np.squeeze(output_data)
    #Output
    return output_data

def metres_to_pixels(metres, cv_image):
  ratio = 1 /convert_pixel_to_metres(cv_image)
  return ratio*metres

def change_frame(y, z, center_y, center_z):
  point = np.array([[0.], [y], [z], [1.]])
  transformation = np.array([[1.,0.,0.,0.],[0., np.cos(np.pi), -np.sin(np.pi), -center_y*np.cos(np.pi)],[np.sin(np.pi), np.cos(np.pi), -center_y*np.sin(np.pi)-center_z*np.cos(np.pi)],[0.,0.,0.,1.]])
  return np.dot(transformation, point)



