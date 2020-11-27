#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
import tensorflow as tf
import os
import sympy as sp
from sympy import symbols, diff, sin, cos, Matrix, Eq, solveset, Interval
import math

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

####These functions define the frame transformations for forward kinematics######################################
def a_0_1(theta1):
  rot_z = Matrix([
    [sp.cos(theta1+(sp.pi/2)), (-1.0)*sp.sin(theta1+(sp.pi/2)), 0, 0],
    [sp.sin(theta1+(sp.pi/2)), sp.cos(theta1+(sp.pi/2)), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ])
  trans_z = Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 2.5],
    [0, 0, 0, 1]
  ])
  rot_x = Matrix([
    [1, 0, 0, 0],
    [0, sp.cos(sp.pi/2), (-1.0)*sp.sin(sp.pi/2), 0],
    [0, sp.sin(sp.pi/2), sp.cos(sp.pi/2), 0],
    [0, 0, 0, 1]
  ])
  mtx_0_1 = rot_z*trans_z*rot_x
  return mtx_0_1

def a_1_2(theta2):
  rot_z = Matrix([
    [sp.cos(theta2+(sp.pi/2)), (-1.0)*sp.sin(theta2+(sp.pi/2)), 0, 0],
    [sp.sin(theta2+(sp.pi/2)), sp.cos(theta2+(sp.pi/2)), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ])
  rot_x = Matrix([
    [1, 0, 0, 0],
    [0, sp.cos(sp.pi/2), (-1.0)*sp.sin(sp.pi/2), 0],
    [0, sp.sin(sp.pi/2), sp.cos(sp.pi/2), 0],
    [0, 0, 0, 1]
  ])
  # print(type(rot_z), type(rot_x))
  mtx_1_2 = rot_z*rot_x
  return mtx_1_2

def a_2_3(theta3):
  rot_z = Matrix([
    [sp.cos(theta3), (-1)*sp.sin(theta3), 0, 0],
    [sp.sin(theta3), sp.cos(theta3), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ])
  trans_x = Matrix([
    [1, 0, 0, 3.5],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ])
  rot_x = Matrix([
    [1, 0, 0, 0],
    [0, sp.cos(-sp.pi/2), (-1)*sp.sin(-sp.pi/2), 0],
    [0, sp.sin(-sp.pi/2), sp.cos(-sp.pi/2), 0],
    [0, 0, 0, 1]
  ])
  mtx_2_3 = rot_z*trans_x*rot_x
  return mtx_2_3

def a_3_4(theta4):
  rot_z = Matrix([
    [sp.cos(theta4), (-1)*sp.sin(theta4), 0, 0],
    [sp.sin(theta4), sp.cos(theta4), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ])
  trans_x = Matrix([
    [1, 0, 0, 3],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ])
  mtx_3_4 = rot_z*trans_x
  return mtx_3_4

##################################################################################################################

#Compute the entire homogeneous transformation matrix
def fk_matrix(theta1, theta2, theta3, theta4):
  mtx_0_1 = a_0_1(theta1)
  mtx_1_2 = a_1_2(theta2)
  mtx_2_3 = a_2_3(theta3)
  mtx_3_4 = a_3_4(theta4)
  fk = mtx_0_1*mtx_1_2*mtx_2_3*mtx_3_4
  np_fk = fk.col(3)[0:3]
  return np_fk

#Compute the homogeneous transformation matrix with the green joint as end effector
def fk_green(theta1, theta2, theta3):
  mtx_0_1 = a_0_1(theta1)
  mtx_1_2 = a_1_2(theta2)
  mtx_2_3 = a_2_3(theta3)
  fk = mtx_0_1*mtx_1_2*mtx_2_3
  np_fk = np.array(fk.col(3)).astype(np.float64)
  return np_fk[0:3]

####Methods for joint angle computation###########################################################################

def get_joint3_angles(joint4):
  #Get x-coordinate of the green joint
  x = joint4[0]

  if x > 3.5:
   x = 3.5
  elif x < -3.5:
    x = -3.5

  theta3 = np.arcsin(x/3.5)

  return theta3

def get_joint2_angles(blue_joint, green_joint):
  m2 = np.arctan2(green_joint[1]-blue_joint[1], green_joint[2]-blue_joint[2])
  return -m2

def get_joint4_angles(theta2, green_joint, end_effector):

  theta4 = -np.arctan2(end_effector[:,1]-green_joint[:,1], end_effector[:,2]-green_joint[:,2])-theta2-0.25 
  return theta4
####################################################################################################################

