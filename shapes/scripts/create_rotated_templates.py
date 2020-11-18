import cv2
import os
import shutil
import numpy as np
#Script file to generate rotated templates from existing images
sphere_folder_path = '../spheres/original'
box_folder_path = '../boxes/original'
output_sphere_folder_path = '../spheres/rotated'
output_box_folder_path=  '../boxes/rotated'

#Different angles to rotate
angles = np.arange(1,360,60)


def apply_rotation(image, angle):
    num_rows, num_cols = image.shape
    matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), angle, 1)
    output = cv2.warpAffine(image, matrix, (num_cols, num_rows))
    return output


#Rotate spheres and save them in rotated folder
#initial path is where the original pictures reside
def generate_rotated_pictures(initial_path,output_dr_path):
    for filename in os.listdir(initial_path):
        img= cv2.imread(os.path.join(initial_path,filename),0)
        for angle in angles:
            output = apply_rotation(img,angle)
            output_filename = os.path.splitext(filename)[0] + '_angle_' +str(angle) + '.png'
            print(output_filename)

            cv2.imwrite(os.path.join(output_dr_path, output_filename), output)

#Create a new rotated folder for spheres and delete previous one
shutil.rmtree(output_sphere_folder_path) #Delete previous rotated path file - MAKE SURE YOU DONT DELETE THE ORIGINAL
os.mkdir(output_sphere_folder_path) #Create a new rotated folder under sphere
#Create a new rotated folder for boxes and delete previous one
shutil.rmtree(output_box_folder_path) #Delete previous rotated folder - MAKE SURE YOU DONT DELETE THE ORIGINAL
os.mkdir(output_box_folder_path) #Create a new rotated folder under box


#rotate spheres
generate_rotated_pictures(sphere_folder_path,output_sphere_folder_path)
#rotate boxes
generate_rotated_pictures(box_folder_path,output_box_folder_path)

