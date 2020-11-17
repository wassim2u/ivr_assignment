import cv2
import os
import shutil
#Utility file to invert some of the files

#Script file to generate inverted templates from existing images
sphere_original_path = '../spheres/original'
sphere_inverted_path = '../spheres/inverted'
sphere_rotated_path = '../spheres/rotated'

box_original_path = '../boxes/original'
box_inverted_path= '../spheres/inverted'
box_rotated_path=  '../boxes/rotated'

#Test folder paths to generate inverted ones to pass to the classifier model
test_original_path = '../tests/original'
test_inverted_path = '../tests/inverted'






def generate_inverted_pictures(initial_path,output_dr_path):
    for filename in os.listdir(initial_path):
        img= cv2.imread(os.path.join(initial_path,filename),0)
        output = cv2.bitwise_not(img)

        output_filename = os.path.splitext(filename)[0] + '_inverted.png'
        print(output_filename)

        cv2.imwrite(os.path.join(output_dr_path, output_filename), output)



#Create a new inverted folder for spheres and delete previous one
shutil.rmtree(sphere_inverted_path) #Delete previous inverted path file - MAKE SURE YOU DONT DELETE THE ORIGINAL
os.mkdir(sphere_inverted_path) #Create a new inverted folder under sphere

#Create a new inverted folder for boxes and delete previous one
shutil.rmtree(box_inverted_path) #Delete previous inverted folder - MAKE SURE YOU DONT DELETE THE ORIGINAL
os.mkdir(box_inverted_path) #Create a new inverted folder under box

#Create a new inverted folder for test images and delete previous one
shutil.rmtree(test_inverted_path) #Delete previous inverted folder - MAKE SURE YOU DONT DELETE THE ORIGINAL
os.mkdir(test_inverted_path) #Create a new inverted folder under box


#invert spheres from original and rotated folder. We will be using these inverted images for training
generate_inverted_pictures(sphere_original_path,sphere_inverted_path)
generate_inverted_pictures(sphere_rotated_path,sphere_inverted_path)

#invert boxes from original and rotated folder. We will be using these inverted imagess for trainig
generate_inverted_pictures(box_original_path,box_inverted_path)
generate_inverted_pictures(box_rotated_path,box_inverted_path)

#invert test images from original test folders - we will be using these for our classifiers as a test set.
generate_inverted_pictures(test_original_path,test_inverted_path)







