# IVR_Assignment - Annabel Jakob and Wassim Jabrane

## Packages Installed 
Sympy and Tensorflow (version 2.3.1) are additional packages that we installed for our projects, with the former allowing us to compute symbolic mathematics and the latter used for target detection.
Other packages installed include numpy and OpenCV.

# --Needs to change
To install the modules used in this project, use: 
pip freeze -r requirements.txt
# --Needs to change

## File Descriptions
image1.py and image2.py handle the image processing part of the project, mainly target detection (Task 2.2) and pinpointing the center of our object.
controller.py takes the coordinates published by the two nodes to reconstruct a 3d representation of the coordinates and use them to generate the FK matrix and control the robot (Task 2.1, Task 3.1, Task 3.2, Task 4.2).
There is also a helper module called helpers.py that we created with utilities to help us solve our tasks.
Other scripts and folders (some not needed to run the project successfully) include sample images of shapes to train our target detection classifer for example.
## To run Task 4.2:
To run Task 4.2 successfully, all python files image1.py, image2.py, and controller.py must be rosrun. However, for task 4.2, before running the controller.py, run the script reset_angles everytime it launches. It is a bash script that resets all joints to 0 by publishing to the relevant topics that move the robot to its original state. This is done because we pass on joint values returned by the function closed_loop_control (Task 3.2) and null_space_control (Task 4.2) to the next iteration to have more accurate results. 
Not reseting may not provide accurate results. To change the initial state of the robot, the joint values 

--Run the following commands initially, need to be done only once: 

catkin_make

source devel/setup.bash

rosrun ivr_assignment image1.py

rosrun ivr_assignment image2.py

--For the following, After every Keyboard interrupt (Ctrl+C) to stop the program, reset_angles must be 
ran everytime before doing rosrun controller.py


source reset_angles
rosrun ivr_assignment controller.py


