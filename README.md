# IVR_Assignment - Annabel Jakob and Wassim Jabrane


**When you clone the repository, please make sure that you rename the folder ivr_assignment-main to ivr_assignment. Otherwise, the code won't run because the model uses the path ivr_assignment.**


## Packages Installed 
Sympy and Tensorflow (version 2.3.1, currently works with python of version 3.7) are additional packages that we installed for our projects, with the former allowing us to compute symbolic mathematics and the latter used for target detection.
Other packages installed include numpy and OpenCV.

To install sympy: pip3 install sympy

To install tensorflow: pip3 install --upgrade tensorflow

Installing mpy might also be needed: pip install empy

Installing catkin_pkg might also be needed : pip install catkin_pkg

Installing rospkg might also be needed : pip install rospkg

## File Descriptions
image1.py and image2.py handle the image processing part of the project, mainly target detection (Task 2.2) and pinpointing the center of our object.
controller.py takes the coordinates published by the two nodes to reconstruct a 3d representation of the coordinates and use them to generate the FK matrix and control the robot (Task 2.1, Task 3.1, Task 3.2, Task 4.2).
There is also a helper module called helpers.py that we created with utilities to help us solve our tasks.
Other scripts and folders (some not needed to run the project successfully) include sample images of shapes to train our target detection classifer for example.
## To run Task 4.2:
To run Task 4.2 successfully, all python files image1.py, image2.py, and controller.py must be rosrun. However, for task 4.2, before running the controller.py, run the script reset_angles everytime it launches. It is a bash script that resets all joints to 0 by publishing to the relevant topics that move the robot to its original state. This is done because we pass on joint values returned by the function closed_loop_control (Task 3.2) and null_space_control (Task 4.2) to the next iteration to have more accurate results. The results are saved in self.q_task4_2 defined in the __init__ function, with an initial state of 0.0 for all joints

Not reseting may not provide accurate results. The 

--Run the following commands initially, need to be done only once: 

catkin_make

source devel/setup.bash

rosrun ivr_assignment image1.py

rosrun ivr_assignment image2.py

--For the following, After every Keyboard interrupt (Ctrl+C) to stop the program, reset_angles must be 
ran everytime before doing rosrun controller.py


source reset_angles

rosrun ivr_assignment controller.py

## Folder Descriptions
This repository contains folders called test_data and training_data. training data contains two folders called circle and not_circle. These contain roughly 2000 binary 32-by-32-bit images on which the classifier for task 2.2 was trained. test_data contains a small number of test images for the classifier, both generated artificially and taken directly from the robot.
Additionally, the repository contains a folder called images. This folder contains graphs for tasks 2.1 through 4.2, the most import of which are included in the report.
