import cv2
#Utility file to invert some of the files
sphere_template_1 = cv2.imread('../shapes/sphere-template.png', 0)
sphere_template_2 = cv2.imread('../shapes/occluded-sphere.png', 0)
sphere_template_3 = cv2.imread('../shapes/sphere-template-full.png', 0)
sphere_template_4 = cv2.imread('../shapes/sphere-template-half-hidden.png', 0)
sphere_template_5 = cv2.imread('../shapes/sphere-template-hidden-moon.png', 0)

box_template = cv2.imread('../shapes/box-template.png', 0)

spheres = [sphere_template_1,sphere_template_2,sphere_template_3,sphere_template_4,sphere_template_5]
#Invert Box
box_inverted = cv2.bitwise_not(box_template)
cv2.imwrite('box_inverted.png', box_inverted)
#invert spheres
for i in range(0,len(spheres)):
    sphere_inverted = cv2.bitwise_not(spheres[i])
    cv2.imshow('what',sphere_inverted)
    cv2.waitKey(0)
    filename = 'sphere_template_inverted_' + str(i) + '.png'
    cv2.imwrite(filename,sphere_inverted)




