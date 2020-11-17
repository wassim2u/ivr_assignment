from PIL import Image, ImageDraw
import random
import os
import math

def generate_squares(num, dim1, dim2, folder):

    for i in range(num):

        x_top = random.randint(dim1/4,3*dim2/4)
        y_left = random.randint(dim1/4,3*dim2/4)

        #height of rectangle
        height = random.randint(dim1/16,dim1/2)

        #width of rectangle
        width = random.randint(dim1/16,dim2/2)

        image = Image.new('1', (dim1,dim2), 'white')

        draw = ImageDraw.Draw(image)

        draw.rectangle((x_top, y_left, x_top+height, y_left+width), fill='black', outline='black')

        basename = str(i)+"rectangle.png"
        path = os.path.join(folder, basename)
        image.save(path)

def generate_circle(num, dim1, dim2, folder):
    for i in range(num):
        x_top = random.randint(dim1/4,3*dim2/4)
        y_left = random.randint(dim1/4,3*dim2/4)

        first_var = random.randint(dim1/8,dim1/2)
        #Randomly decide if the first value computed is the height or the width
        #If 1-> first var is height
        #If 0-> first var is width
        is_height = random.randint(0,1)
        height = None
        width = None
        second_var = None

        #Determine second variable
        #If the first variable is rather small, force second variable to be larger s.t. the ellipse does not come out as a square
        #If first_var is large enough, allow it to be just as large as first_var
        if (first_var < dim1/4):
            second_var = random.randint(math.ceil(first_var+(dim1/7)), 3*dim2/4)
        else :
            second_var = random.randint(first_var, 3*dim2/4)

        if (is_height == 1):
            height = first_var
            width = second_var
        else:
            height = second_var
            width = first_var
        
        image = Image.new('1', (dim1, dim2), 'white')

        draw = ImageDraw.Draw(image)

        basename = str(i)+"circle.png"
        path = os.path.join(folder, basename)

        draw.ellipse((x_top, y_left, x_top+height, y_left+width), fill='black')
        image.save(path)