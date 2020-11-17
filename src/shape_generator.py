from PIL import Image, ImageDraw
import random
import os

def generate_squares(num, dim1, dim2, folder):

    for i in range(num):

        x_top = random.randint(dim1/4,3*dim2/4)
        y_left = random.randint(dim1/4,3*dim2/4)

        #height of rectangle
        height = random.randint(0,dim1/2)

        #width of rectangle
        width = random.randint(0,dim2/2)

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
        height = random.randint(0,dim1/2)
        width = random.randint(0,dim2/2)

        image = Image.new('1', (dim1, dim2), 'white')

        draw = ImageDraw.Draw(image)

        basename = str(i)+"circle.png"
        path = os.path.join(folder, basename)

        draw.ellipse((x_top, y_left, x_top+height, y_left+width), fill='black', outline='black')
        image.save(path)

generate_squares(1000, 32,32, "../small_data/not_circle")
generate_squares(20, 32,32, "../small_test_data")
generate_circle(1000, 32,32, "../small_data/circle")
generate_circle(20, 32,32, "../small_test_data")