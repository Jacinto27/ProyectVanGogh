import os
import numpy as np
from PIL import Image

# define input and output directories
input_dir = 'Data/Vangogh/RootData'
output_dir = 'Data/NormImages'

# define target image size and RGB mode
target_size = (224, 224)
rgb_mode = 'RGB'

# loop through all files in the input directory
for filename in os.listdir(input_dir):

    img_path = os.path.join(input_dir, filename)
    img = Image.open(img_path).convert(rgb_mode)
    img = img.resize(target_size)

    # normalize the pixel values
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0

    # save the image to the output directory
    output_path = os.path.join(output_dir, filename)
    Image.fromarray(np.uint8(img_array * 255.0)).save(output_path)

#Query Image manipulation


# define input and output directories
input_dir = 'Data/QueryImages/Digital Art Files'
output_dir = 'Data/QueryImages/JpegImages'

# loop through all files in the input directory
for filename in os.listdir(input_dir):
    # Check if Png
    if filename.endswith('.png'):

        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path)
        new_filename = filename.replace('.png', '.jpg')
        output_path = os.path.join(output_dir, new_filename)
        img.convert('RGB').save(output_path, 'JPEG')


# define input and output directories
input_dir = 'Data/QueryImages/JpegImages'
output_dir = 'Data/QueryImages/NormQueryImages'

# define target image size and RGB mode
target_size = (224, 224)
rgb_mode = 'RGB'

# Loop through all files in the input directory
for filename in os.listdir(input_dir):
    # load the image and convert to RGB
    img_path = os.path.join(input_dir, filename)
    img = Image.open(img_path).convert(rgb_mode)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    output_path = os.path.join(output_dir, filename)
    Image.fromarray(np.uint8(img_array * 255.0)).save(output_path)



