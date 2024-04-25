

# Import the modules
import os
import shutil
import json

input_dir = "F:/ImageSet/openxl2_dataset"
generation_dir = "F:/ImageSet/openxl2_generation"

# input_dir = "F:/ImageSet/test"
# generation_dir = "F:/ImageSet/test_gen"

caption_dir = "F:/ImageSet/openxl2_slider/caption"
image_dir = "F:/ImageSet/openxl2_slider/image"


for subdir in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, subdir)
    # Loop through the folder and append the image paths to the list
    for file in os.listdir(folder_path):
        # Check if the file is an image by its extension
        if file.endswith((".txt")):
            # copy file to caption_dir using shutil
            caption_file = os.path.join(folder_path, file)
            caption_output_file = os.path.join(caption_dir, f'{subdir}_{file}')
            
            shutil.copy(caption_file, caption_output_file)

            image_file = file.replace('.txt', '.jpg')
            ori_image = caption_file.replace('.txt', '.jpg')
            # copy ori image to high image dir
            high_image_dir = os.path.join(image_dir, 'high')
            # create high image dir
            if not os.path.exists(high_image_dir):
                os.makedirs(high_image_dir)
            
            high_image = os.path.join(high_image_dir, f'{subdir}_{image_file}')
            shutil.copy(ori_image, high_image)

            
            # copy ori image to low image dir
            low_image_dir = os.path.join(image_dir, 'low')
            # create low image dir
            if not os.path.exists(low_image_dir):
                os.makedirs(low_image_dir)
            
            low_image = os.path.join(low_image_dir, f'{subdir}_{image_file}')
            gen_image = os.path.join(generation_dir,subdir, image_file)

            shutil.copy(gen_image, low_image)

