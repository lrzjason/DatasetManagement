

# Import the modules
import os
import json

# Define the folder path and the output file name
# folder_path = "F:/lora_training/hand_gesture_training/images/20_call_test"
# input_dir = "F:/ImageSet/8k_images_captioned"
input_dir = "F:/ImageSet/openxl2_dataset"


for subdir in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, subdir)
    # Loop through the folder and append the image paths to the list
    for file in os.listdir(folder_path):
        # Check if the file is an image by its extension
        if file.endswith((".wd14cap")):
            # remove the file
            os.remove(os.path.join(folder_path, file))
