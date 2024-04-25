

# Import the modules
import os
import json

# Define the folder path and the output file name
input_dir = "F:/ImageSet/openxl2_realism_above_average/1_tags"

for file in os.listdir(input_dir):
    # Check if the file is an image by its extension
    if file.endswith((".wd14_cap")):
        # Join the folder path and the file name to get the full path
        full_path = os.path.join(input_dir, file)
        content = ''
        print(full_path)
        # rename .wd14_cap to .txt
        new_name = full_path.replace('.wd14_cap', '.txt')
        os.rename(full_path, new_name)