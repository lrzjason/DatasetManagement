

# Import the modules
import os
import json

# Define the folder path and the output file name
input_dir = 'F:/ImageSet/openxl2_worst'
caption_dir = 'F:/ImageSet/pickscore_random_captions_pag_ays_parent/pickscore_random_captions_pag_ays'

file_ext = '.webp'

for subdir in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, subdir)
    # Loop through the folder and append the image paths to the list
    for file in os.listdir(folder_path):
        # Check if the file is an image by its extension
        if file.endswith((file_ext)):
            # Join the folder path and the file name to get the full path
            txt_file = file.replace(file_ext, '.txt')
            caption_path = os.path.join(caption_dir, txt_file)
            full_path = os.path.join(folder_path, txt_file)
            
            if os.path.exists(full_path):
                continue
            # copy caption to full path
            if os.path.exists(caption_path):
                with open(caption_path, 'r',encoding='utf-8') as f:
                    caption = f.read()
                with open(full_path, 'w',encoding='utf-8') as f:
                    f.write(caption)