

# Import the modules
import os
import json

input_dir = "F:/ImageSet/openxl2_dataset"

repeat = "30"

for subdir in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, subdir)
    
    if "_" in subdir:
        dir_name = subdir.split('_')
        folder_name = '_'.join(dir_name[1:])
        print(folder_name)
        os.rename(folder_path, os.path.join(input_dir, f'{repeat}_{folder_name}'))
