

# Import the modules
import os
import json

input_dir = "F:/qBTdownload/Vinnegal"
# input_dir = 'F:/qBTdownload/test'


for subdir in os.listdir(input_dir):
    subdir_path = os.path.join(input_dir, subdir)
    # Loop through the folder
    for item in os.listdir(subdir_path):
        item_path = os.path.join(subdir_path, item)
        # check item is dir 
        if os.path.isdir(item_path):
            new_path = os.path.join(input_dir, f'{subdir}_{item}')
            os.rename(item_path, new_path)