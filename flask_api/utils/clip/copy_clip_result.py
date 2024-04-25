

# Import the modules
import os
import json
import clip_sim
import torch
from tqdm import tqdm
import shutil
import numpy as np

def main():
    result_json = "results.json"
    input_dir = "F:/ImageSet/openxl2_realism"
    above_average_dir = "F:/ImageSet/openxl2_realism_above_average"
    # create above_average dir
    if not os.path.exists(above_average_dir):
        os.mkdir(above_average_dir)
    
    # read json file
    with open(result_json, "r") as f:
        result = json.load(f)
    
    above_average_files_list = result['above_average_files_list']
    for item in above_average_files_list:
        subdir = item['subdir']
        subdir_path = os.path.join(above_average_dir, subdir)
        # create subdir
        if not os.path.exists(subdir_path):
            os.mkdir(subdir_path)
        
        input_subdir = os.path.join(input_dir, subdir)

        input_file = os.path.join(input_subdir,f'{item["file_name"]}{item["image_ext"]}')
        target_file = os.path.join(subdir_path,f'{item["file_name"]}{item["image_ext"]}')
        if os.path.exists(input_file) and not os.path.exists(target_file):
            shutil.copy(input_file, target_file)

        input_file = os.path.join(input_subdir,f'{item["file_name"]}{item["caption_ext"]}')
        target_file = os.path.join(subdir_path,f'{item["file_name"]}{item["caption_ext"]}')
        if os.path.exists(input_file) and not os.path.exists(target_file):
            shutil.copy(input_file, target_file)

        input_file = os.path.join(input_subdir,f'{item["file_name"]}.npz')
        target_file = os.path.join(subdir_path,f'{item["file_name"]}.npz')
        if os.path.exists(input_file) and not os.path.exists(target_file):
            shutil.copy(input_file, target_file)
    

if __name__ == '__main__':
    main()
