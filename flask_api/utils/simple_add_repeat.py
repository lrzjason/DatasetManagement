import os

# input_dir = 'F:/ImageSet/8k_images_captioned'
input_dir = 'F:/ImageSet/openxl2_dataset'
repeat = 1
for dir_name in os.listdir(input_dir):
    # rename folder with repeat prefix
    dir_path = os.path.join(input_dir, dir_name)
    rename_dir_path = os.path.join(input_dir, f'{repeat}_{dir_name}')
    os.rename(dir_path,rename_dir_path)
