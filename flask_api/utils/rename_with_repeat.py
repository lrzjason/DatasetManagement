import os
import shutil

input_dir = 'F:/ImageSet/openxl2_dataset'

# output_dir = 'F:/ImageSet/anime_dataset/genshin_classified_with_viewpoint_repeat'

dir_info = {}

# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
max_folder = 0
for dir_name in os.listdir(input_dir):
    dir_path = os.path.join(input_dir, dir_name)
    image_count = 0
    for file in os.listdir(dir_path):
        if file.endswith('.jpg'):
            image_count += 1
    dir_info[dir_name] = image_count
    if dir_info[dir_name] > max_folder:
        max_folder = dir_info[dir_name]
        max_folder_name = dir_name
dir_info['max'] = max_folder
dir_info['max_name'] = max_folder_name
print(dir_info)
# print(dir_info)

repeat_info = {}
target_value = max_folder - 200
for dir_name in os.listdir(input_dir):
    base_value = dir_info[dir_name]
    repeat = int(target_value/base_value)
    if repeat == 0:
        repeat = 1
    repeat_info[dir_name] = repeat
    
    # rename folder with repeat prefix
    dir_path = os.path.join(input_dir, dir_name)
    rename_dir_path = os.path.join(input_dir, f'{repeat}_{dir_name}')
    os.rename(dir_path,rename_dir_path)
    
print(repeat_info)
# shutil.copytree('baz', 'foo', dirs_exist_ok=True)  # Fine