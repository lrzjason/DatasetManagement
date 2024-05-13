

# Import the modules
import os
import json
import re

# Define the folder path and the output file name
# input_dir = "F:/ImageSet/vit_train/hand-classifier/1_good_hand"
input_dir = "F:/ImageSet/openxl2_creative2_fix_saturation/10_lexica"
ref_dir = "F:/ImageSet/openxl2_creative2/10_raw_photo"


prefix = ''
# suffix = ', 8k photo, high quality'
suffix = ''

# for subdir in os.listdir(input_dir):
#     folder_path = os.path.join(input_dir, subdir)
#     # Loop through the folder and append the image paths to the list
#     for file in os.listdir(folder_path):
#         # Check if the file is an image by its extension
#         if file.endswith((".jpg")):
#             # Join the folder path and the file name to get the full path
#             txt_file = file.replace('.jpg', '.txt')
#             full_path = os.path.join(folder_path, txt_file)
#             content = ''
#             if os.path.exists(full_path):
#                 # Append the full path to the list
#                 with open(full_path, "r", encoding="utf-8") as f:
#                     content = f.read()
#                     f.close()
#             else:
#                 with open(full_path, "w", encoding="utf-8") as f:
#                     f.close()
#             content = prefix + ', ' + content
#             if len(suffix)>0:
#                 content =  content + ', ' + suffix
#             with open(full_path, "r+", encoding="utf-8") as out_f:
#                 out_f.write(content)


# def remove_non_ascii(text):
#     return re.sub(r'[^/x00-/x7F]*[/ ]', '_', text)

for file in os.listdir(ref_dir):
    # Check if the file is an image by its extension
    if file.endswith((".txt")):
        # Join the folder path and the file name to get the full path
        ref_path = os.path.join(ref_dir, file)
        content = ''
        print(ref_path)
        # Append the full path to the list
        with open(ref_path, "r", encoding="utf-8") as f:
            content = f.read()
            content = content.replace('/n', '').strip()
            f.close()
        
        of_index = content.index(' of ')
        if of_index>0:
            prefix = content[:of_index+4]
        print(prefix)

        content = ""
        full_path = os.path.join(input_dir, file)
        # Append the full path to the list
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
            content = content.replace('/n', '').strip()
            f.close()
        content = prefix + content
        # content = prefix + content + suffix
        # # content = content.replace(prefix, '')

        # # content = remove_non_ascii(content)
        with open(full_path, "w", encoding="utf-8") as out_f:
            out_f.write(content)