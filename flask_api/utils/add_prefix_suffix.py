

# Import the modules
import os
import json

# Define the folder path and the output file name
# input_dir = "F:/ImageSet/vit_train/hand-classifier/1_good_hand"
input_dir = "F:/ImageSet/openxl2_worst/1_worst_image"

prefix = 'worst quality, worst anatomy, distortion'
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

for file in os.listdir(input_dir):
    # Check if the file is an image by its extension
    if file.endswith((".txt")):
        # Join the folder path and the file name to get the full path
        full_path = os.path.join(input_dir, file)
        content = ''
        print(full_path)
        # Append the full path to the list
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
            f.close()
        
        content = prefix + ', ' + content
        with open(full_path, "r+", encoding="utf-8") as out_f:
            out_f.write(content)