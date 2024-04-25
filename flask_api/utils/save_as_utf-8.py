

# Import the modules
import os
import json

# Define the folder path and the output file name
input_dir = "F:/ImageSet/openxl2_realism_test"

prefix = ''
# suffix = ', 8k photo, high quality'
suffix = ''

for subdir in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, subdir)
    if os.path.isdir(folder_path):
        # Loop through the folder and append the image paths to the list
        for file in os.listdir(folder_path):
            # Check if the file is an image by its extension
            if file.endswith((".wd14_cap")):
                # Join the folder path and the file name to get the full path
                # txt_file = file.replace('.jpg', '.txt')
                full_path = os.path.join(folder_path, file)
                content = ''
                if os.path.exists(full_path):
                    # Append the full path to the list
                    with open(full_path, "r") as f:
                        content = f.read()
                        f.close()
                with open(full_path, "r+", encoding="utf-8") as out_f:
                    out_f.write(content)

# for file in os.listdir(input_dir):
#     # Check if the file is an image by its extension
#     if file.endswith((".txt")):
#         # Join the folder path and the file name to get the full path
#         full_path = os.path.join(input_dir, file)
#         content = ''
#         print(full_path)
#         # Append the full path to the list
#         with open(full_path, "r", encoding="utf-8") as f:
#             content = f.read()
#             f.close()
#         content = prefix + content + suffix
#         with open(full_path, "r+", encoding="utf-8") as out_f:
#             out_f.write(content)