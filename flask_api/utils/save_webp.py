

# Import the modules
import os
import json
import PIL
from PIL import Image
from PIL import ImageOps
from tqdm import tqdm



input_dir = "F:/ImageSet/training_script_cotton_doll/test"

output_dir = "F:/ImageSet/training_script_cotton_doll/test_webp"

# create the output dir
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

remove_original = True

# based on file size, return lossless, quality
def get_webp_params(filesize_mb):
    if filesize_mb <= 2:
        return (False, 100)
    if filesize_mb <= 4:
        return (False, 90)
    return (False, 80)

# Set the maximum pixels to prevent out of memory error
PIL.Image.MAX_IMAGE_PIXELS = 933120000

for subdir in tqdm(os.listdir(input_dir), position=0, desc='Subdir'):
    folder_path = os.path.join(input_dir, subdir)
    if os.path.isdir(folder_path):
        # Loop through the folder and append the image paths to the list
        for file in tqdm(os.listdir(folder_path), position=1, desc='File'):
            # Check if the file is an image by its extension
            if file.endswith((".jpg")) or file.endswith((".png")):
                # get filename and ext from file
                file_name, file_ext = os.path.splitext(file)
                
                output_subdir = os.path.join(output_dir, subdir)
                if not os.path.exists(output_subdir):
                    os.mkdir(output_subdir)
                webp_path = os.path.join(output_subdir, file_name + ".webp")

                if os.path.exists(webp_path):
                    continue

                # Join the folder path and the file name to get the full path
                full_path = os.path.join(folder_path, file)

                # get webp params
                filesize = os.path.getsize(full_path) 
                # print('File: ' + file + ' Size: ' + str(filesize) + ' bytes')
                filesize_mb = filesize / 1024 / 1024
                lossless, quality = get_webp_params(filesize_mb)
                
                try:
                    with Image.open(full_path) as image:
                        # exif = image.info['exif']
                        image = ImageOps.exif_transpose(image)
                        image.save(webp_path, 'webp', optimize = True, quality = quality, lossless = lossless)

                except:
                    print(f"Error in file {full_path}")
                    os.remove(full_path)
                    print(f"Removed file {full_path}")
            
                # print("Saved " + webp_path)
                # remove original image
                if remove_original:
                    os.remove(full_path)
            
# for file in tqdm(os.listdir(input_dir), position=1, desc='File'):
#     # Check if the file is an image by its extension
#     if file.endswith((".jpg")) or file.endswith((".png")):
#         # get filename and ext from file
#         file_name, file_ext = os.path.splitext(file)
        
#         # output_subdir = os.path.join(output_dir, subdir)
#         # if not os.path.exists(output_subdir):
#         #     os.mkdir(output_subdir)
#         webp_path = os.path.join(output_dir, file_name + ".webp")

#         if os.path.exists(webp_path):
#             continue

#         # Join the folder path and the file name to get the full path
#         full_path = os.path.join(input_dir, file)

#         # get webp params
#         filesize = os.path.getsize(full_path) 
#         # print('File: ' + file + ' Size: ' + str(filesize) + ' bytes')
#         filesize_mb = filesize / 1024 / 1024
#         lossless, quality = get_webp_params(filesize_mb)
        
#         try:
#             with Image.open(full_path) as image:
#                 # exif = image.info['exif']
#                 image = ImageOps.exif_transpose(image)
#                 image.save(webp_path, 'webp', optimize = True, quality = quality, lossless = lossless)

#         except:
#             print(f"Error in file {full_path}")
#             os.remove(full_path)
#             print(f"Removed file {full_path}")
        
#         # print("Saved " + webp_path)
#         # remove original image
#         if remove_original:
#             os.remove(full_path)
        