import os
import json
import shutil
from PIL import Image
from PIL import ImageOps

# based on file size, return lossless, quality
def get_webp_params(filesize_mb):
    if filesize_mb <= 2:
        return (False, 100)
    if filesize_mb <= 4:
        return (False, 90)
    return (False, 80)
# provide input dir
# provide temp/below_list.json
# provide temp/upon_list.json
# provide output dir

# loop below_list and upon_list to get .jpg and .txt data
# copy .jpg and .txt to output dir

# input_dir = 'F:/ImageSet/openxl2_realism/temp'
input_dir = 'F:/ImageSet/openxl2_worst'
output_dir = 'F:/ImageSet/vit_train/crop_predict'

count = 0
max_count = 100

# create output dir if not exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for subset in os.listdir(input_dir):
    subset_dir = os.path.join(input_dir, subset)
    
    for image_file in os.listdir(subset_dir):
        if image_file.endswith('.webp') or image_file.endswith('.jpg'):
            filename, ext = os.path.splitext(image_file)
            if "." in filename:
                filename = filename.replace(".", "_")
            print('filename:',f"{subset}_{filename}.webp")   


            output_file = os.path.join(output_dir, f"{subset}_{filename}.webp")
            if os.path.exists(output_file):
                continue
            if count < max_count:
                count += 1
            else:
                break
            
            full_path = os.path.join(subset_dir, image_file)
            # copy file to output dir
            if image_file.endswith('.webp'):
                shutil.copy(full_path, output_file)
            else:

                # get webp params
                filesize = os.path.getsize(full_path) 
                # print('File: ' + file + ' Size: ' + str(filesize) + ' bytes')
                filesize_mb = filesize / 1024 / 1024
                lossless, quality = get_webp_params(filesize_mb)
                
                try:
                    with Image.open(full_path) as image:
                        # exif = image.info['exif']
                        image = ImageOps.exif_transpose(image)
                        image.save(output_file, 'webp', optimize = True, quality = quality, lossless = lossless)

                except:
                    print(f"Error in file {full_path}")
                    os.remove(full_path)
                    print(f"Removed file {full_path}")

print('count',count)
print('done')