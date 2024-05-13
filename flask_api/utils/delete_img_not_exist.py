import glob
import os

input_dir = "F:/ImageSet/openxl2_creative2_fix_saturation/1_lexica_ori"

# loop through each .txt file in input_dir
# if the .jpg file with the same name does not exist, delete the .txt file

for file_path in glob.glob(input_dir + "/*.txt"):
    # get file name without extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    # get image file name
    image_file = file_name + ".jpg"

    # original
    image_file_ori = os.path.join(input_dir, image_file)

    # delete file if image file does not exist
    if not os.path.exists(image_file_ori):
        print("deleting",file_path)
        os.remove(file_path)
        print("deleted",file_path)
    else:
        print("keeping",file_path)