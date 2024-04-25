import os
import shutil
from tqdm import tqdm

filtered_tags = ['genderswap','tentacles_on_male','penis','futanari']

# input_dir = "F:/ImageSet/anime_dataset/genshin_classified"

# output_dir = "F:/ImageSet/anime_dataset/genshin_classified_filtered"


input_dir = "F:/ImageSet/anime_dataset/genshin_classified"

output_dir = "F:/ImageSet/anime_dataset/genshin_classified_filtered"

# create output dir
os.makedirs(output_dir,exist_ok=True)

# list subdir in input_dir
for subdir in tqdm(os.listdir(input_dir)):
    output_subdir = os.path.join(output_dir,subdir)
    input_subdir = os.path.join(input_dir,subdir)
    subdir_files = os.listdir(input_subdir)
    if len(subdir_files)<100:
        if len(subdir_files) == 0:
            shutil.rmtree(input_subdir)
            continue
        # loop all files in subdir
        # get file path
        output_subdir = os.path.join(output_dir,subdir)
        for file in subdir_files:
            # create output sub dir
            os.makedirs(output_subdir,exist_ok=True)
            input_file_path = os.path.join(input_subdir,file)
            output_file_path = os.path.join(output_subdir,file)
            # move file to output_dir
            shutil.move(input_file_path,output_file_path)

            
            # # read file content and check if it contains any of the filtered tags
            # if any(tag in open(file_path).read() for tag in filtered_tags):
            #     # if it contains any of the filtered tags, move the file to output_dir
            #     # create output sub dir
            #     os.makedirs(os.path.join(output_dir,subdir),exist_ok=True)
            #     shutil.move(file_path,os.path.join(output_dir,subdir,file))
            #     file_name = os.path.splitext(file)[0]
            #     shutil.move(os.path.join(input_dir,subdir,file_name+".jpg"),os.path.join(output_dir,subdir,file_name+".jpg"))

# # list subdir in input_dir
# for subdir in tqdm(os.listdir(input_dir)):
#     # loop all files in subdir
#     for file in os.listdir(os.path.join(input_dir,subdir)):
#         if file.endswith(".wd14cap"):
#             # get file path
#             file_path = os.path.join(input_dir,subdir,file)
#             # read file content and check if it contains any of the filtered tags
#             if any(tag in open(file_path).read() for tag in filtered_tags):
#                 # if it contains any of the filtered tags, move the file to output_dir
#                 # create output sub dir
#                 os.makedirs(os.path.join(output_dir,subdir),exist_ok=True)
#                 shutil.move(file_path,os.path.join(output_dir,subdir,file))
#                 file_name = os.path.splitext(file)[0]
#                 shutil.move(os.path.join(input_dir,subdir,file_name+".jpg"),os.path.join(output_dir,subdir,file_name+".jpg"))
