

# Import the modules
import os
import shutil

input_dir = "F:/qBTdownload/[小丁(Fantasy Factory)] 20.08~20.12 21.05~22.05"

# def walk_dir(input_dir,root_dir):
#     # Loop through the folder and append the image paths to the list
#     for item in os.listdir(input_dir):
#         item_path = os.path.join(input_dir, item)
#         if os.path.isdir(item_path):
#             print(item_path)
#             walk_dir(item_path,root_dir)
#         else:
#             if item.endswith((".jpg")) or item.endswith((".png")) or item.endswith((".webp")):
#                 if os.path.exists(os.path.join(root_dir,item)):
#                     # delete file
#                     # os.remove(item_path)
#                     continue
#                 else:
#                     # move image to root dir
#                     os.rename(item_path,os.path.join(root_dir,item))

#                 # check current dir has image or not
#                 if len(os.listdir(sub_dir_path)) == 0:
#                     os.rmdir(sub_dir_path)
#             else:
#                 os.remove(item_path)


# for subdir in os.listdir(input_dir):
#     sub_dir_path = os.path.join(input_dir, subdir)
#     walk_dir(sub_dir_path,sub_dir_path)

# copy sub dir to root dir
for subdir in os.listdir(input_dir):
    sub_dir_path = os.path.join(input_dir, subdir)
    for item in os.listdir(sub_dir_path):
        item_path = os.path.join(sub_dir_path, item)
        if os.path.isdir(item_path):
            # move the item dir to root dir, the name should be f'{subdir}_{item}'
            shutil.move(item_path,os.path.join(input_dir,f'{subdir}_{item}'))


# remove empty dir
for subdir in os.listdir(input_dir):
    sub_dir_path = os.path.join(input_dir, subdir)
    for item in os.listdir(sub_dir_path):
        item_path = os.path.join(sub_dir_path, item)
        if os.path.isdir(item_path):
            if len(os.listdir(item_path)) == 0:
                os.rmdir(item_path)