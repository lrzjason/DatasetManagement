import os
import json
import shutil

# provide input dir
# provide temp/below_list.json
# provide temp/upon_list.json
# provide output dir

# loop below_list and upon_list to get .jpg and .txt data
# copy .jpg and .txt to output dir

below_json = 'temp/below_list.json'
upon_json = 'temp/upon_list.json'

input_dir = 'F:/ImageSet/dump/mobcup_output_clip_failed_recalc_b2'
output_dir = 'F:/ImageSet/dump/mobcup_output'

# create output dir if not exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# read below_list and upon_list
below_list = []
upon_list = []
with open(below_json) as f:
    below_list = json.load(f)
with open(upon_json) as f:
    upon_list = json.load(f)

# concat below_list and upon_list
below_list.extend(upon_list)

below_set = set(item['name'] for item in below_list)

copy_count = 0

# loop input_dir to get .jpg and .txt data
for filename in os.listdir(input_dir):
    if filename in below_set or filename == 'summary':
        continue

    # copy to output_dir
    image_path = os.path.join(input_dir, filename)
    text_path = os.path.join(input_dir, filename.split('.')[0] + '.txt')
    if not os.path.exists(text_path):
        print("skipping",text_path)
        continue
    # use shutil to copy .jpg and .txt to output dir
    shutil.copy(image_path, output_dir)
    shutil.copy(text_path, output_dir)
    copy_count += 1

print("copy_count", copy_count)
