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
output_dir = 'F:/ImageSet/dump/mobcup_output_clip_failed_b3'

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

# loop below_list and upon_list to get .jpg and .txt data
for item in below_list:
    if item['name'] == 'summary':
        continue
    image_path = os.path.join(input_dir, item['name'])
    text_path = os.path.join(input_dir, item['name'].split('.')[0] + '.txt')
    if not os.path.exists(text_path):
        print("skipping",text_path)
        continue
    # use shutil to copy .jpg and .txt to output dir
    shutil.copy(image_path, output_dir)
    shutil.copy(text_path, output_dir)

