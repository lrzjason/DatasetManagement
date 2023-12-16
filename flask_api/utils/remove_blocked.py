import json
import os
from datasets import load_dataset
import shutil


# DATASET = "yuvalkirstain/pickapic_v2_no_images"
DATASET = "F:/ImageSet/PickScore/pickapic_v2_no_images"
SPLIT = "train"


BLOCKED_IDS = [
    280, 331, 437, 641, 718, 729, 783, 984, 1023, 1040, 1059, 1149, 1187, 1177, 1202, 1203, 1220,
    1230, 1227, 1279, 1405, 1460, 1623, 1627, 1758, 1801, 1907, 1917, 1922, 2071, 2215, 2239, 2286, 2322, 2357, 2452,
    2459, 2481, 2513, 2515, 2520, 2545, 2596, 2603, 2617, 2638, 2709, 2783, 2842, 2266, 2899, 3084, 3138, 3243, 3264,
    3265, 3267, 3251, 3292, 3268, 3271, 1961, 3302, 3318, 1689, 3278, 1382, 3542, 3446, 3633, 1526, 4710, 4748, 4762,
    4444, 4870, 4733, 4878, 4928, 4939, 4926, 4942, 5019, 4946, 5006, 4241, 5027, 5015, 5041, 5032, 5047, 5054, 5064,
    5023, 5137, 5281, 4115, 5273, 4347, 3523, 5403, 3589, 5697, 6574, 6573, 6822, 7037, 7277, 8078, 7995, 3604,
    7947, 7277, 8079, 4565, 7931, 4597, 8118, 8176, 8313, 8285, 6032
]


# part 1 load dataset
iterable_dataset = load_dataset(DATASET,split=SPLIT, streaming=True)
# load all from test rather than 40000
subset = list(iterable_dataset.filter(lambda record: record["has_label"]))
# subset = list(iterable_dataset.filter(lambda record: record["has_label"]).take(40000))

caption_dir = 'F:\\ImageSet\\Pickscore_train_10k\\captions'
image_dir = 'F:\\ImageSet\\Pickscore_train_10k\\images'
thumbnail_dir = 'F:\\ImageSet\\Pickscore_train_10k\\images_thumbnails'


caption_dir_blocked = 'F:\\ImageSet\\Pickscore_train_10k\\captions_blocked'
image_dir_blocked = 'F:\\ImageSet\\Pickscore_train_10k\\images_blocked'
thumbnail_dir_blocked = 'F:\\ImageSet\\Pickscore_train_10k\\images_thumbnails_blocked'

indicator_folders = ["low","high"]

for subfolder in indicator_folders:
  #  create blocked image and thumbnail subfolder if not exist
  if not os.path.exists(f"{image_dir_blocked}/{subfolder}"):
    os.makedirs(f"{image_dir_blocked}/{subfolder}")
  if not os.path.exists(f"{thumbnail_dir_blocked}/{subfolder}"):
    os.makedirs(f"{thumbnail_dir_blocked}/{subfolder}")

# create blocked folder if not exist
if not os.path.exists(caption_dir_blocked):
    os.makedirs(caption_dir_blocked)
if not os.path.exists(image_dir_blocked):
    os.makedirs(image_dir_blocked)


saved_json = 'F:\\DatasetManagement\\flask_api\\saved_pairs.json'
deleted_json = 'F:\\DatasetManagement\\flask_api\\deleted_pairs.json'

CAPTION_EXT = '.txt'
processed_captions = os.listdir(caption_dir)


count = 0
log_count = 50

saved_pairs = []
if os.path.exists(saved_json):
    with open(saved_json, "r", encoding='utf-8') as f:
        saved_pairs = json.load(f)

deleted_pairs = []
if os.path.exists(deleted_json):
    with open(deleted_json, "r", encoding='utf-8') as f:
        deleted_pairs = json.load(f)

dataset_count = 0

# loop blocked caption
# for item in os.listdir(caption_dir_blocked):
#     handle_image = item.split('.')[0]
#     # add handle_image to deleted_pairs if not in deleted_pairs
#     if handle_image not in deleted_pairs:
#         deleted_pairs.append(handle_image)

# with open(deleted_json, "w", encoding='utf-8') as f:
#     json.dump(deleted_pairs, f)

# # print(take1)
for item in subset:
  handle_image = item["best_image_uid"]
  # dataset keys
  # ['are_different', 'best_image_uid', 'caption', 
  # 'created_at', 'has_label', 'image_0_uid', 'image_0_url',
  #  'image_1_uid', 'image_1_url', 'jpg_0', 'jpg_1', 'label_0', 
  # 'label_1', 'model_0', 'model_1', 'ranking_id', 'user_id', 
  # 'num_example_per_prompt', '__index_level_0__']
  inprocessed_captions = f"{handle_image}{CAPTION_EXT}" in processed_captions
  if not inprocessed_captions:
    continue
  dataset_count += 1
  # stop process when handled all captions
  if dataset_count > len(processed_captions):
    with open(deleted_json, "w", encoding='utf-8') as f:
        json.dump(deleted_pairs, f)
    break
  # skip processed image
  if handle_image in saved_pairs:
    continue
  # skip processed image
  if handle_image in deleted_pairs:
    continue
  print('process image:',handle_image)
  # move image to skipped if it in blocked list
  if item['user_id'] in BLOCKED_IDS:
    if count % log_count == 0:
      with open(deleted_json, "w", encoding='utf-8') as f:
          json.dump(deleted_pairs, f)
    count += 1
    print("moving to blocked",handle_image)
    # move caption to blocked folder
    caption_path = os.path.join(caption_dir, handle_image + ".txt")
    print("skipping",caption_path)
    shutil.move(caption_path, caption_dir_blocked)

    for subfolder in indicator_folders:
      #  move subfolder image to blocked folder
      image_path = os.path.join(image_dir, subfolder, handle_image + ".png")
      thumbnail_path = os.path.join(thumbnail_dir, subfolder, handle_image + ".png")
      # image blocked subfolder path
      image_blocked_path = os.path.join(image_dir_blocked, subfolder)
      shutil.move(image_path, image_blocked_path)

      # move thumbnail to blocked folder
      thumbnail_blocked_path = os.path.join(thumbnail_dir_blocked, subfolder)
      shutil.move(thumbnail_path, thumbnail_blocked_path)

    # add handle_image to deleted_pairs json
    deleted_pairs.append(handle_image)

