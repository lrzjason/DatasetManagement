import os
import shutil

# input_dir = 'F:/ImageSet/hagrid_test/hand_classifier'
# output_dir = 'F:/ImageSet/hagrid_test/vit_hand_classifier'

# input_dir = 'F:/ImageSet/anime_dataset/fullbody_dataset'
# output_dir = 'F:/ImageSet/anime_dataset/fullbody_dataset_classifier'

input_dir = 'F:/ImageSet/vit_train/crop_predict_cropped/highest'
output_dir = 'F:/ImageSet/vit_train/crop_predict_cropped/highest_train_validation'
copy_txt = True

# mk output dir

if not os.path.exists(output_dir):
  os.mkdir(output_dir)

dataset_configs = [
  {
    'name':'validation',
    'ratio':0.1
  },
  {
    'name':'train',
    'ratio':0.9
  }
]

label_dirs = os.listdir(input_dir)

min_dir_count = 0
for label_dir in label_dirs:
  label_path = os.path.join(input_dir,label_dir)
  label_files = os.listdir(label_path)
  num_files = len(label_files)
  
  if min_dir_count ==0 or num_files < min_dir_count:
    min_dir_count = num_files

print(min_dir_count)
for label_dir in label_dirs:
  label_path = os.path.join(input_dir,label_dir)
  label_files = os.listdir(label_path)
  # working arr to store proccessed file
  processed_files = []
  for config in dataset_configs:
    split_path = os.path.join(output_dir,config["name"])
    if not os.path.exists(split_path):
      # create split dir
      os.mkdir(split_path)
    split_label_path = os.path.join(split_path,label_dir)
    if not os.path.exists(split_label_path):
      # create split dir
      os.mkdir(split_label_path)

    # get split total number
    split_num = int(num_files * config["ratio"])
    count = 0
    for file_name in label_files:
      # break if reach split number
      if count >= split_num or count >= min_dir_count:
        break
      # skip file if processed, avoid duplicated record in train and validation
      if file_name in processed_files:
        continue
      shutil.copy(os.path.join(label_path,file_name),os.path.join(split_label_path,file_name))
      if copy_txt:
        shutil.copy(os.path.join(label_path,file_name.replace('.jpg','.txt')),os.path.join(split_label_path,file_name.replace('.jpg','.txt')))
      processed_files.append(file_name)
      count += 1
