from datasets import load_dataset
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
import torch
import numpy as np
from datasets import load_metric
import os
import shutil
import json
from imgutils.validate import anime_rating, nsfw_pred

def copy_files(sub_dir,file,output_sub_viewpoint_dir):
  file_name, file_ext = os.path.splitext(file)
  # shutil.copy(os.path.join(sub_dir, file), output_sub_viewpoint_dir)
  shutil.copy(os.path.join(sub_dir, file), os.path.join(output_sub_viewpoint_dir, f'{file_name}.capbackup'))
  # copy .jpg to hand dir
  shutil.copy(os.path.join(sub_dir, f'{file_name}.jpg'), output_sub_viewpoint_dir)
  # copy .jpg to hand dir
  shutil.copy(os.path.join(sub_dir, f'{file_name}.wd14cap'), output_sub_viewpoint_dir)

def classifier(model,image):
  inputs = image_processor(image, return_tensors="pt")
  with torch.no_grad():
      logits = model(**inputs).logits

  # model predicts one of the 1000 ImageNet classes
  predicted_label = logits.argmax(-1).item()
  return model.config.id2label[predicted_label]

model_name_or_path = 'F:/Transformers/Vit Classification/vit-base-fullbody-classifier-6'
image_processor = ViTImageProcessor.from_pretrained(model_name_or_path)
model = ViTForImageClassification.from_pretrained(model_name_or_path)

# model predicts one of the 1000 ImageNet classes
# predicted_label = logits.argmax(-1).item()

input_dir = 'F:/ImageSet/anime_dataset/genshin_classified'
# ref_dir = 'F:/ImageSet/hagrid_classified'

# ref_classes = ['left_hand','right_hand']

id_labels = model.config.id2label

subdir = os.listdir(input_dir)

output_dir = 'F:/ImageSet/anime_dataset/genshin_classified_with_viewpoint'

json_file = 'update_caption_with_classifier.json'
data = {}

save_json_per_image = 100
save_count = 0

if os.path.exists(json_file):
  # remove json file
  os.remove(json_file)

# create output dir
if not os.path.exists(output_dir):
  os.mkdir(output_dir)

# sub_index = 0
total_sub = len(subdir)
# loop each subdir
for sub in subdir:
  # sub_index+=1
  sub_dir = os.path.join(input_dir, sub)
  # ref_sub_dir = os.path.join(ref_dir, sub)
  
  # output_sub_dir = os.path.join(output_dir, sub)
  # # create subdir in output dir
  # if not os.path.exists(output_sub_dir):
  #   os.mkdir(output_sub_dir)

  data[sub] = {
    'processed':[],
    'missing':[]
  }
  for index in id_labels:
    label = id_labels[index]
    output_sub_viewpoint_dir = os.path.join(output_dir, f'{sub}_{label}')
    # create subdir in output dir
    if not os.path.exists(output_sub_viewpoint_dir):
      os.mkdir(output_sub_viewpoint_dir)
  
  file_index = 0
  # because it has .jpg and .txt, divide 2
  # total_image = len(os.listdir(subdir_clear))/2
  # loop each file in subdir_clear
  for file in os.listdir(sub_dir):
    # skip when ext not eq to .txt
    if not file.endswith('.txt'):
      continue
    file_index+=1
    save_count+=1
    if save_count >= save_json_per_image:
      save_count = 0
      with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f)
      print(f'json file saved at {json_file}')

    print(f'Processing {sub} {file_index} file: {file}')
    # get file ext
    file_name, file_ext = os.path.splitext(file)
    
    image = Image.open(os.path.join(sub_dir, f'{file_name}.jpg'))
    image_class = classifier(model, image)
    output_sub_viewpoint_dir = os.path.join(output_dir, f'{sub}_{image_class}')
    
    copy_files(sub_dir,file,output_sub_viewpoint_dir)

    image_path = os.path.join(output_sub_viewpoint_dir, f'{file_name}.jpg')

    rating_caption = ''
    rating_label,rating = anime_rating(image_path)
    rating_caption = f', rating:{rating_label}'

    nsfw_label,nsfw_rating = nsfw_pred(image_path)
    nsfw_caption = ''
    if nsfw_label in ['drawings','neutral']:
      nsfw_caption = f', sensitive:neutral'
    else:
      nsfw_caption = f', sensitive:{nsfw_label}'

    new_content = ''
    with open(os.path.join(sub_dir, file), 'r', encoding="utf-8") as f:
      content = f.read()
      new_content = content.strip(' ').strip(',').strip('.')
      new_content = f'{image_class} anime artwork of {new_content}{rating_caption}{nsfw_caption}'
    
    with open(os.path.join(output_sub_viewpoint_dir, file), "w", encoding="utf-8") as out_f:
      out_f.write(new_content)
    

    data[sub]['processed'].append({
      'file_name':file_name,
      'classified':image_class,
      # 'new_caption':new_content,
      # 'before_caption':content,
    })
    # f.close()
    # print(f'sub_index: {sub_index}/{total_sub} file_index: {file_index}/{total_image} file: {file_name} processed.')
  #   break
  # break
