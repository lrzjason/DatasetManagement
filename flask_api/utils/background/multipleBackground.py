import cv2
import numpy as np

from PIL import Image
from transparent_background import Remover
import os
import json
import torch

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

model_name_or_path = 'F:/Transformers/Vit Classification/vit-base-man-classifier'
man_woman_processor = ViTImageProcessor.from_pretrained(model_name_or_path)
man_woman_classifier = ViTForImageClassification.from_pretrained(model_name_or_path)

# Load model
remover = Remover() # default setting

input_dir = 'F:/ImageSet/hagrid_train_all_classified'
# filename = '1ab60b54-e6b3-46f4-a700-b0500a7c0bc4.jpg'

output_dir = 'F:/ImageSet/hagrid_train_all_classified_multiple_background'

# create output folder if not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

subset_list = os.listdir(input_dir)
for subset_index,subset_name in enumerate(subset_list):
    subset_dir = os.path.join(input_dir, subset_name)
    output_subset_dir = os.path.join(output_dir, subset_name)
    if not os.path.exists(output_subset_dir):
        os.makedirs(output_subset_dir)
    file_list = os.listdir(subset_dir)
    # list input_dir
    for file_index,file_name in enumerate(file_list):
        if not file_name.endswith('.jpg'):
            continue
        print(f"processing file {file_index+1}/{len(file_list)}")
        filename = file_name
        # Usage for image
        img = Image.open(os.path.join(subset_dir,filename)).convert('RGB') # read image

        filename = os.path.splitext(filename)[0]
        # save ori
        filename_output = f"{filename}_ori"
        img.save(os.path.join(output_subset_dir, f"{filename_output}.webp"))

        removebg_img = remover.process(img) 

        # color = (153, 153, 255)

        colors = []
        json_file = 'backgroundConfig.json'
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                colors = json.load(f)

        def convert_to_rgb(color):
            color_value = color.split(',')
            # print(color_value)
            return (int(color_value[0].strip()), int(color_value[1].strip()), int(color_value[2].strip()))

        def get_caption(person_desc,gesture_desc,hand_desc,background_desc):
            return f"a {person_desc} is making a {gesture_desc} using {hand_desc}{background_desc}, low quality, jpeg artifacts"

        inputs = man_woman_processor(img, return_tensors="pt")
        with torch.no_grad():
            logits = man_woman_classifier(**inputs).logits

        # model predicts one of the 1000 ImageNet classes
        person_desc = man_woman_classifier.config.id2label[logits.argmax(-1).item()]
        # print(f'predicted_label: {person_desc}')



        # read the .txt file
        with open(os.path.join(subset_dir, f"{filename}.txt"), 'r', encoding='utf-8') as f:
            text = f.read()
            # split text by ,
            text_list = text.split(',')
            hand_desc = text_list[0]
            gesture_desc = text_list[1]

        background_desc = f" raw photo background"
        # save 
        caption = get_caption(person_desc, gesture_desc, hand_desc, background_desc)

        # save caption with the same name of image but in txt ext
        with open(os.path.join(output_subset_dir, f"{filename_output}.txt"), 'w', encoding='utf-8') as f:
            f.write(caption)

        for color_index, color in enumerate(colors):
            print(f"processing file {file_index+1}/{len(file_list)} in {color_index+1}/{len(colors)}")
            # print(color['name'])
            background_desc = f" in {color['name']} color background, simple background"
            color_value = convert_to_rgb(color['rgb'])
            # print(color_value)
            # Create a new image with the desired background color
            background = Image.new("RGB", img.size, color=color_value)

            # Paste the input image on top of the background
            background.paste(removebg_img, (0, 0), removebg_img)


            
            final_name = f"{filename}_{color['name']}"

            # caption = f"a {person_desc} is making a {gesture_desc} using {hand_desc} in {color['name']} background, simple background, low quality, jpeg artifacts"

            filename_output = f"{final_name}.webp"
            # Save the output image
            background.save(os.path.join(output_subset_dir,filename_output))

            caption = get_caption(person_desc, gesture_desc, hand_desc, background_desc)
            # save caption with the same name of image but in txt ext
            with open(os.path.join(output_subset_dir, f"{final_name}.txt"), 'w', encoding='utf-8') as f:
                f.write(caption)