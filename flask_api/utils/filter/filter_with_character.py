import os
import csv
import torch
import numpy as np
import pandas as pd
import onnxruntime
from PIL import Image
import cv2
from pathlib import Path
from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException
from huggingface_hub import hf_hub_download,hf_hub_url
from tqdm import tqdm
from onnxconverter_common import float16
import shutil
import re

CHARACTER_CATEGORY = 4

def download_model_files(model_repo_id):
    # Define the URLs for the model and tags file
    model_url = hf_hub_url(repo_id=model_repo_id, filename='model.onnx')
    tags_url = hf_hub_url(repo_id=model_repo_id, filename='selected_tags.csv')

    # Define local paths to save the files
    local_model_path = hf_hub_download(repo_id=model_repo_id, filename='model.onnx')
    local_tags_path = hf_hub_download(repo_id=model_repo_id, filename='selected_tags.csv')

    return local_model_path, local_tags_path


def preprocess_image(image):
    image = image.convert('RGBA')
    bg = Image.new('RGBA', image.size, 'WHITE')
    bg.paste(image, mask=image)
    image = bg.convert('RGB')
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # convert to BGR format
    h, w = image.shape[:2]
    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    image = np.pad(image, [(pad_h, pad_h), (pad_w, pad_w), (0, 0)], mode='constant', constant_values=255)
    image = cv2.resize(image, (448, 448), interpolation=cv2.INTER_AREA)
    image = np.expand_dims(image, 0)
    return image.astype(np.float32)

def run_model(image_path, model_path, tags_path, session, tag_threshold, filter_tags,selected_columns):
    image = Image.open(image_path)
    processed_image = preprocess_image(image)
    result = session.run(None, {session.get_inputs()[0].name: processed_image})[0]
    tags = pd.read_csv(tags_path)
    tags.reset_index(inplace=True)
    result_df = pd.DataFrame(result[0], columns=['Score'])
    result_with_tags = pd.concat([tags, result_df], axis=1)
    tags_filtered = result_with_tags[selected_columns]
    tags_filtered = tags_filtered[~tags_filtered['name'].isin(filter_tags)]
    # print('tags_filtered')
    # print(tags_filtered)
    return tags_filtered

ILLEGAL_NTFS_CHARS = "[<>:/\\|?*\"]|[\0-\31]"
def __removeIllegalChars(name):
    # removes characters that are invalid for NTFS
    return re.sub(ILLEGAL_NTFS_CHARS, "", name)

if __name__ == '__main__':
    # input_dir = 'F:/ImageSet/AOTW_dataset/images'
    input_dir = 'F:/ImageSet/anime_dataset/genshin'

    output_dir = 'F:/ImageSet/anime_dataset/genshin_classified'
    output_extension = '.wd14cap'

    two_character_dir = 'two_character'
    multiple_character_dir = 'multiple_character'


    sessions = []
    tags_paths = []
    model_repo_id = 'SmilingWolf/wd-v1-4-swinv2-tagger-v2'
    # tags_path = './tags.csv'
    stack_models = False
    tag_threshold = 0.5
    filter_tags = ['questionable', 'general', 'sensitive']
    # if stack_models:
    #     model_repo_ids = ['SmilingWolf/wd-v1-4-convnext-tagger-v2', 'SmilingWolf/wd-v1-4-vit-tagger-v2',
    #                       'SmilingWolf/wd-v1-4-swinv2-tagger-v2']
    model_path, tags_path = download_model_files(model_repo_id)
    try:
        session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    except RuntimeException:
        print("CUDA isn't available. Trying to run on CPU.")
        try:
            session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        except RuntimeException:
            print("Can't run the model. Exiting.")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    subdir = os.listdir(input_dir)
    files = []
    for sub in subdir:
        # list sub dir
        subdir_path = os.path.join(input_dir, sub)
        for f in os.listdir(subdir_path):
            if f.endswith(".jpg") or f.endswith(".webp"):
                files.append(os.path.join(subdir_path, f))

        # add .webp to files

    pbar = tqdm(files)
    print(files[0])

    selected_columns = ['name', 'Score', 'category']
    for file in pbar:
        if file.endswith(".jpg") or file.endswith(".webp"):
            pbar.set_description(f"Processing {file}")
            image_path = os.path.join(input_dir,file)
            tags_scores = []
            tags_filtered = run_model(image_path, model_path, tags_path, session, tag_threshold, filter_tags,selected_columns)
            tags_filtered.columns = ['name', 'Score', 'category']  # rename columns
            tags_filtered = tags_filtered[tags_filtered['Score'] > tag_threshold]
            tags_filtered.sort_values('Score', ascending=False, inplace=True)
            # print(tags_filtered)
            target_dir = os.path.join(output_dir, "none_character")
            skip = True
            content = ""
            character_prefix = ""
            gender_prefix = ""
            character_list = []
            human_count = 0

            # with open(f'{Path(output_dir) / image_path.stem}{output_extension}', 'w') as fw:
            for _, row in tags_filtered.iterrows():
                if row['category'] == CHARACTER_CATEGORY and row['Score'] > 0.9:
                    character = row['name']
                    character_folder = __removeIllegalChars(character)
                    if '_(' in character_folder:
                        character_folder = character_folder.split('_(')[0]
                    # create subfolder for character
                    target_dir = os.path.join(output_dir, character_folder)
                    skip = False
                    character_prefix += f"[character_{row['name']}:{row['Score']:0.2f}], "

                    if '(' in character:
                        character = character.split('(', 1)[0].strip(' ')
                    if not character in character_list:
                        character_list.append(character)
                else:
                    if 'boy' in row['name'] or 'girl' in row['name']:
                        gender_prefix += f"[gender_{row['name']}:{row['Score']:0.2f}], "
                        human_count +=1
                    else:
                        content += f"[{row['name']}:{row['Score']:0.2f}], "
            
            if skip:
                # print(f"Skipping {file} because no character is detected.")
                continue
            content = f"{gender_prefix}{character_prefix}{content}"
            # print('target_dir',target_dir)

            if len(character_list) == 2 or human_count == 2:
                target_dir = os.path.join(output_dir, two_character_dir)
            if len(character_list) > 2 or human_count > 2:
                target_dir = os.path.join(output_dir, multiple_character_dir)

            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
            if not os.path.exists(target_dir):
                print("Error: target_dir doesn't exist")
            filename, file_extension = os.path.splitext(os.path.basename(file))
            # save content to target dir
            with open(f'{target_dir}/{filename}{output_extension}', 'w') as f:
                f.write(content)
            # copy image to target dir
            shutil.copy(image_path, os.path.join(target_dir,f"{filename}{file_extension}"))
            # print(file)
    
    # os.rename(input_dir + '/' + file, input_dir + '/' + file.replace(' ', '_'))