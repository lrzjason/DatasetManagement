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
import re

CHARACTER_CATEGORY = 4
TAG_ONLY = True

REMOVE_EXIST = False

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

def run_model(image_path, model_path, tags_path, session, tag_threshold, filter_tags):
    image = Image.open(image_path)
    processed_image = preprocess_image(image)
    result = session.run(None, {session.get_inputs()[0].name: processed_image})[0]
    tags = pd.read_csv(tags_path)
    tags.reset_index(inplace=True)
    result_df = pd.DataFrame(result[0], columns=['Score'])
    result_with_tags = pd.concat([tags, result_df], axis=1)
    tags_filtered = result_with_tags[['name', 'Score', 'category']]
    tags_filtered = tags_filtered[~tags_filtered['name'].isin(filter_tags)]
    # print('tags_filtered')
    # print(tags_filtered)
    return tags_filtered
    
def main(image_folder, model_repo_ids, tag_threshold, filter_tags, stack_models):
    sessions = []
    tags_paths = []

    if stack_models:
        model_repo_ids = ['SmilingWolf/wd-v1-4-convnext-tagger-v3', 'SmilingWolf/wd-v1-4-vit-tagger-v3',
                          'SmilingWolf/wd-v1-4-swinv2-tagger-v3']

    for model_repo_id in model_repo_ids:
        print('*****************')
        print(model_repo_id)
        # Download the model and tags file
        model_path, tags_path = download_model_files(model_repo_id)

        try:
            session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        except RuntimeException:
            print("CUDA isn't available. Trying to run on CPU.")
            try:
                session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            except RuntimeException:
                print("Can't run the model. Exiting.")
                return

        sessions.append(session)
        tags_paths.append(tags_path)

    image_files = list(Path(image_folder).rglob('*'))
    image_files = [img for img in image_files if img.suffix in ['.jpg', '.jpeg', '.png', '.webp']]
    pbar =  tqdm(image_files, desc="Processing images")
    for image_path in pbar:
        textfile = f'{Path(image_folder) / image_path.stem}{args.output_extension}'
        if os.path.exists(textfile):
            if REMOVE_EXIST:
                os.remove(textfile)
                tqdm.write('remove: ' + str(textfile) )
            else:
                tqdm.write('skipping: ' + str(textfile) )
                pbar.update(1)
                continue
        tags_scores = []
        for session, tags_path in zip(sessions, tags_paths):
            # print(tags_path)
            tags_filtered = run_model(image_path, model_path, tags_path, session, tag_threshold, filter_tags)
            tags_scores.append(tags_filtered.set_index('name'))
            # print(tags_filtered.sort_values('Score', ascending=False).head(10))

        if not stack_models and len(model_repo_ids) == 1:
            averaged_tags_scores = tags_scores[0].reset_index()
        else:
            combined_tags_scores = pd.concat(tags_scores, axis=1, join='outer')
            averaged_tags_scores = combined_tags_scores.mean(axis=1).reset_index()

        averaged_tags_scores.columns = ['name', 'Score', 'category']  # rename columns
        averaged_tags_scores = averaged_tags_scores[averaged_tags_scores['Score'] > tag_threshold]
        averaged_tags_scores.sort_values('Score', ascending=False, inplace=True)
        print(averaged_tags_scores)
        with open(textfile, 'w', encoding='utf-8') as fw:
            for _, row in averaged_tags_scores.iterrows():
                if TAG_ONLY:
                    tag = row['name'].replace('_',' ')
                    if tag == '1girl':
                        tag = 'woman'
                    if tag == '1boy':
                        tag = 'man'
                    fw.write(f"{tag}, ")
                else:
                    if row['category'] == CHARACTER_CATEGORY:
                        fw.write(f"[characeter_{row['name']}: {row['Score']:.2f}], ")
                    else:
                        fw.write(f"[{row['name']}: {row['Score']:.2f}], ")
                



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_directory", help="The folder with the images to process.")
    
    parser.add_argument("--model_repo_id", default=['SmilingWolf/wd-swinv2-tagger-v3'], nargs='+', help="The model repo ids.")
    parser.add_argument("--threshold", type=float, default=0.5, help="wd14 tag confidence threshold")
    parser.add_argument("--output_extension", type=str, default=".txt", help="file extension to save caption with")
    parser.add_argument("--filter", nargs='*', default=['1girl','solo','questionable','general','sensitive'], help="List of tags to filter out.")
    parser.add_argument("--stack_models", action='store_true', help="Whether to stack models. If set, images will be processed with multiple models and their scores averaged.")
    args = parser.parse_args()

    # main(args.input_directory, args.model_repo_id, args.threshold, args.filter, args.stack_models)

    # input_directory = 'F:/ImageSet/Fasion_dataset/all'
    # input_directory = 'F:/ImageSet/openxl2_reg/1_fashion_photo'
    # input_directory = 'F:/ImageSet/openxl2_realism/centered'
    # input_directory =  'F:/ImageSet/hands_dataset_above_average/clean-hands'
    # input_directory = 'F:/ImageSet/handpick_high_quality/cdrama scene'
    input_directory = 'F:/ImageSet/handpick_high_quality_b2_cropped'
    args.filter = ['solo','questionable','general','sensitive']
    args.output_extension = '.wd14'
    for subdir in os.listdir(input_directory):
        subdir_path = os.path.join(input_directory, subdir)
        main(subdir_path, args.model_repo_id, args.threshold, args.filter, args.stack_models)
