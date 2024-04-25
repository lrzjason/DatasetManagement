# target to automatically drop the tags which make low clip similarity

# Import the modules
import os
import json
import clip_sim
import torch
from tqdm import tqdm
import shutil
import numpy as np

def main():
    input_dir = "F:/ImageSet/openxl2_realism_test"
    result_json = "F:/ImageSet/openxl2_realism_test/tags/autodrop_result.json"
    image_ext = '.webp'
    caption_ext = ['.txt','.wd14_cap']
    seperator = ','
    preserved_tag_num = 3
    # read json file
    # with open(result_json, "r") as f:
    #     result = json.load(f)
    
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip_sim.init_model(device)

    for subdir in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, subdir)
        # Loop through the folder and append the image paths to the list
        for file in os.listdir(folder_path):
            filename,ext = os.path.splitext(file)
            if ext in caption_ext:
                file_path = os.path.join(folder_path, file)
                image_path = os.path.join(folder_path, f'{filename}{image_ext}')
                image_features = clip_sim.get_image_feature(model, preprocess, image_path)

                content = open(file_path, "r", encoding="utf-8").read()
                content = content.replace("\n", " ").strip(' ')
                ori_score = clip_sim.cal_similarity(model,preprocess,content,image_features=image_features)
                print(f'{filename}: {ori_score}')
                if content[-1] == seperator:
                    content = content[:-1]
                tags = content.split(seperator)
                preserved_tags = tags[:preserved_tag_num]
                print(f"preserved_tag: {preserved_tags}")
                tags = tags[preserved_tag_num:]
                print(f"skipped tags: {tags}")

                print(f"tags len: {len(tags)}")
                tags_score = {}
                for tag in tags:
                    tag = tag.strip()
                    score = clip_sim.cal_similarity(model,preprocess,tag,image_features=image_features)
                    tags_score[tag] = round(score.item(),4)
                # for idx,tag in enumerate(tags):
                #     tags_except_idx = [x for i,x in enumerate(tags) if i!=tags]
                #     new_tags = ','.join(tags_except_idx)

                print(tags_score)

                # select top twity tags from tags_score
                top_tags = sorted(tags_score, key=tags_score.get, reverse=True)
                adjusted_tags = preserved_tags + top_tags

                new_content = ','.join(adjusted_tags)
                print('new_content',new_content)
                new_score = clip_sim.cal_similarity(model,preprocess,new_content,image_features=image_features)
                
                print(f"ori_score:{ori_score} new_score:{new_score}")

if __name__ == '__main__':
    main()
