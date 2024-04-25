

# Import the modules
import os
import json
import clip_sim
import torch
from tqdm import tqdm

def main():
    input_dir = "F:/ImageSet/openxl2_realism"
    
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip_sim.init_model(device)

    suffix = ''
    clip_targets = ['cinematic photo','anime artwork']

    image_ext = '.webp'
    caption_ext = ['.txt','.wd14_cap']

    for subdir in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, subdir)
        # Loop through the folder and append the image paths to the list
        for file in tqdm(os.listdir(folder_path)):
            # Check if the file is an image by its extension
            if file.endswith((image_ext)):
                for ext in caption_ext:
                    # Join the folder path and the file name to get the full path
                    txt_file = file.replace(image_ext, ext)
                    full_path = os.path.join(folder_path, txt_file)
                    if not os.path.exists(full_path):
                        continue

                    # image_path = "C:/Users/Administrator/Desktop/test.jpg"
                    # image_path = "F:/ImageSet/openxl2_realism/tags/fischl_5599508_preserved.webp"
                    image_path = os.path.join(folder_path, file)
                    # print(image_path)
                    prefix = ''
                    quality_targets = ['low quality','sketch','highly detailed','breathtaking','masterpiece']
                    _,classified_quality,quality_score = clip_sim.classify(model,preprocess,quality_targets,image_path)

                    prefix += classified_quality+' '

                    type_targets = ['anime artwork','cosplay photo','cinematic photo','film','35mm photograph','raw photo','digital illustration']
                    scores,classified_type,type_score = clip_sim.classify(model,preprocess,type_targets,image_path)

                    prefix += classified_type

                    content = ''
                    if os.path.exists(full_path):
                        # Append the full path to the list
                        with open(full_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            f.close()
                    else:
                        with open(full_path, "w", encoding="utf-8") as f:
                            f.close()
                    # prevent double prefix
                    if not content.startswith(prefix):
                        content = f'{prefix}, {content}'
                        if len(suffix)>0:
                            content =  content + ', ' + suffix
                        with open(full_path, "r+", encoding="utf-8") as out_f:
                            out_f.write(content)
                # print('scores',scores)
                # print('classified_type,type_score',classified_type,type_score)
                
                # print('classified_quality,quality_score',classified_quality,quality_score)
                # break

if __name__ == '__main__':
    main()
