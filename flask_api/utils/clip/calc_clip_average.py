

# Import the modules
import os
import json
import clip_sim
import torch
from tqdm import tqdm
import shutil
import numpy as np

def split_text(text):
    results = text.split(".")
    double_split = []
    for splited_text in results:
        if len(splited_text)>77:
            double_split += splited_text.split(",")
        else:
            double_split.append(splited_text)
    return double_split

def main():
    input_dir = "F:/ImageSet/openxl2_realism"
    above_average_dir = "F:/ImageSet/openxl2_realism_above_average"
    underscore_dir = "F:/ImageSet/openxl2_realism_underscore"

    # create above_average dir
    if not os.path.exists(above_average_dir):
        os.mkdir(above_average_dir)
    
    # create underscore dir
    if not os.path.exists(underscore_dir):
        os.mkdir(underscore_dir)
    
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip_sim.init_model(device)
    # print('model')
    # print(model)

    # suffix = ''
    # clip_targets = ['cinematic photo','anime artwork']

    image_ext = '.webp'
    caption_ext = ['.txt','.wd14_cap']

    empty_text_list = []
    missing_text_list = []
    files_list = []
    total_clip_score = 0
    total_files = 0
    subdirs = os.listdir(input_dir)
    for subdir in subdirs:
        folder_path = os.path.join(input_dir, subdir)
        
        above_average_subdir = os.path.join(above_average_dir, subdir)
        if not os.path.exists(above_average_subdir):
            os.makedirs(above_average_subdir)

        underscore_subdir = os.path.join(underscore_dir, subdir)
        if not os.path.exists(underscore_subdir):
            os.makedirs(underscore_subdir)

        
        if os.path.isdir(folder_path):
            # Loop through the folder and append the image paths to the list
            for file in tqdm(os.listdir(folder_path)):
                # Check if the file is an image by its extension
                if file.endswith((image_ext)):
                    text = ""
                    for ext in caption_ext:
                        filename,_ = os.path.splitext(file)
                        # Join the folder path and the file name to get the full path
                        text_file = file.replace(image_ext, ext)
                        text_path = os.path.join(folder_path, text_file)
                        if not os.path.exists(text_path):
                            missing_text_list.append(text_path)
                            continue
                        # open text file and read content
                        with open(text_path, 'r',encoding='utf-8') as f:
                            text = f.read()
                        if text == "":
                            empty_text_list.append(text_path)
                            continue
                    # image_path = "C:/Users/Administrator/Desktop/test.jpg"
                    # image_path = "F:/ImageSet/openxl2_realism/tags/fischl_5599508_preserved.webp"
                    image_path = os.path.join(folder_path, file)
                    
                    # score = clip_sim.cal_similarity(model,preprocess,text,image_path=image_path)
                    text_segement = split_text(text)
                    scores,_,_ = clip_sim.classify(model,preprocess,text_segement,image_path)
                    values = scores.values()
                    score = sum(values)/len(values)
                    score = round(score.item(),2)
                    result = {
                        "subdir": subdir,
                        "file_name":filename,
                        "image_ext":image_ext,
                        "caption_ext":ext,
                        "image_path": image_path,
                        "text_path": text_path,
                        "score": score
                    }
                    print(result)
                    files_list.append(result)
                    total_clip_score += score
                    total_files +=1

    average_score = total_clip_score/total_files
    print(f"Average clip score: {average_score}")
    underscore_files_list = []
    above_average_files_list = []
    for file in files_list:
        if file["score"] < average_score:
            # create subdir in underscore_dir
            subdir = os.path.join(underscore_dir, file["subdir"])
            underscore_files_list.append(file)
            
        else:
            # create subdir in above_average_dir
            subdir = os.path.join(above_average_dir, file["subdir"])
            above_average_files_list.append(file)
        
        # copy image to subdir
        shutil.copy(file["image_path"], os.path.join(subdir,f'{file["file_name"]}{file["image_ext"]}'))
        # copy text to subdir
        shutil.copy(file["text_path"], os.path.join(subdir,f'{file["file_name"]}{file["caption_ext"]}'))
    
    average_above_average_score = 0
    if len(above_average_files_list) > 0:
        print(f'{len(above_average_files_list)} files above average of {len(files_list)}')
        average_above_average_score = sum([file["score"] for file in above_average_files_list]) / len(above_average_files_list)
        print(f'Average score of above_average files: {average_above_average_score}')

    average_underscore = 0
    if len(underscore_files_list) > 0:
        print(f'{len(underscore_files_list)} files above average of {len(files_list)}')
        average_underscore = sum([file["score"] for file in underscore_files_list]) / len(underscore_files_list)
        print(f'Average score of underscore files: {average_underscore}')

    diff_above = abs(average_score - average_above_average_score)
    print('average_score - average_above_average_score',average_score,average_above_average_score,diff_above)
    
    diff_under = abs(average_score - average_underscore)
    print('average_score - average_underscore',average_score,average_underscore,diff_under)

    results = {
        'total_files':len(files_list),
        'total_above':len(above_average_files_list),
        'total_under':len(underscore_files_list),
        'average_score': average_score,
        'average_above_average_score': average_above_average_score,
        'diff_above':diff_above,
        'average_underscore': average_underscore,
        'diff_under':diff_under,
        'underscore_files_list':underscore_files_list,
        'above_average_files_list':above_average_files_list
    }

    output_file = os.path.join(above_average_dir,"results.json")
    # dump the results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
            
                    

if __name__ == '__main__':
    main()
