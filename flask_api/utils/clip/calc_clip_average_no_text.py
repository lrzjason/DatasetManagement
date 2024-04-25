

# Import the modules
import os
import json
import clip_sim
import torch
from tqdm import tqdm
import shutil
import numpy as np
from PIL import Image

def detect_black_border(image_path):
    with Image.open(image_path) as img:
        # Convert image to grayscale
        img_gray = img.convert('L')
        
        # Get image size
        width, height = img_gray.size
        
        # Define the threshold for black color
        black_threshold = 20
        
        # Check the borders
        top_border = all(img_gray.getpixel((x, 0)) < black_threshold for x in range(width))
        bottom_border = all(img_gray.getpixel((x, height - 1)) < black_threshold for x in range(width))
        left_border = all(img_gray.getpixel((0, y)) < black_threshold for y in range(height))
        right_border = all(img_gray.getpixel((width - 1, y)) < black_threshold for y in range(height))
        
        # Return True if all borders are black
        return top_border or bottom_border or left_border or right_border


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
    input_dir = "F:/ImageSet/hands_dataset"
    above_average_dir = "F:/ImageSet/hands_dataset_above_average"
    underscore_dir = "F:/ImageSet/hands_dataset_underscore"

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

    # image_ext = '.webp'
    image_ext = '.jpg'
    caption_ext = ['.txt','.wd14_cap']

    empty_text_list = []
    missing_text_list = []
    files_list = []
    total_clip_score = 0
    total_files = 0
    subdirs = os.listdir(input_dir)

    # read clip_score.json
    clip_score_dict = {}
    if os.path.exists('clip_score.json'):
        with open('clip_score.json', 'r') as f:
            clip_score_dict = json.load(f)
    else:
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
                        text = "hand, anime style"
                        filename,_ = os.path.splitext(file)
                        # for ext in caption_ext:
                        #     # Join the folder path and the file name to get the full path
                        #     text_file = file.replace(image_ext, ext)
                        #     text_path = os.path.join(folder_path, text_file)
                        #     if not os.path.exists(text_path):
                        #         missing_text_list.append(text_path)
                        #         continue
                        #     # open text file and read content
                        #     with open(text_path, 'r',encoding='utf-8') as f:
                        #         text = f.read()
                        #     if text == "":
                        #         empty_text_list.append(text_path)
                        #         continue
                        # image_path = "C:/Users/Administrator/Desktop/test.jpg"
                        # image_path = "F:/ImageSet/openxl2_realism/tags/fischl_5599508_preserved.webp"
                        image_path = os.path.join(folder_path, file)
                        
                        # score = clip_sim.cal_similarity(model,preprocess,text,image_path=image_path)
                        text_segement = split_text(text)
                        scores,_,_ = clip_sim.classify(model,preprocess,text_segement,image_path)
                        values = scores.values()
                        score = sum(values)/len(values)
                        score = round(score.item(),4)
                        result = {
                            "text":text,
                            "subdir": subdir,
                            "file_name":filename,
                            "image_ext":image_ext,
                            # "caption_ext":ext,
                            "image_path": image_path,
                            # "text_path": text_path,
                            "score": score
                        }
                        print(result)
                        files_list.append(result)
                        total_clip_score += score
                        total_files +=1

        average_score = total_clip_score/total_files

        clip_score_dict["average_score"] = average_score
        clip_score_dict["files"] = files_list

    # dump json to file
    with open('clip_score.json', 'w',encoding='utf-8') as f:
        json.dump(clip_score_dict, f, indent=4)

    
    files_list = clip_score_dict["files"]
    average_score = clip_score_dict['average_score']
    print(f"Average clip score: {average_score}")
    underscore_files_list = []
    above_average_files_list = []
    for file in files_list:
        file['black_border'] = detect_black_border(file["image_path"])
        if file["score"] < average_score or file['black_border']:
            # create subdir in underscore_dir
            subdir = os.path.join(underscore_dir, file["subdir"])
            underscore_files_list.append(file)
        else:
            # create subdir in above_average_dir
            subdir = os.path.join(above_average_dir, file["subdir"])
            above_average_files_list.append(file)
        
        # copy image to subdir
        output_file = os.path.join(subdir,f'{file["file_name"]}{file["image_ext"]}')
        if not os.path.exists(output_file):
            shutil.copy(file["image_path"], output_file)
        # copy text to subdir
        # shutil.copy(file["text_path"], os.path.join(subdir,f'{file["file_name"]}{file["caption_ext"]}'))
    
    clip_score_dict["files"] = files_list
    
    # dump json to file
    with open('clip_score.json', 'w',encoding='utf-8') as f:
        json.dump(clip_score_dict, f, indent=4)

    # average_above_average_score = 0
    if len(above_average_files_list) > 0:
        print(f'{len(above_average_files_list)} files above average of {len(files_list)}')
        average_above_average_score = sum([file["score"] for file in above_average_files_list]) / len(above_average_files_list)
        print(f'Average score of above_average files: {average_above_average_score}')

    average_underscore = 0
    if len(underscore_files_list) > 0:
        print(f'{len(underscore_files_list)} files above average of {len(files_list)}')
        average_underscore = sum([file["score"] for file in underscore_files_list]) / len(underscore_files_list)
        print(f'Average score of underscore files: {average_underscore}')

    new_underscore_files_list = []
    # find files that are above average_underscore
    for idx,file in enumerate(underscore_files_list):
        if file["score"] > average_underscore and not file['black_border']:
            # create subdir in above_average_dir
            subdir = os.path.join(above_average_dir, file["subdir"])
            # copy image to subdir
            output_file = os.path.join(subdir,f'{file["file_name"]}{file["image_ext"]}')
            if not os.path.exists(output_file):
                shutil.copy(file["image_path"], os.path.join(subdir,f'{file["file_name"]}{file["image_ext"]}'))

            # add item to above_average_files_list
            above_average_files_list.append(file)

            ori_subdir = os.path.join(underscore_dir, file["subdir"])
            ori_file = os.path.join(ori_subdir,f'{file["file_name"]}{file["image_ext"]}')
            if os.path.exists(ori_file):
                os.remove(ori_file)
        else:
            new_underscore_files_list.append(file)

    underscore_files_list = new_underscore_files_list
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
