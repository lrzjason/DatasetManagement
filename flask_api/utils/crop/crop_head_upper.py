import os
import shutil
from PIL import Image

from imgutils.detect import detect_halfbody,detect_heads
from imgutils.tagging import get_wd14_tags,tags_to_text

def crop_dir(input_dir,character,target_crop_dir,crop_configs):
    viewpoint_subset_path = os.path.join(input_dir,f'{character}_{target_crop_dir}')
    for file in os.listdir(viewpoint_subset_path):
        if not file.endswith('.jpg'):
            continue
        image_path = os.path.join(viewpoint_subset_path,file)
        for crop_config in crop_configs:
            target_viewpoint = crop_config['target_viewpoint']
            dir_path = os.path.join(input_dir, f'{character}_{target_viewpoint}')
            output_viewpoint_dir = os.path.join(output_dir,f'{character}_{target_viewpoint}')
            if not os.path.exists(output_viewpoint_dir):
                os.makedirs(output_viewpoint_dir)
            if int(len(os.listdir(output_viewpoint_dir))/2) > 100:
                break
            if int(len(os.listdir(dir_path))/4) < 100:
                crop(output_dir,image_path,crop_config['detect_fn'],target_viewpoint)

def crop(output_dir,image_path,detect_fn,target_viewpoint):
    # print('handling crop')
    img = Image.open(image_path) # load the image
    file = os.path.basename(image_path)
    result = detect_fn(img)
    if result is None or len(result) == 0:
        return
    box,label,prob = result[0]  # detect it
    if box is not None:
        img2 = img.crop(box) # crop the image
        # create the output directory
        output_viewpoint_dir = os.path.join(output_dir,f'{character}_{target_viewpoint}')
        if not os.path.exists(output_viewpoint_dir):
            os.makedirs(output_viewpoint_dir)
        img2.save(os.path.join(output_viewpoint_dir,file))
        rating, features, chars = get_wd14_tags(img2)
        tag_content = tags_to_text(features, use_spaces=True)
        tag_content = f'{target_viewpoint}, {character}, {tag_content}'
        txt_file = os.path.join(output_viewpoint_dir,file.replace('.jpg','.txt'))
        print(f'writing {txt_file}: {tag_content}')
        with open(txt_file, "w", encoding="utf-8") as out_f:
            out_f.write(tag_content)


character_dir = 'F:/ImageSet/anime_dataset/genshin_classified'

input_dir = 'F:/ImageSet/anime_dataset/genshin_classified_with_viewpoint'
# input_dir = 'F:/ImageSet/anime_dataset/test'

output_dir = 'F:/ImageSet/anime_dataset/genshin_classified_with_viewpoint_crop'

dir_info = {}

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

max_folder = 0
for dir_name in os.listdir(input_dir):
    dir_path = os.path.join(input_dir, dir_name)
    dir_info[dir_name] = int(len(os.listdir(dir_path))/4)
    if dir_info[dir_name] > max_folder:
        max_folder = dir_info[dir_name]
        max_folder_name = dir_name
dir_info['max'] = max_folder
dir_info['max_name'] = max_folder_name
print(dir_info)

skip_subset = ['aether', 'albedo', 'amber', 'arataki_itto', 'barbara', 'beidou', 'boo_tao', 'c.c', 'chongyun', 'diluc', 'eula', 'fischl', 'fu_hua']


character_list = os.listdir(character_dir)
for character in character_list:
    if character in skip_subset:
        continue
    # crop fullbody to halfbody
    crop_configs = [
        {'target_viewpoint':'upperbody','detect_fn':detect_halfbody},
        {'target_viewpoint':'head_only','detect_fn':detect_heads}
    ]
    crop_viewpoints = ['fullbody','knee_level']
    for crop_viewpoint in crop_viewpoints:
        print(f'handling {character} {crop_viewpoint}')
        crop_dir(input_dir,character,crop_viewpoint,crop_configs)
    # viewpoint_subset_path = os.path.join(input_dir,f'{character}_fullbody')
    # for file in os.listdir(viewpoint_subset_path):
    #     if not file.endswith('.jpg'):
    #         continue
    #     image_path = os.path.join(viewpoint_subset_path,file)
    #     dir_path = os.path.join(input_dir, f'{character}_upperbody')
    #     if int(len(os.listdir(dir_path))/4) < 100:
    #         crop(output_dir,image_path,detect_halfbody,target_viewpoint='upperbody')
    #     dir_path = os.path.join(input_dir, f'{character}_head_only')
    #     if int(len(os.listdir(dir_path))/4) < 100:
    #         crop(output_dir,image_path,detect_heads,target_viewpoint='head_only')
    
    # break