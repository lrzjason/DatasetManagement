"""
This is a demo for using CogAgent and CogVLM in CLI
Make sure you have installed vicuna-7b-v1.5 tokenizer model (https://huggingface.co/lmsys/vicuna-7b-v1.5), full checkpoint of vicuna-7b-v1.5 LLM is not required.
In this demo, We us chat template, you can use others to replace such as 'vqa'.
Strongly suggest to use GPU with bfloat16 support, otherwise, it will be slow.
Mention that only one picture can be processed at one conversation, which means you can not replace or insert another picture during the conversation.
"""

import argparse
import torch

from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
import os
import time
import re

def convert_character_tags(text):
    # split the text by ','
    characters = text.split(',')
    # create an empty list to store the converted strings
    result = []
    # loop through each character and game
    for char in characters:
        if '(' in char:
            # split the char by '(' and ')'
            parts = char.split('(')
            # get the character name from the first part
            character = parts[0].strip()
            # get the game name from the second part
            game = parts[1].split(')')[0].strip()
            # format the string and append it to the result list
            result.append(f'{character} from {game}')
        else:
            result.append(char)
    # return the result list
    return result

# model_default = "THUDM/cogagent-chat-hf"
model_default = "THUDM/cogvlm-chat-hf"
parser = argparse.ArgumentParser()
parser.add_argument("--quant", choices=[4], type=int, default=4, help='quantization bits')
parser.add_argument("--from_pretrained", type=str, default=model_default, help='pretrained ckpt')
parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
parser.add_argument("--force_words", type=str, default="", help='force output words')
parser.add_argument("--bad_words", type=str, default="", help='don\'t want to say these words')
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--bf16", action="store_true")

args = parser.parse_args()
MODEL_PATH = args.from_pretrained
TOKENIZER_PATH = args.local_tokenizer
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
if args.bf16:
    torch_type = torch.bfloat16
else:
    torch_type = torch.float16

print("========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE))

if args.quant:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        trust_remote_code=True
    ).eval()
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=args.quant is not None,
        bnb_4bit_compute_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(DEVICE).eval()

# add any transformers params here.
gen_kwargs = {
    'min_new_tokens':100,
    'max_new_tokens':350,
    'num_beams':1,
    'length_penalty':1,
    'top_k':60,
    'top_p':0.6,
    'repetition_penalty': 1.15,
    'no_repeat_ngram_size':0,
    "do_sample": True,
    "temperature": 0.6,
} 
# gen_kwargs = {
#     'min_new_tokens':10,
#     'max_new_tokens':200,
#     'num_beams':2,
#     'length_penalty':1,
#     'top_k':50,
#     'top_p':1,
#     'repetition_penalty': 1,
#     'no_repeat_ngram_size':3,
#     "do_sample": True,
#     "temperature": 0.3
# } 

# text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"
# image_path = 'F:/ImageSet/hagrid_filter/call_2048/clear/00f1315f-b17e-48e2-9464-7c36b025ad4d.jpg'

# input_dir = 'F:/ImageSet/hagrid_filter'
# input_dir = 'F:/ImageSet/anime_dataset/genshin_classified'
# input_dir = 'F:/ImageSet/anime_dataset/genshin_test'

# input_dir = 'F:/ImageSet/training_script_cartoon_dool/train'
# input_dir =  "F:/ImageSet/vit_train/hand-classifier"
input_dir = "F:/ImageSet/openxl2_realism_test_output/image_cog"

subsets = os.listdir(input_dir)
total_subset = len(subsets)
count_subset = 0
# loop all sub dir in input_dir

# print the start time in yyyy-mm-dd hh:mm:ss format
start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

print(f'Start time: {start_time}')

# skip_subset = ['aether', 'albedo', 'amber', 'arataki_itto', 'barbara', 'beidou', 'boo_tao', 'c.c', 'chongyun', 'diluc', 'eula', 'fischl', 'fu_hua', 'ganyu', 'gorou', 'hatsune_miku', 'hu_tao', 'illustrious', 'jean', 'kaedehara_kazuha', 'kaeya', 'kamisato_ayaka', 'kamisato_ayato', 'keqing', 'kiana_kaslana', 'klee', 'kujou_sara', 'kuki_shinobu', 'lisa', 'lumine', 'mona', 'multiple_character', 'nahida', 'nanashi_mumei', 'nilou', 'ningguang', "ninomae_ina'nis", 'noelle', 'ouro_kronii', 'paimon', 'pyra', 'qiqi', 'raiden_mei', 'raiden_shogun', 'razor', 'rosaria', 'sangonomiya_kokomi', 'scaramouche', 'seele_vollerei', 'shenhe', 'slime', 'sucrose', 'tartaglia', 'tennouji_rina', 'thoma', 'two_character', 'usada_pekora', 'venti', 'xiangling', 'xiao', 'xingqiu', 'yae_miko', 'yae_sakura', 'yanfei', 'yelan', 'yoimiya', 'yorha_no._2_type_b', 'zhongli']
# skip_subset = ['aether', 'albedo', 'amber', 'arataki_itto', 'barbara', 'beidou', 'boo_tao', 'c.c', 'chongyun', 'diluc', 'eula', 'fischl', 'fu_hua', 'ganyu', 'gorou', 'hatsune_miku', 'hu_tao', 'illustrious', 'jean', 'kaedehara_kazuha', 'kaeya', 'kamisato_ayaka', 'kamisato_ayato', 'keqing', 'kiana_kaslana', 'klee', 'kujou_sara', 'kuki_shinobu', 'lisa', 'lumine', 'mona', 'multiple_character', 'nahida', 'nanashi_mumei', 'nilou']
# skip_subset = ['aether', 'albedo', 'amber', 'arataki_itto', 'barbara', 'beidou', 'boo_tao', 'c.c', 'chongyun', 'diluc', 'eula', 'fischl', 'fu_hua', 'ganyu', 'gorou', 'hatsune_miku', 'hu_tao', 'illustrious', 'jean', 'kaedehara_kazuha', 'kaeya', 'kamisato_ayaka', 'kamisato_ayato', 'keqing', 'kiana_kaslana', 'klee', 'kujou_sara', 'kuki_shinobu', 'lisa', 'lumine', 'mona', 'multiple_character', 'nahida', 'nanashi_mumei', 'nilou', 'ningguang', "ninomae_ina'nis", 'noelle', 'ouro_kronii', 'paimon', 'pyra', 'qiqi', 'raiden_mei', 'raiden_shogun']
# skip_subset = ['aether', 'albedo', 'amber', 'arataki_itto', 'barbara', 'beidou', 'boo_tao', 'c.c', 'chongyun', 'diluc', 'eula', 'fischl', 'fu_hua', 'ganyu', 'gorou', 'hatsune_miku', 'hu_tao', 'illustrious', 'jean', 'kaedehara_kazuha', 'kaeya', 'kamisato_ayaka', 'kamisato_ayato', 'keqing', 'kiana_kaslana', 'klee', 'kujou_sara', 'kuki_shinobu', 'lisa', 'lumine', 'mona', 'multiple_character', 'nahida', 'nanashi_mumei', 'nilou', 'ningguang', "ninomae_ina'nis", 'noelle', 'ouro_kronii', 'paimon', 'pyra', 'qiqi', 'raiden_mei']

skip_subset = []

# image_ext = '.jpg'
image_ext = '.webp'

for subset_dir in subsets:
    count_subset+=1
    if subset_dir in skip_subset:
        print(f'skip subset_dir: {count_subset}/{total_subset} {subset_dir}')
        continue
    print(f'processing subset_dir: {count_subset}/{total_subset} {subset_dir}')
    character_dir = os.path.join(input_dir, subset_dir)
    images = os.listdir(character_dir)

    # list to store files
    images = []
    # Iterate directory
    for f in os.listdir(character_dir):
        # check only text files
        if f.endswith(image_ext):
            images.append(f)
    total_image = len(images)
    count_image = 0
    # loop all image in clear dir
    for image_name in images:
        count_image+=1
        print(f'processing image: {count_image}/{total_image} {image_name}')
        image_path = os.path.join(character_dir, image_name)
        print(f'Processing {image_path}')
    
        filename = os.path.splitext(os.path.basename(image_path))[0]
        filename, ext = os.path.splitext(image_name)

        file_name = image_name.split('.')[0]
        old_txt_file_path = os.path.join(character_dir, file_name+'.txt')
        txt_file_path = os.path.join(character_dir, filename+'.txt')
        print(txt_file_path)

        if os.path.exists(old_txt_file_path):
            print(f'{old_txt_file_path} already exists, skip')
            # rename the wrongly named file
            os.rename(old_txt_file_path, txt_file_path)
            continue

        image = Image.open(image_path).convert('RGB')

        # prompt = f'Describe the image precisely, detailing every element, interaction and background. Include composition, angle and perspective. Use only facts and concise language; avoid interpretations or speculation:'
        prompt = f'Brief the image as simple as possible'
        starts_with = f'The image showcases '
        query = f'Question: {prompt} Answer: {starts_with}'
        history = []
        input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=history, images=[image])

        prepare_images = []
        if gen_kwargs['num_beams'] > 1:
            prepare_images = [[input_by_model['images'][0].to(DEVICE).to(torch_type)] for _ in range(gen_kwargs['num_beams'])]
        else:
            prepare_images = [[input_by_model['images'][0].to(DEVICE).to(torch_type)]] if image is not None else None
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': prepare_images,
        }
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]

        force_words_ids = None
        force_words = []
        # if args.force_words is not None and args.force_words!="":
        #     force_words = args.force_words.split(",") if args.force_words is not None else []
        # else:
        #   force_words = convert_character_tags(character_name)
        force_words_ids = tokenizer(force_words, add_special_tokens=False)["input_ids"] if force_words else []
        # print(f"** force_words: {force_words}")

        bad_words_ids = None
        # if args.bad_words is not None and args.bad_words!="":
        #     bad_words = args.bad_words.split(",") if args.bad_words is not None else []
        # else:
        bad_words = []
        bad_words_ids = tokenizer(bad_words, add_special_tokens=False)["input_ids"] if bad_words else []

        # print(f"** bad_words: {bad_words}")
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs
                                #      , force_words_ids=force_words_ids,
                                #  bad_words_ids=bad_words_ids
                                 )
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0])
            response = response.split("</s>")[0]
            print(response)
            with open(txt_file_path, 'w', encoding="utf-8") as f:
                f.write(response)
                f.close()
                print(f'write {file_name}.txt')
        # break
    # run one subset for test
    # break

# replace twitter username
# replace , and there are no discernible dialogue or narrative elements.

# print the start time in yyyy-mm-dd hh:mm:ss format
end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

print(f'End time: {end_time}')
