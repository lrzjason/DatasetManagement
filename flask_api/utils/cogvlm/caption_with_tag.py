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
from tqdm import tqdm 


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
    'max_new_tokens':200,
    'num_beams':1,
    'length_penalty':1,
    'top_k':50,
    'top_p':1,
    'repetition_penalty': 1,
    'no_repeat_ngram_size':3,
    "do_sample": True,
    "temperature": 0.3
} 

# text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"
# image_path = 'F:/ImageSet/hagrid_filter/call_2048/clear/00f1315f-b17e-48e2-9464-7c36b025ad4d.jpg'

# input_dir = 'F:/ImageSet/hagrid_filter'
# input_dir = 'F:/ImageSet/anime_dataset/genshin_classified'
# input_dir = 'F:/ImageSet/anime_dataset/genshin_test'

input_dir = 'F:/ImageSet/openxl2_realism'

subsets = os.listdir(input_dir)
total_subset = len(subsets)
count_subset = 0
# loop all sub dir in input_dir

# print the start time in yyyy-mm-dd hh:mm:ss format
start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

print(f'Start time: {start_time}')

# skip_subset = ['aether', 'albedo', 'amber', 'arataki_itto', 'barbara', 'beidou', 'boo_tao', 'c.c', 'chongyun', 'diluc', 'eula', 'fischl', 'fu_hua', 'ganyu', 'gorou', 'hatsune_miku', 'hu_tao', 'illustrious', 'jean', 'kaedehara_kazuha', 'kaeya', 'kamisato_ayaka', 'kamisato_ayato', 'keqing', 'kiana_kaslana', 'klee', 'kujou_sara', 'kuki_shinobu', 'lisa', 'lumine', 'mona', 'multiple_character', 'nahida', 'nanashi_mumei', 'nilou', 'ningguang', "ninomae_ina'nis", 'noelle', 'ouro_kronii', 'paimon', 'pyra', 'qiqi', 'raiden_mei', 'raiden_shogun', 'razor', 'rosaria', 'sangonomiya_kokomi', 'scaramouche', 'seele_vollerei', 'shenhe', 'slime', 'sucrose', 'tartaglia', 'tennouji_rina', 'thoma', 'two_character', 'usada_pekora', 'venti', 'xiangling', 'xiao', 'xingqiu', 'yae_miko', 'yae_sakura', 'yanfei', 'yelan', 'yoimiya', 'yorha_no._2_type_b', 'zhongli']
# skip_subset = ['aether', 'albedo', 'amber', 'arataki_itto', 'barbara', 'beidou', 'boo_tao', 'c.c', 'chongyun', 'diluc', 'eula', 'fischl', 'fu_hua', 'ganyu', 'gorou', 'hatsune_miku', 'hu_tao', 'illustrious', 'jean', 'kaedehara_kazuha', 'kaeya', 'kamisato_ayaka', 'kamisato_ayato', 'keqing', 'kiana_kaslana', 'klee', 'kujou_sara', 'kuki_shinobu', 'lisa', 'lumine', 'mona', 'multiple_character', 'nahida', 'nanashi_mumei', 'nilou', 'ningguang', "ninomae_ina'nis", 'noelle', 'ouro_kronii', 'paimon', 'pyra', 'qiqi', 'raiden_mei','raiden_shogun', 'razor', 'rosaria', 'sangonomiya_kokomi', 'scaramouche', 'seele_vollerei', 'shenhe', 'slime', 'sucrose', 'tartaglia', 'tennouji_rina', 'thoma', 'two_character' ]

# skip_subset = ['noelle', 'ouro_kronii', 'paimon', 'pyra', 'qiqi', 'raiden_mei', 'raiden_shogun', 'razor', 'rosaria', 'sangonomiya_kokomi', 'scaramouche', 'seele_vollerei', 'shenhe', 'slime', 'sucrose', 'tartaglia', 'tennouji_rina', 'thoma', 'two_character', 'usada_pekora', 'venti', 'xiangling', 'xiao', 'xingqiu', 'yae_miko', 'yae_sakura', 'yanfei', 'yelan', 'yoimiya', 'yorha_no._2_type_b', 'zhongli']
skip_subset = []

def replacement(response):
    if 'named' in response:
        response = response.split('named')[1].strip(' ')
    replace_character_arr = ['an animated character',' animated character', 'an anime character', ' anime character', ' character', ' anime figure']
    for replace_character in replace_character_arr:
        if replace_character in response:
            response = response.replace(replace_character, f', {",".join(convert_character_tags(character_name))}')
    
    image_str = ' image '
    if image_str in response:
        index = response.find(image_str)+len(image_str)
        # base on this index, find next ' ' index
        next_index = response[index:].find(' ')
        # skip something like 'The image showcases ' or 'The image indicates ', etc
        response = response[index+next_index+1:]
    pic_str = ' picture of '
    if pic_str in response:
        index = response.find(pic_str)+len(pic_str)
        response = response[index:]
    
    pic_str = ' picture '
    if pic_str in response:
        index = response.find(pic_str)+len(pic_str)
        # base on this index, find next ' ' index
        next_index = response[index:].find(' ')
        # skip something like 'The image showcases ' or 'The image indicates ', etc
        response = response[index+next_index+1:]
    
    skip_start_arr = ['A, ','An, ', 'a, ', 'an, ', 'A ','a ','an ','An ', ', ', ' to be a ', 'to be a ']
    for skip_start in skip_start_arr:
        if response.startswith(skip_start):
            response = response[len(skip_start):]
            break

    
    replace_other_arr = [
        {
            'old': '.cartoon ',
            'new': ' '
        },
    ]
    for replacement in replace_other_arr:
        if replacement['old'] in response:
            response = response.replace(replacement['old'], replacement['new'])
    
    replace_arr = ['A, ','An, ', 'a, ', 'an, ', 'The, ','the, ','seemingly ','possibly ', ' likely','  ', 'animated ','cartoon, ', ' cartoon','cartoon ', ' what appears to be', ' seems to']
    for replacement in replace_arr:
        if replacement in response:
            response = response.replace(replacement, '')

    replace_other_arr = [
        {
            'old': 'appears to be',
            'new': 'is'
        },
        {
            'old': 'appears to me to be',
            'new': 'is'
        },
        {
            'old': 'appearing to be',
            'new': 'is'
        },
        {
            'old': 'appear to be',
            'new': 'are'
        },
        {
            'old': 'seems to be',
            'new': 'is'
        },
        {
            'old': 'seem to be',
            'new': 'are'
        },
        {
            'old': 'seems',
            'new': 'is'
        },
        {
            'old': 'group of,, ',
            'new': 'group of characters, '
        },
    ]
    for replacement in replace_other_arr:
        if replacement['old'] in response:
            response = response.replace(replacement['old'], replacement['new'])
    pure_name = character_name
    if '(' in character_name:
        pure_name = character_name.split('(')[0]
    print('pure_name',pure_name)
    if not pure_name in response:
        response = character_name+', '+response

    regex_replacement = [
        {
            'find': ', and there are no (.*?)\.',
            'new': ''
        },
        {
            'find': 'the artwork does not contain(.*?)\.',
            'new': '.'
        },
        {
            'find': ', and there is no (.*?)\.',
            'new': ''
        },
        {
            'find': ', and there is (.*?)Twitter username(.*?)\.',
            'new': ''
        },
        {
            'find': '. The tags (.*?)\.',
            'new': ''
        },
        {
            'find': 'The image (.*?)tags (.*?)\.',
            'new': ''
        },
        {
            'find': 'twitter(.*?)\'(.*?)\'',
            'new': 'twitter watermark'
        },
        {
            'find': ', and does not (.*?)\.',
            'new': '.'
        },
        {
            'find': ', and it does not contain(.*?)\.',
            'new': '.'
        },
        {
            'find': 'the content is rated as \'(.*?)\.',
            'new': '.'
        }
        
    ]
    # use regex to replace unwanted content
    for regex_replacement in regex_replacement:
        response = re.sub(regex_replacement['find'], regex_replacement['new'], response)
    return response

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
        if f.endswith('.jpg') or f.endswith('.webp') or f.endswith('.jpeg') or f.endswith('.png'):
            images.append(f)
    total_image = len(images)
    count_image = 0
    # loop all image in clear dir
    for image_name in tqdm(images):
        count_image+=1
        print(f'processing image: {count_image}/{total_image} {image_name}')
        image_path = os.path.join(character_dir, image_name)
        print(f'Processing {image_path}')
    
        image = Image.open(image_path).convert('RGB')

        history = []
        
        # caption_ext = '.wd14cap'
        # get filename without extension
        filename = os.path.splitext(os.path.basename(image_path))[0]
        # caption file
        # caption_file = input_folder + filename + caption_ext
        filename, ext = os.path.splitext(image_name)
        # caption_file = os.path.join(character_dir, f'{filename}{caption_ext}')

        txt_file = os.path.join(character_dir, f'{filename}.txt')
        # if txt file exist, skip
        if os.path.exists(txt_file):
            continue

        # if caption not exist, skip
        # if not os.path.exists(caption_file):
        #     continue

        # read tags from caption file
        # with open(caption_file, 'r') as f:
        #     caption = f.read()
        #     tags = caption.split(',')[:-1]
        # genders = []
        # characters = []
        # tag_content = ""
        # for tag in tags:
        #     tag = tag.strip(' ')
        #     if 'gender' in tag:
        #         name = tag.replace('gender_','')
        #         name = name.replace('[','').replace(']','').split(':')[0]
        #         genders.append(name)
        #     elif 'character' in tag or 'characeter' in tag:
        #         name = tag.replace('character_','')
        #         name = name.replace('characeter_','')
        #         name = name.replace('[','').replace(']','').split(':')[0]
        #         # print(name)
        #         characters.append(name)
        #     else:
        #         tag_content += f"{tag},"

        # # print(tag_content)
        # character_desc = ""
        # character_name = ""
        # for gender in genders:
        #     character_desc += f"{gender},"
        # character_desc += " named "
        # for character in characters:
        #     character_desc += f"{character.replace('_',' ').strip(' ')},"
        #     character_name += f"{character.replace('_',' ').strip(' ')},"
        # character_desc = character_desc[:-1]
        # character_name = character_name[:-1]
        # prompt = f'Make use of the following tags to write concise accurate very blunt and direct description of {character_desc} in the image. The description should be in less than 40 words.  tags contains multiple tag. a tag is stored inside a []. a tag name would be the key and a tag probability as the value. Tags: {tag_content}.'

        # print(character_name)
        # break
        prompt = f'Describe the image as short as possible. No longer than 20 words. Precisely, interaction and background. Include composition, angle and perspective. Use only facts and concise language; avoid interpretations or speculation:'
        starts_with = f'The image showcases '
        query = f'Question: {prompt} Answer: {starts_with}'
        input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=history, images=[image])

        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]] if image is not None else None,
        }
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]
        
        # force_words = convert_character_tags(character_name)
        # force_words_ids = tokenizer(force_words, add_special_tokens=False)["input_ids"] if force_words else []
        # print(f"** force_words: {force_words}")
        # bad_words = ['depicting','showcasing','possible','@','Twitter user','posted by','indicating','somewhat','suggest','might','might be','tags','username','twitter','demeanor', 'captures', 'moment', 'tranquility', 'emphasizing','contrasts', 'atmosphere', 'showcase','showcased','the game','character','depicts','showcases','depicted','named','seemingly ','possibly', 'likely', 'animated ', 'cartoon', 'appears', 'seems']
        # bad_words_ids = tokenizer(bad_words, add_special_tokens=False)["input_ids"] if bad_words else []
        # print(f"** bad_words: {bad_words}")
        pass_loop = False
        attempts_count = 0
        attempts_count_limit = 5
        response = ''
        with torch.no_grad():
            while not pass_loop:
                outputs = model.generate(**inputs, **gen_kwargs
                                    # , force_words_ids=force_words_ids,
                                    # bad_words_ids=bad_words_ids
                                    )
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                response = tokenizer.decode(outputs[0])
                response = response.split("</s>")[0]
                response = response.replace("\n","")
                # response = replacement(response)

                if len(response) > 250:
                    pass_loop = True
                if attempts_count > attempts_count_limit:
                    pass_loop = True
                else:
                    attempts_count += 1
            
            file_name = image_name.split('.')[0]
            with open(txt_file, 'w', encoding="utf-8") as f:
                f.truncate(0)
                f.write(response)
                f.close()
                print(response)
                print(txt_file)
                # print(f'write {file_name}.txt')
        # break
    # run one subset for test
    # break

# replace twitter username
# replace , and there are no discernible dialogue or narrative elements.
# yae miko is not engaged in any sexual activities. The tags provided are accurate and do not contain any ambiguous or misleading information. 
# weighing weight pounds
# female figure 

# print the start time in yyyy-mm-dd hh:mm:ss format
end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

print(f'End time: {end_time}')
