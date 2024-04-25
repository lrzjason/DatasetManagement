import os
import hashlib

# input_dir = 'F:/ImageSet/anime_dataset/genshin_classified'
input_dir = 'F:/ImageSet/openxl2_dataset'
subsets = os.listdir(input_dir)
# print(subsets)

# test_string = 'A female, ganyu (genshin impact) with blue hair and red eyes, wearing a black leotard with white sleeves, red and gold patterns, and a black glove on her right hand. Her left hand is bare, revealing a silver bell-shaped ornament. She has a pair of long blue hair tied with a red ribbon, and her eyes are red. She is wearing black pants and black high heels. She stands in front of a white background, with a blue orb floating in front, and there is a black and white pattern on the orb. She looks to the side with a blush on her face.female, ganyu (genshin impact) with blue hair and red eyes, wearing a black leotard with white sleeves, red and gold patterns, and a black glove on her right hand. Her left hand is bare, revealing a silver bell-shaped ornament. She has a pair of long blue hair tied with a red ribbon, and her eyes are red. She is wearing black pants and black high heels. She stands in front of a white background, with a blue orb floating in front, and there is a black and white pattern on the orb. She looks to the side with a blush on her face.'
# test_string = 'female, ganyu (genshin impact) with blue hair and red horns. She is gazing directly at the viewer with a blush on her cheeks. ganyu (genshin impact) is wearing a brown top with a gold bell attached to it. Her hair is long and wavy, and she has a pair of gloves on her hands. The background is a mix of green and blue with leaves and light beams, giving a serene and ethereal ambiance. The art style is detailed and colorful, with a touch of fantasy elements.female, ganyu (genshin impact) with blue hair and red horns. She is gazing directly at the viewer with a blush on her cheeks. ganyu (genshin impact) is wearing a brown top with a gold bell attached to it. Her hair is long and wavy, and she has a pair of gloves on her hands. The background is a mix of green and blue with leaves and light beams, giving a serene and ethereal ambiance. The art style is detailed and colorful, with a touch of fantasy elements.'

# find_arr = ['female, ', '.aether (genshin impact)']

def find_nth(string, substring, n=1):
   if (n == 1):
       return string.find(substring)
   else:
       return string.find(substring, find_nth(string, substring, n - 1) + 1)

# for find in find_arr:
#     if find in test_string:
#         temp_test_string = test_string.lower()
#         index = find_nth(temp_test_string,find,2)
#         if index != -1:
#             result = temp_test_string[0:index]
#         print(result)
#         break

    
# skip_subset = ['noelle', 'ouro_kronii', 'paimon', 'pyra', 'qiqi', 'raiden_mei', 'raiden_shogun', 'razor', 'rosaria', 'sangonomiya_kokomi', 'scaramouche', 'seele_vollerei', 'shenhe', 'slime', 'sucrose', 'tartaglia', 'tennouji_rina', 'thoma', 'two_character', 'usada_pekora', 'venti', 'xiangling', 'xiao', 'xingqiu', 'yae_miko', 'yae_sakura', 'yanfei', 'yelan', 'yoimiya', 'yorha_no._2_type_b', 'zhongli']
skip_subset = []
not_foud_arr = []
# modified_count = 0
count = 0

# test_file = 'F:/ImageSet/anime_dataset/genshin_classified/aether/5377061.txt'
# with open(test_file, 'r+', encoding="utf-8") as f:
#     content = f.read()
#     print(content)
#     temp_test_string = content.lower()
#     for find in find_arr:
#         if find in temp_test_string:
#             print('find: ', find)
#             index = find_nth(temp_test_string,find,1)
#             print('index: ', index)
#             if index != -1:
#                 result = content[0:index]
#                 f.truncate(0)
#                 print(result)
#                 f.write(result)
#                 count += 1
#                 not_found = False
#                 break

# list files in each subset
for subset in subsets:
    if subset in skip_subset:
        continue
    subset_dir = os.path.join(input_dir, subset)
    files = os.listdir(subset_dir)
    for file in files:
        if file.endswith('.txt'):
            count+=1
            # file_path = os.path.join(subset_dir, file)
            # # if file exist, remove it
            # if os.path.exists(file_path):
            #     os.remove(file_path)
            #     count+=1
            # print(file_path)
            # read file content
            # with open(file_path, 'r+', encoding="utf-8") as f:
            #     content = f.read()
            #     temp_test_string = content.lower()
            #     not_found = True
            #     for find in find_arr:
            #         if find in temp_test_string:
            #             index = find_nth(temp_test_string,find,2)
            #             if index != -1:
            #                 result = content[0:index]
            #                 f.truncate(0)
            #                 f.write(result)
            #                 count += 1
            #                 not_found = False
            #                 break
            #     if not_found:
            #         not_foud_arr.append(file_path)

            #     skip_start_arr = ['A, ','An, ', 'a, ', 'an, ', 'A ','a ','an ','An ', ', ']
            #     for skip_start in skip_start_arr:
            #         if content.startswith(skip_start):
            #             content = content[len(skip_start):]
            #             break
            #     modified_count+=1
            #     f.write(content)
print('count:',count)

# # save not_found_arr to file
# with open('not_found_arr.txt', 'w') as f:
#     for item in not_foud_arr:
#         f.write(item + '/n')
                         
                        

# tags = [' character_ningguang_(genshin_impact)']
# for tag in tags:
#     if 'character' in tag or 'characeter' in tag:
#         print('tag is character')
#     else:
#         print('tag is not character')