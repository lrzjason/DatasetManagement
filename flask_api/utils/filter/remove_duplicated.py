import os
import hashlib

# input_dir = 'F:/ImageSet/anime_dataset/genshin_classified'
input_dir = 'F:/ImageSet/anime_dataset/genshin_classified_too_short'
subsets = os.listdir(input_dir)
# output_dir = 'F:/ImageSet/anime_dataset/genshin_classified_too_short'
output_dir = 'F:/ImageSet/anime_dataset/genshin_classified'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

filter_set = set()

def remove_file(subset_dir,file):
    print('remove file:', path)
    os.remove(path)
    txt_path = path.replace('.jpg', '.txt')
    if os.path.exists(txt_path):
        os.remove(txt_path)
        print('remove file:', txt_path)
        os.remove(txt_path)
    wd14cap_path = path.replace('.jpg', '.wd14cap')
    if os.path.exists(wd14cap_path):
        print('remove file:', wd14cap_path)
        os.remove(wd14cap_path)


# target_ext = '.jpg'
target_ext = '.txt'
modified_count = 0
count = 0
debug = False


def filter_duplicated(path,modified_count):
    digest = hashlib.sha1(open(path,'rb').read()).digest()
    if digest not in filter_set:
        filter_set.add(digest)
    else:
        remove_file(subset_dir,file)
        modified_count+=1
        return modified_count

def greater_than(content,threshold=150):
    return len(content) > threshold


def less_than(content,threshold=150):
    return len(content) < threshold

def filter_content_length(path,subset,file,modified_count,debug):
    with open(path, 'r+', encoding="utf-8") as f:
        content = f.read()
        f.close()
        if greater_than(content,150):
            # print('filter_content_length:', path)
            # move file to output folder
            filename,ext = os.path.splitext(file)
            # print('file:', file)
            input_subset = os.path.join(input_dir,subset)
            output_subset = os.path.join(output_dir,subset)
            # create output subset folder if not exists
            if not os.path.exists(output_subset):
                os.mkdir(output_subset)

            copy_file_exts = ['.jpg','.txt','.wd14cap']
            for ext in copy_file_exts:
                input_file = os.path.join(input_subset,f'{filename}{ext}')
                output_file = os.path.join(output_subset, f'{filename}{ext}')
                print('input_file:', input_file, 'output_file:', output_file)
                os.rename(input_file, output_file)

            modified_count+=1
            debug = True
            return debug,modified_count
        

# list files in each subset
for subset in subsets:
    subset_dir = os.path.join(input_dir, subset)
    files = os.listdir(subset_dir)

    # remove dir if no files
    if len(files) == 0:
        print('remove dir:', subset_dir)
        os.rmdir(subset_dir)
    for file in files:
        if file.endswith(target_ext):
            print(file)
            path = os.path.join(subset_dir, file)
            print('subset_dir',subset_dir)

            # os.remove(path)

            # modified_count+=1

            # move file to output folder
            result = filter_content_length(path,subset,file,modified_count,debug)
            if result != None:
                debug = result[0]
                modified_count = result[1]
            # break
            # break
    #     if debug:
    #         break
    
    # if debug:
    #     break
            # file_path = os.path.join(subset_dir, file)
            # print(file_path)
            # # read file content
            # with open(file_path, 'r+', encoding="utf-8") as f:
            #     content = f.read()
            #     skip_start_arr = ['A, ','An, ', 'a, ', 'an, ', 'A ','a ','an ','An ', ', ']
            #     for skip_start in skip_start_arr:
            #         if content.startswith(skip_start):
            #             content = content[len(skip_start):]
            #             break
            #     modified_count+=1
            #     f.write(content)
print('modified_count:',modified_count)
                         
                        

# tags = [' character_ningguang_(genshin_impact)']
# for tag in tags:
#     if 'character' in tag or 'characeter' in tag:
#         print('tag is character')
#     else:
#         print('tag is not character')