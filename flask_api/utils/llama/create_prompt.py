

# Import the modules
import os
import json
import re

# working on santi_3 image
# Define the folder path and the output file name
input_dir = "F:/ImageSet/handpick_high_quality_b2_cropped"

template_path = "./prompt_template.txt"
# read template from file
template = ''
with open(template_path, "r", encoding="utf-8") as f:
    template = f.read()
    template = template.strip()
    f.close()

# suffix = ', 8k photo, high quality'
output_ext = '.llama_template'

text_ext = '.txt'
tags_ext = '.wd14'

for sub in os.listdir(input_dir):
    sub_dir = os.path.join(input_dir, sub)
    for file in os.listdir(sub_dir):
        # Check if the file is an image by its extension
        if file.endswith(text_ext):
            text_path = os.path.join(sub_dir, file)
            tags_path = text_path.replace(text_ext, tags_ext)
            if not os.path.exists(tags_path):
                continue
            output_path = text_path.replace(text_ext, output_ext)
            # avoid duplicate
            if os.path.exists(output_path):
                continue
            # Join the folder path and the file name to get the full path
            print(text_path)
            text = ''
            tags = ''
            # Append the full path to the list
            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read()
                text = text.replace('/n', '').strip()
                f.close()
            
            print(tags_path)
            # Append the full path to the list
            with open(tags_path, "r", encoding="utf-8") as f:
                tags = f.read()
                tags = tags.replace('/n', '').strip()
                f.close()
            
            # clone template
            prompt = template

            # replace text and tags
            prompt = prompt.replace('$text$', text)
            prompt = prompt.replace('$tags$', tags)
            
            # save to file
            print(output_path)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(prompt)
                f.close()
                print(f"{output_path} created")

            