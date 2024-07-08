

# Import the modules
import os
import json

# Define the folder path and the output file name
input_dir = "F:/ImageSet/hunyuan_test_temp/1_creative_photo"

prefix = ''
# suffix = ', 8k photo, high quality'
suffix = ''

# for subdir in os.listdir(input_dir):
#     folder_path = os.path.join(input_dir, subdir)
#     # Loop through the folder and append the image paths to the list
#     for file in os.listdir(folder_path):
#         # Check if the file is an image by its extension
#         if file.endswith((".jpg")):
#             # Join the folder path and the file name to get the full path
#             txt_file = file.replace('.jpg', '.txt')
#             full_path = os.path.join(folder_path, txt_file)
#             content = ''
#             if os.path.exists(full_path):
#                 # Append the full path to the list
#                 with open(full_path, "r", encoding="utf-8") as f:
#                     content = f.read()
#                     f.close()
#             else:
#                 with open(full_path, "w", encoding="utf-8") as f:
#                     f.close()
#             content = prefix + ', ' + content
#             if len(suffix)>0:
#                 content =  content + ', ' + suffix
#             with open(full_path, "r+", encoding="utf-8") as out_f:
#                 out_f.write(content)
empty=[]
for file in os.listdir(input_dir):
    # Check if the file is an image by its extension
    if file.endswith((".png.phi3Vision")):
        # Join the folder path and the file name to get the full path
        full_path = os.path.join(input_dir, file)
        content = ''
        print(full_path)
        # Append the full path to the list
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
            content = content.replace('\n',' ')
            f.close()

        # if "Answer:" in content:
        #     content = content[:content.index("Answer:")]

        # if "watermark" in content:
        #     start = content.find("watermark")
        #     sentence_start = content.rfind('.', 0, start) + 1
        #     content = content[:sentence_start]

        # if "caption " in content:
        #     start = content.find("caption ")
        #     sentence_start = content.rfind('.', 0, start) + 1
        #     content = content[:sentence_start]

        # if "signature " in content:
        #     start = content.find("signature ")
        #     sentence_start = content.rfind('.', 0, start) + 1
        #     content = content[:sentence_start]

        # if "signed by " in content:
        #     start = content.find("signed by ")
        #     sentence_start = content.rfind('.', 0, start) + 1
        #     content = content[:sentence_start]

        if " tag" in content:
            start = content.find(" tag")
            sentence_start = content.rfind('.', 0, start) + 1
            content = content[:sentence_start]
        content = content.strip()
        if len(content)==0:
            empty.append(full_path)
        full_path = full_path.replace('.png.phi3Vision','.txt')
        # save the content
        with open(full_path, "w", encoding="utf-8") as out_f:
            out_f.write(content)
        # content = prefix + content + suffix
        # with open(full_path, "r+", encoding="utf-8") as out_f:
        #     out_f.write(content)

print(empty)