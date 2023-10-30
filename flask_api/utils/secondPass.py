import os
import shutil

raw_dir = 'F:\\ImageSet\\dump\\mobcup_output_raw'
input_dir = 'F:\\ImageSet\\dump\\mobcup_output'
delete_dir = 'F:\\ImageSet\\dump\\mobcup_output_deleted'

if not os.path.exists(delete_dir):
    print("creating delete_dir",delete_dir)
    os.makedirs(delete_dir)

def list_with_ext(directory, extensions):
    return [filename for filename in os.listdir(directory) if filename.endswith(extensions)]

extensions = (".jpg", ".jpeg", ".png")
selected_files = list_with_ext(input_dir, extensions)

print(len(selected_files),"files selected")

# list raw files with extensions, if it not in selected_files, copy it and its text file to delete_dir
raw_files = list_with_ext(raw_dir, extensions)
print(len(raw_files),"raw files found")

for raw_file in raw_files:
    if raw_file not in selected_files:
        # get file name without extension
        raw_file_name = os.path.splitext(os.path.basename(raw_file))[0]
        # get text file name
        raw_file_text = raw_file_name + ".txt"

        # original
        raw_file_ori = os.path.join(raw_dir, raw_file)
        raw_file_text_ori = os.path.join(raw_dir, raw_file_text)

        # copy raw image and raw text to delete_dir
        raw_file_copy = os.path.join(delete_dir, raw_file)
        raw_file_text_copy = os.path.join(delete_dir, raw_file_text)

        print("copying",raw_file_ori,"to",raw_file_copy)
        shutil.copy(raw_file_ori, raw_file_copy)
        print("copying",raw_file_text_ori,"to",raw_file_text_copy)
        shutil.copy(raw_file_text_ori, raw_file_text_copy)
        

# for filename in os.listdir(input_dir):
#     if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
#         image_path = os.path.join(input_dir, filename)
#         output_path = os.path.join(output_dir, filename)
#         if os.path.exists(output_path):
#             print("skipping",output_path)
#             continue
#         with Image.open(image_path) as image:
#             width, height = image.size
#             if width > 1024 or height > 1024:
#                 if width > height:
#                     new_width = 1024
#                     new_height = int(height * (new_width / width))
#                 else:
#                     new_height = 1024
#                     new_width = int(width * (new_height / height))
#                 image.thumbnail((new_width, new_height))
#             output_path = os.path.join(output_dir, filename)
#             print("thumbnail",output_path)
#             image.save(output_path)