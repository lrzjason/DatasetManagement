

# Import the modules
import os
import json
import re
import string

# Define the folder path and the output file name
input_dir = "F:/ImageSet/vit_train/crop_predict"

def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]*[\ ]', '_', text)

exts = [".jpg",".webp"]

printable = set(string.printable)
print(printable)
for file in os.listdir(input_dir):
    # Check if the file is an image by its extension
    for ext in exts:
        # get filename
        filename = os.path.splitext(file)[0]
        # filename = remove_non_ascii(filename)
        new_filename = ''.join(filter(lambda x: x in printable, filename))
        if new_filename != filename:
            # rename the file
            try:
                os.rename(os.path.join(input_dir, file), os.path.join(input_dir, f"{new_filename}{ext}"))
            except Exception as e:
                print(e)