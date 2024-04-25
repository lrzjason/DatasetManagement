

# Import the modules
import os
import json

# Define the folder path and the output file name
input_dir = "F:/ImageSet/openxl2_realism_test_output/ori"

reference_dir = "F:/ImageSet/openxl2_realism_test_output/pag"

captions_folder = "F:/ImageSet/openxl2_realism_test/test/4000"

for file in os.listdir(captions_folder):
    # Check if the file is an image by its extension
    if file.endswith((".txt")):
        # Join the folder path and the file name to get the full path
        caption_path = os.path.join(captions_folder, file)
        reference_path = os.path.join(reference_dir, file.replace(".txt", ".webp"))
        if not os.path.exists(reference_path):
            # delete input file
            os.remove(caption_path)