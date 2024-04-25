

# Import the modules
import os
import json

# Define the folder path and the output file name
folder_path = "F:/ImageSet/hagrid_test/call"
output_file = "images_file.json"

# Create an empty list to store the image paths
image_paths = []

# Loop through the folder and append the image paths to the list
for file in os.listdir(folder_path):
    # Check if the file is an image by its extension
    if file.endswith((".jpg", ".png", ".jpeg")):
        # Join the folder path and the file name to get the full path
        full_path = os.path.join(folder_path, file)
        # Append the full path to the list
        image_paths.append(full_path)

# Open the output file in write mode
with open(output_file, "w") as f:
    # Dump the list of image paths as a JSON array
    json.dump(image_paths, f)

# Print a message to indicate the task is done
print("Saved the image paths to", output_file)
