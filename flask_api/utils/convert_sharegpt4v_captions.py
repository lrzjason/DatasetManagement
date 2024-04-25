# Import the modules
import json
import os

# Define the JSON file name and the output directory
json_file = "F:\InternLM-XComposer\projects\ShareGPT4V\captions.json"
output_dir = "F:\ImageSet\hagrid_test\call"

# Open the JSON file and load the data as a list of dictionaries
with open(json_file, "r") as f:
    data = json.load(f)

# Loop through the data and save the captions to text files
for item in data:
    # Get the image file path and the caption from the dictionary
    for image_file_path, caption in item.items():
      # Split the image file path into directory, name and extension
      image_dir, image_name = os.path.split(image_file_path)
      image_name, image_ext = os.path.splitext(image_name)

    # Construct the output file path by joining the output directory, the image name and the .txt extension
    output_file_path = os.path.join(output_dir, image_name + ".txt")

    # Open the output file in write mode and write the caption
    with open(output_file_path, "w") as f:
        f.write(caption)

# Print a message to indicate the task is done
print("Saved the captions to text files in", output_dir)
