import os
import shutil

# Define the input folder path
input_folder = 'F:/ImageSet/r18/073.葱油饼er'

# Function to move directories from subfolders to the input folder
def move_directories_to_input(folder_path):
    # Loop through each item in the main folder
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        # Check if the item is a directory (subfolder)
        if os.path.isdir(item_path):
            # Loop through each item in the subfolder
            for sub_item in os.listdir(item_path):
                sub_item_path = os.path.join(item_path, sub_item)
                # Check if the sub-item is a directory
                if os.path.isdir(sub_item_path):
                    # Move the directory to the input folder
                    shutil.move(sub_item_path, folder_path)
                    print(f"Moved directory: {sub_item_path} to {folder_path}")

# Call the function with the input folder path
move_directories_to_input(input_folder)

# Print a success message
print("All directories from subfolders have been moved to the input folder.")
