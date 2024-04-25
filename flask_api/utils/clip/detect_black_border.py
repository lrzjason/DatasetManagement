from PIL import Image
import os
import shutil
def detect_black_border(image_path):
    with Image.open(image_path) as img:
        # Convert image to grayscale
        img_gray = img.convert('L')
        
        # Get image size
        width, height = img_gray.size
        
        # Define the threshold for black color
        black_threshold = 20
        
        # Check the borders
        top_border = all(img_gray.getpixel((x, 0)) < black_threshold for x in range(width))
        bottom_border = all(img_gray.getpixel((x, height - 1)) < black_threshold for x in range(width))
        left_border = all(img_gray.getpixel((0, y)) < black_threshold for y in range(height))
        right_border = all(img_gray.getpixel((width - 1, y)) < black_threshold for y in range(height))
        
        # Return True if all borders are black
        return top_border or bottom_border or left_border or right_border

def main():
    # # Example usage
    above_average_dir = 'F:\ImageSet\hands_dataset_above_average\clean-hands'
    underscore_dir = 'F:\ImageSet\hands_dataset_underscore\clean-hands'
    # image_path = '40540.0.82.jpg'  # Replace with your image path
    # file = input_dir + '\\' + image_path
    # if detect_black_border(file):
    #     print(f"The image {file} contains a black border")
    # else:
    #     print(f"The image {image_path} does not contain a black border")


    for file in os.listdir(above_average_dir):
        # Check if the file is an image by its extension
        if file.endswith((".jpg")):
            # Join the folder path and the file name to get the full path
            image_path = os.path.join(above_average_dir, file)
            if detect_black_border(image_path):
                output_file = os.path.join(underscore_dir, file)
                if not os.path.exists(output_file):
                    shutil.copy(image_path, output_file)
                if os.path.exists(image_path):
                    os.remove(image_path)

if __name__ == '__main__':
    main()
