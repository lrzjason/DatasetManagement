from PIL import Image
import os
import numpy as np
from tqdm import tqdm

def analyze_colors_and_saturation(input_path):
    # Load the image
    image = Image.open(input_path)
    image_rgb = image.convert('RGB')
    image_hsv = image.convert('HSV')

    # Convert the images to numpy arrays for easier manipulation
    data_rgb = np.array(image_rgb)
    data_hsv = np.array(image_hsv)

    # Calculate the mean of each RGB channel: Red, Green, Blue
    red_mean = np.mean(data_rgb[:, :, 0])
    green_mean = np.mean(data_rgb[:, :, 1])
    blue_mean = np.mean(data_rgb[:, :, 2])

    # Calculate the mean saturation (from HSV)
    saturation_mean = np.mean(data_hsv[:, :, 1])

    return red_mean, green_mean, blue_mean, saturation_mean

def process_directory(input_dir):
    reds, greens, blues, saturations = [], [], [], []
    
    # Process all images in the directory
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            input_path = os.path.join(input_dir, filename)
            red_mean, green_mean, blue_mean, saturation_mean = analyze_colors_and_saturation(input_path)
            reds.append(red_mean)
            greens.append(green_mean)
            blues.append(blue_mean)
            saturations.append(saturation_mean)
            print(f'Processed {filename}: Red={red_mean}, Green={green_mean}, Blue={blue_mean}, Saturation={saturation_mean}')

    # Calculate and print the average color intensities and saturation for the entire directory
    if reds:
        overall_red = np.mean(reds)
        overall_green = np.mean(greens)
        overall_blue = np.mean(blues)
        overall_saturation = np.mean(saturations)
        print(f"\nOverall Average Color Intensities and Saturation:")
        print(f"Red: {overall_red}")
        print(f"Green: {overall_green}")
        print(f"Blue: {overall_blue}")
        print(f"Saturation: {overall_saturation}")

def main():
    # input_directory = input("F:/ImageSet/openxl2_realism_above_average/1_cog")
    # input_directory = "F:/ImageSet/openxl2_realism_above_average/1_cog"
    # process_directory(input_directory)

    
    input_directory = "F:/ImageSet/openxl2_realism_above_average/1_cog_fix_saturation"
    process_directory(input_directory)

if __name__ == "__main__":
    main()
