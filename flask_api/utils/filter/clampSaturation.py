from PIL import Image
import os
import numpy as np

def clamp_saturation(input_path, output_path, target_saturation):
    # Load the image
    image = Image.open(input_path)
    image_hsv = image.convert('HSV')
    data_hsv = np.array(image_hsv)

    # Calculate current mean saturation
    current_saturation_mean = np.mean(data_hsv[:, :, 1])

    # Calculate the factor by which to adjust the saturation to meet the target
    if current_saturation_mean > target_saturation:
        saturation_adjustment_factor = target_saturation / current_saturation_mean
        # Adjust saturation
        data_hsv[:, :, 1] = (data_hsv[:, :, 1] * saturation_adjustment_factor).clip(0, 255)

    # Convert adjusted HSV data back to an image
    new_image_hsv = Image.fromarray(data_hsv, 'HSV')
    new_image_rgb = new_image_hsv.convert('RGB')

    # Save the modified image
    new_image_rgb.save(output_path)

def process_directory(input_dir, output_dir, target_saturation):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process all images in the directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            clamp_saturation(input_path, output_path, target_saturation)
            print(f'Processed {filename}')

def main():
    # Get user input for directories and target saturation
    # input_directory = input("Enter the path to the input directory: ")
    # output_directory = input("Enter the path to the output directory: ")
    # target_saturation = float(input("Enter the target mean saturation (0-255): "))
    input_directory = "F:/ImageSet/openxl2_realism_above_average/1_cog"
    output_directory = "F:/ImageSet/openxl2_realism_above_average/1_cog_fix_saturation"
    target_saturation = float(76)

    # Process the images
    process_directory(input_directory, output_directory, target_saturation)
    print("All images have been processed and saved to the specified output directory.")

if __name__ == "__main__":
    main()
