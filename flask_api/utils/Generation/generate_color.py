import os
import random
import numpy as np
from PIL import Image, ImageColor 

def generate_color_image(width, height, color_name):
    """
    Generate a pure color image with the given width, height, and color name.
    The color name should be a string that can be converted to a RGB color using
    the Pillow library's Color.getrgb() method.
    """
    color = ImageColor.getrgb(color_name)
    print('color:', color)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = color[0]
    image[:, :, 1] = color[1]
    image[:, :, 2] = color[2]
    return Image.fromarray(image)

def save_color_image(image, output_dir, color_name):
    """
    Save the given color image to the specified output directory.
    The image will be saved as a PNG file with a filename in the format
    "color_{index}.png", where {index} is a zero-padded integer.
    """
    filename = f"{color_name}.png"
    path = os.path.join(output_dir, filename)
    image.save(path, "PNG")
    return filename

def save_color_text(output_dir, color_name):
    """
    Save the given color text to the specified output directory.
    The text will be saved as a TXT file with a filename in the format
    "color_{index}.txt", where {index} is a zero-padded integer.
    """
    filename = f"{color_name}.txt"
    path = os.path.join(output_dir, filename)
    text = f"pure {color_name} background, solid {color_name} background, simple background"
    with open(path, "w") as f:
        f.write(text)
    return filename

def main():
    # Parse command-line arguments
    width = 1024
    height = 1024
    output_dir = "output"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate and save random color images
    color_names = [
        "AliceBlue",
        "AntiqueWhite",
        "Aqua",
        "Aquamarine",
        "Azure",
        "Beige",
        "Bisque",
        "Black",
        "BlanchedAlmond",
        "Blue",
        "BlueViolet",
        "Brown",
        "BurlyWood",
        "CadetBlue",
        "Chartreuse",
        "Chocolate",
        "Coral",
        "CornflowerBlue",
        "Cornsilk",
        "Crimson",
        "Cyan",
        "DarkBlue",
        "DarkCyan",
        "DarkGoldenRod",
        "DarkGray",
        "DarkGreen",
        "DarkKhaki",
        "DarkMagenta",
        "DarkOliveGreen",
        "DarkOrange",
        "DarkOrchid",
        "DarkRed",
        "DarkSalmon",
        "DarkSeaGreen",
        "DarkSlateBlue",
        "DarkSlateGray",
        "DarkTurquoise",
        "DarkViolet",
        "DeepPink",
        "DeepSkyBlue",
        "DimGray",
        "DodgerBlue",
        "FireBrick",
        "FloralWhite",
        "ForestGreen",
        "Fuchsia",
        "Gainsboro",
        "GhostWhite",
        "Gold",
        "GoldenRod",
        "Gray",
        "Green",
        "GreenYellow",
        "HoneyDew",
        "HotPink",
        "IndianRed",
        "Indigo",
        "Ivory",
        "Khaki",
        "Lavender",
        "LavenderBlush",
        "LawnGreen",
        "LemonChiffon",
        "LightBlue",
        "LightCoral",
        "LightCyan",
        "LightGoldenRodYellow",
        "LightGray",
        "LightGreen",
        "LightPink",
        "LightSalmon",
        "LightSeaGreen",
        "LightSkyBlue",
        "LightSlateGray",
        "LightSteelBlue",
        "LightYellow",
        "Lime",
        "LimeGreen",
        "Linen",
        "Magenta",
        "Maroon",
        "MediumAquaMarine",
        "MediumBlue",
        "MediumOrchid",
        "MediumPurple",
        "MediumSeaGreen",
        "MediumSlateBlue",
        "MediumSpringGreen",
        "MediumTurquoise",
        "MediumVioletRed",
        "MidnightBlue",
        "MintCream",
        "MistyRose",
        "Moccasin",
        "NavajoWhite",
        "Navy",
        "OldLace",
        "Olive",
        "OliveDrab",
        "Orange",
        "OrangeRed",
        "Orchid",
        "PaleGoldenRod",
        "PaleGreen",
        "PaleTurquoise",
        "PaleVioletRed",
        "PapayaWhip",
        "PeachPuff",
        "Peru",
        "Pink",
        "Plum",
        "PowderBlue",
        "Purple",
        "Red",
        "RosyBrown",
        "RoyalBlue",
        "SaddleBrown",
        "Salmon",
        "SandyBrown",
        "SeaGreen",
        "SeaShell",
        "Sienna",
        "Silver",
        "SkyBlue",
        "SlateBlue",
        "SlateGray",
        "Snow",
        "SpringGreen",
        "SteelBlue",
        "Tan",
        "Teal",
        "Thistle",
        "Tomato",
        "Turquoise",
        "Violet",
        "Wheat",
        "White",
        "WhiteSmoke",
        "Yellow",
        "YellowGreen",
    ]

    
    num_images = len(color_names)
    for i in range(num_images):
        # Generate a random color name
        color_name = random.choice(color_names)
        
        filename = f"{color_name}.png"
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            continue
        # Generate a color image
        image = generate_color_image(width, height, color_name)
        # Save the color image
        image_filename = save_color_image(image, output_dir, color_name)
        # Save the color text
        text_filename = save_color_text(output_dir, color_name)
        print(f"Generated color image {image_filename} and text {text_filename}")
        # break

if __name__ == "__main__":
    main()
