from imgutils.detect import detect_heads, detection_visualize
# from matplotlib import pyplot as plt
import os
from PIL import Image

input_dir = 'F:/ImageSet/anime_dataset/genshin_classified/aether'
image = '5368752.jpg'
image_path = os.path.join(input_dir, image)
img = Image.open(image_path) # load the image
box,label,prob = detect_heads(img)[0]  # detect it

print(box)

# box = (150, 37, 479, 366) # define the coordinates of the crop rectangle
img2 = img.crop(box) # crop the image
img2.show("cropped.jpg") # save the cropped image


# plt.imshow(detection_visualize(image_path, result))
# plt.show()