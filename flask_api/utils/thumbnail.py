import os
from PIL import Image

input_dir = 'F:\\ImageSet\\Pickscore_train_5k'
output_dir = os.path.join(input_dir, "thumbnails")
print("output_dir",output_dir)

if not os.path.exists(output_dir):
    print("creating output_dir",output_dir)
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        image_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        if os.path.exists(output_path):
            print("skipping",output_path)
            continue
        with Image.open(image_path) as image:
            width, height = image.size
            if width > 1024 or height > 1024:
                if width > height:
                    new_width = 1024
                    new_height = int(height * (new_width / width))
                else:
                    new_height = 1024
                    new_width = int(width * (new_height / height))
                image.thumbnail((new_width, new_height))
            output_path = os.path.join(output_dir, filename)
            print("thumbnail",output_path)
            image.save(output_path)