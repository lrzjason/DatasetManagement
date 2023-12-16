import os
from PIL import Image

input_dir = 'F:\\ImageSet\\Pickscore_train_5k'
caption_folder = f'{input_dir}\\captions'
image_folder = f'{input_dir}\\images'
output_dir = f'{input_dir}\\images_thumbnails'
print("output_dir",output_dir)

if not os.path.exists(output_dir):
    print("creating output_dir",output_dir)
    os.makedirs(output_dir)

count = 0
# list images sub folder
# check image file exist in images folder
for image_subfolder in os.listdir(image_folder):
    count += 1
    print("count",count)
    image_export_folder = os.path.join(output_dir,image_subfolder)
    if not os.path.exists(image_export_folder):
        os.makedirs(image_export_folder)
    for file_id in os.listdir(caption_folder):
        filename = file_id.split('.')[0] + ".png"
        image_path = os.path.join(image_folder,image_subfolder,filename)
        print("image_path",image_path)
        if os.path.exists(image_path):
            output_path = os.path.join(image_export_folder, filename)
            if os.path.exists(output_path):
                print("skipping",output_path)
                continue
            with Image.open(image_path) as image:
                width, height = image.size
                if width > 512 or height > 512:
                    if width > height:
                        new_width = 512
                        new_height = int(height * (new_width / width))
                    else:
                        new_height = 512
                        new_width = int(width * (new_height / height))
                    image.thumbnail((new_width, new_height))
                # output_path = os.path.join(output_dir, filename)
                print("thumbnail",output_path)
                image.save(output_path)
                # break