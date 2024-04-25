import os
from PIL import Image

folder = "E:/Media/0AIpainting/0外包/服装/sample" # the directory where the images are stored
for filename in os.listdir(folder): # loop through all files in the directory
    if filename.endswith(".png"): # check if the file is a jpg image
        image = Image.open(os.path.join(folder, filename)) # open the image
        data = list(image.getdata()) # get the image data without exif info
        image_without_exif = Image.new(image.mode, image.size) # create a new image object
        image_without_exif.putdata(data) # put the image data into the new image object
        new_filename = os.path.splitext(filename)[0] + "_remove" + os.path.splitext(filename)[1] # create the new filename with the suffix
        image_without_exif.save(os.path.join(folder, new_filename)) # save the new image file
        image_without_exif.close() #