# Import flask and request
from flask import Flask, request, send_file
from flask_cors import CORS
import json
from PIL import Image
from io import BytesIO
import os
import shutil
import requests

from utils.BingBrush import BingBrush 
from pathlib import Path
import datetime

# Create a flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/save_generation', methods=['POST'])
def save_generation():
    urls = request.form.get('urls')
    output_folder = request.form.get('output_folder')
    save_name = request.form.get('save_name')
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for url in urls:
        response = requests.get(url)
        if response.status_code == 200:
            file_name = url.split("/")[-1]
            if save_name != "":
                file_name = save_name+"_"+datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            save_path = os.path.join(output_folder, file_name) + ".webp"

            with open(save_path, "wb") as file:
                file.write(response.content)
            print(f"Save image to: {save_path}")
        else:
            print("Download failed!")

@app.route('/generate', methods=['POST'])
def generate():
    # image_ext = ".webp"
    # Get the path parameter from the request
    # output_folder = request.form.get('output_folder')
    # Get the path parameter from the request
    bing_cookie_path = request.form.get('bing_cookie_path')
    # Get the path parameter from the request
    prompt = request.form.get('prompt')
    # Check if the path is valid
    # if not output_folder:
    #     return 'Invalid path', 400
    
    brush = BingBrush(cookie=bing_cookie_path)
    img_urls = brush.process(prompt=prompt)

    if img_urls == -1:
        print('Generation Failed.')
        return 'Generation Failed', 400
    
    return {'img_urls': img_urls}, 200


# Define a route for the list function
@app.route('/export_pairs', methods=['POST'])
def export_pairs():
    image_ext = ".png"
    # Get the path parameter from the request
    caption_folder = request.form.get('caption_folder')
    # Get the path parameter from the request
    image_folder = request.form.get('image_folder')
    # Check if the path is valid
    if not caption_folder:
        return 'Invalid path', 400
    
    image_subfolders = os.listdir(image_folder)
    
    caption_export_folder = f"{caption_folder}_export"
    # create export folder if it doesn't exist
    if not os.path.exists(caption_export_folder):
        os.makedirs(caption_export_folder)
    image_export_folder = f"{image_folder}_export"
    # create export folder if it doesn't exist
    if not os.path.exists(image_export_folder):
        os.makedirs(image_export_folder)

    
    # copy images to export folder
    for image_subfolder in image_subfolders:
        image_export_subfolder = os.path.join(image_export_folder,image_subfolder)
        if not os.path.exists(image_export_subfolder):
            os.makedirs(image_export_subfolder)

    # try:
    # Return the list as a JSON response
    # Read the saved_files.json file and append the saved file names to the list
    json_file = 'saved_pairs.json'
    saved_pairs = []
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for name in data:
            text_file_name = f"{name}.txt"
            # copy captions to export folder
            text_file_path = os.path.join(caption_folder, text_file_name)
            text_file_export_path = os.path.join(caption_export_folder, text_file_name)
            if os.path.exists(text_file_path) and not os.path.exists(text_file_export_path):
                shutil.copy(text_file_path, text_file_export_path)
            
            # copy images to export folder
            for image_subfolder in image_subfolders:
                image_file_name = f"{name}{image_ext}"
                image_file_path = os.path.join(image_folder,image_subfolder,image_file_name)
                image_file_export_path = os.path.join(image_export_folder,image_subfolder,image_file_name)
                if os.path.exists(image_file_path) and not os.path.exists(image_file_export_path):
                    shutil.copy(image_file_path, image_file_export_path)
    # except Exception as e:
    #     return {'message': str(e)}, 500
    # Return the list as a JSON response
    return {'message': 'Pair exported successfully'}, 200

# Define a route for the list function
@app.route('/list_pairs', methods=['POST'])
def list_pairs():
    image_ext = ".png"
    # Get the path parameter from the request
    caption_folder = request.form.get('caption_folder')
    # Get the path parameter from the request
    image_folder = request.form.get('image_folder')
    # Check if the path is valid
    if not caption_folder:
        return 'Invalid path', 400
    
    thumbnail_folder = f"{image_folder}_thumbnails"

    image_subfolders = os.listdir(image_folder)

    pairs = []
    for caption_filename in os.listdir(caption_folder):
        caption_path = os.path.join(caption_folder, caption_filename)
        caption_name,ext = os.path.splitext(caption_filename)
        with open(caption_path, 'r', encoding='utf-8') as f:
            content = f.read()
        images = {}
        thumbnails = {}
        image_exist = True
        for image_subfolder in image_subfolders:
            images[image_subfolder] = os.path.join(image_folder,image_subfolder,f"{caption_name}{image_ext}")
            if not os.path.exists(images[image_subfolder]):
                image_exist = False
                break
            thumbnail_file = os.path.join(thumbnail_folder,image_subfolder,f"{caption_name}{image_ext}")
            # print(f'thumbnail_file: {thumbnail_file}')
            if os.path.exists(thumbnail_file):
                thumbnails[image_subfolder] = thumbnail_file
        if image_exist:
            # print(f'thumbnails: {thumbnails}')
            pairs.append({
                "name": caption_name,
                "caption": content,
                "images":images,
                "thumbnails":thumbnails,

            })

    # Return the list as a JSON response
    # Read the saved_files.json file and append the saved file names to the list
    json_file = 'saved_pairs.json'
    saved_pairs = []
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for file in data:
            saved_pairs.append(file)

    # Return the list as a JSON response
    return {'pairs': pairs, 'saved_pairs': saved_pairs}, 200


# Define a route for the save function
@app.route('/save_pair', methods=['POST'])
def save_pair():
    # Get the file name and content parameters from the request
    file_name = request.form.get('file_name')
    content = request.form.get('content')
    name = request.form.get('name')
    # Check if the file name and content are valid
    if not file_name or not content:
        return 'Invalid file name or content', 400
    # Check if the file name has .txt extension
    if not file_name.endswith('.txt'):
        return 'File name must have .txt extension', 400
    # Import os module to write to the file
    if os.path.exists(file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            existing_content = f.read()
        if not (len(existing_content) == len(content)):
            # Open the file in write mode and overwrite its content
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(content)
    # Append the file name to the JSON file
    json_file = 'saved_pairs.json'
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = []
    if name not in data:
        data.append(name)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    # Return a success message as a JSON response
    return {'message': 'Pair saved successfully','saved_pairs': data}, 200


# Define a route for the switch function
@app.route('/save_to_target', methods=['POST'])
def save_to():

    # Get the file name parameter from the request
    file_name = request.form.get('file_name')
    # Get the file name parameter from the request
    save_from = request.form.get('save_from')
    # Get the file name parameter from the request
    save_to = request.form.get('save_to')
    # Get the file name parameter from the request
    label = request.form.get('label')
    # Get the file name parameter from the request
    self_remove = request.form.get('self_remove')

    source_path = os.path.join(save_from,file_name)
    target_dir = os.path.join(save_to,label)
    target_path = os.path.join(target_dir,file_name)

    json_file = 'saved_dataset.json'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {}

    
    if not os.path.exists(save_to):
        os.mkdir(save_to)
    
    
    for label_dir in os.listdir(save_to):
        if label_dir not in data:
            data[label_dir] = []
        if file_name in os.listdir(os.path.join(save_to,label_dir)):
            exist_file = os.path.join(save_to,label_dir,file_name)
            # skip self
            if os.path.join(save_to,label_dir) == save_from:
                continue
            print('remove exist file', exist_file)
            # remove previous
            os.remove(exist_file)

    # move switch_from to temp folder
    if os.path.exists(source_path):
        shutil.copy(source_path,target_path)
    
    if self_remove:
        os.remove(source_path)
    
    for label_dir in os.listdir(save_to):
        data[label_dir] = os.listdir(os.path.join(save_to,label_dir))

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    # Return a success message as a JSON response
    return {'message': 'File saved successfully','saved_dataset': data}, 200

# Define a route for the switch function
@app.route('/switch_pair', methods=['POST'])
def switch_pair():
    # Get the file name parameter from the request
    switch_from = request.form.get('switch_from')
    # Get the file name parameter from the request
    switch_to = request.form.get('switch_to')

    thumbnail_from = switch_from.replace("images","images_thumbnails")
    thumbnail_to = switch_to.replace("images","images_thumbnails")
    if not os.path.exists(switch_from) or not os.path.exists(switch_from):
        return 'Invalid path', 400

    # create temp folder for switch
    temp_folder = "temp"
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)
    
    temp_thumbnail_folder = "temp_thumbnail"
    if not os.path.exists(temp_thumbnail_folder):
        os.mkdir(temp_thumbnail_folder)

    file_name = os.path.basename (switch_from)
    temp_file = os.path.join(temp_folder,file_name)
    temp_thumbnail = os.path.join(temp_thumbnail_folder,file_name)
    # move switch_from to temp folder
    if os.path.exists(switch_from):
        shutil.move(switch_from,temp_file)
        if os.path.exists(thumbnail_from):
            shutil.move(thumbnail_from,temp_thumbnail)
    
    # move switch_to to switch_from
    if os.path.exists(switch_to):
        shutil.move(switch_to,switch_from)
        if os.path.exists(thumbnail_to):
            shutil.move(thumbnail_to,thumbnail_from)

    # move temp_file to switch_to
    if os.path.exists(temp_file):
        shutil.move(temp_file,switch_to)
        if os.path.exists(temp_thumbnail):
            shutil.move(temp_thumbnail,thumbnail_to)

    # Return a success message as a JSON response
    return {'message': 'File switched successfully'}, 200
    

# Define a route for the delete function
@app.route('/delete_pair', methods=['POST'])
def delete_pair():
    # Get the file name parameter from the request
    name = request.form.get('name')
    text_file_name = f"{name}.txt"
    # Get the path parameter from the request
    caption_folder = request.form.get('caption_folder')
    caption_deleted_folder = f"{caption_folder}_deleted"
    # Get the path parameter from the request
    image_folder = request.form.get('image_folder')
    image_deleted_folder = f"{image_folder}_deleted"
    # Check if the file name is valid
    if not name:
        return 'Invalid name', 400
    # Import os module to check if the file exists and delete it
    import os
    import shutil
    # Check if the file exists in the current directory
    if not os.path.exists(os.path.join(caption_folder,f"{name}.txt")):
        return 'File does not exist', 404
    
    # create deleted dir if not exist
    if not os.path.exists(caption_deleted_folder):
        os.mkdir(caption_deleted_folder)
    # create deleted dir if not exist
    if not os.path.exists(image_deleted_folder):
        os.mkdir(image_deleted_folder)
    # copy file to deleted dir

    # check txt file exist in captions folder
    if os.path.exists(os.path.join(caption_folder,text_file_name)):
        # copy to deleted folder
        shutil.copy(os.path.join(caption_folder,text_file_name),os.path.join(caption_deleted_folder,text_file_name))
        # remove original files
        os.remove(os.path.join(caption_folder,text_file_name))
    
    # check image file exist in images folder
    for image_subfolder in os.listdir(image_folder):
        image_file_name = f"{name}.png"
        original_folder = os.path.join(image_folder,image_subfolder)
        deleted_folder = os.path.join(image_deleted_folder,image_subfolder)
        if not os.path.exists(deleted_folder):
            os.mkdir(deleted_folder)
        if os.path.exists(os.path.join(original_folder,image_file_name)):
            # copy to deleted folder
            shutil.copy(os.path.join(original_folder,image_file_name),os.path.join(deleted_folder,image_file_name))
            # remove original files
            os.remove(os.path.join(original_folder,image_file_name))

    # Remove the file name from the saved_files.json file if it exists
    json_file = 'saved_pairs.json'
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if name in data:
            data.remove(name)
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f)

    # write file into deleted_pairs.
    json_file = 'deleted_pairs.json'
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = []
    if name not in data:
        data.append(name)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    # Return a success message as a JSON response
    return {'message': 'File deleted successfully'}, 200


# Define a route for the list function
@app.route('/list_dataset', methods=['POST'])
def list_dataset():
    # Get the path parameter from the request
    path = request.form.get('path')
    # Check if the path is valid
    if not path:
        return 'Invalid path', 400
    # Import os module to list files
    import os
    # Initialize an empty list to store file names
    files = []
    # Loop through all files in the given path
    for file in os.listdir(path):
        # Check if the file has .png or .jpg extension
        if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.webp'):
            # Append the file name to the list
            files.append(file)
    # Return the list as a JSON response
    # Read the saved_files.json file and append the saved file names to the list
    json_file = 'saved_dataset.json'
    saved_files = []
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for key in data:
                saved_files += data[key]
    # Return the list as a JSON response
    return {'files': files, 'saved_files': saved_files}, 200


# Define a route for the list function
@app.route('/list', methods=['POST'])
def list_files():
    # Get the path parameter from the request
    path = request.form.get('path')
    # Check if the path is valid
    if not path:
        return 'Invalid path', 400
    # Import os module to list files
    import os
    # Initialize an empty list to store file names
    files = []
    # Loop through all files in the given path
    for file in os.listdir(path):
        # Check if the file has .png or .jpg extension
        if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.webp'):
            # Append the file name to the list
            files.append(file)
    # Return the list as a JSON response
    # Read the saved_files.json file and append the saved file names to the list
    json_file = 'saved_files.json'
    saved_files = []
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for file in data:
            saved_files.append(file)
    # Return the list as a JSON response
    return {'files': files, 'saved_files': saved_files}, 200

# Define a route for the save function
@app.route('/save', methods=['POST'])
def save_file():
    # Get the file name and content parameters from the request
    file_name = request.form.get('file_name')
    content = request.form.get('content')
    # Check if the file name and content are valid
    if not file_name or not content:
        return 'Invalid file name or content', 400
    # Check if the file name has .txt extension
    if not file_name.endswith('.txt'):
        return 'File name must have .txt extension', 400
    # Import os module to write to the file
    if os.path.exists(file_name):
        # with open(file_name, 'r', encoding='utf-8') as f:
        #     existing_content = f.read()
        # if not (len(existing_content) == len(content)):
            # Open the file in write mode and overwrite its content
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(content)
    # Append the file name to the JSON file
    json_file = 'saved_files.json'
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = []
    if file_name not in data:
        data.append(file_name)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    # Return a success message as a JSON response
    return {'message': 'File saved successfully','saved_files': data}, 200

# Define a route for the image function
@app.route('/file/<path:file_name>', methods=['GET'])
def get_file(file_name):
    print('-----file_name', file_name)
    # Import os module to check if the file exists
    import os
    # Check if the file exists in the current directory
    if not os.path.exists(file_name):
        return 'File does not exist', 404
    mimetype = 'text/plain'
    is_image = False
    # Check if the file has .png or .jpg extension
    if file_name.endswith('.png'):
        mimetype = 'image/png'
        is_image = True
    elif file_name.endswith('.jpg'):
        mimetype = 'image/jpeg'
        is_image = True
    elif file_name.endswith('.webp'):
        mimetype = 'image/webp'
        is_image = True
    if is_image:
        # replace file dir to thumbnail dir
        # get file dir
        file_dir = os.path.dirname(file_name)
        # get file name
        img_file_name = os.path.basename(file_name)
        # get parent dir
        parent_dir = os.path.dirname(file_dir)
        # get thumbnail dir
        thumbnail = os.path.join(parent_dir, "thumbnails", img_file_name)
        if os.path.exists(thumbnail):
            print("thumbnail exists", thumbnail)
            file_name = thumbnail

    # Return the image file as a response
    return send_file(file_name, mimetype=mimetype)

# Define a route for the delete function
@app.route('/delete', methods=['POST'])
def delete_file():
    # Get the file name parameter from the request
    file_name = request.form.get('file_name')
    # Check if the file name is valid
    if not file_name:
        return 'Invalid file name', 400
    # Import os module to check if the file exists and delete it
    import os
    # Check if the file exists in the current directory
    if not os.path.exists(file_name):
        return 'File does not exist', 404
    # Delete the file
    os.remove(file_name)
    # Delete the corresponding text file
    text_file_name = os.path.splitext(file_name)[0] + '.txt'
    if os.path.exists(text_file_name):
        os.remove(text_file_name)
    # Remove the file name from the saved_files.json file if it exists
    json_file = 'saved_files.json'
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if file_name in data:
            data.remove(file_name)
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f)
    # Return a success message as a JSON response
    return {'message': 'File deleted successfully'}, 200



# Run the app in debug mode
if __name__ == '__main__':
    app.run(debug=True)
