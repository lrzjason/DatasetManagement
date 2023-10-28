# Import flask and request
from flask import Flask, request, send_file
from flask_cors import CORS
import json
from PIL import Image

# Create a flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


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
        if file.endswith('.png') or file.endswith('.jpg'):
            # Append the file name to the list
            files.append(file)
    # Return the list as a JSON response
    # Read the saved_files.json file and append the saved file names to the list
    json_file = 'saved_files.json'
    saved_files = []
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
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
    import os
    # Open the file in write mode and overwrite its content
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(content)
    # Append the file name to the JSON file
    json_file = 'saved_files.json'
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
    else:
        data = []
    if file_name not in data:
        data.append(file_name)
        with open(json_file, 'w') as f:
            json.dump(data, f)
    # Return a success message as a JSON response
    return {'message': 'File saved successfully','saved_files': data}, 200

# Define a route for the image function
@app.route('/file', methods=['POST'])
def get_file():
    # Get the file name parameter from the request
    file_name = request.form.get('file_name')
    print('-----file_name',file_name)
    # Check if the file name is valid
    if not file_name:
        return 'Invalid file name', 400
    mimetype = 'text/plain'
    # Check if the file has .png or .jpg extension
    if file_name.endswith('.png'):
        mimetype = 'image/png'
    elif file_name.endswith('.jpg'):
        mimetype = 'image/jpeg'
    # Import os module to check if the file exists
    import os
    # Check if the file exists in the current directory
    if not os.path.exists(file_name):
        return 'File does not exist', 404
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
        with open(json_file, 'r') as f:
            data = json.load(f)
        if file_name in data:
            data.remove(file_name)
            with open(json_file, 'w') as f:
                json.dump(data, f)
    # Return a success message as a JSON response
    return {'message': 'File deleted successfully'}, 200



# Run the app in debug mode
if __name__ == '__main__':
    app.run(debug=True)
