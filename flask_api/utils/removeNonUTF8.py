import os
import chardet
import codecs
import os

def replace_non_utf8_characters(input_dir):
    # Helper function to filter out non-UTF-8 characters
    def clean_text(text):
        return ''.join([char if ord(char) < 128 else '' for char in text])

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(input_dir, filename)
            
            try:
                # Read file content
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Clean content by removing non-UTF-8 characters
                cleaned_content = clean_text(content)

                # Write cleaned content back to the file
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                    print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage:
input_directory = 'F:/ImageSet/3dkitten'
replace_non_utf8_characters(input_directory)
