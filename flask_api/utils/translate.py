import glob
from googletrans import Translator
import os

# input_dir = "F:/ImageSet/dump/mobcup_output"
output_dir = "F:/ImageSet/dump/mobcup_output_translated"
# create output_dir if not exists
if not os.path.exists(output_dir):
    print("creating output_dir",output_dir)
    os.makedirs(output_dir)
translator = Translator()

# for file_path in glob.glob(input_dir + "/*.txt"):
# read non_ascii_files.txt, and loop through each file
with open("non_ascii_files.txt", "r") as f:
    non_ascii_files = f.read().splitlines()
    for file_path in non_ascii_files:
      with open(file_path, "r", encoding="utf-8") as f:
          text = f.read()
          translation = translator.translate(text, dest="en").text
          output_path = os.path.join(output_dir, os.path.basename(file_path))
          with open(output_path, "w", encoding="utf-8") as out_f:
              out_f.write(translation)

print("Finished translating files.")