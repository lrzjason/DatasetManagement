import glob
import chardet

input_dir = "F:/ImageSet/training_script_cotton_doll/test/cog_full"
non_ascii_files = []

for file_path in glob.glob(input_dir + "/*.txt"):
    with open(file_path, "rb") as f:
        encoding = chardet.detect(f.read())["encoding"]
        if encoding != "ascii":
            non_ascii_files.append(file_path)

with open("non_ascii_files.txt", "w") as f:
    for file_path in non_ascii_files:
        f.write(file_path + "\n")

print("Saved non-ASCII files to non_ascii_files.txt")