from datasets import load_dataset

data_dir = "F:/ImageSet/training_script_cotton_doll/test_webp"
dataset = load_dataset("text", data_dir=data_dir, encoding="utf-8")
dataset.push_to_hub("lrzjason/cotton_doll")

# dataset = load_dataset("lrzjason/cotton_doll")
# print(dataset)