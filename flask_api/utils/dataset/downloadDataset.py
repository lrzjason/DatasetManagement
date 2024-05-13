
# PixArt-alpha/SAM-LLaVA-Captions10M

from datasets import load_dataset


dataset = load_dataset("ptx0/mj-v52-redux", split="Collection_1")
print(dataset)

# subset = dataset.filter(lambda example: "black and white" in example['txt'])
# item = next(iter(dataset))

# count = 0
# max_count = 100

# filename = item['filename']
# caption = item['caption']
# collection = item['collection']

# print(item)
# generate image using text content

# image to image via sdxl checkpoint

# save sdxl image with the same filename as the text content

# save text content using the same filename