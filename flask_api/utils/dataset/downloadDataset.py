from datasets import load_dataset
from huggingface_hub import snapshot_download

# save_dir = "F:/ImageSet/pixart_test"
# snapshot_download(repo_id="lrzjason/cotton_doll",repo_type='dataset',local_dir=save_dir)

save_dir = "F:/models/llm"
snapshot_download(repo_id="google/gemma-2b",repo_type='model',local_dir=save_dir)

# huggingface-cli download terminusresearch/midjourney-v6-520k-raw --local-dir F:\ImageSet\midjourney-v6-520k-raw --type dataset
# huggingface-cli download yuvalkirstain/pickapic_v2 --local-dir F:\ImageSet\pickapic_v2 --type dataset --split test_unique
# huggingface-cli download Kwai-Kolors/Kolors --local-dir E:\Kolors

# ds.save_to_disk(save_dir)
# subset = dataset.filter(lambda example: "black and white" in example['txt'])
# item = next(iter(ds))
# print(item)
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