import torch
import clip
from torch.nn import CosineSimilarity
from PIL import Image

# categories = ["digital art","illustration","anime","cartoon","oil painting","cinematic photo","raw photo","dutch angle","porn","sexy","nipples","r18","r15","r12","nsfw","sfw"]
sfw_categories = ["sfw","not porn","not sexy content","no nipples","image doesn't have private body parts"]
nsfw_categories = ["nsfw","porn","sexy content","nipples","image has private body parts"]

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# sfw
# image_path = "F:/ImageSet/openxl2_leftover/1_3_yae_miko_fullbody/5353229.jpg"
# image_path = "F:/ImageSet/openxl2_leftover/1_2_fischl_upperbody/5868400.jpg"
# image_path = "F:/ImageSet/openxl2_leftover/1_2_fischl_upperbody/5480463.jpg"
# image_path = "F:/ImageSet/openxl2_leftover/1_2_fischl_upperbody/5356828.jpg"

# nsfw
# image_path = "F:/ImageSet/openxl2_leftover/1_3_yae_miko_fullbody/5355172.jpg"
# image_path = "F:/ImageSet/openxl2_leftover/1_3_yae_miko_fullbody/5784557.jpg"
# image_path = "F:/ImageSet/openxl2_leftover/1_3_yae_miko_fullbody/5579784.jpg"
# image_path ="F:/ImageSet/openxl2_leftover/1_1_ganyu_fullbody/5819635.jpg"
image_path ="F:/ImageSet/openxl2_leftover/1_1_ganyu_fullbody/6003224.jpg"


# Prepare your image and text
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

sfw_tokenized_text = []
nsfw_tokenized_text = []

for idx,category in enumerate(sfw_categories):
    sfw_text = clip.tokenize([sfw_categories[idx]]).to(device)
    sfw_tokenized_text.append(sfw_text)
    nsfw_text = clip.tokenize([nsfw_categories[idx]]).to(device)
    nsfw_tokenized_text.append(nsfw_text)

sfw_text_features = []
nsfw_text_features = []

# Obtain the embeddings
with torch.no_grad():
    image_features = model.encode_image(image)
    for text in sfw_tokenized_text:
        sfw_text_features.append(model.encode_text(text))
    for text in nsfw_tokenized_text:
        nsfw_text_features.append(model.encode_text(text))

# Normalize the embeddings
image_features /= image_features.norm(dim=-1, keepdim=True)

sfw_score = 0.0
for idx, text_features in enumerate(sfw_text_features):
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity
    cosine_sim = CosineSimilarity(dim=1, eps=1e-6)
    similarity_score = cosine_sim(image_features, text_features)
    sfw_score+= similarity_score

    # print(f"Similarity score for {sfw_categories[idx]}:", similarity_score.item())

sfw_score = sfw_score/len(sfw_categories)
print(f"sfw_score: {sfw_score[0]}")


nsfw_score = 0.0
for idx, text_features in enumerate(nsfw_text_features):
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity
    cosine_sim = CosineSimilarity(dim=1, eps=1e-6)
    similarity_score = cosine_sim(image_features, text_features)
    nsfw_score+= similarity_score

    # print(f"Similarity score for {nsfw_categories[idx]}:", similarity_score.item())

nsfw_score = nsfw_score/len(nsfw_categories)
print(f"nsfw_score: {nsfw_score[0]}")

if nsfw_score > sfw_score:
    print("NSFW content detected!")