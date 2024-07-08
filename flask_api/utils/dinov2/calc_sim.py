# import faiss
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from torch.nn import CosineSimilarity
import clip
import hpsv2

def init_clip(device):
    model,preprocess = clip.load("ViT-B/32", device=device)
    model.device = device
    return model,preprocess

def get_clip_feature(model,preprocess,image_path):
    device = model.device
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    # Normalize the embeddings
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features

def cal_clip(model,preprocess,image_path_1="",image_path_2="",image_features_1=None,image_features_2=None):
    device = model.device
    # image_path = "image.jpg"
    # text = ["your text"]
    # Prepare your image and text

    if image_features_1 is None:
        image_features_1 = get_clip_feature(model,preprocess,image_path_1)
    
    
    if image_features_2 is None:
        image_features_2 = get_clip_feature(model,preprocess,image_path_2)
    
    

    # Compute cosine similarity
    cosine_sim = CosineSimilarity(dim=1, eps=1e-6)
    similarity_score = cosine_sim(image_features_1, image_features_2)

    # print(similarity_score)
    return similarity_score[0]
    

def init_model(device):
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
    model = AutoModel.from_pretrained('facebook/dinov2-small').to(device)
    return model,processor

def get_image_feature(model,preprocess,image_path):
    device = model.device
    inputs = preprocess(images=Image.open(image_path), return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    embeddings = embeddings.mean(dim=1)
    return embeddings

def cal_similarity(model,preprocess,image_path_1="",image_path_2="",image_features_1=None,image_features_2=None):
    # device = model.device

    if image_features_1 is None:
        image_features_1 = get_image_feature(model,preprocess,image_path_1)
    if image_features_2 is None:
        image_features_2 = get_image_feature(model,preprocess,image_path_2)

    # Compute cosine similarity
    cosine_sim = CosineSimilarity(dim=1, eps=1e-6)
    similarity_score = cosine_sim(image_features_1, image_features_2)

    # print(similarity_score)
    return similarity_score[0]
    
def main():
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = init_model(device)
    # image_path_1 = "F:/ImageSet/handpick_high_quality/cdrama scene/aiqingeryi_1.jpg"
    # image_path_2 = "F:/ImageSet/handpick_high_quality/cdrama scene/changlan_1.jpg"
    image_path_1 = "F:/ImageSet/PickScore/images/high/cc8cc868-c640-48e5-bcc1-4c3c0dc61b74.png"
    image_path_2 = "F:/ImageSet/PickScore/images/low/cc8cc868-c640-48e5-bcc1-4c3c0dc61b74.png"
    image_path_openxl = "F:/ImageSet/pickscore_random_captions_pixart2sdxl/cc8cc868-c640-48e5-bcc1-4c3c0dc61b74.png"

    c_model, c_preprocess = init_clip(device)

    high_dino_sim = cal_similarity(model,preprocess,image_path_1=image_path_openxl,image_path_2=image_path_1)
    high_dino_sim.to(dtype=torch.float16)
    print("openxl vs high pickscore: ", high_dino_sim, " dino_sim")
    

    high_clip_sim = cal_clip(c_model,c_preprocess,image_path_1=image_path_openxl,image_path_2=image_path_1)
    print("openxl vs high pickscore: ", high_clip_sim, " clip_sim")

    
    low_dino_sim = cal_similarity(model,preprocess,image_path_1=image_path_openxl,image_path_2=image_path_2)
    low_dino_sim.to(dtype=torch.float16)
    print("openxl vs low pickscore: ", low_dino_sim, " dino_sim")

    # c_model, c_preprocess = init_clip(device)
    low_clip_sim = cal_clip(c_model,c_preprocess,image_path_1=image_path_openxl,image_path_2=image_path_2)
    print("openxl vs low pickscore: ", low_clip_sim, " clip_sim")

    result = hpsv2.score(image_path_openxl, "creative image of a gorgeous female elf in armor stands on the edge of a building with mountains in the background, in the style of eye-catching resin jewelry, dark white and light gold, gongbi, romantic use of light, exaggerated facial features, rich and immersive, elegant", hps_version="v2.1")[0]
    print("hpsv2: ", result, " image_path_openxl")

    
    high_result = hpsv2.score(image_path_1, "creative image of a gorgeous female elf in armor stands on the edge of a building with mountains in the background, in the style of eye-catching resin jewelry, dark white and light gold, gongbi, romantic use of light, exaggerated facial features, rich and immersive, elegant", hps_version="v2.1")[0]
    high_result = torch.tensor(high_result).to(device=device,dtype=torch.float16)
    print("hpsv2: ", high_result, " image_path_1")

    
    low_result = hpsv2.score(image_path_2, "creative image of a gorgeous female elf in armor stands on the edge of a building with mountains in the background, in the style of eye-catching resin jewelry, dark white and light gold, gongbi, romantic use of light, exaggerated facial features, rich and immersive, elegant", hps_version="v2.1")[0]
    low_result = torch.tensor(low_result).to(device=device,dtype=torch.float16)
    print("hpsv2: ", low_result, " image_path_2")

    high_avg_result = (high_result + high_dino_sim) / 2
    low_avg_result = (low_result + low_dino_sim) / 2
    print("avg high dinov2: ", high_avg_result, " image_path_1")
    print("avg low dinov2: ", low_avg_result, " image_path_2")

if __name__ == '__main__':
    main()
