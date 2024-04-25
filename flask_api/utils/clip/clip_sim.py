import torch
import clip
from torch.nn import CosineSimilarity
from PIL import Image

def init_model(device):
    model,preprocess = clip.load("ViT-B/32", device=device)
    model.device = device
    return model,preprocess

def get_image_feature(model,preprocess,image_path):
    device = model.device
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    # Normalize the embeddings
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features

def cal_similarity(model,preprocess,text,image_path="",image_features=None):
    device = model.device
    # image_path = "image.jpg"
    # text = ["your text"]
    # Prepare your image and text

    if image_features is None:
        image_features = get_image_feature(model,preprocess,image_path)
    text = clip.tokenize(text).to(device)

    # Obtain the embeddings
    with torch.no_grad():
        # image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    # Normalize the embeddings
    # image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity
    cosine_sim = CosineSimilarity(dim=1, eps=1e-6)
    similarity_score = cosine_sim(image_features, text_features)

    # print(similarity_score)
    return similarity_score[0]
    
def classify(model,preprocess,compare_targets,image_path):
    device = model.device
    # image = preprocess(image_path).unsqueeze(0).to(device)
    
    image_features = get_image_feature(model,preprocess,image_path)

    scores = {}
    max_score = 0
    max_target = ''
    for target in compare_targets:
        score = cal_similarity(model,preprocess,target,image_features=image_features)
        scores[target] = score
        if max_score == 0 or score > max_score:
            max_target = target
            max_score = score

    # print(scores,max_target,max_score)
    return scores,max_target,max_score

def main():
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = init_model(device)
    # image_path = "C:/Users/Administrator/Desktop/test.jpg"
    # image_path = "F:/ImageSet/openxl2_realism/tags/fischl_5599508_preserved.webp"
    image_path = "F:/ImageSet/hands_dataset_above_average/clean-hands/5287.0.70.jpg"
    print(image_path)

    # sim_score = cal_similarity(model,preprocess,text,image_path)
    # model, preprocess = clip.load("ViT-B/32", device=device)

    targets = ['palm gesture','fist gesture','peace gesture','ok gesture']

    _,selected,score = classify(model,preprocess,targets,image_path)

    # type_targets = ['single hand','both hands']
    # scores,classified_type,type_score = classify(model,preprocess,type_targets,image_path)
    # print('scores',scores)
    # print('classified_type,type_score',classified_type,type_score)
    
    print('classified_quality,quality_score',selected,score)

if __name__ == '__main__':
    main()
