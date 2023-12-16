
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import clip
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from transformers import CLIPProcessor, CLIPModel
import json


def load_img(path,transform):
    img = Image.open(path)
    if transform is not None:
        img = transform(img)
    return img

def load_txt(path):
    data = []
    with open(path, 'r') as fp:
        # data = fp.read()
        content = fp.read()
        fp.close()
    chunk_size = 77
    for i in range(0, len(content), chunk_size):
      chunk = content[i:i+chunk_size]
    #   prompt_tokens = clip.tokenize(chunk)
      data.append(chunk)
    return data

@torch.no_grad()
def calculate_clip_score(input_dir, model, processor,device):
    # get current dir and create temp dir
    temp_dir = os.path.join(os.getcwd(), 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    result_file = os.path.join(temp_dir, 'result.json')
    if os.path.exists(result_file):
        os.remove(result_file)
        # create result file
    score_results = []
    
    score_acc = 0.
    sample_num = 0.
    logit_scale = model.logit_scale.exp()
    # loop input_dir to get .jpg and .txt data
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            text_path = os.path.join(input_dir, filename.split('.')[0] + '.txt')
            # print("image_path",image_path)
            # print("text_path",text_path)
            if not os.path.exists(text_path):
                print("skipping",text_path)
                continue
            # apply transforms to image
            with open(image_path, 'rb') as f:
                img = Image.open(f)
                #     real_data = preprocess(img).unsqueeze(0).to(device)
                fake_data = load_txt(text_path)

                inputs = processor(text=fake_data, images=img, return_tensors="pt", padding=True)
                outputs = model(**inputs)

                score = outputs.logits_per_image.mean()
                print({'name': filename, 'score': score.item()})
                score_results.append({'name': filename, 'score': score.item()})

                # same the file name and score to result json file in temp_dir
                


                score_acc += score
                sample_num += 1

    # Save the results to a JSON file
    with open(result_file, 'w') as f:
        json.dump(score_results, f)

    return score_acc / sample_num

        
def forward_modality(model, data, flag):
    device = next(model.parameters()).device
    if flag == 'img':
        features = model.encode_image(data.to(device))
    elif flag == 'txt':
        # loop data to get features tensors
        features = []
        for prompt_tokens in data:
            prompt_tokens = prompt_tokens.to(device)
            features.append(model.encode_text(prompt_tokens).mean(dim=0))
        features = torch.stack(features)
    else:
        raise TypeError
    return features


def main():
    # args = parser.parse_args()

    # if args.device is None:
    #     device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    # else:
    #     device = torch.device(args.device)

    # if args.num_workers is None:
    #     try:
    #         num_cpus = len(os.sched_getaffinity(0))
    #     except AttributeError:
    #         # os.sched_getaffinity is not available under Windows, use
    #         # os.cpu_count instead (which may not return the *available* number
    #         # of CPUs).
    #         num_cpus = os.cpu_count()

    #     num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    # else:
    #     num_workers = args.num_workers
    # model = 'F:\\DatasetManagement\\flask_api\\utils\\clip\\model.safetensors'
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    # print('Loading CLIP model: {}'.format(model))
    # model, preprocess = clip.load(model, device=device)
    
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    
    # dataset = DummyDataset(args.real_path, args.fake_path,
    #                        args.real_flag, args.fake_flag,
    #                        transform=preprocess, tokenizer=clip.tokenize)
    # dataloader = DataLoader(dataset, args.batch_size, 
    #                         num_workers=num_workers, pin_memory=True)
    
    print('Calculating CLIP Score:')
    input_dir = 'F:/ImageSet/dump/mobcup_output'
    # CLIP Score:  21.26517120524995
    # 21.165674209594727
    # input_dir = "F:/lora_training/quality_training/20_photo"
    # 23.531064987182617
    # input_dir = "F:/lora_training/qianqian/images/3_q_woman"
    # 23.793838500976562
    clip_score = calculate_clip_score(input_dir, model, processor, device)
    clip_score = clip_score.cpu().item()
    print('CLIP Score: ', clip_score)


if __name__ == '__main__':
    main()