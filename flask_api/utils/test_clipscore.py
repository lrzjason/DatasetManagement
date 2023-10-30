from torchmetrics.functional.multimodal import clip_score
from functools import partial
import torch
import os
from PIL import Image
import numpy as np
import clip


clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-large-patch14")


input_dir = 'F:\\ImageSet\\dump\\mobcup_output'
input_file = '2 best friends - paint two girls'
image_file = input_file + ".jpg"
text_file = input_file + ".txt"

def calculate_clip_score(images, prompts, max_seq_length=77, stride=32):
    images_int = []
    for image in images:
        image_int = (image * 255).astype("uint8")
        images_int.append(image_int)
    images_int = np.array(images_int)
    
    scores = []
    for prompt in prompts:
        prompt_tokens = clip.tokenize(prompt, truncation=True, max_length=max_seq_length, padding="max_length", return_tensors="pt")
        prompt_length = prompt_tokens.input_ids.shape[1]
        for i in range(0, prompt_length, stride):
            prompt_tokens_window = prompt_tokens[:, i:i+max_seq_length]
            score = clip_score_fn(images_int, prompt_tokens_window).detach()
            scores.append(score)
    
    mean_score = torch.mean(torch.stack(scores))
    return round(float(mean_score), 4)


def calc_clip_score(image_file,text_file):
  images = []
  prompts = []
  with open(os.path.join(input_dir, text_file), "r") as f:
    prompts.append(f.read())
  
  image_path = os.path.join(input_dir, image_file)
  image = Image.open(image_path)
  images.append(np.array(image))
  return calculate_clip_score(images, prompts)


clip_score_1 = calc_clip_score(image_file, text_file)
print(f"CLIP score 1: {clip_score_1}")


# clip_score_2 = calc_clip_score(image_file, text_file_2)
# print(f"CLIP score 2: {clip_score_2}")

# average_clip_score = (clip_score_1 + clip_score_2) / 2
# print(f"Average CLIP score: {average_clip_score}")
