from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel
import torch
import os

def predict(model,processor,targets,image_path):
  image = Image.open(image_path)
  inputs = processor(text=targets, images=image, return_tensors="pt",padding=True)
  outputs = model(**inputs)
  logits_per_image = outputs.logits_per_image   
  predicted_label = logits_per_image.argmax(-1).item()
  id2label = {str(i): c for i, c in enumerate(targets)}
  print(image_path)
  print(id2label)
  print(logits_per_image)
  print(id2label[str(predicted_label)])
  return id2label[str(predicted_label)]

def main():
  # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
  # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
  model_path = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
  model = CLIPModel.from_pretrained(model_path)
  processor = CLIPProcessor.from_pretrained(model_path)


  # input_dir = "http://images.cocodataset.org/val2017/000000039769.jpg"
  input_dir = "F:/ImageSet/hands_dataset_above_average/clean-hands"
  file_name = "993849.0.74.jpg"
  image_path = os.path.join(input_dir,file_name)
  
  targets = ['cartoon-style','anime artwork','drawing','painting','sketch']
  result = predict(model,processor,targets,image_path)

  targets = ['fist gesture','no gesture','peace gesture','ok gesture']
  result = predict(model,processor,targets,image_path)
  
  targets = ['holding something','shake hands','shake hands','hands crossed','reach out']
  result = predict(model,processor,targets,image_path)

if __name__ == "__main__":
  main()