# import webdataset as wds
from PIL import Image
import io
import matplotlib.pyplot as plt
import os
import json

from warnings import filterwarnings


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
# from torchvision import datasets, transforms
import tqdm

from os.path import join
# from datasets import load_dataset
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json

import clip


from PIL import Image, ImageFile

def init_model():
    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
    sac_path = 'F:/T2ITrainer/aesthetic/sac+logos+ava1-l14-linearMSE.pth'
    s = torch.load(sac_path)   # load the model you trained previously or the model available in this repo
    model.load_state_dict(s)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    image_encoder, preprocess = clip.load("ViT-L/14", device=device)  #RN50x64   
    return model,image_encoder,preprocess,device

def predict(model,image_encoder,preprocess,image_input,device):
    if isinstance(image_input, str):
        pil_image = Image.open(image_input)
    elif isinstance(image_input, np.ndarray):
        pil_image = Image.fromarray(image_input.astype('uint8'), 'RGB')
    else:
        pil_image = image_input

    image = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = image_encoder.encode_image(image)

    im_emb_arr = normalized(image_features.cpu().detach().numpy() )

    prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
    return prediction.item()


#####  This script will predict the aesthetic score for this image file:

# img_path = "F://ImageSet//dump//mobcup_output_deleted//#Cute girl.jpg"
# img_path = "F://ImageSet//dump//mobcup_output//8k - Autumn Mountains.jpg"

# filename = "x1b413e1dea60b813"
# subdir = "000031"

# ori_img_path = f"F:/ImageSet/improved_aesthetics_6.5plus/output/{subdir}/{filename}.webp"

# centered_img_path = f"F:/T2ITrainer/object_detection/{filename}_centered.webp"

# pixel_preserved_img_path = f"F:/T2ITrainer/object_detection/{filename}_preserved.webp"

# pixel_preserved_img_path = f"F:/T2ITrainer/object_detection/{filename}_simple.webp"

# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


# model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

# s = torch.load("sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo

# model.load_state_dict(s)

# model.to("cuda")
# model.eval()


# device = "cuda" if torch.cuda.is_available() else "cpu"
# model2, preprocess = clip.load("ViT-L/14", device=device)  #RN50x64   

# ori_img_path = f"F:/ImageSet/improved_aesthetics_6.5plus/output/000029/{filename}"

# centered_img_path = f"F:/T2ITrainer/object_detection/283fac54f414613d_crop_resize.webp"

# pixel_preserved_img_path = f"F:/T2ITrainer/object_detection/{filename}"

# ori_score = predict(model,model2,ori_img_path)
# center_score = predict(model,model2,centered_img_path)
# pixel_score = predict(model,model2,pixel_preserved_img_path)

# print(filename)
# print(f"Original: {ori_score}")
# print(f"Centered: {center_score}")
# print(f"Pixel Preserved: {pixel_score}")
