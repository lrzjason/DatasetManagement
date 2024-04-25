import torch

path = 'F:/models/Stable-diffusion/stable_cascade/stage_b_bf16.safetensors'
model = torch.load(path)
print(model)