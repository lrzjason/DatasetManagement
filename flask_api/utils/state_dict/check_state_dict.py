import torch


# model_path = "F:/models/dit/output/checkpoints/epoch_0_step_41.pth"
model_path = "F:/models/Stable-diffusion/pixart/PixArt-Sigma-XL-2-1024-MS.pth"
model = torch.load(model_path)

state_dict = model['state_dict']
with open('keys.txt', 'w') as f:
    print(state_dict.keys(), file=f)