import torch
import safetensors
from safetensors import safe_open
from safetensors.torch import save_file
import json

# model_path = "F:/models/hy/hy_test-2050/model.safetensors"
# porcelain_path = "F:/models/Stable-diffusion/hunyuan/ckpts/t2i/lora/porcelain/adapter_model.safetensors"
# test_path = "F:/models/hy/hy_test-90/pytorch_lora_weights.safetensors"

# model = safetensors.torch.load_file(test_path)
# print(model.keys())

# model_path = "F:/c_user/.cache/huggingface/hub/models--Tencent-Hunyuan--HunyuanDiT-Diffusers/snapshots/014d2051e135a784ff7f62737d264acc85e762a3/transformer/diffusion_pytorch_model.safetensors"

# dora_path = "F:/models/hy/hy_dora-500/pytorch_lora_weights.safetensors"
# model_path = "F:/models/Stable-diffusion/sdxl/o2/openxl2_030.safetensors"
model_path = "F:/models/unet/new_kolors/diffusion_pytorch_model.fp16.safetensors"
t_model = safetensors.torch.load_file(model_path)
json_file = "print_new_kolors_converted_state_dict.json"
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(list(t_model.keys()), f, indent=4)
# transformer_state_dict = pipe.transformer.state_dict()

# cotton_doll_2_path = "F:/models/Stable-diffusion/hunyuan/ckpts/t2i/lora/cotton_doll_2/adapter_model.safetensors"

# cotton_doll_2 = safetensors.torch.load_file(cotton_doll_2_path)
# print("test")
# model_path = "F:/models/Stable-diffusion/hunyuan/ckpts/t2i/model/pytorch_model_module.pt"
# model = torch.load(model_path, weights_only=True)
# print(model.keys())