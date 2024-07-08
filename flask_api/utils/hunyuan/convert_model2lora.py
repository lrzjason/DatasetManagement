import torch
import safetensors
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    model_path = "F:/models/Stable-diffusion/hunyuan/ckpts/t2i/lora/cotton_doll/adapter_model.safetensors"
    new_model_path = "F:/models/Stable-diffusion/hunyuan/ckpts/t2i/lora/cotton_doll/adapter_model_new.safetensors"
    model = safetensors.torch.load_file(model_path)

    model_keys = model.keys()

    new_model = {}
    for key in model_keys:
        if "lora_A" in key or 'lora_B' in key:
            new_key = key
            if ".default" in key:
                new_key = "base_model.model."+key.replace(".default","")
                new_model[new_key] = model[key]
                # print(new_key)
                # return
    print(new_model.keys())         
    # save new model
    save_file(new_model, new_model_path)
    
if __name__ == "__main__":
    main()