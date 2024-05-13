
from diffusers import StableDiffusionXLPipeline

model_path = "F:/models/Stable-diffusion/sdxl/o2/openxl2_023.safetensors"

output_path = "F:/models/Stable-diffusion/sdxl/o2/openxl25_diffusers"

pipe = StableDiffusionXLPipeline.from_single_file(model_path)
pipe.save_pretrained(output_path)