import torch
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm 
from PIL import Image 
from compel import Compel, ReturnedEmbeddingsType
import hpsv2
import os
from accelerate.utils import set_seed
import json
import gc

from diffusers import DPMSolverMultistepScheduler
import random
import numpy as np
from PIL.PngImagePlugin import PngInfo
import piexif
import hashlib

import sys
sys.path.append('F:/DatasetManagement/flask_api/utils/SDXL/aesthetic')
import aesthetic_predict

# Function to calculate the hash of a model file
# def calculate_model_hash(file_path):
#     # Create a hash object
#     sha256_hash = hashlib.sha256()
    
#     # Open the file in binary mode and read chunks
#     with open(file_path, "rb") as f:
#         for byte_block in iter(lambda: f.read(4096), b""):
#             sha256_hash.update(byte_block)
    
#     # Return the hexadecimal digest of the hash
#     return sha256_hash.hexdigest()
def get_exif_parameters(pos_prompt,neg_prompt,steps,sampler,cfg,seed,size,model_name):
    return f"{pos_prompt}\nNegative prompt: {neg_prompt}\n\nSteps: {steps}, Sampler: {sampler}, CFG scale: {cfg}, Seed: {seed}, Size: {size}, Model: {model_name}"
    # return f"{pos_prompt} Negative prompt: {neg_prompt}, Steps: {steps}, Sampler: {sampler}, CFG scale: {cfg}, Seed: {seed}, Size: {size}"

# Add support for setting custom timesteps
class DPMSolverMultistepSchedulerAYS(DPMSolverMultistepScheduler):
    def set_timesteps(
        self, num_inference_steps=None, device=None, 
        timesteps=None
    ):
        if timesteps is None:
            super().set_timesteps(num_inference_steps, device)
            return
        
        all_sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        self.sigmas = torch.from_numpy(all_sigmas[timesteps])
        self.timesteps = torch.tensor(timesteps[:-1]).to(device=device, dtype=torch.int64) # Ignore the last 0
        
        self.num_inference_steps = len(timesteps)

        self.model_outputs = [
            None,
        ] * self.config.solver_order
        self.lower_order_nums = 0

        # add an index counter for schedulers that allow duplicated timesteps
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication


# Define the standard dimensions
height_dimensions = {
    '640, 1536': (640, 1536),
    '768, 1344': (768, 1344),
    '832, 1216': (832, 1216),
    '896, 1152': (896, 1152),
    '1024, 1024': (1024, 1024)
}

width_dimensions = {
    '1024, 1024': (1024, 1024),
    '1152, 896': (1152, 896),
    '1216, 832': (1216, 832),
    '1344, 768': (1344, 768),
    '1536, 640': (1536, 640)
}

def convert_image_to_generation_width_height(width, height):

    if width > height:
        # If the width is greater than the height, use the width as the standard dimension
        standard_dimensions = width_dimensions
    else:
        # If the height is greater than the width, use the height as the standard dimension
        standard_dimensions = height_dimensions

    # Calculate the aspect ratio of the original image
    aspect_ratio = width / height

    # Find the closest standard aspect ratio
    closest_ratio = min(standard_dimensions.items(), key=lambda x: abs((x[1][0] / x[1][1]) - aspect_ratio))

    # Return the dimensions corresponding to the closest aspect ratio
    return closest_ratio[1]
    

# output_dir = "F:/ImageSet/openxl2_realism_above_average_pag"
# output_dir = "F:/ImageSet/openxl2_realism_test_output"
# if not os.path.exists(output_dir):
#     os.mkdir(output_dir)

# # input_dir = "F:/ImageSet/openxl2_realism_test"
# # input_dir = "F:/ImageSet/openxl2_realism_above_average"
# input_dir = "F:/ImageSet/openxl2_realism_test/test"
# # input_dir = ""

# # model_path = "F:/models/Stable-diffusion/sdxl/o2/o2b9_o14_115_00001_.safetensors"
model_path = "F:/models/Stable-diffusion/sdxl/o2/openxl2_016e.safetensors"

# model_hash = calculate_model_hash(model_path)
# print(model_hash)

pipeline = StableDiffusionXLPipeline.from_single_file(
    model_path,variant="fp16", use_safetensors=True, 
    torch_dtype=torch.float16).to("cuda")

compel = Compel(tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2] , text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])

pos_prompt = "photo of a girl"
positive_prompt_embeds,positive_pooled_prompt_embeds = compel(pos_prompt)

neg_prompt = "deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra_limb, ugly, poorly drawn hands, two heads, child, children, kid, gross, mutilated, disgusting, horrible, scary, evil, old, conjoined, morphed, text, error, glitch, lowres, extra digits, watermark, signature, jpeg artifacts, low quality, unfinished, cropped, Siamese twins, robot eyes, loli,"
negative_prompt_embeds,negative_pooled_prompt_embeds = compel(neg_prompt)

image_ext = ['.png','.jpg','.jpeg','.webp']
caption_ext = '.txt'

output_image_ext = '.webp'

seed = 1232


set_seed(seed)

def get_image_file(ori_dir,text_file):
    for ext in image_ext:
        image_file = text_file.replace(caption_ext, ext)
        if os.path.exists(os.path.join(ori_dir, image_file)): 
            return image_file
        else:
            image_file = text_file.replace(caption_ext, output_image_ext)
            return image_file

results = []
save_json_every_progress = 50
count = 0

# eval
ae_model,image_encoder,preprocess,device = aesthetic_predict.init_model()

# scheduler = DefaultDPMSolver(
#     beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, 
#     use_lu_lambdas=True,algorithm_type='dpmsolver++',solver_order=3
# )

ays_scheduler = DPMSolverMultistepSchedulerAYS(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, 
    use_lu_lambdas=True,algorithm_type='dpmsolver++',solver_order=3
)

scheduler = DPMSolverMultistepScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, 
    use_lu_lambdas=True,algorithm_type='dpmsolver++',solver_order=3
)

pipeline.scheduler = ays_scheduler
guidance_scale = 3.5
steps = 10

if random.random() > 0.5:
    values = list(width_dimensions.values())
    width_height = random.choice(values)
else:
    values =  list(height_dimensions.values())
    width_height = random.choice(values)

sampling_schedule = [999, 845, 730, 587, 443, 310, 193, 116, 53, 13, 0]

ays_pag_image = pipeline(prompt_embeds=positive_prompt_embeds, 
                pooled_prompt_embeds=positive_pooled_prompt_embeds, 
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                timesteps=sampling_schedule,
                width=width_height[0], 
                height=width_height[1],
                euler_at_final=True,
                pag_scale=7,
                pag_applied_layers_index=['m0']
                ).images[0]
# show image
ays_pag_image.show()
sampler = "DPM++ 3M Karras"
model_name="openxlVersion23_v23e"
size = f"{width_height[0]}x{width_height[1]}"
# exif_dict = piexif.load(ays_pag_image.info['exif'])
parameters = get_exif_parameters(pos_prompt,neg_prompt,steps,sampler,guidance_scale,seed,size,model_name)
# test_p = '6 boys,ink painting of Six samurai stood back to back together ,The background is high mountains, with a full moon in the sky, rule of thirds composition, beautiful lighting, Ultrahigh Resolution, Volcanic Eruption, futuristic, xxTemplexx Negative prompt: aidv1-neg, animestylenegativeembedding_dreamshaper, bad-artist, bad-artist-anime, sketch by bad-artist, painting by bad-artist, photograph by bad-artist, bad-picture-chill-75v, badhandv4, badhandv5, badv3, badv4, badv5, bad_prompt, bad_prompt_version2, EasyNegative, ng_deepnegative_v1_75t, verybadimagenegative_v1.2-6400, verybadimagenegative_v1.3, Unspeakable-Horrors-Composition-4v,(worst quality, low quality:1.4), (raw photo) Steps: 20, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 1073211052, Size: 1024x1024, Model hash: 92b799d43e, Model: pastelboys2D_v30, ENSD: 31337'
metadata = PngInfo()
# Add metadata
metadata.add_text("parameters", parameters)

# print(exif_dict)
ays_pag_image.save("test.png", pnginfo=metadata)
# del pag_image


# pipeline.scheduler = scheduler
# steps = 30

# pag_image = pipeline(prompt_embeds=positive_prompt_embeds, 
#                 pooled_prompt_embeds=positive_pooled_prompt_embeds, 
#                 negative_prompt_embeds=negative_prompt_embeds,
#                 negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
#                 num_inference_steps=steps,
#                 guidance_scale=guidance_scale,
#                 width=width_height[0], 
#                 height=width_height[1],
#                 euler_at_final=True,
#                 pag_scale=7,
#                 pag_applied_layers_index=['m0']
#                 ).images[0]

# pag_image.show()

