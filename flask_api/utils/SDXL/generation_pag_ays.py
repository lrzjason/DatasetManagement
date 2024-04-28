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

from diffusers import DPMSolverMultistepScheduler,DEISMultistepScheduler
import random
from PIL.PngImagePlugin import PngInfo
import numpy as np

import sys
sys.path.append('F:/DatasetManagement/flask_api/utils/SDXL/aesthetic')
import aesthetic_predict

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
output_dir = "F:/ImageSet/pickscore_random_captions_pag_ays"
# output_dir = "F:/ImageSet/pickscore_random_captions_ays"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# input_dir = "F:/ImageSet/openxl2_realism_test"
# input_dir = "F:/ImageSet/openxl2_realism_above_average"
# input_dir = "F:/ImageSet/openxl2_realism_test"
input_dir = "F:/ImageSet/pickscore_random_captions"

# model_path = "F:/models/Stable-diffusion/sdxl/o2/o2b9_o14_115_00001_.safetensors"
model_path = "F:/models/Stable-diffusion/sdxl/o2/openxl2_016e.safetensors"

pipeline = StableDiffusionXLPipeline.from_single_file(
    model_path,variant="fp16", use_safetensors=True, 
    torch_dtype=torch.float16).to("cuda")

compel = Compel(tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2] , text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])

neg_prompt = "deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra_limb, ugly, poorly drawn hands, two heads, child, children, kid, gross, mutilated, disgusting, horrible, scary, evil, old, conjoined, morphed, text, error, glitch, lowres, extra digits, watermark, signature, jpeg artifacts, low quality, unfinished, cropped, Siamese twins, robot eyes, loli, "
negative_prompt_embeds,negative_pooled_prompt_embeds = compel(neg_prompt)

image_ext = ['.png','.jpg','.jpeg','.webp']
caption_ext = '.txt'

# output_image_ext = '.webp'
output_image_ext = '.webp'

seed = 1231231


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

# scheduler = DPMSolverMultistepScheduler(
#     beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, 
#     use_lu_lambdas=True,algorithm_type='dpmsolver++',solver_order=3
# )

ays_scheduler = DPMSolverMultistepSchedulerAYS(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, 
    use_lu_lambdas=True,algorithm_type='dpmsolver++',solver_order=3
)


# scheduler = DEISMultistepScheduler(
#     beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, 
#     solver_order=3
# )

pipeline.scheduler = ays_scheduler
guidance_scale = 3.5
steps = 10

sampling_schedule = [999, 845, 730, 587, 443, 310, 193, 116, 53, 13, 0]

sampler = "DPM++ 3M SDE Karras"
model_name="openxlVersion23_v23e"



for subdir in tqdm(os.listdir(input_dir),position=0):
    subdir_path = os.path.join(input_dir, subdir)
    files = os.listdir(subdir_path)
    text_files = [file for file in files if file.endswith(caption_ext)]
    print('text_files:',text_files)
    for file in tqdm(text_files,position=1):
        if file.endswith(caption_ext):
            count +=1
            # create output subdir
            image_file = get_image_file(subdir_path,file)

            # only generate one file
            output_pag_file_path = os.path.join(output_dir, image_file)
            
            if os.path.exists(output_pag_file_path): continue
            
            # read file
            prompt = ''
            file_path = os.path.join(subdir_path, file)
            image_path = os.path.join(subdir_path, image_file)

            if os.path.exists(image_path):
                ori_image = Image.open(image_path) 
                width_height = convert_image_to_generation_width_height(ori_image.width,ori_image.height)
            else:
                # roll random, if > 50, randomly select a width_height from width set
                if random.random() > 0.5:
                    values = list(width_dimensions.values())
                    width_height = random.choice(values)
                else:
                    values =  list(height_dimensions.values())
                    width_height = random.choice(values)
            
            # read prompt from file
            with open(file_path, "r", encoding="utf-8") as f:
                prompt = f.read()
                f.close()
            prompt = prompt.replace('/n', ' ')
            
            positive_prompt_embeds, positive_pooled_prompt_embeds = compel(prompt)
            pag_image = pipeline(prompt_embeds=positive_prompt_embeds, 
                            pooled_prompt_embeds=positive_pooled_prompt_embeds, 
                            negative_prompt_embeds=negative_prompt_embeds,
                            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                            num_inference_steps=steps,
                            guidance_scale=guidance_scale,
                            timesteps=sampling_schedule,
                            width=width_height[0], 
                            height=width_height[1],
                            # euler_at_final=True,
                            # pag_scale=7,
                            # pag_applied_layers_index=['m0']
                            ).images[0]
            
            
            size = f"{width_height[0]}x{width_height[1]}"
            # exif_dict = piexif.load(ays_pag_image.info['exif'])
            parameters = get_exif_parameters(prompt,neg_prompt,steps,sampler,guidance_scale,seed,size,model_name)
            metadata = PngInfo()
            # Add metadata
            metadata.add_text("parameters", parameters)
            pag_image.save(output_pag_file_path, pnginfo=metadata)
            del pag_image

            # save metadata as seperate file
            metadata_file_path = output_pag_file_path.replace(output_image_ext, '.metadata')
            with open(metadata_file_path, 'w', encoding="utf-8") as f:
                f.write(parameters)
            
            # save text as seperate file
            txt_file_path = output_pag_file_path.replace(output_image_ext, '.txt')
            with open(txt_file_path, 'w', encoding="utf-8") as f:
                f.write(prompt)

            # release memory
            gc.collect()
            torch.cuda.empty_cache()
