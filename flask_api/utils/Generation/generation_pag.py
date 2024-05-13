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

import sys
sys.path.append('F:/DatasetManagement/flask_api/utils/SDXL/aesthetic')
import aesthetic_predict

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
output_dir = "F:/ImageSet/openxl2_realism_test_output"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# input_dir = "F:/ImageSet/openxl2_realism_test"
# input_dir = "F:/ImageSet/openxl2_realism_above_average"
input_dir = "F:/ImageSet/openxl2_realism_test/test"
# input_dir = ""

# model_path = "F:/models/Stable-diffusion/sdxl/o2/o2b9_o14_115_00001_.safetensors"
model_path = "F:/models/Stable-diffusion/sdxl/o2/openxl2_014.safetensors"

pipeline = StableDiffusionXLPipeline.from_single_file(
    model_path,variant="fp16", use_safetensors=True, 
    torch_dtype=torch.float16).to("cuda")

compel = Compel(tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2] , text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])

neg_prompt = "deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra_limb, ugly, poorly drawn hands, two heads, child, children, kid, gross, mutilated, disgusting, horrible, scary, evil, old, conjoined, morphed, text, error, glitch, lowres, extra digits, watermark, signature, jpeg artifacts, low quality, unfinished, cropped, Siamese twins, robot eyes, loli, "
negative_prompt_embeds,negative_pooled_prompt_embeds = compel(neg_prompt)

image_ext = ['.png','.jpg','.jpeg','.webp']
caption_ext = '.txt'

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

scheduler = DPMSolverMultistepScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, 
    use_lu_lambdas=True,algorithm_type='dpmsolver++',solver_order=3
)

# scheduler = DEISMultistepScheduler(
#     beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, 
#     solver_order=3
# )

pipeline.scheduler = scheduler

for subdir in tqdm(os.listdir(input_dir),position=0):
    subdir_path = os.path.join(input_dir, subdir)
    files = os.listdir(subdir_path)
    text_files = [file for file in files if file.endswith(caption_ext)]
    print('text_files:',text_files)
    for file in tqdm(text_files,position=1):
        if file.endswith(caption_ext):
            count +=1
            # create output subdir
            # output_high_dir = os.path.join(output_dir, "high")
            # if not os.path.exists(output_high_dir):
            #     os.mkdir(output_high_dir)

            # output_low_dir = os.path.join(output_dir, "low")
            # if not os.path.exists(output_low_dir):
            #     os.mkdir(output_low_dir)
            
            output_ori_dir = os.path.join(output_dir, "ori")
            if not os.path.exists(output_ori_dir):
                os.mkdir(output_ori_dir)

            output_pag_dir = os.path.join(output_dir, "pag")
            if not os.path.exists(output_pag_dir):
                os.mkdir(output_pag_dir)
            
            image_file = get_image_file(subdir_path,file)

            # output_high_file_path = os.path.join(output_high_dir, image_file)
            # output_low_file_path = os.path.join(output_low_dir, image_file)
            output_ori_file_path = os.path.join(output_ori_dir, image_file)
            output_pag_file_path = os.path.join(output_pag_dir, image_file)
            # skip exist image
            # if os.path.exists(output_high_file_path) and os.path.exists(output_low_file_path): continue
            
            if os.path.exists(output_ori_file_path) and os.path.exists(output_pag_file_path): continue
            
            # read file
            prompt = ''
            file_path = os.path.join(subdir_path, file)
            image_path = os.path.join(subdir_path, image_file)

            # image_path = file_path.replace(caption_ext, image_ext)
            # print(image_path)
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
                
            # print(width_height)
            # break
            with open(file_path, "r", encoding="utf-8") as f:
                prompt = f.read()
                f.close()
            prompt = prompt.replace('/n', ' ')
            
            conditioning, pooled = compel(prompt)
            guidance_scale = 3.5
            steps = 30
            if not os.path.exists(output_ori_file_path):
                # generate image
                ori_image = pipeline(prompt_embeds=conditioning, 
                                pooled_prompt_embeds=pooled, 
                                negative_prompt_embeds=negative_prompt_embeds,
                                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                                num_inference_steps=steps,
                                guidance_scale=guidance_scale,
                                width=width_height[0], 
                                height=width_height[1],
                                euler_at_final=True,
                                #  pag_scale=1.5,
                                #  pag_applied_layers_index=['m0']
                                ).images[0]
                ori_image.save(output_ori_file_path)
                del ori_image
            
            if not os.path.exists(output_pag_file_path):
                # generate image
                pag_image = pipeline(prompt_embeds=conditioning, 
                                pooled_prompt_embeds=pooled, 
                                negative_prompt_embeds=negative_prompt_embeds,
                                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                                num_inference_steps=steps,
                                guidance_scale=guidance_scale,
                                width=width_height[0], 
                                height=width_height[1],
                                euler_at_final=True,
                                pag_scale=7,
                                pag_applied_layers_index=['m0']
                                ).images[0]
                pag_image.save(output_pag_file_path)
                del pag_image

            # release memory
            gc.collect()
            torch.cuda.empty_cache()
