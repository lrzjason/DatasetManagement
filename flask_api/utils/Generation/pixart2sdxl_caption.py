import torch
from tqdm import tqdm 
from PIL import Image 
from compel import Compel, ReturnedEmbeddingsType
import hpsv2
import os
from accelerate.utils import set_seed
import json
import gc

from transformers import T5EncoderModel, T5Tokenizer
from diffusers import StableDiffusionXLPipeline, Transformer2DModel, PixArtSigmaPipeline, DPMSolverMultistepScheduler, StableDiffusionXLImg2ImgPipeline, AutoencoderKL, HeunDiscreteScheduler
import random
from PIL.PngImagePlugin import PngInfo
import numpy as np

import sys
# sys.path.append('F:/DatasetManagement/flask_api/utils/SDXL/aesthetic')
# import aesthetic_predict


from diffusers import DPMSolverMultistepScheduler as DefaultDPMSolver

MAX_SEED = np.iinfo(np.int32).max
def sdxl_img2img(pipeline,compel,image,prompt, negative_prompt="", steps=30, strength=0.6,seed=None, guidance_scale=3.5, positive_prompt_embeds=None,positive_pooled_prompt_embeds=None,negative_prompt_embeds=None,negative_pooled_prompt_embeds=None):
    if seed is None:
        seed = random.randint(0, MAX_SEED)
    set_seed(seed)
    if positive_prompt_embeds is None and positive_pooled_prompt_embeds is None:
        positive_prompt_embeds, positive_pooled_prompt_embeds = compel(prompt)
    if negative_prompt_embeds is None and negative_pooled_prompt_embeds is None:
        negative_prompt_embeds,negative_pooled_prompt_embeds = compel(negative_prompt)
    sdxl_image = pipeline(
        image= image,
        strength=strength,
        prompt_embeds=positive_prompt_embeds, 
        pooled_prompt_embeds=positive_pooled_prompt_embeds, 
        negative_prompt_embeds=negative_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=image.width, 
        height=image.height,
    ).images[0]

    del positive_prompt_embeds,positive_pooled_prompt_embeds
    
    return sdxl_image

def pixart_encode(device,tokenizer,text_encoder,prompt):
    prompt_tokens = [prompt]
    # prompt_tokens = pipeline._text_preprocessing(prompt_tokens)
    # max_length = prompt.shape[1]
    prompt_tokens = tokenizer(
        prompt_tokens,
        padding="max_length",
        max_length=300,
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    ).to("cpu")
    # tokenizer(prompts, max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt").to("cpu")
    prompt_attention_mask = prompt_tokens.attention_mask
    prompt_attention_mask = prompt_attention_mask.to("cpu")

    prompt_embeds = text_encoder(
        prompt_tokens.input_ids.to("cpu"), attention_mask=prompt_attention_mask
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds.to(device).to(torch.float16), prompt_attention_mask.to(device).to(torch.float16)
def pixart_generation(pipeline,tokenizer,text_encoder,prompt="", negative_prompt="", steps=30, seed=None, width=512, height=512, guidance_scale=3.5, return_upscaled=True,prompt_embeds=None, prompt_attention_mask=None, negative_prompt_embeds=None, negative_prompt_attention_mask=None):
    if seed is None:
        seed = random.randint(0, MAX_SEED)
    set_seed(seed)
    if prompt_embeds is None:
        prompt_embeds, prompt_attention_mask = pixart_encode(device,tokenizer,text_encoder,prompt)
    if negative_prompt_embeds is None:
        negative_prompt_embeds, negative_prompt_attention_mask = pixart_encode(device,tokenizer,text_encoder,negative_prompt)
    image = pipeline(
        prompt_embeds = prompt_embeds,
        prompt_attention_mask = prompt_attention_mask,
        negative_prompt = None,
        negative_prompt_embeds = negative_prompt_embeds,
        negative_prompt_attention_mask = negative_prompt_attention_mask,
        num_inference_steps = steps,
        guidance_scale = guidance_scale,
        width = width,
        height = height
    ).images[0]

    del prompt_embeds,prompt_attention_mask

    if return_upscaled:
    # upscale image to 1024
        return image.resize((image.width * 2, image.height * 2), resample=Image.Resampling.LANCZOS)
    return image

def init_sdxl(device, weight_dtype, scheduler,vae):
    model_path = "F:/models/Stable-diffusion/sdxl/o2/openxl2_022.safetensors"
    sdxl_pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file(
        model_path, use_safetensors=True, 
        vae=vae,
        torch_dtype=weight_dtype).to(device)
    sdxl_pipeline.scheduler = scheduler
    compel = Compel(tokenizer=[sdxl_pipeline.tokenizer, sdxl_pipeline.tokenizer_2] , text_encoder=[sdxl_pipeline.text_encoder, sdxl_pipeline.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])
    # sdxl_pipeline.unet = torch.compile(sdxl_pipeline.unet, mode="reduce-overhead", fullgraph=True)
    sdxl_pipeline.enable_model_cpu_offload()

    return sdxl_pipeline, compel
    

def init_pixart(device, weight_dtype, scheduler, vae):
    transformer_path = "F:/models/Stable-diffusion/pixart/PixArt-Sigma-XL-2-512-MS"
    transformer = Transformer2DModel.from_pretrained(
        transformer_path, 
        subfolder='transformer', 
        torch_dtype=weight_dtype,
        use_safetensors=True,
        # scheduler = scheduler
    )
    pixart_pipeline_path = "F:/PixArt-sigma/output/pixart_sigma_sdxlvae_T5_diffusers"
    
    text_encoder = T5EncoderModel.from_pretrained(pixart_pipeline_path, subfolder="text_encoder")
    tokenizer = T5Tokenizer.from_pretrained(pixart_pipeline_path, subfolder="tokenizer")
    pixart_pipeline = PixArtSigmaPipeline.from_pretrained(
        pixart_pipeline_path,
        vae = vae,
        text_encoder = None,
        tokenizer = tokenizer,
        transformer=transformer,
        torch_dtype=weight_dtype,
        use_safetensors=True,
    )
    pixart_pipeline.to(device)
    pixart_pipeline.enable_model_cpu_offload()

    return pixart_pipeline, text_encoder, tokenizer


def get_exif_parameters(pos_prompt,neg_prompt,steps,sampler,cfg,seed,size,model_name):
    return f"{pos_prompt}\nNegative prompt: {neg_prompt}\n\nSteps: {steps}, Sampler: {sampler}, CFG scale: {cfg}, Seed: {seed}, Size: {size}, Model: {model_name}"
    # return f"{pos_prompt} Negative prompt: {neg_prompt}, Steps: {steps}, Sampler: {sampler}, CFG scale: {cfg}, Seed: {seed}, Size: {size}"




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
    

output_dir = "F:/ImageSet/pickscore_random_captions_pixart2sdxl"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

input_dir = "F:/ImageSet/pickscore_random_captions/temp"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weight_dtype = torch.float16

scheduler  = DPMSolverMultistepScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, 
    use_lu_lambdas=True,algorithm_type='sde-dpmsolver++',solver_order=2,
    thresholding=True
)
vae_path = "F:/models/VAE/sdxl_vae.safetensors"
vae = AutoencoderKL.from_single_file(
    vae_path
).to(device).to(weight_dtype)
sdxl_pipeline,compel = init_sdxl(device, weight_dtype, scheduler, vae)
pixart_pipeline,t5_text_encoder, t5_tokenizer = init_pixart(device, weight_dtype, scheduler, vae)

# neg_prompt = "deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra_limb, ugly, poorly drawn hands, two heads, child, children, kid, gross, mutilated, disgusting, horrible, scary, evil, old, conjoined, morphed, text, error, glitch, lowres, extra digits, watermark, signature, jpeg artifacts, low quality, unfinished, cropped, Siamese twins, robot eyes, loli, "
neg_prompt = "worst anatomy, worst quality, distortion, greyscale, japenes character,chinese character, blurry background"
sdxl_negative_prompt_embeds,sdxl_negative_pooled_prompt_embeds = compel(neg_prompt)


pixart_negative_prompt_embeds,pixart_negative_prompt_attention_mask = pixart_encode(device,t5_tokenizer,t5_text_encoder,neg_prompt)

image_ext = ['.png','.jpg','.jpeg','.webp']
caption_ext = '.txt'

# output_image_ext = '.webp'
output_image_ext = '.png'

seed = 41254125


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


scheduler = DPMSolverMultistepScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, 
    use_lu_lambdas=True,algorithm_type='dpmsolver++',solver_order=3
)
# scheduler = HeunDiscreteScheduler(
#     beta_start=0.00085, beta_end=0.012, beta_schedule="linear"
# )
sdxl_pipeline.scheduler = scheduler
guidance_scale = 15
sdxl1_guidance_scale = 3.5
sdxl2_guidance_scale = 5
pixart_steps = 50
sdxl1_steps = 40
sdxl2_steps = 30
sdxl1_strength = 0.75
sdxl2_strength = 0.2

debug = False

# sampling_schedule = [999, 845, 730, 587, 443, 310, 193, 116, 53, 13, 0]

sampler = "DPM++ 3M SDE Karras"
model_name="openxlVersion25_v25"

# prefix = 'worst quality, worst anatomy, distortion, abstract, '
prefix = 'creative image of '
suffix = ''

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
            output_file_path = os.path.join(output_dir, image_file)

            filename = file.replace(caption_ext, "")
            pa_output_file_path = os.path.join(output_dir, f"{filename}_pixart{output_image_ext}")
            
            if os.path.exists(output_file_path): continue
            
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

            prompt = f'{prefix}{prompt}{suffix}'
            
            pixart_image = pixart_generation(
                pixart_pipeline,
                tokenizer=t5_tokenizer,
                text_encoder=t5_text_encoder,
                prompt=prompt,
                negative_prompt_embeds=pixart_negative_prompt_embeds,
                negative_prompt_attention_mask = pixart_negative_prompt_attention_mask,
                steps=pixart_steps,
                width=int(width_height[0]/2),
                height=int(width_height[1]/2),
                guidance_scale=guidance_scale,
                seed=seed,
                )
            
            sdxl_image = sdxl_img2img(
                sdxl_pipeline,
                compel,
                pixart_image,
                prompt,
                # neg_prompt,
                steps=sdxl1_steps,
                strength=sdxl1_strength,
                seed=seed,
                guidance_scale=sdxl1_guidance_scale,
                negative_prompt_embeds=sdxl_negative_prompt_embeds,
                negative_pooled_prompt_embeds=sdxl_negative_pooled_prompt_embeds)
            
            
            # sdxl_image_2 = sdxl_img2img(
            #     sdxl_pipeline,
            #     compel,
            #     sdxl_image,
            #     prompt,
            #     # neg_prompt,
            #     steps=sdxl2_steps,
            #     strength=sdxl2_strength,
            #     seed=seed,
            #     guidance_scale=sdxl2_guidance_scale,
            #     negative_prompt_embeds=sdxl_negative_prompt_embeds,
            #     negative_pooled_prompt_embeds=sdxl_negative_pooled_prompt_embeds)

            size = f"{width_height[0]}x{width_height[1]}"
            # exif_dict = piexif.load(ays_pag_image.info['exif'])
            # parameters = get_exif_parameters(prompt,neg_prompt,pixart_steps+sdxl1_steps+sdxl2_steps,sampler,guidance_scale,seed,size,model_name)
            parameters = get_exif_parameters(prompt,neg_prompt,pixart_steps+sdxl1_steps,sampler,guidance_scale,seed,size,model_name)
            metadata = PngInfo()
            # Add metadata
            metadata.add_text("parameters", parameters)

            if debug:
                pixart_image.save(pa_output_file_path, pnginfo=metadata)

                sdxl1_output_file_path = os.path.join(output_dir, f"{filename}_sdxl1{output_image_ext}")
                sdxl_image.save(sdxl1_output_file_path, pnginfo=metadata)

                
                # sdxl2_output_file_path = os.path.join(output_dir, f"{filename}_sdxl2{output_image_ext}")
                # sdxl_image_2.save(sdxl2_output_file_path, pnginfo=metadata)
                # sdxl_image_2.save(output_file_path, pnginfo=metadata)

                sdxl_image.save(output_file_path, pnginfo=metadata)
                
            else:
                # sdxl_image_2.save(output_file_path, pnginfo=metadata)
                sdxl_image.save(output_file_path, pnginfo=metadata)
            # del pixart_image, sdxl_image, sdxl_image_2
            del pixart_image, sdxl_image


            # save text as seperate file
            txt_file_path = output_file_path.replace(output_image_ext, '.txt')
            with open(txt_file_path, 'w', encoding="utf-8") as f:
                f.write(prompt)

            # release memory
            gc.collect()
            torch.cuda.empty_cache()
