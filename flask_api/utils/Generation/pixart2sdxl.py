import torch
from diffusers import Transformer2DModel, PixArtSigmaPipeline, DPMSolverMultistepScheduler, StableDiffusionXLImg2ImgPipeline
from PIL import Image
from compel import Compel, ReturnedEmbeddingsType
from accelerate.utils import set_seed
import random
import numpy as np

MAX_SEED = np.iinfo(np.int32).max
def sdxl_img2img(pipeline,compel,image,prompt, negative_prompt, steps, strength=0.6,seed=None, guidance_scale=3.5):
    if seed is None:
        seed = random.randint(0, MAX_SEED)
    set_seed(seed)
    positive_prompt_embeds, positive_pooled_prompt_embeds = compel(prompt)
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
    
    return sdxl_image
def pixart_generation(pipeline,prompt, negative_prompt, steps, seed=None, width=512, height=512, guidance_scale=3.5, return_upscaled=True):
    if seed is None:
        seed = random.randint(0, MAX_SEED)
    set_seed(seed)
    image = pipeline(
        prompt = prompt,
        neg_prompt = negative_prompt,
        num_inference_steps = steps,
        guidance_scale = guidance_scale,
        width = width,
        height = height
    ).images[0]

    if return_upscaled:
    # upscale image to 1024
        return image.resize((image.width * 2, image.height * 2), resample=Image.Resampling.LANCZOS)
    return image

def init_sdxl(device, weight_dtype, scheduler):
    model_path = "F:/models/Stable-diffusion/sdxl/o2/openxl2_022.safetensors"
    sdxl_pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file(
        model_path, use_safetensors=True, 
        torch_dtype=weight_dtype).to(device)
    sdxl_pipeline.scheduler = scheduler
    compel = Compel(tokenizer=[sdxl_pipeline.tokenizer, sdxl_pipeline.tokenizer_2] , text_encoder=[sdxl_pipeline.text_encoder, sdxl_pipeline.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])

    return sdxl_pipeline, compel
    

def init_pixart(device, weight_dtype, scheduler):
    transformer_path = "F:/models/Stable-diffusion/pixart/PixArt-Sigma-XL-2-512-MS"
    transformer = Transformer2DModel.from_pretrained(
        transformer_path, 
        subfolder='transformer', 
        torch_dtype=weight_dtype,
        use_safetensors=True,
        scheduler = scheduler
    )
    pixart_pipeline_path = "F:/PixArt-sigma/output/pixart_sigma_sdxlvae_T5_diffusers"
    
    pixart_pipeline = PixArtSigmaPipeline.from_pretrained(
        pixart_pipeline_path,
        transformer=transformer,
        torch_dtype=weight_dtype,
        use_safetensors=True,
    )
    pixart_pipeline.to(device)

    # Enable memory optimizations.
    pixart_pipeline.enable_model_cpu_offload()

    return pixart_pipeline

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16

    scheduler  = DPMSolverMultistepScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, 
        use_lu_lambdas=True,algorithm_type='dpmsolver++',solver_order=3
    )
    pixart_pipeline = init_pixart(device, weight_dtype, scheduler)
    sdxl_pipeline,compel = init_sdxl(device, weight_dtype, scheduler)
    prompt = "The painting portrays a group of panda soldiers standing guard on the snow-covered peaks of a mountain range. Their faces are stern and resolute, a testament to their unwavering dedication to their duty. They wear camouflage gear that blends seamlessly with the surrounding snow and rock, and their weapons are slung over their shoulders, ready for action at a moment's notice. Despite the harsh conditions, the soldiers remain vigilant, their eyes scanning the horizon for any signs of movement. The painting captures the essence of their unyielding spirit, their commitment to protecting their homeland and its people, even in the face of adversity. The snow-covered peaks and the vast expanse of the sky serve as a backdrop to their steadfast determination, a symbol of the challenges they face and the heights they are willing to scale to overcome them."
    neg_prompt = "black and white image, japenes character,chinese character,white background, simple background, black background,worst anatomy, worst quality, distortion"
    steps = 50
    guidance_scale = 3
    width = 512
    height = 512

    seed = 1234

    pixart_image = pixart_generation(
        pixart_pipeline,
        prompt,
        neg_prompt,
        steps,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        # seed=seed
        )
    pixart_image.save("./pixart_image.png")

    strength = 0.65
    sdxl_image = sdxl_img2img(
        sdxl_pipeline,
        compel,
        pixart_image,
        prompt,
        neg_prompt,
        steps,
        strength=strength,
        # seed=seed,
        guidance_scale=guidance_scale)
    sdxl_image.save("./sdxl_image.png")

if __name__ == "__main__":
    main()