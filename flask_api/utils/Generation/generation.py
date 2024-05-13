import torch
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm 
from PIL import Image 
from compel import Compel, ReturnedEmbeddingsType
# import hpsv2
import os

def convert_image_to_generation_width_height(width, height):
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
    

output_dir = "F:/ImageSet/openxl2_generation"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

input_dir = "F:/ImageSet/openxl2_dataset"

# model_path = "F:/models/Stable-diffusion/sdxl/o2/o2b9_o14_115_00001_.safetensors"
model_path = "F:/models/Stable-diffusion/sdxl/b9/openxl2_b9.safetensors"

pipeline = StableDiffusionXLPipeline.from_single_file(
    model_path,variant="fp16", use_safetensors=True, 
    torch_dtype=torch.float16).to("cuda")

compel = Compel(tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2] , text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])

for subdir in tqdm(os.listdir(input_dir),position=0):
    subdir_path = os.path.join(input_dir, subdir)
    for file in tqdm(os.listdir(subdir_path),position=1):
        if file.endswith(".txt"):
            # create output subdir
            output_subdir_path = os.path.join(output_dir, subdir)
            output_file_path = os.path.join(output_subdir_path, file.replace('.txt', '.jpg'))
            if not os.path.exists(output_subdir_path):
                os.mkdir(output_subdir_path)
            
            # skip exist image
            if os.path.exists(output_file_path): continue
            # read file
            prompt = ''
            file_path = os.path.join(subdir_path, file)

            image_path = file_path.replace('.txt', '.jpg')
            # print(image_path)
            ori_image = Image.open(image_path) 
            width_height = convert_image_to_generation_width_height(ori_image.width,ori_image.height)
            # print(width_height)
            # break
            with open(file_path, "r") as f:
                prompt = f.read()
                f.close()
            prompt = prompt.replace('\n', ' ')
            
            conditioning, pooled = compel(prompt)
            guidance_scale = 7
            steps = 30
            # generate image
            image = pipeline(prompt_embeds=conditioning, 
                             pooled_prompt_embeds=pooled, 
                             num_inference_steps=steps,
                             guidance_scale=guidance_scale,
                             width=width_height[0], 
                             height=width_height[1]
                             ).images[0]

            # image.show()
            
            image.save(output_file_path)

            # Tried HPSv2 compared to original image
            # generated result: 0.36572265625
            # original result: 0.2137451171875
            # as of this point, HPSv2 is able to evaluate images quality

            # generated_result = hpsv2.score(image, prompt, hps_version="v2.1")[0]
            # hpsv2.score(image, prompt, hps_version="v2.1")[0]
            # print(f'generated_result:{generated_result}')
            # image_path = file_path.replace('.txt', '.jpg')
            # original_result = hpsv2.score(image_path, prompt, hps_version="v2.1")[0]
            # print(f'original_result:{original_result}')
            # break
    # break
