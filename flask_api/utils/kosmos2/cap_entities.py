from PIL import Image
import requests
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration
import cv2
import os
from tqdm import tqdm 
import numpy
import torch
import clip

from torch.nn import CosineSimilarity
import numpy as np

from transformers import AutoModelForCausalLM, LlamaTokenizer

def get_cog_result(cog_model,cog_tokenizer,device,torch_type,image,prompt="",starts_with='The image showcases '):
    # add any transformers params here.
    gen_kwargs = {
        'min_new_tokens':100,
        'max_new_tokens':350,
        'num_beams':1,
        'length_penalty':1,
        'top_k':60,
        'top_p':0.6,
        'repetition_penalty': 1.15,
        'no_repeat_ngram_size':0,
        "do_sample": True,
        "temperature": 0.6,
    } 
    if prompt == "":
        prompt = f'Describe the image precisely, detailing every element, interaction and background. Include composition, angle and perspective. Use only facts and concise language; avoid interpretations or speculation:'
    
    # if starts_with == "":
    #     starts_with = f'The image showcases '
    query = f'Question: {prompt} Answer: {starts_with}'
    history = []
    input_by_model = cog_model.build_conversation_input_ids(cog_tokenizer, query=query, history=history, images=[image])

    prepare_images = []
    if gen_kwargs['num_beams'] > 1:
        prepare_images = [[input_by_model['images'][0].to(device).to(torch_type)] for _ in range(gen_kwargs['num_beams'])]
    else:
        prepare_images = [[input_by_model['images'][0].to(device).to(torch_type)]] if image is not None else None
    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(device),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(device),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(device),
        'images': prepare_images,
    }
    if 'cross_images' in input_by_model and input_by_model['cross_images']:
        inputs['cross_images'] = [[input_by_model['cross_images'][0].to(device).to(torch_type)]]

    
    response = ""
    with torch.no_grad():
        outputs = cog_model.generate(**inputs, **gen_kwargs
                                )
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = cog_tokenizer.decode(outputs[0])
        response = response.split("</s>")[0]
    return response


def get_clip_score(clip_model,clip_preprocess,image,text,device):
    image = Image.fromarray(np.uint8(image)).convert('RGB')
    # Prepare your image and text
    image_features = clip_preprocess(image).unsqueeze(0).to(device)
    text = clip.tokenize([text]).to(device)

    # Obtain the embeddings
    with torch.no_grad():
        image_features = clip_model.encode_image(image_features)
        text_features = clip_model.encode_text(text)

    # Normalize the embeddings
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity
    cosine_sim = CosineSimilarity(dim=1, eps=1e-6)
    similarity_score = cosine_sim(image_features, text_features)

    return similarity_score[0]

def get_biggest_features_bbox(model,processor,image,draw=False):
    open_cv_image = numpy.array(image)
    # # Convert RGB to BGR
    image = open_cv_image[:, :, ::-1].copy()

    height, width, _ = image.shape
    prompt = "<grounding> the main objects of this image are:"

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=64,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    _, entities = processor.post_process_generation(generated_text)

    names = []
    # please help to draw the bounding box
    for entity in entities:
        print(entity)
        name,_,bboxes = entity
        names.append(name)
        if draw:
            for bbox in bboxes:
                x, y, x2, y2 = bbox
                start_pt = (int(x*width+0.5), int(y*height+0.5))
                end_pt = (int(x2*width+0.5), int(y2*height+0.5))
                # print(start_pt, end_pt)
                image = cv2.rectangle(image, start_pt, end_pt, (0, 255, 0), 2)
                image = cv2.putText(image, name, start_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
    
    if draw:
        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return names



def main():
    input_dir = "F:/ImageSet/openxl2_realism"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load kosmos2 model
    model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224").to(device)
    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224", return_tensors="pt")

    
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    torch_type = torch.bfloat16

    cog_tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
    cog_model = AutoModelForCausalLM.from_pretrained(
        "THUDM/cogagent-chat-hf",
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch_type,
        trust_remote_code=True
    ).eval()

    for subdir in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, subdir)
        for file in tqdm(os.listdir(folder_path)):
            # Check if the file is an image by its extension
            if file.endswith((".webp")) or file.endswith((".jpg")) or file.endswith((".png")):
                file_path = os.path.join(folder_path,file)
                filename,_ = os.path.splitext(file)
                image = Image.open(file_path).convert('RGB')
                # kosmos2_image = image.copy()
                # names = get_biggest_features_bbox(model,processor,kosmos2_image,draw=True)
                
                # kosmos2_image = image.copy()
                # for name in names:
                #     score = get_clip_score(clip_model, clip_preprocess,image,name,device)
                #     print(f"{name}:{score}")

                #     not_score = get_clip_score(clip_model, clip_preprocess,image,f"not {name}",device)
                #     print(f"not {name}:{not_score}")
                # for name in names:
                    # prompt = f"dose '{name}' exist in image? if not please correct it."
                response = get_cog_result(cog_model,cog_tokenizer,device,torch_type,image, prompt="descript the image as short as possible, no more than 10 words.")
                print(f"cog:{response}")
                
                # save response to file
                with open(f"{input_dir}/{filename}.cog_cap", "w",encoding='utf-8') as f:
                    f.write(response)

if __name__ == "__main__":
    main()