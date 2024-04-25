from transformers import AutoProcessor, Kosmos2ForConditionalGeneration
import numpy
import torch
from PIL import Image
import os


input_dir = "F:/ImageSet/openxl2_realism"
file_path = os.path.join(input_dir, "cog/00001_preserved.webp")
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224").to(device)
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224", return_tensors="pt")

image = Image.open(file_path).convert('RGB')
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

print(entities)

# specify the location indexes of some input tokens
START_INDEX_of_IMAGE_TOKENS = 1
END_INDEX_of_IMAGE_TOKENS = 1
NUM_of_TOKENS_IN_THE_PROMPT = 2048

key_position = {
  "image_start": START_INDEX_of_IMAGE_TOKENS, 
  "image_end": END_INDEX_of_IMAGE_TOKENS, 
  "response_start": NUM_of_TOKENS_IN_THE_PROMPT,
}


generated_ids = model.generate(
    pixel_values=inputs["pixel_values"],
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    image_embeds=None,
    image_embeds_position_mask=inputs["image_embeds_position_mask"],
    use_cache=True,
    max_new_tokens=64,
    # opera
    opera_decoding=True,
    key_position=key_position,
    scale_factor=50,
    threshold=25,
    num_attn_candidates=1,
    penalty_weights=1,
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

_, entities = processor.post_process_generation(generated_text)

print(entities)