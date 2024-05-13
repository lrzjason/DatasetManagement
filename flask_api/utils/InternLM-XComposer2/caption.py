import torch
from transformers import AutoModel, AutoTokenizer

torch.set_grad_enabled(False)

# init model and tokenizer
model = AutoModel.from_pretrained('internlm/internlm-xcomposer2-vl-7b',              
          torch_dtype=torch.bfloat16,
          low_cpu_mem_usage=True,
          load_in_4bit=True,
          bnb_4bit_compute_dtype=torch.bfloat16,
          trust_remote_code=True
        ).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-vl-7b', trust_remote_code=True)

query = '<ImageHere>Please describe this image in detail.'
image = './image1.webp'
with torch.cuda.amp.autocast():
  response, _ = model.chat(tokenizer, query=query, image=image, history=[], do_sample=False)
print(response)