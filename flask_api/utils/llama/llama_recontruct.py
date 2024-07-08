import transformers
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from tqdm import tqdm

def main():
    input_dir = "F:/ImageSet/openxl2_reg/llama3_output"

    template_path = "./prompt_template_tags_recontruct.txt"
    template = ''
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
        template = template.strip()
        f.close()

    text_ext = '.txt'

    model_id = "refuelai/Llama-3-Refueled"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")



    for sub in os.listdir(input_dir):
        sub_dir = os.path.join(input_dir, sub)
        for file in tqdm(os.listdir(sub_dir)):
            if file.endswith(text_ext):
                text_path = os.path.join(sub_dir, file)
                print(text_path)
                text = ''
                # Append the full path to the list
                with open(text_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    text = text.replace('/n', '').strip()
                    f.close()

                # clone template
                prompt = template

                # replace text and tags
                prompt = prompt.replace('$tags$', text)
                messages = [{"role": "user", "content": prompt}]

                inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")

                outputs = model.generate(inputs, max_new_tokens=200)
                response = tokenizer.decode(outputs[0])
                # print("ori response")
                # print(response)
                response = response.split("<|start_header_id|>assistant<|end_header_id|>")[1].replace("\n","").replace("<|eot_id|><|end_of_text|>","")
                print(text_path)
                print(text_path.replace('.txt','.webp'))
                print("response")
                print(response)
                
                
                with open(text_path, 'w', encoding="utf-8") as f:
                    f.write(response)
                    f.close()
                    print(f'write {text_path}')
        #         break
        # break
        

if __name__ == "__main__":
    main()