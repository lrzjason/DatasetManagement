from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import os
import torch

def predict(model,tokenizer,file_path,question = "Describe this image in one sentence."):
    image = Image.open(file_path)
    enc_image = model.encode_image(image)
    # print(model.answer_question(enc_image, "Describe this image in one sentence.", tokenizer))

    
    chat_history=""
    prompt = f"<image>\n\n{chat_history}Question: {question}\n\nAnswer:"
    answer = model.generate(
        enc_image,
        prompt,
        tokenizer=tokenizer,
        max_new_tokens=512,
    )[0]
    cleaned_answer = answer.strip()
    return cleaned_answer

def main():
    input_dir = "F:/ImageSet/vit_train/hand-classifier/good_hand"
    file_path = os.path.join(input_dir, "472.0.61.jpg")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # device = "cpu"

    model_id = "vikhyatk/moondream2"
    revision = "2024-04-02"


    torch_type = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        revision=revision,
        # torch_dtype=torch_type,
        # low_cpu_mem_usage=True,
        # load_in_4bit=True,
        # bnb_4bit_compute_dtype=torch_type,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    for file in os.listdir(input_dir):
        # Check if the file is an image by its extension
        if file.endswith((".jpg")):
            text_path = file_path.replace(".jpg", ".txt")
            if os.path.exists(text_path):
                continue
            # Join the folder path and the file name to get the full path
            file_path = os.path.join(input_dir, file)
            cleaned_answer = predict(model,tokenizer,file_path)
            cleaned_answer = cleaned_answer.replace("\n", " ").strip()
            print(file_path+"\n")
            print("prompt:\n")
            print(cleaned_answer+"\n")
            print(text_path+"\n")
            # save the answer
            with open(text_path, "w",encoding="utf-8") as f:
                f.write(cleaned_answer)


if __name__ == "__main__":
    main()

# question = "Describe this image in one sentence."
# chat_history=""
# prompt = f"<image>\n\n{chat_history}Question: {question}\n\nAnswer:"
# answer = model.generate(
#     enc_image,
#     prompt,
#     tokenizer=tokenizer,
#     max_new_tokens=512,

#     output_attentions=True,
#     opera_decoding=True,
#     scale_factor=50,
#     threshold=15.0,
#     num_attn_candidates=5,
# )[0]
# cleaned_answer = answer.strip()

# print(cleaned_answer)