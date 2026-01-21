import pandas as pd
import torch
from PIL import Image
from module.model import load_models_cuda, MultimodalPhi2
from peft import PeftModel
import os

test_dataset = pd.read_csv("./open/test.csv")

def inference(model, tokenizer, image_processor, prompt, image_path):
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

    prompt_tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = prompt_tokens.input_ids.to(device)
    attention_mask = prompt_tokens.attention_mask.to(device)
    PIL_image = Image.open(image_path).convert("RGB")
    image = image_processor(images=PIL_image, return_tensors="pt").pixel_values.to(device) 

    with torch.no_grad():
        image_outputs = model.vision_encoder(image)
        image_patch_features = image_outputs.last_hidden_state
        model.image_features_cache = model.vision_projection(image_patch_features)
    
    generated_ids = model.llm.generate(
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        max_new_tokens=128,  # 새로 생성할 최대 토큰 수
        do_sample=False,     # 샘플링을 활성화
        temperature=1,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    model.image_features_cache = None

    input_token_len = input_ids.shape[1]
    generated_text_ids = generated_ids[:, input_token_len:]
    
    generated_text = tokenizer.batch_decode(generated_text_ids, skip_special_tokens=True)[0]
    return generated_text.strip()

def load_trained_model(save_directory):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    llm, tokenizer, vision_encoder, image_processor = load_models_cuda()
    lora_save_path = os.path.join(save_directory, "llm_adapters2") 
    peft_llm = PeftModel.from_pretrained(llm, lora_save_path)

    model = MultimodalPhi2(peft_llm, vision_encoder)

    vision_projection_path = os.path.join(save_directory, "vision_projection2.pt")
    model.vision_projection.load_state_dict(torch.load(vision_projection_path, map_location='cpu'))

    cross_attentions_path = os.path.join(save_directory, "cross_attentions2.pt")
    model.cross_attentions.load_state_dict(torch.load(cross_attentions_path, map_location='cpu'))
    
    model.to(device)
    model.eval()

    return model, tokenizer, image_processor

if __name__ == "__main__":
    save_directory = "./saved_models/2epoch"
    model, tokenizer, image_processor = load_trained_model(save_directory)

    index = 16

    image_path = "./open/"+test_dataset['img_path'][index]

    question = test_dataset['Question'][index]
    A = test_dataset['A'][index]
    B = test_dataset['B'][index]
    C = test_dataset['C'][index]
    D = test_dataset['D'][index]

    # 새로운 프롬프트 구조

    prompt = f"""ROLE: I will give you an image, a question, and several answer choices.
    Choose the one correct answer based on the image.
    Only return the letter corresponding to the correct answer (e.g., A, B, C, or D).\n
            HUMAN: <Image>\n{question}\n
            [STEP 1] Infer the most correct answer of A, B, C, or D\n
            [STEP 2] Based on your inference, say only one of A, B, C, or D.\n
            ASSISTANT:"""

    result = inference(model, tokenizer, image_processor, prompt, image_path)

    print("Generated Answer: ", result)
    
       