from transformers import CLIPImageProcessor, AutoTokenizer
from datasets import Dataset
from tqdm.auto import tqdm
import json
from PIL import Image

image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def flatten_generator(dataset):
    for item in tqdm(dataset, desc="Flattening dataset"):
        captions = [line for line in item['txt'].split('\n')]
        for caption in captions:
            yield {'jpg': item['jpg'], 'caption': caption}

def preprocess_function(dataset):
    images = [img.convert("RGB") for img in dataset['jpg']]
    captions = dataset['caption']
    model_inputs = image_processor(images, return_tensors="pt")
    text_inputs = tokenizer(captions, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    model_inputs['input_ids'] = text_inputs['input_ids']
    model_inputs['attention_mask'] = text_inputs['attention_mask']
    model_inputs['labels'] = text_inputs['input_ids'].clone()
    return model_inputs

def save_preprocessed_dataset1(dataset, save_path):
    expanded_dataset = Dataset.from_generator(flatten_generator, gen_kwargs={"dataset": dataset['train']})
    final_dataset = expanded_dataset.map(
    function=preprocess_function,
    batched=True,
    remove_columns=expanded_dataset.column_names
    )
    final_dataset.save_to_disk(save_path)
    print(f"Preprocessed dataset saved to {save_path}")



# ------------- second dataset (llava_instruct)---------------

def instruction_generator(dataset):
    for item in tqdm(dataset, desc="Generating instruction pairs"):
        conv = item['conversations']
        
        # 2개씩 짝지어 (human, gpt) 턴 처리
        for i in range(0, len(conv), 2):
            human_turn = conv[i]
            gpt_turn = conv[i+1]
            
            # 질문과 답변 텍스트를 분리
            question = human_turn['value']
            answer = gpt_turn['value']
            
            # 나중에 처리하기 쉽도록 분리된 정보를 yield
            yield {
                'image_path': item['image'],
                'question': question,
                'answer': answer
            }

def preprocess_function(dataset):

    image_paths = dataset['image_path'] 
    questions = dataset['question']
    answers = dataset['answer']

    # 이미지 처리 
    images = [Image.open(f"coco/train2017/{path}").convert("RGB") for path in image_paths]
    processed_images = image_processor(images, return_tensors="pt")

    # 텍스트 처리 및 라벨 마스킹 
    question_tokenized = tokenizer(questions, padding="max_length", truncation=True, max_length=256)
    question_lengths = [len([tok for tok in ids if tok != tokenizer.pad_token_id]) for ids in question_tokenized['input_ids']]
    
    full_texts = [q + a for q, a in zip(questions, answers)]
    model_inputs = tokenizer(full_texts, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

    # 위에서 계산한 질문 길이만큼 labels의 앞부분을 -100으로 마스킹
    labels = model_inputs['input_ids'].clone()
    for i, question_len in enumerate(question_lengths):
        labels[i, :question_len] = -100

    return {
        "pixel_values": processed_images.pixel_values,
        "input_ids": model_inputs.input_ids,
        "attention_mask": model_inputs.attention_mask,
        "labels": labels
    }

def save_preprocessed_dataset2(dataset, save_path):
    expanded_instruction_dataset = Dataset.from_generator(instruction_generator, gen_kwargs={"dataset": dataset})
    instruction_dataset = expanded_instruction_dataset.map(
        function=preprocess_function,
        batched=True,
        remove_columns=expanded_instruction_dataset.column_names
    )
    instruction_dataset.save_to_disk(save_path)
    print(f"Preprocessed instruction dataset saved to {save_path}")