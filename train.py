from onnx import save
from module.model import load_models_cuda, create_LoRA_model
from datasets import load_from_disk
from transformers import default_data_collator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import torch
import os

device= "cuda" if torch.cuda.is_available() else "cpu"
config1 = {"num_epochs": 4, 
            "batch_size": 8,
            "learning_rate": 5e-5,
            "LoRA_R": 32,
            "LoRA_alpha": 64,
            "LoRA_dropout": 0.05}
data_path1 = "./dataset/preprocessed_mscoco"
model_save_path = "./saved_models"

llm, tokenizer, vision_encoder, image_processor = load_models_cuda()

model = create_LoRA_model(llm, vision_encoder, 
                          r = config1["LoRA_R"], 
                          lora_alpha = config1["LoRA_alpha"], 
                          lora_dropout = config1["LoRA_dropout"], 
                          )

dataset = load_from_disk(data_path1)
train_dataloader = DataLoader(dataset, batch_size=config1["batch_size"], collate_fn=default_data_collator, shuffle=True)
trainable_params = [p for p in model.parameters() if p.requires_grad] # require_grad가 허용된(미분가능) 부분에만 optimizer적용
optimizer = AdamW(trainable_params, lr=config1["learning_rate"])

num_training_steps = config1["num_epochs"] * len(train_dataloader)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0, # 몇 step동안 천천히 증가(웜업) 할지 설정
    num_training_steps=num_training_steps, # 몇 step에 걸쳐 천천히 감소할지 설정
)

# 훈련 루프
for epoch in range(config1["num_epochs"]):
    model.train()
    progress_bar = tqdm(train_dataloader, desc="Training")
    print(train_dataloader)
    for step, batch in enumerate(progress_bar):
        outputs = model(batch['input_ids'].to(device),
                        batch['pixel_values'].to(device),
                        batch['attention_mask'].to(device),
                        batch['labels'].to(device))
        loss = outputs.loss
        
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress_bar.set_postfix({"loss": loss.item()})

# ------------- instruct train ---------------
config2 = {"num_epochs": 2,
            "batch_size": 7,
            "learning_rate":1e-4} 
data_path2 = "./dataset/preprocessed_llava_instruct"
instruct_dataset = load_from_disk(data_path2)
instruct_dataloader = DataLoader(instruct_dataset, batch_size=config2["batch_size"], collate_fn=default_data_collator, shuffle=True)

instruct_trainable_params = [p for p in model.parameters() if p.requires_grad]

instruct_optimizer = AdamW(instruct_trainable_params, lr=config2["learning_rate"])

num_training_steps = config2["num_epochs"] * len(instruct_dataloader)
num_warmup_steps = int(num_training_steps * 0.05)

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=instruct_optimizer,
    num_warmup_steps= num_warmup_steps, # 몇 step동안 천천히 증가(웜업) 할지 설정
    num_training_steps=num_training_steps, # 몇 step에 걸쳐 천천히 감소할지 설정
)

for epoch in range(config2["num_epochs"]):
    model.train()
    progress_bar = tqdm(instruct_dataloader, desc="Training")
    print(instruct_dataloader)
    for step, batch in enumerate(progress_bar):
        outputs = model(batch['input_ids'].to(device),
                                    batch['pixel_values'].to(device),
                                    batch['attention_mask'].to(device),
                                    batch['labels'].to(device))
        
        loss = outputs.loss
        
        loss.backward()
        
        instruct_optimizer.step()
        lr_scheduler.step()
        instruct_optimizer.zero_grad()
        
        progress_bar.set_postfix({"loss": loss.item()})



# 모델 저장
os.makedirs(model_save_path+f"/{epoch}epoch", exist_ok=True)

lora_save_path = os.path.join(model_save_path+f"/{epoch}epoch", f"llm_adapters{epoch}")
model.llm.save_pretrained(lora_save_path)
print(f"LoRA adapters saved to {lora_save_path}")

vision_projection_path = os.path.join(model_save_path+f"/{epoch}epoch", f"vision_projection{epoch}.pt")
torch.save(model.vision_projection.state_dict(), vision_projection_path)
print(f"Vision projection saved to {vision_projection_path}")

cross_attentions_path = os.path.join(model_save_path+f"/{epoch}epoch", f"cross_attentions{epoch}.pt")
torch.save(model.cross_attentions.state_dict(), cross_attentions_path)
print(f"Cross attentions saved to {cross_attentions_path}")


