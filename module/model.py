import torch
import torch.nn as nn
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
from peft import get_peft_model, LoraConfig, TaskType

def load_models_cuda():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

    llm = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,     
    ).to(device)

    vision_encoder = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        torch_dtype=torch.bfloat16
    ).to(device)

    image_processor = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    return llm, tokenizer, vision_encoder, image_processor


class CrossAttention(nn.Module):
    def __init__(self, model_dims: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=model_dims, 
            num_heads=num_heads, 
            batch_first=True,
            dtype=torch.bfloat16 # float16
        )
        self.layer_norm = nn.LayerNorm(model_dims)

    def forward(self, text_features, image_features):
        attn_output, _ = self.attention(text_features, image_features, image_features)
        output = self.layer_norm(text_features + attn_output)
        return output

class MultimodalPhi2(nn.Module):
    def __init__(self, peft_llm, vision_encoder):
        super().__init__()
        
        self.vision_encoder = vision_encoder
        self.llm = peft_llm

        target_dtype = self.llm.dtype
        model_dims = self.llm.config.hidden_size
        vit_dims = self.vision_encoder.config.hidden_size
        num_heads = self.llm.config.num_attention_heads
        num_llm_layers = self.llm.config.num_hidden_layers

        self.target_layers = range(num_llm_layers - 4, num_llm_layers)

        self.vision_projection = nn.Linear(vit_dims, model_dims)
        self.vision_projection.to(device=self.llm.device, dtype=target_dtype) 
        
        self.cross_attentions = nn.ModuleDict({
        str(i): CrossAttention(model_dims, num_heads) for i in self.target_layers
        }) 

        self.image_features_cache = None # 이미지 특징을 임시 저장할 공간

        for layer_idx in self.target_layers:
            layer = self.llm.model.model.layers[layer_idx] # peft_llm의 경우, model을 한번 더 거쳐야 함
            layer.self_attn.register_forward_hook(
                partial(self.cross_attention_hook, layer_idx=layer_idx)
            )
        
        self.cross_attentions.to(device=self.llm.device, dtype=target_dtype)

    def cross_attention_hook(self, module, input, output, layer_idx):
        hidden_states = output[0]
        
        # ModuleDict의 키는 문자열이므로, 인덱싱할 때 str(layer_idx)를 사용
        cross_attn_output = self.cross_attentions[str(layer_idx)](
            hidden_states, self.image_features_cache
        )

        return (cross_attn_output,) + output[1:]

    def forward(self, input_ids: torch.Tensor, pixel_values: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None):
        # 이미지 특징을 계산하고, 훅 함수가 사용할 수 있도록 캐시에 저장
        image_outputs = self.vision_encoder(pixel_values)
        image_patch_features = image_outputs.last_hidden_state
        self.image_features_cache = self.vision_projection(
            image_patch_features.to(self.llm.dtype)
        )

        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels 
        )
        
        self.image_features_cache = None
        
        return outputs
    
def create_LoRA_model(llm, vision_encoder , r = 32, lora_alpha = 64, lora_dropout = 0.05):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lora_config = LoraConfig(
        r=r, # LoRA를 이용해 몇차원 으로 줄일지 설정
        lora_alpha=lora_alpha, # LoRA의 영향력 조절
        target_modules=["q_proj", "v_proj", "k_proj", "dense"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    peft_llm = get_peft_model(llm, lora_config)
    peft_llm.print_trainable_parameters()

    model = MultimodalPhi2(peft_llm, vision_encoder).to(device)

    return model