from module.preprocessor import save_preprocessed_dataset1, save_preprocessed_dataset2
from datasets import load_dataset
import json

dataset = load_dataset("clip-benchmark/wds_mscoco_captions2017")
save_path1 = "./dataset/preprocessed_mscoco"
save_preprocessed_dataset1(dataset, save_path1)

# ------------- second dataset (llava_instruct)---------------
with open("llava_instruct_150k.json", "r", encoding="utf-8") as f:
    instruct_data = json.load(f)

save_path2 = "./dataset/preprocessed_llava_instruct"
save_preprocessed_dataset2(instruct_data, save_path2)