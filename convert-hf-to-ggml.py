from transformers import AutoModelForCausalLM
import torch
import os

os.makedirs("models/vicuna/7B", exist_ok=True)

model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")
torch.save(model.state_dict(), "models/vicuna/7B/consolidated.00.pth")
