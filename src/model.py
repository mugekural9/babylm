# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("babylm/babyllama-10m-2024")
model = AutoModelForCausalLM.from_pretrained("babylm/babyllama-10m-2024")

model.save_pretrained("./babylm/babyllama", from_pt=True) 
tokenizer.save_pretrained('./babylm/babyllama')
