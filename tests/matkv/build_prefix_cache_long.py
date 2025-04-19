import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

model_name = "meta-llama/Llama-3.2-3B"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name) # padding_side=left 필요?
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

model = AutoModelForCausalLM.from_pretrained(
  model_name,
  quantization_config=None,
  device_map="auto",
)

doc_path = "/home/s2/dongseob/preprocessing/qa_data/documents/doc_0.txt"
with open(doc_path, "r") as f:
    doc = f.read()

for i, doc in enumerate([doc]):
    _input = tokenizer(doc, max_length=1024, padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**_input, use_cache=True) # run only prefill

    torch.save(output.past_key_values, os.path.join("./", f"{i}_0.pt"))