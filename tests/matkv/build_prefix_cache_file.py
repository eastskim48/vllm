import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

sample_docs = [
    "You are an expert school principal, skilled in effectively managing "
    "faculty and staff. Draft 10-15 questions for a potential first grade "
    "Head Teacher for my K-12, all-girls', independent school that emphasizes "
    "community, joyful discovery, and life-long learning. The candidate is "
    "coming in for a first-round panel interview for a 8th grade Math "
    "teaching role. They have 5 years of previous teaching experience "
    "as an assistant teacher at a co-ed, public school with experience "
    "in middle school math teaching. Based on these information, fulfill "
    "the following paragraph: "
]

model_name = "facebook/opt-125m"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name) # padding_side=left 필요?
# tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
  model_name,
  quantization_config=None,
  device_map="auto",
)

for i, doc in enumerate(sample_docs):
    _input = tokenizer(doc, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**_input, use_cache=True) # run only prefill

    torch.save(output.past_key_values, os.path.join("./", f"{i}.pt"))