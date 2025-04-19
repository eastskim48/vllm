import torch
import os
from transformers import AutoTokenizer
from typing import List, Tuple
import time
import json

from vllm import LLM, SamplingParams, LLMEngine
from vllm.sequence import SequenceGroup, Sequence
from vllm.inputs.data import token_inputs

block_size = 16
device = "cuda" if torch.cuda.is_available() else "cpu"
sampling_params = SamplingParams(temperature=0.0)
model_name = "meta-llama/Llama-3.2-3B"

doc_path = "/home/s2/dongseob/preprocessing/qa_data/documents/doc_0.txt"
query_path = "/home/s2/dongseob/preprocessing/qa_data/questions/query.jsonl"
with open(query_path) as f:
    query = json.loads(f.readline())["query"]
with open(doc_path, "r") as f:
    doc = f.read()
docs = [doc]
queries = [f"\n\nAnswer the following Question, given the relevant documents above. Answer without explanation. \n\nQuestion: {query}\n\nAnswer:"]


def load_and_convert_cache(file_name: str):
    cache_tuples = torch.load(os.path.join("./", file_name),
                              weights_only=False, map_location=torch.device(device))
    caches = []
    for layer in cache_tuples:
        key_cache = layer[0].squeeze(dim=0).permute(1, 0, 2)
        value_cache = layer[1].squeeze(dim=0).permute(1, 0, 2) # (44, 12, 64)
        caches.append((key_cache, value_cache))
    return caches


def store_loaded_cache_to_engine(
        loaded_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        kv_cache_tensor: torch.Tensor
):
    num_blocks = loaded_caches[0][0].shape[0] // block_size
    for layer_idx, (_key_cache, _value_cache) in enumerate(loaded_caches):
        for block_idx in range(num_blocks):
            # Assumption: key_cache의 shape[0] size는 block_size로 나누어 떨어진다 (padding 붙여서)
            # PagedAttention.write_to_paged_cache로 쓰는 것과 뭔가 차이가 있을 수는 있음. 확인 필요
            key_cache = _key_cache[block_idx * block_size:(block_idx + 1) * block_size]
            value_cache = _value_cache[block_idx * block_size:(block_idx + 1) * block_size]

            kv_cache_tensor[0][layer_idx][0][block_idx].copy_(key_cache)
            kv_cache_tensor[0][layer_idx][1][block_idx].copy_(value_cache)


def load_block_to_engine(engine: LLMEngine, prompt_text: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    tokenized_inputs = tokenizer(prompt_text, max_length=1024, padding="max_length")["input_ids"]
    # tokenized_inputs = tokenizer(prompt_text)["input_ids"]
    # assume parallel size = 1
    # req_id, seq_id를 근데 절대 나올 수 없는 숫자를 줘야 함. 아니면 seq id generate 부분을 속이던지
    seq_group = SequenceGroup(request_id="100",
                  seqs=[
                      Sequence(
                          seq_id=100, inputs=token_inputs(tokenized_inputs), block_size=block_size
                      )
                  ],
                  arrival_time=0
                  )
    engine.scheduler[0].block_manager.allocate(seq_group=seq_group)
    engine.scheduler[0].block_manager.mark_blocks_as_computed(
        seq_group=seq_group, token_chunk_size=0
    )

def main(use_cache: bool=True):
    # Load the cache from the file
    caches = load_and_convert_cache("0_0.pt")

    # Create an LLM with prefix caching enabled.
    sampling_params = SamplingParams(temperature=0.0)
    llm = LLM(model=model_name,
              enable_prefix_caching=use_cache,
              gpu_memory_utilization=0.8,
              block_size=block_size,
              max_model_len=4096
              )

    if use_cache:
        store_loaded_cache_to_engine(
            loaded_caches=caches,
            kv_cache_tensor=llm.llm_engine.model_executor.driver_worker.worker.kv_cache
        )
        load_block_to_engine(engine=llm.llm_engine, prompt_text=docs[0])
    start_time = time.perf_counter()
    prompt = docs[0] + queries[0]
    outputs = llm.generate(prompts=prompt, sampling_params=sampling_params)
    print(f"time taken: {time.perf_counter() - start_time:.4f} seconds")

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")

if __name__ == "__main__":
    main(use_cache=False)
