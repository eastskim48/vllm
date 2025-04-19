import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple
import time

from vllm import LLM, SamplingParams, LLMEngine
from vllm.attention.ops.paged_attn import PagedAttention
from vllm.sequence import SequenceGroup, Sequence
from vllm.inputs.data import token_inputs

block_size = 16
device = "cuda" if torch.cuda.is_available() else "cpu"
sampling_params = SamplingParams(temperature=0.0)
model_name = "facebook/opt-125m"

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

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

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
    tokenizer.padding_side = "left"
    # tokenized_inputs = tokenizer(prompt_text, max_length=128, padding="max_length")["input_ids"]
    tokenized_inputs = tokenizer(prompt_text)["input_ids"]
    # assume parallel size = 1
    seq = Sequence(seq_id=0, inputs=token_inputs(tokenized_inputs), block_size=block_size)
    engine.scheduler[0].block_manager.allocate(
        seq_group=SequenceGroup(request_id="0", seqs=[seq], arrival_time=0)
    )
    block_ids = [block.block_id for block in engine.scheduler[0].block_manager.block_tables[seq.seq_id].blocks]
    engine.scheduler[0].block_manager.block_allocator.mark_blocks_as_computed(block_ids)


def write_cache():
    # PagedAttention.write_to_paged_cache(
    #     key, value, key_cache,
    #     value_cache, slot_mapping, kv_cache_dtype,
    #     k_scale, v_scale)
    pass

def main(use_cache: bool=True):
    # Load the cache from the file
    caches = load_and_convert_cache("0.pt")

    # Create an LLM with prefix caching enabled.
    llm = LLM(model=model_name,
              enable_prefix_caching=use_cache,
              gpu_memory_utilization=0.4, block_size=block_size, max_model_len=256)

    if use_cache:
        store_loaded_cache_to_engine(
            loaded_caches=caches,
            kv_cache_tensor=llm.llm_engine.model_executor.driver_worker.worker.kv_cache
        )
        load_block_to_engine(engine=llm.llm_engine, prompt_text=sample_docs[0])
    start_time = time.perf_counter()
    outputs = llm.generate_with_id(req_id="0", prompts=sample_docs[0] + prompts[0])
    print(f"time taken: {time.perf_counter() - start_time:.4f} seconds")

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")

    # caches = load_and_convert_cache("0.pt")

if __name__ == "__main__":
    main(use_cache=True)
