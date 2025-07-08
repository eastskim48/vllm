from fastapi import FastAPI, Request
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import argparse

import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
import json
from tqdm import tqdm
import torch

from vllm import LLM, SamplingParams, RequestOutput
from vllm.core.interfaces import BlockSpaceManager
from vllm.sequence import SequenceGroup, Sequence
from vllm.inputs.data import token_inputs, TokensPrompt
from vllm.utils import Device

from utils import Tokenizer, VectorDB, TimeEstimator

class MatKVCache:
    def __init__(self, dir_path: str, cache_ids: List[str], device: str, max_len: int):
        self.layers = []
        self.device = device
        self.bos_cache = torch.load("./bos.pt", map_location=device, weights_only=False)
        self.max_len = max_len
        self._load(dir_path, cache_ids)

    def _load(self, dir_path: str, cache_ids: List[str]):
        chunked_caches = [
            torch.load(
                os.path.join(dir_path, f"{cache_id}.pt"),
                weights_only=False, map_location=torch.device(self.device)
            ) for cache_id in cache_ids
        ]
        assert len(chunked_caches) != 0, "no cache file found"
        num_layers = len(chunked_caches[0])
        for layer_idx in range(num_layers):
            # in: (num_heads, num_tokens, head_dim), out: (num_tokens, num_heads, head_dim)
            key_cache = torch.cat(
                list(map(lambda x: x[layer_idx][0].squeeze(dim=0), chunked_caches)), dim=1
            ).permute(1, 0, 2)

            value_cache = torch.cat(
                list(map(lambda x: x[layer_idx][1].squeeze(dim=0), chunked_caches)), dim=1
            ).permute(1, 0, 2)

            pad = torch.zeros(
                (self.max_len - key_cache.shape[0], key_cache.shape[1], key_cache.shape[2]),
                dtype=key_cache.dtype, device=self.device
            )

            layer_cache = MatKVLayerCache(
                key=torch.cat([pad, key_cache], dim=0),
                value=torch.cat([pad, value_cache], dim=0)
            )
            self.layers.append(layer_cache)

    @property
    def num_tokens(self) -> int:
        return self.layers[0].key.shape[0]


class LLMManager:
    def __init__(
        self,
        model_name: str,
        tokenizer: Tokenizer,
        use_cache: bool,
        block_size: int,
        max_token_len: int = 1536,
        device: str = "cuda",
        temperature: int = 0.3,
        max_gpu_util=0.3
    ):
        self.llm = LLM(
            model=model_name,
            enable_prefix_caching=use_cache,
            gpu_memory_utilization=max_gpu_util,
            block_size=block_size,
            max_model_len=4096
        )
        self.sampling_params = SamplingParams(temperature=temperature)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_token_len = max_token_len
        self.device = device

    def __del__(self):
        del self.llm

    @property
    def kv_cache(self) -> List[torch.Tensor]:
        assert len(self.worker.kv_cache) == 1, "only support single gpu worker"
        return self.worker.kv_cache[0]

    @property
    def worker(self):
        return self.llm.llm_engine.model_executor.driver_worker.worker

    @property
    def block_manager(self) -> BlockSpaceManager:
        assert len(self.llm.llm_engine.scheduler) == 1, "only support single scheduler (single gpu worker)"
        return self.llm.llm_engine.scheduler[0].block_manager

    def generate(self, prompts: List[Any]) -> List[Tuple[str, int]]:
        outputs = self.llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=prompt) for prompt in prompts],
                                    sampling_params=self.sampling_params
        )
        return [(output.outputs[0].text, output.num_cached_tokens) for output in outputs]

    def _print_outputs(self, outputs: List[RequestOutput]):
        for output in outputs:
            prompt = output.prompt_token_ids
            generated_text = output.outputs[0].text
            print(f"Prompt: {self.tokenizer.tokenizer.decode(prompt, skip_special_tokens=True)}"
                  f"\n{generated_text}")
            print(f"cache hit: {output.num_cached_tokens} tokens")

    def load_block_to_engine(self, tokenized_inputs: List[int]):
        seq_group = SequenceGroup(
            request_id=next(self.llm.request_counter),
            seqs=[
                Sequence(
                    seq_id=next(self.llm.llm_engine.seq_counter),
                    inputs=token_inputs(tokenized_inputs),
                    block_size=self.block_size
                )
            ],
            arrival_time=time.time()
        )
        self.block_manager.allocate(seq_group=seq_group)
        # parameters have no meaning. not used
        self.block_manager.mark_blocks_as_computed(seq_group=seq_group, token_chunk_size=0)

    def load_external_cache(self, cache: MatKVCache):
        # add caches to blocks
        target_blocks = \
        self.block_manager.block_tables[self.llm.llm_engine.seq_counter.counter - 1]._blocks._block_ids
        num_blocks = cache.num_tokens // self.block_size
        assert cache.num_tokens % self.block_size == 0, \
            f"num_tokens must be divisible by block_size but num_tokens:{cache.num_tokens}"

        assert len(target_blocks) == num_blocks, f"target_blocks: {target_blocks}, num_blocks: {num_blocks}"

        for layer_idx, layer_cache in enumerate(cache.layers):
            # cpu_cache: num_blocks, self.block_size, self.num_heads, self.head_size
            # -> (2, num_blocks, block_size(16) * num_kv_heads(8) * head_size(128))
            # layer_cache.key = (block_size*num_blocks, num_kv_heads, head_size)
            key_cache_reshaped = layer_cache.key.view(num_blocks, self.block_size, *layer_cache.key.shape[1:])
            value_cache_reshaped = layer_cache.value.view(num_blocks, self.block_size, *layer_cache.value.shape[1:])
            # (num_blocks, block_size, num_heads, head_dim)
            # self.kv_cache[layer_idx][0].shape = torch.Size([3548, 16, 8, 128])
            if self.device == "cpu":
                key_cache_reshaped = key_cache_reshaped.permute(0, 2, 1, 3).reshape(num_blocks, -1)
                value_cache_reshaped = value_cache_reshaped.permute(0, 2, 1, 3).reshape(num_blocks, -1)
            index = torch.tensor(target_blocks, device=self.device)
            self.kv_cache[layer_idx][0].index_copy_(0, index, key_cache_reshaped.to(torch.bfloat16))
            self.kv_cache[layer_idx][1].index_copy_(0, index, value_cache_reshaped.to(torch.bfloat16))

@dataclass
class MatKVLayerCache:
    key: torch.Tensor
    value: torch.Tensor


def get_template(query: str) -> str:
    template = (
        f"\n\nAnswer the following Question, "
        f"given the relevant documents above. Answer without explanation. "
        f"\n\nQuestion: {query}\n\nAnswer:"
    )
    return template


def get_nearest_blocked_size(len: int, block_size: int) -> int:
    return ((len + (block_size - 1)) // block_size) * block_size



# server

class PromptRequest(BaseModel):
    prompt: str
    top_k: int

llm = None
db = None
cache_dir = None
bg_executor = ThreadPoolExecutor(max_workers=2)
app: FastAPI = None


def refresh_blocktable_if_full(args):
    global llm
    num_free_gpu_blocks = llm.block_manager.block_allocator.get_num_free_blocks(
        device=Device.GPU)
    num_required_blocks = llm.max_token_len // llm.block_size

    if num_free_gpu_blocks < num_required_blocks * 2:  # HACK. Needs to be optimized
        print(f"Only {num_free_gpu_blocks} blocks left. refreshing LLM Engine...")
        del llm
        llm = LLMManager(model_name=args.model_name, tokenizer=Tokenizer(args.model_name), use_cache=True,
                         block_size=args.block_size,
                         device="cuda", max_gpu_util=args.max_gpu_util, temperature=args.temperature)


def create_app(args):
    global db, llm, app

    db = VectorDB(db_dir=args.db_dir)
    llm = LLMManager(model_name=args.model_name, tokenizer=Tokenizer(args.model_name), use_cache=True,
                     block_size=args.block_size,
                     device="cuda", max_gpu_util=args.max_gpu_util, temperature=args.temperature)
    app = FastAPI()

    @app.post("/generate")
    def generate_text(request: PromptRequest):
        try:
            results = db.batch_query(request.prompt, request.top_k)
            cache_ids = sorted(results["ids"][0])
            doc_text = "".join(results["documents"][0])

            max_cache_len = get_nearest_blocked_size(
                sum([m["token_len"] for m in results["metadatas"][0]]), args.block_size
            )

            tokenized_docs = llm.tokenizer.tokenize(doc_text,
                                                max_len=max_cache_len,
                                                add_special_tokens=True)["input_ids"]
            prompt = tokenized_docs + \
                     llm.tokenizer.tokenize(get_template(request.prompt), padding="do_not_pad",
                                        add_special_tokens=False)["input_ids"]

            # load caches
            cache = MatKVCache(args.cache_dir, cache_ids, "cuda", max_len=max_cache_len)

            # store caches as blocks
            llm.load_block_to_engine(tokenized_docs)
            llm.load_external_cache(cache)

            outputs = llm.generate(prompts=[prompt])
            assert len(outputs) == 1
            del cache

            bg_executor.submit(refresh_blocktable_if_full)

            return {"response": [{"prompt": request.prompt,"generated": generated, "num_cache_hit": num_cache_hit} for (generated, num_cache_hit) in outputs]}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--max_gpu_util", type=float, default=0.3)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--db_dir", type=str, default="/home/dongseob/preprocessing/db_3b")
    parser.add_argument("--cache_dir", type=str, default="/home/dongseob/preprocessing/cache_3b")
    args = parser.parse_args()

    global app, cache_dir
    cache_dir = args.cache_dir
    app = create_app(args)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="debug")

if __name__ == "__main__":
    main()
