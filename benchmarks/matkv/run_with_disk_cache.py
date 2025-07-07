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
from vllm.inputs.data import token_inputs
from vllm.utils import Device
from vllm.inputs.data import TokensPrompt

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

    def generate(self, prompts: List[Any], print_outputs: bool = True) -> List[RequestOutput]:
        outputs = self.llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=prompt) for prompt in prompts],
                                    sampling_params=self.sampling_params
        )
        if print_outputs:
            self._print_outputs(outputs)
        return outputs

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


def analyze_logged_times(key: str, log_path: str) -> Optional[Tuple[float, float, float, float]]:
    assert os.path.exists(log_path), f"log file not found: {log_path}"
    with open(log_path) as f:
        data = json.load(f).get(key, {})

    if data.get("prefill", []) == []:
        return None
    decode = data.get("decode", [])
    return sum(data["prefill"]), sum(decode), len(data["prefill"]), len(decode)


def get_nearest_blocked_size(len: int, block_size: int) -> int:
    return ((len + (block_size - 1)) // block_size) * block_size


def main(
    use_cache: bool,
    model_name: str,
    db_dir: str,
    cache_dir: str,
    query_path: str,
    log_path: str,
    top_k: int,
    batch_size: int,
    block_size: int,
    max_gpu_util: float = 0.3,
    temperature: float = 0.3,
    max_samples: Optional[int] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # Create an LLM with prefix caching enabled.
    db = VectorDB(db_dir=db_dir)
    tokenizer = Tokenizer(model_name)

    prefill_times = []
    decode_times = []
    engine_times = []
    load_times = []
    overall_times = []
    search_times = []
    decode_counts = []
    prefill_counts = []
    num_errors = 0

    with open(query_path) as f:
        batch_queries = []

        llm = LLMManager(model_name=model_name, tokenizer=tokenizer, use_cache=use_cache,
                         block_size=block_size,
                         device=device, max_gpu_util=max_gpu_util, temperature=temperature)

        for sample_idx, line in enumerate(tqdm(f)):
            if max_samples is not None and sample_idx >= max_samples:
                break
            batch_queries.append(json.loads(line)["query"])
            if len(batch_queries) != batch_size:
                continue

            overall_te = TimeEstimator()
            search_te = TimeEstimator()
            results = db.batch_query(batch_queries, top_k)
            search_times.append(search_te.stop())

            prompts = []

            max_cache_len = get_nearest_blocked_size(
                max(sum([m["token_len"] for m in batch_meta])
                    for batch_meta in results["metadatas"]), block_size
            )

            if use_cache:
                load_time = 0
                for batch_idx, query in enumerate(batch_queries):
                    load_te = TimeEstimator()

                    cache_ids = sorted(results["ids"][batch_idx])
                    doc_text = "".join(results["documents"][batch_idx])

                    tokenized_docs = tokenizer.tokenize(doc_text,
                                                        max_len=max_cache_len,
                                                        add_special_tokens=True)["input_ids"]
                    prompt = tokenized_docs + \
                             tokenizer.tokenize(get_template(query), padding="do_not_pad",
                                                add_special_tokens=False)["input_ids"]
                    prompts.append(prompt)

                    # load caches
                    cache = MatKVCache(cache_dir, cache_ids, device, max_len=max_cache_len)

                    # store caches as blocks
                    llm.load_block_to_engine(tokenized_docs)
                    llm.load_external_cache(cache)

                    load_time += load_te.stop()
                    del cache
                load_times.append(load_time)

            # batch run
            engine_te = TimeEstimator()
            _ = llm.generate(prompts=prompts, print_outputs=True)
            engine_times.append(engine_te.stop())

            overall_times.append(overall_te.stop())

            times = analyze_logged_times(str(id(llm.worker))+str(llm.llm.request_counter.counter-1), log_path)
            if times is not None:
                prefill_time, decode_time, prefill_count, decode_count = times
                prefill_times.append(prefill_time)
            else:
                num_errors += 1
            decode_times.append(decode_time)
            prefill_counts.append(prefill_count)
            decode_counts.append(decode_count)

            batch_queries = []

            # refresh llm object if block space is not enough
            num_free_gpu_blocks = llm.block_manager.block_allocator.get_num_free_blocks(
                device=Device.GPU)
            num_required_blocks = llm.max_token_len // llm.block_size

            if num_free_gpu_blocks < num_required_blocks * 2:  # HACK. Needs to be optimized
                print(f"Only {num_free_gpu_blocks} blocks left. refreshing LLM Engine...")
                del llm
                llm = LLMManager(model_name=model_name, tokenizer=tokenizer, use_cache=use_cache,
                                 block_size=block_size,
                                 device=device, max_gpu_util=max_gpu_util, temperature=temperature)


    # print time analysis
    print("-" * 20 + "Results" + "-" * 20)
    print(f"Avg search time: {sum(search_times) / len(search_times)}")
    if use_cache:
        print(f"Avg load time: {sum(load_times) / len(load_times)}")
    print(f"Avg engine time: {sum(engine_times) / len(engine_times)}")
    print(f"Avg prefill time: {sum(prefill_times) / len(prefill_times)}")
    print(f"Avg decode time: {sum(decode_times) / len(decode_times)}")
    print(f"Avg generated tokens: {sum(decode_counts) / len(decode_counts)}")
    print(f"Avg decode time per step: {sum(decode_times) / sum(decode_counts)}")
    print(f"Avg prefill time per step: {sum(prefill_times) / sum(prefill_counts)}")
    print(f"Avg overall time: {sum(overall_times) / len(overall_times)}")
    print(f"Num errors: {num_errors}")


if __name__ == "__main__":
    main(
        use_cache=True,
        model_name="meta-llama/Llama-3.2-3B",
        db_dir="/home/dongseob/preprocessing/db_3b",
        cache_dir="/home/dongseob/preprocessing/cache_3b",
        query_path="/home/dongseob/preprocessing/qa_data/questions/query.jsonl",
        log_path="./profile/matkv.json",
        top_k=1,
        batch_size=1,
        block_size=16,
        max_samples=96,
        device="cuda"
    )
