import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json
from tqdm import tqdm
import torch

from vllm import LLM, SamplingParams, RequestOutput
from vllm.core.interfaces import BlockSpaceManager
from vllm.sequence import SequenceGroup, Sequence
from vllm.inputs.data import token_inputs
from vllm.worker.worker import Worker

from utils import Tokenizer, VectorDB, TimeEstimator


class MatKVCache:
    def __init__(self, dir_path: str, cache_ids: List[str]):
        self.layers = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load(dir_path, cache_ids)

    def _load(self, dir_path: str, cache_ids: List[str]):
        # TODO: sorting
        chunked_caches = [
            torch.load(
                os.path.join(dir_path, f"{cache_id}.pt"),
                weights_only=False, map_location=torch.device(self.device)
            ) for cache_id in cache_ids
        ]
        assert len(chunked_caches) != 0, "no cache file found"
        num_layers = len(chunked_caches[0])
        for layer_idx in range(num_layers):
            # TODO: optimize with torch kernel
            # in: (num_heads, num_tokens, head_dim), out: (num_tokens, num_heads, head_dim)
            key_caches = list(map(lambda x: x[layer_idx][0].squeeze(dim=0), chunked_caches))
            value_caches = list(map(lambda x: x[layer_idx][1].squeeze(dim=0), chunked_caches))
            layer_cache = MatKVLayerCache(
                key=torch.cat(key_caches, dim=1).permute(1, 0, 2),
                value=torch.cat(value_caches, dim=1).permute(1, 0, 2)
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
        max_token_len: int = 1536
    ):
        self.llm = LLM(
            model=model_name,
            enable_prefix_caching=use_cache,
            gpu_memory_utilization=0.8,
            block_size=block_size,
            max_model_len=4096
        )
        self.sampling_params = SamplingParams(temperature=0.0)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_token_len = max_token_len

    def __del__(self):
        del self.llm

    @property
    def kv_cache(self) -> List[torch.Tensor]:
        assert len(self.worker.kv_cache) == 1, "only support single gpu worker"
        return self.worker.kv_cache[0]

    @property
    def worker(self) -> Worker:
        return self.llm.llm_engine.model_executor.driver_worker.worker

    @property
    def block_manager(self) -> BlockSpaceManager:
        assert len(self.llm.llm_engine.scheduler) == 1, "only support single scheduler (single gpu worker)"
        return self.llm.llm_engine.scheduler[0].block_manager

    def generate(self, prompts: List[str], print_outputs: bool = False) -> List[RequestOutput]:
        outputs = self.llm.generate(prompts=prompts, sampling_params=self.sampling_params)
        if print_outputs:
            self._print_outputs(outputs)
        return outputs

    @staticmethod
    def _print_outputs(outputs: List[RequestOutput]):
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")

    def load_block_to_engine(self, prompt_text: str):
        tokenized_inputs = self.tokenizer.tokenize(prompt_text, max_len=self.max_token_len)["input_ids"]
        seq_group = SequenceGroup(
            request_id=next(self.llm.request_counter),
            seqs=[
                Sequence(
                    seq_id=next(self.llm.llm_engine.seq_counter),
                    inputs=token_inputs(tokenized_inputs),
                    block_size=self.block_size
                )
            ],
            arrival_time=0
        )
        self.block_manager.allocate(seq_group=seq_group)
        # parameters have no meaning. not used
        self.block_manager.mark_blocks_as_computed(seq_group=seq_group, token_chunk_size=0)

    def load_external_cache(self, cache: MatKVCache):
        # this only add caches to blocks at front
        num_blocks = cache.num_tokens // self.block_size
        assert cache.num_tokens % self.block_size == 0, \
            f"num_tokens must be divisible by block_size but num_tokens:{cache.num_tokens}"

        for layer_idx, layer_cache in enumerate(cache.layers):
            for block_idx in range(num_blocks):
                start_pos = block_idx * self.block_size

                key_cache = layer_cache.key[start_pos:start_pos + self.block_size]
                value_cache = layer_cache.value[start_pos:start_pos + self.block_size]
                self.kv_cache[layer_idx][0][block_idx].copy_(key_cache)
                self.kv_cache[layer_idx][1][block_idx].copy_(value_cache)


def get_prompt(doc_text: str, query: str) -> str:
    template = (
        f"\n\nAnswer the following Question, "
        f"given the relevant documents above. Answer without explanation. "
        f"\n\nQuestion: {query}\n\nAnswer:"
    )
    return doc_text + template


@dataclass
class MatKVLayerCache:
    key: torch.Tensor
    value: torch.Tensor


def analyze_logged_times(key: str, log_path: str) -> Optional[Tuple[float, float]]:
    assert os.path.exists(log_path), f"log file not found: {log_path}"
    with open(log_path) as f:
        data = json.load(f).get(key, {})

    if len(data.get("prefill", [])) != 1 or data.get("prefill", []) == []:
        return None
    return sum(data["prefill"]), sum(data["decode"])


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
    max_samples: Optional[int] = None
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
    num_errors = 0

    with open(query_path) as f:
        batch_queries = []

        for sample_idx, line in enumerate(tqdm(f)):
            if sample_idx >= max_samples:
                break
            batch_queries.append(json.loads(line)["query"])
            if len(batch_queries) != batch_size:
                continue

            # run batch
            # HACK: only supports batch_size=1 for now
            llm = LLMManager(model_name=model_name, tokenizer=tokenizer, use_cache=use_cache, block_size=block_size)

            overall_te = TimeEstimator()
            search_te = TimeEstimator()
            results = db.batch_query(batch_queries, top_k)
            search_times.append(search_te.stop())

            prompts = []
            for idx, query in enumerate(batch_queries):
                cache_ids = sorted(results["ids"][idx])
                doc_text = "".join(results["documents"][idx])
                prompt = get_prompt(doc_text, query)
                prompts.append(prompt)

                if use_cache:
                    # load blocks
                    llm.load_block_to_engine(doc_text)

                    # load caches
                    cache_te = TimeEstimator()

                    cache = MatKVCache(cache_dir, cache_ids)
                    llm.load_external_cache(cache)

                    load_times.append(cache_te.stop())

                    del cache

            # batch run
            engine_te = TimeEstimator()
            _ = llm.generate(prompts=prompts, print_outputs=False)
            engine_times.append(engine_te.stop())

            overall_times.append(overall_te.stop())

            times = analyze_logged_times(str(id(llm.worker)), log_path)
            if times is not None:
                prefill_time, decode_time = times
                prefill_times.append(prefill_time)
            else:
                num_errors += 1
            decode_times.append(decode_time)

            del llm
            batch_queries = []

    # print time analysis
    print("-" * 20 + "Results" + "-" * 20)
    print(f"Avg search time: {sum(search_times) / len(search_times)}")
    if use_cache:
        print(f"Avg load time: {sum(load_times) / len(load_times)}")
    print(f"Avg engine time: {sum(engine_times) / len(engine_times)}")
    print(f"Avg prefill time: {sum(prefill_times) / len(prefill_times)}")
    print(f"Avg decode time: {sum(decode_times) / len(decode_times)}")
    print(f"Avg overall time: {sum(overall_times) / len(overall_times)}")
    print(f"Num errors: {num_errors}")


if __name__ == "__main__":
    main(
        use_cache=True,
        model_name="meta-llama/Llama-3.2-3B",
        db_dir="/home/s2/dongseob/preprocessing/db_3b",
        cache_dir="/home/s2/dongseob/preprocessing/cache_3b",
        query_path="/home/s2/dongseob/preprocessing/qa_data/questions/query.jsonl",
        log_path="./profile/matkv.json",
        top_k=2,
        batch_size=1,
        block_size=16,
        max_samples=1,
    )
