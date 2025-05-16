import os
import time

import torch
import chromadb
from transformers import AutoTokenizer

from typing import List, Any, Optional


class KVCacheStore:
    def __init__(self, dir: str):
        self.dir = dir

    def save_kv_cache(self, cache: torch.Tensor, cache_id: str):
        try:
            torch.save(cache, os.path.join(self.dir, f"{cache_id}.pt"))
        except Exception as e:
            print(f"cache file {cache_id} save fail!\n{e}")

    def load_kv_cache(self, cache_id: str) -> torch.Tensor:
        try:
            cache_file = os.path.join(self.dir, f"{cache_id}.pt")
            return torch.load(cache_file, weights_only=True, map_location="cuda")
        except Exception as e:
            print(f"cache file {cache_id} load fail!\n{e}")


class DocumentChunk:
  def __init__(
    self,
    chunk_id: str,
    text: str,
  ):
    self.chunk_id = chunk_id
    self.text = text


class VectorDB:
    def __init__(self, db_dir: str):
        self.dir = db_dir
        self.collection = self._get_collection()

    def add_documents(self, chunks: List[DocumentChunk]):
        self.collection.upsert(
            documents=[chunk.text for chunk in chunks],
            ids=[chunk.chunk_id for chunk in chunks]
        )

    def _get_collection(self):
        chroma_client = chromadb.PersistentClient(path = self.dir)
        return chroma_client.get_or_create_collection(name="matkv")

    def batch_query(self, queries: List[str], top_k: int = 1) -> Optional[chromadb.QueryResult]:
        try:
            return self.collection.query(query_texts=queries, n_results=top_k)
        except Exception as e:
            print(f"vectordb query fail!\n{e}")
            return None


class Tokenizer:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

    def tokenize(self, test_input: str,  return_tensors: bool = False, max_len: Optional[int] = None) -> Any:
        if return_tensors:
            tokenized = self.tokenizer(
                test_input, max_length=max_len, padding="max_length", return_tensors="pt", truncation=True
            )
        else:
            tokenized = self.tokenizer(test_input, max_length=max_len, padding="max_length", truncation=True)
        return tokenized

    def split_document(self, filepath: str, chunk_size: int) -> List[DocumentChunk]:
        filename = filepath.split("/")[-1]
        with open(filepath) as f:
            text = f.read()
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            chunks = [
                DocumentChunk(
                    chunk_id=f"{filename}-{i}",
                    text=self.tokenizer.decode(
                        tokens[i:i + chunk_size], skip_special_tokens=True
                    )
                ) for i in range(0, len(tokens), chunk_size)
            ]
        return chunks


class TimeEstimator:
    def __init__(self):
        self.start()

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    def stop(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        return self.end_time - self.start_time