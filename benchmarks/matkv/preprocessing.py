import os
import torch
from transformers import AutoModelForCausalLM

from utils import DocumentChunk, Tokenizer, VectorDB


class KVCacheBuilder:
    def __init__(self, model_name: str, tokenizer: Tokenizer, cache_dir: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=None,
            device_map="auto",
        )
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir = cache_dir

    def save_cache(self, doc: DocumentChunk, chunk_size: int):
        tokenized_input = (
            self.tokenizer.tokenize(doc.text, return_tensors=True, max_len=chunk_size).to(self.device)
        )
        print(tokenized_input["input_ids"][0])
        with torch.no_grad():
            output = self.model(**tokenized_input, use_cache=True)  # run only prefill

        torch.save(output.past_key_values, os.path.join(self.cache_dir, f"{doc.chunk_id}.pt"))


def main(
    model_name: str,
    db_dir: str,
    doc_dir: str,
    cache_dir: str,
    chunk_size: int,
    max_samples: int
):
    vectordb = VectorDB(db_dir=db_dir)
    tokenizer = Tokenizer(model_name)
    cache_manager = KVCacheBuilder(model_name, tokenizer, cache_dir)

    for filename in sorted(os.listdir(doc_dir))[:max_samples]:
        doc_chunks = tokenizer.split_document(
            filepath=os.path.join(doc_dir, filename),
            chunk_size=chunk_size
        )

        # save text and ids
        vectordb.add_documents(doc_chunks)

        # save cache
        for chunk in doc_chunks:
            cache_manager.save_cache(chunk, chunk_size=chunk_size)


if __name__ == "__main__":
    main(
        model_name="meta-llama/Llama-3.2-3B",
        db_dir="/home/s2/dongseob/preprocessing/db_3b",
        doc_dir="/home/s2/dongseob/preprocessing/qa_data/documents",
        cache_dir="./test_caches",
        chunk_size=512,
        max_samples=1
    )
