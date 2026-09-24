"""Sentence embeddings with ONNX Runtime instead of sentence-transformers/torch.

Same model weights (the ONNX export published in the model's own HF repo), same
tokenizer, same pooling, so vectors match sentence-transformers (cosine 1.00000 on our
corpus) while the image drops torch (~1 GB of dependencies).

The model is loaded once per process (in the app lifespan or once per CLI run).
Inference is CPU-bound and would block the event loop for the whole batch, so it runs
in a worker thread via asyncio.to_thread.
"""

import asyncio
import json
import logging
from typing import Protocol

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from tokenizers import Tokenizer

from app.chunking import CHUNK_TOKENS, Span

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384  # must match chunks.embedding VECTOR(384) in migrations/001_init.sql
BATCH_SIZE = 16


class Embedder(Protocol):
    def token_spans(self, text: str) -> list[Span]: ...

    async def embed_passages(self, texts: list[str]) -> np.ndarray: ...

    async def embed_query(self, text: str) -> np.ndarray: ...


class OnnxEmbedder:
    def __init__(self, model_name: str) -> None:
        tokenizer_path = hf_hub_download(model_name, "tokenizer.json")
        self.max_seq_length = _max_seq_length(model_name)

        # Two instances: chunking needs every token of a long document, inference needs
        # truncation to the model limit plus padding for batches. tokenizer.json files may
        # ship their own settings (all-MiniLM-L6-v2 truncates at 128), so set both explicitly.
        self._span_tokenizer = Tokenizer.from_file(tokenizer_path)
        self._span_tokenizer.no_truncation()
        self._span_tokenizer.no_padding()
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(self.max_seq_length)
        pad_token = "<pad>" if self._tokenizer.token_to_id("<pad>") is not None else "[PAD]"
        self._tokenizer.enable_padding(
            pad_id=self._tokenizer.token_to_id(pad_token) or 0, pad_token=pad_token
        )

        self._session = ort.InferenceSession(
            hf_hub_download(model_name, "onnx/model.onnx"), providers=["CPUExecutionProvider"]
        )
        self._needs_token_types = any(
            i.name == "token_type_ids" for i in self._session.get_inputs()
        )

        # E5 models are trained with these prefixes; without them retrieval quality drops.
        is_e5 = "e5" in model_name.lower()
        self._query_prefix = "query: " if is_e5 else ""
        self._passage_prefix = "passage: " if is_e5 else ""

        dim = self._encode_sync(["probe"]).shape[1]
        if dim != EMBEDDING_DIM:
            raise ValueError(
                f"{model_name} produces {dim}-dim vectors, schema expects {EMBEDDING_DIM}"
            )
        if self.max_seq_length < CHUNK_TOKENS + 16:
            logger.warning(
                "%s truncates input at %d tokens but chunks are %d: chunk tails are lost",
                model_name,
                self.max_seq_length,
                CHUNK_TOKENS,
            )
        logger.info(
            "loaded embedding model %s (onnx, max %d tokens)", model_name, self.max_seq_length
        )

    def token_spans(self, text: str) -> list[Span]:
        # Offsets may include the space before a word ("▁word"); chunk_text strips chunk edges.
        return self._span_tokenizer.encode(text, add_special_tokens=False).offsets

    async def embed_passages(self, texts: list[str]) -> np.ndarray:
        return await asyncio.to_thread(self._encode_sync, [self._passage_prefix + t for t in texts])

    async def embed_query(self, text: str) -> np.ndarray:
        return (await asyncio.to_thread(self._encode_sync, [self._query_prefix + text]))[0]

    def _encode_sync(self, texts: list[str]) -> np.ndarray:
        out = [
            self._encode_batch(texts[i : i + BATCH_SIZE]) for i in range(0, len(texts), BATCH_SIZE)
        ]
        return np.concatenate(out) if out else np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        encoded = self._tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if self._needs_token_types:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self._session.run(None, feeds)[0]  # (batch, seq, dim)
        # Mean pooling over real tokens, as in the sentence-transformers config of e5/MiniLM.
        weights = mask[..., None].astype(np.float32)
        vectors = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        # Normalised vectors: cosine distance in pgvector then equals 1 - dot product.
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _max_seq_length(model_name: str) -> int:
    try:
        with open(hf_hub_download(model_name, "sentence_bert_config.json")) as f:
            return int(json.load(f)["max_seq_length"])
    except (EntryNotFoundError, KeyError):
        return 512
