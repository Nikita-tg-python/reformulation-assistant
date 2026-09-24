"""Split text into overlapping chunks measured in tokens.

"Tokens" here are the embedding model's own tokenizer tokens, not words. The model's
input limit (max_seq_length) is counted in these units, so measuring chunks the same
way guarantees a chunk is embedded whole instead of being silently truncated. Words
would be misleading: for Ukrainian text one word is ~2 tokens in multilingual-e5-small
and ~5 in the English-only all-MiniLM-L6-v2.

The tokenizer is passed in as a function returning character spans, so this module
stays pure: no model, no I/O, trivially testable with a whitespace tokenizer.
"""

from collections.abc import Callable

Span = tuple[int, int]
Tokenize = Callable[[str], list[Span]]

CHUNK_TOKENS = 400
OVERLAP_TOKENS = 50


def chunk_text(
    text: str,
    tokenize: Tokenize,
    chunk_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[str]:
    """Return windows of `chunk_tokens` tokens, each overlapping the previous by `overlap_tokens`.

    Chunks are sliced from the original text by character offsets, so the stored text
    keeps its casing, punctuation and letters like "й"/"ї" exactly as written.
    A window may end mid-word (subword tokens); the overlap carries the rest of the word
    into the next chunk.
    """
    if chunk_tokens <= 0 or not 0 <= overlap_tokens < chunk_tokens:
        raise ValueError("require chunk_tokens > 0 and 0 <= overlap_tokens < chunk_tokens")
    if not text.strip():
        return []

    spans = [(start, end) for start, end in tokenize(text) if end > start]
    step = chunk_tokens - overlap_tokens
    chunks = []
    for first in range(0, len(spans), step):
        window = spans[first : first + chunk_tokens]
        chunks.append(text[window[0][0] : window[-1][1]].strip())
        if first + chunk_tokens >= len(spans):
            break
    return chunks
