import itertools
import re

import pytest

from app.chunking import CHUNK_TOKENS, OVERLAP_TOKENS, chunk_text


def words(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in re.finditer(r"\S+", text)]


def numbered(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


@pytest.mark.parametrize("text", ["", "   ", "\n\t \n"])
def test_empty_or_blank_text_gives_no_chunks(text):
    assert chunk_text(text, words) == []


def test_text_without_tokens_gives_no_chunks():
    assert chunk_text("some text", lambda _: []) == []


def test_short_text_is_one_stripped_chunk():
    assert chunk_text("  Аквафаба замінює яйце.  ", words) == ["Аквафаба замінює яйце."]


def test_defaults_are_400_tokens_with_50_overlap():
    assert (CHUNK_TOKENS, OVERLAP_TOKENS) == (400, 50)


def test_chunk_sizes_and_overlap():
    chunks = chunk_text(numbered(1000), words)

    assert [len(c.split()) for c in chunks] == [400, 400, 300]
    for prev, nxt in itertools.pairwise(chunks):
        assert prev.split()[-50:] == nxt.split()[:50]


@pytest.mark.parametrize(
    ("n_tokens", "sizes"),
    [(1, [1]), (400, [400]), (401, [400, 51]), (750, [400, 400]), (751, [400, 400, 51])],
)
def test_window_boundaries(n_tokens, sizes):
    assert [len(c.split()) for c in chunk_text(numbered(n_tokens), words)] == sizes


def test_every_token_is_covered_in_order():
    tokens = numbered(1234).split()
    chunks = chunk_text(" ".join(tokens), words)

    rebuilt = chunks[0].split()
    for chunk in chunks[1:]:
        rebuilt += chunk.split()[OVERLAP_TOKENS:]
    assert rebuilt == tokens


def test_chunks_keep_original_text_exactly():
    text = "Кокосове   молоко й закваска: їх треба замінити!\nДив. SPEC-009."
    assert chunk_text(text, words, chunk_tokens=4, overlap_tokens=1) == [
        "Кокосове   молоко й закваска:",
        "закваска: їх треба замінити!",
        "замінити!\nДив. SPEC-009.",
    ]


def test_custom_sizes():
    chunks = chunk_text(numbered(10), words, chunk_tokens=4, overlap_tokens=2)
    assert chunks == ["w0 w1 w2 w3", "w2 w3 w4 w5", "w4 w5 w6 w7", "w6 w7 w8 w9"]


def test_zero_width_tokens_are_ignored():
    # Some tokenizers emit empty spans (e.g. a lone "▁" piece).
    def with_empty(text):
        return [(0, 0), *words(text), (len(text), len(text))]

    assert chunk_text("a b c", with_empty, chunk_tokens=2, overlap_tokens=0) == ["a b", "c"]


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-1, 0), (10, 10), (10, 11), (10, -1)])
def test_invalid_sizes_are_rejected(size, overlap):
    with pytest.raises(ValueError, match="chunk_tokens"):
        chunk_text("a b c", words, chunk_tokens=size, overlap_tokens=overlap)
