"""Metrics of eval/run.py, without a database or the model."""

from eval.run import recall_and_rr, report, top_documents


def test_top_documents_deduplicates_chunks_by_document_keeping_rank_order():
    chunks = ["SPEC-007", "SPEC-007", "GUIDE-002", "SPEC-008", "GUIDE-002", "A", "B", "C"]
    assert top_documents(chunks) == ["SPEC-007", "GUIDE-002", "SPEC-008", "A", "B"]


def test_recall_and_reciprocal_rank_use_the_first_expected_document():
    assert recall_and_rr(["A", "B", "C"], {"C", "B"}) == (True, 0.5)
    assert recall_and_rr(["A", "B"], {"X"}) == (False, 0.0)


def test_report_fails_below_the_recall_threshold():
    def rows(found: list[bool]):
        return [
            {"question": f"q{n}", "kind": "semantic", "found": f, "rr": 1.0 if f else 0.0}
            for n, f in enumerate(found)
        ]

    table, ok = report({"vector": rows([True] * 8 + [False] * 2), "hybrid": rows([True] * 10)})
    assert ok  # 0.8 is enough
    assert "| vector | 0.80 | 0.80 | 0/0 |" in table
    _, ok = report({"vector": rows([True] * 7 + [False] * 3)})
    assert not ok
