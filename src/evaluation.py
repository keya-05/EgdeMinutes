"""Lightweight evaluation: length statistics, compression, timing and ROUGE.

None of these metrics prove factual accuracy. ROUGE measures n-gram overlap
with a reference; a summary can score well and still contain a hallucinated
name or deadline. Human review against the transcript is still required.
"""

from __future__ import annotations

import re
import time
from contextlib import contextmanager


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w[\w'-]*\b", text or ""))


def compression_ratio(transcript: str, summary: str) -> float:
    """Summary words / transcript words (lower = more compressed)."""
    source = word_count(transcript)
    return round(word_count(summary) / source, 3) if source else 0.0


@contextmanager
def timer(results: dict, key: str):
    """Record elapsed seconds of a block into results[key]."""
    start = time.perf_counter()
    try:
        yield
    finally:
        results[key] = round(time.perf_counter() - start, 2)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _f1(overlap: int, cand: int, ref: int) -> dict:
    p = overlap / cand if cand else 0.0
    r = overlap / ref if ref else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def _ngram_overlap(c: list[str], r: list[str], n: int) -> dict:
    from collections import Counter

    cg = Counter(tuple(c[i:i + n]) for i in range(len(c) - n + 1))
    rg = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
    return _f1(sum((cg & rg).values()), sum(cg.values()), sum(rg.values()))


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_scores(candidate: str, reference: str) -> tuple[dict, str]:
    """ROUGE-1/2/L. Uses the `rouge-score` package when installed, otherwise
    an equivalent built-in implementation (no stemming). Returns
    (scores, implementation_name)."""
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        raw = scorer.score(reference, candidate)
        scores = {k: {"precision": round(v.precision, 4), "recall": round(v.recall, 4),
                      "f1": round(v.fmeasure, 4)} for k, v in raw.items()}
        return scores, "rouge-score (Porter stemmer)"
    except ImportError:
        c, r = _tokens(candidate), _tokens(reference)
        scores = {
            "rouge1": _ngram_overlap(c, r, 1),
            "rouge2": _ngram_overlap(c, r, 2),
            "rougeL": _f1(_lcs(c, r), len(c), len(r)),
        }
        return scores, "built-in (no stemming)"


def summary_statistics(transcript: str, summary: str, timings: dict | None = None) -> dict:
    stats = {
        "transcript_words": word_count(transcript),
        "summary_words": word_count(summary),
        "compression_ratio": compression_ratio(transcript, summary),
    }
    stats.update(timings or {})
    return stats
