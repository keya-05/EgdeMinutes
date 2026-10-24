"""Transcript summarization.

Primary method: DistilBART (sshleifer/distilbart-cnn-12-6), a distilled
BART encoder-decoder Transformer fine-tuned on CNN/DailyMail news summaries.
It is NOT instruction-tuned: it receives only the transcript text, never a
prompt.

Fallback method: a frequency-based extractive summarizer (pure Python). It
selects existing sentences; it is not a neural model and not an LLM.

Optional: an instruction-tuned model (google/flan-t5-small) that DOES receive
the prompt templates from meeting_minutes.py, used for the prompt comparison
feature.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

SUMMARY_MODEL = "sshleifer/distilbart-cnn-12-6"
INSTRUCTION_MODEL = "google/flan-t5-small"
CHUNK_TOKENS = 900          # DistilBART accepts at most 1024 input tokens
DEFAULT_MAX_CHUNKS = 4      # bounds CPU time on long meetings
INSTRUCTION_MAX_TOKENS = 512

METHOD_TRANSFORMER = "transformer"
METHOD_EXTRACTIVE = "extractive"

STOPWORDS = set(
    """a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers herself him himself his how
    i if in into is it its itself just let lets me more most my myself no nor not now of
    off on once only or other our ours ourselves out over own same she should so some such
    than that the their theirs them themselves then there these they this those through to
    too under until up very was we were what when where which while who whom why will with
    would you your yours yourself yourselves okay ok yes yeah sure thanks thank also still
    i'm i'll we'll it's that's don't let's going get got really well good""".split()
)


class SummarizationError(RuntimeError):
    """Raised when the transformer summarizer cannot run."""


@dataclass
class SummaryResult:
    text: str
    method: str                 # METHOD_TRANSFORMER or METHOD_EXTRACTIVE
    model: str
    processing_time_s: float
    chunks_used: int = 1
    chunks_total: int = 1
    truncated: bool = False
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- utilities

def split_sentences(text: str) -> list[str]:
    """Split text into sentences, treating each transcript line separately."""
    sentences = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", line)
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences


def _strip_speaker(sentence: str) -> str:
    return re.sub(r"^[A-Z][\w .'-]{0,30}:\s+", "", sentence)


# ------------------------------------------------------ extractive fallback

def extractive_summary(text: str, max_sentences: int = 3) -> str:
    """Score sentences by normalised content-word frequency; keep the top ones
    in their original order. Purely extractive: every output sentence exists
    verbatim (minus speaker label) in the input."""
    sentences = split_sentences(text)
    if not sentences:
        return ""
    if len(sentences) <= max_sentences:
        return " ".join(_strip_speaker(s) for s in sentences)

    def words(s: str) -> list[str]:
        return [w for w in re.findall(r"[a-z']+", _strip_speaker(s).lower())
                if w not in STOPWORDS and len(w) > 2]

    freq = Counter(w for s in sentences for w in words(s))
    if not freq:
        return " ".join(_strip_speaker(s) for s in sentences[:max_sentences])
    top = max(freq.values())

    def score(s: str) -> float:
        ws = words(s)
        if not ws:
            return 0.0
        return sum(freq[w] / top for w in ws) / math.sqrt(len(ws))

    ranked = sorted(range(len(sentences)), key=lambda i: score(sentences[i]), reverse=True)
    keep = sorted(ranked[:max_sentences])
    return " ".join(_strip_speaker(sentences[i]) for i in keep)


# ------------------------------------------------------ transformer models

@lru_cache(maxsize=2)
def _load_seq2seq(model_name: str):
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise SummarizationError(
            "transformers/torch are not installed. Run: pip install -r requirements.txt"
        ) from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.eval()
        return tokenizer, model
    except Exception as exc:
        raise SummarizationError(
            f"Could not load model '{model_name}'. The first run needs internet "
            f"access to download it. Details: {exc}"
        ) from exc


def load_summarizer():
    return _load_seq2seq(SUMMARY_MODEL)


def _chunk_by_tokens(sentences: list[str], tokenizer, budget: int) -> list[str]:
    chunks, current, size = [], [], 0
    for s in sentences:
        n = len(tokenizer.encode(s, add_special_tokens=False))
        if current and size + n > budget:
            chunks.append(" ".join(current))
            current, size = [], 0
        current.append(s)
        size += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def _generate(tokenizer, model, text: str, max_input: int, **gen_kwargs) -> str:
    import torch

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input)
    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def transformer_summary(text: str, max_chunks: int = DEFAULT_MAX_CHUNKS) -> SummaryResult:
    """Abstractive summary with DistilBART. Long transcripts are split into
    ~900-token chunks; at most `max_chunks` are summarised and the result
    records explicitly if the rest of the transcript was skipped."""
    start = time.perf_counter()
    tokenizer, model = load_summarizer()
    sentences = split_sentences(text)
    chunks = _chunk_by_tokens(sentences, tokenizer, CHUNK_TOKENS)
    used = chunks[:max_chunks]
    notes = []
    if len(chunks) > max_chunks:
        notes.append(
            f"Transcript was long: only the first {max_chunks} of {len(chunks)} "
            f"chunks (~{CHUNK_TOKENS} tokens each) were summarised. The rest was NOT processed."
        )
    parts = []
    for chunk in used:
        n_tokens = len(tokenizer.encode(chunk))
        # Keep the summary shorter than its input for short meetings.
        max_len = max(30, min(142, int(n_tokens * 0.6)))
        min_len = min(20, max_len // 2)
        parts.append(_generate(
            tokenizer, model, chunk, 1024,
            num_beams=2, max_length=max_len, min_length=min_len,
            no_repeat_ngram_size=3, early_stopping=True,
        ))
    text = re.sub(r"\s+([.,;:!?])", r"\1", " ".join(parts))  # "May 14 ." -> "May 14."
    return SummaryResult(
        text=text,
        method=METHOD_TRANSFORMER,
        model=SUMMARY_MODEL,
        processing_time_s=round(time.perf_counter() - start, 2),
        chunks_used=len(used),
        chunks_total=len(chunks),
        truncated=len(chunks) > max_chunks,
        notes=notes,
    )


def summarize_transcript(
    transcript: str,
    method: str = METHOD_TRANSFORMER,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> SummaryResult:
    """Summarise a transcript. Falls back to the extractive method (and says
    so in `notes`) if the transformer model cannot be loaded or run."""
    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty.")
    fallback_note = ""
    if method == METHOD_TRANSFORMER:
        try:
            return transformer_summary(transcript, max_chunks=max_chunks)
        except Exception as exc:
            fallback_note = f"Transformer summarizer unavailable, used extractive fallback. Reason: {exc}"

    start = time.perf_counter()
    text = extractive_summary(transcript)
    return SummaryResult(
        text=text,
        method=METHOD_EXTRACTIVE,
        model="Frequency-based extractive summarizer (not a neural model)",
        processing_time_s=round(time.perf_counter() - start, 3),
        notes=[fallback_note] if fallback_note else [],
    )


def run_instruction_model(prompt: str) -> tuple[str, bool, float]:
    """Send a full prompt to the instruction-tuned Flan-T5-small model.
    Returns (output, input_was_truncated, seconds)."""
    start = time.perf_counter()
    tokenizer, model = _load_seq2seq(INSTRUCTION_MODEL)
    truncated = len(tokenizer.encode(prompt)) > INSTRUCTION_MAX_TOKENS
    output = _generate(
        tokenizer, model, prompt, INSTRUCTION_MAX_TOKENS,
        num_beams=2, max_new_tokens=120, no_repeat_ngram_size=3,
    )
    return output, truncated, round(time.perf_counter() - start, 2)
