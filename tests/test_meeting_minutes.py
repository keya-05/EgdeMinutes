"""Unit tests for the deterministic parts of EdgeMinutes.

The summarizer is mocked, so no model is downloaded or loaded.
"""

import json
from pathlib import Path

import pytest

from src import summarization
from src.evaluation import compression_ratio, rouge_scores, summary_statistics, word_count
from src.meeting_minutes import (
    MINUTES_PROMPT_TEMPLATE,
    NOT_SPECIFIED,
    SYSTEM_ROLE,
    build_prompt,
    extract_deadline,
    extract_structure,
    generate_minutes,
)
from src.summarization import METHOD_EXTRACTIVE, SummaryResult, extractive_summary

SAMPLE = Path(__file__).resolve().parents[1] / "sample_data" / "sample_transcript.txt"


def fake_summarizer(text: str) -> SummaryResult:
    return SummaryResult(text="Mock summary.", method="mock", model="mock-model", processing_time_s=0.0)


def failing_summarizer(text: str) -> SummaryResult:
    raise RuntimeError("model download failed")


def actions(transcript: str):
    return extract_structure(transcript)["action_items"]


# ------------------------------------------------------------ empty input

@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_empty_transcript_raises(text):
    with pytest.raises(ValueError):
        generate_minutes(text, summarizer=fake_summarizer)


def test_summarize_transcript_rejects_empty():
    with pytest.raises(ValueError):
        summarization.summarize_transcript("  ")


# ------------------------------------------------------------ action items

def test_explicit_action_item_is_extracted():
    items = actions("Aditi: I will complete the literature review by Friday.")
    assert len(items) == 1
    assert items[0].task == "Complete the literature review"


def test_explicit_owner_and_deadline():
    items = actions("Priya will send the budget report by tomorrow.")
    assert (items[0].owner, items[0].deadline) == ("Priya", "tomorrow")


def test_first_person_owner_is_the_speaker():
    items = actions("Rahul: I will prepare the slides by next Monday.")
    assert items[0].owner == "Rahul"
    assert items[0].deadline == "next Monday"


def test_request_addressed_to_named_person():
    items = actions("Aditi: Rahul, can you record a demo video before the submission deadline?")
    assert items[0].owner == "Rahul"
    assert items[0].task == "Record a demo video"
    assert items[0].deadline == "before the submission deadline"


def test_task_without_owner_is_not_specified():
    items = actions("We need to fix the login bug by Monday.")
    assert items[0].owner == NOT_SPECIFIED
    assert items[0].deadline == "Monday"


def test_task_without_deadline_is_not_specified():
    items = actions("Meera will update the README.")
    assert items[0].owner == "Meera"
    assert items[0].deadline == NOT_SPECIFIED


def test_no_action_items_when_none_present():
    text = "Aditi: The demo went well.\nRahul: Yes, the audience liked the interface."
    assert actions(text) == []


def test_unknown_names_are_not_invented_as_owners():
    # With speaker labels present, a non-participant capitalised word
    # (a tool name) must not become an owner.
    items = actions("Aditi: Whisper will handle the audio.\nRahul: Okay.")
    assert all(i.owner != "Whisper" for i in items)


def test_question_is_not_an_action_item():
    out = extract_structure("Aditi: Do we have the reference summaries ready?")
    assert out["action_items"] == []
    assert len(out["open_questions"]) == 1


def test_deadline_extraction_variants():
    assert extract_deadline("Send it before March 3.") == "before March 3"
    assert extract_deadline("Karan to book the room by 5 pm.") == "5 pm"
    assert extract_deadline("Review it soon.") == NOT_SPECIFIED


# ------------------------------------------------------------ decisions

def test_explicit_decision_is_extracted():
    out = extract_structure("Rahul: We decided to use DistilBART for summarization.")
    assert len(out["decisions"]) == 1
    assert "DistilBART" in out["decisions"][0]


def test_undecided_statement_is_not_a_decision():
    out = extract_structure("Aditi: We have not decided who will test on Android.")
    assert out["decisions"] == []
    assert out["open_questions"]


# ------------------------------------------------------------ sample transcript

def test_sample_transcript_end_to_end():
    minutes = generate_minutes(SAMPLE.read_text(encoding="utf-8"), summarizer=fake_summarizer)
    assert minutes.participants == ["Aditi", "Rahul"]
    assert len(minutes.decisions) == 1
    by_task = {i.task: i for i in minutes.action_items}
    assert by_task["Complete the literature review"].owner == "Aditi"
    assert by_task["Complete the literature review"].deadline == "Friday"
    assert by_task["Prepare the presentation slides"].owner == "Rahul"
    # The ambiguous task must stay unassigned, not be guessed.
    android = next(i for i in minutes.action_items if "Android" in i.task)
    assert android.owner == NOT_SPECIFIED


# ------------------------------------------------------------ output structure

def test_output_structure_and_download_text():
    minutes = generate_minutes("Meera will update the README by Friday.", summarizer=fake_summarizer)
    data = minutes.to_dict()
    for key in ("summary", "key_points", "decisions", "action_items", "open_questions", "participants"):
        assert key in data
    assert set(data["action_items"][0]) == {"task", "owner", "deadline", "source"}
    text = minutes.to_text()
    for heading in ("MEETING SUMMARY", "KEY DISCUSSION POINTS", "DECISIONS", "ACTION ITEMS"):
        assert heading in text
    assert "Owner: Meera" in text and "Deadline: Friday" in text
    assert json.loads(minutes.to_json())["summary"] == "Mock summary."


def test_summarizer_failure_falls_back_to_extractive():
    minutes = generate_minutes(SAMPLE.read_text(encoding="utf-8"), summarizer=failing_summarizer)
    assert minutes.summary_method == METHOD_EXTRACTIVE
    assert minutes.summary
    assert any("model download failed" in n for n in minutes.notes)


def test_transformer_failure_falls_back(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no internet")

    monkeypatch.setattr(summarization, "transformer_summary", boom)
    result = summarization.summarize_transcript("Aditi: We met today. Rahul: The plan is ready.")
    assert result.method == METHOD_EXTRACTIVE
    assert "no internet" in result.notes[0]


def test_extractive_summary_only_uses_input_sentences():
    text = SAMPLE.read_text(encoding="utf-8")
    plain = text.replace("Aditi: ", "").replace("Rahul: ", "")
    for sentence in extractive_summary(text).split(". "):
        assert sentence.rstrip(".") in plain


# ------------------------------------------------------------ prompt + evaluation

def test_prompt_template_contains_role_and_transcript():
    prompt = build_prompt("Aditi: hello")
    assert SYSTEM_ROLE in prompt and "Aditi: hello" in prompt
    assert "Not specified" in MINUTES_PROMPT_TEMPLATE


def test_evaluation_metrics():
    assert word_count("one two three four") == 4
    assert compression_ratio("one two three four", "one two") == 0.5
    assert compression_ratio("", "x") == 0.0
    scores, _ = rouge_scores("the cat sat", "the cat sat")
    assert scores["rouge1"]["f1"] == pytest.approx(1.0)
    stats = summary_statistics("a b c d", "a b", {"summarization_s": 1.5})
    assert stats["compression_ratio"] == 0.5 and stats["summarization_s"] == 1.5
