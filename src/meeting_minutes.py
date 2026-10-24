"""Structured meeting minutes: summary + rule-based extraction.

Design principle: extraction is conservative. Owners, deadlines, decisions
and tasks are copied from transcript text matched by explicit patterns; if
something is not stated it is reported as NOT_SPECIFIED rather than guessed.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Callable

from src.summarization import (
    METHOD_EXTRACTIVE,
    SummaryResult,
    extractive_summary,
    split_sentences,
    summarize_transcript,
)

NOT_SPECIFIED = "Not specified"

# --------------------------------------------------------------------------
# Prompt engineering
#
# DistilBART (the default summarizer) is NOT instruction-tuned: it only ever
# receives raw transcript text, so these templates are never sent to it.
# They are sent, verbatim, to the instruction-tuned Flan-T5 model in the
# "Prompt comparison" feature, and are written so they can be reused with a
# larger local instruction model (e.g. a quantised Llama/Phi/Qwen) later.
# --------------------------------------------------------------------------

SYSTEM_ROLE = (
    "You are a careful professional meeting secretary. Summarize only "
    "information supported by the supplied transcript."
)

MINUTES_PROMPT_TEMPLATE = SYSTEM_ROLE + """

Instructions:
1. Be concise and factual.
2. Preserve every important decision.
3. Extract action items.
4. Give an owner or deadline only if the transcript states it.
5. Do not add facts, names or dates that are not in the transcript.
6. Write "Not specified" when information is missing.
7. Use exactly this structure:
MEETING SUMMARY: <2-3 sentences>
KEY DISCUSSION POINTS: <bullets>
DECISIONS: <bullets or "None recorded">
ACTION ITEMS: <Task | Owner | Deadline>

Transcript:
{transcript}
"""

# Two short instructions for the side-by-side comparison. Kept short because
# Flan-T5-small reads at most 512 tokens.
PROMPT_VARIANTS = {
    "A: Plain instruction": "Summarize this meeting.\n\n{transcript}",
    "B: Role + constraints": (
        SYSTEM_ROLE
        + " Write a concise, factual summary that states the decisions made "
        "and who will do which task by when. Do not invent details.\n\n"
        "Transcript:\n{transcript}"
    ),
}


def build_prompt(transcript: str, template: str = MINUTES_PROMPT_TEMPLATE) -> str:
    return template.format(transcript=transcript.strip())


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------

@dataclass
class ActionItem:
    task: str
    owner: str = NOT_SPECIFIED
    deadline: str = NOT_SPECIFIED
    source: str = ""            # transcript sentence the item came from


@dataclass
class MeetingMinutes:
    summary: str
    summary_method: str
    summary_model: str
    key_points: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    participants: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_text(self) -> str:
        def bullets(items: list[str], empty: str) -> list[str]:
            return [f"- {i}" for i in items] if items else [f"- {empty}"]

        lines = ["MEETING SUMMARY", self.summary or NOT_SPECIFIED, ""]
        if self.participants:
            lines += ["PARTICIPANTS", ", ".join(self.participants), ""]
        lines += ["KEY DISCUSSION POINTS", *bullets(self.key_points, "None identified"), ""]
        lines += ["DECISIONS", *bullets(self.decisions, "None recorded"), ""]
        lines.append("ACTION ITEMS")
        if self.action_items:
            for item in self.action_items:
                lines += [f"- Task: {item.task}", f"  Owner: {item.owner}",
                          f"  Deadline: {item.deadline}", ""]
        else:
            lines += ["- None recorded", ""]
        if self.open_questions:
            lines += ["OPEN QUESTIONS / UNRESOLVED", *bullets(self.open_questions, ""), ""]
        lines += [
            "---",
            f"Summary method: {self.summary_method} ({self.summary_model})",
            "Decisions and action items: rule-based extraction from the transcript.",
            *[f"Note: {n}" for n in self.notes],
            "AI-generated draft. Review against the transcript before distributing.",
        ]
        return "\n".join(lines).strip() + "\n"


# --------------------------------------------------------------------------
# Extraction rules
# --------------------------------------------------------------------------

_SPEAKER_RE = re.compile(r"^([A-Z][A-Za-z.'-]*(?: [A-Z][A-Za-z.'-]*){0,2}):\s+(.*)$")
_NOT_SPEAKERS = {"Action Item", "Decision", "Note", "Agenda", "Task", "Deadline", "Summary"}

# Capitalised words that start sentences but are not people.
_NON_NAMES = {
    "I", "We", "You", "He", "She", "They", "It", "The", "This", "That", "These",
    "Those", "There", "Here", "Also", "So", "And", "But", "Then", "Okay", "Ok",
    "Yes", "No", "Sure", "Maybe", "Everyone", "Someone", "Somebody", "Anyone",
    "Nobody", "Team", "Who", "What", "When", "Where", "Why", "How", "Which",
    "Let", "Lets", "Please", "Thanks", "Well", "Now", "Next", "Today",
    "Tomorrow", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday", "If", "After", "Before", "Once", "Our", "My", "Your",
}

_NAME = r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)?"
_MODAL = (r"(?:will|shall|'ll|is going to|is gonna|needs to|has to|must|should|"
          r"agreed to|volunteered to|is responsible for|will be responsible for)")

_WEEKDAY = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|"
          r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_TIME_WORD = re.compile(
    rf"\b(?:{_WEEKDAY}|{_MONTH}|tomorrow|today|tonight|eod|noon|midnight|"
    r"week|weekend|month|sprint|quarter|deadline|"
    r"\d{1,2}(?::\d{2})?\s*[ap]\.?m\.?|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{1,2}(?:st|nd|rd|th)?)\b",
    re.IGNORECASE,
)
_DEADLINE_PREP = re.compile(
    r"\b(?:by|before|due(?:\s+on|\s+by)?|no later than|until)\s+([^,.;!?]{1,50})",
    re.IGNORECASE,
)
_DEADLINE_BARE = re.compile(
    rf"\b(?:(?:on|this|next)\s+{_WEEKDAY}|next\s+(?:week|month|sprint|meeting)|"
    r"this\s+(?:week|weekend|month)|tomorrow|tonight|end of (?:the )?(?:day|week|month))\b",
    re.IGNORECASE,
)

_DECISION_RE = re.compile(
    r"\b(?:decided|agreed (?:to|that|on)|finali[sz]ed|approved|"
    r"(?:the )?(?:final )?decision (?:is|was)|let'?s go with|"
    r"we(?:'ll| will| are) go(?:ing)? with|we(?:'ve| have) chosen)\b",
    re.IGNORECASE,
)
_UNDECIDED_RE = re.compile(
    r"\b(?:not (?:yet )?decided|haven't decided|undecided|to be decided|tbd|"
    r"not sure|unclear|keep (?:that|this|it) open|open question|no decision)\b",
    re.IGNORECASE,
)

_REQUEST_PATTERNS = [
    # "Rahul, can you also record a demo?"
    re.compile(rf"^(?P<owner>{_NAME}),?\s+(?:can|could|would|will)\s+you\s+(?:please\s+)?(?P<task>.+)$"),
    # "Rahul, please record a demo."
    re.compile(rf"^(?P<owner>{_NAME}),?\s+please\s+(?P<task>.+)$"),
    # "Can you record a demo, Rahul?"
    re.compile(rf"^(?:can|could|would)\s+you\s+(?:please\s+)?(?P<task>.+?),\s+(?P<owner>{_NAME})\s*\?$",
               re.IGNORECASE),
]
# "Aditi will finish the review" / "Rahul to send the report"
_NAMED_RE = re.compile(rf"(?:^|[,;]\s*|\band\s+)(?P<owner>{_NAME})\s+{_MODAL}\s+(?P<task>.+)$")
_NAMED_TO_RE = re.compile(rf"^(?P<owner>{_NAME})\s+to\s+(?P<task>.+)$")
_ASSIGNED_RE = re.compile(
    rf"^(?P<task>.+?)\s+(?:is\s+|was\s+|has been\s+|will be\s+)?assigned to\s+(?P<owner>{_NAME})\b")
_FIRST_PERSON_RE = re.compile(
    r"\bI(?:'ll| will| shall| am going to|'m going to|'m gonna| am gonna)\s+(?P<task>.+)$"
)
_UNASSIGNED_RE = re.compile(
    r"\b(?:we|someone|somebody|anyone|the team)\s+(?:also\s+|still\s+)*"
    r"(?:(?:need|needs|have|has)\s+to|must|should)\s+(?P<task>.+)$",
    re.IGNORECASE,
)
_ACTION_LABEL_RE = re.compile(r"^action item:?\s+(?P<task>.+)$", re.IGNORECASE)

_CLAUSE_TAIL = re.compile(r",?\s+\b(?:but|because|although|though|since|so that|which)\b.*$", re.IGNORECASE)
_VAGUE_TASKS = {"do that", "do it", "do this", "handle it", "handle that", "that", "it",
                "take care of it", "take care of that", "help", "check"}
_CLOSING_RE = re.compile(r"\b(?:good morning|good afternoon|hello|thanks everyone|thank you all|see you|"
                         r"let'?s (?:start|begin|get started))\b",
                         re.IGNORECASE)


def parse_transcript(transcript: str) -> list[tuple[str | None, str]]:
    """Return (speaker, sentence) pairs. Speaker is None for unlabelled text
    (e.g. raw Whisper output, which has no speaker diarisation)."""
    pairs: list[tuple[str | None, str]] = []
    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue
        speaker, text = None, line
        m = _SPEAKER_RE.match(line)
        if m and m.group(1) not in _NOT_SPEAKERS:
            speaker, text = m.group(1), m.group(2)
        for sentence in split_sentences(text):
            pairs.append((speaker, sentence))
    return pairs


def extract_deadline(sentence: str) -> str:
    """Return the deadline phrase exactly as stated, or NOT_SPECIFIED."""
    for m in _DEADLINE_PREP.finditer(sentence):
        phrase = m.group(0)
        times = list(_TIME_WORD.finditer(phrase))
        # The time word must be inside the object of the preposition.
        times = [t for t in times if t.start() >= m.start(1) - m.start(0)]
        if times:
            # "by Friday" -> "Friday"; "before ..." / "until ..." keep their meaning.
            phrase = re.sub(r"^(?:by|due(?:\s+on|\s+by)?)\s+", "", phrase[: times[-1].end()].strip(),
                            flags=re.IGNORECASE)
            return phrase
    m = _DEADLINE_BARE.search(sentence)
    return m.group(0) if m else NOT_SPECIFIED


def _clean_task(task: str, deadline: str) -> str:
    task = _CLAUSE_TAIL.sub("", task)
    if deadline != NOT_SPECIFIED:
        task = re.sub(r"\s*\b(?:(?:by|due(?:\s+on|\s+by)?|on|no later than)\s+)?" + re.escape(deadline),
                      "", task, flags=re.IGNORECASE)
    task = re.sub(r"^(?:also|please|then)\s+", "", task.strip(), flags=re.IGNORECASE)
    task = re.sub(r"\s+(?:also|please)\b", "", task)
    task = re.sub(r"\s{2,}", " ", task).strip(" ,.;:!?-")
    return task[:1].upper() + task[1:] if task else ""


def _is_name(word: str, speakers: set[str]) -> bool:
    first = word.split()[0]
    if first in _NON_NAMES:
        return False
    # When the transcript has speaker labels, require a known participant to
    # avoid treating products/tools ("Whisper will...") as owners.
    if speakers:
        return any(first == s.split()[0] for s in speakers)
    return True


def _match_action(speaker: str | None, sentence: str, speakers: set[str]) -> ActionItem | None:
    text = sentence.strip()
    deadline = extract_deadline(text)

    def make(task: str, owner: str) -> ActionItem | None:
        task = _clean_task(task, deadline)
        if not task or task.lower() in _VAGUE_TASKS or len(task.split()) < 2:
            return None
        return ActionItem(task=task, owner=owner, deadline=deadline, source=sentence)

    for pattern in _REQUEST_PATTERNS:
        m = pattern.match(text)
        # A directly addressed name ("Rahul, can you...") is a person even if
        # they never speak in the transcript, so skip the participant check.
        if m and _is_name(m.group("owner"), set()):
            return make(m.group("task"), m.group("owner"))
    if text.endswith("?"):
        return None  # other questions are open questions, not commitments
    m = _ACTION_LABEL_RE.match(text)
    if m:
        inner = _match_action(speaker, m.group("task"), speakers)
        return inner or make(m.group("task"), NOT_SPECIFIED)
    for pattern in (_NAMED_TO_RE, _NAMED_RE, _ASSIGNED_RE):
        m = pattern.search(text)
        if m and _is_name(m.group("owner"), speakers):
            return make(m.group("task"), m.group("owner"))
    m = _FIRST_PERSON_RE.search(text)
    if m:
        owner = speaker or f"{NOT_SPECIFIED} (speaker unknown)"
        return make(m.group("task"), owner)
    m = _UNASSIGNED_RE.search(text)
    if m:
        return make(m.group("task"), NOT_SPECIFIED)
    return None


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 3}


def _drop_covered_unassigned(items: list[ActionItem]) -> list[ActionItem]:
    """Drop a generic unassigned task ("we need to finish X") when a later or
    earlier item assigns the same work to someone explicitly."""
    assigned = [i for i in items if not i.owner.startswith(NOT_SPECIFIED)]
    kept = []
    for item in items:
        if item.owner.startswith(NOT_SPECIFIED):
            words = _content_words(item.task)
            covered = any(
                (aw := _content_words(a.task)) and len(aw & words) / len(aw) >= 0.6
                for a in assigned
            )
            if covered:
                continue
        kept.append(item)
    return kept


def _fmt(speaker: str | None, sentence: str) -> str:
    return f"{speaker}: {sentence}" if speaker else sentence


def extract_structure(transcript: str) -> dict:
    """Rule-based extraction of participants, decisions, action items,
    open questions and key discussion points."""
    pairs = parse_transcript(transcript)
    speakers = {s for s, _ in pairs if s}
    participants = list(dict.fromkeys(s for s, _ in pairs if s))

    decisions, actions, questions, used = [], [], [], set()
    for idx, (speaker, sentence) in enumerate(pairs):
        undecided = bool(_UNDECIDED_RE.search(sentence))
        if not undecided and _DECISION_RE.search(sentence):
            decisions.append(_fmt(speaker, sentence))
            used.add(idx)
            continue
        item = _match_action(speaker, sentence, speakers)
        if item:
            actions.append(item)
            used.add(idx)
        if undecided or (sentence.endswith("?") and not item):
            questions.append(_fmt(speaker, sentence))
            used.add(idx)

    # Key points: highest-ranked remaining sentences, in original order.
    remaining = [
        (i, s, t) for i, (s, t) in enumerate(pairs)
        if i not in used and len(t.split()) >= 6 and not _CLOSING_RE.search(t)
    ]
    ranked_text = extractive_summary("\n".join(t for _, _, t in remaining), max_sentences=4)
    key_points = [_fmt(s, t) for _, s, t in remaining if t in ranked_text]

    return {
        "participants": participants,
        "decisions": decisions,
        "action_items": _drop_covered_unassigned(actions),
        "open_questions": questions,
        "key_points": key_points,
    }


def generate_minutes(
    transcript: str,
    summarizer: Callable[[str], SummaryResult] = summarize_transcript,
) -> MeetingMinutes:
    """Build structured minutes. `summarizer` is injectable for testing; if
    it fails, the extractive fallback is used and noted."""
    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty. Record, upload or paste a transcript first.")

    notes: list[str] = []
    try:
        result = summarizer(transcript)
    except Exception as exc:
        result = SummaryResult(
            text=extractive_summary(transcript),
            method=METHOD_EXTRACTIVE,
            model="Frequency-based extractive summarizer (not a neural model)",
            processing_time_s=0.0,
            notes=[f"Summarizer failed, used extractive fallback. Reason: {exc}"],
        )
    notes.extend(result.notes)

    structure = extract_structure(transcript)
    return MeetingMinutes(
        summary=result.text,
        summary_method=result.method,
        summary_model=result.model,
        notes=notes,
        **structure,
    )
