"""EdgeMinutes: Streamlit UI. Run with:  streamlit run app.py"""

from __future__ import annotations

import platform
import socket
from functools import partial
from pathlib import Path

import pandas as pd
import streamlit as st

from src.evaluation import rouge_scores, summary_statistics, timer
from src.meeting_minutes import (
    MINUTES_PROMPT_TEMPLATE,
    PROMPT_VARIANTS,
    build_prompt,
    generate_minutes,
)
from src.mic_recorder import mic_recorder
from src.summarization import (
    INSTRUCTION_MODEL,
    METHOD_EXTRACTIVE,
    METHOD_TRANSFORMER,
    SUMMARY_MODEL,
    load_summarizer,
    run_instruction_model,
    summarize_transcript,
)
from src.transcription import (
    MODEL_SIZES,
    SUPPORTED_EXTENSIONS,
    TranscriptionError,
    load_whisper_model,
    transcribe_bytes,
)

SAMPLE_PATH = Path(__file__).parent / "sample_data" / "sample_transcript.txt"
LANGUAGES = {"Auto-detect": None, "English": "en", "Hindi": "hi", "Marathi": "mr",
             "Tamil": "ta", "Telugu": "te", "Bengali": "bn", "Gujarati": "gu",
             "French": "fr", "German": "de", "Spanish": "es"}
SOURCES = ["Record audio", "Upload audio", "Paste transcript", "Demo sample transcript"]

st.set_page_config(page_title="EdgeMinutes", page_icon="📝", layout="wide")
state = st.session_state

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    whisper_size = st.selectbox("Whisper model", MODEL_SIZES, index=0,
                                help="tiny ≈ 75 MB (fastest), base ≈ 145 MB, small ≈ 480 MB.")
    language_name = st.selectbox("Spoken language", list(LANGUAGES), index=0)
    method_label = st.radio(
        "Summarizer",
        ["Transformer (DistilBART, abstractive)", "Extractive fallback (no neural model)"],
        help="The extractive option selects existing sentences and needs no model download.",
    )
    method = METHOD_TRANSFORMER if method_label.startswith("Transformer") else METHOD_EXTRACTIVE
    max_chunks = st.slider("Max transcript chunks to summarise", 1, 8, 4,
                           help="Each chunk is ~900 tokens. Larger values are slower on CPU.")

    with st.expander("Execution information", expanded=False):
        st.markdown(f"""
**This app's Python process runs on:** `{socket.gethostname()}`
({platform.system()} {platform.release()}, {platform.machine()}).

- **Transcription:** faster-whisper `{whisper_size}` (int8) on **this computer's CPU**.
- **Summarization:** `{SUMMARY_MODEL}` on **this computer's CPU**.
- **Extraction:** rule-based Python on this computer.
- **Remote servers:** none. Audio and transcripts are not sent to any API.
- **Internet:** only for the first download of model weights from Hugging Face.
- **Phone:** if you open this page from a phone, the phone is only a browser
  (upload/record). **All AI inference still runs on this computer**, not on the phone.
""")

# ------------------------------------------------------------------ header
st.title("EdgeMinutes — AI Meeting Minutes Summarizer")
st.caption(
    "Converts meeting audio into text and generates concise, structured minutes with key "
    "decisions and action items, using local open-source models (no API keys, no cloud)."
)

# ------------------------------------------------------------------ input
source = st.radio("Input source", SOURCES, horizontal=True)
audio_bytes, audio_suffix, typed_text = None, None, ""

if source == "Record audio":
    st.markdown("**Record the meeting**")
    recording = mic_recorder(key="mic")
    if recording is not None:
        audio_bytes, audio_suffix, rec_info = recording
        st.audio(audio_bytes, format=rec_info["mime"].split(";")[0] or None)
        st.caption(f"Recorded {rec_info['duration_ms'] / 1000:.1f}s · {rec_info['mime']} · "
                   f"{rec_info['size'] / 1024:.0f} KB")
    st.caption("Browsers only allow microphone access on `localhost` or HTTPS pages.")
    if hasattr(st, "audio_input"):
        with st.expander("Alternative: Streamlit's built-in recorder"):
            # sample_rate=None keeps the browser's native rate: resampling to 16 kHz
            # in the browser fails on some Android phones, and Whisper resamples anyway.
            rec = st.audio_input("Built-in recorder", sample_rate=None)
            if rec is not None:
                audio_bytes, audio_suffix = rec.getvalue(), ".wav"
elif source == "Upload audio":
    up = st.file_uploader("Upload meeting audio", type=list(SUPPORTED_EXTENSIONS))
    if up is not None:
        audio_bytes, audio_suffix = up.getvalue(), Path(up.name).suffix.lower()
        st.audio(audio_bytes)
elif source == "Paste transcript":
    typed_text = st.text_area(
        "Paste a transcript (optionally with `Name: text` speaker labels)", height=220,
        placeholder="Aditi: I will finish the report by Friday.\nRahul: We decided to ...")
else:
    typed_text = SAMPLE_PATH.read_text(encoding="utf-8")
    st.info("**DEMO DATA:** this is a hand-written sample transcript, not recognised speech. "
            "The summary generated from it is real model output.")
    with st.expander("View sample transcript"):
        st.text(typed_text)

# ------------------------------------------------------------------ run
if st.button("Generate Meeting Minutes", type="primary", width="stretch"):
    timings: dict = {}
    transcript, asr_info = "", None
    try:
        if audio_bytes is not None:
            with st.status("Processing audio…", expanded=True) as status:
                st.write(f"Loading Whisper `{whisper_size}` (first run downloads the model)…")
                load_whisper_model(whisper_size)
                st.write("Transcribing on CPU…")
                asr_info = transcribe_bytes(audio_bytes, audio_suffix, whisper_size,
                                            LANGUAGES[language_name])
                transcript = asr_info.text
                timings["transcription_s"] = asr_info.processing_time_s
                status.update(label="Transcription complete", state="complete")
        elif source in ("Record audio", "Upload audio"):
            st.error("No audio yet. Record or upload a file first.")
            st.stop()
        else:
            transcript = typed_text.strip()
            if not transcript:
                st.error("The transcript is empty. Paste some text first.")
                st.stop()

        if method == METHOD_TRANSFORMER:
            with st.spinner(f"Loading {SUMMARY_MODEL} (first run downloads ~1.2 GB)…"):
                try:
                    load_summarizer()
                except Exception as exc:  # summarize_transcript falls back and notes it
                    st.warning(f"Transformer model unavailable: {exc}")
        with st.spinner("Summarising and extracting decisions and action items…"):
            summarizer = partial(summarize_transcript, method=method, max_chunks=max_chunks)
            with timer(timings, "minutes_generation_s"):
                minutes = generate_minutes(transcript, summarizer=summarizer)
        state.update(transcript=transcript, minutes=minutes, asr=asr_info,
                     timings=timings, is_demo=source.startswith("Demo"))
    except TranscriptionError as exc:
        st.error(f"Transcription error: {exc}")
        st.info("You can still use **Paste transcript** or **Demo sample transcript**.")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")

# ------------------------------------------------------------------ results
if "minutes" in state:
    minutes, transcript = state.minutes, state.transcript
    if state.is_demo:
        st.info("Results below were generated from the **demo sample transcript**.")

    col_t, col_m = st.columns([2, 3])
    with col_t:
        st.subheader("Transcript")
        if state.asr:
            a = state.asr
            st.caption(f"Whisper {a.model_size} · language `{a.language}` "
                       f"(p={a.language_probability}) · audio {a.audio_duration_s}s · "
                       f"processed in {a.processing_time_s}s")
        st.text_area("transcript", transcript, height=420, label_visibility="collapsed")

    with col_m:
        st.subheader("Meeting Minutes")
        method_name = ("DistilBART transformer (abstractive)" if minutes.summary_method == METHOD_TRANSFORMER
                       else "Extractive fallback — selected sentences, not a neural model")
        for note in minutes.notes:
            st.warning(note)
        st.markdown("**Meeting summary**")
        st.write(minutes.summary)
        st.caption(f"Summary method: {method_name}")
        if minutes.participants:
            st.markdown(f"**Participants:** {', '.join(minutes.participants)}")
        st.markdown("**Key discussion points**")
        st.markdown("\n".join(f"- {p}" for p in minutes.key_points) or "- None identified")
        st.markdown("**Decisions**")
        st.markdown("\n".join(f"- {d}" for d in minutes.decisions) or "- None recorded")
        st.markdown("**Action items**")
        if minutes.action_items:
            st.dataframe(pd.DataFrame([{"Task": i.task, "Owner": i.owner, "Deadline": i.deadline,
                                        "Source sentence": i.source} for i in minutes.action_items]),
                         hide_index=True, width="stretch")
        else:
            st.markdown("- None recorded")
        if minutes.open_questions:
            st.markdown("**Open questions / unresolved**")
            st.markdown("\n".join(f"- {q}" for q in minutes.open_questions))
        st.caption("Decisions, action items and open questions are rule-based extractions "
                   "copied from the transcript. AI-generated draft: review before sharing.")

    d1, d2, d3 = st.columns(3)
    d1.download_button("Download transcript (.txt)", transcript, "transcript.txt", "text/plain",
                       width="stretch")
    d2.download_button("Download minutes (.txt)", minutes.to_text(), "meeting_minutes.txt",
                       "text/plain", width="stretch")
    d3.download_button("Download minutes (.json)", minutes.to_json(), "meeting_minutes.json",
                       "application/json", width="stretch")

    # -------------------------------------------------------------- evaluation
    st.divider()
    st.subheader("Evaluation")
    stats = summary_statistics(transcript, minutes.summary, state.timings)
    m = st.columns(5)
    m[0].metric("Transcript words", stats["transcript_words"])
    m[1].metric("Summary words", stats["summary_words"])
    m[2].metric("Compression ratio", stats["compression_ratio"])
    m[3].metric("Transcription time (s)", stats.get("transcription_s", "—"))
    m[4].metric("Summary + extraction time (s)", stats.get("minutes_generation_s", "—"))
    reference = st.text_area("Optional reference summary (for ROUGE)", height=100)
    if st.button("Compute ROUGE") and reference.strip():
        scores, impl = rouge_scores(minutes.summary, reference)
        st.dataframe(pd.DataFrame(scores).T, width="stretch")
        st.caption(f"ROUGE implementation: {impl}")
    st.caption("Compression ratio and ROUGE measure length and word overlap only. They do not "
               "prove factual accuracy; a human must check for hallucinated or omitted information.")

    # -------------------------------------------------------------- prompts
    st.divider()
    st.subheader("Prompt engineering")
    st.markdown(
        f"`{SUMMARY_MODEL}` is a summarization model **without instruction tuning**: it receives "
        "only the transcript, never a prompt. The structured prompt below is the template "
        f"designed for instruction-tuned models. It is sent, verbatim, to `{INSTRUCTION_MODEL}` "
        "in the comparison below.")
    with st.expander("Structured minutes prompt template (filled with this transcript)"):
        st.code(build_prompt(transcript, MINUTES_PROMPT_TEMPLATE), language="text")
    if st.button(f"Compare two prompts with {INSTRUCTION_MODEL} (downloads ~300 MB first time)"):
        cols = st.columns(len(PROMPT_VARIANTS))
        for col, (name, template) in zip(cols, PROMPT_VARIANTS.items()):
            with col, st.spinner(f"Running {name}…"):
                try:
                    out, truncated, secs = run_instruction_model(build_prompt(transcript, template))
                    st.markdown(f"**{name}**")
                    st.code(template.split("\n")[0][:300], language="text")
                    st.write(out or "(empty output)")
                    st.caption(f"{secs}s on CPU" + (" · input truncated to 512 tokens" if truncated else ""))
                except Exception as exc:
                    st.error(f"Instruction model unavailable: {exc}")
