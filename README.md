# EdgeMinutes — AI Meeting Minutes Summarizer

> **Problem statement:** To develop an Edge AI-based NLP system that converts meeting audio into text and generates concise, structured meeting minutes with key decisions and action items directly on a smartphone.

## Objective

EdgeMinutes accepts meeting audio (recorded or uploaded) or a pasted transcript. It converts speech to text with a pretrained Whisper model, summarizes the transcript with a pretrained Transformer, and extracts structured minutes: summary, key points, decisions, action items with owner and deadline, and open questions. Everything runs locally with free, open-source models. It needs no API keys and does not use cloud AI.

**Current status (please read):** the full pipeline runs **on the laptop's CPU**. A phone can open the web page and upload audio, but inference still happens on the laptop. Running it fully on the phone is *not* implemented yet. See [Edge AI](#edge-ai-where-does-inference-run).

## Architecture and data flow

```
Meeting Audio (record / upload WAV, MP3, M4A, FLAC, OGG, WEBM)
    -> Audio decoding (PyAV, resample to 16 kHz mono)
    -> Speech Recognition (faster-whisper "tiny", int8, CPU)
    -> Transcript  <---------------------- or: pasted transcript / demo sample
    -> Transformer Summarization (DistilBART, abstractive)
         \-> fallback: frequency-based extractive summary (no model)
    -> Rule-based extraction (decisions, action items, owners, deadlines, open questions)
    -> Structured Meeting Minutes (Python dataclass)
    -> Display / Download (.txt, .json) / Evaluation (compression, timing, ROUGE)
```

## Directory structure

```
EdgeMinutes/
    app.py                     Streamlit user interface
    minutes_cli.py             Command-line version (no UI; extractive mode is stdlib-only)
    requirements.txt
    README.md
    .gitignore
    .streamlit/config.toml     Disables usage telemetry and the file watcher
    src/
        transcription.py       Whisper speech-to-text
        summarization.py       DistilBART summarizer, extractive fallback, Flan-T5 prompt runner
        meeting_minutes.py     Prompt templates, data structures, rule-based extraction
        evaluation.py          Word counts, compression ratio, timing, ROUGE
    tests/
        test_meeting_minutes.py  23 unit tests (models mocked)
    sample_data/
        sample_transcript.txt  Hand-written demo transcript
    outputs/                   CLI output files
```

## Installation (Windows, Python 3.12)

```powershell
cd D:\MeetingSummarizer\EdgeMinutes
python -m venv venv                      # skip if venv\ already exists
.\venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only build, avoids a multi-GB CUDA download
pip install -r requirements.txt
```

If PowerShell blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `venv\Scripts\activate.bat` in cmd.

Models download automatically from Hugging Face the first time they are used, then load from `%USERPROFILE%\.cache\huggingface`:

| Model | Purpose | Download size |
|---|---|---|
| `Systran/faster-whisper-tiny` | Speech recognition | ~75 MB |
| `sshleifer/distilbart-cnn-12-6` | Summarization | ~1.2 GB (cache folder may show more) |
| `google/flan-t5-small` | Optional prompt comparison only | ~300 MB |

After the first download the app works **offline**. To force offline mode: `$env:HF_HUB_OFFLINE = "1"`.

## Running

```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

Open http://localhost:8501. Run the tests with `python -m pytest -q`.

Command-line alternative: `python minutes_cli.py sample_data\sample_transcript.txt --method extractive` (or `--method transformer`).

## Using the app

1. **Record audio**: tap **Start recording**, speak, tap **Stop recording**. This is a small custom component (`src/mic_recorder.py` + `src/mic_recorder_frontend/index.html`) using the browser's MediaRecorder API, because Streamlit's built-in `st.audio_input` did not start recording on an iPhone. The built-in recorder is still available under "Alternative". Browsers only allow the microphone on `localhost` or HTTPS pages (e.g. a VS Code dev tunnel).
2. **Upload audio**: WAV, MP3, M4A, FLAC, OGG, WEBM, or a phone video (MP4/MOV; only the sound is used). On an iPhone, the in-browser recorder may not start; record in Voice Memos (or as a video) and upload it instead. Decoding uses the FFmpeg libraries bundled with PyAV, so no separate FFmpeg install is needed.
3. **Paste transcript**: paste any text. `Name: sentence` speaker labels improve owner detection ("I will…" is assigned to the speaker).
4. **Demo sample transcript**: loads `sample_data/sample_transcript.txt`. It is labelled as demo data in the UI. The *transcript* is hand-written, but the summary and extraction run live on it. Nothing is pre-computed.

Then click **Generate Meeting Minutes**. The sidebar lets you choose the Whisper model size, the spoken language (or auto-detect), the summarizer (transformer or extractive fallback) and the maximum number of chunks.

## Models, libraries and NLP concepts

### What is implemented

| Component | Implementation | Concept demonstrated |
|---|---|---|
| Speech recognition | faster-whisper (CTranslate2), Whisper `tiny`, int8 quantised | Transformer **encoder-decoder**: audio encoder plus autoregressive text decoder; multilingual ASR with language ID |
| Summarization | `sshleifer/distilbart-cnn-12-6` via Hugging Face `transformers` | Pretrained **encoder-decoder** (BART), **knowledge distillation**, fine-tuning on CNN/DailyMail, abstractive generation with beam search |
| Long input handling | Sentence-aware chunking to ~900 tokens (limit 1024); chunk cap with an explicit warning when text is skipped | Context-length limits of self-attention |
| Fallback summarizer | Frequency-based extractive scoring | Classic extractive summarization. **Not** a neural model, and labelled as such |
| Structured extraction | Conservative regex rules | Information extraction, structured output, refusing to fabricate |
| Prompt engineering | Role prompt template plus two-prompt A/B comparison on Flan-T5-small | Role prompting, constraints, structured output instructions |
| Evaluation | Word counts, compression ratio, timings, ROUGE-1/2/L | Automatic summarization metrics and their limits |

### Concepts explained (not separately implemented)

- **RNNs → Transformers → LLMs.** RNN/LSTM seq2seq models read tokens one at a time, which makes long-range dependencies hard and training hard to parallelise. The Transformer (Vaswani et al., 2017) replaces recurrence with **self-attention**: every token attends to every other token. **Multi-head attention** runs several such attention maps in parallel to capture different relations. Scaling Transformers on large corpora produced pretrained language models and then LLMs.
- **Encoder-decoder vs decoder-only.** BART, T5 and Whisper are encoder-decoders: the encoder reads the whole input bidirectionally, the decoder generates output with cross-attention to it. That suits summarization, translation and ASR. GPT-style LLMs are decoder-only.
- **Pretraining objectives.** *Masked language modelling* (BERT) predicts hidden tokens using context from both sides. *Autoregressive* LM (GPT) predicts the next token. BART uses *denoising*: it corrupts text with masking, deletion and shuffling, then autoregressively reconstructs it. T5 uses span corruption.
- **Pretraining → fine-tuning.** DistilBART is BART distilled into fewer decoder layers, then fine-tuned for news summarization. Flan-T5 is T5 *instruction-tuned* on many tasks phrased as instructions, which is why it can follow a prompt and DistilBART cannot. Parameter-efficient fine-tuning (LoRA, adapters) on meeting data (e.g. the AMI corpus) would be the natural next step. It is not done here.
- **Machine translation (future extension).** Whisper can already transcribe many languages, and also has a `translate` task (speech → English). A MarianMT/NLLB encoder-decoder could translate the minutes. Not implemented.

## Prompt engineering

`src/meeting_minutes.py` defines:

- `SYSTEM_ROLE`: *"You are a careful professional meeting secretary. Summarize only information supported by the supplied transcript."*
- `MINUTES_PROMPT_TEMPLATE`: a role plus numbered constraints (concise and factual, keep decisions, extract action items, owners and deadlines only if stated, no fabricated facts, write "Not specified" when missing) and a fixed output structure.
- `PROMPT_VARIANTS`: **A** (plain "Summarize this meeting") vs **B** (role + constraints).

**What actually receives the prompt:**

- **DistilBART does not receive any prompt.** It is not instruction-tuned and only ever sees the transcript text. Calling its output "prompted" would be false.
- The *Prompt engineering* section of the app sends prompts A and B verbatim to the instruction-tuned **Flan-T5-small** and shows both outputs side by side.
- The full structured template is displayed, filled with the current transcript, ready for a larger local instruction model (future work).

**Observed result (honest finding):** on the sample transcript, Flan-T5-small (77M parameters) produced fluent but **hallucinated** summaries with both prompts. For example, it attributed the demo-video task to "Adi" when the transcript assigns it to Rahul, and it invented "start the meeting on November 14". This is why EdgeMinutes takes decisions, owners and deadlines from rule-based extraction over the transcript and not from a small generative model.

## Evaluation

Implemented in `src/evaluation.py` and shown in the app:

- Transcript words, summary words, **compression ratio** (summary words ÷ transcript words)
- **Processing time** for transcription and for summary + extraction
- **ROUGE-1/2/L** against an optional reference summary you type in, using `rouge-score` (with a built-in fallback implementation)

Measured on this laptop (Intel i3-1005G1, 12 GB RAM, CPU only):

| Step | Time |
|---|---|
| Whisper tiny, 24 s English speech (synthetic TTS audio) | ~3 s |
| DistilBART first load from disk | ~11 s (first call only) |
| DistilBART summary of the 216-word sample | ~2.4 s (compression ratio 0.12) |
| Rule-based extraction | < 0.1 s |

**Limits of these metrics:** compression ratio only measures length. ROUGE only measures n-gram overlap with a reference. A summary can score well and still contain a wrong name or deadline, or leave out a decision. **Human review against the transcript is required.** That is why the UI shows the source sentence for every action item.

## Responsible AI, privacy and security

- **Privacy:** audio and transcripts are processed in-process on the local machine and are never sent to a remote API. Uploaded audio goes to a temporary file that is deleted right after transcription. Streamlit usage telemetry is disabled in `.streamlit/config.toml`. The only network traffic is the one-time model download from Hugging Face.
- **Hallucination control:** the abstractive summary is labelled as model-generated. Decisions, action items, owners and deadlines are copied from transcript sentences. Missing owners and deadlines appear as **"Not specified"**, never guessed. Undecided items ("we have not decided who…") are listed as open questions, not decisions.
- **Transparency:** the UI says which method produced the summary, warns when a long transcript was truncated, and warns when the fallback summarizer was used.
- **Consent:** record meetings only with the participants' consent.
- **Bias / accuracy:** Whisper tiny is less accurate on accented speech, noisy audio and uncommon names. In testing it heard the name "Priya" as "Pre-AWIL", so that task was not attributed to anyone. Errors in the transcript carry through into the minutes.
- **Security:** no accounts, no stored data. If you expose the app on a network (`--server.address 0.0.0.0`), anyone on that network can use it.

## Edge AI: where does inference run?

| Stage | Where it runs now |
|---|---|
| Audio capture | Browser on whichever device opens the page (laptop, or a phone over Wi-Fi) |
| Speech recognition | **Laptop CPU** (faster-whisper, int8) |
| Summarization | **Laptop CPU** (DistilBART, PyTorch fp32, ~1.2 GB RAM) |
| Extraction / evaluation | **Laptop CPU** (pure Python) |
| Remote servers | None. Internet is needed only for the first model download |

This is **local, offline, on-device inference on a laptop**, which is an edge deployment in the sense that no cloud is involved. It is **not** yet the smartphone deployment the problem statement asks for. Opening the app from a phone (`streamlit run app.py --server.address 0.0.0.0`, then `http://<laptop-IP>:8501` on the same Wi-Fi) only makes the phone a client. Also, the phone's microphone recorder will not work over plain HTTP. Use **Upload audio** from the phone instead.

**Feasibility of running on the phone (future work, not tested on a phone):**

1. *Speech recognition:* Whisper tiny/base runs on modern Android phones via **whisper.cpp**. It has an official Android example app, and it can also be built in Termux. Quantised tiny is ~40–75 MB.
2. *Extraction + extractive summary:* `minutes_cli.py --method extractive` was verified to import **only the Python standard library**, so `src/` plus that CLI should run unchanged in Termux's Python on Android. This was checked on the laptop, not on a phone.
3. *Abstractive summary:* DistilBART (306M parameters) is heavy for a phone. Options: export to ONNX and int8-quantise for ONNX Runtime Mobile, use a smaller model (e.g. Flan-T5-small, with the hallucination caveat above), or use an on-device LLM runtime (MediaPipe LLM Inference / llama.cpp with a ~1B quantised model) driven by `MINUTES_PROMPT_TEMPLATE`.
4. A native app would chain (1) → (2)/(3) fully offline on the phone.

Edge Impulse was not used: it targets classification and keyword-spotting models, not sequence-to-sequence ASR and summarization.

## Known issues and limitations

- Whisper `tiny` favours speed over accuracy. Choose `base`/`small` in the sidebar for better transcripts (slower).
- Whisper output has **no speaker labels** (no diarisation), so "I will…" from audio gets owner "Not specified (speaker unknown)". Pasted transcripts with `Name:` labels work best.
- The rules are conservative and English-only. They catch patterns like "X will…", "I will…", "X, can you…", "we need to…", "assigned to X", "decided / agreed to…". Paraphrased or implicit commitments are missed, by design; it would rather miss an item than invent one.
- Without speaker labels, any capitalised word before "will" (e.g. a product name) may be read as an owner.
- DistilBART was fine-tuned on news, not meetings, so its summaries of conversational text can be generic. Transcripts longer than `max_chunks × ~900` tokens are partly skipped, with a warning.
- The first run downloads ~1.5 GB of models.
- Transformers prints a harmless `forced_bos_token_id` notice in the terminal.

## Future improvements

- Fully on-device Android build (whisper.cpp + quantised summarizer, see above).
- Speaker diarisation (e.g. pyannote) to attribute "I will…" in audio.
- Fine-tune the summarizer on meeting corpora (AMI, ICSI, QMSum), possibly with LoRA.
- A small local instruction LLM (quantised, via llama.cpp) using `MINUTES_PROMPT_TEMPLATE`, with outputs checked against the rule-based extraction.
- Translation of minutes (MarianMT / NLLB) and Whisper's speech-translation mode.
- Human evaluation form (faithfulness / completeness ratings) alongside ROUGE.
