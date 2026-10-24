"""Command-line minutes generator (no Streamlit needed).

    python minutes_cli.py sample_data/sample_transcript.txt --method extractive

`--method extractive` needs only the Python standard library, so this file
plus src/ can run anywhere Python runs (e.g. Termux on Android), with the
transcript coming from an on-device ASR such as whisper.cpp.
"""

import argparse
from functools import partial
from pathlib import Path

from src.meeting_minutes import generate_minutes
from src.summarization import METHOD_EXTRACTIVE, METHOD_TRANSFORMER, summarize_transcript


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("transcript", type=Path, help="UTF-8 transcript text file")
    parser.add_argument("--method", choices=[METHOD_TRANSFORMER, METHOD_EXTRACTIVE],
                        default=METHOD_TRANSFORMER)
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="output directory")
    args = parser.parse_args()

    transcript = args.transcript.read_text(encoding="utf-8")
    minutes = generate_minutes(transcript, summarizer=partial(summarize_transcript, method=args.method))
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.transcript.stem
    (args.out / f"{stem}_minutes.txt").write_text(minutes.to_text(), encoding="utf-8")
    (args.out / f"{stem}_minutes.json").write_text(minutes.to_json(), encoding="utf-8")
    print(minutes.to_text())
    print(f"Saved to {args.out / (stem + '_minutes.txt')} and .json")


if __name__ == "__main__":
    main()
