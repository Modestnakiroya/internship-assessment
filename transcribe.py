"""
CLI: transcribe an audio file with Sunbird AI STT.

Rejects audio longer than 5 minutes. Token: SUNBIRD_API_TOKEN in .env.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv
from tinytag import TinyTag

from languages import LANGUAGE_CODE_BY_NAME, SUPPORTED_LANGUAGES

API_BASE = "https://api.sunbird.ai"
STT_URL = f"{API_BASE}/tasks/modal/stt"

MAX_DURATION_SECONDS = 5 * 60


def _load_token() -> str:
    load_dotenv()
    token = os.environ.get("SUNBIRD_API_TOKEN", "").strip()
    if not token:
        print(
            "Error: SUNBIRD_API_TOKEN is not set. Add it to a .env file in this directory.",
            file=sys.stderr,
        )
        sys.exit(1)
    return token


def _auth_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _audio_duration_seconds(path: Path) -> Optional[float]:
    try:
        tag = TinyTag.get(str(path))
    except Exception as exc:
        print(f"Error: could not read audio metadata: {exc}", file=sys.stderr)
        sys.exit(1)
    if tag.duration is None:
        return None
    return float(tag.duration)


def _prompt_target_language() -> str:
    print("\nTarget language (improves STT accuracy) — choose a number:")
    for i, name in enumerate(SUPPORTED_LANGUAGES, start=1):
        print(f"  {i}. {name}")
    while True:
        raw = input("Enter number: ").strip()
        if not raw.isdigit():
            print("Please enter a valid number.")
            continue
        idx = int(raw)
        if 1 <= idx <= len(SUPPORTED_LANGUAGES):
            return SUPPORTED_LANGUAGES[idx - 1]
        print(f"Please enter a number between 1 and {len(SUPPORTED_LANGUAGES)}.")


def _extract_transcript(payload: Any) -> Optional[str]:
    """Support both flat STT payloads and wrapped `output` shapes."""
    if not isinstance(payload, dict):
        return None
    if "audio_transcription" in payload and payload["audio_transcription"]:
        return str(payload["audio_transcription"])
    out = payload.get("output")
    if isinstance(out, dict):
        if out.get("text"):
            return str(out["text"])
        if out.get("audio_transcription"):
            return str(out["audio_transcription"])
    return None


def _print_http_error(context: str, resp: requests.Response) -> None:
    print(f"{context} request failed: HTTP {resp.status_code}", file=sys.stderr)
    try:
        detail = resp.json()
    except ValueError:
        print(resp.text[:2000], file=sys.stderr)
        return
    if isinstance(detail, dict) and "detail" in detail:
        print(detail["detail"], file=sys.stderr)
    else:
        print(detail, file=sys.stderr)


def main() -> None:
    token = _load_token()
    raw_path = input("\nPath to audio file: ").strip().strip('"')
    path = Path(raw_path).expanduser()
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    duration = _audio_duration_seconds(path)
    if duration is None:
        print(
            "Error: could not determine audio duration. Use a supported format "
            "(e.g. MP3, WAV, OGG, M4A, AAC).",
            file=sys.stderr,
        )
        sys.exit(1)
    if duration > MAX_DURATION_SECONDS:
        mins = duration / 60.0
        print(
            f"Error: audio is too long ({mins:.1f} minutes). "
            f"Maximum allowed length is 5 minutes.",
            file=sys.stderr,
        )
        sys.exit(1)

    target_name = _prompt_target_language()
    code = LANGUAGE_CODE_BY_NAME[target_name]

    with path.open("rb") as audio_fp:
        files = {"audio": (path.name, audio_fp, "application/octet-stream")}
        data = {"language": code}
        resp = requests.post(
            STT_URL,
            headers=_auth_headers(token),
            files=files,
            data=data,
            timeout=300,
        )

    if not resp.ok:
        _print_http_error("Speech-to-text", resp)
        sys.exit(1)

    body = resp.json()
    text = _extract_transcript(body)
    if not text:
        print("Error: no transcription returned.", file=sys.stderr)
        print(body, file=sys.stderr)
        sys.exit(1)

    print("\n--- Transcript ---\n")
    print(text)


if __name__ == "__main__":
    main()
